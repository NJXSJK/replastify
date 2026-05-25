# backend/app/routes/predict.py
import asyncio
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Literal

from app.config import settings
from app.services.classifier import predict, get_model
from app.services.gemini import get_ai_suggestions
from app.services.plastic_info import PLASTIC_DATABASE, get_plastic_info
from app.utils.image_utils import validate_and_load_image

logger = logging.getLogger(__name__)
router = APIRouter()


# ══════════════════════════════════════════════════════════════════
# Response models — explicit schemas drive Swagger UI auto-docs
# ══════════════════════════════════════════════════════════════════

class Top3Item(BaseModel):
    type: str
    confidence: float


class PredictionInfo(BaseModel):
    plastic_type: str
    full_name: str
    resin_code: int
    confidence: float
    is_uncertain: bool
    uncertainty_message: str | None
    top3: list[Top3Item]
    all_probabilities: dict[str, float]


class PlasticInfoResponse(BaseModel):
    common_uses: list[str]
    recyclability: str
    recyclability_score: int
    health_concerns: str
    decomposition_years: int
    warning: str | None
    fun_fact: str


class SuggestionsResponse(BaseModel):
    recycling_tips: list[str]
    reuse_ideas: list[str]
    eco_alternatives: list[str]
    environmental_note: str
    source: Literal["ai", "static"]


class PredictResponse(BaseModel):
    prediction: PredictionInfo
    info: PlasticInfoResponse
    suggestions: SuggestionsResponse


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str
    classes: int
    confidence_threshold: float
    gemini_enabled: bool


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictResponse, summary="Identify a plastic type from an image")
async def predict_plastic(file: UploadFile = File(..., description="Image of the plastic item (jpg/png/webp, max 10 MB)")):
    """
    Upload a plastic image to receive:
    - Plastic type classification (EfficientNet-B0)
    - Recycling and environmental info
    - AI-generated disposal suggestions (Gemini, with static fallback)
    """
    # Step 1: Validate and decode image → raises HTTPException on bad input
    image = await validate_and_load_image(file)

    # Step 2: ML inference (offloaded to thread to keep the event loop non-blocking)
    result = await asyncio.to_thread(predict, image)

    # Step 3: Static plastic knowledge
    info = get_plastic_info(result.plastic_type)

    # Step 4: AI suggestions — async with timeout + silent fallback
    suggestions, source = await get_ai_suggestions(
        plastic_type=result.plastic_type,
        confidence=result.confidence,
        is_uncertain=result.is_uncertain,
    )

    # Log prediction details for auditability/drift analysis
    logger.info(
        f"Prediction: file='{file.filename}' -> detected='{result.plastic_type}' "
        f"(conf={result.confidence:.2f}, uncertain={result.is_uncertain}, suggestions_source='{source}')"
    )

    # Step 5: Assemble and return
    return PredictResponse(
        prediction=PredictionInfo(
            plastic_type=result.plastic_type,
            full_name=info["full_name"],
            resin_code=info["resin_code"],
            confidence=result.confidence,
            is_uncertain=result.is_uncertain,
            uncertainty_message=result.uncertainty_message,
            top3=[Top3Item(type=t.plastic_type, confidence=t.confidence) for t in result.top3],
            all_probabilities=result.all_probabilities,
        ),
        info=PlasticInfoResponse(
            common_uses=info["common_uses"],
            recyclability=info["recyclability"],
            recyclability_score=info["recyclability_score"],
            health_concerns=info["health_concerns"],
            decomposition_years=info["decomposition_years"],
            warning=info.get("warning"),
            fun_fact=info["fun_fact"],
        ),
        suggestions=SuggestionsResponse(
            recycling_tips=suggestions["recycling_tips"],
            reuse_ideas=suggestions["reuse_ideas"],
            eco_alternatives=suggestions["eco_alternatives"],
            environmental_note=suggestions["environmental_note"],
            source=source,
        ),
    )


@router.get("/health", response_model=HealthResponse, summary="Server and model health check")
async def health_check():
    """Returns current status of the server and loaded model."""
    try:
        get_model()
        model_loaded = True
    except RuntimeError:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model=settings.model_name,
        classes=settings.num_classes,
        confidence_threshold=settings.confidence_threshold,
        gemini_enabled=bool(settings.gemini_api_key),
    )


@router.get("/plastic-types", summary="Get full knowledge base for all plastic types")
async def get_all_plastic_types():
    """
    Returns static info for all 6 supported plastic types.
    Useful for frontend reference / info pages without uploading an image.
    """
    return {
        "plastic_types": PLASTIC_DATABASE,
        "count": len(PLASTIC_DATABASE),
        "types": list(PLASTIC_DATABASE.keys()),
    }
