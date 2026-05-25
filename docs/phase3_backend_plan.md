# Phase 3 — Backend API: Comprehensive Implementation Plan
## Part 1 of 2: Architecture, Config, Inference Engine

**Phase Status:** Ready to implement  
**Inputs from Phase 2:** `best_efficientnet_b0.pth` | 6 classes | Test Acc: 88.94% | Macro F1: 0.8641  
**Key constraint from Phase 2 results:** PP class has F1=0.606. The API must surface uncertainty, not silently return wrong answers.

---

## 1. Architecture Philosophy & Design Decisions

Before writing any code, these decisions shape every file:

### Why FastAPI over Flask?
- **Automatic OpenAPI/Swagger docs** at `/docs` — zero extra work, great for demos
- **Async-native** — can handle multiple concurrent image uploads without blocking
- **Pydantic models** — request/response schemas are validated and type-safe automatically
- **Startup/shutdown events** — lets us load the 20 MB model file once at boot, not on every request

### Why a Singleton for the Model?
Loading `best_efficientnet_b0.pth` takes ~0.5–1 second. If we loaded the model on every `/predict` request, the API would be unusable. The singleton pattern ensures the model is loaded once at server startup and shared across all requests. On CPU inference, a single forward pass on a 224×224 image takes ~50–100ms — acceptable for a web app.

### Why Static Knowledge Base Instead of a Database?
We have exactly 6 plastic types with fixed, rarely-changing data. A PostgreSQL/SQLite database would add complexity and a new failure point for zero benefit. A Python dictionary in `plastic_info.py` loads at startup, lives in memory, and can be version-controlled alongside the code. No connection strings, no migrations, no DB crashes.

### Why Gemini as Optional Enrichment, Not Core?
The Gemini API is a third-party external service. External services go down, hit rate limits, and cost money per call. If Gemini fails and the entire `/predict` endpoint fails as a result, the whole application is broken. We treat Gemini as **optional enrichment**: the static knowledge base provides the baseline response, and Gemini adds richer AI suggestions on top — with a timeout and silent fallback.

### Single Server vs. Separate Frontend Server
FastAPI will serve the frontend HTML/JS/CSS as static files mounted at `/`. This means one `uvicorn` process serves everything — simpler deployment, no CORS issues in production, one port to open. During development, CORS is enabled for `localhost` to allow hot-reload frontend servers.

---

## 2. Complete Project Structure

```
backend/
├── app/
│   ├── __init__.py                 (exists — empty)
│   ├── main.py                     [NEW] App factory, CORS, startup, static files
│   ├── config.py                   [NEW] All settings from .env via pydantic-settings
│   ├── routes/
│   │   ├── __init__.py             (exists — empty)
│   │   └── predict.py              [NEW] All HTTP endpoints
│   ├── services/
│   │   ├── __init__.py             (exists — empty)
│   │   ├── classifier.py           [NEW] EfficientNet-B0 inference singleton
│   │   ├── plastic_info.py         [NEW] Static knowledge base — all 6 plastics
│   │   └── gemini.py               [NEW] Gemini AI integration with fallback
│   └── utils/
│       ├── __init__.py             (exists — empty)
│       └── image_utils.py          [NEW] PIL validation, RGB conversion, size checks
├── models/
│   └── best_efficientnet_b0.pth    ← copy from Kaggle download
├── temp/                           [auto-created at runtime]
├── .env                            [copy from .env.example and fill in]
├── .env.example                    (exists)
├── requirements.txt                [MODIFY] add pydantic-settings
└── Dockerfile                      [NEW]
```

> [!IMPORTANT]
> `pydantic-settings` is a **separate package** from `pydantic` (split in v2). Add it to `requirements.txt`:
> ```
> pydantic-settings>=2.0.0
> ```

---

## 3. `config.py` — Centralized Settings

**Why this exists:** Hardcoding paths, API keys, or thresholds anywhere in the codebase is a maintainability and security risk. Every configurable value lives here, loaded from `.env`.

```python
# backend/app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Model
    model_path: Path = Path("models/best_efficientnet_b0.pth")
    model_name: str = "efficientnet_b0"
    num_classes: int = 6

    # Inference thresholds (from Phase 2 analysis)
    confidence_threshold: float = 0.70   # below → is_uncertain=True
    pp_extra_warning: bool = True        # always add PP disclaimer

    # File upload limits
    max_file_size_mb: int = 10
    allowed_extensions: set[str] = {"jpg", "jpeg", "png", "webp"}

    # Paths
    temp_dir: Path = Path("temp")

    # External APIs
    gemini_api_key: str = ""             # empty string = Gemini disabled
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_seconds: int = 8

    # Server
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

# Instantiate once — imported everywhere else
settings = Settings()
```

**Key design decisions:**
- `pydantic-settings` auto-validates types at startup. If `model_path` points to a non-existent file, we catch it in the startup hook — not silently at first request.
- `gemini_api_key: str = ""` — empty string means Gemini is disabled. No key = graceful fallback to static DB. No crash.
- `confidence_threshold: float = 0.70` — derived from Phase 2 analysis. PP (our weakest class) has F1=0.606. Any prediction we're less than 70% sure about should be flagged.

---

## 4. `utils/image_utils.py` — Image Validation & Preprocessing

**Why a dedicated utils file:** The route handler should not contain image logic. Separating concerns makes the validation testable in isolation.

```python
# backend/app/utils/image_utils.py
import io
from PIL import Image, UnidentifiedImageError
from fastapi import UploadFile, HTTPException
from app.config import settings

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024

async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """
    Validates an uploaded file and returns a clean RGB PIL Image.
    Raises HTTPException with appropriate status codes on failure.
    """
    # 1. Extension check
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}"
        )

    # 2. Read bytes
    contents = await file.read()

    # 3. Size check
    if len(contents) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)//1024//1024} MB). Max allowed: {settings.max_file_size_mb} MB"
        )

    # 4. Valid image check
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()          # detects truncated/corrupt files
        image = Image.open(io.BytesIO(contents))  # re-open after verify() exhausts it
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Cannot process image: {str(e)}")

    # 5. Convert to RGB — handles RGBA PNGs, grayscale, CMYK, etc.
    image = image.convert("RGB")

    return image
```

**Why `image.verify()` then re-open?** PIL's `verify()` is destructive — it reads the file pointer to the end. You must re-open the image from the original bytes after calling it. Forgetting this step causes a subtle bug where the image appears valid but inference crashes.

**Why convert to RGB always?** Our model was trained on 3-channel RGB images. PNG files with transparency (RGBA) have 4 channels. Grayscale images have 1 channel. The model's first conv layer expects exactly 3 channels — passing 4 channels will crash with a dimension mismatch error. The `convert("RGB")` call handles all cases.

---

## 5. `services/classifier.py` — The Inference Engine

This is the most technically critical file. Every design decision here has direct impact on accuracy, speed, and reliability.

### 5.1 Model Architecture Must Match Training Exactly

```python
# backend/app/services/classifier.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass
from app.config import settings

# Class order MUST match training — from 03_train_model.py output:
# {'HDPE': 0, 'LDPE': 1, 'PET': 2, 'PP': 3, 'PS': 4, 'PVC': 5}
CLASS_NAMES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

# ImageNet normalization — MUST match val_transform in 03_train_model.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Val transform — NOT train transform (no random flips/crops at inference)
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

> **Critical gotcha:** Using train transforms at inference (RandomResizedCrop, RandomHorizontalFlip) would randomly modify the image differently each time, making the same image give different predictions on different calls. Always use the deterministic val transform at inference.

### 5.2 Building the Model Architecture

The model architecture must be rebuilt identically to how it was created in `03_train_model.py`. We cannot just call `torch.load(path)` on a state_dict and expect a complete model — we must first construct the architecture, then load the weights.

```python
def _build_model() -> nn.Module:
    """Rebuild EfficientNet-B0 with custom head — must match 03_train_model.py exactly."""
    model = models.efficientnet_b0(weights=None)  # no pretrained weights — we load ours

    feature_dim = model.classifier[1].in_features  # 1280

    # Custom head — must be IDENTICAL to build_model() in 03_train_model.py
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 6),          # 6 classes
    )
    return model
```

### 5.3 Singleton Loading

```python
_model: nn.Module | None = None

def load_model() -> nn.Module:
    """Load model from disk. Called once at app startup."""
    global _model
    model = _build_model()
    state_dict = torch.load(
        settings.model_path,
        map_location="cpu",         # GPU-trained model → load on CPU server
        weights_only=True           # security: prevents arbitrary code execution
    )
    model.load_state_dict(state_dict)
    model.eval()                    # CRITICAL: disables Dropout and BatchNorm training mode
    _model = model
    return _model

def get_model() -> nn.Module:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model
```

**Why `map_location="cpu"`?** The model was trained on a Kaggle T4 GPU. The production server likely has no GPU. Without `map_location`, PyTorch tries to load the CUDA tensors onto a CUDA device — which crashes on a CPU-only server.

**Why `weights_only=True`?** PyTorch's `torch.load()` by default uses Python's `pickle`, which can execute arbitrary code embedded in the file. `weights_only=True` restricts deserialization to tensors only — a critical security hardening step when loading model files from external sources.

**Why `model.eval()`?** Two layers behave differently between training and inference:
- **Dropout:** In training, randomly zeros neurons. In eval, passes all values through. We need deterministic predictions.
- **BatchNorm:** In training, computes batch statistics. In eval, uses running mean/variance. Forgetting `.eval()` causes inconsistent, degraded predictions.

### 5.4 The Predict Function

```python
@dataclass
class Top3Prediction:
    plastic_type: str
    confidence: float

@dataclass  
class PredictionResult:
    plastic_type: str
    confidence: float
    is_uncertain: bool
    uncertainty_message: str | None
    top3: list[Top3Prediction]
    all_probabilities: dict[str, float]

def predict(image: Image.Image) -> PredictionResult:
    """Run inference on a PIL Image. Returns structured prediction result."""
    model = get_model()

    # Preprocess
    tensor = _inference_transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Inference — no_grad() prevents memory accumulation (critical for long-running servers)
    with torch.no_grad():
        logits = model(tensor)                          # [1, 6]
        probabilities = torch.softmax(logits, dim=1)   # [1, 6] → sum to 1.0
        probabilities = probabilities.squeeze(0)        # [6]

    # Convert to Python
    probs_list = probabilities.tolist()
    all_probs = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs_list)}

    # Top-1
    top1_idx = int(probabilities.argmax())
    top1_class = CLASS_NAMES[top1_idx]
    top1_conf = round(probs_list[top1_idx], 4)

    # Top-3
    top3_indices = probabilities.topk(3).indices.tolist()
    top3 = [
        Top3Prediction(CLASS_NAMES[i], round(probs_list[i], 4))
        for i in top3_indices
    ]

    # Uncertainty check (from Phase 2 analysis)
    is_uncertain = top1_conf < settings.confidence_threshold
    uncertainty_msg = None

    if is_uncertain:
        uncertainty_msg = (
            f"Confidence is {top1_conf:.0%} — below the reliable threshold of "
            f"{settings.confidence_threshold:.0%}. Consider checking the resin code on the item."
        )
    elif top1_class == "PP" and settings.pp_extra_warning:
        uncertainty_msg = (
            "PP detection has reduced accuracy in this model. "
            "Verify by checking for resin code #5 on the item."
        )

    return PredictionResult(
        plastic_type=top1_class,
        confidence=top1_conf,
        is_uncertain=is_uncertain,
        uncertainty_message=uncertainty_msg,
        top3=top3,
        all_probabilities=all_probs,
    )
```

**Why `torch.no_grad()`?** Without it, PyTorch builds a computation graph for every forward pass (for backpropagation support). During inference we never do backprop, so the graph is pure wasted memory and computation. On a long-running server processing hundreds of requests, this would cause memory leaks.

---

## 6. `services/plastic_info.py` — Knowledge Base

Complete static database for all 6 plastic types. Covers every field the frontend and API response schema needs.

```python
PLASTIC_DATABASE = {
    "HDPE": {
        "full_name": "High-Density Polyethylene",
        "resin_code": 2,
        "common_uses": ["Milk jugs", "Detergent bottles", "Shampoo bottles", "Plastic lumber", "Grocery bags"],
        "recyclability": "Widely recycled",
        "recyclability_score": 5,
        "health_concerns": "Considered one of the safest plastics. Non-toxic, does not leach chemicals under normal use.",
        "decomposition_years": 500,
        "recycling_tips": ["Rinse containers before recycling", "Remove pumps and sprayers", "Check local curbside acceptance"],
        "reuse_ideas": ["Storage containers", "Garden planters", "Toy storage bins"],
        "eco_alternatives": ["Glass jars", "Stainless steel containers", "Bamboo dispensers"],
        "fun_fact": "Recycled HDPE is used to make park benches, plastic lumber, and even playground equipment.",
        "warning": None,
    },
    "LDPE": {
        "full_name": "Low-Density Polyethylene",
        "resin_code": 4,
        "common_uses": ["Grocery bags", "Bread bags", "Cling wrap", "Squeezable bottles", "Six-pack rings"],
        "recyclability": "Limited — not typically curbside; drop-off required",
        "recyclability_score": 3,
        "health_concerns": "Generally considered safe. Does not leach harmful chemicals under normal use.",
        "decomposition_years": 500,
        "recycling_tips": ["Return to grocery store drop-off bins", "Bundle multiple bags together", "Keep dry before drop-off"],
        "reuse_ideas": ["Bin liners", "Packing material", "Waterproofing layer in gardens"],
        "eco_alternatives": ["Canvas tote bags", "Silicone food bags", "Beeswax wraps"],
        "fun_fact": "Some grocery store drop-off programs recycle LDPE bags into composite decking material.",
        "warning": None,
    },
    "PET": {
        "full_name": "Polyethylene Terephthalate",
        "resin_code": 1,
        "common_uses": ["Water bottles", "Soda bottles", "Food jars", "Salad dressing containers", "Polyester clothing"],
        "recyclability": "Widely recycled",
        "recyclability_score": 5,
        "health_concerns": "Safe for single use. Prolonged reuse may allow bacteria to harbour in scratches. Can leach antimony if heated.",
        "decomposition_years": 450,
        "recycling_tips": ["Rinse thoroughly", "Remove the bottle cap (different plastic)", "Flatten to save space in bin"],
        "reuse_ideas": ["Planters", "Bird feeders", "Piggy banks"],
        "eco_alternatives": ["Glass bottles", "Stainless steel bottles", "Aluminum cans"],
        "fun_fact": "Recycling one PET bottle saves enough energy to power a laptop for approximately 25 minutes.",
        "warning": None,
    },
    "PP": {
        "full_name": "Polypropylene",
        "resin_code": 5,
        "common_uses": ["Yogurt tubs", "Bottle caps", "Straws", "Microwave-safe containers", "Medicine bottles", "Tupperware"],
        "recyclability": "Increasingly accepted in curbside programs — check locally",
        "recyclability_score": 4,
        "health_concerns": "Considered safe. Heat-resistant — one of the few plastics safe for microwave use. Does not leach chemicals at normal temperatures.",
        "decomposition_years": 20,
        "recycling_tips": ["Check local municipal guidelines", "Clean thoroughly (food residue prevents recycling)", "Separate lids from bottles"],
        "reuse_ideas": ["Food storage containers", "Seed trays", "Workshop organizers"],
        "eco_alternatives": ["Glass jars (for food storage)", "Stainless steel containers", "Bamboo straws"],
        "fun_fact": "PP actually has one of the shortest decomposition times of common plastics — just 20–30 years vs 450+ for PET.",
        "warning": None,
    },
    "PS": {
        "full_name": "Polystyrene (Styrofoam)",
        "resin_code": 6,
        "common_uses": ["Foam cups", "Takeout containers", "Packing peanuts", "CD cases", "Disposable plates"],
        "recyclability": "Difficult — rarely accepted curbside",
        "recyclability_score": 1,
        "health_concerns": "Can leach styrene, a possible carcinogen, especially when heated. Avoid microwaving food in PS containers.",
        "decomposition_years": 500,
        "recycling_tips": ["Check for specialized drop-off facilities", "Avoid placing in general recycling (it contaminates batches)", "Reuse for packing fragile items"],
        "reuse_ideas": ["Packing material for shipping", "Insulation in garden cold frames", "Foam blocks for craft projects"],
        "eco_alternatives": ["Paper/fiber cups", "Compostable PLA containers", "Reusable stainless steel mugs"],
        "fun_fact": "Polystyrene is 95% air by volume. It is incredibly light but very difficult to recycle because of this.",
        "warning": "Avoid heating food in PS containers — styrene may migrate into food.",
    },
    "PVC": {
        "full_name": "Polyvinyl Chloride",
        "resin_code": 3,
        "common_uses": ["Pipes", "Shower curtains", "Cling wrap", "Vinyl flooring", "Window frames", "Electrical cable insulation"],
        "recyclability": "Difficult — specialized facilities only",
        "recyclability_score": 2,
        "health_concerns": "Contains harmful phthalates and chlorine. Can release dioxins when burned. Avoid for food/drink contact.",
        "decomposition_years": 1000,
        "recycling_tips": ["Do not place in general recycling — it contaminates entire batches", "Contact specialist PVC recyclers", "Consider take-back programs from manufacturers"],
        "reuse_ideas": ["Pipe sections as garden edging", "PVC sheet as workshop surface protector"],
        "eco_alternatives": ["Copper or PEX pipes", "Silicone-based wraps", "Natural fiber shower curtains"],
        "fun_fact": "PVC is one of the most produced plastics globally, but also one of the most difficult to safely dispose of.",
        "warning": "Contains chlorine and phthalates. Do not burn PVC — releases toxic dioxins.",
    },
}

def get_plastic_info(class_name: str) -> dict:
    """Return full plastic info dict. Raises KeyError if class_name invalid."""
    if class_name not in PLASTIC_DATABASE:
        raise KeyError(f"Unknown plastic type: '{class_name}'. Valid types: {list(PLASTIC_DATABASE.keys())}")
    return PLASTIC_DATABASE[class_name]
```

---

*→ Continue in Part 2: Gemini Service, API Routes, main.py, Dockerfile, Error Handling, Testing*
# Phase 3 — Backend API: Comprehensive Implementation Plan
## Part 2 of 2: Gemini, Routes, App Factory, Security & Testing

*(Continued from Part 1: Architecture, Config, Classifier, Knowledge Base)*

---

## 7. `services/gemini.py` — AI Enrichment with Graceful Fallback

### Design Philosophy
Gemini must **never** be a single point of failure. If the key is missing, quota is exhausted, or the API times out, the `/predict` endpoint still returns a complete, useful response using the static knowledge base. The frontend never knows the difference — the response shape is identical.

### Prompt Engineering
The prompt is structured to request JSON output, which makes parsing reliable. We include the confidence level so Gemini can give appropriately hedged advice for uncertain detections.

```python
# backend/app/services/gemini.py
import json
import asyncio
from google import genai
from app.config import settings
from app.services.plastic_info import get_plastic_info

_client: genai.Client | None = None

def _get_client() -> genai.Client | None:
    """Lazy-initialize Gemini client. Returns None if no API key configured."""
    global _client
    if _client is None and settings.gemini_api_key:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client

def _build_prompt(plastic_type: str, confidence: float, is_uncertain: bool) -> str:
    info = get_plastic_info(plastic_type)  # already imported at top of file

    confidence_note = (
        f"Note: Identification confidence is {confidence:.0%}, which is below the reliable threshold. "
        "Provide general advice for this plastic type." if is_uncertain
        else f"Confidence: {confidence:.0%}."
    )

    return f"""You are a plastic waste and recycling expert helping a user identify and responsibly dispose of plastic.

The user's item has been identified as: {plastic_type} ({info['full_name']}, Resin Code #{info['resin_code']}).
{confidence_note}

Provide specific, actionable advice. Respond ONLY with a valid JSON object matching this exact schema:
{{
  "recycling_tips": ["tip1", "tip2", "tip3"],
  "reuse_ideas": ["idea1", "idea2", "idea3"],
  "eco_alternatives": ["alt1", "alt2"],
  "environmental_note": "One sentence about the environmental impact of this specific plastic type."
}}

Be concise. Each item should be one sentence or less. Do not include markdown, code blocks, or any text outside the JSON object."""

def _parse_gemini_response(text: str) -> dict:
    """Extract JSON from Gemini response. Handles responses with surrounding text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown or surrounding text
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return None to signal parsing failed
    return None

async def get_ai_suggestions(
    plastic_type: str,
    confidence: float,
    is_uncertain: bool
) -> tuple[dict, str]:
    """
    Returns (suggestions_dict, source) where source is 'ai' or 'static'.
    Never raises — always returns something usable.
    """
    client = _get_client()

    if client is None:
        # No API key configured — use static
        return _static_fallback(plastic_type), "static"

    prompt = _build_prompt(plastic_type, confidence, is_uncertain)

    try:
        # Use get_running_loop() — get_event_loop() is deprecated in Python 3.10+
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt
                )
            ),
            timeout=settings.gemini_timeout_seconds
        )

        parsed = _parse_gemini_response(response.text)
        if parsed is None:
            return _static_fallback(plastic_type), "static"

        return parsed, "ai"

    except asyncio.TimeoutError:
        # Gemini took too long — don't block the response
        return _static_fallback(plastic_type), "static"
    except Exception:
        # Any other error (quota, network, invalid key) — silent fallback
        return _static_fallback(plastic_type), "static"

def _static_fallback(plastic_type: str) -> dict:
    """Build suggestions from static knowledge base in the same schema as Gemini response."""
    info = get_plastic_info(plastic_type)
    return {
        "recycling_tips": info["recycling_tips"],
        "reuse_ideas": info["reuse_ideas"],
        "eco_alternatives": info["eco_alternatives"],
        "environmental_note": (
            f"{info['full_name']} takes approximately {info['decomposition_years']} years to decompose. "
            f"Recyclability: {info['recyclability']}."
        ),
    }
```

**Why `asyncio.wait_for` + `run_in_executor`?** The `google-genai` SDK is synchronous. If we call it directly in an async FastAPI route, it blocks the entire event loop — no other requests can be processed during that Gemini call. `run_in_executor` offloads it to a thread, and `wait_for` adds the 8-second timeout so a slow Gemini response doesn't hang the user's browser.

---

## 8. `routes/predict.py` — All API Endpoints

### 8.1 Pydantic Response Models

Defining explicit response models serves two purposes: FastAPI auto-generates accurate API docs, and returning the wrong shape is caught at development time, not in production.

```python
# backend/app/routes/predict.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.config import settings          # needed by health_check()
from app.services.classifier import predict
from app.services.plastic_info import get_plastic_info, PLASTIC_DATABASE
from app.services.gemini import get_ai_suggestions
from app.utils.image_utils import validate_and_load_image

router = APIRouter()

# ── Response Models ──────────────────────────────────────
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
    source: str   # "ai" | "static"

class PredictResponse(BaseModel):
    prediction: PredictionInfo
    info: PlasticInfoResponse
    suggestions: SuggestionsResponse
```

### 8.2 `POST /predict` — The Core Endpoint

```python
@router.post("/predict", response_model=PredictResponse)
async def predict_plastic(file: UploadFile = File(...)):
    """
    Upload an image to identify the plastic type.
    Returns prediction, static info, and AI-generated recycling suggestions.
    """
    # Step 1: Validate image — raises HTTPException on bad input
    image = await validate_and_load_image(file)

    # Step 2: Run ML inference
    result = predict(image)

    # Step 3: Get static plastic info
    info = get_plastic_info(result.plastic_type)

    # Step 4: Get AI suggestions (async, with timeout + fallback)
    suggestions, source = await get_ai_suggestions(
        plastic_type=result.plastic_type,
        confidence=result.confidence,
        is_uncertain=result.is_uncertain,
    )

    # Step 5: Assemble response
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
```

### 8.3 `GET /health`

```python
@router.get("/health")
async def health_check():
    """Returns server and model status. Used by monitoring and deployment checks."""
    from app.services.classifier import get_model
    try:
        model = get_model()
        model_loaded = model is not None
    except RuntimeError:
        model_loaded = False

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model": "efficientnet_b0",
        "classes": 6,
        "confidence_threshold": settings.confidence_threshold,
        "gemini_enabled": bool(settings.gemini_api_key),
    }
```

### 8.4 `GET /plastic-types`

```python
@router.get("/plastic-types")
async def get_all_plastic_types():
    """
    Returns the complete knowledge base for all 6 plastic types.
    Useful for frontend reference/info pages without making a prediction.
    """
    return {
        "plastic_types": PLASTIC_DATABASE,
        "count": len(PLASTIC_DATABASE),
    }
```

---

## 9. `main.py` — Application Factory

```python
# backend/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import logging

from app.config import settings
from app.routes.predict import router
from app.services.classifier import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App instantiation ─────────────────────────────────────
app = FastAPI(
    title="Replastify API",
    description="EfficientNet-B0 powered plastic identification. Upload an image → get plastic type, recycling info, and AI suggestions.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI — auto-generated, no extra work
    redoc_url="/redoc",     # Alternative API docs
)

# ── CORS ─────────────────────────────────────────────────
# During dev: allow localhost. In production: restrict to your domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Startup: Load model once ──────────────────────────────
# on_event("startup") is deprecated since FastAPI 0.93. lifespan is the
# correct modern approach — no deprecation warnings.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Loading EfficientNet-B0 model...")
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {settings.model_path}\n"
            "Download best_efficientnet_b0.pth from Kaggle and place it in backend/models/"
        )
    load_model()
    settings.temp_dir.mkdir(exist_ok=True)
    logger.info(f"Model loaded. Confidence threshold: {settings.confidence_threshold}")
    logger.info(f"Gemini: {'enabled' if settings.gemini_api_key else 'disabled (static fallback)'}")
    yield
    # --- Shutdown ---

app = FastAPI(
    title="Replastify API",
    description="EfficientNet-B0 plastic identification API.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Global exception handler ──────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."}
    )

# ── Register routes ───────────────────────────────────────
app.include_router(router)

# ── Serve frontend static files ───────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
```

**Why the startup `FileNotFoundError` check?** It is far better to crash loudly at server startup than to start successfully and then fail silently on the first request. If the model file isn't there, every `/predict` call will fail — we should fail fast and give the developer a clear error message with the fix.

---

## 10. `Dockerfile` — Containerization

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for Pillow (image processing)
RUN apt-get update && apt-get install -y \
    curl \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Why `--workers 1`?** The model is loaded as a module-level singleton. With multiple workers, each worker process loads its own copy of the model. On a small server, 2 workers × 20 MB model = 40 MB VRAM/RAM for models alone. `workers=1` is fine for a project/demo. For true production scale, use a model server (TorchServe/Triton) instead.

---

## 11. Error Handling Taxonomy

Every failure mode should return the right HTTP status with a human-readable `detail` field:

| Scenario | HTTP Status | `detail` Message |
|---|---|---|
| Non-image file uploaded | `400 Bad Request` | `"Unsupported file type '.pdf'. Allowed: jpg, jpeg, png, webp"` |
| File > 10 MB | `413 Request Entity Too Large` | `"File too large (14 MB). Max allowed: 10 MB"` |
| Corrupt/truncated image | `422 Unprocessable Entity` | `"Cannot process image: file is truncated"` |
| No file attached to request | `422 Unprocessable Entity` | FastAPI auto-generates this |
| Model file missing at startup | Server won't start | Clear `FileNotFoundError` in logs |
| Gemini timeout / quota | Transparent — silent fallback | `source: "static"` in response, no error |
| Unexpected server error | `500 Internal Server Error` | `"Internal server error. Please try again."` |

---

## 12. Security Hardening

| Concern | Implementation |
|---|---|
| API key exposure | `.env` file, never committed. `.gitignore` entry already exists. |
| `torch.load` code execution | `weights_only=True` — prevents pickle-based code injection |
| File upload path traversal | Using `io.BytesIO` in memory — never writing user-controlled filenames to disk |
| Large file DoS | 10 MB size check before loading into memory |
| Corrupt file crash | `PIL.Image.verify()` before inference |
| CORS | Explicit allow-list of origins — no wildcard `*` in production |

---

## 13. Build Order (Step-by-Step)

Build and test incrementally in this order. Each step is independently testable before the next:

```
Step 1: config.py
        → python -c "from app.config import settings; print(settings)"

Step 2: utils/image_utils.py
        → pytest tests/test_image_utils.py (write a quick test)

Step 3: services/plastic_info.py
        → python -c "from app.services.plastic_info import get_plastic_info; print(get_plastic_info('PET'))"

Step 4: services/classifier.py
        → Need model file first. Test with: python -c "from app.services.classifier import load_model, predict; ..."

Step 5: services/gemini.py
        → Test with GEMINI_API_KEY="" (should use static fallback silently)

Step 6: routes/predict.py + main.py
        → uvicorn app.main:app --reload
        → Test /health, /plastic-types, then /predict

Step 7: Dockerfile
        → docker build -t replastify-backend .
        → docker run -p 8000:8000 --env-file .env replastify-backend
```

---

## 14. Full Verification & Testing Plan

### Smoke Tests (after `uvicorn app.main:app --reload`)
```bash
# Health check
curl http://localhost:8000/health

# All plastic types
curl http://localhost:8000/plastic-types | python3 -m json.tool

# API docs (open in browser)
open http://localhost:8000/docs
```

### Functional Tests
```bash
# Happy path: valid plastic image
curl -X POST http://localhost:8000/predict \
  -F "file=@test_images/pet_bottle.jpg" \
  | python3 -m json.tool

# Verify response fields exist
# Check: prediction.plastic_type, prediction.confidence, prediction.is_uncertain
# Check: info.recyclability_score (1-5)
# Check: suggestions.source ("ai" or "static")
```

### Edge Case Tests
```bash
# Non-image file
curl -X POST http://localhost:8000/predict -F "file=@document.pdf"
# Expected: 400

# Oversized file (create a >10MB dummy)
dd if=/dev/urandom of=big.jpg bs=1M count=11
curl -X POST http://localhost:8000/predict -F "file=@big.jpg"
# Expected: 413

# RGBA PNG (transparency)
curl -X POST http://localhost:8000/predict -F "file=@logo_with_alpha.png"
# Expected: 200 with valid prediction

# Gemini disabled (empty key in .env)
# Check suggestions.source == "static" in all responses
```

### Per-Class Sanity Check
Use one known-good test image per plastic type:

| Class | Test Image | Expected Min Confidence |
|---|---|---|
| LDPE | grocery bag photo | > 0.85 |
| PVC | clear pipe section | > 0.80 |
| HDPE | detergent bottle | > 0.80 |
| PS | styrofoam cup | > 0.75 |
| PET | clear water bottle | > 0.75 |
| PP | yogurt container | > 0.60 (known weak class) |

---

## 15. Open Questions Before Starting

> [!IMPORTANT]
> **Q1 — Model file:** Will you manually copy `best_efficientnet_b0.pth` into `backend/models/` locally, or should the Docker image download it from a URL? For now, manual copy is simplest.

> [!IMPORTANT]
> **Q2 — Gemini API key:** Do you have one ready? If not, leave `GEMINI_API_KEY=""` in `.env` — the backend will use static fallback for all suggestions. Gemini can be enabled later.

> [!IMPORTANT]
> **Q3 — Frontend serving:** The current plan mounts the frontend from FastAPI. If you plan to use a React/Vite frontend (Phase 4), we may need CORS enabled between two separate servers during dev. Confirm the frontend tech stack before Phase 4 begins.
