# backend/app/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.predict import router
from app.services.classifier import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Lifespan — replaces deprecated @app.on_event("startup")
# Correct modern pattern since FastAPI 0.93+
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("  Replastify API — starting up")
    logger.info("═" * 55)

    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"\n\nModel file not found: {settings.model_path}\n"
            "Fix: download best_efficientnet_b0.pth from Kaggle and copy it to backend/models/\n"
        )

    load_model()
    settings.temp_dir.mkdir(exist_ok=True)

    logger.info(f"  ✅ Model loaded:           {settings.model_name}")
    logger.info(f"  ✅ Classes:                {settings.num_classes}")
    logger.info(f"  ✅ Confidence threshold:   {settings.confidence_threshold}")
    logger.info(f"  ✅ Gemini:                 {'enabled' if settings.gemini_api_key else 'disabled (static fallback)'}")
    logger.info("═" * 55)

    yield  # ← server runs here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Replastify API — shutting down")


# ══════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Replastify API",
    description=(
        "EfficientNet-B0 powered plastic identification. "
        "Upload a photo → get plastic type, recycling info, and AI disposal suggestions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ══════════════════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# Global error handler — prevents raw tracebacks reaching the client
# ══════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ══════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════

app.include_router(router)


# ══════════════════════════════════════════════════════════════════
# Serve frontend (Phase 4)
# Mounted last so API routes take priority over static file fallback
# ══════════════════════════════════════════════════════════════════

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info(f"Frontend served from: {_frontend_dir}")
