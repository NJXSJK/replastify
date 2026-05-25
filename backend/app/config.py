# backend/app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Model ────────────────────────────────────────────────
    model_path: Path = Path("models/best_efficientnet_b0.pth")
    model_name: str = "efficientnet_b0"
    num_classes: int = 6

    # ── Inference thresholds (from Phase 2 analysis) ─────────
    # PP class has F1=0.606 — flag anything below 70% as uncertain
    confidence_threshold: float = 0.70
    pp_extra_warning: bool = True   # always append PP-specific disclaimer

    # ── File upload limits ───────────────────────────────────
    max_file_size_mb: int = 10
    allowed_extensions: set[str] = {"jpg", "jpeg", "png", "webp"}

    # ── Paths ────────────────────────────────────────────────
    temp_dir: Path = Path("temp")

    # ── Gemini API ───────────────────────────────────────────
    # Leave empty to disable Gemini — static knowledge base used as fallback
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_seconds: int = 8

    # ── Server / CORS ────────────────────────────────────────
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",   # VS Code Live Server
    ]


# Instantiated once at import time — all other modules import this object
settings = Settings()
