"""Centralised configuration for the production ML pipeline.

All values come from environment variables with sane defaults. Override at
runtime via env vars or a `.env` file at the project root.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables.

    Override any field with an env var, e.g. `MODEL_PATH=foo.onnx python app.py`.
    """

    # ── Paths ────────────────────────────────────────────────────────────
    model_path: str = "models/baseline.onnx"
    vectorizer_path: str = "models/dv.joblib"
    models_path: str = "models/models.joblib"
    data_path: str = "yellow_tripdata_2026-01.parquet"

    random_state: int = 42
    test_size: float = 0.2

    # ── ONNX / Runtime ───────────────────────────────────────────────────
    onnx_provider: Optional[str] = None        # auto-detect if None
    onnx_opset: int = 21
    intra_op_threads: int = 1

    # ── Service ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Pydantic-settings reads env vars matching field names
    # (case-insensitive), falling back to a .env file at project root.
    # extra="ignore" means unknown env vars don't crash startup.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()
