# app/config.py

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class AppSettings(BaseModel):
    """
    Central configuration for the web app.

    In a real project you'd probably use pydantic BaseSettings,
    but for this assignment a simple BaseModel + env reading is enough.
    """

    # STT config
    stt_model_size: str = "small"
    stt_device: str = "cuda"          # "cuda" or "cpu"
    stt_compute_type: str = "float16" # "float16", "int8", etc.

    # Summarizer (Phi-4) config
    phi_model_name: str = "microsoft/Phi-4-mini-instruct"
    phi_max_input_tokens: int = 8000
    phi_max_new_tokens_chunk: int = 256
    phi_max_new_tokens_final: int = 512

    # General
    cache_dir: Path = Path("data/cache")


def get_settings() -> AppSettings:
    """
    Return app settings. Later you can extend this to read env vars.
    """
    # If you want env-aware settings:
    #   import os
    #   return AppSettings(
    #       stt_model_size=os.getenv("STT_MODEL_SIZE", "small"),
    #       ...
    #   )
    return AppSettings()
