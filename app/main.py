# app/main.py

from __future__ import annotations

import logging
import asyncio
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings
from app.services.pipeline import (
    TranscriptionPipeline,
    TranscriptionPipelineConfig,
)
from app.services.summarizer import (
    Phi4Summarizer,
    SummarizerConfig,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline YouTube Summarizer",
        version="0.1.0",
        description=(
            "Offline YouTube video summarizer using faster-whisper for STT "
            "and microsoft/Phi-4-mini-instruct for summarization."
        ),
    )

    settings = get_settings()

    # Ensure cache dir exists
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

    # Initialize singletons once (so models are loaded once per process)
    stt_config = TranscriptionPipelineConfig(
        model_size=settings.stt_model_size,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        output_dir=settings.cache_dir,
    )
    pipeline = TranscriptionPipeline(config=stt_config)

    sum_config = SummarizerConfig(
        model_name=settings.phi_model_name,
        device=settings.stt_device,  # share same device
        max_input_tokens=settings.phi_max_input_tokens,
        max_new_tokens_chunk=settings.phi_max_new_tokens_chunk,
        max_new_tokens_final=settings.phi_max_new_tokens_final,
    )
    summarizer = Phi4Summarizer(config=sum_config)

    # Store in app.state for dependency injection
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.summarizer = summarizer
    app.state.lock = asyncio.Lock()

    # Include API router under /api
    app.include_router(api_router, prefix="/api")

    # Serve static files (Bonus: Web Interface)
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def read_index():
            return FileResponse(static_dir / "index.html")
            
    except ImportError:
        logger.warning("StaticFiles not available (pip install fastapi[standard] or aiofiles required?)")


    return app


app = create_app()
