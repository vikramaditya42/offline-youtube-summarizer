# app/services/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .stt_engine import STTEngine, STTResult
from .youtube_downloader import YouTubeDownloader, DownloadResult

import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionPipelineConfig:
    """
    Configuration for the URL → audio → transcript pipeline.
    """

    model_size: str = "small"
    device: str = "cuda"          # "cuda" or "cpu"
    compute_type: str = "float16" # "float16", "int8", "int8_float16", etc.
    output_dir: Path = Path("data/cache")


class TranscriptionPipeline:
    """
    End-to-end pipeline:
        YouTube URL → downloaded audio → offline STT → transcript.
    """

    def __init__(
        self,
        config: Optional[TranscriptionPipelineConfig] = None,
        stt_engine: Optional[STTEngine] = None,
        downloader: Optional[YouTubeDownloader] = None,
    ) -> None:
        self.config = config or TranscriptionPipelineConfig()

        self.downloader = downloader or YouTubeDownloader(
            output_dir=self.config.output_dir
        )

        self.stt_engine = stt_engine or STTEngine(
            model_size=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def run(
        self,
        url: str,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> tuple[DownloadResult, STTResult]:
        """
        Execute the full pipeline.

        Args:
            url: YouTube URL.
            language: Optional language code for STT (e.g., "en").
            beam_size: Beam size for decoding (higher = more accurate but slower).

        Returns:
            (DownloadResult, STTResult)
        """
        if language:
            logger.info(f"Starting pipeline for {url} with language={language}")
        else:
            logger.info(f"Starting pipeline for {url} (auto-detect language)")

        download_result = self.downloader.download_audio(url)
        logger.info(f"Audio downloaded: {download_result.audio_path}")

        stt_result = self.stt_engine.transcribe(
            download_result.audio_path,
            language=language,
            beam_size=beam_size,
        )
        logger.info(f"Transcription complete. Length: {len(stt_result.text)} chars")

        return download_result, stt_result


__all__ = ["TranscriptionPipeline", "TranscriptionPipelineConfig"]
