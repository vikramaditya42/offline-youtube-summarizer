# app/services/stt_engine.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from faster_whisper import WhisperModel, BatchedInferencePipeline


@dataclass
class STTSegment:
    """Single segment of a transcript."""

    start: float
    end: float
    text: str


@dataclass
class STTResult:
    """Full transcription result."""

    text: str
    segments: List[STTSegment]
    language: Optional[str] = None


class STTEngine:
    """Offline speech-to-text engine using `faster-whisper`.

    Supports both standard and batched transcription.
    """

    def __init__(
        self,
        model_size: str = "small",      # "tiny", "base", "small", "medium", "large-v3", etc.
        device: str = "cpu",            # "cpu" or "cuda"
        compute_type: str = "int8",     # "float16", "int8", "int8_float16", etc.
        use_batched: bool = False,      # enable BatchedInferencePipeline
        default_batch_size: int = 8,    # used when use_batched=True and no batch_size passed
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_batched = use_batched
        self.default_batch_size = default_batch_size

        # Lazy-loaded objects
        self._model: Optional[WhisperModel] = None
        self._batched_pipeline: Optional[BatchedInferencePipeline] = None

    def _load_model(self) -> WhisperModel:
        """Load (or return cached) WhisperModel."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _load_batched_pipeline(self) -> BatchedInferencePipeline:
        """Load (or return cached) batched pipeline."""
        if self._batched_pipeline is None:
            model = self._load_model()
            self._batched_pipeline = BatchedInferencePipeline(model=model)
        return self._batched_pipeline

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        beam_size: int = 5,
        batch_size: Optional[int] = None,
        use_batched: Optional[bool] = None,
    ) -> STTResult:
        """Transcribe an audio file into text.

        Args:
            audio_path: Path to a local audio file.
            language: Optional BCP-47 language code (e.g., "en"). If None,
                the model will attempt to auto-detect the language.
            beam_size: Beam size for decoding (higher = more accurate but slower).
            batch_size: Number of segments processed in parallel when using
                batched transcription. If None, falls back to default_batch_size.
            use_batched: Override for this call; if None, uses self.use_batched.

        Returns:
            STTResult object with full text and per-segment metadata.
        """

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        effective_use_batched = self.use_batched if use_batched is None else use_batched
        effective_batch_size = batch_size or self.default_batch_size

        if effective_use_batched:
            pipeline = self._load_batched_pipeline()
            segments_iter, info = pipeline.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                batch_size=effective_batch_size,
            )
        else:
            model = self._load_model()
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
            )

        stt_segments: List[STTSegment] = []
        full_text_parts: List[str] = []

        # segments_iter is a generator – iterating actually runs the transcription
        for seg in segments_iter:
            segment_text = seg.text.strip()
            stt_segments.append(
                STTSegment(
                    start=seg.start,
                    end=seg.end,
                    text=segment_text,
                )
            )
            full_text_parts.append(segment_text)

        full_text = " ".join(full_text_parts).strip()

        return STTResult(
            text=full_text,
            segments=stt_segments,
            language=getattr(info, "language", None),
        )


__all__ = [
    "STTEngine",
    "STTSegment",
    "STTResult",
]
