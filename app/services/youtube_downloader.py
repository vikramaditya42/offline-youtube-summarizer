# app/services/youtube_downloader.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Represents the result of a YouTube audio download."""

    url: str
    audio_path: Path
    title: Optional[str] = None
    video_id: Optional[str] = None


class YouTubeDownloader:
    """
    Simple wrapper around yt-dlp to download audio only.

    Default behaviour:
    - Downloads the best available audio
    - Converts it to WAV using ffmpeg
    - Saves into `output_dir` as <video_id>.wav
    """

    def __init__(self, output_dir: str | Path = "data/cache") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_options(self) -> dict:
        """
        Build yt-dlp options for audio-only download.

        Returns:
            Dictionary of options passed to yt_dlp.YoutubeDL.
        """
        outtmpl = str(self.output_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }
        return ydl_opts

    def download_audio(self, url: str) -> DownloadResult:
        """
        Download audio from a YouTube URL and return the local WAV path.

        Args:
            url: Public YouTube video URL.

        Returns:
            DownloadResult with the path to the audio file and some metadata.

        Raises:
            RuntimeError: If download fails for any reason.
        """
        ydl_opts = self._build_options()

        logger.info("Starting download of audio from URL: %s", url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to download audio from %s", url)
                raise RuntimeError(f"Failed to download audio: {exc}") from exc

        video_id = info.get("id")
        title = info.get("title")

        # Because of post-processing, extension should be 'wav'
        audio_path = self.output_dir / f"{video_id}.wav"
        if not audio_path.exists():
            # Fallback: sometimes yt-dlp may not use the video_id template exactly.
            # Try to locate a .wav file in output_dir.
            wav_files = list(self.output_dir.glob("*.wav"))
            if wav_files:
                audio_path = wav_files[-1]  # most recent file
            else:
                raise RuntimeError(
                    f"Audio file not found after download in {self.output_dir}"
                )

        logger.info(
            "Downloaded audio to %s (title=%s, video_id=%s)",
            audio_path,
            title,
            video_id,
        )

        return DownloadResult(
            url=url,
            audio_path=audio_path,
            title=title,
            video_id=video_id,
        )


__all__ = ["YouTubeDownloader", "DownloadResult"]
