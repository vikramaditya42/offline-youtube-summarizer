# scripts/transcribe_youtube.py

"""
Download a YouTube video's audio → transcribe → print transcript.

Usage:
python -m scripts.transcribe_youtube \
    --url "https://www.youtube.com/watch?v=oOxV1wt4OFs" \
    --model-size small \
    --device cpu \
    --compute-type int8 \
    --use-batched \
    --batch-size 8
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
from pathlib import Path
import yt_dlp

from app.services.stt_engine import STTEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube → Offline STT CLI")

    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")

    parser.add_argument("--model-size", type=str, default="small",
                        help="Whisper model size (tiny, base, small, medium, large-v3)")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device: cpu or cuda")

    parser.add_argument("--compute-type", type=str, default="int8",
                        help="Compute type: float16, int8, int8_float16")

    parser.add_argument("--language", type=str, default=None,
                        help="Optional language code (e.g., en)")

    parser.add_argument("--output-dir", type=str, default="data/cache",
                        help="Where to save downloaded audio")

    parser.add_argument("--keep-audio", action="store_true",
                        help="Do not delete temporary audio file")

    # 🔥 NEW ARGUMENTS FOR BATCHED TRANSCRIPTION
    parser.add_argument("--use-batched", action="store_true",
                        help="Use BatchedInferencePipeline for faster inference")

    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for batched transcription")

    return parser.parse_args()


def download_audio(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out = output_dir / "%(title)s.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_out),
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        wav_file = Path(filename).with_suffix(".wav")
        return wav_file


def main() -> None:
    args = parse_args()

    print("Downloading audio...")
    audio_path = download_audio(args.url, Path(args.output_dir))

    print(f"Downloaded to: {audio_path}")

    engine = STTEngine(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        use_batched=args.use_batched,
        default_batch_size=args.batch_size,
    )

    print("Transcribing...")
    result = engine.transcribe(
        audio_path,
        language=args.language,
        beam_size=5,
    )

    print("\n==== TRANSCRIPT ====\n")
    print(result.text)

    if not args.keep_audio:
        try:
            audio_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
