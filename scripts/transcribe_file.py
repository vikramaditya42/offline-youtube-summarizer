from __future__ import annotations


import argparse
from pathlib import Path


from app.services.stt_engine import STTEngine

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline STT test CLI")
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the input audio file (wav/mp3/m4a/etc.)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run on: "cuda" or "cpu"',
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        help="Compute type for faster-whisper (float16, int8, int8_float16, etc.)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code (e.g., en). If not set, auto-detect.",
    )

    parser.add_argument(
        "--use-batched",
        action="store_true",
        help="Use batched transcription (BatchedInferencePipeline).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched transcription.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()


    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")


    engine = STTEngine(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        use_batched=args.use_batched,
        default_batch_size=args.batch_size,
    )

    result = engine.transcribe(
        audio_path,
        language=args.language,
        beam_size=5,
        # batch_size can be left None here, it uses default_batch_size
    )



    print("==== Transcription Result ====")
    if result.language:
        print(f"Detected language: {result.language}")
    print("\nFull transcript:\n")
    print(result.text)


    print("\nFirst 5 segments:")
    for seg in result.segments[:5]:
        print(f"[{seg.start:7.2f}s → {seg.end:7.2f}s] {seg.text}")  


if __name__ == "__main__":
    main()