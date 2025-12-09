# scripts/summarize_youtube.py

"""
End-to-end CLI:
    YouTube URL -> audio download -> offline STT (Whisper) -> Phi-4 summary.

Usage (GPU):
    python -m scripts.summarize_youtube \\
        --url "https://www.youtube.com/watch?v=XXXX" \\
        --stt-model-size small \\
        --device cuda

Usage (CPU, smaller compute type):
    python -m scripts.summarize_youtube \\
        --url "https://www.youtube.com/watch?v=XXXX" \\
        --stt-model-size small \\
        --device cpu \\
        --stt-compute-type int8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.pipeline import (
    TranscriptionPipeline,
    TranscriptionPipelineConfig,
)
from app.services.summarizer import (
    Phi4Summarizer,
    SummarizerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline YouTube video summarizer (Whisper STT + Phi-4-mini-instruct)."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Public YouTube video URL.",
    )
    # STT options
    parser.add_argument(
        "--stt-model-size",
        type=str,
        default="small",
        help="Whisper model size (tiny, base, small, medium, large-v3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use for STT and Phi-4: "cuda" or "cpu".',
    )
    parser.add_argument(
        "--stt-compute-type",
        type=str,
        default="float16",
        help="Compute type for faster-whisper (float16, int8, int8_float16, etc.).",
    )
    parser.add_argument(
        "--stt-language",
        type=str,
        default=None,
        help="Optional language code for STT (e.g., en). If not set, auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cache",
        help="Directory where audio files will be cached.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="If set, do not delete the downloaded audio file after processing.",
    )

    # Summarization options
    parser.add_argument(
        "--phi-model-name",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Hugging Face model name or local path for Phi-4-mini-instruct.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=8000,
        help="Maximum input tokens per chunk for summarization.",
    )
    parser.add_argument(
        "--max-new-tokens-chunk",
        type=int,
        default=256,
        help="Max new tokens per chunk summary.",
    )
    parser.add_argument(
        "--max-new-tokens-final",
        type=int,
        default=512,
        help="Max new tokens for final global summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transcription pipeline
    stt_config = TranscriptionPipelineConfig(
        model_size=args.stt_model_size,
        device=args.device,
        compute_type=args.stt_compute_type,
        output_dir=output_dir,
    )
    transcription_pipeline = TranscriptionPipeline(config=stt_config)

    download_result, stt_result = transcription_pipeline.run(
        url=args.url,
        language=args.stt_language,
    )

    print("==== Download Info ====")
    print(f"URL      : {download_result.url}")
    print(f"Title    : {download_result.title}")
    print(f"Video ID : {download_result.video_id}")
    print(f"Audio    : {download_result.audio_path}\n")

    print("==== Transcription ====")
    if stt_result.language:
        print(f"Detected language: {stt_result.language}")
    print(f"Transcript length (chars): {len(stt_result.text)}\n")

    # 2) Summarization with Phi-4-mini-instruct
    sum_config = SummarizerConfig(
        model_name=args.phi_model_name,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens_chunk=args.max_new_tokens_chunk,
        max_new_tokens_final=args.max_new_tokens_final,
    )
    summarizer = Phi4Summarizer(config=sum_config)

    summary_result = summarizer.summarize_transcript(stt_result.text)

    print("==== Final Summary ====\n")
    print(summary_result.final_summary)

    print("\n==== Per-chunk Summaries (debug) ====\n")
    for i, chunk_sum in enumerate(summary_result.chunk_summaries, start=1):
        print(f"--- Chunk {i} ---")
        print(chunk_sum)
        print()

    # 3) Cleanup
    if not args.keep_audio:
        try:
            download_result.audio_path.unlink()
            print(f"(cleanup) Deleted audio file: {download_result.audio_path}")
        except OSError:
            print(f"(cleanup) Failed to delete audio file: {download_result.audio_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
