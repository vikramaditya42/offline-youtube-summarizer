# app/api/routes.py

from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import SummarizeRequest, SummarizeResponse
from app.services.pipeline import TranscriptionPipeline
from app.services.summarizer import Phi4Summarizer

router = APIRouter()


def get_pipeline(request: Request) -> TranscriptionPipeline:
    pipeline: TranscriptionPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("TranscriptionPipeline not initialized")
    return pipeline


def get_summarizer(request: Request) -> Phi4Summarizer:
    summarizer: Phi4Summarizer | None = getattr(request.app.state, "summarizer", None)
    if summarizer is None:
        raise RuntimeError("Phi4Summarizer not initialized")
    return summarizer


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_video(
    request: Request,
    payload: SummarizeRequest,
    pipeline: TranscriptionPipeline = Depends(get_pipeline),
    summarizer: Phi4Summarizer = Depends(get_summarizer),
) -> SummarizeResponse:
    """
    Main endpoint:
        YouTube URL -> audio download -> STT -> Phi-4 summary.
    """
    # 1. Acquire lock to ensure we don't run multiple heavy pipelines in parallel (OOM risk)
    lock: asyncio.Lock = request.app.state.lock
    
    async with lock:
        try:
            # Run blocking pipeline in threadpool
            download_result, stt_result = await run_in_threadpool(
                pipeline.run,
                url=str(payload.url),
                language=payload.language,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download or transcribe video: {exc}",
            ) from exc

        try:
            # Run blocking summarization in threadpool
            summary_result = await run_in_threadpool(
                summarizer.summarize_transcript,
                stt_result.text,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to summarize transcript: {exc}",
            ) from exc

    return SummarizeResponse(
        url=payload.url,
        title=download_result.title,
        video_id=download_result.video_id,
        detected_language=stt_result.language,
        transcript_length_chars=len(stt_result.text),
        final_summary=summary_result.final_summary,
        chunk_summaries=summary_result.chunk_summaries,
    )
