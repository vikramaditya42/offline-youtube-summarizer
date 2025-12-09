# app/api/schemas.py

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, HttpUrl


class SummarizeRequest(BaseModel):
    url: HttpUrl
    language: Optional[str] = None  # e.g. "en"


class SummarizeResponse(BaseModel):
    url: HttpUrl
    title: Optional[str]
    video_id: Optional[str]

    detected_language: Optional[str]
    transcript_length_chars: int

    final_summary: str
    chunk_summaries: List[str]
