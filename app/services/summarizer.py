# app/services/summarizer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.services.prompts import (
    CHUNK_SUMMARY_USER_PROMPT_TEMPLATE,
    FINAL_SUMMARY_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class SummarizerConfig:
    """
    Configuration for Phi-4-mini-instruct summarizer.
    """

    model_name: str = "microsoft/Phi-4-mini-instruct"
    device: str = "cuda"  # "cuda" or "cpu"
    max_input_tokens: int = 2000          # per chunk
    max_new_tokens_chunk: int = 256       # tokens generated per chunk summary
    max_new_tokens_final: int = 512       # tokens for final global summary
    temperature: float = 0.1
    do_sample: bool = True               # deterministic by default


@dataclass
class SummaryResult:
    """
    Result of summarization.
    """

    final_summary: str
    chunk_summaries: List[str]


class Phi4Summarizer:
    """
    Offline summarizer using microsoft/Phi-4-mini-instruct.

    Strategy for long transcripts:
    - Tokenize transcript and split into chunks of <= max_input_tokens.
    - For each chunk: ask Phi-4 to create a concise summary.
    - If multiple chunk summaries exist: summarize those into a final summary.
    """

    def __init__(self, config: Optional[SummarizerConfig] = None) -> None:
        self.config = config or SummarizerConfig()

        # Load model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Use float16 on GPU, bfloat16/float32 fallback on CPU
        torch_dtype = torch.float16 if self.config.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto" if self.config.device == "cuda" else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Use HF pipeline so we can feed chat-style messages directly,
        # as in the model card example. :contentReference[oaicite:2]{index=2}
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    # ---------- internal helpers ----------

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Tokenize `text` and split into chunks of <= max_input_tokens.

        This keeps us under the context limit and avoids OOM on small GPUs.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"][0]
        max_len = self.config.max_input_tokens

        chunks: List[str] = []
        for start in range(0, len(input_ids), max_len):
            slice_ids = input_ids[start : start + max_len]
            chunk_text = self.tokenizer.decode(
                slice_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk_text.strip())

        return chunks

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """
        Call Phi-4-mini-instruct using chat messages format.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        out = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
        )

        # HF pipeline returns list[dict] with "generated_text"
        text = out[0]["generated_text"]
        return text.strip()

    # ---------- public API ----------

    def summarize_transcript(self, transcript: str) -> SummaryResult:
        """
        Summarize an entire transcript (potentially very long).

        Returns:
            SummaryResult containing the final summary and per-chunk summaries.
        """
        if not transcript.strip():
            return SummaryResult(final_summary="", chunk_summaries=[])

        chunks = self._split_into_chunks(transcript)
        num_chunks = len(chunks)

        system_prompt = SYSTEM_PROMPT

        chunk_summaries: List[str] = []

        logger.info(f"Summarizing {num_chunks} chunks...")

        for idx, chunk in enumerate(chunks, start=1):
            user_prompt = CHUNK_SUMMARY_USER_PROMPT_TEMPLATE.format(
                idx=idx,
                num_chunks=num_chunks,
                chunk=chunk,
            )

            logger.debug(f"Generating summary for chunk {idx}/{num_chunks}")
            summary_i = self._generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=self.config.max_new_tokens_chunk,
            )
            chunk_summaries.append(summary_i)

        # If only one chunk, that summary is the final summary.
        if len(chunk_summaries) == 1:
            return SummaryResult(
                final_summary=chunk_summaries[0],
                chunk_summaries=chunk_summaries,
            )

        # Otherwise, create a higher-level summary.
        logger.info("Combining chunk summaries into final summary...")
        merged = ""
        for i, s in enumerate(chunk_summaries, start=1):
            merged += f"Segment {i} summary:\n{s}\n\n"

        final_user_prompt = FINAL_SUMMARY_USER_PROMPT_TEMPLATE.format(merged=merged)

        final_summary = self._generate(
            system_prompt=system_prompt,
            user_prompt=final_user_prompt,
            max_new_tokens=self.config.max_new_tokens_final,
        )

        return SummaryResult(
            final_summary=final_summary,
            chunk_summaries=chunk_summaries,
        )


__all__ = ["Phi4Summarizer", "SummarizerConfig", "SummaryResult"]
