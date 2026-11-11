"""
agents/slide_generator.py — Slide-Generator Agent.

Generates an academic presentation slide deck (10–12 slides) with structured
bullet points, presenter notes, and verified page references.

Public API
----------
run(retrieval_result=None, doc_id=None, llm=None) -> SlideDeck

Flow (LCEL Chain)
-----------------
    retrieval_result.context_str
        → SLIDE_PROMPT (ChatPromptTemplate)
        → llm (ChatGoogleGenerativeAI)
        → clean markdown fences
        → PydanticOutputParser[SlideDeck]
        → validate sequential slide numbering (5–15 slides, 2–6 bullets)
        → retry on ValidationError (up to settings.LLM_MAX_RETRIES)
        → attach retrieval_result.sources to SlideDeck
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser

from config.settings import settings
from core.llm import get_llm, DailyQuotaExceededError
from core.rag_context import (
    retrieve_for_agent,
    RetrievalResult,
    DocumentNotIngestedError,
    InsufficientContextError,
)
from prompts.slide_prompts import SLIDE_PROMPT
from schemas.slide_schema import SlideDeck

logger = logging.getLogger(__name__)


def _clean_json_string(raw: str) -> str:
    """Extract valid JSON from raw text, removing markdown code fences if present."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return text


def run(
    retrieval_result: RetrievalResult | None = None,
    doc_id: str | None = None,
    llm: Any | None = None,
) -> SlideDeck:
    """
    Execute the Slide-Generator Agent to create a structured presentation deck.

    Parameters
    ----------
    retrieval_result : RetrievalResult | None
        Pre-computed retrieval result from core.rag_context.
    doc_id : str | None
        If retrieval_result is omitted, doc_id is used to retrieve context.
    llm : Any | None
        LLM instance (defaults to shared get_llm()).

    Returns
    -------
    SlideDeck
        Validated slide deck with structured bullets and source citations.

    Raises
    ------
    DocumentNotIngestedError : if both retrieval_result and doc_id are missing.
    InsufficientContextError : if retrieved context is empty or lacks relevance.
    RuntimeError             : if LLM generation or JSON parsing fails after retries.
    """
    # ── Step 1: Resolve Retrieval Context ────────────────────────────────────
    if retrieval_result is None:
        if not doc_id or not doc_id.strip():
            raise DocumentNotIngestedError(
                "SlideGenerator requires an active doc_id or pre-computed retrieval_result."
            )
        retrieval_result = retrieve_for_agent("slide_generator", doc_id=doc_id)

    if not retrieval_result.documents or not retrieval_result.context_str.strip():
        raise InsufficientContextError(
            "Cannot generate slides: retrieved context is empty."
        )

    # ── Step 2: Build Parser and LCEL Chain ──────────────────────────────────
    active_llm = llm if llm is not None else get_llm()
    parser = PydanticOutputParser(pydantic_object=SlideDeck)
    prompt = SLIDE_PROMPT.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | active_llm

    # ── Step 3: Invoke with Retry on Validation Errors ───────────────────────
    max_retries = max(1, settings.LLM_MAX_RETRIES + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = chain.invoke({"context": retrieval_result.context_str})
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            cleaned_json = _clean_json_string(raw_text)

            slide_deck: SlideDeck = parser.parse(cleaned_json)

            # Attach verified retrieval sources to the validated output
            slide_deck.sources = list(retrieval_result.sources)
            logger.info("SlideGenerator: Successfully generated %d slides on attempt %d", len(slide_deck.slides), attempt)
            return slide_deck

        except DailyQuotaExceededError:
            logger.error("SlideGenerator: Daily quota exceeded. Halting retries immediately.")
            raise

        except Exception as exc:
            last_error = exc
            logger.warning(
                "SlideGenerator: Attempt %d failed with error: %s. Retrying...",
                attempt, exc,
            )

    raise RuntimeError(
        f"SlideGeneratorAgent failed to produce valid structured output after {max_retries} attempts: {last_error}"
    ) from last_error
