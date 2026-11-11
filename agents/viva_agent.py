"""
agents/viva_agent.py — Viva Q&A Agent.

Generates comprehensive academic viva questions with detailed model answers
strictly grounded in the retrieved research paper content.

Public API
----------
run(retrieval_result=None, doc_id=None, count=20, llm=None) -> VivaOutput

Flow (LCEL Chain)
-----------------
    retrieval_result.context_str
        → VIVA_PROMPT (ChatPromptTemplate)
        → llm (ChatGoogleGenerativeAI)
        → clean markdown fences
        → PydanticOutputParser[VivaOutput]
        → validate Q&A pairs (5–15 or 20–30) and 5 examination categories
        → resolve verified question-level source_pages (no blanket page attribution)
        → attach retrieval_result.sources to VivaOutput
        → retry on ValidationError (up to settings.LLM_MAX_RETRIES)
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
from prompts.viva_prompts import VIVA_PROMPT
from schemas.viva_schema import VivaOutput, VivaQA

logger = logging.getLogger(__name__)


def _clean_json_string(raw: str) -> str:
    """Extract valid JSON from raw text, removing markdown code fences if present."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return text


def _resolve_question_source_pages(qa: VivaQA, available_pages: set[int]) -> list[int]:
    """
    Ensure a question's source_pages contains ONLY verified supporting pages.

    1. If the LLM supplied source_pages, filter to keep only those pages that
       actually exist in the retrieved context (pruning any hallucinated pages).
    2. If source_pages is empty, inspect the question and model answer for explicit
       page markers like [Page X] or p. X that match available retrieved pages.
    3. If no specific supporting page is established, return an empty list [].
       Never assign all retrieved pages to a question.
    """
    supporting_pages: set[int] = set()

    # Case 1: Check LLM-reported source_pages against actual retrieved pages
    if qa.source_pages:
        for p in qa.source_pages:
            if isinstance(p, int) and p in available_pages:
                supporting_pages.add(p)

    # Case 2: If empty, extract explicit page citations from the question and answer text
    if not supporting_pages:
        text = f"{qa.question} {qa.model_answer}"
        matches = re.findall(
            r"(?:\[Page\s*(\d+)\]|\(Page\s*(\d+)\)|(?:\[|\(|\b)p(?:age|\.)\s*(\d+))",
            text,
            re.IGNORECASE,
        )
        for match_tuple in matches:
            for g in match_tuple:
                if g and int(g) in available_pages:
                    supporting_pages.add(int(g))

    return sorted(supporting_pages)


def run(
    retrieval_result: RetrievalResult | None = None,
    doc_id: str | None = None,
    count: int = 20,
    llm: Any | None = None,
) -> VivaOutput:
    """
    Execute the Viva Agent to generate exam/defense Q&A pairs.

    Parameters
    ----------
    retrieval_result : RetrievalResult | None
        Pre-computed retrieval result from core.rag_context.
    doc_id : str | None
        If retrieval_result is omitted, doc_id is used to retrieve context.
    count : int
        Target number of questions (default: 20).
    llm : Any | None
        LLM instance (defaults to shared get_llm()).

    Returns
    -------
    VivaOutput
        Validated set of viva questions, model answers, and source citations.

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
                "VivaAgent requires an active doc_id or pre-computed retrieval_result."
            )
        retrieval_result = retrieve_for_agent("viva", doc_id=doc_id)

    if not retrieval_result.documents or not retrieval_result.context_str.strip():
        raise InsufficientContextError(
            "Cannot generate viva questions: retrieved context is empty."
        )

    # Set of valid page numbers from retrieved chunks for verification
    available_pages = {
        s.page_number for s in retrieval_result.sources if s.page_number > 0
    }

    # ── Step 2: Build Parser and LCEL Chain ──────────────────────────────────
    active_llm = llm if llm is not None else get_llm()
    parser = PydanticOutputParser(pydantic_object=VivaOutput)
    prompt = VIVA_PROMPT.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | active_llm

    # ── Step 3: Invoke with Retry on Validation Errors ───────────────────────
    max_retries = max(1, settings.LLM_MAX_RETRIES + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = chain.invoke({"context": retrieval_result.context_str})
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            cleaned_json = _clean_json_string(raw_text)

            viva_output: VivaOutput = parser.parse(cleaned_json)

            # Resolve verified question-level source pages (no blanket attribution)
            for qa in viva_output.qa_pairs:
                qa.source_pages = _resolve_question_source_pages(qa, available_pages)

            # Attach verified retrieval sources to the validated output for deck-level traceability
            viva_output.sources = list(retrieval_result.sources)
            logger.info(
                "VivaAgent: Successfully generated %d questions on attempt %d",
                len(viva_output.qa_pairs),
                attempt,
            )
            return viva_output

        except DailyQuotaExceededError:
            logger.error("VivaAgent: Daily quota exceeded. Halting retries immediately.")
            raise

        except Exception as exc:
            last_error = exc
            logger.warning(
                "VivaAgent: Attempt %d failed with error: %s. Retrying...",
                attempt,
                exc,
            )

    raise RuntimeError(
        f"VivaAgent failed to produce valid structured output after {max_retries} attempts: {last_error}"
    ) from last_error
