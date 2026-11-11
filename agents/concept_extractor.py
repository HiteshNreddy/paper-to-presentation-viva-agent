"""
agents/concept_extractor.py — Concept-Extractor Agent.

Extracts technical contributions, methodology, datasets, results,
limitations, and future scope grounded strictly in the paper chunks.

Public API
----------
run(retrieval_result=None, doc_id=None, llm=None) -> ConceptOutput

Flow (LCEL Chain)
-----------------
    retrieval_result.context_str
        → CONCEPT_PROMPT (ChatPromptTemplate)
        → llm (ChatGoogleGenerativeAI)
        → clean markdown fences
        → PydanticOutputParser[ConceptOutput]
        → retry on ValidationError (up to settings.LLM_MAX_RETRIES)
        → attach retrieval_result.sources to output
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
from prompts.concept_prompts import CONCEPT_PROMPT
from schemas.concept_schema import ConceptOutput

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
) -> ConceptOutput:
    """
    Execute the Concept-Extractor Agent to extract core technical sections.

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
    ConceptOutput
        Structured concept extraction with verified source page records.

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
                "ConceptExtractor requires an active doc_id or pre-computed retrieval_result."
            )
        retrieval_result = retrieve_for_agent("concept_extractor", doc_id=doc_id)

    if not retrieval_result.documents or not retrieval_result.context_str.strip():
        raise InsufficientContextError(
            "Cannot extract concepts: retrieved context is empty."
        )

    # ── Step 2: Build Parser and LCEL Chain ──────────────────────────────────
    active_llm = llm if llm is not None else get_llm()
    parser = PydanticOutputParser(pydantic_object=ConceptOutput)
    prompt = CONCEPT_PROMPT.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | active_llm

    # ── Step 3: Invoke with Retry on Validation Errors ───────────────────────
    max_retries = max(1, settings.LLM_MAX_RETRIES + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = chain.invoke({"context": retrieval_result.context_str})
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            cleaned_json = _clean_json_string(raw_text)

            concept_output: ConceptOutput = parser.parse(cleaned_json)

            # Attach verified retrieval sources to the validated output
            concept_output.sources = list(retrieval_result.sources)
            logger.info("ConceptExtractor: Successfully extracted concepts on attempt %d", attempt)
            return concept_output

        except DailyQuotaExceededError:
            logger.error("ConceptExtractor: Daily quota exceeded. Halting retries immediately.")
            raise

        except Exception as exc:
            last_error = exc
            logger.warning(
                "ConceptExtractor: Attempt %d failed with error: %s. Retrying...",
                attempt, exc,
            )

    raise RuntimeError(
        f"ConceptExtractorAgent failed to produce valid structured output after {max_retries} attempts: {last_error}"
    ) from last_error
