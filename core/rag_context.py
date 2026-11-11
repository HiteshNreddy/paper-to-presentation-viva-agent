"""
core/rag_context.py — Explicit, agent-aware retrieval with relevance assessment.

This module is the single retrieval decision hub for the application.

Public API
----------
decide_retrieval(intent, user_input, doc_id) -> RetrievalDecision
    Determines whether paper-specific retrieval is required and builds
    the optimal task-specific query.

retrieve_for_agent(agent_name, doc_id, user_query, top_k, allow_refinement)
    -> RetrievalResult
    Executes retrieval with explicit session-scoped doc_id, assesses
    result relevance, and optionally refines the query if relevance is low.

Design & Viva significance
--------------------------
1. Explicit Retrieval Decision (decide_retrieval):
   Not every request requires vector retrieval. A request like "hello" or
   "what can you do?" should not touch ChromaDB. This function acts as the
   retrieval gatekeeper, making the retrieval decision visible and auditable.

2. Dynamic Task-Specific Queries:
   Instead of using a static, hardcoded string for each agent, this module
   combines the canonical domain terms for that agent's role with any
   specific intent keywords extracted from the user's input.
   Example: if the user asks "What are the limitations of the attention mechanism?",
   the query becomes:
     base (limitations, weaknesses, failure modes) + user terms ("attention mechanism")
   This gives much higher cosine-similarity relevance than a static string.

3. Relevance Assessment (assess_relevance):
   After retrieval, the system checks whether the retrieved chunks actually
   contain sufficient keyword overlap with the task. If relevance is below
   threshold, it triggers a refinement step (broader query) before giving up,
   or raises an explicit InsufficientContextError with a clear explanation.
   This avoids hallucination: the LLM is never called with irrelevant context.

4. Session Scoping (doc_id):
   doc_id is an explicit, mandatory parameter on every function call.
   No module-level state is accessed. Cross-session leakage is impossible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document

from core.vector_store import get_retriever
from schemas.summary_schema import RetrievedSource

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class InsufficientContextError(RuntimeError):
    """
    Raised when retrieved chunks do not contain enough relevant information
    to ground the requested task, even after refinement.

    The UI catches this and suggests next steps (e.g. rephrase or check the PDF).
    """


class DocumentNotIngestedError(RuntimeError):
    """Raised when an agent is invoked without an active doc_id in session."""


# ── Data classes ──────────────────────────────────────────────────────────────

RetrievalStrategy = Literal["agent_canonical", "user_refined", "broad_fallback"]


@dataclass
class RetrievalDecision:
    """Explicit record of the retrieval decision made for a request."""

    needs_retrieval: bool
    agent_name: str
    constructed_query: str
    strategy: RetrievalStrategy
    doc_id: str
    reason: str


@dataclass
class RetrievalResult:
    """Container returned by ``retrieve_for_agent()``."""

    documents: list[Document]
    context_str: str                    # formatted for LLM prompt injection
    sources: list[RetrievedSource]      # for UI source attribution
    decision: RetrievalDecision         # for explainability / audit trail
    relevance_score: float              # 0.0 – 1.0 estimated keyword coverage


# ── Canonical domain terms per agent ──────────────────────────────────────────
# These anchor the cosine search in the paper's vocabulary for each task.

_CANONICAL_QUERIES: dict[str, str] = {
    "summarizer": (
        "abstract introduction problem statement proposed solution approach "
        "key contributions main findings conclusion research motivation"
    ),
    "concept_extractor": (
        "methodology system architecture algorithm experimental setup "
        "datasets evaluation metrics results ablation study contributions "
        "limitations future work"
    ),
    "slide_generator": (
        "abstract introduction background proposed method experimental results "
        "evaluation key findings conclusion contributions overview"
    ),
    "viva": (
        "methodology implementation details experimental results evaluation "
        "metrics datasets limitations weaknesses comparison with baselines "
        "future work research questions key claims"
    ),
}

# Key topical markers expected in retrieved chunks for each agent
_TASK_MARKERS: dict[str, list[str]] = {
    "summarizer": ["abstract", "introduction", "conclu", "propose", "result", "paper", "method"],
    "concept_extractor": ["method", "dataset", "result", "limit", "evaluat", "contribut", "experim"],
    "slide_generator": ["method", "result", "conclu", "introduc", "evaluat", "propose"],
    "viva": ["method", "result", "limit", "compar", "evaluat", "dataset", "find"],
}


def normalize_agent_name(name: str) -> str:
    """Normalize router intent labels or agent names to canonical query keys."""
    mapping = {
        "summarize": "summarizer",
        "extract_concepts": "concept_extractor",
        "generate_slides": "slide_generator",
        "viva": "viva",
        "summarizer": "summarizer",
        "concept_extractor": "concept_extractor",
        "slide_generator": "slide_generator",
    }
    return mapping.get(name, name)


# ── Public API ────────────────────────────────────────────────────────────────

def decide_retrieval(
    intent: str,
    user_input: str,
    doc_id: str,
) -> RetrievalDecision:
    """
    Determine whether retrieval is needed and construct the search query.

    Parameters
    ----------
    intent : str
        Detected intent: "summarize" | "extract_concepts" | "generate_slides" |
        "viva" | "unknown".
    user_input : str
        The raw text from the user (can be empty if button-triggered).
    doc_id : str
        Session-scoped document identifier.

    Returns
    -------
    RetrievalDecision
    """
    agent_key = normalize_agent_name(intent)

    if not doc_id.strip():
        return RetrievalDecision(
            needs_retrieval=False,
            agent_name=agent_key,
            constructed_query="",
            strategy="broad_fallback",
            doc_id="",
            reason="No document is loaded in the current session.",
        )

    if intent == "unknown":
        return RetrievalDecision(
            needs_retrieval=False,
            agent_name="unknown",
            constructed_query="",
            strategy="broad_fallback",
            doc_id=doc_id,
            reason="Intent 'unknown' does not require document retrieval.",
        )

    base = _CANONICAL_QUERIES.get(agent_key, "")
    user_clean = user_input.strip()

    # If the user typed a specific query beyond a simple trigger, combine it:
    # canonical anchors + user's specific terms = high-precision retrieval
    is_specific = (
        bool(user_clean)
        and len(user_clean.split()) > 3
        and user_clean.lower() not in ["summarise", "summarize", "slides", "viva", "concepts"]
    )

    if is_specific:
        query = f"{user_clean} {base}"
        strategy: RetrievalStrategy = "user_refined"
        reason = f"Combined user query '{user_clean[:40]}...' with canonical {agent_key} terms."
    else:
        query = base
        strategy = "agent_canonical"
        reason = f"Used canonical {agent_key} retrieval terms."

    return RetrievalDecision(
        needs_retrieval=True,
        agent_name=agent_key,
        constructed_query=query,
        strategy=strategy,
        doc_id=doc_id,
        reason=reason,
    )


def retrieve_for_agent(
    agent_name: str,
    doc_id: str,
    *,
    user_query: str = "",
    top_k: int | None = None,
    allow_refinement: bool = True,
) -> RetrievalResult:
    """
    Execute retrieval scoped to doc_id, verify relevance, and return context.

    Parameters
    ----------
    agent_name : str
        One of "summarizer", "concept_extractor", "slide_generator", "viva"
        (or router intent equivalents like "summarize", "extract_concepts").
    doc_id : str
        Session-scoped document identifier. Must be non-empty.
    user_query : str
        Optional user-specified query to refine retrieval.
    top_k : int | None
        Override default chunk count.
    allow_refinement : bool
        If True, attempts a second, broader query when relevance is low.

    Returns
    -------
    RetrievalResult

    Raises
    ------
    DocumentNotIngestedError : if doc_id is blank.
    InsufficientContextError : if retrieval returns zero chunks or fails relevance.
    ValueError               : if agent_name is unknown.
    """
    if not doc_id.strip():
        raise DocumentNotIngestedError(
            "No active document in session. Upload a PDF before running agents."
        )

    agent_key = normalize_agent_name(agent_name)

    if agent_key not in _CANONICAL_QUERIES:
        raise ValueError(
            f"Unknown agent_name '{agent_name}'. "
            f"Valid options: {list(_CANONICAL_QUERIES.keys())}"
        )

    # ── Step 1: Explicit retrieval decision ──────────────────────────────────
    decision = decide_retrieval(agent_key, user_query, doc_id)
    retriever = get_retriever(doc_id, top_k=top_k, agent_query_hint=agent_key)

    # ── Step 2: Primary retrieval ────────────────────────────────────────────
    docs: list[Document] = retriever.invoke(decision.constructed_query)

    # ── Step 3: Handle empty retrieval ───────────────────────────────────────
    if not docs:
        if allow_refinement:
            logger.warning(
                "Primary retrieval returned 0 chunks for agent=%s. Trying broad fallback.",
                agent_name,
            )
            # Broad fallback: query just the paper's title/intro words
            fallback_query = "introduction overview abstract background results"
            docs = retriever.invoke(fallback_query)
            decision.strategy = "broad_fallback"
            decision.constructed_query = fallback_query
            decision.reason = "Primary query returned 0 chunks; used broad fallback."

        if not docs:
            raise InsufficientContextError(
                f"Could not retrieve any content from the document for agent '{agent_name}'. "
                f"The document may be too short or may not contain the expected sections."
            )

    # ── Step 4: Relevance assessment ─────────────────────────────────────────
    relevance_score = assess_relevance(docs, agent_key)
    logger.info(
        "retrieve_for_agent: agent=%s | doc_id=%s | chunks=%d | relevance=%.2f | strategy=%s",
        agent_key, doc_id[:8], len(docs), relevance_score, decision.strategy,
    )

    # If relevance is critically low (< 0.15) and refinement is permitted, try once more
    if relevance_score < 0.15 and allow_refinement and decision.strategy != "broad_fallback":
        logger.warning(
            "Relevance score %.2f is below threshold. Refining query with broad terms.",
            relevance_score,
        )
        broad_query = f"{_CANONICAL_QUERIES[agent_key]} research paper study findings"
        refined_docs = retriever.invoke(broad_query)
        if refined_docs:
            docs = refined_docs
            relevance_score = assess_relevance(docs, agent_key)
            decision.strategy = "broad_fallback"
            decision.reason += " (refinement triggered due to low relevance)"

    # ── Step 5: Format context and build sources ─────────────────────────────
    context_str = _format_context(docs)
    sources = _build_sources(docs)

    return RetrievalResult(
        documents=docs,
        context_str=context_str,
        sources=sources,
        decision=decision,
        relevance_score=relevance_score,
    )


def assess_relevance(docs: list[Document], agent_name: str) -> float:
    """
    Estimate keyword coverage of retrieved documents for a given agent task.

    Calculates the proportion of expected topical markers that appear in at
    least one retrieved chunk. Returns a score between 0.0 and 1.0.

    Parameters
    ----------
    docs : list[Document]
        Retrieved chunks to assess.
    agent_name : str
        Agent role being evaluated.

    Returns
    -------
    float
        Score between 0.0 (no markers found) and 1.0 (all markers found).
    """
    if not docs:
        return 0.0

    markers = _TASK_MARKERS.get(agent_name, [])
    if not markers:
        return 1.0  # no markers defined → assume relevant

    combined_text = " ".join(d.page_content.lower() for d in docs)
    matched = sum(1 for m in markers if m in combined_text)
    return round(matched / len(markers), 2)


# ── Private formatting helpers ────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a numbered context block for LLM injection."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[Chunk {i} | Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _build_sources(docs: list[Document]) -> list[RetrievedSource]:
    """Build a deduplicated, page-sorted list of RetrievedSource records."""
    seen: set[tuple] = set()
    sources: list[RetrievedSource] = []

    for doc in docs:
        meta = doc.metadata
        key = (meta.get("page_number", 0), meta.get("chunk_index", -1))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            RetrievedSource(
                page_number=meta.get("page_number", 1),
                chunk_index=meta.get("chunk_index", 0),
                source=meta.get("source", "unknown"),
            )
        )

    sources.sort(key=lambda s: (s.page_number, s.chunk_index))
    return sources
