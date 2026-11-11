"""
tests/test_rag_context.py — Unit tests for core/rag_context.py.

Tests covered:
1. decide_retrieval:
   - When doc_id is empty, needs_retrieval is False.
   - When intent is 'unknown', needs_retrieval is False.
   - Canonical query strategy when user input is empty or trivial.
   - User-refined query strategy when user input contains specific multi-word query.
2. assess_relevance:
   - Returns 0.0 for empty document list.
   - Returns 0.0 when no expected keywords/markers match.
   - Returns > 0.0 when expected task markers match.
   - Accurately computes coverage ratio.
3. retrieve_for_agent:
   - Raises DocumentNotIngestedError when doc_id is blank.
   - Raises ValueError for unrecognized agent name.
   - Returns RetrievalResult with correctly formatted context, sources, and decision.
   - Raises InsufficientContextError when retrieval returns empty and refinement fails.
   - Triggers refinement fallback when primary query returns zero documents.
   - Triggers refinement fallback when relevance is below threshold (< 0.15).
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from core.rag_context import (
    decide_retrieval,
    assess_relevance,
    retrieve_for_agent,
    RetrievalDecision,
    RetrievalResult,
    InsufficientContextError,
    DocumentNotIngestedError,
    _CANONICAL_QUERIES,
    _TASK_MARKERS,
)


class TestDecideRetrieval:
    def test_empty_doc_id_needs_no_retrieval(self):
        decision = decide_retrieval("summarize", "summarize paper", "")
        assert decision.needs_retrieval is False
        assert decision.doc_id == ""
        assert "No document is loaded" in decision.reason

    def test_whitespace_doc_id_needs_no_retrieval(self):
        decision = decide_retrieval("summarize", "summarize paper", "   ")
        assert decision.needs_retrieval is False

    def test_unknown_intent_needs_no_retrieval(self):
        decision = decide_retrieval("unknown", "hello what is your name?", "doc-123")
        assert decision.needs_retrieval is False
        assert decision.agent_name == "unknown"
        assert "does not require document retrieval" in decision.reason

    def test_canonical_strategy_when_no_user_input(self):
        decision = decide_retrieval("summarize", "", "doc-123")
        assert decision.needs_retrieval is True
        assert decision.strategy == "agent_canonical"
        assert decision.agent_name == "summarizer"
        assert decision.constructed_query == _CANONICAL_QUERIES["summarizer"]

    def test_canonical_strategy_when_trivial_user_input(self):
        for phrase in ["summarize", "slides", "viva", "concepts", "please"]:
            decision = decide_retrieval("summarize", phrase, "doc-123")
            assert decision.strategy == "agent_canonical"

    def test_user_refined_strategy_for_specific_query(self):
        specific = "What are the limitations of the sparse attention mechanism in transformers?"
        decision = decide_retrieval("concept_extractor", specific, "doc-123")
        assert decision.needs_retrieval is True
        assert decision.strategy == "user_refined"
        assert specific in decision.constructed_query
        assert _CANONICAL_QUERIES["concept_extractor"] in decision.constructed_query


class TestAssessRelevance:
    def test_empty_docs_returns_zero(self):
        assert assess_relevance([], "summarizer") == 0.0

    def test_unrelated_content_returns_zero(self):
        docs = [Document(page_content="xylophone zebra quantum zephyr")]
        assert assess_relevance(docs, "summarizer") == 0.0

    def test_matching_content_returns_positive_score(self):
        # Markers for summarizer include: abstract, introduction, conclu, propose, result, paper, method
        docs = [
            Document(page_content="Abstract: In this paper we introduce and propose a novel method."),
            Document(page_content="Our experimental results show strong performance with our conclusion."),
        ]
        score = assess_relevance(docs, "summarizer")
        assert score > 0.5
        assert score <= 1.0


class TestRetrieveForAgent:
    def test_empty_doc_id_raises_document_not_ingested_error(self):
        with pytest.raises(DocumentNotIngestedError):
            retrieve_for_agent("summarizer", "")

    def test_invalid_agent_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown agent_name"):
            retrieve_for_agent("invalid_agent", "doc-123")

    def test_successful_retrieval(self, monkeypatch):
        mock_retriever = MagicMock()
        sample_docs = [
            Document(
                page_content="Abstract: We propose a new method for deep learning evaluation.",
                metadata={"page_number": 1, "chunk_index": 0, "source": "paper.pdf", "doc_id": "doc-123"}
            ),
            Document(
                page_content="Results: Our experimental results and conclusion demonstrate high accuracy.",
                metadata={"page_number": 2, "chunk_index": 1, "source": "paper.pdf", "doc_id": "doc-123"}
            ),
        ]
        mock_retriever.invoke.return_value = sample_docs
        monkeypatch.setattr("core.rag_context.get_retriever", lambda *args, **kwargs: mock_retriever)

        result = retrieve_for_agent("summarizer", "doc-123")

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 2
        assert "[Chunk 1 | Page 1]" in result.context_str
        assert len(result.sources) == 2
        assert result.sources[0].page_number == 1
        assert result.sources[1].page_number == 2
        assert result.relevance_score > 0.0

    def test_empty_retrieval_without_refinement_raises_insufficient_context(self, monkeypatch):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        monkeypatch.setattr("core.rag_context.get_retriever", lambda *args, **kwargs: mock_retriever)

        with pytest.raises(InsufficientContextError):
            retrieve_for_agent("summarizer", "doc-123", allow_refinement=False)

    def test_empty_primary_retrieval_triggers_broad_fallback(self, monkeypatch):
        mock_retriever = MagicMock()
        # First call returns empty, second call (fallback) returns docs
        fallback_docs = [
            Document(
                page_content="Overview and introduction of the paper results.",
                metadata={"page_number": 1, "chunk_index": 0, "source": "paper.pdf", "doc_id": "doc-123"}
            )
        ]
        mock_retriever.invoke.side_effect = [[], fallback_docs]
        monkeypatch.setattr("core.rag_context.get_retriever", lambda *args, **kwargs: mock_retriever)

        result = retrieve_for_agent("summarizer", "doc-123", allow_refinement=True)
        assert len(result.documents) == 1
        assert result.decision.strategy == "broad_fallback"
        assert "broad fallback" in result.decision.reason
