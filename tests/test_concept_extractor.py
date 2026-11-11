"""
tests/test_concept_extractor.py — Unit tests for agents/concept_extractor.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.concept_extractor import run, _clean_json_string
from core.rag_context import (
    RetrievalDecision,
    RetrievalResult,
    DocumentNotIngestedError,
    InsufficientContextError,
)
from schemas.concept_schema import ConceptOutput
from schemas.summary_schema import RetrievedSource


SAMPLE_CONCEPT_JSON = {
    "key_contributions": [
        "First transduction model based entirely on self-attention.",
        "Significant training speedup and parallelizability compared to RNNs.",
    ],
    "methodology": [
        "Multi-head self-attention layers with scaled dot-product attention.",
        "Positional encodings added to input embeddings.",
        "Layer normalization and residual connections around sub-layers.",
    ],
    "datasets_used": [
        "WMT 2014 English-to-German translation dataset.",
        "WMT 2014 English-to-French translation dataset.",
    ],
    "results_and_metrics": [
        "28.4 BLEU on WMT 2014 English-to-German.",
        "41.8 BLEU on WMT 2014 English-to-French.",
    ],
    "limitations": [
        "O(n²) memory complexity with sequence length n.",
        "Lack of inductive bias for sequential order requires positional embeddings.",
    ],
    "future_work": [
        "Extend attention mechanisms to images, audio, and video inputs.",
        "Investigate local restricted attention mechanisms for long inputs.",
    ],
}


def _dummy_retrieval_result(empty: bool = False) -> RetrievalResult:
    decision = RetrievalDecision(
        needs_retrieval=True,
        agent_name="concept_extractor",
        constructed_query="test concept query",
        strategy="agent_canonical",
        doc_id="doc-456",
        reason="Testing concept extractor",
    )
    if empty:
        return RetrievalResult(
            documents=[],
            context_str="",
            sources=[],
            decision=decision,
            relevance_score=0.0,
        )
    sources = [
        RetrievedSource(
            page_number=3,
            chunk_index=2,
            source="paper.pdf",
        ),
        RetrievedSource(
            page_number=4,
            chunk_index=3,
            source="paper.pdf",
        ),
    ]
    return RetrievalResult(
        documents=[Document(page_content="Concepts and methodology of Transformer.")],
        context_str="Concepts and methodology of Transformer.",
        sources=sources,
        decision=decision,
        relevance_score=0.88,
    )


class TestConceptExtractorAgent:
    def test_clean_json_string_with_fences(self):
        fenced = "```json\n" + json.dumps(SAMPLE_CONCEPT_JSON) + "\n```"
        cleaned = _clean_json_string(fenced)
        assert cleaned.startswith("{")
        assert len(json.loads(cleaned)["key_contributions"]) == 2

    def test_clean_json_string_without_fences(self):
        raw = json.dumps(SAMPLE_CONCEPT_JSON)
        assert _clean_json_string(raw) == raw

    def test_run_success_with_retrieval_result(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(SAMPLE_CONCEPT_JSON)))

        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert isinstance(output, ConceptOutput)
        assert len(output.key_contributions) == 2
        assert len(output.methodology) == 3
        assert len(output.datasets_used) == 2
        assert len(output.results_and_metrics) == 2
        assert len(output.limitations) == 2
        assert len(output.future_work) == 2
        assert len(output.sources) == 2
        assert output.sources[0].page_number == 3

    def test_run_success_with_doc_id(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(SAMPLE_CONCEPT_JSON)))

        with patch("agents.concept_extractor.retrieve_for_agent", return_value=retrieval) as mock_ret:
            output = run(doc_id="doc-456", llm=mock_llm)
            mock_ret.assert_called_once_with("concept_extractor", doc_id="doc-456")
            assert len(output.key_contributions) == 2

    def test_missing_doc_id_and_retrieval_result_raises(self):
        with pytest.raises(DocumentNotIngestedError):
            run(retrieval_result=None, doc_id=None)

    def test_empty_doc_id_raises(self):
        with pytest.raises(DocumentNotIngestedError):
            run(retrieval_result=None, doc_id="   ")

    def test_empty_retrieval_context_raises(self):
        empty_retrieval = _dummy_retrieval_result(empty=True)
        with pytest.raises(InsufficientContextError):
            run(retrieval_result=empty_retrieval)

    def test_retry_on_initial_failure(self):
        retrieval = _dummy_retrieval_result()
        calls = 0

        def flaky_llm(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content="Malformed text instead of JSON")
            return AIMessage(content=json.dumps(SAMPLE_CONCEPT_JSON))

        mock_llm = RunnableLambda(flaky_llm)
        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert calls == 2
        assert len(output.results_and_metrics) == 2

    def test_exhausted_retries_raises_runtime_error(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content="Never valid JSON"))

        with pytest.raises(RuntimeError, match="ConceptExtractorAgent failed"):
            run(retrieval_result=retrieval, llm=mock_llm)

