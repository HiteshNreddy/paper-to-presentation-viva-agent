"""
tests/test_summarizer.py — Unit tests for agents/summarizer.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.summarizer import run, _clean_json_string
from core.rag_context import (
    RetrievalDecision,
    RetrievalResult,
    DocumentNotIngestedError,
    InsufficientContextError,
)
from schemas.summary_schema import RetrievedSource, SummaryOutput


SAMPLE_SUMMARY_JSON = {
    "title": "Attention Is All You Need",
    "abstract_summary": "The paper introduces the Transformer, a novel sequence-to-sequence architecture based entirely on attention mechanisms.",
    "problem_statement": "Recurrent neural networks suffer from sequential computation bottlenecks.",
    "proposed_solution": "Multi-head self-attention combined with positional encodings.",
    "key_findings": [
        "Achieved 28.4 BLEU on WMT 2014 English-to-German.",
        "Trained significantly faster than recurrent models.",
    ],
    "conclusion": "The Transformer establishes a new state of the art in machine translation.",
}


def _dummy_retrieval_result(empty: bool = False) -> RetrievalResult:
    decision = RetrievalDecision(
        needs_retrieval=True,
        agent_name="summarizer",
        constructed_query="test query",
        strategy="agent_canonical",
        doc_id="doc-123",
        reason="Testing summarizer",
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
            page_number=1,
            chunk_index=0,
            source="transformer.pdf",
        ),
        RetrievedSource(
            page_number=2,
            chunk_index=1,
            source="transformer.pdf",
        ),
    ]
    return RetrievalResult(
        documents=[Document(page_content="Doc content on Transformer architecture.")],
        context_str="Doc content on Transformer architecture.",
        sources=sources,
        decision=decision,
        relevance_score=0.9,
    )


class TestSummarizerAgent:
    def test_clean_json_string_with_fences(self):
        fenced = "```json\n" + json.dumps(SAMPLE_SUMMARY_JSON) + "\n```"
        cleaned = _clean_json_string(fenced)
        assert cleaned.startswith("{")
        assert json.loads(cleaned)["title"] == "Attention Is All You Need"

    def test_clean_json_string_without_fences(self):
        raw = json.dumps(SAMPLE_SUMMARY_JSON)
        assert _clean_json_string(raw) == raw

    def test_run_success_with_retrieval_result(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(SAMPLE_SUMMARY_JSON)))

        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert isinstance(output, SummaryOutput)
        assert output.title == "Attention Is All You Need"
        assert len(output.key_findings) == 2
        assert len(output.sources) == 2
        assert output.sources[0].chunk_index == 0
        assert output.sources[0].page_number == 1

    def test_run_success_with_doc_id(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(SAMPLE_SUMMARY_JSON)))

        with patch("agents.summarizer.retrieve_for_agent", return_value=retrieval) as mock_ret:
            output = run(doc_id="test-doc-id", llm=mock_llm)
            mock_ret.assert_called_once_with("summarizer", doc_id="test-doc-id")
            assert output.title == "Attention Is All You Need"

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
                return AIMessage(content="Invalid non-JSON response")
            return AIMessage(content=json.dumps(SAMPLE_SUMMARY_JSON))

        mock_llm = RunnableLambda(flaky_llm)
        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert calls == 2
        assert output.title == "Attention Is All You Need"

    def test_exhausted_retries_raises_runtime_error(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content="Always broken JSON"))

        with pytest.raises(RuntimeError, match="SummarizerAgent failed"):
            run(retrieval_result=retrieval, llm=mock_llm)

    def test_daily_quota_exceeded_stops_immediately_without_retries(self):
        from core.llm import DailyQuotaExceededError

        retrieval = _dummy_retrieval_result()
        calls = 0

        def quota_failing_llm(_):
            nonlocal calls
            calls += 1
            raise DailyQuotaExceededError("Daily limit reached", retry_delay="16h 50m")

        mock_llm = RunnableLambda(quota_failing_llm)

        with pytest.raises(DailyQuotaExceededError) as exc_info:
            run(retrieval_result=retrieval, llm=mock_llm)

        # Verified: called exactly once, no retry attempts 2 or 3!
        assert calls == 1
        assert exc_info.value.retry_delay == "16h 50m"
