"""
tests/test_viva_agent.py — Unit tests for agents/viva_agent.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.viva_agent import run, _clean_json_string
from core.rag_context import (
    RetrievalDecision,
    RetrievalResult,
    DocumentNotIngestedError,
    InsufficientContextError,
)
from schemas.summary_schema import RetrievedSource
from schemas.viva_schema import VivaOutput


def _build_sample_viva(num_questions: int = 20, source_pages_per_q: list[int] | None = None) -> dict:
    categories = ["conceptual", "methodological", "result-based", "critical", "comparison"]
    difficulties = ["easy", "medium", "hard"]
    qa_pairs = []

    for i in range(1, num_questions + 1):
        cat = categories[(i - 1) % len(categories)]
        diff = difficulties[(i - 1) % len(difficulties)]
        pages = source_pages_per_q if source_pages_per_q is not None else [1]
        qa_pairs.append({
            "question": f"Question {i}: Can you explain the {cat} aspect of this system?",
            "model_answer": f"Answer {i}: The paper establishes {cat} properties using specific empirical derivations.",
            "category": cat,
            "difficulty": diff,
            "source_pages": list(pages),
        })

    return {
        "paper_title": "Attention Is All You Need",
        "qa_pairs": qa_pairs,
    }


def _dummy_retrieval_result(empty: bool = False, page_numbers: list[int] | None = None) -> RetrievalResult:
    decision = RetrievalDecision(
        needs_retrieval=True,
        agent_name="viva",
        constructed_query="test viva query",
        strategy="agent_canonical",
        doc_id="doc-viva-1",
        reason="Testing viva agent",
    )
    if empty:
        return RetrievalResult(
            documents=[],
            context_str="",
            sources=[],
            decision=decision,
            relevance_score=0.0,
        )
    pages = page_numbers or [1, 2]
    sources = [
        RetrievedSource(
            page_number=p,
            chunk_index=i,
            source="paper.pdf",
        )
        for i, p in enumerate(pages)
    ]
    return RetrievalResult(
        documents=[Document(page_content="Viva content on experimental validation.")],
        context_str="Viva content on experimental validation.",
        sources=sources,
        decision=decision,
        relevance_score=0.91,
    )


class TestVivaAgent:
    def test_clean_json_string_with_fences(self):
        sample = _build_sample_viva(20)
        fenced = "```json\n" + json.dumps(sample) + "\n```"
        cleaned = _clean_json_string(fenced)
        assert cleaned.startswith("{")
        assert len(json.loads(cleaned)["qa_pairs"]) == 20

    def test_clean_json_string_without_fences(self):
        raw = json.dumps(_build_sample_viva(20))
        assert _clean_json_string(raw) == raw

    def test_run_success_with_retrieval_result_20_questions(self):
        sample = _build_sample_viva(20, source_pages_per_q=[1])
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert isinstance(output, VivaOutput)
        assert output.paper_title == "Attention Is All You Need"
        assert len(output.qa_pairs) == 20
        assert len(output.sources) == 2
        assert output.sources[0].page_number == 1

        # Verify all 5 categories are represented
        categories_present = {qa.category for qa in output.qa_pairs}
        assert "conceptual" in categories_present
        assert "methodological" in categories_present
        assert "result-based" in categories_present
        assert "critical" in categories_present
        assert "comparison" in categories_present

    def test_accurate_page_attribution_only_assigned_pages(self):
        """Each question retains only its specifically attributed supporting pages."""
        sample = _build_sample_viva(20)
        # Give Q1 page 1, Q2 page 2, Q3 pages 1 & 2
        sample["qa_pairs"][0]["source_pages"] = [1]
        sample["qa_pairs"][1]["source_pages"] = [2]
        sample["qa_pairs"][2]["source_pages"] = [1, 2]

        retrieval = _dummy_retrieval_result(page_numbers=[1, 2, 3])
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert output.qa_pairs[0].source_pages == [1]
        assert output.qa_pairs[1].source_pages == [2]
        assert output.qa_pairs[2].source_pages == [1, 2]
        # Page 3 was retrieved in context, but must NOT be assigned to Q1 or Q2
        assert 3 not in output.qa_pairs[0].source_pages
        assert 3 not in output.qa_pairs[1].source_pages

    def test_missing_citations_yields_empty_list_no_blanket_fallback(self):
        """When LLM omits source_pages and no citations appear in text, source_pages is empty."""
        sample = _build_sample_viva(20, source_pages_per_q=[])
        retrieval = _dummy_retrieval_result(page_numbers=[1, 2, 3])
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert len(output.qa_pairs) == 20
        for qa in output.qa_pairs:
            # Must be empty list, NOT [1, 2, 3] blanket attribution
            assert qa.source_pages == []

    def test_page_extracted_from_answer_text_when_field_empty(self):
        """When source_pages is empty, explicit [Page X] tags in answer text are extracted."""
        sample = _build_sample_viva(20, source_pages_per_q=[])
        sample["qa_pairs"][0]["model_answer"] = "Described in the architecture section [Page 2]."

        retrieval = _dummy_retrieval_result(page_numbers=[1, 2])
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert output.qa_pairs[0].source_pages == [2]
        # Other questions without citations remain empty
        assert output.qa_pairs[1].source_pages == []

    def test_hallucinated_page_numbers_pruned(self):
        """Page numbers cited by the LLM that do NOT exist in retrieved chunks are pruned."""
        sample = _build_sample_viva(20)
        # LLM hallucinates page 99 which was never in the PDF
        sample["qa_pairs"][0]["source_pages"] = [1, 99]

        retrieval = _dummy_retrieval_result(page_numbers=[1, 2])
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert output.qa_pairs[0].source_pages == [1]
        assert 99 not in output.qa_pairs[0].source_pages

    def test_run_success_with_doc_id(self):
        sample = _build_sample_viva(20, source_pages_per_q=[1])
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        with patch("agents.viva_agent.retrieve_for_agent", return_value=retrieval) as mock_ret:
            output = run(doc_id="doc-viva-1", count=20, llm=mock_llm)
            mock_ret.assert_called_once_with("viva", doc_id="doc-viva-1")
            assert len(output.qa_pairs) == 20

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
        sample = _build_sample_viva(20, source_pages_per_q=[1])
        retrieval = _dummy_retrieval_result()
        calls = 0

        def flaky_llm(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content="Parse error: not a JSON string")
            return AIMessage(content=json.dumps(sample))

        mock_llm = RunnableLambda(flaky_llm)
        output = run(retrieval_result=retrieval, count=20, llm=mock_llm)

        assert calls == 2
        assert len(output.qa_pairs) == 20

    def test_exhausted_retries_raises_runtime_error(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content="Always broken JSON"))

        with pytest.raises(RuntimeError, match="VivaAgent failed"):
            run(retrieval_result=retrieval, count=20, llm=mock_llm)
