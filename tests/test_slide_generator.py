"""
tests/test_slide_generator.py — Unit tests for agents/slide_generator.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.slide_generator import run, _clean_json_string
from core.rag_context import (
    RetrievalDecision,
    RetrievalResult,
    DocumentNotIngestedError,
    InsufficientContextError,
)
from schemas.slide_schema import SlideDeck
from schemas.summary_schema import RetrievedSource


def _build_sample_deck(num_slides: int = 10) -> dict:
    slides = []
    titles = [
        "Title Slide",
        "Introduction & Motivation",
        "Problem Statement",
        "Research Objectives",
        "Related Work",
        "Proposed Architecture",
        "Methodology Details",
        "Implementation",
        "Experimental Evaluation",
        "Conclusion & Future Scope",
        "Extra Discussion",
        "Appendix",
    ]
    for i in range(1, num_slides + 1):
        slides.append({
            "slide_number": i,
            "title": titles[i - 1] if i - 1 < len(titles) else f"Slide {i}",
            "bullets": [
                f"Key discussion point {i}.1 regarding the architecture.",
                f"Key experimental insight {i}.2 on baseline comparison.",
                f"Methodological takeaway {i}.3 from the study.",
            ],
            "speaker_note": f"Presenter should elaborate on slide {i} mechanisms.",
        })
    return {
        "paper_title": "Attention Is All You Need",
        "slides": slides,
    }


def _dummy_retrieval_result(empty: bool = False) -> RetrievalResult:
    decision = RetrievalDecision(
        needs_retrieval=True,
        agent_name="slide_generator",
        constructed_query="test slide query",
        strategy="agent_canonical",
        doc_id="doc-789",
        reason="Testing slide generator",
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
            source="paper.pdf",
        ),
        RetrievedSource(
            page_number=2,
            chunk_index=1,
            source="paper.pdf",
        ),
    ]
    return RetrievalResult(
        documents=[Document(page_content="Slide content on Transformer architecture.")],
        context_str="Slide content on Transformer architecture.",
        sources=sources,
        decision=decision,
        relevance_score=0.85,
    )


class TestSlideGeneratorAgent:
    def test_clean_json_string_with_fences(self):
        sample = _build_sample_deck(10)
        fenced = "```json\n" + json.dumps(sample) + "\n```"
        cleaned = _clean_json_string(fenced)
        assert cleaned.startswith("{")
        assert len(json.loads(cleaned)["slides"]) == 10

    def test_clean_json_string_without_fences(self):
        raw = json.dumps(_build_sample_deck(10))
        assert _clean_json_string(raw) == raw

    def test_run_success_with_retrieval_result(self):
        sample = _build_sample_deck(10)
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert isinstance(output, SlideDeck)
        assert output.paper_title == "Attention Is All You Need"
        assert len(output.slides) == 10
        assert output.slides[0].slide_number == 1
        assert len(output.slides[0].bullets) == 3
        assert output.slides[0].speaker_note is not None
        assert len(output.sources) == 2
        assert output.sources[0].page_number == 1

    def test_run_success_with_doc_id(self):
        sample = _build_sample_deck(10)
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content=json.dumps(sample)))

        with patch("agents.slide_generator.retrieve_for_agent", return_value=retrieval) as mock_ret:
            output = run(doc_id="doc-789", llm=mock_llm)
            mock_ret.assert_called_once_with("slide_generator", doc_id="doc-789")
            assert len(output.slides) == 10

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
        sample = _build_sample_deck(10)
        retrieval = _dummy_retrieval_result()
        calls = 0

        def flaky_llm(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content="Invalid response")
            return AIMessage(content=json.dumps(sample))

        mock_llm = RunnableLambda(flaky_llm)
        output = run(retrieval_result=retrieval, llm=mock_llm)

        assert calls == 2
        assert len(output.slides) == 10

    def test_exhausted_retries_raises_runtime_error(self):
        retrieval = _dummy_retrieval_result()
        mock_llm = RunnableLambda(lambda _: AIMessage(content="Always broken JSON"))

        with pytest.raises(RuntimeError, match="SlideGeneratorAgent failed"):
            run(retrieval_result=retrieval, llm=mock_llm)

