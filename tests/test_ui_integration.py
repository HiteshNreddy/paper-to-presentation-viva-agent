"""
tests/test_ui_integration.py — Integration tests for Streamlit UI modules and end-to-end flow.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import router
from core.pdf_pipeline import process_pdf
from schemas.concept_schema import ConceptOutput
from schemas.slide_schema import Slide, SlideDeck
from schemas.summary_schema import RetrievedSource, SummaryOutput
from schemas.viva_schema import VivaOutput, VivaQA
from ui.result_views import (
    render_summary,
    render_concepts,
    render_slides,
    render_viva,
)
from ui.sidebar import reset_document_session
from utils.pptx_builder import build_presentation_bytes


@pytest.fixture
def sample_summary() -> SummaryOutput:
    return SummaryOutput(
        title="SparseDocNet: Dynamic Sparse Attention for Documents",
        abstract_summary="SparseDocNet addresses the quadratic computational bottleneck of dense attention.",
        problem_statement="Dense self-attention exhibits quadratic complexity.",
        proposed_solution="Adaptive Dynamic Routing partitions tokens into blocks.",
        key_findings=[
            "Achieves 94.2% accuracy on RVL-CDIP.",
            "Delivers 3.8x inference speedup over BERT.",
        ],
        conclusion="Dynamic sparse attention is an effective paradigm for long documents.",
        sources=[
            RetrievedSource(page_number=1, chunk_index=0, source="sample.pdf"),
            RetrievedSource(page_number=2, chunk_index=1, source="sample.pdf"),
        ],
    )


@pytest.fixture
def sample_concepts() -> ConceptOutput:
    return ConceptOutput(
        key_contributions=["Adaptive Dynamic Routing mechanism."],
        methodology=["Tokens partitioned into blocks of 64."],
        datasets_used=["RVL-CDIP", "DocBank"],
        results_and_metrics=["94.2% classification accuracy."],
        limitations=["Small overhead on very short sequences."],
        future_work=["Extension to autoregressive models."],
        sources=[
            RetrievedSource(page_number=2, chunk_index=1, source="sample.pdf"),
        ],
    )


@pytest.fixture
def sample_deck() -> SlideDeck:
    slides = [
        Slide(
            slide_number=i,
            title=f"Slide {i} Analysis",
            bullets=[f"Point {i}.1 regarding methodology", f"Point {i}.2 regarding evaluation"],
            speaker_note=f"Presenter notes for slide {i}",
        )
        for i in range(1, 11)
    ]
    return SlideDeck(
        paper_title="SparseDocNet: Dynamic Sparse Attention for Documents",
        slides=slides,
        sources=[
            RetrievedSource(page_number=1, chunk_index=0, source="sample.pdf"),
        ],
    )


@pytest.fixture
def sample_viva() -> VivaOutput:
    qa_pairs = [
        VivaQA(
            question=f"Question {i}: Explain aspect {i}",
            model_answer=f"Model answer {i} grounded in paper derivations [Page 1].",
            category="conceptual" if i % 2 == 0 else "methodological",
            difficulty="medium",
            source_pages=[1],
        )
        for i in range(1, 21)
    ]
    return VivaOutput(
        paper_title="SparseDocNet: Dynamic Sparse Attention for Documents",
        qa_pairs=qa_pairs,
        sources=[
            RetrievedSource(page_number=1, chunk_index=0, source="sample.pdf"),
        ],
    )


class TestUIIntegration:
    def test_ui_imports(self):
        """Ensure all UI and app modules import without errors."""
        import app
        import ui.sidebar
        import ui.result_views
        import utils.pptx_builder
        assert hasattr(app, "main")
        assert hasattr(ui.sidebar, "render_sidebar")
        assert hasattr(ui.result_views, "render_summary")

    def test_reset_document_session(self):
        """Test reset_document_session removes all session keys."""
        import streamlit as st
        mock_state = {
            "doc_id": "test-doc-123",
            "doc_info": {"filename": "paper.pdf"},
            "summary_output": "cached_summary",
            "concept_output": "cached_concepts",
            "slide_deck": "cached_deck",
            "pptx_bytes": b"bytes",
            "viva_output": "cached_viva",
            "chat_history": [{"role": "user", "content": "hi"}],
            "keep_me": "persists",
        }
        with patch.object(st, "session_state", mock_state):
            reset_document_session()
            assert "doc_id" not in mock_state
            assert "summary_output" not in mock_state
            assert "viva_output" not in mock_state
            assert mock_state["keep_me"] == "persists"

    def test_render_summary_view_succeeds(self, sample_summary):
        """render_summary executes without raising exceptions."""
        with patch("streamlit.markdown") as mock_md, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.success") as mock_succ, \
             patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]):
            render_summary(sample_summary)
            mock_info.assert_called_once_with(sample_summary.abstract_summary)

    def test_render_concepts_view_succeeds(self, sample_concepts):
        """render_concepts executes without raising exceptions."""
        mock_tabs = [MagicMock() for _ in range(6)]
        with patch("streamlit.markdown"), \
             patch("streamlit.tabs", return_value=mock_tabs):
            render_concepts(sample_concepts)

    def test_render_slides_view_succeeds(self, sample_deck):
        """render_slides executes and generates PPTX bytes without crashing."""
        raw_pptx = build_presentation_bytes(sample_deck)
        with patch("streamlit.markdown"), \
             patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]), \
             patch("streamlit.download_button") as mock_dl, \
             patch("streamlit.container"):
            render_slides(sample_deck, pptx_bytes=raw_pptx)
            mock_dl.assert_called_once()
            call_kwargs = mock_dl.call_args.kwargs
            assert "pptx" in call_kwargs["file_name"]

    def test_render_viva_view_succeeds(self, sample_viva):
        """render_viva executes, supports category filtering, and renders questions."""
        with patch("streamlit.markdown"), \
             patch("streamlit.selectbox", return_value="All Categories"), \
             patch("streamlit.expander"):
            render_viva(sample_viva)

    def test_router_dispatch_end_to_end(self, sample_summary):
        """Router dispatch properly delegates to agent and returns typed result."""
        with patch("agents.summarizer.run", return_value=sample_summary) as mock_sum:
            res = router.dispatch("summarize", doc_id="test-doc-id")
            mock_sum.assert_called_once_with(retrieval_result=None, doc_id="test-doc-id", llm=None)
            assert isinstance(res, SummaryOutput)
            assert res.title == sample_summary.title

    def test_document_switching_isolation(self, minimal_pdf_bytes):
        """Different document bytes produce different doc_ids and isolated metadata."""
        doc_a_bytes = minimal_pdf_bytes
        doc_b_bytes = minimal_pdf_bytes + b"\n%extra trailing content"

        res_a, chunks_a = process_pdf(doc_a_bytes, filename="paper_a.pdf")
        res_b, chunks_b = process_pdf(doc_b_bytes, filename="paper_b.pdf")

        assert res_a.doc_id != res_b.doc_id
        assert chunks_a[0].metadata["doc_id"] == res_a.doc_id
        assert chunks_b[0].metadata["doc_id"] == res_b.doc_id

