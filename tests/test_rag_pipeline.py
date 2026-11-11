"""
tests/test_rag_pipeline.py — Unit and integration tests for Phase 4:
Complete RAG Pipeline with Gemini, LCEL, Source Attribution, and Hallucination Control.

Tests covered:
1. Successful retrieval and answer generation with grounded source citations.
2. Insufficient context handling and refusal to hallucinate on missing topics.
3. Conversational / greeting queries routed directly without unnecessary vector retrieval.
4. Query refinement triggered when initial retrieval relevance is weak.
5. Missing document (empty doc_id) raises DocumentNotIngestedError.
6. Empty or whitespace question raises ValueError.
7. Multi-session document isolation within the RAG pipeline.
8. Gemini API failure handling (graceful degradation without crashing).
9. Source attribution verification (exact 1-based page numbers and chunk IDs).
10. LCEL chain construction and execution (prompt | llm | StrOutputParser).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config.settings import settings
from core.pdf_pipeline import process_pdf
from core.rag_context import DocumentNotIngestedError
from core.rag_pipeline import RAGPipeline
from core.vector_store import (
    ingest_documents,
    delete_document,
    reset_store_connection,
)
from schemas.rag_schema import RAGResponse
from schemas.summary_schema import RetrievedSource


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_chroma_db(tmp_path, monkeypatch):
    """Ensure every test runs against an isolated ChromaDB directory."""
    temp_dir = str(tmp_path / "rag_pipeline_test_chroma")
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", temp_dir)
    reset_store_connection()
    yield temp_dir
    reset_store_connection()


@pytest.fixture
def sample_paper_ingested(isolated_chroma_db):
    """Ingest the 4-page sample research paper and return its doc_id."""
    pdf_path = Path("tests/data/sample_research_paper.pdf")
    assert pdf_path.exists(), "sample_research_paper.pdf must exist"
    result, chunks = process_pdf(pdf_path)
    ingest_documents(chunks, result.doc_id)
    return result.doc_id


def _create_mock_llm(return_text: str):
    """Create a LangChain RunnableLambda mock that returns return_text."""
    return RunnableLambda(lambda _: return_text)


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Tests for the end-to-end RAG pipeline."""

    def test_successful_retrieval_and_answer_generation(self, sample_paper_ingested):
        """Pipeline retrieves relevant chunks, cites [Page 2], and returns grounded response."""
        doc_id = sample_paper_ingested
        question = "Explain the proposed SparseDocNet architecture."

        mock_answer = (
            "The proposed SparseDocNet architecture introduces the Adaptive Dynamic Routing (ADR) "
            "layer [Page 2]. It clusters tokens into blocks of size 64 and applies top-k sparse attention [Page 2]."
        )
        mock_llm = _create_mock_llm(mock_answer)

        response: RAGResponse = RAGPipeline.ask(question, doc_id=doc_id, top_k=2, llm=mock_llm)

        assert response.question == question
        assert "SparseDocNet" in response.answer
        assert response.is_grounded is True
        assert response.has_sufficient_context is True
        assert response.error is None
        assert len(response.sources) > 0
        assert any(s.page_number == 2 for s in response.sources)

    def test_insufficient_context_and_hallucination_control(self, sample_paper_ingested):
        """When queried on topics absent from the paper, context is flagged insufficient and sources cleared."""
        doc_id = sample_paper_ingested
        question = "What audio speech recognition accuracy does SparseDocNet report?"

        mock_refusal = (
            "The provided research paper does not contain sufficient information to answer this question. "
            "The paper evaluates exclusively on text benchmarks and does not mention audio or speech recognition."
        )
        mock_llm = _create_mock_llm(mock_refusal)

        response: RAGResponse = RAGPipeline.ask(question, doc_id=doc_id, top_k=2, llm=mock_llm)

        assert response.has_sufficient_context is False
        assert response.is_grounded is False
        assert "does not contain sufficient information" in response.answer.lower()
        # Unsupported answers must NOT claim source attribution
        assert len(response.sources) == 0

    def test_conversational_query_handled_without_vector_retrieval(self, sample_paper_ingested):
        """Greetings like 'hello' bypass vector retrieval and return helpful instructions."""
        doc_id = sample_paper_ingested
        mock_llm = _create_mock_llm("Hello! How can I help?")
        response = RAGPipeline.ask("Hello! Who are you?", doc_id=doc_id, llm=mock_llm)

        assert response.strategy_used == "direct_no_retrieval"
        assert response.assessment.chunk_count_retrieved == 0
        assert len(response.sources) == 0
        assert "AI Research Assistant" in response.answer

    def test_query_refinement_on_weak_retrieval(self, sample_paper_ingested, monkeypatch):
        """When initial query matches poorly, query refinement is triggered."""
        doc_id = sample_paper_ingested
        # A vague question that triggers refinement to locate the methodology section
        question = "What methodology was used?"

        mock_llm = _create_mock_llm("Methodology uses Adaptive Dynamic Routing [Page 2].")
        response = RAGPipeline.ask(question, doc_id=doc_id, top_k=2, llm=mock_llm)

        # Refinement or keyword focus was executed
        assert response.strategy_used in ["user_refined", "agent_canonical"]
        assert response.assessment.chunk_count_retrieved > 0

    def test_missing_document_raises_document_not_ingested_error(self):
        """Calling ask with an empty or whitespace doc_id raises DocumentNotIngestedError."""
        with pytest.raises(DocumentNotIngestedError):
            RAGPipeline.ask("What is SparseDocNet?", doc_id="")
        with pytest.raises(DocumentNotIngestedError):
            RAGPipeline.ask("What is SparseDocNet?", doc_id="   ")

    def test_empty_question_raises_value_error(self, sample_paper_ingested):
        """Passing an empty or whitespace question raises ValueError."""
        doc_id = sample_paper_ingested
        with pytest.raises(ValueError, match="Question cannot be empty"):
            RAGPipeline.ask("", doc_id=doc_id)
        with pytest.raises(ValueError, match="Question cannot be empty"):
            RAGPipeline.ask("   \n\t  ", doc_id=doc_id)

    def test_multi_session_document_isolation(self, sample_paper_ingested):
        """Session A cannot retrieve Session B's document content, and vice versa."""
        doc_id_a = sample_paper_ingested

        # Ingest Paper B in a separate session
        doc_id_b = "session-b-cryptography-id"
        dummy_chunks = [
            Document(
                page_content="Quantum Cryptography utilizes polarized photons for key exchange.",
                metadata={"page_number": 1, "chunk_index": 0, "source": "quantum.pdf", "doc_id": doc_id_b},
            )
        ]
        ingest_documents(dummy_chunks, doc_id_b)

        # Query Session A for Quantum Cryptography
        mock_llm = _create_mock_llm("The paper does not contain information on quantum cryptography.")
        response_a = RAGPipeline.ask("Quantum cryptography photons", doc_id=doc_id_a, llm=mock_llm)

        # Session A must NOT contain chunks from quantum.pdf
        for s in response_a.sources:
            assert s.source != "quantum.pdf"
            assert s.source == "sample_research_paper.pdf"

        # Query Session B for SparseDocNet
        response_b = RAGPipeline.ask("SparseDocNet attention", doc_id=doc_id_b, llm=mock_llm)
        for s in response_b.sources:
            assert s.source != "sample_research_paper.pdf"

    def test_gemini_api_failure_handled_gracefully(self, sample_paper_ingested):
        """When LLM API throws an exception (quota, connection), pipeline returns structured error without crash."""
        doc_id = sample_paper_ingested
        def _raise_error(_):
            raise RuntimeError("ResourceExhausted: 429 Rate limit exceeded.")
        failing_llm = RunnableLambda(_raise_error)

        response = RAGPipeline.ask("What datasets were used?", doc_id=doc_id, llm=failing_llm)

        assert response.is_grounded is False
        assert response.error is not None
        assert "ResourceExhausted" in response.error
        assert "Generation failed due to an LLM API error" in response.answer

    def test_source_attribution_preserves_page_numbers_and_chunks(self, sample_paper_ingested):
        """Retrieved sources have 1-based page numbers, non-negative chunk IDs, and valid filenames."""
        doc_id = sample_paper_ingested
        mock_llm = _create_mock_llm("SparseDocNet was evaluated on arXiv-LongDoc and PubMed-MultiDoc [Page 3].")

        response = RAGPipeline.ask("What datasets were used?", doc_id=doc_id, top_k=2, llm=mock_llm)

        assert len(response.sources) > 0
        for s in response.sources:
            assert isinstance(s, RetrievedSource)
            assert s.page_number >= 1
            assert s.chunk_index >= 0
            assert s.source == "sample_research_paper.pdf"

    def test_lcel_chain_execution(self):
        """build_chain constructs a valid LCEL runnable chain: prompt | llm | StrOutputParser."""
        mock_llm = _create_mock_llm("LCEL output answer.")
        chain = RAGPipeline.build_chain(llm=mock_llm)

        result = chain.invoke({
            "context": "[Chunk 1 | Page 1]\nSample context text.",
            "question": "What is the sample text?",
        })
        assert result == "LCEL output answer."

    def test_chromadb_squared_l2_to_cosine_similarity_conversion(self):
        """
        Verify the mathematical conversion from ChromaDB's default squared L2 distance
        to cosine similarity for unit-normalized vectors.
        Formula: cos(theta) = 1 - (dist / 2.0), clamped to [0.0, 1.0].
        """
        # Identical unit vectors: ||u - v||^2 = 0 -> sem_sim = 1.0
        d_identical = 0.0
        assert max(0.0, 1.0 - (d_identical / 2.0)) == pytest.approx(1.0)

        # 60-degree separation: cos(60) = 0.5 -> ||u - v||^2 = 2 - 2(0.5) = 1.0 -> sem_sim = 0.5
        d_60deg = 1.0
        assert max(0.0, 1.0 - (d_60deg / 2.0)) == pytest.approx(0.5)

        # Orthogonal unit vectors: cos(90) = 0.0 -> ||u - v||^2 = 2.0 -> sem_sim = 0.0
        d_orthogonal = 2.0
        assert max(0.0, 1.0 - (d_orthogonal / 2.0)) == pytest.approx(0.0)

        # Opposed vectors: cos(180) = -1.0 -> ||u - v||^2 = 4.0 -> clamped to 0.0
        d_opposed = 4.0
        assert max(0.0, 1.0 - (d_opposed / 2.0)) == 0.0

    def test_chromadb_collection_default_metric_is_l2(self, isolated_chroma_db):
        """Verify the underlying ChromaDB collection space is default 'l2' (squared Euclidean)."""
        import chromadb.segment.impl.vector.local_hnsw as hnsw
        # When collection metadata is None, Chroma HnswParams assigns space='l2'
        params = hnsw.HnswParams(metadata=None)
        assert params.space == "l2"

    def test_hybrid_scoring_formula(self):
        """Verify composite hybrid scoring: composite = 0.5 * sem_sim + 0.5 * lex_score."""
        # Case A: Strong semantic (dist=0.4 -> sem_sim=0.8) and 100% lexical match (lex_score=1.0)
        sem_sim_a = max(0.0, 1.0 - (0.4 / 2.0))  # 0.8
        lex_score_a = 1.0
        composite_a = (0.5 * sem_sim_a) + (0.5 * lex_score_a)
        assert composite_a == pytest.approx(0.9)
        assert composite_a >= settings.RAG_RELEVANCE_THRESHOLD

        # Case B: Weak semantic (dist=1.7 -> sem_sim=0.15) and 0% lexical match (lex_score=0.0)
        sem_sim_b = max(0.0, 1.0 - (1.7 / 2.0))  # 0.15
        lex_score_b = 0.0
        composite_b = (0.5 * sem_sim_b) + (0.5 * lex_score_b)
        assert composite_b == pytest.approx(0.075)
        # Must be below rejection threshold 0.25
        assert composite_b < 0.25

