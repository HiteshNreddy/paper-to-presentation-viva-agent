"""
tests/test_vector_store.py — Unit and integration tests for Phase 3:
Embeddings and ChromaDB vector store.

Covers:
1. Embedding generation, dimensions (384), normalization, and validation.
2. ChromaDB document ingestion and metadata preservation.
3. Duplicate ingestion prevention.
4. Semantic similarity retrieval and distance scoring.
5. Multi-session document isolation with real database operations.
6. Target document deletion without cross-session contamination.
7. Persistence across store connection resets / restarts.
8. Error handling and input validation.
"""

import math
import pytest
from pathlib import Path
from langchain_core.documents import Document

from config.settings import settings
from core.embeddings import get_embedding_model, embed_texts, embed_query
from core.vector_store import (
    ingest_documents,
    get_retriever,
    similarity_search,
    similarity_search_with_score,
    is_document_ingested,
    get_document_chunk_count,
    delete_document,
    reset_store_connection,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_chroma_env(tmp_path, monkeypatch):
    """
    Ensure every test runs against an isolated ChromaDB directory,
    resetting the store connection before and after each test.
    """
    temp_chroma_dir = str(tmp_path / "chroma_test_db")
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", temp_chroma_dir)
    reset_store_connection()
    yield temp_chroma_dir
    reset_store_connection()


def _make_sample_chunks(doc_id: str, source_name: str, topic: str, count: int = 3) -> list[Document]:
    """Helper to generate sample Documents with complete metadata."""
    chunks = []
    for i in range(count):
        chunks.append(
            Document(
                page_content=(
                    f"Research paper on {topic}. "
                    f"Section {i+1} provides detailed analysis of {topic} algorithms and experimental benchmarks. "
                    f"Results demonstrate high accuracy and efficiency."
                ),
                metadata={
                    "doc_id": doc_id,
                    "source": source_name,
                    "page_number": i + 1,
                    "chunk_index": i,
                    "start_index": i * 150,
                },
            )
        )
    return chunks


# ── Task 1: Embedding Generation Tests ────────────────────────────────────────

class TestEmbeddingGeneration:
    """Verify Sentence Transformers embeddings model and vector properties."""

    def test_embedding_model_singleton(self):
        """get_embedding_model returns the same cached instance across calls."""
        model_a = get_embedding_model()
        model_b = get_embedding_model()
        assert model_a is model_b

    def test_embed_texts_returns_correct_dimensions(self):
        """all-MiniLM-L6-v2 outputs exactly 384-dimensional dense vectors."""
        texts = ["Sparse attention mechanisms in neural networks", "Efficient transformer architectures"]
        vectors = embed_texts(texts)
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384

    def test_embed_texts_numerical_validity_and_normalization(self):
        """Vectors must contain finite floats and be L2-normalized (length ≈ 1.0)."""
        texts = ["Validation of embedding normalization and finite values."]
        vector = embed_texts(texts)[0]

        # Check finite floats
        for val in vector:
            assert isinstance(val, float)
            assert not math.isnan(val)
            assert not math.isinf(val)

        # Check L2 normalization (cosine-similarity friendly)
        norm = math.sqrt(sum(v ** 2 for v in vector))
        assert pytest.approx(norm, rel=1e-3) == 1.0

    def test_embed_query_valid(self):
        """embed_query returns a valid 384-d vector for a query string."""
        query = "What is the speedup of SparseDocNet?"
        vector = embed_query(query)
        assert len(vector) == 384
        norm = math.sqrt(sum(v ** 2 for v in vector))
        assert pytest.approx(norm, rel=1e-3) == 1.0

    def test_embed_texts_empty_list_raises_value_error(self):
        """Passing an empty list to embed_texts must raise ValueError."""
        with pytest.raises(ValueError, match="empty list"):
            embed_texts([])

    def test_embed_query_empty_or_whitespace_raises_value_error(self):
        """Passing empty or whitespace query to embed_query must raise ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace"):
            embed_query("")
        with pytest.raises(ValueError, match="empty or whitespace"):
            embed_query("   ")


# ── Tasks 2, 3, 4: Ingestion, Retrieval, Isolation & Persistence Tests ────────

class TestChromaVectorStore:
    """Verify ChromaDB document ingestion, retrieval, and session isolation."""

    def test_ingest_documents_and_metadata_preservation(self):
        """Ingested chunks must preserve all metadata fields in ChromaDB."""
        doc_id = "test-doc-id-001"
        chunks = _make_sample_chunks(doc_id, "sparse_doc_net.pdf", "transformer sparsity", count=3)

        count = ingest_documents(chunks, doc_id)
        assert count == 3
        assert is_document_ingested(doc_id) is True
        assert get_document_chunk_count(doc_id) == 3

        # Retrieve and verify metadata preservation
        results = similarity_search("transformer sparsity", doc_id, top_k=3)
        assert len(results) == 3

        retrieved_metadata = results[0].metadata
        assert retrieved_metadata["doc_id"] == doc_id
        assert retrieved_metadata["source"] == "sparse_doc_net.pdf"
        assert "page_number" in retrieved_metadata
        assert "chunk_index" in retrieved_metadata
        assert "start_index" in retrieved_metadata

    def test_duplicate_ingestion_prevention(self):
        """Re-ingesting the same doc_id must be skipped to prevent duplication."""
        doc_id = "duplicate-check-doc-id"
        chunks = _make_sample_chunks(doc_id, "paper.pdf", "deep learning", count=2)

        # First ingestion
        first_count = ingest_documents(chunks, doc_id)
        assert first_count == 2
        assert get_document_chunk_count(doc_id) == 2

        # Second ingestion should be skipped
        second_count = ingest_documents(chunks, doc_id)
        assert second_count == 0
        assert get_document_chunk_count(doc_id) == 2

        # Ingestion with force=True allows explicit re-ingestion
        forced_count = ingest_documents(chunks, doc_id, force=True)
        assert forced_count == 2
        assert get_document_chunk_count(doc_id) == 4

    def test_similarity_search_with_score(self):
        """similarity_search_with_score returns documents and cosine distance scores."""
        doc_id = "scored-search-doc"
        chunks = _make_sample_chunks(doc_id, "paper.pdf", "graph neural networks", count=2)
        ingest_documents(chunks, doc_id)

        scored_results = similarity_search_with_score("graph neural networks", doc_id, top_k=2)
        assert len(scored_results) == 2

        for doc, score in scored_results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            # Chroma cosine distance is >= 0.0 (0.0 is perfect match)
            assert score >= 0.0

    def test_retriever_interface(self):
        """get_retriever returns a working VectorStoreRetriever filtered to doc_id."""
        doc_id = "retriever-interface-doc"
        chunks = _make_sample_chunks(doc_id, "paper.pdf", "quantum computing", count=3)
        ingest_documents(chunks, doc_id)

        retriever = get_retriever(doc_id, top_k=2)
        retrieved_docs = retriever.invoke("quantum algorithms")
        assert len(retrieved_docs) <= 2
        for d in retrieved_docs:
            assert d.metadata["doc_id"] == doc_id

    def test_multi_session_document_isolation(self):
        """
        Ingest Paper A (Session A) and Paper B (Session B).
        Queries for Session A must NEVER return Session B chunks, and vice versa.
        """
        doc_id_a = "session-a-paper-id"
        doc_id_b = "session-b-paper-id"

        chunks_a = _make_sample_chunks(doc_id_a, "alpha.pdf", "astronomy astrophysics", count=3)
        chunks_b = _make_sample_chunks(doc_id_b, "beta.pdf", "biochemistry genetics", count=3)

        ingest_documents(chunks_a, doc_id_a)
        ingest_documents(chunks_b, doc_id_b)

        assert get_document_chunk_count(doc_id_a) == 3
        assert get_document_chunk_count(doc_id_b) == 3

        # Query Session A for Session B content
        results_a = similarity_search("biochemistry genetics", doc_id=doc_id_a, top_k=5)
        for doc in results_a:
            assert doc.metadata["doc_id"] == doc_id_a
            assert doc.metadata["source"] == "alpha.pdf"
            assert "biochemistry" not in doc.page_content

        # Query Session B for Session A content
        results_b = similarity_search("astronomy astrophysics", doc_id=doc_id_b, top_k=5)
        for doc in results_b:
            assert doc.metadata["doc_id"] == doc_id_b
            assert doc.metadata["source"] == "beta.pdf"
            assert "astronomy" not in doc.page_content

    def test_delete_document_preserves_other_sessions(self):
        """Deleting doc_id_a removes only doc_id_a, leaving doc_id_b intact."""
        doc_id_a = "to-be-deleted-id"
        doc_id_b = "to-be-preserved-id"

        chunks_a = _make_sample_chunks(doc_id_a, "delete_me.pdf", "topic a", count=2)
        chunks_b = _make_sample_chunks(doc_id_b, "keep_me.pdf", "topic b", count=2)

        ingest_documents(chunks_a, doc_id_a)
        ingest_documents(chunks_b, doc_id_b)

        assert is_document_ingested(doc_id_a) is True
        assert is_document_ingested(doc_id_b) is True

        # Delete document A
        delete_document(doc_id_a)

        assert is_document_ingested(doc_id_a) is False
        assert get_document_chunk_count(doc_id_a) == 0

        # Document B must remain intact
        assert is_document_ingested(doc_id_b) is True
        assert get_document_chunk_count(doc_id_b) == 2
        results_b = similarity_search("topic b", doc_id_b)
        assert len(results_b) == 2

    def test_persistence_across_connection_resets(self):
        """Simulate an application restart by resetting the store connection."""
        doc_id = "persistent-doc-id"
        chunks = _make_sample_chunks(doc_id, "persistent_paper.pdf", "reinforcement learning", count=3)
        ingest_documents(chunks, doc_id)

        # Simulate restarting the application by disconnecting ChromaDB singleton
        reset_store_connection()

        # Re-check and query: ChromaDB must re-read from disk
        assert is_document_ingested(doc_id) is True
        assert get_document_chunk_count(doc_id) == 3

        reloaded_results = similarity_search("reinforcement learning", doc_id, top_k=2)
        assert len(reloaded_results) == 2
        assert reloaded_results[0].metadata["doc_id"] == doc_id
        assert reloaded_results[0].metadata["source"] == "persistent_paper.pdf"

    def test_error_handling_invalid_inputs(self):
        """Vector store operations validate parameters and raise meaningful errors."""
        doc_id = "valid-doc-id"
        chunks = _make_sample_chunks(doc_id, "paper.pdf", "nlp", count=1)

        # Ingest validation
        with pytest.raises(ValueError, match="docs list is empty"):
            ingest_documents([], doc_id)
        with pytest.raises(ValueError, match="non-empty string"):
            ingest_documents(chunks, "")
        with pytest.raises(ValueError, match="non-empty string"):
            ingest_documents(chunks, "   ")

        # Similarity search validation
        with pytest.raises(ValueError, match="non-empty doc_id"):
            similarity_search("query", "")
        with pytest.raises(ValueError, match="non-empty query"):
            similarity_search("", doc_id)
        with pytest.raises(ValueError, match="non-empty doc_id"):
            similarity_search_with_score("query", "")
        with pytest.raises(ValueError, match="non-empty query"):
            similarity_search_with_score("", doc_id)

        # Delete validation
        with pytest.raises(ValueError, match="non-empty string"):
            delete_document("")

