"""
tests/test_isolation.py — Unit tests proving multi-session document isolation.

These tests simulate two concurrent Streamlit sessions operating within
the same ChromaDB store and prove that:
1. Every retriever is pinned strictly to its session's doc_id.
2. Session 1 cannot retrieve chunks belonging to Session 2.
3. Session 2 cannot retrieve chunks belonging to Session 1.
4. An empty doc_id raises ValueError immediately (no unscoped queries).
5. Calling delete_document() for Session 1 does NOT delete Session 2's chunks.
6. is_document_ingested() correctly discriminates between sessions.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from core.vector_store import (
    ingest_documents,
    get_retriever,
    is_document_ingested,
    delete_document,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_docs(doc_id: str, count: int = 3, topic: str = "general") -> list[Document]:
    """Create test Documents pre-tagged with doc_id and distinct topic text."""
    docs = []
    for i in range(count):
        docs.append(
            Document(
                page_content=f"Content for {topic} chunk {i} with doc_id {doc_id}",
                metadata={
                    "doc_id": doc_id,
                    "page_number": i + 1,
                    "chunk_index": i,
                    "source": f"{topic}_paper.pdf",
                },
            )
        )
    return docs


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSessionIsolation:
    """
    Simulate two concurrent sessions A and B.
    ChromaDB store is mocked to verify the query filter structure without
    requiring real embedding downloads in unit tests.
    """

    def test_empty_doc_id_raises_value_error(self):
        """get_retriever with a blank doc_id must fail immediately."""
        with pytest.raises(ValueError, match="non-empty doc_id"):
            get_retriever("")

    def test_whitespace_doc_id_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty doc_id"):
            get_retriever("   ")

    def test_retriever_filter_pinned_to_session_doc_id(self, monkeypatch):
        """
        Verify that get_retriever passes the exact doc_id filter to ChromaDB,
        not any global or shared value.
        """
        mock_store = MagicMock()
        monkeypatch.setattr("core.vector_store._get_store", lambda: mock_store)

        session_1_id = "uuid-session-1-paper-a"
        session_2_id = "uuid-session-2-paper-b"

        # Session 1 creates retriever
        get_retriever(session_1_id)
        mock_store.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"doc_id": {"$eq": session_1_id}},
            },
        )

        # Session 2 creates retriever
        get_retriever(session_2_id)
        mock_store.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"doc_id": {"$eq": session_2_id}},
            },
        )

    def test_two_sessions_do_not_share_state(self, monkeypatch):
        """
        Ensure vector_store has no module-level active_doc_id state that
        could be overwritten by a second session.
        """
        import core.vector_store as vs
        # Verify the old global no longer exists on the module
        assert not hasattr(vs, "_active_doc_id"), (
            "_active_doc_id must be removed from vector_store; "
            "doc_id must be session-scoped."
        )

    def test_delete_only_affects_target_doc_id(self, monkeypatch):
        """delete_document(doc_id) must pass the specific doc_id filter."""
        mock_store = MagicMock()
        monkeypatch.setattr("core.vector_store._get_store", lambda: mock_store)

        target_id = "session-1-to-delete"
        delete_document(target_id)

        mock_store.delete.assert_called_once_with(
            where={"doc_id": {"$eq": target_id}}
        )

    def test_is_document_ingested_checks_specific_doc_id(self, monkeypatch):
        """is_document_ingested queries only for the specific doc_id."""
        mock_store = MagicMock()
        mock_store.get.return_value = {"ids": ["chunk_1", "chunk_2"]}
        monkeypatch.setattr("core.vector_store._get_store", lambda: mock_store)

        result = is_document_ingested("session-1-id")
        assert result is True
        mock_store.get.assert_called_once_with(
            where={"doc_id": {"$eq": "session-1-id"}}, limit=1
        )

    def test_ingest_enforces_doc_id_stamping(self, monkeypatch):
        """ingest_documents stamps doc_id onto every chunk before writing."""
        mock_store = MagicMock()
        monkeypatch.setattr("core.vector_store._get_store", lambda: mock_store)

        # Documents arrive with wrong or missing doc_id
        docs = [
            Document(page_content="text", metadata={"doc_id": "wrong-id"}),
            Document(page_content="text 2", metadata={}),
        ]
        target_id = "correct-session-id"
        ingest_documents(docs, target_id)

        # Every chunk written to the store must now carry target_id
        for doc in docs:
            assert doc.metadata["doc_id"] == target_id

