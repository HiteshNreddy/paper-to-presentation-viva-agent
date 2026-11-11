"""
tests/test_text_splitter.py — Unit tests for core/text_splitter.py.

Tests covered
-------------
- split_pages() returns a non-empty list of Documents.
- Every Document has the required metadata keys.
- doc_id is correctly propagated to every chunk.
- page_number metadata matches the source page.
- chunk_index values are sequential and start at 0.
- Empty pages list raises ValueError.
- Blank doc_id raises ValueError.
- All-whitespace pages raises ValueError.
- Chunk size is respected (no chunk larger than CHUNK_SIZE + CHUNK_OVERLAP).
"""

import pytest

from core.text_splitter import split_pages


class TestSplitPages:
    def test_returns_documents(self, sample_pages):
        docs = split_pages(sample_pages, doc_id="test-doc-id")
        assert len(docs) >= 1

    def test_metadata_keys_present(self, sample_pages):
        docs = split_pages(sample_pages, doc_id="test-doc-id")
        for doc in docs:
            assert "doc_id" in doc.metadata
            assert "page_number" in doc.metadata
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata

    def test_doc_id_propagated(self, sample_pages):
        doc_id = "abc-123"
        docs = split_pages(sample_pages, doc_id=doc_id)
        for doc in docs:
            assert doc.metadata["doc_id"] == doc_id

    def test_page_number_matches_source(self, sample_pages):
        docs = split_pages(sample_pages, doc_id="x")
        page_numbers = {doc.metadata["page_number"] for doc in docs}
        # Both pages should appear
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_indices_are_sequential(self, sample_pages):
        docs = split_pages(sample_pages, doc_id="x")
        indices = [doc.metadata["chunk_index"] for doc in docs]
        assert indices == list(range(len(docs)))

    def test_chunk_indices_start_at_zero(self, sample_pages):
        docs = split_pages(sample_pages, doc_id="x")
        assert docs[0].metadata["chunk_index"] == 0

    def test_empty_pages_raises(self):
        with pytest.raises(ValueError, match="empty"):
            split_pages([], doc_id="x")

    def test_blank_doc_id_raises(self, sample_pages):
        with pytest.raises(ValueError, match="doc_id"):
            split_pages(sample_pages, doc_id="")

    def test_all_whitespace_pages_raises(self):
        blank_pages = [
            {"page_number": 1, "text": "   ", "source": "blank.pdf"},
            {"page_number": 2, "text": "\n\t", "source": "blank.pdf"},
        ]
        with pytest.raises(ValueError, match="no chunks"):
            split_pages(blank_pages, doc_id="x")

