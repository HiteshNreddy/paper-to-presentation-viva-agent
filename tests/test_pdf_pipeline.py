"""
tests/test_pdf_pipeline.py — Unit and integration tests for core/pdf_pipeline.py.

Tests covered:
1. Processing an actual multi-page research paper PDF.
2. Verification of page count, total characters, and chunk count.
3. Verification of 1-based page numbers and metadata preservation.
4. Determinism of content-addressed doc_id across paths vs. bytes.
5. Verification of sequential chunk_index and start_index offset.
6. Custom chunk_size and chunk_overlap parameterization.
7. Rejection of corrupted, empty, non-PDF, and password-protected PDFs.
"""

from pathlib import Path
import pytest
import fitz

from core.pdf_pipeline import process_pdf, PDFProcessingResult

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PAPER_PATH = DATA_DIR / "sample_research_paper.pdf"


@pytest.fixture(scope="session")
def sample_paper_path() -> Path:
    """Ensure the sample research paper PDF exists on disk."""
    if not SAMPLE_PAPER_PATH.exists():
        from scripts.generate_sample_paper import generate_sample_paper
        generate_sample_paper()
    return SAMPLE_PAPER_PATH


@pytest.fixture
def encrypted_pdf_bytes() -> bytes:
    """Create a temporary password-protected PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential Research Paper Content")
    # Save encrypted with AES-256
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="password123",
        owner_pw="ownerpass"
    )
    doc.close()
    return pdf_bytes


class TestPDFPipeline:
    def test_process_actual_research_paper(self, sample_paper_path):
        result, chunks = process_pdf(sample_paper_path)

        assert isinstance(result, PDFProcessingResult)
        assert result.page_count == 4
        assert result.filename == "sample_research_paper.pdf"
        assert result.total_characters > 1500
        assert result.chunk_count >= 4
        assert len(chunks) == result.chunk_count

        # Check page records
        assert len(result.page_records) == 4
        for i, page_rec in enumerate(result.page_records, start=1):
            assert page_rec.page_number == i
            assert page_rec.character_count > 100
            assert len(page_rec.text_preview) > 0

    def test_metadata_preservation_and_traceability(self, sample_paper_path):
        result, chunks = process_pdf(sample_paper_path)

        seen_pages = set()
        for idx, chunk in enumerate(chunks):
            meta = chunk.metadata
            assert meta["doc_id"] == result.doc_id
            assert meta["source"] == "sample_research_paper.pdf"
            assert meta["chunk_index"] == idx
            assert 1 <= meta["page_number"] <= 4
            assert "start_index" in meta
            seen_pages.add(meta["page_number"])

        # All 4 pages must have generated at least one chunk
        assert seen_pages == {1, 2, 3, 4}

    def test_doc_id_deterministic_between_path_and_bytes(self, sample_paper_path):
        result_path, _ = process_pdf(sample_paper_path)
        raw_bytes = sample_paper_path.read_bytes()
        result_bytes, _ = process_pdf(raw_bytes, filename="sample_research_paper.pdf")

        assert result_path.doc_id == result_bytes.doc_id
        assert result_path.total_characters == result_bytes.total_characters
        assert result_path.chunk_count == result_bytes.chunk_count

    def test_custom_chunk_parameters(self, sample_paper_path):
        # Default is 1000/200; with smaller chunks (300/50), we should get significantly more chunks
        result_default, chunks_default = process_pdf(sample_paper_path)
        result_small, chunks_small = process_pdf(
            sample_paper_path,
            chunk_size=300,
            chunk_overlap=50
        )

        assert result_small.chunk_count > result_default.chunk_count
        for chunk in chunks_small:
            assert len(chunk.page_content) <= 350  # allows small boundary wiggle

    def test_invalid_chunk_overlap_raises(self, sample_paper_path):
        with pytest.raises(ValueError, match="strictly less than"):
            process_pdf(sample_paper_path, chunk_size=500, chunk_overlap=500)

    def test_empty_pdf_bytes_raises(self):
        with pytest.raises(ValueError, match="empty"):
            process_pdf(b"", filename="empty.pdf")

    def test_non_pdf_content_raises(self):
        with pytest.raises(ValueError, match="not a valid PDF"):
            process_pdf(b"PlainTextNotPDF", filename="fake.pdf")

    def test_corrupted_pdf_raises(self):
        corrupted = b"%PDF-1.4\nCorruptedGarbageTrailerEndingWithoutCatalog"
        with pytest.raises(ValueError, match="Corrupted or unreadable"):
            process_pdf(corrupted, filename="corrupted.pdf")

    def test_encrypted_pdf_rejected(self, encrypted_pdf_bytes):
        with pytest.raises(ValueError, match="password-protected or encrypted"):
            process_pdf(encrypted_pdf_bytes, filename="encrypted.pdf")

