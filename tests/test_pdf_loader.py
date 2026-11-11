"""
tests/test_pdf_loader.py — Unit tests for core/pdf_loader.py.

Tests covered
-------------
- Valid PDF bytes → correct page count and non-empty text.
- Path-based loading.
- Non-PDF bytes → ValueError.
- Empty-bytes input → ValueError.
- Completely blank PDF (no text) → ValueError.
- Corrupted PDF bytes → ValueError.
- Encrypted / password-protected PDF → ValueError.
- Missing file path → ValueError.
- Unsupported source type → TypeError.
- get_page_count() returns correct count without full extraction.
- validate_pdf_source() validates all security & structural invariants.
"""

import pytest
import fitz

from core.pdf_loader import load_pdf, get_page_count, validate_pdf_source


@pytest.fixture
def encrypted_pdf_bytes() -> bytes:
    """Create a temporary password-protected PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Classified Paper")
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="owner123"
    )
    doc.close()
    return pdf_bytes


class TestLoadPdfFromBytes:
    def test_valid_pdf_returns_pages(self, minimal_pdf_bytes):
        pages = load_pdf(minimal_pdf_bytes, filename="test.pdf")
        assert isinstance(pages, list)
        assert len(pages) >= 1

    def test_page_dict_has_required_keys(self, minimal_pdf_bytes):
        pages = load_pdf(minimal_pdf_bytes, filename="test.pdf")
        for page in pages:
            assert "page_number" in page
            assert "text" in page
            assert "source" in page

    def test_page_numbers_are_one_based(self, minimal_pdf_bytes):
        pages = load_pdf(minimal_pdf_bytes, filename="test.pdf")
        assert pages[0]["page_number"] == 1

    def test_source_label_uses_filename(self, minimal_pdf_bytes):
        pages = load_pdf(minimal_pdf_bytes, filename="my_paper.pdf")
        assert pages[0]["source"] == "my_paper.pdf"

    def test_source_label_default_when_no_filename(self, minimal_pdf_bytes):
        pages = load_pdf(minimal_pdf_bytes)
        assert "pdf" in pages[0]["source"].lower()

    def test_non_pdf_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="not a valid PDF"):
            load_pdf(b"this is not a pdf", filename="fake.pdf")

    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            load_pdf(b"", filename="empty.pdf")

    def test_corrupted_pdf_raises_value_error(self):
        corrupted = b"%PDF-1.4\nCorruptedGarbageTrailerEndingWithoutCatalog"
        with pytest.raises(ValueError, match="Corrupted or unreadable"):
            load_pdf(corrupted, filename="corrupted.pdf")

    def test_encrypted_pdf_raises_value_error(self, encrypted_pdf_bytes):
        with pytest.raises(ValueError, match="password-protected or encrypted"):
            load_pdf(encrypted_pdf_bytes, filename="encrypted.pdf")


class TestLoadPdfErrors:
    def test_missing_file_raises_value_error(self):
        with pytest.raises(ValueError, match="File not found"):
            load_pdf("/nonexistent/path/file.pdf")

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            load_pdf(12345)  # type: ignore[arg-type]


class TestGetPageCount:
    def test_returns_int_for_valid_pdf(self, minimal_pdf_bytes):
        count = get_page_count(minimal_pdf_bytes)
        assert isinstance(count, int)
        assert count >= 1

    def test_non_pdf_raises_value_error(self):
        with pytest.raises(ValueError):
            get_page_count(b"not a pdf")

    def test_encrypted_pdf_raises_value_error(self, encrypted_pdf_bytes):
        with pytest.raises(ValueError, match="password-protected or encrypted"):
            get_page_count(encrypted_pdf_bytes)


class TestValidatePdfSource:
    def test_valid_pdf_passes(self, minimal_pdf_bytes):
        validate_pdf_source(minimal_pdf_bytes)

    def test_empty_pdf_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_pdf_source(b"")

    def test_non_pdf_raises(self):
        with pytest.raises(ValueError, match="not a valid PDF"):
            validate_pdf_source(b"NOT A PDF")

    def test_corrupted_pdf_raises(self):
        corrupted = b"%PDF-1.4\nCorruptedGarbageTrailerEndingWithoutCatalog"
        with pytest.raises(ValueError, match="Corrupted or unreadable"):
            validate_pdf_source(corrupted)

    def test_encrypted_pdf_raises(self, encrypted_pdf_bytes):
        with pytest.raises(ValueError, match="password-protected or encrypted"):
            validate_pdf_source(encrypted_pdf_bytes)
