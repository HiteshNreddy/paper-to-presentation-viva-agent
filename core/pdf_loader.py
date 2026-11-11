"""
core/pdf_loader.py — PDF ingestion and validation layer using PyMuPDF.

Public API
----------
load_pdf(source, filename) -> list[dict]
    source   : str | Path | bytes  — file path or raw PDF bytes
    filename : str | None          — original filename (used when source is bytes)

    Returns a list of page dicts:
        {
            "page_number": int,   # 1-based index
            "text":        str,   # extracted plain text
            "source":      str,   # filename for traceability
        }

get_page_count(source) -> int
    Returns the total page count without full text extraction.

validate_pdf_source(source) -> None
    Validates magic bytes, non-emptiness, uncorrupted structure, and encryption status.

Design decisions & Viva points
------------------------------
1. Defense-in-depth validation:
   - File level: Verifies the standard %PDF magic byte header.
   - Structure level: Verifies the PDF trailer and catalog are readable by PyMuPDF.
   - Security level: Detects encrypted/password-protected PDFs (`doc.is_encrypted` /
     `doc.needs_pass`) and rejects them immediately with a helpful error.
   - Content level: Rejects scanned/image-only PDFs where zero extractable text exists.
2. Traceability:
   - Text is extracted page-by-page and tagged with 1-based page numbers and the original
     filename, ensuring that all downstream RAG chunks retain exact source page attribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import fitz  # PyMuPDF


def validate_pdf_source(source: Union[str, Path, bytes]) -> None:
    """
    Validate that source is a non-empty, uncorrupted, unencrypted PDF.

    Raises
    ------
    ValueError : if validation fails.
    TypeError  : if source is not str, Path, or bytes.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError("The PDF file is empty (0 bytes).")
        try:
            with open(path, "rb") as f:
                header = f.read(1024)
            _validate_pdf_magic_bytes(header)
            doc = fitz.open(str(path))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable PDF file: {exc}") from exc

    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ValueError("The PDF file is empty (0 bytes).")
        _validate_pdf_magic_bytes(source)
        try:
            doc = fitz.open(stream=source, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable PDF bytes: {exc}") from exc

    else:
        raise TypeError(
            f"source must be str, Path, or bytes — got {type(source).__name__}"
        )

    # Check encryption / password protection
    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise ValueError(
            "The PDF document is password-protected or encrypted. "
            "Please provide an unencrypted PDF."
        )

    if doc.page_count == 0:
        doc.close()
        raise ValueError("The uploaded PDF contains no pages.")

    doc.close()


def load_pdf(
    source: Union[str, Path, bytes],
    filename: str | None = None,
) -> list[dict]:
    """
    Extract plain text from every page of a PDF.

    Parameters
    ----------
    source : str | Path | bytes
        A file-system path or raw PDF bytes (e.g. from Streamlit's file_uploader).
    filename : str | None
        Human-readable filename to embed in metadata. Required when source
        is bytes and caller knows the original name; ignored otherwise.

    Returns
    -------
    list[dict]
        One dict per page: {"page_number": int, "text": str, "source": str}.

    Raises
    ------
    ValueError : empty document, corrupted/encrypted PDF, or no extractable text.
    TypeError  : unsupported source type.
    """
    # ── Step 1: Open and validate document ──────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError("The PDF file is empty (0 bytes).")
        source_label = filename or path.name

        with open(path, "rb") as f:
            header = f.read(1024)
        _validate_pdf_magic_bytes(header)

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable PDF file: {exc}") from exc

    elif isinstance(source, bytes):
        if len(source) == 0:
            raise ValueError("The PDF file is empty (0 bytes).")
        _validate_pdf_magic_bytes(source)
        source_label = filename or "uploaded_document.pdf"

        try:
            doc = fitz.open(stream=source, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable PDF bytes: {exc}") from exc

    else:
        raise TypeError(
            f"source must be str, Path, or bytes — got {type(source).__name__}"
        )

    # ── Step 2: Security & structural checks ────────────────────────────────
    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise ValueError(
            "The PDF document is password-protected or encrypted. "
            "Please provide an unencrypted PDF."
        )

    if doc.page_count == 0:
        doc.close()
        raise ValueError("The uploaded PDF contains no pages.")

    # ── Step 3: Page-by-page text extraction ────────────────────────────────
    pages: list[dict] = []
    for idx in range(doc.page_count):
        page = doc[idx]
        text = page.get_text("text").strip()
        pages.append(
            {
                "page_number": idx + 1,   # 1-based for human readability
                "text": text,
                "source": source_label,
            }
        )
    doc.close()

    # ── Step 4: Reject image-only / empty PDFs ──────────────────────────────
    total_text = " ".join(p["text"] for p in pages).strip()
    if not total_text:
        raise ValueError(
            "No readable text could be extracted from the PDF. "
            "It may be a scanned image-only document. "
            "Please use a text-based PDF."
        )

    return pages


def get_page_count(source: Union[str, Path, bytes]) -> int:
    """
    Return the page count of a PDF without extracting text.

    Parameters
    ----------
    source : str | Path | bytes
        Same as ``load_pdf``.

    Returns
    -------
    int
        Number of pages in the document.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        with open(path, "rb") as f:
            header = f.read(1024)
        _validate_pdf_magic_bytes(header)
        doc = fitz.open(str(path))
    elif isinstance(source, bytes):
        _validate_pdf_magic_bytes(source)
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        raise TypeError(f"source must be str, Path, or bytes — got {type(source).__name__}")

    if doc.is_encrypted or doc.needs_pass:
        doc.close()
        raise ValueError("The PDF document is password-protected or encrypted.")

    count = doc.page_count
    doc.close()
    return count


# ── Internal helpers ────────────────────────────────────────────────────────

def _validate_pdf_magic_bytes(data: bytes) -> None:
    """Raise ValueError if `data` does not begin with the PDF magic bytes (%PDF)."""
    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The uploaded file is not a valid PDF (missing '%PDF' header). "
            "Please upload a valid PDF document."
        )
