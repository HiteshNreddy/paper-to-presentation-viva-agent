"""
core/pdf_pipeline.py — End-to-end PDF processing pipeline.

Public API
----------
process_pdf(source, filename=None, chunk_size=None, chunk_overlap=None) -> PDFProcessingResult
    Validates, extracts, identifies, and chunks an uploaded PDF.

PDFProcessingResult: Pydantic model encapsulating the complete extraction & chunking state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field
from langchain_core.documents import Document

from core.pdf_loader import load_pdf, validate_pdf_source, get_page_count
from core.text_splitter import split_pages
from utils.doc_id import make_doc_id


class PDFPageRecord(BaseModel):
    """Structured record of an individual extracted page."""
    page_number: int = Field(..., ge=1, description="1-based page number")
    character_count: int = Field(..., ge=0, description="Number of text characters extracted")
    text_preview: str = Field(..., description="First 100 characters preview")


class PDFProcessingResult(BaseModel):
    """
    Structured outcome of the PDF processing pipeline.
    Encapsulates all metadata and documents needed for downstream RAG ingestion.
    """
    doc_id: str = Field(..., description="Content-addressed SHA-256 UUID identifier")
    filename: str = Field(..., description="Original filename of the research paper")
    page_count: int = Field(..., ge=1, description="Total pages in the PDF")
    total_characters: int = Field(..., ge=0, description="Total characters extracted across all pages")
    chunk_count: int = Field(..., ge=1, description="Total chunks produced")
    page_records: list[PDFPageRecord] = Field(..., description="Summary of each page")
    chunks: list[dict] = Field(..., description="Serialized chunk contents and metadata")

    model_config = {"frozen": False}


def process_pdf(
    source: Union[str, Path, bytes],
    filename: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[PDFProcessingResult, list[Document]]:
    """
    Execute the full PDF processing pipeline:
    1. Validation (magic bytes, file integrity, encryption check)
    2. Document Identification (SHA-256 of entire file)
    3. Page-wise Text Extraction (PyMuPDF)
    4. Text Chunking (RecursiveCharacterTextSplitter with page attribution)

    Parameters
    ----------
    source : str | Path | bytes
        File path or raw bytes of the uploaded PDF.
    filename : str | None
        Original name of the file (e.g. 'paper.pdf').
    chunk_size : int | None
        Optional chunk size override.
    chunk_overlap : int | None
        Optional chunk overlap override.

    Returns
    -------
    tuple[PDFProcessingResult, list[Document]]
        Structured summary result and the list of LangChain Document objects.

    Raises
    ------
    ValueError : for invalid, empty, encrypted, or corrupted PDFs.
    """
    # ── 1. Read bytes for deterministic content-hashing ─────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        content_bytes = path.read_bytes()
        effective_filename = filename or path.name
    elif isinstance(source, bytes):
        content_bytes = source
        effective_filename = filename or "uploaded_document.pdf"
    else:
        raise TypeError(f"source must be str, Path, or bytes — got {type(source).__name__}")

    # ── 2. Validate PDF structure and security ──────────────────────────────
    validate_pdf_source(content_bytes)

    # ── 3. Content-addressed Document Identification ────────────────────────
    doc_id = make_doc_id(content_bytes)

    # ── 4. Page-by-page text extraction ─────────────────────────────────────
    pages = load_pdf(content_bytes, filename=effective_filename)

    total_chars = sum(len(p["text"]) for p in pages)
    page_records = [
        PDFPageRecord(
            page_number=p["page_number"],
            character_count=len(p["text"]),
            text_preview=p["text"][:100].replace("\n", " ") + ("..." if len(p["text"]) > 100 else "")
        )
        for p in pages
    ]

    # ── 5. Page-level chunking with metadata preservation ───────────────────
    chunks = split_pages(
        pages=pages,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    serialized_chunks = [
        {
            "chunk_index": c.metadata["chunk_index"],
            "page_number": c.metadata["page_number"],
            "source": c.metadata["source"],
            "doc_id": c.metadata["doc_id"],
            "start_index": c.metadata.get("start_index", 0),
            "character_length": len(c.page_content),
            "text": c.page_content,
        }
        for c in chunks
    ]

    result = PDFProcessingResult(
        doc_id=doc_id,
        filename=effective_filename,
        page_count=len(pages),
        total_characters=total_chars,
        chunk_count=len(chunks),
        page_records=page_records,
        chunks=serialized_chunks,
    )

    return result, chunks

