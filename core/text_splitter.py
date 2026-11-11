"""
core/text_splitter.py — Document chunking for the RAG pipeline.

Public API
----------
split_pages(pages, doc_id, chunk_size=None, chunk_overlap=None) -> list[Document]
    pages  : list[dict] — output of pdf_loader.load_pdf()
    doc_id : str        — stable identifier for this paper (UUID from utils.doc_id)

    Returns a list of LangChain Document objects. Every chunk carries:
        page_content : str  — the chunk text
        metadata     : dict — doc_id, page_number, chunk_index,
                               source (filename), start_index

Design decisions & Viva points
------------------------------
1. Page-level chunking:
   - Chunking is performed on each page's extracted text individually.
   - Benefit: Every chunk is deterministically tied to its exact origin page (`page_number`).
     If chunks were split across merged document text, a chunk spanning pages 3 and 4 would
     have ambiguous page attribution, leading to inaccurate citations.
2. Recursive Character Splitting:
   - Evaluates separators in hierarchical order: `["\n\n", "\n", " ", ""]`.
   - Preserves semantic units (paragraphs and sentences) before breaking words.
3. Traceability Metadata:
   - `doc_id` guarantees multi-user / session isolation in the vector database.
   - `page_number` enables exact page citations in viva answers and summaries.
   - `chunk_index` preserves global document sequence order.
   - `start_index` records character offset on the origin page.
"""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings


def split_pages(
    pages: list[dict],
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split page-level text into overlapping chunks suitable for embedding.

    Parameters
    ----------
    pages : list[dict]
        Output of ``pdf_loader.load_pdf()``.
        Each dict has keys: page_number (int), text (str), source (str).
    doc_id : str
        Stable identifier for the parent document (UUID, stored in metadata).
    chunk_size : int | None
        Characters per chunk. Defaults to settings.CHUNK_SIZE (1000).
    chunk_overlap : int | None
        Character overlap between consecutive chunks. Defaults to settings.CHUNK_OVERLAP (200).

    Returns
    -------
    list[Document]
        LangChain Document objects with chunk text and full traceability metadata.

    Raises
    ------
    ValueError : if ``pages`` is empty, ``doc_id`` is blank, or splitting
                 produces no non-empty chunks.
    """
    if not pages:
        raise ValueError("split_pages: pages list is empty.")
    if not doc_id or not doc_id.strip():
        raise ValueError("split_pages: doc_id must be a non-empty string.")

    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if c_overlap >= c_size:
        raise ValueError(
            f"chunk_overlap ({c_overlap}) must be strictly less than chunk_size ({c_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        length_function=len,
        add_start_index=True,  # adds 'start_index' to metadata for character offset traceability
    )

    documents: list[Document] = []
    global_chunk_index = 0

    for page in pages:
        text: str = page.get("text", "").strip()
        if not text:
            continue  # skip blank / image-only pages

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "page_number": page["page_number"],
                    "source": page["source"],
                }
            ],
        )

        for chunk in chunks:
            chunk.metadata["chunk_index"] = global_chunk_index
            global_chunk_index += 1

        documents.extend(chunks)

    if not documents:
        raise ValueError(
            "Text splitting produced no chunks. "
            "The PDF may contain only images or whitespace."
        )

    return documents
