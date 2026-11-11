"""
core/vector_store.py — ChromaDB vector store management (session-scoped).

Uses langchain-chroma (the dedicated, non-deprecated integration package).

Public API
----------
ingest_documents(docs, doc_id, force=False) -> int
    Embeds and stores Documents tagged with doc_id. Returns chunk count.
    Prevents duplicate ingestion if chunks for doc_id already exist.

get_retriever(doc_id, top_k, agent_query_hint) -> VectorStoreRetriever
    Returns a retriever whose ChromaDB filter is pinned to the given doc_id.
    The caller (rag_context / app.py) holds the doc_id from the session and
    passes it explicitly on every call.

similarity_search(query, doc_id, top_k) -> list[Document]
    Performs similarity search strictly filtered to doc_id.

similarity_search_with_score(query, doc_id, top_k) -> list[tuple[Document, float]]
    Performs similarity search strictly filtered to doc_id, returning (doc, score).

is_document_ingested(doc_id) -> bool
    Returns True if at least one chunk with this doc_id already exists in
    the store. Used to skip redundant re-ingestion of the same paper.

get_document_chunk_count(doc_id) -> int
    Returns the total number of chunks stored for the given doc_id.

delete_document(doc_id) -> None
    Removes all chunks belonging to doc_id. Called when the user uploads a
    new paper so stale chunks do not accumulate indefinitely.

reset_store_connection() -> None
    Resets the cached store instance (useful for testing persistence).

Multi-user / multi-session isolation design
-------------------------------------------
The module-level _active_doc_id global has been removed entirely.

Previously: the module held a single _active_doc_id, which would be shared
across all Streamlit sessions running in the same Python process (a process
is shared by all browser tabs in Streamlit's default threading model).
Session A uploading a new paper would silently overwrite the global and
cause Session B to query the wrong document.

Now: the doc_id is *never stored here*. It is owned by the caller — in
practice by Streamlit's st.session_state in app.py. Every function that
needs a doc_id receives it as an explicit argument. The ChromaDB metadata
filter {"doc_id": {"$eq": doc_id}} is the only isolation mechanism, and it
uses the caller-supplied value, not any module-level state.

This makes it safe to run concurrent Streamlit sessions: each session owns
its own doc_id in st.session_state and passes it to every call site.

The shared Chroma store object (_chroma_store) is still module-level because
the underlying SQLite connection is process-scoped, but it carries no
session-specific state — all scoping is done via the doc_id filter.
"""

from __future__ import annotations

import logging

# Suppress harmless posthog capture signature warning in chromadb telemetry
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# ── Shared store object (process-scoped, no session state) ───────────────────
# This is intentionally NOT a session-level singleton. It holds only the
# connection to ChromaDB; all session-specific scoping is done via doc_id
# filters in every query. It is safe to share across concurrent sessions.
_chroma_store: Chroma | None = None


def _get_store() -> Chroma:
    """Return (or lazily initialise) the process-level Chroma store object."""
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    return _chroma_store


def reset_store_connection() -> None:
    """
    Reset the process-level Chroma store instance to None.
    Forces the next call to _get_store() to reconnect from disk.
    Used for verifying database persistence across restarts.
    """
    global _chroma_store
    _chroma_store = None


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_documents(
    docs: list[Document],
    doc_id: str,
    force: bool = False,
) -> int:
    """
    Embed and store Documents in ChromaDB, tagged with the given doc_id.

    Every chunk in ``docs`` must already have ``metadata["doc_id"] = doc_id``
    set by ``text_splitter.split_pages()``. This function enforces that
    invariant and also stamps the metadata defensively before writing, so the
    calling order (split → ingest) is not load-bearing for correctness.

    If chunks for this doc_id already exist and ``force`` is False, ingestion
    is skipped to prevent duplicate chunks in the database.

    Parameters
    ----------
    docs : list[Document]
        Output of ``core.text_splitter.split_pages(pages, doc_id)``.
    doc_id : str
        Stable identifier for this paper (UUID from utils.doc_id.make_doc_id).
        Caller owns this value and stores it in session state.
    force : bool
        If True, re-ingests even if chunks for this doc_id already exist.

    Returns
    -------
    int
        Number of chunks successfully ingested (0 if skipped as duplicate).

    Raises
    ------
    ValueError   : if ``docs`` is empty or ``doc_id`` is blank.
    RuntimeError : if ChromaDB ingestion fails.
    """
    if not docs:
        raise ValueError("ingest_documents: docs list is empty.")
    if not doc_id.strip():
        raise ValueError("ingest_documents: doc_id must be a non-empty string.")

    # Prevent duplicate ingestion unless forced
    if not force and is_document_ingested(doc_id):
        logger.info(
            "ingest_documents: doc_id=%s is already ingested; skipping duplicate ingestion.",
            doc_id,
        )
        return 0

    # Defensive stamp: ensure every chunk carries the correct doc_id even if
    # the splitter forgot to set it (belt-and-suspenders).
    for chunk in docs:
        chunk.metadata["doc_id"] = doc_id

    store = _get_store()
    try:
        store.add_documents(docs)
    except Exception as exc:
        raise RuntimeError(f"ChromaDB ingestion failed: {exc}") from exc

    logger.info(
        "ingest_documents: ingested %d chunks | doc_id=%s", len(docs), doc_id
    )
    return len(docs)


def get_retriever(
    doc_id: str,
    *,
    top_k: int | None = None,
    agent_query_hint: str = "",
) -> VectorStoreRetriever:
    """
    Return a metadata-filtered VectorStoreRetriever pinned to doc_id.

    The ChromaDB filter ``{"doc_id": {"$eq": doc_id}}`` guarantees that
    retrieved chunks belong exclusively to this document. No other session's
    data can be returned, regardless of what other sessions have ingested.

    Parameters
    ----------
    doc_id : str
        The document identifier owned by the caller's session. Must not be
        empty. If the caller passes an empty string, a ValueError is raised
        immediately rather than issuing an unfiltered query.
    top_k : int | None
        Number of chunks to retrieve. Defaults to settings.RAG_TOP_K.
    agent_query_hint : str
        Informational label used only for logging (e.g. "summarizer").

    Returns
    -------
    VectorStoreRetriever

    Raises
    ------
    ValueError : if ``doc_id`` is blank (prevents unscoped queries).
    """
    if not doc_id.strip():
        raise ValueError(
            "get_retriever requires a non-empty doc_id. "
            "Ensure a PDF has been uploaded and ingested in this session."
        )

    k = top_k if top_k is not None else settings.RAG_TOP_K
    store = _get_store()

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_id": {"$eq": doc_id}},
        },
    )
    logger.debug(
        "get_retriever: agent=%s | doc_id=%s | top_k=%d",
        agent_query_hint or "unspecified",
        doc_id,
        k,
    )
    return retriever


def similarity_search(
    query: str,
    doc_id: str,
    top_k: int | None = None,
) -> list[Document]:
    """
    Search for documents most similar to query, strictly scoped to doc_id.

    Parameters
    ----------
    query : str
        Semantic search text.
    doc_id : str
        Document identifier to filter against.
    top_k : int | None
        Number of documents to return. Defaults to settings.RAG_TOP_K.

    Returns
    -------
    list[Document]
        List of matching Document objects, empty if none match or if doc_id
        is not found.

    Raises
    ------
    ValueError : if `doc_id` is blank or `query` is blank.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("similarity_search requires a non-empty doc_id.")
    if not query or not query.strip():
        raise ValueError("similarity_search requires a non-empty query.")

    k = top_k if top_k is not None else settings.RAG_TOP_K
    store = _get_store()
    try:
        return store.similarity_search(
            query=query,
            k=k,
            filter={"doc_id": {"$eq": doc_id}},
        )
    except Exception as exc:
        logger.error(
            "similarity_search failed for doc_id=%s, query=%r: %s",
            doc_id, query, exc,
        )
        return []


def similarity_search_with_score(
    query: str,
    doc_id: str,
    top_k: int | None = None,
) -> list[tuple[Document, float]]:
    """
    Search for documents most similar to query, strictly scoped to doc_id,
    returning each document alongside its similarity/distance score.

    Note on ChromaDB scores:
    By default, Chroma returns cosine distance (0.0 = identical, 1.0 = orthogonal).
    Lower distance indicates higher semantic similarity.

    Parameters
    ----------
    query : str
        Semantic search text.
    doc_id : str
        Document identifier to filter against.
    top_k : int | None
        Number of documents to return. Defaults to settings.RAG_TOP_K.

    Returns
    -------
    list[tuple[Document, float]]
        List of (Document, score) tuples, empty if none match.

    Raises
    ------
    ValueError : if `doc_id` is blank or `query` is blank.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("similarity_search_with_score requires a non-empty doc_id.")
    if not query or not query.strip():
        raise ValueError("similarity_search_with_score requires a non-empty query.")

    k = top_k if top_k is not None else settings.RAG_TOP_K
    store = _get_store()
    try:
        return store.similarity_search_with_score(
            query=query,
            k=k,
            filter={"doc_id": {"$eq": doc_id}},
        )
    except Exception as exc:
        logger.error(
            "similarity_search_with_score failed for doc_id=%s, query=%r: %s",
            doc_id, query, exc,
        )
        return []


def is_document_ingested(doc_id: str) -> bool:
    """
    Return True if at least one chunk with this doc_id exists in the store.

    Use this to avoid re-ingesting the same paper when the user re-runs the
    Streamlit app without clearing session state.

    Parameters
    ----------
    doc_id : str
        Document identifier to check.

    Returns
    -------
    bool
    """
    if not doc_id or not doc_id.strip():
        return False
    store = _get_store()
    try:
        results = store.get(where={"doc_id": {"$eq": doc_id}}, limit=1)
        return len(results.get("ids", [])) > 0
    except Exception as exc:
        logger.warning(
            "is_document_ingested: ChromaDB check failed for doc_id=%s: %s",
            doc_id, exc,
        )
        return False


def get_document_chunk_count(doc_id: str) -> int:
    """
    Return the total number of chunks stored for the given doc_id.

    Parameters
    ----------
    doc_id : str
        Document identifier to query.

    Returns
    -------
    int
        Count of chunks stored for this doc_id.
    """
    if not doc_id or not doc_id.strip():
        return 0
    store = _get_store()
    try:
        results = store.get(where={"doc_id": {"$eq": doc_id}})
        return len(results.get("ids", []))
    except Exception as exc:
        logger.warning(
            "get_document_chunk_count: ChromaDB query failed for doc_id=%s: %s",
            doc_id, exc,
        )
        return 0


def delete_document(doc_id: str) -> None:
    """
    Delete all chunks belonging to doc_id from the store.

    Called when the user uploads a new paper so the store does not grow
    unbounded. The caller is responsible for clearing doc_id from session
    state after this call.

    Parameters
    ----------
    doc_id : str
        Document identifier whose chunks should be removed.

    Raises
    ------
    ValueError : if ``doc_id`` is blank.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("delete_document: doc_id must be a non-empty string.")

    store = _get_store()
    try:
        # Chroma.delete() accepts a where-filter to remove matching documents.
        store.delete(where={"doc_id": {"$eq": doc_id}})
        logger.info("delete_document: removed chunks for doc_id=%s", doc_id)
    except Exception as exc:
        logger.warning(
            "delete_document: could not remove chunks for doc_id=%s: %s",
            doc_id, exc,
        )
