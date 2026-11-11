"""
core/embeddings.py — Local embedding model wrapper.

Uses langchain-huggingface (the officially maintained, non-deprecated
integration) instead of the legacy langchain-community HuggingFaceEmbeddings.

Public API
----------
get_embedding_model() -> HuggingFaceEmbeddings
    Returns a cached, LangChain-compatible embedding object.

embed_texts(texts) -> list[list[float]]
    Thin helper for direct embedding outside of LangChain chains.

Design decisions
----------------
- @st.cache_resource / module-level singleton: the Sentence Transformers
  model is large (~90 MB). Loading it once and reusing it avoids a 2-3 second
  penalty on every Streamlit interaction.
- The model runs fully locally; no API key is needed.
- Caching is done at the module level (not Streamlit-level) so unit tests
  that import this module work without a Streamlit runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

# If the Hugging Face model is already downloaded in the local cache, enable offline
# mode before importing huggingface_hub or sentence_transformers. This prevents
# huggingface_hub from attempting remote HEAD version checks to huggingface.co,
# eliminating DNS timeouts in offline or sandboxed environments and speeding up startup.
_hf_home = os.environ.get("HF_HOME")
_hf_cache_dir = Path(_hf_home) / "hub" if _hf_home else Path.home() / ".cache" / "huggingface" / "hub"
if _hf_cache_dir.exists():
    _cached_models = list(_hf_cache_dir.glob("*MiniLM*"))
    if _cached_models:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

# Module-level singleton — loaded once, shared for the entire process lifetime.
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the shared HuggingFaceEmbeddings instance.

    The model is loaded from the local Hugging Face cache on first call (or
    downloaded if not present) and cached in-memory thereafter. Subsequent
    calls return the already-loaded object instantly.

    Returns
    -------
    HuggingFaceEmbeddings
        A LangChain Embeddings object backed by Sentence Transformers.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine-sim friendly
        )
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings directly (outside of a LangChain chain).

    Parameters
    ----------
    texts : list[str]
        Strings to embed.

    Returns
    -------
    list[list[float]]
        One float vector per input string.

    Raises
    ------
    ValueError : if `texts` is empty.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string directly.

    Parameters
    ----------
    query : str
        Search query or prompt to embed.

    Returns
    -------
    list[float]
        384-dimensional dense float vector.

    Raises
    ------
    ValueError : if `query` is empty or only whitespace.
    """
    if not query or not query.strip():
        raise ValueError("embed_query received an empty or whitespace string.")
    model = get_embedding_model()
    return model.embed_query(query)
