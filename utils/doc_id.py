"""
utils/doc_id.py — Stable document identifier generation.

Public API
----------
make_doc_id(content_bytes) -> str
    Returns a deterministic UUID-format string derived from a SHA-256 hash
    of the *entire* PDF byte content.

    The original filename is NOT part of the hash. It is stored separately
    in chunk metadata (the "source" field set by text_splitter.split_pages).

    This separation means:
      - Two files with identical byte content always get the same doc_id,
        regardless of their names. This correctly identifies them as the
        same document and skips redundant re-ingestion.
      - Two files with different byte content always get different doc_ids,
        even if only a single byte differs anywhere — including beyond the
        first 64 KB (which the previous implementation missed).

Why the full content, not a 64 KB sample?
------------------------------------------
The old implementation hashed filename + first 64 KB. This created a subtle
collision hazard: two papers that share an identical cover page, abstract,
and introduction (e.g. a preprint and its published version) but differ only
in the later sections would receive the same doc_id. The first paper would be
silently reused when the second was uploaded, corrupting retrieval results.

Hashing the full content costs negligible time for typical PDFs (1–50 MB),
eliminates the collision risk entirely, and makes the behaviour fully
deterministic and provable.

UUID formatting
---------------
The 256-bit SHA-256 digest is truncated to its first 128 bits (32 hex chars)
and formatted as a RFC-4122-style UUID string for readability and ChromaDB
key compatibility. The truncation does not reintroduce collision risk in
practice: 2^128 ≈ 3.4 × 10^38 possible values.
"""

from __future__ import annotations

import hashlib
import uuid


def make_doc_id(content_bytes: bytes) -> str:
    """
    Generate a stable, collision-resistant document identifier from PDF bytes.

    Parameters
    ----------
    content_bytes : bytes
        The *complete* raw bytes of the PDF file.
        Pass an empty bytes object only in tests; pdf_loader validates content.

    Returns
    -------
    str
        A UUID-format string, e.g. "3f2504e0-4f89-11d3-9a0c-0305e82c3301".
        Deterministic: identical content always produces the identical ID.
        Content-addressed: different content always produces a different ID.

    Notes
    -----
    The original filename must be stored separately in chunk metadata
    (via the "source" field in text_splitter.split_pages). It is not part
    of the hash and must not be passed to this function.
    """
    # Hash the full content — no truncation, no filename mixing.
    digest = hashlib.sha256(content_bytes).hexdigest()

    # Format as UUID (8-4-4-4-12). Take the first 32 hex chars = 128 bits.
    doc_id = str(uuid.UUID(digest[:32]))
    return doc_id
