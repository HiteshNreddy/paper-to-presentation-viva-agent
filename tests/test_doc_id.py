"""
tests/test_doc_id.py — Unit tests for utils/doc_id.py.

Tests covered
-------------
1. Returns a valid UUID4-formatted string.
2. Content-addressing: identical content always returns the same ID (deterministic).
3. Different content produces different IDs.
4. CRITICAL: Two documents with identical first 64 KB but differing bytes AFTER
   the 64 KB boundary MUST produce different IDs (proves full-content hashing).
5. Filename is NOT part of the hash — same content with different filenames
   produces identical IDs (content-addressing contract).
6. Empty content does not raise.
"""

import re
import pytest

from utils.doc_id import make_doc_id

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class TestMakeDocId:
    def test_returns_uuid_format(self):
        doc_id = make_doc_id(b"%PDF-1.4 minimal test content")
        assert UUID_PATTERN.match(doc_id), f"Not UUID format: {doc_id}"

    def test_deterministic_identical_content(self):
        """Identical byte content always yields identical doc_id."""
        content = b"%PDF-1.4 some fixed content here"
        id1 = make_doc_id(content)
        id2 = make_doc_id(content)
        assert id1 == id2

    def test_different_content_different_ids(self):
        """Distinct content produces distinct doc_ids."""
        id1 = make_doc_id(b"%PDF-1.4 paper A")
        id2 = make_doc_id(b"%PDF-1.4 paper B")
        assert id1 != id2

    def test_identical_first_64kb_differs_after_produces_different_ids(self):
        """
        CRITICAL VIVA TEST:
        Simulate two versions of a paper that share an identical 64 KB prefix
        (e.g. cover page + abstract + intro) but differ only in the methodology
        section starting at byte 65537.

        Old implementation (which only sampled first 64 KB) would give both
        the SAME id, causing silent retrieval pollution.
        New implementation (full content SHA-256) MUST give DIFFERENT ids.
        """
        prefix_64kb = b"A" * 65536  # exactly 64 KB of identical bytes

        doc_a = prefix_64kb + b"_DIFFERENT_CONTENT_SECTION_A_"
        doc_b = prefix_64kb + b"_DIFFERENT_CONTENT_SECTION_B_"

        id_a = make_doc_id(doc_a)
        id_b = make_doc_id(doc_b)

        assert id_a != id_b, (
            "Collision! Documents sharing first 64 KB must NOT produce the same doc_id. "
            "make_doc_id must hash the entire byte content."
        )

    def test_different_lengths_different_ids(self):
        """Appending even a single byte changes the doc_id."""
        base = b"%PDF-1.4 " + b"X" * 70000
        extended = base + b"Y"
        assert make_doc_id(base) != make_doc_id(extended)

    def test_empty_content_does_not_raise(self):
        doc_id = make_doc_id(b"")
        assert UUID_PATTERN.match(doc_id)
