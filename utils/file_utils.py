"""
utils/file_utils.py — File handling utilities for uploads and outputs.

Public API
----------
ensure_directories() -> None
    Create uploads/ and outputs/ directories if they do not exist.

save_upload(uploaded_file) -> tuple[Path, bytes]
    Save a Streamlit UploadedFile to uploads/ and return (path, raw_bytes).

cleanup_uploads(max_age_seconds) -> int
    Remove old files from uploads/; returns count of deleted files.

validate_pdf_bytes(data) -> None
    Raise ValueError if bytes are not a valid PDF (magic-byte check).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Create uploads/ and outputs/ directories if they do not exist."""
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(uploaded_file) -> tuple[Path, bytes]:
    """
    Persist a Streamlit UploadedFile to the uploads/ directory.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The object returned by ``st.file_uploader()``.

    Returns
    -------
    (Path, bytes)
        Absolute path of the saved file and the raw bytes.
        Pass the raw bytes to ``utils.doc_id.make_doc_id()`` to derive
        the session-scoped document identifier.

    Raises
    ------
    ValueError : if the file is not a valid PDF.
    OSError    : if writing to uploads/ fails.
    """
    ensure_directories()

    raw_bytes: bytes = uploaded_file.read()
    validate_pdf_bytes(raw_bytes)

    # Use the original filename; it is already unique per Streamlit session.
    dest: Path = settings.UPLOADS_DIR / uploaded_file.name
    dest.write_bytes(raw_bytes)
    logger.info("Saved upload: %s (%d bytes)", dest, len(raw_bytes))
    return dest, raw_bytes


def cleanup_uploads(max_age_seconds: int = 3600) -> int:
    """
    Remove files from uploads/ older than ``max_age_seconds``.

    Parameters
    ----------
    max_age_seconds : int
        Files older than this many seconds are deleted. Default: 1 hour.

    Returns
    -------
    int
        Number of files deleted.
    """
    ensure_directories()
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for f in settings.UPLOADS_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
                logger.debug("Deleted stale upload: %s", f.name)
            except OSError as exc:
                logger.warning("Could not delete %s: %s", f.name, exc)
    return deleted


def validate_pdf_bytes(data: bytes) -> None:
    """
    Raise ValueError if `data` does not begin with the PDF magic bytes.

    Parameters
    ----------
    data : bytes
        Raw file content to validate.

    Raises
    ------
    ValueError : if the magic bytes are missing.
    """
    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The uploaded file is not a valid PDF (missing '%PDF' header). "
            "Please upload a PDF document."
        )
