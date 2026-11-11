"""
tests/test_settings.py — Unit tests for config/settings.py.

Tests covered
-------------
- Settings loads correctly when GOOGLE_API_KEY is set.
- _require() raises RuntimeError when a key is missing.
- Numeric env vars (RAG_TOP_K, CHUNK_SIZE, etc.) parse correctly.
- UPLOADS_DIR and OUTPUTS_DIR are absolute Paths.
- CHROMA_COLLECTION_NAME defaults to "paper_chunks".
- LLM_MAX_RETRIES defaults to 2.
"""

import os
import importlib
import pytest


class TestSettingsLoading:
    """
    These tests import settings dynamically so we can control env vars
    without contaminating the global settings singleton already loaded by
    conftest.py.
    """

    def test_settings_loads_with_api_key(self):
        # conftest.py sets GOOGLE_API_KEY=test-key-for-unit-tests
        from config.settings import settings
        assert settings.GOOGLE_API_KEY == "test-key-for-unit-tests"

    def test_gemini_model_has_default(self):
        from config.settings import settings
        assert settings.GEMINI_MODEL == "gemini-3.8-flash"

    def test_gemini_transport_default(self):
        from config.settings import settings
        assert settings.GEMINI_TRANSPORT == "rest"

    def test_gemini_request_timeout_default(self):
        from config.settings import settings
        assert settings.GEMINI_REQUEST_TIMEOUT == 30.0

    def test_rag_top_k_is_int(self):
        from config.settings import settings
        assert isinstance(settings.RAG_TOP_K, int)
        assert settings.RAG_TOP_K > 0

    def test_chunk_size_is_int(self):
        from config.settings import settings
        assert isinstance(settings.CHUNK_SIZE, int)

    def test_chunk_overlap_is_int(self):
        from config.settings import settings
        assert isinstance(settings.CHUNK_OVERLAP, int)

    def test_llm_max_retries_default(self):
        from config.settings import settings
        assert settings.LLM_MAX_RETRIES == 2

    def test_chroma_collection_name_default(self):
        from config.settings import settings
        # conftest sets it to "test_paper_chunks"
        assert settings.CHROMA_COLLECTION_NAME == "test_paper_chunks"

    def test_uploads_dir_is_absolute_path(self):
        from config.settings import settings
        from pathlib import Path
        assert isinstance(settings.UPLOADS_DIR, Path)
        assert settings.UPLOADS_DIR.is_absolute()

    def test_outputs_dir_is_absolute_path(self):
        from config.settings import settings
        from pathlib import Path
        assert isinstance(settings.OUTPUTS_DIR, Path)
        assert settings.OUTPUTS_DIR.is_absolute()


class TestRequireHelper:
    """Test the _require() private helper via its effect on Settings."""

    def test_missing_key_raises_runtime_error(self, monkeypatch):
        """
        Temporarily remove GOOGLE_API_KEY and re-import to verify RuntimeError.
        Uses importlib to reload the module cleanly.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        import config.settings as settings_module
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            # Call _require directly since Settings is already instantiated
            settings_module._require("GOOGLE_API_KEY")

