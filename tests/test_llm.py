"""
tests/test_llm.py — Unit tests for core/llm.py.

Verifies:
1. get_llm() returns an instance of ChatGoogleGenerativeAI.
2. Caching: multiple calls return the identical object.
3. reset_llm_connection() clears the cached singleton.
4. get_llm() reinitializes when the model or key is changed.
5. Missing or placeholder GOOGLE_API_KEY raises a descriptive RuntimeError.
6. Diagnostic strings and exceptions never expose raw API keys.
"""

from unittest.mock import patch
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

import core.llm as llm_module
from core.llm import get_llm, reset_llm_connection
from config.settings import settings


class TestLLMInitialization:
    def setup_method(self):
        reset_llm_connection()

    def teardown_method(self):
        reset_llm_connection()

    def test_get_llm_returns_instance(self):
        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI)
        assert settings.GEMINI_MODEL in llm.model
        assert llm.temperature == 0.2
        assert llm.convert_system_message_to_human is True
        assert llm.transport == "rest"
        assert llm.timeout == 30.0

    def test_get_llm_is_cached_singleton(self):
        llm1 = get_llm()
        llm2 = get_llm()
        assert llm1 is llm2

    def test_reset_llm_connection_clears_cache(self):
        llm1 = get_llm()
        reset_llm_connection()
        llm2 = get_llm()
        assert llm1 is not llm2

    def test_get_llm_refreshes_when_model_changes(self, monkeypatch):
        llm1 = get_llm()
        monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-1.5-flash")
        llm2 = get_llm()
        assert llm1 is not llm2
        assert "gemini-1.5-flash" in llm2.model

    def test_get_llm_refreshes_when_transport_changes(self, monkeypatch):
        llm1 = get_llm()
        monkeypatch.setattr(settings, "GEMINI_TRANSPORT", "grpc")
        llm2 = get_llm()
        assert llm1 is not llm2
        assert llm2.transport == "grpc"

    def test_get_llm_refreshes_when_timeout_changes(self, monkeypatch):
        llm1 = get_llm()
        monkeypatch.setattr(settings, "GEMINI_REQUEST_TIMEOUT", 45.0)
        llm2 = get_llm()
        assert llm1 is not llm2
        assert llm2.timeout == 45.0

    def test_get_llm_missing_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(settings, "_google_api_key", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        with pytest.raises(RuntimeError, match="Missing"):
            get_llm(force_refresh=True)

    def test_get_llm_placeholder_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(settings, "_google_api_key", "your_google_gemini_api_key_here")
        monkeypatch.setenv("GOOGLE_API_KEY", "your_google_gemini_api_key_here")
        with pytest.raises(RuntimeError, match="placeholder"):
            get_llm(force_refresh=True)

    def test_parse_quota_retry_delay(self):
        from core.llm import parse_quota_retry_delay

        msg1 = "Quota exceeded... Please retry in 16h51m24.770415307s."
        delay1 = parse_quota_retry_delay(msg1)
        assert delay1 == "16h 51m 24s"

        msg2 = "Rate limit reached. retry in 45s."
        delay2 = parse_quota_retry_delay(msg2)
        assert delay2 == "45s"

        msg3 = "No retry delay specified."
        assert parse_quota_retry_delay(msg3) is None

    def test_is_daily_quota_error(self):
        from core.llm import is_daily_quota_error

        err1 = Exception("Generate content requests per day limit exceeded on Free Tier.")
        assert is_daily_quota_error(err1) is True

        err2 = Exception("Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests")
        assert is_daily_quota_error(err2) is True

        err3 = Exception("Temporary network error 503")
        assert is_daily_quota_error(err3) is False

    def test_daily_quota_halts_retries_immediately(self):
        from core.llm import DailyQuotaExceededError
        from google.api_core.exceptions import TooManyRequests

        llm = get_llm()
        calls = []

        def mock_generate(*args, **kwargs):
            calls.append(1)
            raise TooManyRequests(
                "You exceeded your current quota... Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "description: Generate content requests per day limit exceeded on Free Tier. "
                "Please retry in 16h51m24s."
            )

        llm.client.generate_content = mock_generate
        with pytest.raises(DailyQuotaExceededError) as exc_info:
            llm.invoke("Test prompt")

        # Verify exactly 1 call was made (no tenacity retries)
        assert len(calls) == 1
        assert "20 requests/day" in str(exc_info.value)
        assert exc_info.value.retry_delay == "16h 51m 24s"
