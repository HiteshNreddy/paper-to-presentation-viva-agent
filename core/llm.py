"""
core/llm.py — LangChain-compatible Gemini LLM factory.

Public API
----------
get_llm(force_refresh=False) -> ChatGoogleGenerativeAI
    Returns a cached ChatGoogleGenerativeAI instance. Automatically reinitialises
    if GOOGLE_API_KEY or GEMINI_MODEL in settings has changed.

reset_llm_connection() -> None
    Clears the cached LLM instance, forcing recreation on the next get_llm() call.

Design decisions
----------------
- langchain-google-genai reads GOOGLE_API_KEY from settings; we pass it
  explicitly to ensure the active .env configuration is respected.
- temperature=0.2 gives mostly-deterministic structured output while
  allowing minor phrasing variation.
- convert_system_message_to_human=True merges system turn into human turn
  as required by Gemini models via this adapter.
- Never prints, logs, or exposes the actual API key in diagnostic output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

logger = logging.getLogger(__name__)


class DailyQuotaExceededError(RuntimeError):
    """
    Raised when the Google Gemini API daily request quota has been exhausted.
    Carries structured information about the retry delay.
    """

    def __init__(
        self,
        message: str,
        retry_delay: str | None = None,
        raw_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_delay = retry_delay
        self.raw_error = raw_error


def parse_quota_retry_delay(error_str: str) -> str | None:
    """Extract human-readable retry delay from a Google API quota error string."""
    match = re.search(r"retry in\s+([0-9a-zA-Z\.]+)", error_str, re.IGNORECASE)
    if match:
        delay = match.group(1).rstrip(".")
        # Clean microsecond fractions e.g. 16h51m24.770415307s -> 16h 51m 24s
        delay = re.sub(r"(\.[0-9]+s)", "s", delay)
        # Add space between time units e.g. 16h51m24s -> 16h 51m 24s
        delay = re.sub(r"([0-9]+[hms])", r"\1 ", delay).strip()
        return delay
    return None


def is_daily_quota_error(exc: Exception) -> bool:
    """Return True if the exception indicates daily request quota exhaustion."""
    msg = str(exc).lower()
    daily_indicators = [
        "requests per day limit exceeded",
        "free_tier_requests",
        "generaterequestsperday",
        "requests per day",
        "daily limit exceeded",
        "daily request quota",
        "per day limit",
    ]
    return any(ind in msg for ind in daily_indicators)


class AppChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Subclass of ChatGoogleGenerativeAI ensuring:
    1. The configured timeout is explicitly passed on every generation call.
    2. Daily quota exhaustion (HTTP 429) is detected immediately to prevent
       futile retries from Tenacity and agent loops.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Guarantee finite request timeout is dispatched to Google GAPIC client
        if "timeout" not in kwargs and self.timeout:
            kwargs["timeout"] = self.timeout

        original_generate = self.client.generate_content

        def _quota_aware_generate(*args: Any, **gen_kwargs: Any) -> Any:
            try:
                return original_generate(*args, **gen_kwargs)
            except Exception as exc:
                if is_daily_quota_error(exc):
                    delay = parse_quota_retry_delay(str(exc))
                    delay_str = f" Retry available in {delay}." if delay else ""
                    logger.error(
                        "Gemini daily quota exhaustion detected. Stopping retries immediately.%s",
                        delay_str,
                    )
                    raise DailyQuotaExceededError(
                        f"Gemini API daily request quota exceeded (Free Tier limit: 20 requests/day).{delay_str}",
                        retry_delay=delay,
                        raw_error=exc,
                    ) from exc
                raise

        try:
            self.client.generate_content = _quota_aware_generate
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        finally:
            self.client.generate_content = original_generate


_llm_instance: Optional[ChatGoogleGenerativeAI] = None
_cached_key: Optional[str] = None
_cached_model: Optional[str] = None
_cached_transport: Optional[str] = None
_cached_timeout: Optional[float] = None


def reset_llm_connection() -> None:
    """Clear the cached LLM instance, forcing recreation with fresh config."""
    global _llm_instance, _cached_key, _cached_model, _cached_transport, _cached_timeout
    _llm_instance = None
    _cached_key = None
    _cached_model = None
    _cached_transport = None
    _cached_timeout = None
    logger.info("LLM connection cache reset.")


def get_llm(force_refresh: bool = False) -> ChatGoogleGenerativeAI:
    """
    Return the shared ChatGoogleGenerativeAI instance.

    Automatically refreshes if the configured API key, model, transport,
    or timeout in settings has changed, or if force_refresh is True.

    Returns
    -------
    ChatGoogleGenerativeAI
        A LangChain chat model backed by the Gemini API configured with
        REST transport and finite timeout.

    Raises
    ------
    RuntimeError : if GOOGLE_API_KEY is missing or default placeholder.
    """
    global _llm_instance, _cached_key, _cached_model, _cached_transport, _cached_timeout

    # Check key and model from settings
    current_key = settings.GOOGLE_API_KEY
    if not current_key or current_key == "your_google_gemini_api_key_here":
        raise RuntimeError(
            "Missing or placeholder GOOGLE_API_KEY in .env. "
            "Please configure your Google Gemini API key in the .env file."
        )

    current_model = settings.GEMINI_MODEL
    current_transport = settings.GEMINI_TRANSPORT
    current_timeout = settings.GEMINI_REQUEST_TIMEOUT

    # Recreate client if model, key, transport, or timeout changed or force_refresh requested
    if (
        force_refresh
        or _llm_instance is None
        or _cached_key != current_key
        or _cached_model != current_model
        or _cached_transport != current_transport
        or _cached_timeout != current_timeout
    ):
        _llm_instance = AppChatGoogleGenerativeAI(
            model=current_model,
            google_api_key=current_key,
            temperature=0.2,
            convert_system_message_to_human=True,
            transport=current_transport,
            timeout=current_timeout,
        )
        _cached_key = current_key
        _cached_model = current_model
        _cached_transport = current_transport
        _cached_timeout = current_timeout
        # Note: Never log or print the raw API key!
        logger.info(
            "Gemini LLM initialized: model=%s, transport=%s, timeout=%.1fs (key length: %d)",
            current_model,
            current_transport,
            current_timeout,
            len(current_key),
        )

    return _llm_instance
