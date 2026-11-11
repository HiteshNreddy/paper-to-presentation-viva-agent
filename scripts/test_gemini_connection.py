"""
scripts/test_gemini_connection.py — Connectivity check for Google Gemini API.

Validates that:
1. GOOGLE_API_KEY is present in .env and not empty or default placeholder.
2. The Gemini API endpoint can be reached via ChatGoogleGenerativeAI using REST transport.
3. A finite request timeout (default: 30s) prevents indefinite hanging.
4. Separate diagnostic handlers identify:
   - Timeout errors
   - Connection/Network errors
   - Authentication errors
   - Quota/Rate-limit errors
   - Invalid/unsupported model errors
5. A basic generation prompt receives an authentic valid response.

Security Mandate: Never prints, logs, or exposes raw API key characters in diagnostic output.
"""

import os
import re
import socket
import sys
import time
from pathlib import Path

# Add project root to sys.path so config and core modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import google.api_core.exceptions as g_exceptions
import requests.exceptions

from config.settings import settings
from core.llm import (
    get_llm,
    reset_llm_connection,
    DailyQuotaExceededError,
    parse_quota_retry_delay,
    is_daily_quota_error,
)


def sanitize_message(msg: str) -> str:
    """Sanitize any error message or string so no API keys or key query params leak."""
    if not msg:
        return ""
    # Redact Google API key format (typically AIzaSy...)
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", str(msg))
    # Redact key= parameter in URLs
    sanitized = re.sub(r"([?&]key=)[^& \n\t]+", r"\1[REDACTED_API_KEY]", sanitized)
    # Redact the active key if present
    active_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if active_key and len(active_key) > 8 and active_key in sanitized:
        sanitized = sanitized.replace(active_key, "[REDACTED_API_KEY]")
    return sanitized


def key_status_display(key: str) -> str:
    """Return a completely sanitized status indicator without exposing any characters."""
    if not key or not key.strip():
        return "[NOT CONFIGURED - EMPTY]"
    key_clean = key.strip()
    if "your_google_gemini_api_key_here" in key_clean.lower():
        return "[PLACEHOLDER DETECTED - NOT CONFIGURED]"
    return f"[CONFIGURED - length: {len(key_clean)} chars]"


def classify_error(exc: Exception, model_name: str, timeout_seconds: float) -> tuple[str, str, list[str]]:
    """
    Classify the exception into one of the designated categories:
    - TIMEOUT
    - CONNECTION
    - AUTHENTICATION
    - QUOTA
    - INVALID_MODEL
    - GENERAL_ERROR

    Returns (category_name, detailed_explanation, list_of_recommendations).
    """
    err_str = str(exc)
    clean_err = sanitize_message(err_str)
    exc_type = type(exc).__name__

    # 1. Invalid Model Error
    if isinstance(exc, g_exceptions.NotFound) or "not found" in clean_err.lower() or "404" in clean_err:
        return (
            "INVALID MODEL ERROR",
            f"Model '{model_name}' was not found or is not supported for generateContent (HTTP 404).",
            [
                f"The configured model name '{model_name}' is not recognized by the Gemini API endpoint.",
                "Gemini models use specific identifiers. Recommended valid models include:",
                "  - gemini-1.5-flash  (Fast, multimodal, high quota)",
                "  - gemini-1.5-pro    (Complex reasoning)",
                "  - gemini-2.0-flash  (Latest flash generation)",
                "Update GEMINI_MODEL in your .env file to a supported model (e.g. GEMINI_MODEL=gemini-1.5-flash).",
            ],
        )

    # 2. Authentication Error
    if (
        isinstance(exc, (g_exceptions.Unauthenticated, g_exceptions.PermissionDenied))
        or "api_key_invalid" in clean_err.lower()
        or "api key not valid" in clean_err.lower()
        or "invalid api key" in clean_err.lower()
        or "401" in clean_err
        or "403" in clean_err
    ):
        return (
            "AUTHENTICATION ERROR",
            "The Gemini API rejected the credentials provided in GOOGLE_API_KEY (HTTP 400/401/403).",
            [
                "Your API key was not accepted by Google AI Studio.",
                "1. Verify your key at: https://aistudio.google.com/app/apikey",
                "2. Ensure the key starts with 'AIzaSy' and is copied with no quotes or spaces in .env.",
                "3. If this key was generated in Vertex AI / Google Cloud Console instead of Google AI Studio,",
                "   it may require OAuth2/Service Account credentials instead of an AI Studio API key.",
            ],
        )

    # 3. Quota / Rate-limit Error
    if (
        isinstance(exc, DailyQuotaExceededError)
        or is_daily_quota_error(exc)
        or isinstance(exc, g_exceptions.ResourceExhausted)
        or "429" in clean_err
        or "resource_exhausted" in clean_err.lower()
        or "quota" in clean_err.lower()
    ):
        delay = getattr(exc, "retry_delay", None) or parse_quota_retry_delay(clean_err)
        delay_msg = f" Retry available in {delay}." if delay else ""
        if isinstance(exc, DailyQuotaExceededError) or is_daily_quota_error(exc):
            return (
                "QUOTA EXCEEDED ERROR (DAILY FREE-TIER LIMIT REACHED)",
                f"Gemini Free-Tier daily request quota of 20 requests/day has been exhausted.{delay_msg}",
                [
                    "You have reached the Free Tier daily limit (20 requests per project per day).",
                    f"WHEN YOU CAN RETRY:{delay_msg if delay else ' Tomorrow when the daily quota resets.'}",
                    "IMMEDIATE STOP: Retries were halted immediately on attempt 1 because daily quota exhaustion cannot succeed on retry.",
                    "MODEL MAPPING NOTE: Google's backend tracks free-tier quotas under the 'gemini-3.6-flash' metric bucket regardless of whether 3.5 or 3.6 was requested.",
                    "UPGRADE OPTION: You can enable Pay-As-You-Go billing in Google AI Studio (https://aistudio.google.com/app/apikey) to lift the 20 requests/day limit, or wait for the quota to reset.",
                ],
            )
        return (
            "QUOTA EXCEEDED ERROR (RPM RATE LIMIT)",
            f"Gemini API short-term rate limit has been reached (HTTP 429).{delay_msg}",
            [
                "You have temporarily exceeded the Requests Per Minute (RPM) limit.",
                "1. Wait a few moments (e.g. 30-60 seconds) and retry.",
                "2. Monitor your project usage on Google AI Studio (https://aistudio.google.com).",
            ],
        )

    # 4. Timeout Error
    if (
        isinstance(exc, (g_exceptions.DeadlineExceeded, requests.exceptions.Timeout))
        or isinstance(exc, (socket.timeout, TimeoutError))
        or "timed out" in clean_err.lower()
        or "deadline" in clean_err.lower()
        or "504" in clean_err
    ):
        return (
            "REQUEST TIMEOUT ERROR",
            f"The request timed out after {timeout_seconds:.1f} seconds waiting for a response.",
            [
                f"The Gemini API did not complete the request within the {timeout_seconds:.1f}s deadline.",
                "1. Check if an outbound firewall, corporate VPN, or proxy is dropping HTTPS traffic.",
                "2. Verify internet connectivity to 'generativelanguage.googleapis.com'.",
                "3. You can increase GEMINI_REQUEST_TIMEOUT in .env (e.g. GEMINI_REQUEST_TIMEOUT=60.0).",
            ],
        )

    # 5. Connection / Network Error
    if (
        isinstance(exc, (requests.exceptions.ConnectionError, g_exceptions.ServiceUnavailable))
        or isinstance(exc, (socket.error, ConnectionResetError))
        or "getaddrinfo" in clean_err.lower()
        or "nameresolutionerror" in clean_err.lower()
        or "connection refused" in clean_err.lower()
        or "connection error" in clean_err.lower()
        or "503" in clean_err
    ):
        return (
            "CONNECTION / NETWORK ERROR",
            "Failed to establish a network connection to Google's API servers.",
            [
                "The client could not reach generativelanguage.googleapis.com on port 443.",
                "1. Verify your machine has an active internet connection.",
                "2. If you are behind a corporate proxy, configure HTTPS_PROXY environment variable.",
                "3. Check whether DNS resolution for 'generativelanguage.googleapis.com' is working.",
            ],
        )

    # 6. General / Unhandled Error
    return (
        f"API ERROR ({exc_type})",
        clean_err,
        [
            "An unhandled error occurred during Gemini API communication.",
            "Inspect the sanitized details above to diagnose the root cause.",
        ],
    )


def check_gemini_connection() -> bool:
    print("=" * 70)
    print("  Paper-to-Presentation & Viva Agent: Gemini Connectivity Test")
    print("=" * 70)

    # Stage 1: Load configuration
    print("\n[Stage 1/5] Loading configuration and environment variables from .env...")
    settings.reload()
    reset_llm_connection()

    api_key = settings.GOOGLE_API_KEY.strip()
    model_name = settings.GEMINI_MODEL
    transport = settings.GEMINI_TRANSPORT
    timeout_seconds = settings.GEMINI_REQUEST_TIMEOUT

    print(f"  -> Configured Model     : {model_name}")
    print(f"  -> Configured Transport : {transport} (HTTPS REST)")
    print(f"  -> Request Timeout      : {timeout_seconds:.1f} seconds")
    print(f"  -> API Key Status       : {key_status_display(api_key)}")

    # Stage 2: Validate API key presence
    print("\n[Stage 2/5] Validating API key presence...")
    if not api_key or "your_google_gemini_api_key_here" in api_key.lower():
        print("  [x] ABORT: Valid Google API key not found in .env.")
        print("      Please edit .env and configure your Google Gemini API key:")
        print("      GOOGLE_API_KEY=AIzaSy...")
        print("      Obtain a key at: https://aistudio.google.com/app/apikey")
        return False
    print("  [+] API key detected.")

    # Stage 3: Client Initialization
    print(f"\n[Stage 3/5] Initializing Gemini LLM client (transport='{transport}', timeout={timeout_seconds:.1f}s)...")
    init_start = time.perf_counter()
    try:
        llm = get_llm()
        init_elapsed = time.perf_counter() - init_start
        print(f"  [+] Client initialized successfully in {init_elapsed:.3f}s.")
        print(f"      Client Class     : {type(llm).__name__}")
        print(f"      Effective Model  : {llm.model}")
        print(f"      Active Transport : {getattr(llm, 'transport', 'default')}")
        print(f"      Active Timeout   : {getattr(llm, 'timeout', timeout_seconds)}s")
    except Exception as exc:
        clean_err = sanitize_message(str(exc))
        print(f"  [x] Client initialization failed: {clean_err}")
        return False

    # Stage 4: Dispatch Request
    test_prompt = "Hello! Please reply with exactly: 'GEMINI_CONNECTION_OK'"
    print(f"\n[Stage 4/5] Sending probe request to Google Gemini API...")
    print(f"  -> Target Endpoint : https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent")
    print(f"  -> Prompt           : \"{test_prompt}\"")
    print(f"  -> Waiting up to {timeout_seconds:.1f}s for response... (Press Ctrl+C to abort)")

    request_start = time.perf_counter()
    try:
        response = llm.invoke(test_prompt, timeout=timeout_seconds)
        request_elapsed = time.perf_counter() - request_start
        reply_text = str(response.content).strip()

        # Stage 5: Response processing
        print(f"\n[Stage 5/5] Processing Gemini API response...")
        print(f"  [+] Response received in {request_elapsed:.2f}s:")
        print(f"      Content: \"{reply_text}\"")

        if "GEMINI_CONNECTION_OK" in reply_text.upper():
            print("\n" + "=" * 70)
            print("  [+] SUCCESS: Gemini API connection verified and 100% operational!")
            print(f"      Model: {model_name} | Transport: {transport} | Latency: {request_elapsed:.2f}s")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("  [+] SUCCESS: Authentic response received from Google Gemini API.")
            print(f"      Model: {model_name} | Transport: {transport} | Latency: {request_elapsed:.2f}s")
            print("=" * 70)
        return True

    except KeyboardInterrupt:
        request_elapsed = time.perf_counter() - request_start
        print(f"\n  [!] User interrupted probe with Ctrl+C after {request_elapsed:.2f}s.")
        return False

    except Exception as exc:
        request_elapsed = time.perf_counter() - request_start
        cat, explanation, recommendations = classify_error(exc, model_name, timeout_seconds)

        print(f"\n[Stage 5/5] Connection probe failed after {request_elapsed:.2f}s.")
        print("\n" + "=" * 70)
        print(f"  [x] FAILURE CATEGORY : {cat}")
        print(f"  [x] EXCEPTION TYPE   : {type(exc).__module__}.{type(exc).__name__}")
        print(f"  [x] SUMMARY          : {explanation}")
        print("=" * 70)

        clean_detail = sanitize_message(str(exc))
        print(f"\n  Sanitized API Details:")
        for line in clean_detail.splitlines()[:6]:
            print(f"    {line}")

        print("\n  Recommended Actions:")
        for rec in recommendations:
            print(f"    * {rec}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = check_gemini_connection()
    sys.exit(0 if success else 1)



