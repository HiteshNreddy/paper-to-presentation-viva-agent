"""
scripts/verify_failure_handling.py — Verifies missing API key & API failure handling.

Security Rule: Never logs or prints raw API keys.
"""

import os
import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings


def sanitize(msg: str) -> str:
    """Scrub potential API keys or secrets from messages."""
    return re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_KEY]", str(msg))


def test_missing_api_key_handling():
    print("\n--- Test 1: Missing API Key Handling ---")
    import config.settings as settings_module
    try:
        # Call _require with an empty env variable
        os.environ["MISSING_TEST_KEY"] = ""
        settings_module._require("MISSING_TEST_KEY")
        print("[FAIL] Missing key did not raise RuntimeError")
        return False
    except RuntimeError as exc:
        print(f"[PASS] Successfully caught missing key error: {sanitize(exc)}")
        return True


def test_invalid_api_key_handling():
    print("\n--- Test 2: Invalid API Key Failure Handling ---")
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        # Intentionally provide a dummy invalid key
        bad_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key="AIzaSy_INVALID_TEST_KEY_FOR_FAILURE_CHECK_0000",
            temperature=0.0,
        )
        print(f"[*] Sending probe with invalid key to {settings.GEMINI_MODEL}...")
        bad_llm.invoke("ping")
        print("[FAIL] Invalid key unexpectedly succeeded")
        return False
    except Exception as exc:
        error_type = type(exc).__name__
        print(f"[PASS] Successfully caught invalid API key failure ({error_type}):")
        clean_err = sanitize(exc).replace("AIzaSy_INVALID_TEST_KEY_FOR_FAILURE_CHECK_0000", "[MASKED_KEY]")
        print(f"       Sanitized Detail: {clean_err[:120]}...")
        return True


if __name__ == "__main__":
    t1 = test_missing_api_key_handling()
    t2 = test_invalid_api_key_handling()
    if t1 and t2:
        print("\n[+] All failure handling checks passed!")
        sys.exit(0)
    else:
        sys.exit(1)


