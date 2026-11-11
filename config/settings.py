"""
config/settings.py — Centralised application configuration.

Loads all environment variables from the .env file and exposes a single
`settings` instance imported by the rest of the application.

Design decisions
----------------
- Loads .env using python-dotenv with override=True so local .env values
  strictly take precedence over stale/system environment variables.
- Dynamic properties allow re-reading GOOGLE_API_KEY and GEMINI_MODEL if .env
  is updated without requiring a complete process restart.
- Default GEMINI_MODEL is set to 'gemini-3.8-flash'.
- Never prints, logs, or exposes raw API keys in logs or exceptions.
- All directory paths are resolved to absolute Path objects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Locate project root and load .env ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _is_testing() -> bool:
    """Return True if running in a pytest or automated test context."""
    return "pytest" in sys.modules or bool(os.getenv("PYTEST_CURRENT_TEST"))


# Load .env with override=True so local .env values take precedence over system/stale vars,
# except in automated test sessions where test fixtures may configure test values.
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=not _is_testing())


def _require(key: str) -> str:
    """Return the value of an env variable or raise a helpful RuntimeError."""
    value = os.getenv(key, "").strip()
    if not value or value == "your_google_gemini_api_key_here":
        raise RuntimeError(
            f"Missing required environment variable: '{key}'.\n"
            f"Please configure your Google Gemini API key in { _ENV_FILE.name }."
        )
    return value


class Settings:
    """
    Application configuration object populated from environment variables.
    """

    def __init__(self) -> None:
        self.reload()

    def reload(self, override: bool | None = None) -> None:
        """Reload configuration from .env file, refreshing all properties."""
        if override is None:
            override = not _is_testing()
        if _ENV_FILE.exists():
            load_dotenv(_ENV_FILE, override=override)

        self._google_api_key: str = os.getenv("GOOGLE_API_KEY", "").strip()
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        self.GEMINI_TRANSPORT: str = os.getenv("GEMINI_TRANSPORT", "rest").strip()
        self.GEMINI_REQUEST_TIMEOUT: float = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "30.0"))
        self.LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        self.CHROMA_PERSIST_DIR: str = os.getenv(
            "CHROMA_PERSIST_DIR",
            str(_PROJECT_ROOT / "chroma_store"),
        )
        self.CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "paper_chunks").strip()
        self.RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
        self.RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.30"))
        self.RAG_MAX_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "1.85"))
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.UPLOADS_DIR: Path = _PROJECT_ROOT / "uploads"
        self.OUTPUTS_DIR: Path = _PROJECT_ROOT / "outputs"

    @property
    def GOOGLE_API_KEY(self) -> str:
        """
        Dynamically retrieve GOOGLE_API_KEY.
        If empty in memory, attempts fresh load from .env.
        """
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key or key == "your_google_gemini_api_key_here":
            if _ENV_FILE.exists() and not _is_testing():
                load_dotenv(_ENV_FILE, override=True)
                key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "Missing required environment variable: 'GOOGLE_API_KEY'.\n"
                f"Copy .env.example -> .env and add your valid Gemini API key."
            )
        return key

    @property
    def is_api_key_configured(self) -> bool:
        """Return True if a non-empty, non-placeholder API key is present."""
        try:
            key = self.GOOGLE_API_KEY
            return bool(
                key
                and key != "your_google_gemini_api_key_here"
                and not key.startswith("test-key")
            )
        except RuntimeError:
            return False


# Single shared instance — import `settings` everywhere.
settings = Settings()
