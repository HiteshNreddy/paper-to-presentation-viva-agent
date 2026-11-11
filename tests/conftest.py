"""
tests/conftest.py — Shared pytest fixtures and test environment setup.

Sets GOOGLE_API_KEY=test-key before any module is imported so settings.py
does not raise RuntimeError during test collection.

Also provides lightweight test shims for external framework packages
(langchain, langchain_core, langchain_chroma, fitz) if they are not yet
installed in the active Python environment. This allows testing all core
logic, schemas, RAG decisions, isolation rules, and document hashing
without requiring network downloads during development.
"""

import importlib.util
import os
import sys
import types
import pytest

# ── Set required env vars before any application import ───────────────────────
os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.8-flash")
os.environ.setdefault("GEMINI_TRANSPORT", "rest")
os.environ.setdefault("GEMINI_REQUEST_TIMEOUT", "30.0")
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma_test")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_paper_chunks")
os.environ.setdefault("HF_HUB_OFFLINE", "1")



# ── Lightweight Mock Shims for Uninstalled External Packages ──────────────────
# These only activate if the real package is not installed.
def _install_test_shims():
    # 1. langchain_core
    if importlib.util.find_spec("langchain_core") is None:
        lc_core = types.ModuleType("langchain_core")

        # langchain_core.documents
        docs_mod = types.ModuleType("langchain_core.documents")
        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata if metadata is not None else {}
            def __repr__(self):
                return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"
        docs_mod.Document = Document

        # langchain_core.vectorstores
        vs_mod = types.ModuleType("langchain_core.vectorstores")
        class VectorStoreRetriever:
            def __init__(self, *args, **kwargs):
                pass
            def invoke(self, query):
                return []
        vs_mod.VectorStoreRetriever = VectorStoreRetriever

        # langchain_core.messages
        msg_mod = types.ModuleType("langchain_core.messages")
        class AIMessage:
            def __init__(self, content=""):
                self.content = content
        msg_mod.AIMessage = AIMessage

        # langchain_core.output_parsers
        op_mod = types.ModuleType("langchain_core.output_parsers")
        class PydanticOutputParser:
            def __init__(self, pydantic_object=None):
                self.pydantic_object = pydantic_object
            def get_format_instructions(self):
                return "JSON Schema Instructions"
        op_mod.PydanticOutputParser = PydanticOutputParser

        # langchain_core.prompts
        prompt_mod = types.ModuleType("langchain_core.prompts")
        class ChatPromptTemplate:
            def __init__(self, messages=None):
                self.messages = messages or []
            @classmethod
            def from_messages(cls, messages):
                return cls(messages)
            def partial(self, **kwargs):
                return self
            def __or__(self, other):
                from unittest.mock import MagicMock
                return MagicMock()
        prompt_mod.ChatPromptTemplate = ChatPromptTemplate

        lc_core.documents = docs_mod
        lc_core.vectorstores = vs_mod
        lc_core.messages = msg_mod
        lc_core.output_parsers = op_mod
        lc_core.prompts = prompt_mod

        sys.modules["langchain_core"] = lc_core
        sys.modules["langchain_core.documents"] = docs_mod
        sys.modules["langchain_core.vectorstores"] = vs_mod
        sys.modules["langchain_core.messages"] = msg_mod
        sys.modules["langchain_core.output_parsers"] = op_mod
        sys.modules["langchain_core.prompts"] = prompt_mod

    # 2. langchain_chroma
    if importlib.util.find_spec("langchain_chroma") is None:
        lc_chroma = types.ModuleType("langchain_chroma")
        class Chroma:
            def __init__(self, *args, **kwargs):
                pass
            def as_retriever(self, **kwargs):
                return sys.modules["langchain_core.vectorstores"].VectorStoreRetriever()
            def add_documents(self, docs):
                pass
            def delete(self, where=None):
                pass
            def get(self, where=None, limit=None):
                return {"ids": []}
        lc_chroma.Chroma = Chroma
        sys.modules["langchain_chroma"] = lc_chroma

    # 3. langchain_huggingface
    if importlib.util.find_spec("langchain_huggingface") is None:
        lc_hf = types.ModuleType("langchain_huggingface")
        class HuggingFaceEmbeddings:
            def __init__(self, *args, **kwargs):
                pass
            def embed_documents(self, texts):
                return [[0.0] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.0] * 384
        lc_hf.HuggingFaceEmbeddings = HuggingFaceEmbeddings
        sys.modules["langchain_huggingface"] = lc_hf

    # 4. langchain_google_genai
    if importlib.util.find_spec("langchain_google_genai") is None:
        lc_genai = types.ModuleType("langchain_google_genai")
        class ChatGoogleGenerativeAI:
            def __init__(self, *args, **kwargs):
                pass
        lc_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
        sys.modules["langchain_google_genai"] = lc_genai

    # 5. langchain (including text_splitter)
    if importlib.util.find_spec("langchain") is None:
        lc_mod = types.ModuleType("langchain")
        ts_mod = types.ModuleType("langchain.text_splitter")
        DocumentClass = sys.modules["langchain_core.documents"].Document

        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
                self.add_start_index = add_start_index

            def create_documents(self, texts, metadatas=None):
                docs = []
                for i, text in enumerate(texts):
                    meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
                    start = 0
                    while start < len(text):
                        end = min(start + self.chunk_size, len(text))
                        chunk_text = text[start:end]
                        chunk_meta = dict(meta)
                        if self.add_start_index:
                            chunk_meta["start_index"] = start
                        docs.append(DocumentClass(page_content=chunk_text, metadata=chunk_meta))
                        if end >= len(text):
                            break
                        start += max(1, self.chunk_size - self.chunk_overlap)
                return docs

        ts_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        lc_mod.text_splitter = ts_mod
        sys.modules["langchain"] = lc_mod
        sys.modules["langchain.text_splitter"] = ts_mod

    # 6. fitz (PyMuPDF)
    if importlib.util.find_spec("fitz") is None:
        fitz_mod = types.ModuleType("fitz")
        class _MockFitzDoc:
            def __init__(self, pages=None):
                self._pages = pages or []
            @property
            def page_count(self):
                return len(self._pages)
            def __getitem__(self, idx):
                return self._pages[idx]
            def close(self):
                pass

        class _MockFitzPage:
            def __init__(self, text=""):
                self._text = text
            def get_text(self, fmt="text"):
                return self._text

        def _open(path_or_stream=None, stream=None, filetype=None):
            if stream is not None:
                if not stream.startswith(b"%PDF"):
                    raise ValueError("Invalid PDF stream header")
                if b"Hello PDF" in stream:
                    return _MockFitzDoc([_MockFitzPage("Hello PDF")])
                return _MockFitzDoc([_MockFitzPage("Sample extracted text from uploaded PDF")])
            if path_or_stream is not None:
                from pathlib import Path
                p = Path(path_or_stream)
                if not p.exists():
                    raise FileNotFoundError(f"File not found: {p}")
                return _MockFitzDoc([_MockFitzPage("Sample extracted text from path")])
            return _MockFitzDoc([])

        fitz_mod.open = _open
        sys.modules["fitz"] = fitz_mod


_install_test_shims()


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Return raw bytes of a minimal valid PDF containing 'Hello PDF'."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n441\n%%EOF"
    )


@pytest.fixture
def sample_pages() -> list[dict]:
    """Return a minimal list of page dicts (output of pdf_loader.load_pdf)."""
    return [
        {
            "page_number": 1,
            "text": (
                "Abstract: This paper proposes a novel approach to solving "
                "the travelling salesman problem using deep reinforcement learning. "
                "We demonstrate state-of-the-art results on standard benchmarks."
            ),
            "source": "sample_paper.pdf",
        },
        {
            "page_number": 2,
            "text": (
                "Methodology: We employ a pointer-network architecture trained with "
                "REINFORCE. The model is evaluated on TSP-50 and TSP-100 datasets. "
                "Our approach achieves 95% optimality gap on TSP-50."
            ),
            "source": "sample_paper.pdf",
        },
    ]
