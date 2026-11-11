"""
ui/sidebar.py — Apple-inspired Sidebar Component for Paper-to-Presentation & Viva Agent.

Features:
- iOS 27 glassmorphic layout with frosted glass styling.
- Real-time Light / Dark Mode toggle switch.
- Robust PDF upload validation and error messaging.
- SHA-256 document identification and session-scoped ChromaDB indexing.
- Active document card with metadata pills (pages, chunks, size, doc_id hash).
- System runtime badges (Embeddings, ChromaDB isolation, Gemini LLM).
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from config.settings import settings
from core.pdf_loader import validate_pdf_source
from core.pdf_pipeline import process_pdf
from core.vector_store import ingest_documents, is_document_ingested
from utils.doc_id import make_doc_id
from utils.file_utils import validate_pdf_bytes

logger = logging.getLogger(__name__)


def reset_document_session() -> None:
    """Clear active document state and all cached agent outputs in session state."""
    keys_to_clear = [
        "doc_id",
        "doc_info",
        "summary_output",
        "concept_output",
        "slide_deck",
        "pptx_bytes",
        "viva_output",
        "chat_history",
        "current_query_intent",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def render_sidebar() -> dict[str, Any]:
    """
    Render the Apple-styled sidebar with theme controls, PDF upload, and document management.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'active_doc_id' and 'doc_info'.
    """
    with st.sidebar:
        # ── Header & Workspace Branding ────────────────────────────────────────
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem;">
                <div style="background: linear-gradient(135deg, #0A84FF, #0071E3); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.25rem; font-weight: bold; box-shadow: 0 4px 12px rgba(10, 132, 255, 0.4);">
                    ✦
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 1.15rem; letter-spacing: -0.02em; line-height: 1.2;">
                        Paper Duo
                    </div>
                    <div style="font-size: 0.75rem; color: gray; letter-spacing: 0.04em; text-transform: uppercase;">
                        Research OS 27
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme Toggle ───────────────────────────────────────────────────────
        current_theme = st.session_state.get("theme", "dark")
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.caption(f"Appearance: **{current_theme.capitalize()} Mode**")
        with col_t2:
            toggle_label = "☀️ Light" if current_theme == "dark" else "🌙 Dark"
            if st.button(toggle_label, key="theme_toggle_btn", use_container_width=True):
                st.session_state["theme"] = "light" if current_theme == "dark" else "dark"
                st.rerun()

        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        # ── Step 1: Upload Paper ───────────────────────────────────────────────
        st.markdown("### 📄 Research Paper")
        uploaded_file = st.file_uploader(
            "Drop PDF here or browse",
            type=["pdf"],
            help="Accepts academic papers in standard PDF format (up to 50 MB).",
            key="pdf_file_uploader",
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                if not raw_bytes:
                    st.error("Uploaded file is empty (0 bytes).")
                    return {"active_doc_id": None, "doc_info": None}

                # Validate PDF magic bytes and integrity
                validate_pdf_bytes(raw_bytes)
                validate_pdf_source(raw_bytes)

                # Compute full-file SHA-256 doc_id
                computed_doc_id = make_doc_id(raw_bytes)

                # Check if this is a newly uploaded paper or existing session paper
                active_doc_id = st.session_state.get("doc_id")

                if active_doc_id != computed_doc_id:
                    with st.spinner("Indexing paper into ChromaDB with 384d embeddings..."):
                        # Clear old document's cached outputs
                        reset_document_session()

                        # Run PDF processing pipeline (validation, extraction, chunking)
                        proc_result, chunks = process_pdf(raw_bytes, filename=uploaded_file.name)

                        # Ingest chunks into ChromaDB if not already present
                        if not is_document_ingested(computed_doc_id):
                            ingest_documents(chunks, doc_id=computed_doc_id)

                        # Store in session state
                        st.session_state["doc_id"] = computed_doc_id
                        st.session_state["doc_info"] = {
                            "filename": uploaded_file.name,
                            "page_count": proc_result.page_count,
                            "chunk_count": len(chunks),
                            "file_size_kb": round(len(raw_bytes) / 1024, 1),
                        }
                        st.success(f"Indexed **{uploaded_file.name}**")
                        st.rerun()

            except ValueError as e:
                st.error(f"Validation error: {e}")
            except TypeError as e:
                st.error(f"File error: {e}")
            except Exception as e:
                logger.exception("Unexpected error processing uploaded PDF: %s", e)
                st.error(f"Processing error: {e}")

        # ── Step 2: Active Document Card ───────────────────────────────────────
        doc_id = st.session_state.get("doc_id")
        doc_info = st.session_state.get("doc_info", {})

        if doc_id and doc_info:
            filename = doc_info.get("filename", "Paper.pdf")
            page_count = doc_info.get("page_count", 0)
            chunk_count = doc_info.get("chunk_count", 0)
            file_size_kb = doc_info.get("file_size_kb", 0.0)

            st.markdown(
                f"""
                <div class="apple-glass-card-sm" style="margin-top: 1rem;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                        ● Active Document
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.6rem; word-break: break-word;">
                        {filename}
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.6rem;">
                        <span class="apple-pill apple-pill-blue">📄 {page_count} Pages</span>
                        <span class="apple-pill">🧩 {chunk_count} Chunks</span>
                        <span class="apple-pill">💾 {file_size_kb} KB</span>
                    </div>
                    <div style="font-size: 0.72rem; color: gray; font-family: monospace;">
                        ID: {doc_id[:16]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔄 Change Document", key="btn_reset_paper", use_container_width=True):
                reset_document_session()
                st.rerun()
        else:
            st.markdown(
                """
                <div class="apple-glass-card-sm" style="margin-top: 1rem; border-style: dashed;">
                    <div style="font-size: 0.85rem; color: gray; text-align: center;">
                        No paper currently loaded.<br>Upload a PDF above to activate all 4 AI agents.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Step 3: Runtime Spec Footer ────────────────────────────────────────
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="padding: 0.8rem; border-radius: 12px; background: rgba(128, 128, 128, 0.08); font-size: 0.75rem; color: gray; line-height: 1.5;">
                <div style="font-weight: 600; margin-bottom: 0.2rem; color: var(--apple-text);">System Architecture</div>
                • Model: <code>{settings.GEMINI_MODEL}</code><br>
                • Embeddings: <code>all-MiniLM-L6-v2</code><br>
                • Storage: <code>ChromaDB (Isolated)</code><br>
                • Transport: <code>REST (Timeout: 30s)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "active_doc_id": st.session_state.get("doc_id"),
            "doc_info": st.session_state.get("doc_info"),
        }
