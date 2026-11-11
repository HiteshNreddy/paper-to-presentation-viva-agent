"""
ui/result_views.py — Presentation components for agent outputs and interactive Q&A.

Responsibilities:
- Provide formatted, responsive Streamlit views for each specialized agent:
    render_summary(summary: SummaryOutput)
    render_concepts(concepts: ConceptOutput)
    render_slides(slide_deck: SlideDeck, pptx_bytes: bytes | None)
    render_viva(viva: VivaOutput)
    render_rag_chat(doc_id: str)
- Accept validated Pydantic models (SummaryOutput, ConceptOutput, SlideDeck, VivaOutput).
- Render Apple iPhone Duo / iOS 27 glass cards, badges, expanders, and download controls.
- Display source page attributions without fabricating citations.
- Does NOT contain agent logic or direct database access.
"""

from __future__ import annotations

import logging
from typing import Sequence

import streamlit as st

from core.rag_pipeline import RAGPipeline
from schemas.concept_schema import ConceptOutput
from schemas.slide_schema import SlideDeck
from schemas.summary_schema import RetrievedSource, SummaryOutput
from schemas.viva_schema import VivaOutput
from utils.pptx_builder import build_presentation_bytes, _sanitize_filename

logger = logging.getLogger(__name__)


def _render_sources_expander(
    sources: Sequence[RetrievedSource],
    title: str = "Grounding Sources & Traceability",
) -> None:
    """Render an expandable list of retrieved chunk sources with page citations."""
    if not sources:
        return

    with st.expander(f"📚 {title} ({len(sources)} verified chunks)", expanded=False):
        for s in sources:
            st.markdown(
                f"- **Chunk {s.chunk_index}** · **Page {s.page_number}** "
                f"*(Source: `{s.source}`)*"
            )


# ── View 1: Summarizer Output ──────────────────────────────────────────────────

def render_summary(summary: SummaryOutput) -> None:
    """Render structured research paper summary with Apple glass panels."""
    st.markdown(
        f"""
        <div class="apple-glass-card" style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                ✦ Agent 1 · Research Summary
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25; margin-bottom: 0.6rem;">
                {summary.title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Abstract summary callout (preserving st.info for test compatibility)
    st.markdown("### 📝 Abstract Summary")
    st.info(summary.abstract_summary)

    # Core Problem & Proposed Solution side-by-side cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Problem Statement")
        st.warning(summary.problem_statement)

    with col2:
        st.markdown("### 💡 Proposed Solution")
        st.success(summary.proposed_solution)

    # Key Findings & Contributions
    st.markdown("### 🔍 Key Findings & Empirical Results")
    findings_html = ""
    for i, finding in enumerate(summary.key_findings, start=1):
        findings_html += f"""
        <div class="apple-glass-card-sm" style="display: flex; align-items: flex-start; gap: 0.8rem;">
            <div style="background: var(--apple-accent); color: white; border-radius: 8px; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;">
                {i}
            </div>
            <div style="font-size: 0.96rem; line-height: 1.5; color: var(--apple-text);">
                {finding}
            </div>
        </div>
        """
    st.markdown(findings_html, unsafe_allow_html=True)

    # Conclusion
    st.markdown("### 🏁 Conclusion & Impact")
    st.markdown(
        f"""
        <div class="apple-glass-card" style="border-left: 4px solid var(--apple-accent);">
            <div style="font-size: 1.02rem; line-height: 1.6; color: var(--apple-text);">
                {summary.conclusion}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_sources_expander(summary.sources, "Retrieved Sources Grounding This Summary")


# ── View 2: Concept-Extractor Output ───────────────────────────────────────────

def render_concepts(concepts: ConceptOutput) -> None:
    """Render technical concepts breakdown in 6 clear academic sections."""
    st.markdown(
        """
        <div class="apple-glass-card" style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                ✦ Agent 2 · Deep Concept Extraction
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25;">
                Technical Concepts, Architecture & Methodology
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Contributions",
        "⚙️ Methodology",
        "📊 Datasets",
        "📈 Results & Metrics",
        "⚠️ Limitations",
        "🔮 Future Work",
    ])

    def _render_list_items(items: Sequence[str], icon: str = "•") -> None:
        if not items:
            st.info("No items identified in this category.")
            return
        items_html = ""
        for item in items:
            items_html += f"""
            <div class="apple-glass-card-sm" style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <span style="color: var(--apple-accent); font-size: 1.1rem; line-height: 1.3;">{icon}</span>
                <span style="font-size: 0.95rem; line-height: 1.5; color: var(--apple-text);">{item}</span>
            </div>
            """
        st.markdown(items_html, unsafe_allow_html=True)

    with tab1:
        st.markdown("#### Novel Claims & Key Contributions")
        _render_list_items(concepts.key_contributions, "✦")

    with tab2:
        st.markdown("#### Step-by-Step Methodology & System Architecture")
        _render_list_items(concepts.methodology, "⚙️")

    with tab3:
        st.markdown("#### Datasets & Evaluation Benchmarks")
        _render_list_items(concepts.datasets_used, "📊")

    with tab4:
        st.markdown("#### Quantitative Empirical Results & Metrics")
        _render_list_items(concepts.results_and_metrics, "📈")

    with tab5:
        st.markdown("#### Acknowledged Weaknesses & Limitations")
        _render_list_items(concepts.limitations, "⚠️")

    with tab6:
        st.markdown("#### Future Research Directions & Open Questions")
        _render_list_items(concepts.future_work, "🔮")

    _render_sources_expander(concepts.sources, "Retrieved Sources Grounding These Concepts")


# ── View 3: Slide Deck Output & PPTX Export ───────────────────────────────────

def render_slides(slide_deck: SlideDeck, pptx_bytes: bytes | None = None) -> None:
    """Render presentation slides preview with 16:9 cards, speaker notes, and PPTX export."""
    st.markdown(
        f"""
        <div class="apple-glass-card" style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                ✦ Agent 3 · Presentation Slide Deck
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25; margin-bottom: 0.5rem;">
                {slide_deck.paper_title}
            </div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span class="apple-pill apple-pill-blue">📊 {len(slide_deck.slides)} Slides</span>
                <span class="apple-pill">🖥️ 16:9 Widescreen</span>
                <span class="apple-pill apple-pill-green">🎙️ Embedded Speaker Notes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # In-memory PPTX generation if not already cached
    if pptx_bytes is None:
        try:
            pptx_bytes = build_presentation_bytes(slide_deck)
        except Exception as e:
            logger.exception("Error generating PPTX bytes: %s", e)
            pptx_bytes = None

    # Download action bar
    col_dl, col_info = st.columns([2, 3])
    with col_dl:
        if pptx_bytes is not None:
            filename = f"{_sanitize_filename(slide_deck.paper_title)}.pptx"
            st.download_button(
                label="📥 Download Presentation (.pptx)",
                data=pptx_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True,
                help="Download native PowerPoint presentation with full speaker notes.",
            )
        else:
            st.error("PowerPoint file generation currently unavailable.")

    with col_info:
        st.caption("PowerPoint file formatted in 16:9 widescreen with academic slide hierarchy and embedded notes.")

    st.markdown("### 🖼️ Slides Preview")

    # Render each slide card in 16:9 widescreen card style
    for slide in slide_deck.slides:
        with st.container():
            bullets_html = "".join(
                f'<div class="apple-slide-bullet"><span>{b}</span></div>'
                for b in slide.bullets
            )

            st.markdown(
                f"""
                <div class="apple-slide-card">
                    <div class="apple-slide-header">
                        <span class="apple-slide-num">Slide {slide.slide_number} of {len(slide_deck.slides)}</span>
                        <span class="apple-pill">Academic 16:9</span>
                    </div>
                    <div class="apple-slide-title" style="margin-bottom: 1.2rem;">
                        {slide.title}
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        {bullets_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if slide.speaker_note:
                with st.expander(f"🎙️ Speaker Notes (Slide {slide.slide_number})", expanded=False):
                    st.markdown(f"*{slide.speaker_note}*")

    _render_sources_expander(slide_deck.sources, "Retrieved Sources Grounding This Presentation")


# ── View 4: Viva Defense Q&A Output ───────────────────────────────────────────

def render_viva(viva: VivaOutput) -> None:
    """Render viva defense examination questions with category filter and model answers."""
    st.markdown(
        f"""
        <div class="apple-glass-card" style="margin-bottom: 1.5rem;">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                ✦ Agent 4 · Oral Defense & Viva Examination
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25; margin-bottom: 0.5rem;">
                {viva.paper_title}
            </div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <span class="apple-pill apple-pill-blue">🎓 {len(viva.qa_pairs)} Defense Questions</span>
                <span class="apple-pill">5 Academic Categories</span>
                <span class="apple-pill apple-pill-green">📖 Grounded Model Answers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Category Filter
    all_categories = sorted(list({qa.category for qa in viva.qa_pairs}))
    filter_cat = st.selectbox(
        "Filter Questions by Category",
        ["All Categories"] + [c.capitalize() for c in all_categories],
    )

    filtered_pairs = [
        qa for qa in viva.qa_pairs
        if filter_cat == "All Categories" or qa.category.lower() == filter_cat.lower()
    ]

    st.markdown(f"Showing **{len(filtered_pairs)}** defense questions:")

    for i, qa in enumerate(filtered_pairs, start=1):
        diff = str(qa.difficulty).lower()
        if diff == "easy":
            diff_pill = '<span class="apple-pill apple-pill-green">🟢 Foundational</span>'
        elif diff == "hard":
            diff_pill = '<span class="apple-pill apple-pill-red">🔴 Critical Analysis</span>'
        else:
            diff_pill = '<span class="apple-pill apple-pill-amber">🟠 Methodological</span>'

        if qa.source_pages:
            pages_str = ", ".join(f"p. {p}" for p in qa.source_pages)
            source_pill = f'<span class="apple-pill">📄 {pages_str}</span>'
        else:
            source_pill = '<span class="apple-pill">📄 Citation pending</span>'

        cat_pill = f'<span class="apple-pill apple-pill-blue">{qa.category.upper()}</span>'

        st.markdown(
            f"""
            <div class="apple-viva-card">
                <div class="apple-viva-meta">
                    <span style="font-weight: 700; font-size: 0.85rem; color: var(--apple-accent); margin-right: 0.3rem;">Q{i}</span>
                    {cat_pill}
                    {diff_pill}
                    {source_pill}
                </div>
                <div class="apple-viva-q">
                    {qa.question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📖 Reveal Model Answer", expanded=False):
            st.markdown(qa.model_answer)

    _render_sources_expander(viva.sources, "Retrieved Sources Grounding These Questions")


# ── View 5: Interactive Paper RAG Q&A ──────────────────────────────────────────

def render_rag_chat(doc_id: str) -> None:
    """Render Apple-styled interactive paper chat powered by RAGPipeline."""
    st.markdown(
        """
        <div class="apple-glass-card" style="margin-bottom: 1.2rem;">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">
                ✦ Interactive Intelligence · Ask Your Paper
            </div>
            <div style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.25; margin-bottom: 0.4rem;">
                Paper Conversation
            </div>
            <div style="font-size: 0.95rem; color: var(--apple-text-secondary); line-height: 1.5;">
                Ask free-form questions about the paper's methodology, architecture, datasets, or findings.
                Every response is strictly grounded in retrieved chunks with verified page citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested Prompts Chips
    st.markdown("<div style='font-size: 0.82rem; color: gray; font-weight: 600; margin-bottom: 0.4rem;'>SUGGESTED EXPLORATIONS</div>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    suggested_query = None

    with chip_col1:
        if st.button("🚀 What is the core contribution?", key="chip_contrib", use_container_width=True):
            suggested_query = "What is the core technical contribution of this paper?"
    with chip_col2:
        if st.button("📊 Which datasets were tested?", key="chip_datasets", use_container_width=True):
            suggested_query = "Which datasets and benchmarks were used for experimental evaluation?"
    with chip_col3:
        if st.button("⚠️ What are the limitations?", key="chip_limits", use_container_width=True):
            suggested_query = "What limitations or trade-offs do the authors acknowledge?"

    # Display past exchanges
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 0.8rem;">
                    <div class="apple-chat-user">
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 0.8rem;">
                    <div class="apple-chat-assistant">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--apple-accent); font-weight: 700; margin-bottom: 0.4rem;">✦ Research Assistant</div>
                        <div style="line-height: 1.55;">{msg["content"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander("📚 Source Citations", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- **Chunk {s['chunk_index']}** (Page {s['page_number']})")

    # Handle suggested chip query or standard chat_input
    user_query = st.chat_input("Enter your question about this paper...")
    active_prompt = suggested_query or user_query

    if active_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": active_prompt})

        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 0.8rem;">
                <div class="apple-chat-user">
                    {active_prompt}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Searching paper excerpts and formulating grounded answer..."):
            try:
                response = RAGPipeline.ask(question=active_prompt, doc_id=doc_id)

                serialized_sources = [
                    {"chunk_index": s.chunk_index, "page_number": s.page_number, "source": s.source}
                    for s in response.sources
                ]

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": serialized_sources,
                })
                st.rerun()

            except Exception as e:
                logger.exception("Error in RAG Q&A: %s", e)
                error_msg = f"Failed to retrieve or generate answer: {e}"
                st.error(error_msg)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
                st.rerun()
