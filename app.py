"""
app.py — Streamlit Application Entry Point for Paper-to-Presentation & Viva Agent.
Apple iPhone Duo / iOS 27 Inspired Frontend.

Orchestrates:
1. Dynamic theme-based Apple CSS styling (Light & Dark modes).
2. PDF file upload and document session management (via ui/sidebar.py).
3. Routing user requests to the four specialized agents (via agents/router.py).
4. Presenting validated Pydantic outputs and interactive Q&A (via ui/result_views.py).
5. PowerPoint (.pptx) file generation and export (via utils/pptx_builder.py).
"""

from __future__ import annotations

import logging
import streamlit as st

# 1. Page Configuration (must be called first)
st.set_page_config(
    page_title="Paper Duo · Research-to-Presentation OS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agents import router
from core.llm import DailyQuotaExceededError
from core.rag_context import InsufficientContextError
from ui.result_views import (
    render_summary,
    render_concepts,
    render_slides,
    render_viva,
    render_rag_chat,
)
from ui.sidebar import render_sidebar
from ui.styles import get_apple_css
from utils.pptx_builder import build_presentation_bytes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("app")


def _render_quota_error(e: DailyQuotaExceededError) -> None:
    """Display an informative Apple-styled message when the daily free-tier quota is reached."""
    delay_msg = f" You can retry in **{e.retry_delay}**." if e.retry_delay else " Please wait until your daily quota resets."
    st.error(
        f"⏳ **Gemini Free-Tier Daily Quota Exceeded (Limit: 20 requests/day)**\n\n"
        f"Your Google Gemini API key has exhausted its daily Free Tier limit of 20 requests.{delay_msg}\n\n"
        f"*(Retries were halted immediately to prevent redundant quota burn.)*\n\n"
        f"💡 **Tip:** To lift this daily limit, enable Pay-As-You-Go billing on "
        f"[Google AI Studio](https://aistudio.google.com/app/apikey) or wait for your daily quota to reset."
    )


def main():
    # ── Initialize Theme & Inject Dynamic Apple CSS ────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    theme = st.session_state["theme"]
    st.markdown(get_apple_css(theme), unsafe_allow_html=True)

    # ── Render Apple Sidebar ───────────────────────────────────────────────────
    sidebar_state = render_sidebar()
    doc_id = sidebar_state.get("active_doc_id")
    doc_info = sidebar_state.get("doc_info")

    # ── Top Navigation Bar ─────────────────────────────────────────────────────
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        if doc_id and doc_info:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.8rem;">
                    <span class="apple-pill apple-pill-green">● Active Session</span>
                    <span class="apple-pill"><b>{doc_info.get('filename')}</b></span>
                    <span class="apple-pill">📄 {doc_info.get('page_count')} Pages</span>
                    <span class="apple-pill">🧩 {doc_info.get('chunk_count')} Chunks</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.8rem;">
                    <span class="apple-pill apple-pill-amber">○ Standby</span>
                    <span class="apple-pill">No Research Paper Loaded</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_nav2:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 0.82rem; color: gray; padding-top: 0.4rem;">
                OS 27 · <b>{theme.capitalize()}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── State A: No Document Active (Hero View) ────────────────────────────────
    if not doc_id:
        st.markdown(
            """
            <div class="apple-hero">
                <div class="apple-hero-badge">
                    ✦ Autonomous Multi-Agent Intelligence
                </div>
                <div class="apple-hero-title">
                    Turn Research Papers Into Presentations
                </div>
                <div class="apple-hero-subtitle">
                    Transform complex academic PDFs into executive summaries, technical methodology breakdowns,
                    widescreen presentation decks with speaker notes, and comprehensive viva defense examinations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: var(--apple-text);">
                Specialized Autonomous Agents
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="apple-glass-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Agent 1 · Research Summarizer
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); line-height: 1.5;">
                        Extracts structured executive summaries including the core problem statement, proposed methodology,
                        key empirical findings, and academic conclusions grounded in source citations.
                    </div>
                </div>
                <div class="apple-glass-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Agent 3 · Slide Deck & PPTX
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); line-height: 1.5;">
                        Constructs 10–12 widescreen (16:9) presentation slides with clear bullet hierarchies,
                        deep presenter notes, and instant downloadable .pptx export.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="apple-glass-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Agent 2 · Deep Concept Extractor
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); line-height: 1.5;">
                        Deconstructs novel contributions, mathematical architectures, datasets used, quantitative
                        metrics, acknowledged limitations, and open future research directions.
                    </div>
                </div>
                <div class="apple-glass-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎓</div>
                    <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Agent 4 · Viva Defense Prep
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); line-height: 1.5;">
                        Formulates 20 rigorous oral examination questions across 5 academic categories with
                        difficulty indicators and verified source-grounded model answers.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # ── State B: Document Active — Multi-Agent Workspace ───────────────────────
    # Action Header
    col_bar1, col_bar2 = st.columns([3, 1])
    with col_bar1:
        st.markdown(
            f"""
            <div style="font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem; color: var(--apple-text);">
                Workspace · {doc_info.get('filename')}
            </div>
            <div style="font-size: 0.88rem; color: var(--apple-text-secondary);">
                SHA-256: <code>{doc_id[:16]}...</code> · {doc_info.get('page_count')} Pages · {doc_info.get('chunk_count')} Chunks
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bar2:
        if st.button("⚡ Generate Full Suite", use_container_width=True, type="primary"):
            progress_bar = st.progress(0, text="Launching Multi-Agent Pipeline...")
            try:
                progress_bar.progress(15, text="1/4: Synthesizing Research Summary...")
                st.session_state["summary_output"] = router.dispatch("summarize", doc_id=doc_id)

                progress_bar.progress(40, text="2/4: Extracting Technical Concepts & Methods...")
                st.session_state["concept_output"] = router.dispatch("extract_concepts", doc_id=doc_id)

                progress_bar.progress(65, text="3/4: Designing Academic Slide Deck & PPTX...")
                deck = router.dispatch("generate_slides", doc_id=doc_id)
                st.session_state["slide_deck"] = deck
                st.session_state["pptx_bytes"] = build_presentation_bytes(deck)

                progress_bar.progress(90, text="4/4: Formulating 20 Viva Defense Questions...")
                st.session_state["viva_output"] = router.dispatch("viva", doc_id=doc_id)

                progress_bar.progress(100, text="All agents generated successfully!")
                st.success("Complete research suite generated successfully!")
                st.rerun()
            except DailyQuotaExceededError as q_err:
                logger.error("Full suite generation daily quota exceeded: %s", q_err)
                _render_quota_error(q_err)
            except Exception as e:
                logger.exception("Full suite generation failed: %s", e)
                st.error(f"Generation failed: {e}")

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # ── Workspace Tabs ─────────────────────────────────────────────────────────
    tab_summary, tab_concepts, tab_slides, tab_viva, tab_chat = st.tabs([
        "📄 Paper Summary",
        "🧠 Concepts & Methodology",
        "📊 Slide Deck & PPTX",
        "🎓 Viva Defense Prep",
        "💬 Ask Paper (Q&A)",
    ])

    # ── Tab 1: Summarizer ─────────────────────────────────────────────────────
    with tab_summary:
        if "summary_output" in st.session_state:
            render_summary(st.session_state["summary_output"])
        else:
            st.markdown(
                """
                <div class="apple-glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Executive Summary Pending
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); max-width: 500px; margin: 0 auto 1.2rem auto;">
                        Synthesize the core problem statement, proposed methodology, empirical findings, and conclusions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Generate Paper Summary", type="primary", key="btn_gen_summary"):
                with st.spinner("Summarizer Agent is analyzing paper sections..."):
                    try:
                        summary = router.dispatch("summarize", doc_id=doc_id)
                        st.session_state["summary_output"] = summary
                        st.rerun()
                    except InsufficientContextError as e:
                        st.warning(f"Insufficient context for summary: {e}")
                    except DailyQuotaExceededError as q_err:
                        _render_quota_error(q_err)
                    except Exception as e:
                        logger.exception("Summarizer error: %s", e)
                        st.error(f"Summarizer failed: {e}")

    # ── Tab 2: Concept Extractor ──────────────────────────────────────────────
    with tab_concepts:
        if "concept_output" in st.session_state:
            render_concepts(st.session_state["concept_output"])
        else:
            st.markdown(
                """
                <div class="apple-glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Technical Concepts Pending
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); max-width: 500px; margin: 0 auto 1.2rem auto;">
                        Extract novel contributions, step-by-step algorithms, datasets, benchmark metrics, and limitations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Extract Technical Concepts", type="primary", key="btn_gen_concepts"):
                with st.spinner("Concept-Extractor Agent is analyzing architecture and algorithms..."):
                    try:
                        concepts = router.dispatch("extract_concepts", doc_id=doc_id)
                        st.session_state["concept_output"] = concepts
                        st.rerun()
                    except InsufficientContextError as e:
                        st.warning(f"Insufficient context for concepts: {e}")
                    except DailyQuotaExceededError as q_err:
                        _render_quota_error(q_err)
                    except Exception as e:
                        logger.exception("Concept extractor error: %s", e)
                        st.error(f"Concept extractor failed: {e}")

    # ── Tab 3: Slide Deck & PPTX Export ───────────────────────────────────────
    with tab_slides:
        if "slide_deck" in st.session_state:
            pptx_bytes = st.session_state.get("pptx_bytes")
            render_slides(st.session_state["slide_deck"], pptx_bytes=pptx_bytes)
        else:
            st.markdown(
                """
                <div class="apple-glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Presentation Slide Deck Pending
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); max-width: 500px; margin: 0 auto 1.2rem auto;">
                        Generate 10–12 academic 16:9 slides with structured bullets, presenter notes, and instant PowerPoint download.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Generate Presentation Deck & PPTX", type="primary", key="btn_gen_slides"):
                with st.spinner("Slide-Generator Agent is constructing 16:9 presentation slides..."):
                    try:
                        deck = router.dispatch("generate_slides", doc_id=doc_id)
                        pptx_bytes = build_presentation_bytes(deck)
                        st.session_state["slide_deck"] = deck
                        st.session_state["pptx_bytes"] = pptx_bytes
                        st.rerun()
                    except InsufficientContextError as e:
                        st.warning(f"Insufficient context for slide deck: {e}")
                    except DailyQuotaExceededError as q_err:
                        _render_quota_error(q_err)
                    except Exception as e:
                        logger.exception("Slide generator error: %s", e)
                        st.error(f"Slide generator failed: {e}")

    # ── Tab 4: Viva Defense Prep ──────────────────────────────────────────────
    with tab_viva:
        if "viva_output" in st.session_state:
            render_viva(st.session_state["viva_output"])
        else:
            st.markdown(
                """
                <div class="apple-glass-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎓</div>
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; color: var(--apple-text);">
                        Viva Defense Preparation Pending
                    </div>
                    <div style="font-size: 0.92rem; color: var(--apple-text-secondary); max-width: 500px; margin: 0 auto 1.2rem auto;">
                        Formulate 20 rigorous oral defense questions with difficulty ratings and verified source citations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Generate Viva Defense Questions", type="primary", key="btn_gen_viva"):
                with st.spinner("Viva Agent is preparing 20 examination defense questions..."):
                    try:
                        viva = router.dispatch("viva", doc_id=doc_id)
                        st.session_state["viva_output"] = viva
                        st.rerun()
                    except InsufficientContextError as e:
                        st.warning(f"Insufficient context for viva questions: {e}")
                    except DailyQuotaExceededError as q_err:
                        _render_quota_error(q_err)
                    except Exception as e:
                        logger.exception("Viva agent error: %s", e)
                        st.error(f"Viva agent failed: {e}")

    # ── Tab 5: Ask Paper (RAG Chat) ───────────────────────────────────────────
    with tab_chat:
        render_rag_chat(doc_id=doc_id)


if __name__ == "__main__":
    main()
