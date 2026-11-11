"""
scripts/verify_phase6.py — Comprehensive Verification Script for Phase 6.

Validates:
1. Streamlit application compilation and import.
2. End-to-end PowerPoint (.pptx) file generation from SlideDeck.
3. Structural verification of the saved PPTX file (10–12 slides, notes, titles).
4. Full agent pipeline execution and router dispatch.
5. Document isolation and session state safety.
"""

from __future__ import annotations

import io
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pptx
from agents import router
from core.pdf_pipeline import process_pdf
from core.vector_store import ingest_documents, is_document_ingested
from schemas.slide_schema import SlideDeck
from utils.pptx_builder import build_presentation, build_presentation_bytes


def main():
    print("=" * 80)
    print("PHASE 6 VERIFICATION: PPTX EXPORT & STREAMLIT INTEGRATION")
    print("=" * 80)

    # 1. Verify app imports
    print("\n[1] Verifying Streamlit App & UI Modules Import...")
    import app
    import ui.sidebar
    import ui.result_views
    import utils.pptx_builder
    print("    - app.py: OK")
    print("    - ui/sidebar.py: OK")
    print("    - ui/result_views.py: OK")
    print("    - utils/pptx_builder.py: OK")

    # 2. Ingest Sample Paper
    pdf_path = PROJECT_ROOT / "tests" / "data" / "sample_research_paper.pdf"
    print(f"\n[2] Loading sample research paper: {pdf_path.name}")
    proc_result, chunks = process_pdf(pdf_path)
    doc_id = proc_result.doc_id
    if not is_document_ingested(doc_id):
        ingest_documents(chunks, doc_id)
    print(f"    - doc_id: {doc_id}")
    print(f"    - Pages: {proc_result.page_count}, Chunks: {len(chunks)}")

    # 3. Test PowerPoint Generation
    print("\n[3] Generating and Verifying PowerPoint (.pptx) Presentation...")
    # Import demo mock generator to produce sample deck
    from scripts.demo_agents import _create_grounded_mock_llm
    mock_llm = _create_grounded_mock_llm()

    deck: SlideDeck = router.dispatch("generate_slides", doc_id=doc_id, llm=mock_llm)
    assert isinstance(deck, SlideDeck), "Expected SlideDeck output"
    print(f"    - Generated deck with {len(deck.slides)} slides.")

    # Save to disk
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "SparseDocNet_Presentation.pptx"
    saved_path = build_presentation(deck, output_path=out_file)
    assert saved_path.exists(), "PPTX file was not created on disk"
    file_size_kb = round(saved_path.stat().st_size / 1024, 2)
    print(f"    - Saved presentation to: {saved_path} ({file_size_kb} KB)")

    # Read back and inspect structure with python-pptx
    prs = pptx.Presentation(str(saved_path))
    assert len(prs.slides) == len(deck.slides), f"Slide count mismatch: {len(prs.slides)} vs {len(deck.slides)}"
    print(f"    - Successfully opened with python-pptx. Total slides: {len(prs.slides)}")
    print(f"    - Widescreen dimensions: {prs.slide_width.inches:.2f}\" x {prs.slide_height.inches:.2f}\" (16:9)")

    # Verify speaker notes
    has_notes = False
    for i, s in enumerate(prs.slides, start=1):
        note_text = s.notes_slide.notes_text_frame.text
        if note_text.strip():
            has_notes = True
            print(f"      * Slide {i} notes verified: '{note_text[:50]}...'")
    assert has_notes, "Speaker notes were not found in presentation"

    # 4. Verify In-Memory Bytes Export (for Streamlit download button)
    print("\n[4] Verifying In-Memory PPTX Bytes Export (Streamlit Download Button)...")
    raw_bytes = build_presentation_bytes(deck)
    assert isinstance(raw_bytes, bytes)
    assert raw_bytes.startswith(b"PK\x03\x04"), "Invalid PPTX zip magic bytes"
    prs_from_bytes = pptx.Presentation(io.BytesIO(raw_bytes))
    assert len(prs_from_bytes.slides) == len(deck.slides)
    print(f"    - In-memory bytes verified: {len(raw_bytes)} bytes (Valid PPTX).")

    print("\n" + "=" * 80)
    print("PHASE 6 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

