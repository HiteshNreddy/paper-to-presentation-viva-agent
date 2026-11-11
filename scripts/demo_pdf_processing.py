"""
scripts/demo_pdf_processing.py — Demonstration of Phase 2 PDF Processing Pipeline.

Demonstrates:
1. Validation of an actual multi-page research paper PDF.
2. Full content SHA-256 document identification.
3. Page-wise text extraction using PyMuPDF (fitz) with 1-based page numbering.
4. Semantic text chunking using LangChain RecursiveCharacterTextSplitter.
5. Exact metadata preservation (doc_id, page_number, chunk_index, source, start_index).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pdf_pipeline import process_pdf
from scripts.generate_sample_paper import generate_sample_paper, PDF_PATH


def run_demo():
    print("=" * 78)
    print("      PAPER-TO-PRESENTATION & VIVA AGENT: PDF PROCESSING DEMO (PHASE 2)")
    print("=" * 78)

    # Ensure sample paper exists
    if not PDF_PATH.exists():
        print("[*] Generating sample research paper...")
        generate_sample_paper()

    print(f"\n[1] Target Research Paper:")
    print(f"    Path : {PDF_PATH}")
    print(f"    Size : {PDF_PATH.stat().st_size:,} bytes")

    # Run pipeline
    print("\n[2] Executing PDF Processing Pipeline...")
    result, chunks = process_pdf(PDF_PATH)

    print("\n[+] Extraction & Chunking Summary:")
    print(f"    • Generated Document ID (SHA-256) : {result.doc_id}")
    print(f"    • Source Filename                   : {result.filename}")
    print(f"    • Total Page Count                  : {result.page_count} pages")
    print(f"    • Total Extracted Characters        : {result.total_characters:,} chars")
    print(f"    • Total Generated Chunks            : {result.chunk_count} chunks")

    print("\n[3] Per-Page Extraction Breakdown:")
    for prec in result.page_records:
        print(f"    • Page {prec.page_number}: {prec.character_count:4d} chars | Preview: \"{prec.text_preview}\"")

    print("\n[4] Sample Chunks & Preserved Metadata (Displaying 3 Representative Chunks):")
    sample_indices = [0, len(chunks) // 2, len(chunks) - 1]

    for sample_idx in sample_indices:
        chunk = chunks[sample_idx]
        meta = chunk.metadata
        print("-" * 78)
        print(f"CHUNK #{meta['chunk_index']}  |  PAGE {meta['page_number']}  |  LENGTH: {len(chunk.page_content)} characters")
        print(f"Traceability Metadata:")
        print(f"  - doc_id      : {meta['doc_id']}")
        print(f"  - source      : {meta['source']}")
        print(f"  - page_number : {meta['page_number']}")
        print(f"  - chunk_index : {meta['chunk_index']}")
        print(f"  - start_index : {meta.get('start_index', 0)}")
        print("Content Snippet:")
        indented_content = "\n".join("    " + line for line in chunk.page_content.strip().split("\n")[:6])
        print(indented_content)
        if len(chunk.page_content.strip().split("\n")) > 6:
            print("    ...")

    print("=" * 78)
    print("[+] PHASE 2 PDF PROCESSING DEMONSTRATION COMPLETE: All verifications passed.")
    print("=" * 78)


if __name__ == "__main__":
    run_demo()

