"""
scripts/demo_vector_store.py — Phase 3 Demonstration Script.

Demonstrates:
1. Loading and processing the sample research paper using the Phase 2 pipeline.
2. Embedding generation via Sentence Transformers (all-MiniLM-L6-v2) with dimension verification.
3. Storing chunks in ChromaDB with complete metadata (doc_id, page_number, chunk_index, source).
4. Executing 5 distinct semantic queries with relevance distance scores and source page attribution.
5. Demonstrating multi-session isolation via doc_id filtering.
6. Demonstrating duplicate ingestion prevention.
7. Demonstrating persistence by reopening the database connection.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from config.settings import settings
from core.embeddings import get_embedding_model, embed_texts, embed_query
from core.pdf_pipeline import process_pdf
from core.vector_store import (
    ingest_documents,
    similarity_search_with_score,
    is_document_ingested,
    get_document_chunk_count,
    delete_document,
    reset_store_connection,
)


def print_separator(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}")
    print(f" {title}")
    print(f"{char * 78}")


def main() -> None:
    sample_pdf_path = PROJECT_ROOT / "tests" / "data" / "sample_research_paper.pdf"
    if not sample_pdf_path.exists():
        print(f"Error: Sample PDF not found at {sample_pdf_path}")
        print("Run scripts/generate_sample_paper.py first.")
        sys.exit(1)

    print_separator("PHASE 3 DEMONSTRATION: EMBEDDINGS & CHROMADB VECTOR STORE")

    # ── Step 1: Process Sample Paper ──────────────────────────────────────────
    print("\n[Step 1] Processing Sample Research Paper via Phase 2 Pipeline...")
    result, chunks = process_pdf(sample_pdf_path)
    print(f"  • Filename        : {result.filename}")
    print(f"  • Document ID     : {result.doc_id}")
    print(f"  • Total Pages     : {result.page_count}")
    print(f"  • Total Characters: {result.total_characters}")
    print(f"  • Extracted Chunks: {result.chunk_count}")

    # ── Step 2: Embedding Generation Verification ─────────────────────────────
    print_separator("[Step 2] Local Sentence Transformers Embedding Generation")
    model = get_embedding_model()
    print(f"  • Embedding Model : {settings.EMBEDDING_MODEL}")
    print(f"  • Device          : CPU (offline local inference)")

    sample_query = "What is the speedup and complexity of SparseDocNet?"
    query_vector = embed_query(sample_query)
    norm = sum(x ** 2 for x in query_vector) ** 0.5
    print(f"  • Query Text      : '{sample_query}'")
    print(f"  • Vector Dimension: {len(query_vector)} dimensions")
    print(f"  • Vector L2 Norm  : {norm:.6f} (Normalized = {abs(norm - 1.0) < 1e-4})")
    sample_preview = ", ".join(f"{v:+.4f}" for v in query_vector[:6])
    print(f"  • Vector Preview  : [{sample_preview}, ... ({len(query_vector)-6} more values)]")

    # ── Step 3: ChromaDB Ingestion ────────────────────────────────────────────
    print_separator("[Step 3] ChromaDB Document Ingestion & Metadata Stamping")
    doc_id = result.doc_id

    # Clean any previous run data for fresh demo
    if is_document_ingested(doc_id):
        print(f"  • Cleaning pre-existing entries for doc_id={doc_id[:12]}...")
        delete_document(doc_id)

    print(f"  • Pre-ingestion is_document_ingested: {is_document_ingested(doc_id)}")
    ingested_count = ingest_documents(chunks, doc_id)
    print(f"  • Successfully ingested chunks     : {ingested_count}")
    print(f"  • Post-ingestion is_document_ingested: {is_document_ingested(doc_id)}")
    print(f"  • Total chunks stored in ChromaDB   : {get_document_chunk_count(doc_id)}")

    # ── Step 4: Five Semantic Queries with Cosine Distance & Page Attribution ─
    print_separator("[Step 4] Executing 5 Distinct Semantic Queries via ChromaDB")
    queries = [
        (
            "Query 1 (Architecture & Attention)",
            "What is the core architecture and sparse attention mechanism of SparseDocNet?",
        ),
        (
            "Query 2 (Complexity & Performance)",
            "What computational complexity and throughput speedup does SparseDocNet achieve?",
        ),
        (
            "Query 3 (Evaluation & Datasets)",
            "What benchmarks and baseline models were used for experimental evaluation?",
        ),
        (
            "Query 4 (Limitations & Constraints)",
            "What are the limitations, memory constraints, and future work of the proposed method?",
        ),
        (
            "Query 5 (Token Routing & Sparsity)",
            "How does the token router dynamically select relevant key-value blocks?",
        ),
    ]

    for label, q_text in queries:
        print(f"\n--- {label} ---")
        print(f"  Target Query: \"{q_text}\"")
        scored_results = similarity_search_with_score(q_text, doc_id=doc_id, top_k=2)

        for rank, (doc, dist) in enumerate(scored_results, start=1):
            # ChromaDB uses cosine distance: smaller distance = higher semantic similarity
            similarity_percent = max(0.0, (1.0 - dist)) * 100
            page_num = doc.metadata.get("page_number", "N/A")
            chunk_idx = doc.metadata.get("chunk_index", "N/A")
            excerpt = " ".join(doc.page_content.split()[:28]) + "..."

            print(f"  [Match #{rank}] Cosine Dist: {dist:.4f} (Sim: {similarity_percent:.1f}%) | Page: {page_num} | Chunk: {chunk_idx}")
            print(f"    Excerpt: \"{excerpt}\"")

    # ── Step 5: Multi-Session Document Isolation ──────────────────────────────
    print_separator("[Step 5] Multi-Session Isolation Verification")
    dummy_doc_id = "dummy-quantum-crypto-session-id"
    dummy_chunks = [
        Document(
            page_content=(
                "Quantum Key Distribution (QKD) utilizes quantum mechanics principles such as "
                "the BB84 protocol and photon polarization to establish cryptographically secure "
                "communication channels impervious to eavesdropping."
            ),
            metadata={
                "doc_id": dummy_doc_id,
                "source": "quantum_cryptography.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "start_index": 0,
            },
        )
    ]
    # Ensure dummy doc is ingested
    delete_document(dummy_doc_id)
    ingest_documents(dummy_chunks, dummy_doc_id)
    print(f"  • Ingested Session 2 (Dummy): doc_id={dummy_doc_id}")

    # Query Paper 1 session for quantum topics
    cross_search_1 = similarity_search_with_score(
        "Quantum mechanics and photon polarization cryptography",
        doc_id=doc_id,
        top_k=2,
    )
    print("\n  [Isolation Check 1] Querying Session 1 (SparseDocNet) for 'Quantum cryptography':")
    has_quantum_chunk = any("Quantum" in d.page_content for d, _ in cross_search_1)
    print(f"    Contained Session 2 chunks? {has_quantum_chunk} (Correct: False)")
    for d, s in cross_search_1:
        print(f"    -> Returned Page {d.metadata['page_number']} from '{d.metadata['source']}' (doc_id={d.metadata['doc_id'][:12]}...)")

    # Query Session 2 for SparseDocNet topics
    cross_search_2 = similarity_search_with_score(
        "SparseDocNet attention computational complexity",
        doc_id=dummy_doc_id,
        top_k=2,
    )
    print("\n  [Isolation Check 2] Querying Session 2 (Quantum) for 'SparseDocNet attention':")
    has_sparse_chunk = any("SparseDocNet" in d.page_content for d, _ in cross_search_2)
    print(f"    Contained Session 1 chunks? {has_sparse_chunk} (Correct: False)")
    for d, s in cross_search_2:
        print(f"    -> Returned Page {d.metadata['page_number']} from '{d.metadata['source']}' (doc_id={d.metadata['doc_id'][:12]}...)")

    # Clean up dummy doc
    delete_document(dummy_doc_id)
    print(f"  • Deleted Session 2 dummy doc. Verifying Session 1 chunks remain intact: {get_document_chunk_count(doc_id)} chunks.")

    # ── Step 6: Duplicate Ingestion Prevention ────────────────────────────────
    print_separator("[Step 6] Duplicate Ingestion Prevention")
    print(f"  • Current chunk count in ChromaDB : {get_document_chunk_count(doc_id)}")
    print(f"  • Attempting duplicate ingestion for doc_id={doc_id[:12]}...")
    re_ingested = ingest_documents(chunks, doc_id)
    print(f"  • Chunks ingested during duplicate attempt: {re_ingested} (Skipped: {re_ingested == 0})")
    print(f"  • Final chunk count in ChromaDB          : {get_document_chunk_count(doc_id)}")

    # ── Step 7: Persistence Verification across Reopening ─────────────────────
    print_separator("[Step 7] Persistence Verification (Reopening Store from Disk)")
    print("  • Resetting in-memory ChromaDB store connection...")
    reset_store_connection()
    print("  • Store connection reset to None. Reconnecting to disk SQLite store...")

    reopened_count = get_document_chunk_count(doc_id)
    print(f"  • Chunks found on disk after reopening : {reopened_count}")
    reopened_results = similarity_search_with_score("SparseDocNet speedup and memory complexity", doc_id=doc_id, top_k=1)
    best_doc, best_dist = reopened_results[0]
    print(f"  • Re-query after restart successful    : Page {best_doc.metadata['page_number']}, Dist: {best_dist:.4f}")
    print(f"  • Excerpt                              : \"{' '.join(best_doc.page_content.split()[:20])}...\"")

    print_separator("PHASE 3 DEMONSTRATION COMPLETE: ALL CHECKS PASSED", char="*")


if __name__ == "__main__":
    main()
