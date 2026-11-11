import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pdf_pipeline import process_pdf
from core.vector_store import _get_store, similarity_search_with_score
import re

res, chunks = process_pdf(Path("tests/data/sample_research_paper.pdf"))

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "on", "of", "for", "to",
    "and", "or", "did", "does", "do", "how", "why", "who", "which", "where",
    "when", "explain", "describe", "detail", "paper", "research", "authors",
    "report", "proposed", "method"
}

def extract_keywords(query: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    return [w for w in words if w not in STOPWORDS]

def hybrid_rerank(query: str, doc_id: str, top_k: int = 5):
    # 1. Fetch top_k candidates from vector store
    candidates = similarity_search_with_score(query, doc_id, top_k=top_k)
    keywords = extract_keywords(query)
    
    scored = []
    for doc, dist in candidates:
        text_lower = doc.page_content.lower()
        # Semantic score: normalize distance (d=0 -> 1.0, d=2 -> 0.0)
        sem_sim = max(0.0, 1.0 - dist / 2.0)
        
        # Lexical score: keyword presence
        if keywords:
            matches = sum(1 for kw in keywords if kw in text_lower)
            lex_score = matches / len(keywords)
        else:
            lex_score = 0.5
            
        # Composite score (60% semantic + 40% keyword match)
        # Bonus if exact primary keywords (like 'limitations', 'datasets', 'architecture') are in text
        composite = 0.5 * sem_sim + 0.5 * lex_score
        scored.append((doc, composite, sem_sim, lex_score, dist))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, keywords

queries = [
    "What problem does the research paper address?",
    "Explain the proposed architecture.",
    "What methodology was used?",
    "What datasets were used?",
    "What results did the authors achieve?",
    "What are the limitations?",
    "What future improvements are proposed?",
    "What audio speech recognition accuracy does SparseDocNet report?"
]

print("HYBRID RERANKING RESULTS:")
for q in queries:
    ranked, kws = hybrid_rerank(q, res.doc_id, top_k=6)
    top_doc, comp, sem, lex, dist = ranked[0]
    p = top_doc.metadata["page_number"]
    chunk_idx = top_doc.metadata["chunk_index"]
    print(f"\nQ: {q}")
    print(f"  Keywords : {kws}")
    print(f"  Top Match: Page {p} (Chunk {chunk_idx}) | Composite: {comp:.2f} (Lexical: {lex:.2f}, Semantic: {sem:.2f}, Dist: {dist:.2f})")
    print(f"  Snippet  : {top_doc.page_content[:100]}...")
