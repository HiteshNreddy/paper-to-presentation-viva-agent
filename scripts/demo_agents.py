"""
scripts/demo_agents.py — Phase 5 Demonstration Script: The Four Specialized Agents.

Demonstrates:
1. Document ingestion into ChromaDB (SparseDocNet sample paper).
2. Summarizer Agent: structured summary (title, problem, method, findings, conclusion, sources).
3. Concept-Extractor Agent: 6 technical extraction categories with grounded facts.
4. Slide-Generator Agent: 10–12 presentation slides with bullets, speaker notes, and page numbers.
5. Viva Agent: 20 comprehensive oral defense questions across 5 exam categories with difficulty ratings.
6. Router Agent: intent detection and dispatching to each agent.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("demo_agents")

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import summarizer, concept_extractor, slide_generator, viva_agent, router
from config.settings import settings
from core.llm import get_llm
from core.pdf_pipeline import process_pdf
from core.vector_store import ingest_documents, is_document_ingested


def _is_live_key_configured() -> bool:
    """Check if a real Google API key is configured in the environment."""
    key = settings.GOOGLE_API_KEY.strip()
    return bool(key and key != "your_google_gemini_api_key_here" and not key.startswith("test-"))


def _create_grounded_mock_llm():
    """
    Simulated grounded LLM for demonstration when a live API key is not present
    or when running in an offline test harness.
    """
    def mock_llm_fn(input_val) -> AIMessage:
        # Determine prompt type by inspecting prompt messages
        prompt_str = ""
        if hasattr(input_val, "to_string"):
            prompt_str = input_val.to_string()
        elif hasattr(input_val, "messages"):
            prompt_str = " ".join([m.content for m in input_val.messages])
        else:
            prompt_str = str(input_val)

        lower_prompt = prompt_str.lower()

        # 1. Intent Router
        if "intent classifier" in lower_prompt:
            user_part = lower_prompt.split("user request:")[-1] if "user request:" in lower_prompt else lower_prompt
            if "summar" in user_part:
                return AIMessage(content=json.dumps({"intent": "summarize", "confidence": 0.98}))
            elif "concept" in user_part or "method" in user_part:
                return AIMessage(content=json.dumps({"intent": "extract_concepts", "confidence": 0.96}))
            elif "slide" in user_part or "deck" in user_part or "presentation" in user_part:
                return AIMessage(content=json.dumps({"intent": "generate_slides", "confidence": 0.97}))
            elif "viva" in user_part or "question" in user_part or "exam" in user_part:
                return AIMessage(content=json.dumps({"intent": "viva", "confidence": 0.99}))
            return AIMessage(content=json.dumps({"intent": "unknown", "confidence": 0.0}))

        # 2. Summarizer Agent
        if "structured summary" in lower_prompt or "expert research analyst" in lower_prompt:
            payload = {
                "title": "SparseDocNet: Efficient Document Representation with Dynamic Sparse Attention",
                "abstract_summary": "SparseDocNet addresses the quadratic computational bottleneck of dense self-attention for long-document understanding. By introducing an Adaptive Dynamic Routing (ADR) mechanism, it achieves 3.8x faster inference and a 62% reduction in memory footprint while maintaining 94.2% document classification accuracy across standard benchmarks.",
                "problem_statement": "Dense self-attention in Transformer architectures exhibits quadratic O(n^2) computational complexity and GPU memory consumption, severely constraining sequence length and scalability on multi-page document analysis tasks.",
                "proposed_solution": "The paper introduces Adaptive Dynamic Routing (ADR), which dynamically clusters input tokens into hierarchical blocks and computes sparse multi-head self-attention only across top-k most relevant token clusters.",
                "key_findings": [
                    "Achieves 94.2% classification accuracy on multi-page legal and scientific document datasets.",
                    "Delivers 3.8x inference speedup over standard dense BERT and RoBERTa baselines.",
                    "Reduces peak GPU memory footprint by 62%, enabling efficient processing of 8,192-token sequences on a single GPU.",
                    "Ablation studies demonstrate that dynamic cluster routing retains 98.7% of the attention mass of dense models."
                ],
                "conclusion": "SparseDocNet demonstrates that dynamic sparse attention guided by token clustering provides an effective and memory-efficient paradigm for document representation without sacrificing downstream task accuracy."
            }
            return AIMessage(content=json.dumps(payload))

        # 3. Concept-Extractor Agent
        if "detailed technical analysis" in lower_prompt or "six categories of information" in lower_prompt:
            payload = {
                "key_contributions": [
                    "Adaptive Dynamic Routing (ADR) mechanism that dynamically routes attention between token clusters.",
                    "Block-sparse attention kernel achieving 3.8x throughput speedup on modern tensor accelerators.",
                    "Extensive empirical benchmarking across DocBank, RVL-CDIP, and Contract-8K datasets."
                ],
                "methodology": [
                    "Tokens are partitioned into non-overlapping local blocks of size B=64.",
                    "A lightweight routing network predicts relevance scores between block centroids using cosine similarity.",
                    "Top-k relevant block pairs are selected for dense attention computation, while remaining pairs are pruned.",
                    "Residual connections and pre-layer normalization stabilize optimization across deep routing stacks."
                ],
                "datasets_used": [
                    "DocBank (500K document pages with token-level layout annotations).",
                    "RVL-CDIP (400K multi-page document images across 16 categories).",
                    "Contract-8K (specialized legal contracts dataset)."
                ],
                "results_and_metrics": [
                    "94.2% document classification accuracy on RVL-CDIP.",
                    "3.8x end-to-end inference speedup over standard BERT-base.",
                    "62% lower peak GPU memory usage on 8,192 token sequences.",
                    "91.7 F1-score on DocBank token labeling."
                ],
                "limitations": [
                    "Routing overhead incurs a 4-7% latency penalty on short sequences (n < 512 tokens).",
                    "Requires fixed block size tuning (B=64) per hardware architecture for optimal kernel tiling."
                ],
                "future_work": [
                    "Extend Adaptive Dynamic Routing to autoregressive decoder models for document generation.",
                    "Explore learned hardware-aware block sizing for edge devices."
                ]
            }
            return AIMessage(content=json.dumps(payload))

        # 4. Slide-Generator Agent
        if "presentation designer" in lower_prompt or "10–12 slides" in lower_prompt:
            slides = [
                {
                    "slide_number": 1,
                    "title": "SparseDocNet: Dynamic Sparse Attention for Documents",
                    "bullets": [
                        "Efficient document representation for long sequences",
                        "Addresses quadratic O(n^2) Transformer bottlenecks",
                        "Introduces Adaptive Dynamic Routing (ADR)",
                        "Evaluated on multi-page scientific and legal corpora"
                    ],
                    "speaker_note": "Welcome everyone. Today we present SparseDocNet, an architecture that overcomes memory limits in long-document processing."
                },
                {
                    "slide_number": 2,
                    "title": "Introduction & Motivation",
                    "bullets": [
                        "Real-world documents often span thousands of tokens",
                        "Dense Transformers scale quadratically with sequence length",
                        "Standard truncation discards critical trailing context",
                        "Need for scalable, sub-quadratic attention mechanisms"
                    ],
                    "speaker_note": "Emphasize why long document modeling is vital in domains like legal tech, healthcare, and research."
                },
                {
                    "slide_number": 3,
                    "title": "Problem Statement",
                    "bullets": [
                        "Standard self-attention computes n x n similarity matrix",
                        "8,192 token sequences exceed consumer GPU memory limits",
                        "Existing sparse patterns use fixed heuristic masks",
                        "Fixed masks fail to capture arbitrary cross-page dependencies"
                    ],
                    "speaker_note": "Highlight the failure of static windowed attention when dependencies span distant paragraphs."
                },
                {
                    "slide_number": 4,
                    "title": "Research Objectives",
                    "bullets": [
                        "Develop content-dependent dynamic routing for attention",
                        "Preserve high-relevance token interactions across pages",
                        "Reduce memory consumption to enable 8K+ token contexts",
                        "Match or exceed dense Transformer classification accuracy"
                    ],
                    "speaker_note": "Our goal is full expressive power without the quadratic memory overhead."
                },
                {
                    "slide_number": 5,
                    "title": "Background & Related Work",
                    "bullets": [
                        "Longformer & BigBird: Static sliding window + global tokens",
                        "Reformer: Locality-sensitive hashing with high constant factors",
                        "Routing Transformers: K-means clustering of attention keys",
                        "SparseDocNet differs by introducing hierarchical block routing"
                    ],
                    "speaker_note": "Position SparseDocNet against Longformer and Reformer, pointing out why block routing is hardware-friendly."
                },
                {
                    "slide_number": 6,
                    "title": "Proposed Architecture: SparseDocNet",
                    "bullets": [
                        "Two-tiered hierarchical attention framework",
                        "Block-level routing network predicts inter-block affinity",
                        "Fine-grained token attention computed only on active block pairs",
                        "Hardware-aligned block tiling (B=64) maximizes GPU tensor core utilization"
                    ],
                    "speaker_note": "Explain the two levels: block-level selection first, then dense computation only where needed."
                },
                {
                    "slide_number": 7,
                    "title": "Methodology: Adaptive Dynamic Routing (ADR)",
                    "bullets": [
                        "Step 1: Partition sequence into blocks of size B=64",
                        "Step 2: Compute centroid summary vectors via mean pooling",
                        "Step 3: Lightweight router evaluates inter-block cosine similarity",
                        "Step 4: Top-k clusters attend to each other; remainder pruned"
                    ],
                    "speaker_note": "Walk through the four mathematical stages of the Adaptive Dynamic Routing algorithm."
                },
                {
                    "slide_number": 8,
                    "title": "System Implementation & Training",
                    "bullets": [
                        "Implemented in PyTorch with custom Triton CUDA kernels",
                        "Trained on 8x NVIDIA A100 GPUs (40GB) with FP16 mixed precision",
                        "AdamW optimizer with warmup-decay cosine schedule",
                        "Combined routing loss and cross-entropy classification objective"
                    ],
                    "speaker_note": "Note the use of custom Triton kernels to achieve actual wall-clock speedup on GPUs."
                },
                {
                    "slide_number": 9,
                    "title": "Experimental Setup & Datasets",
                    "bullets": [
                        "RVL-CDIP: 400,000 multi-page scanned document images",
                        "DocBank: 500,000 document pages with layout bounding boxes",
                        "Contract-8K: Multi-page legal agreements with complex clauses",
                        "Baselines: Standard BERT, Longformer, BigBird, and Reformer"
                    ],
                    "speaker_note": "Review the diverse experimental corpus covering images, text layouts, and legal agreements."
                },
                {
                    "slide_number": 10,
                    "title": "Results & Quantitative Evaluation",
                    "bullets": [
                        "94.2% classification accuracy on RVL-CDIP (matches dense BERT)",
                        "3.8x faster end-to-end inference throughput on 8K sequences",
                        "62% reduction in peak GPU memory usage during training",
                        "Outperforms static Longformer by +1.4% accuracy"
                    ],
                    "speaker_note": "Emphasize that SparseDocNet matched dense accuracy while slashing latency and memory."
                },
                {
                    "slide_number": 11,
                    "title": "Limitations & Critical Discussion",
                    "bullets": [
                        "Routing network adds 4-7% computational overhead on short sequences",
                        "Requires hardware-specific tuning of block size B for optimal speed",
                        "Hyperparameter k (top-k clusters) creates an accuracy-speed trade-off",
                        "Evaluated primarily on English-language document corpora"
                    ],
                    "speaker_note": "Discuss limitations honestly: for very short inputs, standard dense attention is still faster."
                },
                {
                    "slide_number": 12,
                    "title": "Conclusion & Future Work",
                    "bullets": [
                        "Dynamic block routing solves the long-document scalability challenge",
                        "Maintains full accuracy while reducing memory by 62%",
                        "Future work: Extension to generative autoregressive language models",
                        "Future work: Hardware-aware adaptive block sizes for edge devices"
                    ],
                    "speaker_note": "Conclude by summarizing the main contribution and outlining exciting future directions."
                }
            ]
            payload = {
                "paper_title": "SparseDocNet: Efficient Document Representation with Dynamic Sparse Attention",
                "slides": slides
            }
            return AIMessage(content=json.dumps(payload))

        # 5. Viva Agent
        if "technical viva panel" in lower_prompt or "viva" in lower_prompt:
            categories = [
                ("conceptual", "easy", "What is the core problem that SparseDocNet aims to solve?", "SparseDocNet addresses the quadratic computational O(n^2) and GPU memory bottleneck inherent in standard dense Transformer self-attention when processing long documents exceeding standard sequence limits [Page 1].", [1]),
                ("conceptual", "medium", "Explain why fixed sparse attention patterns are inadequate for multi-page documents.", "Fixed patterns like local sliding windows or fixed stride patterns assume spatial locality. In complex multi-page documents, related information often appears in distant sections (e.g., introduction and conclusion), which fixed patterns miss entirely [Page 1, Page 2].", [1, 2]),
                ("conceptual", "hard", "How does Adaptive Dynamic Routing preserve theoretical attention mass?", "By computing cosine similarity between block centroids and selecting top-k clusters, ADR dynamically directs attention to clusters that contain 98.7% of the total attention weight, effectively approximating full dense attention with bounded loss [Page 2, Page 3].", [2, 3]),
                ("conceptual", "medium", "What constitutes a block centroid in the SparseDocNet architecture?", "A block centroid is calculated by applying mean pooling across the token embeddings within each non-overlapping contiguous block of size B=64 [Page 2].", [2]),

                ("methodological", "easy", "What block size B is utilized in SparseDocNet and why?", "SparseDocNet uses a block size of B=64 because it aligns with GPU tensor core memory alignment and warp execution sizes, minimizing thread divergence [Page 2].", [2]),
                ("methodological", "medium", "Describe the step-by-step process of Adaptive Dynamic Routing.", "First, input tokens are partitioned into blocks of 64. Second, mean-pooled centroids are created. Third, the routing network computes pairwise centroid cosine similarities. Fourth, top-k blocks are selected for standard attention while others are pruned [Page 2].", [2]),
                ("methodological", "hard", "How does the training objective balance routing accuracy and classification loss?", "SparseDocNet uses a multi-task composite loss combining cross-entropy classification loss with a routing consistency regularization term that penalizes routing decisions deviating from dense attention mass [Page 2, Page 3].", [2, 3]),
                ("methodological", "medium", "What custom software optimization was developed to realize practical wall-clock speedups?", "The authors implemented custom Triton CUDA kernels for block-sparse matrix multiplication, avoiding memory allocation for zeroed-out blocks [Page 3].", [3]),

                ("result-based", "easy", "What inference throughput speedup did SparseDocNet achieve over BERT?", "SparseDocNet achieved a 3.8x inference speedup over standard BERT-base on 8,192-token sequences [Page 3].", [3]),
                ("result-based", "medium", "What accuracy did SparseDocNet achieve on the RVL-CDIP benchmark?", "SparseDocNet achieved 94.2% document classification accuracy on RVL-CDIP, matching the dense Transformer baseline within 0.1% [Page 3].", [3]),
                ("result-based", "hard", "How did SparseDocNet compare to Longformer and BigBird on the Contract-8K benchmark?", "On Contract-8K, SparseDocNet outperformed Longformer by 1.4% and BigBird by 1.1% in accuracy, while using 24% less peak GPU memory due to its content-adaptive routing [Page 3].", [3]),
                ("result-based", "medium", "What was the reported reduction in peak GPU memory usage?", "Peak GPU memory usage decreased by 62%, enabling the processing of 8,192 token sequences on a single 16GB GPU [Page 3, Page 4].", [3, 4]),

                ("critical", "easy", "Under what conditions does SparseDocNet underperform or incur overhead?", "On short sequences under 512 tokens, the overhead of computing block centroids and the routing matrix incurs a 4-7% latency penalty compared to standard dense attention [Page 4].", [4]),
                ("critical", "medium", "What are the primary hyperparameter sensitivities in SparseDocNet?", "The model is sensitive to the choice of block size B and the top-k cluster selection threshold; setting k too low degrades accuracy while setting it too high reduces speedup [Page 3, Page 4].", [3, 4]),
                ("critical", "hard", "How might the centroid mean-pooling step fail in heterogeneous document pages?", "If a single block contains both dense tabular data and sparse text, mean pooling can dilute distinct feature vectors, leading to suboptimal routing for outlier tokens [Page 4].", [4]),
                ("critical", "hard", "What are the limitations regarding multi-lingual or non-standard document formats?", "SparseDocNet was evaluated primarily on English documents; its tokenization and routing patterns have not been validated on ideographic or right-to-left scripts [Page 4].", [4]),

                ("comparison", "easy", "How does SparseDocNet differ from standard sliding-window attention?", "Sliding-window attention only attends to adjacent tokens regardless of relevance, whereas SparseDocNet dynamically selects distant blocks based on learned content similarity [Page 1, Page 2].", [1, 2]),
                ("comparison", "medium", "Compare SparseDocNet with Reformer's locality-sensitive hashing.", "While Reformer uses LSH which involves high constant factor overhead and multi-round hashing, SparseDocNet uses block-level routing with GPU-friendly dense block operations, providing faster real-world wall-clock time [Page 2, Page 3].", [2, 3]),
                ("comparison", "hard", "Why is SparseDocNet more memory efficient than Routing Transformer?", "Routing Transformer clusters individual tokens via k-means at every step, creating high memory fragmentation; SparseDocNet clusters contiguous blocks, enabling coalesced GPU memory reads [Page 2, Page 3].", [2, 3]),
                ("comparison", "medium", "How does SparseDocNet's accuracy compare to full dense attention baselines?", "SparseDocNet achieves 94.2% accuracy versus 94.3% for full dense attention, retaining 99.9% of the performance while cutting memory by 62% [Page 3, Page 4].", [3, 4])
            ]

            qa_pairs = []
            for cat, diff, q, a, pages in categories:
                qa_pairs.append({
                    "question": q,
                    "model_answer": a,
                    "category": cat,
                    "difficulty": diff,
                    "source_pages": list(pages)
                })

            payload = {
                "paper_title": "SparseDocNet: Efficient Document Representation with Dynamic Sparse Attention",
                "qa_pairs": qa_pairs
            }
            return AIMessage(content=json.dumps(payload))

        # Default fallback
        return AIMessage(content="{}")

    return RunnableLambda(mock_llm_fn)


def main():
    print("=" * 80)
    print("PHASE 5 DEMONSTRATION: THE FOUR SPECIALIZED AGENTS")
    print("=" * 80)

    # Step 1: Ingest sample paper
    pdf_path = PROJECT_ROOT / "tests" / "data" / "sample_research_paper.pdf"
    if not pdf_path.exists():
        print(f"Error: Sample paper not found at {pdf_path}")
        sys.exit(1)

    print(f"\n[1] Processing PDF: {pdf_path.name}")
    proc_result, chunks = process_pdf(pdf_path)
    doc_id = proc_result.doc_id
    print(f"    - SHA-256 doc_id : {doc_id}")
    print(f"    - Pages extracted : {proc_result.page_count}")
    print(f"    - Chunks created  : {len(chunks)}")

    if not is_document_ingested(doc_id):
        print(f"    - Ingesting {len(chunks)} chunks into ChromaDB...")
        ingest_documents(chunks, doc_id)
        print("    - Ingestion complete.")
    else:
        print("    - Document already present in ChromaDB collection.")

    # Step 2: Resolve LLM
    if _is_live_key_configured():
        print("\n[2] Using live Google Gemini LLM instance.")
        active_llm = get_llm()
    else:
        print("\n[2] Using grounded offline mock LLM (validates all schemas and pipelines).")
        active_llm = _create_grounded_mock_llm()

    # Step 3: Run Summarizer Agent
    print("\n" + "=" * 80)
    print("AGENT 1: SUMMARIZER AGENT")
    print("=" * 80)
    summary_output = summarizer.run(doc_id=doc_id, llm=active_llm)
    print(f"Title: {summary_output.title}")
    print(f"\nAbstract Summary:\n{summary_output.abstract_summary}")
    print(f"\nProblem Statement:\n{summary_output.problem_statement}")
    print(f"\nProposed Solution:\n{summary_output.proposed_solution}")
    print("\nKey Findings:")
    for i, finding in enumerate(summary_output.key_findings, start=1):
        print(f"  {i}. {finding}")
    print(f"\nConclusion:\n{summary_output.conclusion}")
    print("\nGrounding Sources:")
    for s in summary_output.sources:
        print(f"  - Chunk {s.chunk_index} | Page {s.page_number} ({s.source})")

    # Step 4: Run Concept-Extractor Agent
    print("\n" + "=" * 80)
    print("AGENT 2: CONCEPT-EXTRACTOR AGENT")
    print("=" * 80)
    concepts_output = concept_extractor.run(doc_id=doc_id, llm=active_llm)
    print("Key Contributions:")
    for c in concepts_output.key_contributions:
        print(f"  * {c}")
    print("\nMethodology Highlights:")
    for m in concepts_output.methodology:
        print(f"  * {m}")
    print("\nDatasets Used:")
    for d in concepts_output.datasets_used:
        print(f"  * {d}")
    print("\nResults and Quantitative Metrics:")
    for r in concepts_output.results_and_metrics:
        print(f"  * {r}")
    print("\nLimitations:")
    for lim in concepts_output.limitations:
        print(f"  * {lim}")
    print("\nFuture Work:")
    for f in concepts_output.future_work:
        print(f"  * {f}")
    print("\nGrounding Sources:")
    for s in concepts_output.sources:
        print(f"  - Chunk {s.chunk_index} | Page {s.page_number} ({s.source})")

    # Step 5: Run Slide-Generator Agent
    print("\n" + "=" * 80)
    print("AGENT 3: SLIDE-GENERATOR AGENT")
    print("=" * 80)
    slides_output = slide_generator.run(doc_id=doc_id, llm=active_llm)
    print(f"Presentation Title: {slides_output.paper_title}")
    print(f"Total Slides Generated: {len(slides_output.slides)}")
    print("-" * 60)
    for slide in slides_output.slides:
        print(f"Slide {slide.slide_number}: {slide.title}")
        for bullet in slide.bullets:
            print(f"   - {bullet}")
        if slide.speaker_note:
            print(f"   [Speaker Note]: {slide.speaker_note}")
        print()
    print("Grounding Sources:")
    for s in slides_output.sources:
        print(f"  - Chunk {s.chunk_index} | Page {s.page_number} ({s.source})")

    # Step 6: Run Viva Agent
    print("\n" + "=" * 80)
    print("AGENT 4: VIVA AGENT (ORAL DEFENSE PREPARATION)")
    print("=" * 80)
    viva_output = viva_agent.run(doc_id=doc_id, count=20, llm=active_llm)
    print(f"Paper: {viva_output.paper_title}")
    print(f"Total Viva Questions: {len(viva_output.qa_pairs)}")
    print("-" * 60)
    for i, qa in enumerate(viva_output.qa_pairs, start=1):
        pages_str = ", ".join(f"p.{p}" for p in qa.source_pages) if qa.source_pages else "N/A"
        print(f"Q{i} [{qa.category.upper()} | {qa.difficulty.upper()} | Sources: {pages_str}]:")
        print(f"  Question: {qa.question}")
        print(f"  Model Answer: {qa.model_answer}\n")

    # Step 7: Router Agent Dispatch Demo
    print("\n" + "=" * 80)
    print("ROUTER AGENT: INTENT CLASSIFICATION & DISPATCH")
    print("=" * 80)
    sample_queries = [
        "Please summarize the main ideas of this research paper.",
        "What methodology and datasets did the authors use?",
        "Create a 10-slide presentation deck for my seminar.",
        "Generate viva defense questions with model answers.",
    ]
    for q in sample_queries:
        intent_res = router.detect_intent(q, active_llm)
        print(f"Query: '{q}'")
        print(f"  -> Detected Intent: {intent_res.intent} (confidence: {intent_res.confidence})")
        dispatched_res = router.dispatch(intent_res, doc_id=doc_id, llm=active_llm)
        print(f"  -> Dispatched successfully to {type(dispatched_res).__name__}")
        print()

    print("=" * 80)
    print("ALL 4 AGENTS EXECUTED SUCCESSFULLY AND GROUNDED IN CITATIONS.")
    print("=" * 80)


if __name__ == "__main__":
    main()
