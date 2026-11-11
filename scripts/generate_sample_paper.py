"""
scripts/generate_sample_paper.py — Creates a realistic 4-page academic research paper PDF.

Uses PyMuPDF (fitz) to create an actual research paper PDF with standard academic
sections (Abstract, Introduction, Methodology, Experiments, Results, Limitations, Conclusion).
Saved to: tests/data/sample_research_paper.pdf
"""

import fitz
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = DATA_DIR / "sample_research_paper.pdf"

PAGES_CONTENT = [
    # Page 1
    (
        "Efficient Deep Attention Networks for Sparse Document Modeling\n"
        "Authors: Alex Thorne, Elena Rostova, Marcus Vance\n"
        "Department of Computer Science, AI Research Institute\n\n"
        "Abstract\n"
        "Modern transformer architectures suffer from quadratic time and memory complexity with respect "
        "to sequence length. In this paper, we introduce SparseDocNet, an efficient self-attention architecture "
        "designed for long-form academic document modeling. By combining dynamic block-sparse routing with "
        "hierarchical memory compression, SparseDocNet reduces attention complexity from O(N^2) to O(N log N). "
        "We evaluate our framework on three standard benchmarks and achieve a 94.2% F1 score while reducing "
        "inference latency by 3.4x compared to full-attention baselines.\n\n"
        "1. Introduction\n"
        "The quadratic scaling of standard multi-head self-attention poses a fundamental bottleneck when "
        "processing lengthy documents such as academic research papers, legal briefs, and technical manuals. "
        "Prior works have explored fixed stride windows or low-rank kernel approximations. However, these "
        "methods often sacrifice long-range semantic dependencies.\n\n"
        "Problem Statement: How can an attention mechanism preserve global semantic coherence across multi-page "
        "documents while maintaining sub-quadratic computational complexity? This paper addresses this challenge "
        "by introducing a dynamic routing mechanism that selectively attends only to top-k informative token clusters."
    ),
    # Page 2
    (
        "2. Related Work\n"
        "Early sparse transformers such as Longformer and BigBird utilized combinations of sliding window attention, "
        "dilated patterns, and dedicated global tokens. While effective for moderate contexts, the fixed window size "
        "fails to adapt to document structures with uneven information density.\n\n"
        "3. Proposed Methodology: SparseDocNet\n"
        "The core innovation of SparseDocNet is the Adaptive Dynamic Routing (ADR) layer. The architecture consists of:\n"
        "1. Hierarchical Token Clustering: Consecutive tokens are grouped into semantic blocks of size B=64.\n"
        "2. Inter-block Routing Matrix: A lightweight feed-forward gating network estimates relevance scores between all block pairs.\n"
        "3. Top-k Sparse Attention: For each block, attention is restricted to the k most relevant blocks, where k << total blocks.\n"
        "4. Compressed Global Memory: A residual summary vector maintains document-wide context across all layers.\n\n"
        "Mathematical Formulation:\n"
        "Given query block Q_i and key block K_j, the routing score R_ij is computed as Softmax(W_r * [Q_i; K_j] / sqrt(d)). "
        "Only block pairs with R_ij exceeding a dynamic threshold tau participate in the scaled dot-product attention calculation."
    ),
    # Page 3
    (
        "4. Experimental Setup & Datasets\n"
        "We conduct empirical evaluations on three challenging benchmarks:\n"
        "- arXiv-LongDoc: 50,000 multi-page scientific papers across physics and computer science.\n"
        "- PubMed-MultiDoc: 35,000 biomedical papers requiring deep domain reasoning.\n"
        "- GovReport-Long: 19,000 government research reports with average sequence length of 9,400 tokens.\n\n"
        "Hardware Setup: All models were trained on 8x NVIDIA A100 GPUs (80GB) using PyTorch with mixed precision (bfloat16). "
        "AdamW optimizer was configured with learning rate 1e-4, linear warmup for 2,000 steps, and cosine decay.\n\n"
        "5. Results & Quantitative Evaluation\n"
        "Across all datasets, SparseDocNet consistently outperforms competitive baselines:\n"
        "- arXiv-LongDoc: Achieves 94.2% F1 score (vs. 91.8% for Longformer and 92.4% for BigBird).\n"
        "- PubMed-MultiDoc: 93.6% accuracy with a 3.4x wall-clock inference speedup.\n"
        "- Memory Footprint: Peak GPU memory utilization dropped by 62% on sequence lengths exceeding 8,000 tokens.\n"
        "- Ablation Study: Disabling the Compressed Global Memory leads to a 4.1% drop in F1 score on multi-hop queries, "
        "confirming its essential role in long-range coherence."
    ),
    # Page 4
    (
        "6. Limitations & Discussion\n"
        "Despite compelling results, SparseDocNet has three notable limitations:\n"
        "1. Short-sequence overhead: For documents under 512 tokens, the clustering and routing network introduces "
        "unnecessary latency overhead of approximately 8% compared to standard dense attention.\n"
        "2. Hyperparameter sensitivity: The dynamic threshold tau requires dataset-specific tuning; overly aggressive "
        "filtering can omit crucial transition sentences.\n"
        "3. High GPU memory requirements during training: While inference is lightweight, the clustering step during "
        "backpropagation requires substantial intermediate buffer memory.\n\n"
        "7. Future Work & Conclusion\n"
        "In this work, we presented SparseDocNet, an efficient attention architecture tailored for long document analysis. "
        "Future directions include extending dynamic routing to multi-modal document understanding (incorporating tables "
        "and figures) and developing self-tuning threshold algorithms for arbitrary document domains.\n\n"
        "References\n"
        "[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.\n"
        "[2] Beltagy et al. Longformer: The Long-Document Transformer. arXiv 2020.\n"
        "[3] Zaheer et al. Big Bird: Transformers for Longer Sequences. NeurIPS 2020."
    )
]


def generate_sample_paper() -> Path:
    doc = fitz.open()

    for page_num, text in enumerate(PAGES_CONTENT, start=1):
        # Create standard A4 page (595 x 842 points)
        page = doc.new_page(width=595, height=842)
        # Margin: left=50, top=60, right=545, bottom=780
        rect = fitz.Rect(50, 60, 545, 780)
        page.insert_textbox(
            rect,
            text,
            fontsize=10.5,
            fontname="helv",
            color=(0.1, 0.1, 0.1),
            align=fitz.TEXT_ALIGN_LEFT
        )

    doc.save(str(PDF_PATH))
    doc.close()
    print(f"Generated sample research paper at: {PDF_PATH} ({PDF_PATH.stat().st_size} bytes, 4 pages)")
    return PDF_PATH


if __name__ == "__main__":
    generate_sample_paper()

