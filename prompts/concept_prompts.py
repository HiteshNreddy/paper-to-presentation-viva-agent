"""
prompts/concept_prompts.py — LangChain PromptTemplate for the Concept-Extractor Agent.

Exposes: CONCEPT_PROMPT (ChatPromptTemplate)

Template variables:
  {context}            — formatted retrieved chunks from rag_context
  {format_instructions}— injected by PydanticOutputParser

Prompt-engineering techniques used
------------------------------------
1. Role priming       — domain expert framing anchors the extraction task.
2. Strict grounding   — LLM must extract, not infer or extend.
3. Fallback instruction — if a field is not mentioned in the paper, the LLM
                          must write ["Not explicitly mentioned"] rather than
                          hallucinating content.
4. Structured output  — format_instructions enforces exact JSON schema.
5. Per-field examples — brief inline examples show the expected granularity
                        (e.g. "95.3% accuracy on CIFAR-10" for metrics).
"""

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are a senior machine learning researcher performing a detailed technical \
analysis of a research paper. Extract ONLY information that is explicitly \
stated in the context excerpts below. Do not infer, extrapolate, or add \
knowledge from outside the provided text.

CONTEXT FROM THE PAPER:
{context}

Extract the following six categories of information:

1. key_contributions — The novel ideas, claims, or technical contributions \
   the authors make (e.g. "Introduces a sparse attention mechanism that \
   reduces complexity from O(n²) to O(n log n)").
2. methodology — Step-by-step description of the proposed method or system \
   architecture (e.g. "1. Tokenise input, 2. Apply sparse attention, ...").
3. datasets_used — All datasets or benchmarks mentioned (e.g. "ImageNet, \
   COCO, MS-COCO val2017"). Write ["Not explicitly mentioned"] if absent.
4. results_and_metrics — Quantitative results with numbers where available \
   (e.g. "92.1% top-1 accuracy on ImageNet"). Write ["Not explicitly \
   mentioned"] if absent.
5. limitations — Weaknesses, constraints, or failure modes the authors \
   acknowledge.
6. future_work — Research directions or improvements the authors suggest. \
   Write ["Not explicitly mentioned"] if absent.

Output a single JSON object matching the schema below. \
No text before or after the JSON.

{format_instructions}
"""

_HUMAN = "Produce the structured concept extraction now."

CONCEPT_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)
