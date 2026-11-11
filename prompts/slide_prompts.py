"""
prompts/slide_prompts.py — LangChain PromptTemplate for the Slide-Generator Agent.

Exposes: SLIDE_PROMPT (ChatPromptTemplate)

Template variables:
  {context}            — formatted retrieved chunks from rag_context
  {format_instructions}— injected by PydanticOutputParser

Prompt-engineering techniques used
------------------------------------
1. Role priming       — presentation designer framing.
2. Slide-count constraint — the LLM is told 8–10 slides; this guides it
                            within the Pydantic 5–15 validation range.
3. Per-slide structure — explicit instruction for title + 3–4 bullets per
                         slide, within the Pydantic 2–6 bullet constraint.
4. Canonical slide order — the LLM is given a recommended slide sequence to
                           ensure a coherent college-level presentation flow.
5. No fabrication rule — bullets must come from the context only.
6. Speaker notes — the LLM is encouraged (not required) to add concise
                   presenter notes to aid delivery.
"""

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert academic presentation designer. Your task is to create a \
clear, well-structured slide deck for a college-level seminar based ONLY \
on the context excerpts from the research paper below.

CONTEXT FROM THE PAPER:
{context}

Design a presentation with 10–12 slides following this recommended structure:
  Slide 1: Title Slide — paper title, authors (if mentioned), and a concise subtitle
  Slide 2: Introduction & Research Motivation
  Slide 3: Problem Statement & Research Challenges
  Slide 4: Research Objectives & Scope
  Slide 5: Background & Related Work
  Slide 6: Proposed Architecture & High-Level Design
  Slide 7: Methodology & Technical Algorithms
  Slide 8: System Implementation Details
  Slide 9: Experimental Setup & Datasets
  Slide 10: Results & Quantitative Evaluation
  Slide 11: Limitations & Critical Discussion
  Slide 12: Conclusion & Future Scope

Rules:
- Each slide must have a clear, descriptive title.
- Each slide must have exactly 3–4 concise bullet points (max 15 words each).
- Bullets must be strictly grounded in the context — do not fabricate content.
- Assign slide_number starting from 1, incrementing by 1 sequentially.
- Add an insightful speaker_note (1–2 sentences) for each slide.
- Extract the paper_title from the context for the top-level field.

Output a single JSON object matching the schema below. \
No text before or after the JSON.

{format_instructions}
"""

_HUMAN = "Produce the structured slide deck now."

SLIDE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)
