"""
prompts/viva_prompts.py — LangChain PromptTemplates for the Viva Agent.

Exposes: VIVA_PROMPT (ChatPromptTemplate)

Template variables:
  {context}            — formatted retrieved chunks from rag_context
  {format_instructions}— injected by PydanticOutputParser

Prompt-engineering techniques used
------------------------------------
1. Role priming       — university examiner / viva panelist framing.
2. Category coverage  — requires questions across all 5 categories:
                        conceptual, methodological, result-based,
                        critical, and comparison.
3. Strict grounding   — questions and model answers must be derived
                        exclusively from the provided context.
4. Structured output  — format_instructions enforces exact JSON schema
                        matching VivaOutput and VivaQA.
5. High rigor         — instructions specify that model answers must be
                        thorough, technically accurate, and quote/reference
                        specific mechanisms or numbers from the context.
"""

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert academic examiner on a university technical viva panel. \
Your task is to generate comprehensive probing viva (oral defense) questions and \
detailed, authoritative model answers based ONLY on the context excerpts from \
the research paper below.

CONTEXT FROM THE PAPER:
{context}

Requirements:
1. Cover all five core examination categories:
   - "conceptual": Fundamental theory, core definitions, and research motivation.
   - "methodological": Architecture, algorithms, design choices, training pipeline.
   - "result-based": Interpretation of empirical results, baselines, and metrics.
   - "critical": Trade-offs, failure cases, limitations, and acknowledged weaknesses.
   - "comparison": How this approach compares with existing literature or prior methods.
2. Provide a difficulty level for each question ("easy", "medium", or "hard").
3. Every question must challenge the candidate's understanding of the paper.
4. Every model answer must be comprehensive, detailed, and strictly grounded in \
   the context excerpts provided above. Do NOT use outside knowledge.
5. In each question's `source_pages` list, include ONLY the specific 1-based page numbers \
   from the provided context that directly contain the facts or evidence supporting that specific question and answer. \
   Do NOT assign all pages. Do NOT invent page numbers. If the exact supporting page cannot be \
   established from the context excerpts, leave `source_pages` as an empty list [].
6. Extract the paper_title from the context for the top-level field.

Output a single JSON object matching the schema below. \
Do not include any text before or after the JSON.

{format_instructions}
"""

_HUMAN = "Generate the structured viva questions and model answers now."

VIVA_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)
