"""
prompts/summarizer_prompts.py — LangChain PromptTemplate for the Summarizer Agent.

Exposes: SUMMARIZER_PROMPT (ChatPromptTemplate)

Template variables expected by the LCEL chain in agents/summarizer.py:
  {context}            — formatted retrieved chunks from rag_context
  {format_instructions}— injected by PydanticOutputParser at chain build time

Prompt-engineering techniques used
------------------------------------
1. Role priming   — the system turn establishes the LLM as a research
                    analyst, anchoring its register and vocabulary.
2. Explicit scope — the LLM is told to work ONLY from the provided context,
                    preventing it from supplementing with outside knowledge.
3. Chain-of-thought hint — "Think step by step" before the JSON output
                    improves structured-output accuracy.
4. Output schema  — the format_instructions block (injected by the parser)
                    provides the exact JSON skeleton, field names, and types.
5. Grounding rule — explicit instruction not to invent information absent
                    from the context, reducing hallucination.
"""

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are an expert research analyst. Your task is to produce a structured \
summary of an academic research paper based ONLY on the context excerpts \
provided below. Do not use any external knowledge or information not present \
in the context.

CONTEXT FROM THE PAPER:
{context}

Think step by step:
1. Identify the paper title from the context.
2. Summarise the core problem the paper addresses.
3. Describe the proposed solution or method.
4. List the key findings and contributions.
5. State the conclusion.

Then output your answer as a single JSON object matching the schema below. \
Do not include any text before or after the JSON.

{format_instructions}
"""

_HUMAN = "Produce the structured summary now."

SUMMARIZER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ]
)
