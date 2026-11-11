"""
agents/router.py — Intent detection and agent dispatcher.

Public API
----------
detect_intent(user_input, llm) -> IntentResult
    Classifies a free-text string into one of four intents using a
    lightweight LCEL classification chain.

dispatch(intent, retrieval_result=None, doc_id=None, llm=None) -> AgentOutput
    Calls the appropriate agent function and returns its output.

Design decisions
----------------
- Intent detection uses a small ChatPromptTemplate + Pydantic output parser,
  NOT a full tool-calling agent loop. This keeps latency low (<1 s) and
  makes the classification step transparent and testable.
- The router receives a pre-computed RetrievalResult or a doc_id to pass down.
- Unrecognised intents return IntentResult(intent="unknown") so the UI can
  ask the user to clarify rather than crashing.
- The four valid intents map 1-to-1 to the four agents:
    "summarize"        → agents.summarizer.run()
    "extract_concepts" → agents.concept_extractor.run()
    "generate_slides"  → agents.slide_generator.run()
    "viva"             → agents.viva_agent.run()
"""

from __future__ import annotations

from typing import Any, Literal, Union

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import agents.concept_extractor as concept_extractor
import agents.slide_generator as slide_generator
import agents.summarizer as summarizer
import agents.viva_agent as viva_agent
from core.rag_context import RetrievalResult
from schemas.concept_schema import ConceptOutput
from schemas.slide_schema import SlideDeck
from schemas.summary_schema import SummaryOutput
from schemas.viva_schema import VivaOutput

AgentOutput = Union[SummaryOutput, ConceptOutput, SlideDeck, VivaOutput]

# ── Schema for the intent classification response ─────────────────────────────

IntentLabel = Literal["summarize", "extract_concepts", "generate_slides", "viva", "unknown"]


class IntentResult(BaseModel):
    """Structured output of the intent-detection chain."""

    intent: IntentLabel = Field(
        ...,
        description=(
            "Detected intent. Must be exactly one of: "
            "summarize | extract_concepts | generate_slides | viva | unknown"
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence (0–1).",
    )


# ── Prompt template for intent classification ─────────────────────────────────

_INTENT_SYSTEM = """\
You are an intent classifier for a research-paper AI assistant.
Given the user's request, output ONLY valid JSON matching the schema below.
Do not explain. Do not add extra keys.

Valid intents:
- "summarize"        : user wants a summary of the paper
- "extract_concepts" : user wants key concepts, methodology, results, or contributions
- "generate_slides"  : user wants a PowerPoint / slide deck
- "viva"             : user wants viva questions, exam prep, or Q&A
- "unknown"          : request does not match any of the above

{format_instructions}
"""

_INTENT_HUMAN = "User request: {user_input}"


def _build_intent_chain(llm):
    """Build and return the LCEL intent-classification chain."""
    parser = PydanticOutputParser(pydantic_object=IntentResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _INTENT_SYSTEM),
            ("human", _INTENT_HUMAN),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def detect_intent(user_input: str, llm) -> IntentResult:
    """
    Classify a free-text user request into one of four agent intents.

    Parameters
    ----------
    user_input : str
        The raw text typed by the user (e.g. "give me viva questions").
    llm : ChatGoogleGenerativeAI
        The shared LLM instance from core.llm.get_llm().

    Returns
    -------
    IntentResult
        ``intent`` is one of the five labels; ``confidence`` is advisory.
    """
    if not user_input.strip():
        return IntentResult(intent="unknown", confidence=0.0)

    chain = _build_intent_chain(llm)
    try:
        result: IntentResult = chain.invoke({"user_input": user_input})
    except Exception:
        # Parsing failure → treat as unknown rather than propagating
        result = IntentResult(intent="unknown", confidence=0.0)

    return result


def dispatch(
    intent: IntentLabel | IntentResult | str,
    retrieval_result: RetrievalResult | None = None,
    doc_id: str | None = None,
    llm: Any | None = None,
) -> AgentOutput:
    """
    Dispatch an intent to the corresponding specialized agent.

    Parameters
    ----------
    intent : IntentLabel | IntentResult | str
        Target intent ("summarize", "extract_concepts", "generate_slides", "viva").
    retrieval_result : RetrievalResult | None
        Pre-computed retrieval result to pass to the agent.
    doc_id : str | None
        Active doc_id for on-demand retrieval if retrieval_result is None.
    llm : Any | None
        Optional LLM instance override.

    Returns
    -------
    AgentOutput
        Output model from the dispatched agent.

    Raises
    ------
    ValueError
        If the intent is unrecognized or unsupported.
    """
    label = intent.intent if isinstance(intent, IntentResult) else str(intent).strip().lower()

    if label == "summarize":
        return summarizer.run(retrieval_result=retrieval_result, doc_id=doc_id, llm=llm)
    elif label == "extract_concepts":
        return concept_extractor.run(retrieval_result=retrieval_result, doc_id=doc_id, llm=llm)
    elif label == "generate_slides":
        return slide_generator.run(retrieval_result=retrieval_result, doc_id=doc_id, llm=llm)
    elif label == "viva":
        return viva_agent.run(retrieval_result=retrieval_result, doc_id=doc_id, llm=llm)
    else:
        raise ValueError(
            f"Cannot dispatch unrecognized or unsupported intent: '{label}'. "
            f"Valid intents are: summarize, extract_concepts, generate_slides, viva."
        )
