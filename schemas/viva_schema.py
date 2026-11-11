"""
schemas/viva_schema.py — Pydantic models for Viva Agent output.

Validation rules
----------------
- category must be one of five allowed values (Literal enforces this).
- VivaOutput must contain 5–15 Q&A pairs.
- question and model_answer must be non-empty strings.
- sources carry the retrieved chunks that grounded the answers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from schemas.summary_schema import RetrievedSource

# Five core exam categories (supporting canonical and user-friendly aliases)
VivaCategory = Literal[
    "conceptual",
    "methodological",
    "result-based",
    "critical",
    "comparison",
    "fundamentals",
    "technical concepts",
    "methodology",
    "results and evaluation",
    "critical thinking",
]


class VivaQA(BaseModel):
    """A single viva question paired with a detailed, paper-grounded answer."""

    question: str = Field(..., description="The exam question.")
    model_answer: str = Field(
        ..., description="Detailed answer grounded in the paper content."
    )
    category: VivaCategory = Field(
        ...,
        description=(
            "Question category: conceptual | methodological | "
            "result-based | critical | comparison."
        ),
    )
    difficulty: Literal["easy", "medium", "hard"] | str = Field(
        default="medium",
        description="Question difficulty level: easy | medium | hard.",
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers from which the answer was derived.",
    )

    @field_validator("question", "model_answer")
    @classmethod
    def not_empty(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be empty.")
        return v.strip()

    model_config = {"frozen": False, "protected_namespaces": ()}


class VivaOutput(BaseModel):
    """Full set of viva Q&A pairs produced by the Viva Agent."""

    paper_title: str = Field(..., description="Title of the source research paper.")
    qa_pairs: list[VivaQA] = Field(
        ..., description="Ordered list of question-answer pairs."
    )
    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Retrieved chunks used to generate these questions.",
    )

    @field_validator("qa_pairs")
    @classmethod
    def qa_count(cls, v: list[VivaQA]) -> list[VivaQA]:
        # Allows standard 5-15 batches as well as comprehensive 20-30 college viva sets
        if not (5 <= len(v) <= 15 or 20 <= len(v) <= 30):
            raise ValueError(
                f"VivaOutput must contain 5–15 Q&A pairs (or 20–30 in comprehensive viva mode); got {len(v)}."
            )
        return v

    model_config = {"frozen": False, "protected_namespaces": ()}
