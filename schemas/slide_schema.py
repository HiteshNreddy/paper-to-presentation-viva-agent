"""
schemas/slide_schema.py — Pydantic models for Slide-Generator Agent output.

Validation rules
----------------
- Each Slide must have 2–6 bullets (not too sparse, not a wall of text).
- A SlideDeck must have 5–15 slides (a reasonable college presentation range).
- slide_number must be sequential (validated at deck level).
- sources carry the retrieved chunks that grounded the slide content.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas.summary_schema import RetrievedSource


class Slide(BaseModel):
    """A single presentation slide."""

    slide_number: int = Field(..., ge=1, description="1-based slide index.")
    title: str = Field(..., description="Slide title.")
    bullets: list[str] = Field(
        ..., description="Concise bullet points for this slide."
    )
    speaker_note: str | None = Field(
        default=None,
        description="Optional presenter note for this slide.",
    )

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Slide title must not be empty.")
        return v.strip()

    @field_validator("bullets")
    @classmethod
    def bullet_count(cls, v: list[str]) -> list[str]:
        if not (2 <= len(v) <= 6):
            raise ValueError(
                f"Each slide must have 2–6 bullets; got {len(v)}."
            )
        return v

    model_config = {"frozen": False}


class SlideDeck(BaseModel):
    """An ordered collection of slides forming a full presentation."""

    paper_title: str = Field(..., description="Title of the source research paper.")
    slides: list[Slide] = Field(..., description="Ordered list of slides.")
    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Retrieved chunks that grounded the slide content.",
    )

    @field_validator("slides")
    @classmethod
    def slide_count(cls, v: list[Slide]) -> list[Slide]:
        if not (5 <= len(v) <= 15):
            raise ValueError(
                f"A presentation must have 5–15 slides; got {len(v)}."
            )
        return v

    @model_validator(mode="after")
    def sequential_slide_numbers(self) -> "SlideDeck":
        for i, slide in enumerate(self.slides, start=1):
            if slide.slide_number != i:
                raise ValueError(
                    f"Slide numbers must be sequential starting at 1; "
                    f"slide at position {i} has slide_number={slide.slide_number}."
                )
        return self

    model_config = {"frozen": False}
