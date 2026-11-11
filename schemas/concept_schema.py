"""
schemas/concept_schema.py — Pydantic model for Concept-Extractor Agent output.

Source attribution
------------------
`sources` lists the retrieved chunks that grounded each section.
Agents populate this after retrieval; the LLM never invents page numbers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from schemas.summary_schema import RetrievedSource   # reuse the same record type


class ConceptOutput(BaseModel):
    """
    Structured concept breakdown produced by the Concept-Extractor Agent.

    Validation rules
    ----------------
    - Every list field must contain at least 1 entry (empty = no extraction).
    - datasets_used and future_work may be ["Not mentioned"] if absent in paper.
    """

    key_contributions: list[str] = Field(
        ..., description="Novel claims or contributions of the paper."
    )
    methodology: list[str] = Field(
        ..., description="Step-by-step description of the proposed method."
    )
    datasets_used: list[str] = Field(
        ..., description="Datasets or benchmarks used for evaluation."
    )
    results_and_metrics: list[str] = Field(
        ..., description="Quantitative results (e.g. accuracy, F1, BLEU)."
    )
    limitations: list[str] = Field(
        ..., description="Limitations acknowledged by the authors."
    )
    future_work: list[str] = Field(
        ..., description="Future research directions suggested by the authors."
    )
    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Retrieved chunk metadata used to ground this extraction.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator(
        "key_contributions", "methodology", "results_and_metrics", "limitations"
    )
    @classmethod
    def at_least_one(cls, v: list[str], info) -> list[str]:
        if not v:
            raise ValueError(f"{info.field_name} must contain at least one entry.")
        return v

    model_config = {"frozen": False}
