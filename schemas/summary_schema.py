"""
schemas/summary_schema.py — Pydantic model for Summarizer Agent output.

Source attribution
------------------
`sources` carries the page numbers and chunk IDs of the retrieved chunks
that grounded the LLM response.  This list is populated by the agent
(not by the LLM) using the retriever's returned Documents, so citations
are never fabricated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetrievedSource(BaseModel):
    """Traceability record for a single retrieved chunk."""

    page_number: int = Field(..., ge=1, description="1-based page number in the PDF.")
    chunk_index: int = Field(..., ge=0, description="Global chunk index assigned at split time.")
    source: str = Field(..., description="Original PDF filename.")

    model_config = {"frozen": True}


class SummaryOutput(BaseModel):
    """
    Structured summary produced by the Summarizer Agent.

    Validation rules
    ----------------
    - title must not be empty.
    - key_findings must contain between 2 and 8 entries.
    - sources must be non-empty (at least one retrieved chunk).
    """

    title: str = Field(..., description="Paper title as found in the document.")
    abstract_summary: str = Field(
        ..., description="2–4 sentence plain-language summary of the paper."
    )
    problem_statement: str = Field(
        ..., description="The core research problem the paper addresses."
    )
    proposed_solution: str = Field(
        ..., description="The method or approach proposed by the authors."
    )
    key_findings: list[str] = Field(
        ..., description="3–6 key findings or contributions."
    )
    conclusion: str = Field(
        ..., description="Concluding takeaway from the paper."
    )
    sources: list[RetrievedSource] = Field(
        default_factory=list,
        description="Retrieved chunk metadata used to ground this summary.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty.")
        return v.strip()

    @field_validator("key_findings")
    @classmethod
    def findings_count(cls, v: list[str]) -> list[str]:
        if not (2 <= len(v) <= 8):
            raise ValueError(
                f"key_findings must have 2–8 entries; got {len(v)}."
            )
        return v

    model_config = {"frozen": False}
