"""
tests/test_schemas.py — Unit tests for all Pydantic output schemas.

Tests covered
-------------
SummaryOutput:
  - Valid construction.
  - Empty title raises ValidationError.
  - key_findings < 2 or > 8 raises ValidationError.

ConceptOutput:
  - Valid construction.
  - Empty key_contributions raises ValidationError.

SlideDeck:
  - Valid construction.
  - < 5 or > 15 slides raises ValidationError.
  - Non-sequential slide_number raises ValidationError.
  - bullets < 2 or > 6 raises ValidationError.

VivaOutput:
  - Valid construction.
  - < 5 or > 15 qa_pairs raises ValidationError.
  - Invalid category raises ValidationError.
  - Empty question raises ValidationError.

RetrievedSource:
  - page_number < 1 raises ValidationError.
  - chunk_index < 0 raises ValidationError.
"""

import pytest
from pydantic import ValidationError

from schemas.summary_schema import SummaryOutput, RetrievedSource
from schemas.concept_schema import ConceptOutput
from schemas.slide_schema import Slide, SlideDeck
from schemas.viva_schema import VivaQA, VivaOutput


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_source() -> RetrievedSource:
    return RetrievedSource(page_number=1, chunk_index=0, source="paper.pdf")


def _make_slide(n: int) -> Slide:
    return Slide(
        slide_number=n,
        title=f"Slide {n}",
        bullets=["Point A", "Point B", "Point C"],
    )


def _make_qa(category="conceptual") -> VivaQA:
    return VivaQA(
        question="What is the problem addressed?",
        model_answer="The paper addresses the travelling salesman problem.",
        category=category,
    )


# ── RetrievedSource ───────────────────────────────────────────────────────────

class TestRetrievedSource:
    def test_valid(self):
        src = _make_source()
        assert src.page_number == 1

    def test_page_number_less_than_one_raises(self):
        with pytest.raises(ValidationError):
            RetrievedSource(page_number=0, chunk_index=0, source="f.pdf")

    def test_chunk_index_negative_raises(self):
        with pytest.raises(ValidationError):
            RetrievedSource(page_number=1, chunk_index=-1, source="f.pdf")


# ── SummaryOutput ─────────────────────────────────────────────────────────────

class TestSummaryOutput:
    def _valid(self, **kwargs) -> dict:
        base = dict(
            title="A Novel Approach",
            abstract_summary="This paper does something novel.",
            problem_statement="TSP is NP-hard.",
            proposed_solution="Deep RL pointer network.",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            conclusion="The approach is effective.",
        )
        base.update(kwargs)
        return base

    def test_valid_construction(self):
        s = SummaryOutput(**self._valid())
        assert s.title == "A Novel Approach"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="title"):
            SummaryOutput(**self._valid(title="   "))

    def test_too_few_findings_raises(self):
        with pytest.raises(ValidationError, match="2–8"):
            SummaryOutput(**self._valid(key_findings=["Only one"]))

    def test_too_many_findings_raises(self):
        with pytest.raises(ValidationError, match="2–8"):
            SummaryOutput(**self._valid(key_findings=[f"F{i}" for i in range(9)]))

    def test_sources_default_empty(self):
        s = SummaryOutput(**self._valid())
        assert s.sources == []

    def test_sources_accepted(self):
        s = SummaryOutput(**self._valid(), sources=[_make_source()])
        assert len(s.sources) == 1


# ── ConceptOutput ─────────────────────────────────────────────────────────────

class TestConceptOutput:
    def _valid(self) -> dict:
        return dict(
            key_contributions=["Novel loss function"],
            methodology=["Step 1: preprocess data"],
            datasets_used=["CIFAR-10"],
            results_and_metrics=["95% accuracy"],
            limitations=["Only tested on small datasets"],
            future_work=["Extend to larger benchmarks"],
        )

    def test_valid_construction(self):
        c = ConceptOutput(**self._valid())
        assert c.key_contributions == ["Novel loss function"]

    def test_empty_contributions_raises(self):
        data = self._valid()
        data["key_contributions"] = []
        with pytest.raises(ValidationError):
            ConceptOutput(**data)

    def test_empty_methodology_raises(self):
        data = self._valid()
        data["methodology"] = []
        with pytest.raises(ValidationError):
            ConceptOutput(**data)


# ── SlideDeck ─────────────────────────────────────────────────────────────────

class TestSlideDeck:
    def test_valid_deck(self):
        slides = [_make_slide(i) for i in range(1, 7)]
        deck = SlideDeck(paper_title="My Paper", slides=slides)
        assert len(deck.slides) == 6

    def test_too_few_slides_raises(self):
        slides = [_make_slide(i) for i in range(1, 4)]   # only 3
        with pytest.raises(ValidationError, match="5–15"):
            SlideDeck(paper_title="P", slides=slides)

    def test_too_many_slides_raises(self):
        slides = [_make_slide(i) for i in range(1, 17)]  # 16
        with pytest.raises(ValidationError, match="5–15"):
            SlideDeck(paper_title="P", slides=slides)

    def test_non_sequential_slide_numbers_raises(self):
        slides = [_make_slide(i) for i in range(1, 7)]
        slides[2] = Slide(
            slide_number=99,     # wrong number
            title="Bad",
            bullets=["A", "B"],
        )
        with pytest.raises(ValidationError, match="sequential"):
            SlideDeck(paper_title="P", slides=slides)

    def test_too_few_bullets_raises(self):
        with pytest.raises(ValidationError, match="2–6"):
            Slide(slide_number=1, title="X", bullets=["only one"])

    def test_too_many_bullets_raises(self):
        with pytest.raises(ValidationError, match="2–6"):
            Slide(slide_number=1, title="X", bullets=[f"B{i}" for i in range(7)])


# ── VivaOutput ────────────────────────────────────────────────────────────────

class TestVivaOutput:
    def _valid_output(self, count: int = 5) -> VivaOutput:
        categories = ["conceptual", "methodological", "result-based", "critical", "comparison"]
        pairs = [_make_qa(categories[i % len(categories)]) for i in range(count)]
        return VivaOutput(paper_title="My Paper", qa_pairs=pairs)

    def test_valid_construction(self):
        vo = self._valid_output(5)
        assert len(vo.qa_pairs) == 5

    def test_too_few_pairs_raises(self):
        pairs = [_make_qa() for _ in range(4)]
        with pytest.raises(ValidationError, match="5–15"):
            VivaOutput(paper_title="P", qa_pairs=pairs)

    def test_too_many_pairs_raises(self):
        pairs = [_make_qa() for _ in range(16)]
        with pytest.raises(ValidationError, match="5–15"):
            VivaOutput(paper_title="P", qa_pairs=pairs)

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            VivaQA(
                question="Q?",
                model_answer="A.",
                category="invalid_category",   # type: ignore[arg-type]
            )

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            VivaQA(question="  ", model_answer="A.", category="conceptual")

