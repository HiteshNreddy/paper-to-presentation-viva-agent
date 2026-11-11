"""
tests/test_pptx_builder.py — Unit tests for utils/pptx_builder.py.
"""

from __future__ import annotations

import io
from pathlib import Path

import pptx
import pytest

from schemas.slide_schema import Slide, SlideDeck
from utils.pptx_builder import (
    build_presentation,
    build_presentation_bytes,
    _sanitize_filename,
    SLIDE_WIDTH_INCHES,
    SLIDE_HEIGHT_INCHES,
)


@pytest.fixture
def sample_slide_deck() -> SlideDeck:
    """Construct a validated 10-slide SlideDeck."""
    slides = [
        Slide(
            slide_number=1,
            title="SparseDocNet: Dynamic Sparse Attention for Documents",
            bullets=[
                "Efficient document representation for long sequences.",
                "Overcomes quadratic memory bottlenecks of standard attention.",
                "Introduces Adaptive Dynamic Routing mechanism.",
            ],
            speaker_note="Welcome everyone. Today we present SparseDocNet.",
        ),
        Slide(
            slide_number=2,
            title="Introduction & Motivation",
            bullets=[
                "Long documents span thousands of tokens.",
                "Standard Transformers fail due to O(n^2) scaling.",
                "Sub-quadratic attention mechanisms are required.",
            ],
            speaker_note="Focus on real-world document lengths.",
        ),
    ]
    # Add slides 3..10 to satisfy the 5-15 slide constraint
    for i in range(3, 11):
        slides.append(
            Slide(
                slide_number=i,
                title=f"Section {i}: Academic Analysis",
                bullets=[
                    f"Core technical finding {i}.1 regarding evaluation.",
                    f"Methodology consideration {i}.2 on ablation studies.",
                ],
                speaker_note=f"Elaborate on details of section {i}.",
            )
        )

    return SlideDeck(
        paper_title="SparseDocNet: Dynamic Sparse Attention for Documents",
        slides=slides,
    )


class TestPPTXBuilder:
    def test_sanitize_filename(self):
        assert _sanitize_filename("Attention: Is All You Need?!") == "Attention_Is_All_You_Need"
        assert _sanitize_filename("   ") == "presentation"

    def test_build_presentation_bytes(self, sample_slide_deck):
        raw_bytes = build_presentation_bytes(sample_slide_deck)
        assert isinstance(raw_bytes, bytes)
        assert len(raw_bytes) > 0
        # Standard PPTX / ZIP magic bytes: PK\x03\x04
        assert raw_bytes.startswith(b"PK\x03\x04")

        # Verify it can be parsed back into a Presentation object
        prs = pptx.Presentation(io.BytesIO(raw_bytes))
        assert len(prs.slides) == 10
        assert round(prs.slide_width.inches, 2) == round(SLIDE_WIDTH_INCHES, 2)
        assert round(prs.slide_height.inches, 2) == round(SLIDE_HEIGHT_INCHES, 2)

    def test_build_presentation_file(self, sample_slide_deck, tmp_path):
        target_path = tmp_path / "test_presentation.pptx"
        result_path = build_presentation(sample_slide_deck, output_path=target_path)

        assert result_path == target_path
        assert target_path.exists()
        assert target_path.stat().st_size > 0

        # Verify presentation structure
        prs = pptx.Presentation(str(target_path))
        assert len(prs.slides) == 10

    def test_speaker_notes_embedded(self, sample_slide_deck):
        raw_bytes = build_presentation_bytes(sample_slide_deck)
        prs = pptx.Presentation(io.BytesIO(raw_bytes))

        # Check Slide 1 speaker notes
        slide_1 = prs.slides[0]
        assert "Welcome everyone" in slide_1.notes_slide.notes_text_frame.text

        # Check Slide 2 speaker notes
        slide_2 = prs.slides[1]
        assert "Focus on real-world" in slide_2.notes_slide.notes_text_frame.text

    def test_slide_titles_and_content_presence(self, sample_slide_deck):
        raw_bytes = build_presentation_bytes(sample_slide_deck)
        prs = pptx.Presentation(io.BytesIO(raw_bytes))

        # Slide 2 should have content
        slide_2 = prs.slides[1]
        texts = [shape.text for shape in slide_2.shapes if shape.has_text_frame]
        combined_text = " ".join(texts)
        assert "Introduction & Motivation" in combined_text
        assert "Long documents span thousands of tokens" in combined_text

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError, match="Expected SlideDeck instance"):
            build_presentation("invalid_string")

        with pytest.raises(TypeError, match="Expected SlideDeck instance"):
            build_presentation_bytes(None)

