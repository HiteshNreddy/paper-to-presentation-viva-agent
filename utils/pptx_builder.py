"""
utils/pptx_builder.py — Professional PowerPoint Presentation Generator.

Generates a formatted 16:9 widescreen presentation (.pptx) from a
validated SlideDeck Pydantic model using python-pptx.

Public API
----------
build_presentation(slide_deck, output_path=None) -> Path
    Generates and saves the PPTX file to disk. Returns absolute Path.

build_presentation_bytes(slide_deck) -> bytes
    Generates the PPTX file entirely in memory and returns raw bytes.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from config.settings import settings
from schemas.slide_schema import Slide, SlideDeck
from utils.file_utils import ensure_directories

logger = logging.getLogger(__name__)

# ── Aesthetic Academic Design Constants (16:9 Widescreen) ──────────────────────
SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5

# Palette
COLOR_PRIMARY = RGBColor(24, 43, 73)      # Academic Deep Navy (#182B49)
COLOR_SECONDARY = RGBColor(70, 80, 95)    # Slate Gray (#46505F)
COLOR_ACCENT = RGBColor(41, 128, 185)     # Classic Academic Blue (#2980B9)
COLOR_TEXT = RGBColor(33, 37, 41)         # Off-black Charcoal (#212529)
COLOR_MUTED = RGBColor(108, 117, 125)     # Muted Footer Gray (#6C757D)
COLOR_BG_CARD = RGBColor(248, 249, 250)   # Off-white Card (#F8F9FA)
COLOR_WHITE = RGBColor(255, 255, 255)

FONT_HEADING = "Arial"
FONT_BODY = "Arial"


def _sanitize_filename(name: str) -> str:
    """Derive a safe filesystem filename from paper title."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean[:50] or "presentation"


def _create_base_presentation() -> pptx.Presentation:
    """Create a new Presentation with 16:9 widescreen dimensions."""
    prs = pptx.Presentation()
    prs.slide_width = Inches(SLIDE_WIDTH_INCHES)
    prs.slide_height = Inches(SLIDE_HEIGHT_INCHES)
    return prs


def _add_title_slide(
    prs: pptx.Presentation,
    slide_data: Slide,
    paper_title: str,
) -> None:
    """Render the opening Title Slide."""
    blank_layout = prs.slide_layouts[6]  # Blank slide
    slide = prs.slides.add_slide(blank_layout)

    # Top accent band
    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0),
        Inches(0),
        Inches(SLIDE_WIDTH_INCHES),
        Inches(0.4),
    )
    band.fill.solid()
    band.fill.fore_color.rgb = COLOR_ACCENT
    band.line.color.rgb = COLOR_ACCENT

    # Main Card Box
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.0),
        Inches(1.2),
        Inches(11.333),
        Inches(5.2),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_BG_CARD
    card.line.color.rgb = RGBColor(222, 226, 230)

    # Title & Subtitle Text Box
    tx_box = slide.shapes.add_textbox(
        Inches(1.5),
        Inches(1.6),
        Inches(10.333),
        Inches(4.4),
    )
    tf = tx_box.text_frame
    tf.word_wrap = True

    # Main Paper Title
    p_title = tf.paragraphs[0]
    p_title.text = paper_title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY
    p_title.space_after = Pt(14)

    # Subtitle / Presentation Type
    p_sub = tf.add_paragraph()
    p_sub.text = "Research Paper Analysis & Technical Defense Presentation"
    p_sub.font.name = FONT_HEADING
    p_sub.font.size = Pt(16)
    p_sub.font.bold = True
    p_sub.font.color.rgb = COLOR_ACCENT
    p_sub.space_after = Pt(20)

    # Highlights from Slide 1 bullets
    for b in slide_data.bullets:
        p_b = tf.add_paragraph()
        p_b.text = f"•  {b}"
        p_b.font.name = FONT_BODY
        p_b.font.size = Pt(14)
        p_b.font.color.rgb = COLOR_SECONDARY
        p_b.space_after = Pt(8)

    # Embed speaker notes
    if slide_data.speaker_note:
        slide.notes_slide.notes_text_frame.text = slide_data.speaker_note.strip()


def _add_content_slide(
    prs: pptx.Presentation,
    slide_data: Slide,
    paper_title: str,
    total_slides: int,
) -> None:
    """Render a standard academic content slide with bullets and speaker notes."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Header Title Area
    header_box = slide.shapes.add_textbox(
        Inches(0.8),
        Inches(0.5),
        Inches(11.733),
        Inches(1.0),
    )
    tf_header = header_box.text_frame
    tf_header.word_wrap = True

    p_header = tf_header.paragraphs[0]
    p_header.text = slide_data.title
    p_header.font.name = FONT_HEADING
    p_header.font.size = Pt(24)
    p_header.font.bold = True
    p_header.font.color.rgb = COLOR_PRIMARY

    # Accent Divider Line below title
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.8),
        Inches(1.4),
        Inches(11.733),
        Inches(0.04),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_ACCENT
    line.line.color.rgb = COLOR_ACCENT

    # Content Body Box
    body_box = slide.shapes.add_textbox(
        Inches(0.8),
        Inches(1.7),
        Inches(11.733),
        Inches(4.8),
    )
    tf_body = body_box.text_frame
    tf_body.word_wrap = True

    for i, bullet in enumerate(slide_data.bullets):
        p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
        p.text = f"•   {bullet.strip()}"
        p.font.name = FONT_BODY
        p.font.size = Pt(17)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(16)
        p.line_spacing = 1.15

    # Footer Area
    footer_box = slide.shapes.add_textbox(
        Inches(0.8),
        Inches(6.8),
        Inches(11.733),
        Inches(0.4),
    )
    tf_footer = footer_box.text_frame
    p_footer = tf_footer.paragraphs[0]
    
    short_title = paper_title if len(paper_title) <= 55 else paper_title[:52] + "..."
    footer_text = f"{short_title}  |  Slide {slide_data.slide_number} of {total_slides}"
    p_footer.text = footer_text
    p_footer.font.name = FONT_BODY
    p_footer.font.size = Pt(10)
    p_footer.font.color.rgb = COLOR_MUTED

    # Embed speaker notes
    if slide_data.speaker_note:
        slide.notes_slide.notes_text_frame.text = slide_data.speaker_note.strip()


def build_presentation(
    slide_deck: SlideDeck,
    output_path: Path | str | None = None,
) -> Path:
    """
    Generate an academic PowerPoint (.pptx) file from a validated SlideDeck.

    Parameters
    ----------
    slide_deck : SlideDeck
        Validated SlideDeck Pydantic model with sequential slides and notes.
    output_path : Path | str | None
        Optional target file path. If None, saves to settings.OUTPUTS_DIR.

    Returns
    -------
    Path
        Absolute path to the created presentation file.
    """
    if not isinstance(slide_deck, SlideDeck):
        raise TypeError(f"Expected SlideDeck instance, got {type(slide_deck).__name__}")

    ensure_directories()
    prs = _create_base_presentation()
    total_slides = len(slide_deck.slides)

    for i, slide in enumerate(slide_deck.slides):
        if i == 0:
            _add_title_slide(prs, slide, slide_deck.paper_title)
        else:
            _add_content_slide(prs, slide, slide_deck.paper_title, total_slides)

    if output_path is None:
        safe_name = _sanitize_filename(slide_deck.paper_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_file = settings.OUTPUTS_DIR / f"{safe_name}_{timestamp}.pptx"
    else:
        dest_file = Path(output_path).resolve()

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(dest_file))
    logger.info("Successfully generated PowerPoint presentation at: %s", dest_file)
    return dest_file


def build_presentation_bytes(slide_deck: SlideDeck) -> bytes:
    """
    Generate an academic PowerPoint (.pptx) file in memory and return raw bytes.
    Useful for direct delivery via Streamlit st.download_button.

    Parameters
    ----------
    slide_deck : SlideDeck
        Validated SlideDeck Pydantic model.

    Returns
    -------
    bytes
        Raw PPTX file bytes.
    """
    if not isinstance(slide_deck, SlideDeck):
        raise TypeError(f"Expected SlideDeck instance, got {type(slide_deck).__name__}")

    prs = _create_base_presentation()
    total_slides = len(slide_deck.slides)

    for i, slide in enumerate(slide_deck.slides):
        if i == 0:
            _add_title_slide(prs, slide, slide_deck.paper_title)
        else:
            _add_content_slide(prs, slide, slide_deck.paper_title, total_slides)

    buffer = io.BytesIO()
    prs.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
