"""
ui/styles.py — Apple iPhone Duo / iOS 27 Design System for Streamlit.

Provides a unified CSS system supporting both Light and Dark modes with:
- Translucent frosted glass surfaces (backdrop-filter: blur(24px)).
- SF Pro / Inter typography hierarchy with ultra-clean letter-spacing.
- Apple-grade color palettes (iOS system grays, Apple blue accent, semantic tints).
- Fluid responsive cards, 16:9 slide previews, pill badges, and chat bubbles.
- Modern streamlined tabs, selectboxes, buttons, and expanders.
"""

from __future__ import annotations


def get_apple_css(theme: str = "dark") -> str:
    """
    Generate the complete CSS string matching Apple iOS 27 / iPhone Duo aesthetic.

    Parameters
    ----------
    theme : str
        "dark" (default) or "light".
    """
    is_dark = theme.lower() == "dark"

    # Color tokens
    if is_dark:
        bg_primary = "#0B0C10"
        bg_secondary = "#12141C"
        surface_glass = "rgba(22, 25, 34, 0.72)"
        surface_glass_hover = "rgba(30, 34, 48, 0.85)"
        surface_elevated = "rgba(28, 32, 44, 0.82)"
        border_subtle = "rgba(255, 255, 255, 0.08)"
        border_focus = "rgba(10, 132, 255, 0.5)"
        text_primary = "#F5F5F7"
        text_secondary = "#8E8E93"
        text_tertiary = "#636366"
        accent_blue = "#0A84FF"
        accent_blue_hover = "#0071E3"
        accent_blue_subtle = "rgba(10, 132, 255, 0.15)"
        accent_green = "#30D158"
        accent_green_subtle = "rgba(48, 209, 88, 0.14)"
        accent_amber = "#FF9F0A"
        accent_amber_subtle = "rgba(255, 159, 10, 0.14)"
        accent_red = "#FF453A"
        accent_red_subtle = "rgba(255, 69, 58, 0.14)"
        shadow_ambient = "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
        shadow_card = "0 4px 20px rgba(0, 0, 0, 0.25)"
        slide_card_bg = "linear-gradient(135deg, rgba(20, 24, 33, 0.95), rgba(13, 16, 23, 0.98))"
    else:
        bg_primary = "#F5F5F7"
        bg_secondary = "#E5E5EA"
        surface_glass = "rgba(255, 255, 255, 0.78)"
        surface_glass_hover = "rgba(255, 255, 255, 0.92)"
        surface_elevated = "rgba(255, 255, 255, 0.88)"
        border_subtle = "rgba(0, 0, 0, 0.06)"
        border_focus = "rgba(0, 113, 227, 0.45)"
        text_primary = "#1D1D1F"
        text_secondary = "#6E6E73"
        text_tertiary = "#8E8E93"
        accent_blue = "#0071E3"
        accent_blue_hover = "#0077ED"
        accent_blue_subtle = "rgba(0, 113, 227, 0.1)"
        accent_green = "#34C759"
        accent_green_subtle = "rgba(52, 199, 89, 0.12)"
        accent_amber = "#FF9500"
        accent_amber_subtle = "rgba(255, 149, 0, 0.12)"
        accent_red = "#FF3B30"
        accent_red_subtle = "rgba(255, 59, 48, 0.1)"
        shadow_ambient = "0 8px 30px rgba(0, 0, 0, 0.05)"
        shadow_card = "0 4px 16px rgba(0, 0, 0, 0.04)"
        slide_card_bg = "linear-gradient(135deg, #FFFFFF, #F8F9FA)"

    return f"""
<style>
    /* ── Apple iOS 27 Design Foundation ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --apple-bg: {bg_primary};
        --apple-surface: {surface_glass};
        --apple-surface-elevated: {surface_elevated};
        --apple-border: {border_subtle};
        --apple-border-focus: {border_focus};
        --apple-text: {text_primary};
        --apple-text-secondary: {text_secondary};
        --apple-accent: {accent_blue};
        --apple-accent-hover: {accent_blue_hover};
        --apple-accent-subtle: {accent_blue_subtle};
        --apple-shadow: {shadow_ambient};
    }}

    /* Global Streamlit App Overrides */
    .stApp {{
        background: {bg_primary} !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif !important;
        color: {text_primary} !important;
        letter-spacing: -0.012em;
    }}

    /* Sidebar Glass Surface */
    [data-testid="stSidebar"] {{
        background: {surface_glass} !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border-right: 1px solid {border_subtle} !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
    }}

    /* Main container padding and max-width */
    .main .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }}

    /* ── Typography ─────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {{
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif !important;
        color: {text_primary} !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }}

    p, span, label, li {{
        color: {text_primary};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
    }}

    /* ── Apple Glass Cards ───────────────────────────────────────── */
    .apple-glass-card {{
        background: {surface_glass} !important;
        backdrop-filter: blur(24px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(160%) !important;
        border: 1px solid {border_subtle} !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: {shadow_card} !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease !important;
        margin-bottom: 1.25rem !important;
    }}

    .apple-glass-card:hover {{
        border-color: {border_focus} !important;
        box-shadow: {shadow_ambient} !important;
    }}

    .apple-glass-card-sm {{
        background: {surface_glass} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid {border_subtle} !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }}

    /* Hero Banner */
    .apple-hero {{
        text-align: center;
        padding: 3.5rem 1.5rem 2.5rem 1.5rem;
        background: radial-gradient(circle at 50% 20%, {accent_blue_subtle} 0%, transparent 70%);
        border-radius: 28px;
        margin-bottom: 2.2rem;
        border: 1px solid {border_subtle};
    }}

    .apple-hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        background: {accent_blue_subtle};
        color: {accent_blue};
        border: 1px solid {border_focus};
        margin-bottom: 1rem;
    }}

    .apple-hero-title {{
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -0.04em;
        margin-bottom: 0.85rem;
        background: linear-gradient(135deg, {text_primary} 30%, {text_secondary} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .apple-hero-subtitle {{
        font-size: 1.15rem;
        color: {text_secondary};
        max-width: 680px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.55;
    }}

    /* ── Metadata Pills ─────────────────────────────────────────── */
    .apple-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: {surface_elevated};
        border: 1px solid {border_subtle};
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        color: {text_primary};
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}

    .apple-pill-blue {{
        background: {accent_blue_subtle};
        border-color: {border_focus};
        color: {accent_blue};
    }}

    .apple-pill-green {{
        background: {accent_green_subtle};
        border-color: rgba(48, 209, 88, 0.4);
        color: {accent_green};
    }}

    .apple-pill-amber {{
        background: {accent_amber_subtle};
        border-color: rgba(255, 159, 10, 0.4);
        color: {accent_amber};
    }}

    .apple-pill-red {{
        background: {accent_red_subtle};
        border-color: rgba(255, 69, 58, 0.4);
        color: {accent_red};
    }}

    /* ── Streamlit Tabs Styling (iOS 27 Segmented Control) ───────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {surface_glass} !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid {border_subtle} !important;
        border-radius: 16px !important;
        padding: 4px !important;
        gap: 6px !important;
        margin-bottom: 1.5rem !important;
        display: flex !important;
        justify-content: stretch !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px !important;
        padding: 0.55rem 1.1rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: {text_secondary} !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        flex: 1 1 auto !important;
        text-align: center !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {accent_blue} !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(10, 132, 255, 0.35) !important;
        font-weight: 600 !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* ── Apple Primary & Secondary Buttons ──────────────────────── */
    .stButton > button {{
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1.4rem !important;
        border: 1px solid {border_subtle} !important;
        background: {surface_elevated} !important;
        color: {text_primary} !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }}

    .stButton > button:hover {{
        border-color: {accent_blue} !important;
        color: {accent_blue} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
    }}

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, {accent_blue}, {accent_blue_hover}) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(10, 132, 255, 0.4) !important;
    }}

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        box-shadow: 0 6px 20px rgba(10, 132, 255, 0.55) !important;
        transform: translateY(-1.5px) !important;
    }}

    /* Download Button */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {accent_blue}, {accent_blue_hover}) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.6rem !important;
        box-shadow: 0 4px 18px rgba(10, 132, 255, 0.45) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    .stDownloadButton > button:hover {{
        box-shadow: 0 6px 24px rgba(10, 132, 255, 0.6) !important;
        transform: translateY(-2px) !important;
    }}

    /* ── 16:9 Presentation Slide Card ────────────────────────────── */
    .apple-slide-card {{
        background: {slide_card_bg} !important;
        border: 1px solid {border_subtle} !important;
        border-radius: 20px !important;
        padding: 1.8rem 2rem !important;
        margin-bottom: 1.6rem !important;
        position: relative !important;
        box-shadow: {shadow_ambient} !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }}

    .apple-slide-card:hover {{
        border-color: {border_focus} !important;
        transform: translateY(-2px) !important;
    }}

    .apple-slide-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid {border_subtle};
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }}

    .apple-slide-num {{
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: {accent_blue};
        letter-spacing: 0.05em;
    }}

    .apple-slide-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {text_primary};
        margin: 0;
    }}

    .apple-slide-bullet {{
        font-size: 0.98rem;
        line-height: 1.6;
        color: {text_primary};
        margin-bottom: 0.65rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }}

    .apple-slide-bullet::before {{
        content: "•";
        color: {accent_blue};
        font-size: 1.3rem;
        line-height: 1;
    }}

    /* ── Viva Defense Q&A Card ───────────────────────────────────── */
    .apple-viva-card {{
        background: {surface_glass};
        border: 1px solid {border_subtle};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }}

    .apple-viva-card:hover {{
        border-color: {border_focus};
    }}

    .apple-viva-meta {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }}

    .apple-viva-q {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {text_primary};
        line-height: 1.45;
        margin-bottom: 0.6rem;
    }}

    /* ── Chat Messages (Apple Duo Style) ─────────────────────────── */
    .stChatMessage {{
        background: transparent !important;
        padding: 0.8rem 0 !important;
    }}

    .apple-chat-user {{
        background: linear-gradient(135deg, {accent_blue}, {accent_blue_hover});
        color: #FFFFFF !important;
        border-radius: 20px 20px 4px 20px;
        padding: 0.9rem 1.3rem;
        max-width: 82%;
        margin-left: auto;
        box-shadow: 0 4px 14px rgba(10, 132, 255, 0.3);
    }}

    .apple-chat-assistant {{
        background: {surface_glass};
        backdrop-filter: blur(20px);
        border: 1px solid {border_subtle};
        color: {text_primary};
        border-radius: 20px 20px 20px 4px;
        padding: 1.1rem 1.4rem;
        max-width: 90%;
        box-shadow: {shadow_card};
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background: {surface_glass} !important;
        border-radius: 12px !important;
        border: 1px solid {border_subtle} !important;
        font-weight: 500 !important;
        color: {text_primary} !important;
    }}

    /* Inputs and Selectbox */
    div[data-baseweb="select"] > div {{
        background: {surface_glass} !important;
        border-color: {border_subtle} !important;
        border-radius: 12px !important;
        color: {text_primary} !important;
    }}

    .stTextInput > div > div > input {{
        background: {surface_glass} !important;
        border: 1px solid {border_subtle} !important;
        border-radius: 14px !important;
        color: {text_primary} !important;
        padding: 0.6rem 1rem !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {accent_blue} !important;
        box-shadow: 0 0 0 2px {accent_blue_subtle} !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background: {surface_glass} !important;
        backdrop-filter: blur(16px) !important;
        border: 1.5px dashed {border_subtle} !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        transition: border-color 0.2s ease !important;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {accent_blue} !important;
    }}

    /* Hide standard Streamlit header anchor links */
    .css-15zrgzn, .e104v570-0 {{
        display: none !important;
    }}
</style>
"""

