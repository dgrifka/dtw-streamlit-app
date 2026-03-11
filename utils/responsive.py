"""
Shared responsive CSS injection for mobile/tablet support.

Call inject_responsive_css() once per page, right after st.set_page_config().
All media queries live here so responsive behavior is consistent across pages.
"""

import base64
import os

import streamlit as st

_RESPONSIVE_CSS = """
<style>
/* ── Tablet / landscape phone (≤768px) ───────────────────────────── */
@media (max-width: 768px) {
    /* Force multi-column layouts to wrap (2-per-row) */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }

    /* Shrink hero text */
    .hero-title { font-size: 1.8rem !important; }
    .hero-subtitle { font-size: 1rem !important; }

    /* Compact metric labels */
    [data-testid="stMetric"] label { font-size: 0.75rem !important; }

    /* Card images: allow natural height */
    .responsive-card-img {
        height: auto !important;
        max-height: 200px !important;
    }

    /* Center page logo on mobile, slightly smaller for balance */
    .page-logo-container {
        justify-content: center !important;
    }
    .page-logo {
        width: 60px !important;
    }

    /* Prevent long page titles from wrapping */
    h1 { font-size: 1.6rem !important; }

    /* Narrower sidebar */
    [data-testid="stSidebar"] {
        min-width: 180px !important;
        max-width: 180px !important;
    }
}

/* ── Phone portrait (≤480px) ─────────────────────────────────────── */
@media (max-width: 480px) {
    /* Full single-column stacking */
    [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    .hero-title { font-size: 1.4rem !important; }
    .hero-subtitle { font-size: 0.85rem !important; }

    /* Smaller page titles on phones */
    h1 { font-size: 1.4rem !important; }

    /* Slightly smaller logo on phones */
    .page-logo { width: 50px !important; }

    /* Reduce padding in game cards */
    .game-card { padding: 0.6rem !important; }
    .game-info-box { padding: 0.6rem !important; }

    /* Smaller team scores in Game Detail */
    .team-score { font-size: 1.1rem !important; }
}
</style>
"""


def inject_responsive_css(extra_css: str = ""):
    """Inject responsive media queries into the current page.

    Parameters
    ----------
    extra_css : str, optional
        Additional CSS rules (wrapped in <style> tags) to inject alongside
        the shared responsive rules.  Useful for page-specific overrides.
    """
    st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)
    if extra_css:
        st.markdown(f"<style>{extra_css}</style>", unsafe_allow_html=True)


def render_page_logo(logo_path: str, width: int = 85):
    """Render the page logo as an HTML img that centers on mobile via CSS.

    Uses base64 encoding so the image works as an <img> tag with a CSS class,
    allowing the responsive media query to center and resize it on narrow screens.
    """
    if not os.path.exists(logo_path):
        return
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div class="page-logo-container" style="display:flex; justify-content:flex-end;">'
        f'<img src="data:image/png;base64,{logo_b64}" class="page-logo" '
        f'style="width:{width}px; height:auto;">'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_home_link():
    """Render a subtle 'Return to Home' link at the bottom of the page."""
    st.divider()
    st.page_link("Home.py", label="← Return to Home", icon="🏠")
