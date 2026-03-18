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
/* ── Gradient dividers ────────────────────────────────────────────── */
[data-testid="stDivider"] { margin: 1.25rem 0 !important; }
[data-testid="stDivider"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(to right, transparent, #CBD5E0, transparent) !important;
}

/* ── Metric card backgrounds ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #e8ecf1;
}

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

    # Clean up any sticky player bar left over from a previous page.
    # The bar is injected into window.parent.document by Profile/Comparison pages
    # and survives Streamlit page navigation because the iframe hosting its
    # cleanup observer is destroyed on navigation.  Each page that needs a bar
    # will re-create it immediately after this runs.
    import streamlit.components.v1 as _components
    _components.html("""
<script>
(function() {
    var doc = window.parent.document;
    var bar = doc.getElementById('sticky-player-bar');
    if (bar) bar.remove();
})();
</script>
""", height=0)


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


def upset_badge_html(is_upset: bool) -> str:
    """Return a styled HTML badge for upset games, or empty string."""
    if not is_upset:
        return ""
    return ('<span style="background:#E53E3E; color:white; font-size:0.65em; '
            'font-weight:700; padding:2px 8px; border-radius:10px; '
            'vertical-align:middle; margin-left:6px;">UPSET</span>')


def render_home_link():
    """Render a subtle 'Return to Home' link at the bottom of the page."""
    st.divider()
    st.page_link("_home.py", label="← Return to Home")
