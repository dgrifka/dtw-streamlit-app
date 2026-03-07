"""
Shared player-related helpers used by Player Profile and Player Comparison pages.
"""

import pandas as pd
import streamlit as st
import unicodedata

# Plotly interaction config
PLOTLY_CONFIG = {"scrollZoom": False, "displayModeBar": False, "doubleClick": False}
PLOTLY_CONFIG_STATIC = {"staticPlot": True}

# Actual total bases mapping
TB_MAP = {
    "Single": 1, "Double": 2, "Triple": 3, "Home Run": 4,
    "Out": 0, "Field Out": 0, "Grounded Into Dp": 0,
    "Fielders Choice": 0, "Fielders Choice Out": 0,
    "Sac Fly": 0, "Sac Bunt": 0, "Double Play": 0,
    "Force Out": 0, "Field Error": 0,
}


def normalize_name(name):
    """Strip accents for fuzzy search."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()


def build_headshot_url(player_id):
    """MLB CDN headshot URL (silo PNG format)."""
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:silo:current.png/w_213,q_auto:best/v1/people/{player_id}/headshot/silo/current.png"


def build_video_url(play_id):
    """Baseball Savant video link from play_id."""
    if pd.isna(play_id) or play_id == "":
        return None
    return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


def categorize_launch_angle(la):
    """Categorize launch angle into batted ball type."""
    if pd.isna(la):
        return "Unknown"
    if la < 10:
        return "Ground Ball"
    elif la < 25:
        return "Line Drive"
    elif la < 50:
        return "Fly Ball"
    else:
        return "Pop Up"


def is_barrel(ev, la):
    """Check if a batted ball is a barrel (Statcast definition)."""
    if pd.isna(ev) or pd.isna(la):
        return False
    if ev < 98:
        return False
    extra = ev - 98
    la_low = max(8, 26 - extra)
    la_high = min(50, 30 + 2 * extra)
    return la_low <= la <= la_high


def _ordinal(n):
    """Return ordinal string for a number (e.g., 1st, 2nd, 3rd, 72nd)."""
    n = int(n)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{('th','st','nd','rd')[min(n%10,4) if n%10<4 else 0]}"


def _percentile_color(pct):
    """3-stop gradient: blue (0) -> gray (50) -> red (100)."""
    if pct <= 50:
        t = pct / 50
        r = int(59 + (156 - 59) * t)
        g = int(130 + (163 - 130) * t)
        b = int(246 + (175 - 246) * t)
    else:
        t = (pct - 50) / 50
        r = int(156 + (239 - 156) * t)
        g = int(163 + (68 - 163) * t)
        b = int(175 + (68 - 175) * t)
    return f"rgb({r},{g},{b})"


def render_percentile_bar(percentile, label=None, container=None):
    """Render a horizontal percentile bar with colored circle indicator."""
    if percentile is None:
        return
    target = container if container is not None else st
    pct = max(0, min(100, percentile))
    color = _percentile_color(pct)
    if label is None:
        label = f"{_ordinal(int(pct))} percentile"
    target.markdown(f"""
    <div style="margin:-8px 0 4px 0;">
        <div style="position:relative; height:20px; margin:0 10px;">
            <div style="position:absolute; top:7px; left:0; right:0; height:6px;
                        background:rgba(180,180,180,0.3); border-radius:3px;"></div>
            <div style="position:absolute; top:0; left:{pct}%;
                        width:20px; height:20px; margin-left:-10px;
                        background:{color}; border-radius:50%;
                        box-shadow:0 1px 3px rgba(0,0,0,0.15);"></div>
        </div>
        <div style="text-align:center; font-size:11px; color:rgba(150,150,150,0.9);
                    margin-top:2px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def luck_tier_label(percentile):
    """Return 5-tier luck label based on percentile."""
    if percentile is None:
        return None
    if percentile < 10:
        return "Very Unlucky"
    elif percentile < 30:
        return "Unlucky"
    elif percentile < 70:
        return "Luck Neutral"
    elif percentile < 90:
        return "Lucky"
    else:
        return "Very Lucky"
