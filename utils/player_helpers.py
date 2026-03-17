"""
Shared player-related helpers used by Hitter Profile, Pitcher Profile, and Hitter Comparison pages.
"""

import numpy as np
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


def is_barrel_vectorized(ev: pd.Series, la: pd.Series) -> pd.Series:
    """Vectorized barrel check — same logic as is_barrel but on entire columns."""
    extra = ev - 98
    la_low = np.maximum(8, 26 - extra)
    la_high = np.minimum(50, 30 + 2 * extra)
    return (ev >= 98) & (la >= la_low) & (la <= la_high)


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
        <div style="text-align:center; font-size:13px; color:rgba(150,150,150,0.9);
                    margin-top:2px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_comparison_metric(label, v1_str, v2_str, v1_num, v2_num, pct1=None, pct2=None, bold_higher=True, invert=False):
    """Return HTML for a side-by-side metric row with optional percentile bars.

    Parameters
    ----------
    label : str
        Metric name displayed in center column.
    v1_str, v2_str : str
        Formatted display values for player 1 and player 2.
    v1_num, v2_num : float
        Numeric values for comparison (bold the higher one).
    pct1, pct2 : float | None
        Percentile values (0-100) for bar display.  None = no bar.
    bold_higher : bool
        Whether to bold the higher value.
    invert : bool
        If True, bold the *lower* value instead.
    """
    w1 = w2 = ""
    if bold_higher and v1_num is not None and v2_num is not None:
        if invert:
            w1 = "font-weight:700;" if v1_num < v2_num else ""
            w2 = "font-weight:700;" if v2_num < v1_num else ""
        else:
            w1 = "font-weight:700;" if v1_num > v2_num else ""
            w2 = "font-weight:700;" if v2_num > v1_num else ""

    def _bar_html(pct):
        if pct is None:
            return ""
        pct = max(0, min(100, pct))
        color = _percentile_color(pct)
        return (
            f'<div style="position:relative; height:14px; margin:2px 10px 0 10px;">'
            f'<div style="position:absolute; top:4px; left:0; right:0; height:5px;'
            f' background:rgba(180,180,180,0.25); border-radius:3px;"></div>'
            f'<div style="position:absolute; top:0; left:{pct}%;'
            f' width:14px; height:14px; margin-left:-7px;'
            f' background:{color}; border-radius:50%;'
            f' box-shadow:0 1px 2px rgba(0,0,0,0.12);"></div>'
            f'</div>'
            f'<div style="text-align:center; font-size:12px; color:rgba(150,150,150,0.85);'
            f' margin-top:1px;">{_ordinal(int(pct))} pct</div>'
        )

    left_bar = _bar_html(pct1)
    right_bar = _bar_html(pct2)

    return (
        f'<div style="display:flex; align-items:flex-start; border-bottom:1px solid #EDF2F7; padding:8px 0;">'
        f'<div style="flex:1; text-align:right; padding-right:12px;">'
        f'<div style="{w1} color:#1a1a1a; font-size:0.95rem;">{v1_str}</div>'
        f'{left_bar}'
        f'</div>'
        f'<div style="width:130px; text-align:center; font-weight:600; color:#1E3A5F; font-size:0.85rem; padding-top:2px;">{label}</div>'
        f'<div style="flex:1; text-align:left; padding-left:12px;">'
        f'<div style="{w2} color:#1a1a1a; font-size:0.95rem;">{v2_str}</div>'
        f'{right_bar}'
        f'</div>'
        f'</div>'
    )


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


def render_radar_chart(percentiles, primary_color, secondary_color=None):
    """
    Create a Plotly Scatterpolar radar chart from percentile dict.

    Parameters
    ----------
    percentiles : dict
        {axis_label: percentile_value (0-100)}
    primary_color : str
        Team primary color hex (e.g., '#003087')
    secondary_color : str, optional
        Not used currently, reserved for future dual-player overlays.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    labels = list(percentiles.keys())
    values = list(percentiles.values())
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    # Percentile annotations on each point
    hover_text = [f"{l}: {_ordinal(int(v))} percentile" for l, v in zip(labels, values)]
    hover_text.append(hover_text[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba({int(primary_color.lstrip('#')[0:2], 16)},{int(primary_color.lstrip('#')[2:4], 16)},{int(primary_color.lstrip('#')[4:6], 16)},0.2)",
        line=dict(color=primary_color, width=2),
        marker=dict(size=6, color=primary_color),
        text=hover_text,
        hoverinfo="text",
        customdata=[[_ordinal(int(v))] for v in values_closed],
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25th", "50th", "75th"],
                tickfont=dict(size=10, color="rgba(150,150,150,0.7)"),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#333"),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=30, b=30),
        height=350,
        template="plotly_white",
        dragmode=False,
    )

    return fig


def render_archetype_badge(archetype_name, description, primary_color):
    """Render an HTML archetype badge with team-colored pill."""
    return (
        f'<div style="margin-bottom:12px;">'
        f'<span style="display:inline-block; background:{primary_color}; color:white; '
        f'padding:4px 14px; border-radius:16px; font-weight:600; font-size:0.95rem; '
        f'letter-spacing:0.3px;">{archetype_name}</span>'
        f'</div>'
        f'<div style="color:#4A5568; font-size:0.9rem; line-height:1.5;">{description}</div>'
    )


def render_similar_players(similar_list):
    """Render a compact list of similar players with similarity scores."""
    if not similar_list:
        return '<div style="color:#718096; font-size:0.9rem;">Not enough data for comparison</div>'

    items = []
    for i, p in enumerate(similar_list, 1):
        sim_pct = int(p["similarity"])
        items.append(
            f'<div style="padding:3px 0; font-size:0.9rem;">'
            f'<span style="color:#718096; font-weight:600;">{i}.</span> '
            f'{p["player"]} <span style="color:#A0AEC0;">({p["team"]})</span>'
            f'<span style="float:right; color:#718096; font-size:0.85rem;">{sim_pct}% match</span>'
            f'</div>'
        )
    return "".join(items)
