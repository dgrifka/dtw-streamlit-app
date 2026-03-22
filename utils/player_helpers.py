"""
Shared player-related helpers used by Hitter Profile, Pitcher Profile, and Hitter Comparison pages.
"""

import html
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import unicodedata

# Plotly interaction config
PLOTLY_CONFIG = {"scrollZoom": False, "displayModeBar": False, "doubleClick": False}
PLOTLY_CONFIG_STATIC = {"staticPlot": True}
PLOTLY_CONFIG_FOREST = {
    "scrollZoom": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
    ],
    "displaylogo": False,
    "doubleClick": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "player_rankings",
        "height": 800,
        "width": 1200,
        "scale": 2,
    },
}

# Actual total bases mapping
TB_MAP = {
    "Single": 1, "Double": 2, "Triple": 3, "Home Run": 4,
    "Out": 0, "Field Out": 0, "Grounded Into Dp": 0,
    "Fielders Choice": 0, "Fielders Choice Out": 0,
    "Sac Fly": 0, "Sac Bunt": 0, "Double Play": 0,
    "Force Out": 0, "Field Error": 0,
}


def safe_html(text):
    """Escape text for safe HTML rendering in unsafe_allow_html=True contexts."""
    return html.escape(str(text)) if text else ""


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


def plotly_download_config(filename, width=1200, height=800, scale=2):
    """Plotly config with only the camera (download PNG) button visible."""
    return {
        "scrollZoom": False,
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        ],
        "displaylogo": False,
        "doubleClick": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": height,
            "width": width,
            "scale": scale,
        },
    }


def _player_filename_slug(name):
    """Convert player name to clean filename slug (e.g., 'Aaron Judge' -> 'judge')."""
    slug = name.split()[-1].lower() if " " in name else name.lower()
    return normalize_name(slug).replace(" ", "_")


def render_comparison_bar_html(
    player_name, player_val, player_low, player_high, player_color,
    best_name, best_val, best_low, best_high, best_color,
    league_mean, league_sd,
    value_fmt=".3f", value_suffix="", caption=None,
):
    """Render a 3-row comparison bar (player vs best vs league avg).

    Returns HTML string. Caller wraps in st.markdown(..., unsafe_allow_html=True).
    Sized to fit in a 1/3-width column.
    """
    lg_low = league_mean - league_sd
    lg_high = league_mean + league_sd
    all_vals = [player_low, player_high, best_low, best_high, lg_low, lg_high]
    pad = (max(all_vals) - min(all_vals)) * 0.1 + 0.01
    d_min = min(all_vals) - pad
    d_max = max(all_vals) + pad
    d_range = d_max - d_min
    if d_range <= 0:
        return ""

    def _pos(v):
        return max(0, min(100, (v - d_min) / d_range * 100))

    p_left = _pos(player_low)
    p_width = max(_pos(player_high) - p_left, 0.5)
    p_marker = _pos(player_val)

    b_left = _pos(best_low)
    b_width = max(_pos(best_high) - b_left, 0.5)
    b_marker = _pos(best_val)

    lg_left_pos = _pos(lg_low)
    lg_width_pos = max(_pos(lg_high) - lg_left_pos, 0.5)
    lg_marker_pos = _pos(league_mean)

    player_label = safe_html(player_name.split(" ")[-1][:8] if " " in player_name else player_name[:8])
    best_label = safe_html(best_name.split(" ")[-1][:8] if " " in best_name else best_name[:8])

    fmt = f"{{:{value_fmt}}}"
    p_str = fmt.format(player_val) + value_suffix
    b_str = fmt.format(best_val) + value_suffix
    lg_str = fmt.format(league_mean) + value_suffix

    caption_html = ""
    if caption:
        caption_html = (
            f'<div style="font-size:11px; color:rgba(120,120,120,0.8); '
            f'margin-top:2px; text-align:center;">{safe_html(caption)}</div>'
        )

    return (
        f'<div class="comp-bar-container" style="font-size:12px; margin:4px 0 2px 0;">'
        # Row 1: Player
        f'<div style="display:flex; align-items:center; height:22px; margin-bottom:3px;">'
        f'<div class="comp-bar-label" style="width:55px; text-align:right; padding-right:6px; font-weight:600; color:{player_color}; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:11px;">{player_label}</div>'
        f'<div style="flex:1; position:relative; height:14px; min-width:0;">'
        f'<div style="position:absolute; top:1px; left:{p_left:.1f}%; width:{p_width:.1f}%; '
        f'height:12px; background:{player_color}; opacity:0.7; border-radius:6px;"></div>'
        f'<div style="position:absolute; top:0px; left:{p_marker:.1f}%; '
        f'width:14px; height:14px; margin-left:-7px; '
        f'background:white; border:2.5px solid {player_color}; border-radius:50%;"></div>'
        f'</div>'
        f'<div class="comp-bar-value" style="width:44px; padding-left:4px; font-size:11px; color:{player_color}; font-weight:600;">{p_str}</div>'
        f'</div>'
        # Row 2: Best player
        f'<div style="display:flex; align-items:center; height:22px; margin-bottom:3px;">'
        f'<div class="comp-bar-label" style="width:55px; text-align:right; padding-right:6px; color:{best_color}; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:11px;" '
        f'title="{safe_html(best_name)}">{best_label}</div>'
        f'<div style="flex:1; position:relative; height:14px; min-width:0;">'
        f'<div style="position:absolute; top:1px; left:{b_left:.1f}%; width:{b_width:.1f}%; '
        f'height:12px; background:{best_color}; opacity:0.5; border-radius:6px;"></div>'
        f'<div style="position:absolute; top:0px; left:{b_marker:.1f}%; '
        f'width:12px; height:12px; margin-left:-6px; '
        f'background:{best_color}; transform:rotate(45deg);"></div>'
        f'</div>'
        f'<div class="comp-bar-value" style="width:44px; padding-left:4px; font-size:11px; color:{best_color};">{b_str}</div>'
        f'</div>'
        # Row 3: League average
        f'<div style="display:flex; align-items:center; height:22px;">'
        f'<div class="comp-bar-label" style="width:55px; text-align:right; padding-right:6px; color:rgba(120,120,120,0.9); font-size:11px;">Lg Avg</div>'
        f'<div style="flex:1; position:relative; height:14px; min-width:0;">'
        f'<div style="position:absolute; top:1px; left:{lg_left_pos:.1f}%; width:{lg_width_pos:.1f}%; '
        f'height:12px; background:rgba(160,160,160,0.35); border-radius:6px;"></div>'
        f'<div style="position:absolute; top:1px; left:{lg_marker_pos:.1f}%; '
        f'width:10px; height:10px; margin-left:-5px; '
        f'background:rgba(150,150,150,0.7); border-radius:50%;"></div>'
        f'</div>'
        f'<div class="comp-bar-value" style="width:44px; padding-left:4px; font-size:11px; color:rgba(120,120,120,0.9);">{lg_str}</div>'
        f'</div>'
        f'{caption_html}'
        f'</div>'
    )


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
                    margin-top:2px;">{safe_html(label)}</div>
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


def render_comparison_radar_chart(pcts_1, pcts_2, name_1, name_2, color_1, color_2):
    """
    Create a Plotly Scatterpolar radar chart with two overlaid player traces.

    Parameters
    ----------
    pcts_1, pcts_2 : dict
        {axis_label: percentile_value (0-100)} for each player.
    name_1, name_2 : str
        Player names for legend and hover.
    color_1, color_2 : str
        Hex color for each player (e.g., '#003087').

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    labels = list(pcts_1.keys())
    vals_1 = list(pcts_1.values())
    vals_2 = list(pcts_2.values())

    # Close the polygons
    labels_closed = labels + [labels[0]]
    vals_1_closed = vals_1 + [vals_1[0]]
    vals_2_closed = vals_2 + [vals_2[0]]

    hover_1 = [f"{name_1} — {l}: {_ordinal(int(v))} percentile" for l, v in zip(labels, vals_1)]
    hover_1.append(hover_1[0])
    hover_2 = [f"{name_2} — {l}: {_ordinal(int(v))} percentile" for l, v in zip(labels, vals_2)]
    hover_2.append(hover_2[0])

    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = _hex_to_rgb(color_1)
    r2, g2, b2 = _hex_to_rgb(color_2)

    fig = go.Figure()

    # Player 1 — stronger fill
    fig.add_trace(go.Scatterpolar(
        r=vals_1_closed, theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba({r1},{g1},{b1},0.25)",
        line=dict(color=color_1, width=2.5),
        marker=dict(size=7, color=color_1),
        text=hover_1, hoverinfo="text",
        name=name_1,
    ))

    # Player 2 — lighter fill
    fig.add_trace(go.Scatterpolar(
        r=vals_2_closed, theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba({r2},{g2},{b2},0.15)",
        line=dict(color=color_2, width=2),
        marker=dict(size=7, color=color_2),
        text=hover_2, hoverinfo="text",
        name=name_2,
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
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=13)),
        margin=dict(l=50, r=50, t=50, b=30),
        height=400,
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
        f'letter-spacing:0.3px;">{safe_html(archetype_name)}</span>'
        f'</div>'
        f'<div style="color:#4A5568; font-size:0.9rem; line-height:1.5;">{safe_html(description)}</div>'
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
            f'{safe_html(p["player"])} <span style="color:#A0AEC0;">({safe_html(p["team"])})</span>'
            f'<span style="float:right; color:#718096; font-size:0.85rem;">{sim_pct}% match</span>'
            f'</div>'
        )
    return "".join(items)


def render_sticky_player_bar(player_name, team_short, primary_color, headshot_url="", subtitle=""):
    """Inject a fixed identity bar that appears when the hero section scrolls out of view.

    Uses IntersectionObserver on ``#hero-section-sentinel`` to toggle visibility.
    The bar is injected into ``window.parent.document`` via ``components.html()``.
    """
    safe_name = safe_html(player_name)
    safe_team = safe_html(team_short)
    safe_sub = safe_html(subtitle)
    safe_color = safe_html(primary_color)
    safe_img = html.escape(headshot_url, quote=True) if headshot_url else ""

    img_tag = ""
    if safe_img:
        img_tag = (
            f'<img src="{safe_img}" '
            f'style="width:36px;height:36px;object-fit:contain;border-radius:50%;flex-shrink:0;" '
            f'onerror="this.style.display=\'none\'">'
        )

    components.html(f"""
<script>
(function() {{
    const doc = window.parent.document;
    const BAR_ID = 'sticky-player-bar';

    // Remove any existing bar (handles reruns / player changes)
    const old = doc.getElementById(BAR_ID);
    if (old) old.remove();

    // Build bar element
    const bar = doc.createElement('div');
    bar.id = BAR_ID;
    bar.innerHTML = `
        <div style="display:flex;align-items:center;gap:10px;max-width:720px;margin:0 auto;padding:0 16px;">
            {img_tag}
            <div style="min-width:0;">
                <div style="font-weight:700;font-size:0.95rem;color:#1a1a1a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {safe_name}
                </div>
                <div style="font-size:0.8rem;color:#4A5568;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {safe_team}{(' | ' + safe_sub) if safe_sub else ''}
                </div>
            </div>
        </div>
    `;

    // Detect Streamlit header height for positioning
    const stHeader = doc.querySelector('[data-testid="stHeader"]');
    const headerH = stHeader ? stHeader.offsetHeight : 0;

    Object.assign(bar.style, {{
        position: 'fixed',
        top: headerH + 'px',
        left: '0',
        right: '0',
        height: '52px',
        display: 'flex',
        alignItems: 'center',
        background: 'rgba(255,255,255,0.97)',
        borderBottom: '1px solid #e2e8f0',
        borderLeft: '4px solid {safe_color}',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        zIndex: '999',
        transform: 'translateY(-100%)',
        transition: 'transform 0.25s ease',
        backdropFilter: 'blur(8px)',
    }});

    // Responsive: smaller bar on mobile
    const style = doc.createElement('style');
    style.textContent = `
        @media (max-width: 480px) {{
            #sticky-player-bar {{
                height: 44px !important;
            }}
            #sticky-player-bar img {{
                width: 28px !important;
                height: 28px !important;
            }}
        }}
    `;
    doc.head.appendChild(style);
    doc.body.appendChild(bar);

    // Watch the hero sentinel
    function setupObserver() {{
        const sentinel = doc.getElementById('hero-section-sentinel');
        if (!sentinel) return;

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(e => {{
                bar.style.transform = e.isIntersecting ? 'translateY(-100%)' : 'translateY(0)';
            }});
        }}, {{ threshold: 0, rootMargin: '-' + headerH + 'px 0px 0px 0px' }});
        observer.observe(sentinel);

        // Cleanup when sentinel is removed (page navigation)
        const cleanup = new MutationObserver(() => {{
            if (!doc.getElementById('hero-section-sentinel')) {{
                bar.style.transform = 'translateY(-100%)';
                setTimeout(() => {{
                    const b = doc.getElementById(BAR_ID);
                    if (b) b.remove();
                    style.remove();
                }}, 300);
                cleanup.disconnect();
            }}
        }});
        cleanup.observe(doc.body, {{ childList: true, subtree: true }});
    }}

    // Sentinel may not be in parent DOM yet — poll briefly
    let attempts = 0;
    const poll = setInterval(() => {{
        if (doc.getElementById('hero-section-sentinel') || attempts > 20) {{
            clearInterval(poll);
            setupObserver();
        }}
        attempts++;
    }}, 100);
}})();
</script>
""", height=0)


def render_sticky_comparison_bar(
    name_1, team_1, color_1, headshot_url_1,
    name_2, team_2, color_2, headshot_url_2,
):
    """Inject a fixed bar showing both players that appears when the hero scrolls away.

    Same IntersectionObserver mechanism as ``render_sticky_player_bar`` but
    displays two players side-by-side with their respective team colors.
    """
    safe_n1 = safe_html(name_1)
    safe_t1 = safe_html(team_1)
    safe_c1 = safe_html(color_1)
    safe_n2 = safe_html(name_2)
    safe_t2 = safe_html(team_2)
    safe_c2 = safe_html(color_2)
    safe_img1 = html.escape(headshot_url_1, quote=True) if headshot_url_1 else ""
    safe_img2 = html.escape(headshot_url_2, quote=True) if headshot_url_2 else ""

    def _img(url):
        if not url:
            return ""
        return (
            f'<img src="{url}" '
            f'style="width:32px;height:32px;object-fit:contain;border-radius:50%;flex-shrink:0;" '
            f'onerror="this.style.display=\'none\'">'
        )

    img1 = _img(safe_img1)
    img2 = _img(safe_img2)

    components.html(f"""
<script>
(function() {{
    const doc = window.parent.document;
    const BAR_ID = 'sticky-player-bar';

    const old = doc.getElementById(BAR_ID);
    if (old) old.remove();

    const bar = doc.createElement('div');
    bar.id = BAR_ID;
    bar.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;gap:6px;max-width:780px;margin:0 auto;padding:0 12px;width:100%;">
            <div style="display:flex;align-items:center;gap:8px;flex:1;justify-content:flex-end;min-width:0;">
                <div style="text-align:right;min-width:0;">
                    <div style="font-weight:700;font-size:0.85rem;color:#1a1a1a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{safe_n1}</div>
                    <div style="font-size:0.72rem;color:#4A5568;white-space:nowrap;">{safe_t1}</div>
                </div>
                {img1}
            </div>
            <div style="font-weight:700;color:#8B8FA3;font-size:0.8rem;padding:0 6px;flex-shrink:0;">vs</div>
            <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:0;">
                {img2}
                <div style="min-width:0;">
                    <div style="font-weight:700;font-size:0.85rem;color:#1a1a1a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{safe_n2}</div>
                    <div style="font-size:0.72rem;color:#4A5568;white-space:nowrap;">{safe_t2}</div>
                </div>
            </div>
        </div>
    `;

    const stHeader = doc.querySelector('[data-testid="stHeader"]');
    const headerH = stHeader ? stHeader.offsetHeight : 0;

    Object.assign(bar.style, {{
        position: 'fixed',
        top: headerH + 'px',
        left: '0',
        right: '0',
        height: '52px',
        display: 'flex',
        alignItems: 'center',
        background: 'rgba(255,255,255,0.97)',
        borderBottom: '1px solid #e2e8f0',
        borderImage: 'linear-gradient(to right, {safe_c1}, {safe_c2}) 1',
        borderImageSlice: '0 0 0 1',
        borderLeft: '4px solid',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        zIndex: '999',
        transform: 'translateY(-100%)',
        transition: 'transform 0.25s ease',
        backdropFilter: 'blur(8px)',
    }});

    const style = doc.createElement('style');
    style.textContent = `
        @media (max-width: 480px) {{
            #sticky-player-bar {{
                height: 44px !important;
            }}
            #sticky-player-bar img {{
                width: 24px !important;
                height: 24px !important;
            }}
        }}
    `;
    doc.head.appendChild(style);
    doc.body.appendChild(bar);

    function setupObserver() {{
        const sentinel = doc.getElementById('hero-section-sentinel');
        if (!sentinel) return;

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(e => {{
                bar.style.transform = e.isIntersecting ? 'translateY(-100%)' : 'translateY(0)';
            }});
        }}, {{ threshold: 0, rootMargin: '-' + headerH + 'px 0px 0px 0px' }});
        observer.observe(sentinel);

        const cleanup = new MutationObserver(() => {{
            if (!doc.getElementById('hero-section-sentinel')) {{
                bar.style.transform = 'translateY(-100%)';
                setTimeout(() => {{
                    const b = doc.getElementById(BAR_ID);
                    if (b) b.remove();
                    style.remove();
                }}, 300);
                cleanup.disconnect();
            }}
        }});
        cleanup.observe(doc.body, {{ childList: true, subtree: true }});
    }}

    let attempts = 0;
    const poll = setInterval(() => {{
        if (doc.getElementById('hero-section-sentinel') || attempts > 20) {{
            clearInterval(poll);
            setupObserver();
        }}
        attempts++;
    }}, 100);
}})();
</script>
""", height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Player Snapshot components
# ─────────────────────────────────────────────────────────────────────────────

def _score_letter_grade(score):
    """Map 0-100 composite score to letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C+"
    elif score >= 40:
        return "C"
    elif score >= 30:
        return "D"
    else:
        return "F"


def render_snapshot_section(metrics, composite_score, archetype_name, primary_color, subtitle=None):
    """Render the combined overall score badge + percentile bars as a single HTML block.

    Parameters
    ----------
    metrics : list of dict
        Each dict: {"label": str, "pct": float 0-100, "value": str, "group": int (0 or 1)}
        group=0 for contact/power metrics, group=1 for discipline metrics.
    composite_score : float
        0-100 overall score (mean of radar percentiles).
    archetype_name : str
        Player archetype label.
    primary_color : str
        Team primary hex color.
    subtitle : str or None
        Optional small text shown below the archetype (e.g., "Includes 2026 projections").
    """
    grade = _score_letter_grade(composite_score)
    score_int = int(round(composite_score))

    # Parse hex color for rgba
    h = primary_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # Build bar rows HTML
    bar_rows = []
    prev_group = None
    for m in metrics:
        pct = max(0, min(100, m["pct"]))
        # Opacity scales with percentile: higher = darker (better)
        fill_opacity = 0.30 + 0.70 * (pct / 100)  # 0.30 at 0th to 1.0 at 100th

        # Separator between groups
        if prev_group is not None and m.get("group", 0) != prev_group:
            bar_rows.append(
                '<div style="border-top:1px dashed #E2E8F0; margin:6px 0;"></div>'
            )
        prev_group = m.get("group", 0)

        bar_rows.append(
            f'<div class="snapshot-bar-row" style="display:flex; align-items:center; height:30px; gap:8px;">'
            f'<div class="snapshot-label" style="width:90px; text-align:right; font-size:0.82rem; color:#4A5568; font-weight:500; white-space:nowrap;">{safe_html(m["label"])}</div>'
            f'<div style="flex:1; position:relative; height:18px; background:#EDF2F7; border-radius:9px; overflow:hidden;">'
            f'<div style="position:absolute; top:0; left:0; width:{pct:.1f}%; height:100%; background:rgba({r},{g},{b},{fill_opacity:.2f}); border-radius:9px; transition:width 0.3s ease;"></div>'
            f'<div style="position:absolute; top:0; left:50%; width:2px; height:100%; background:rgba(0,0,0,0.25); transform:translateX(-1px);"></div>'
            f'</div>'
            f'<div class="snapshot-value" style="width:100px; text-align:right; font-size:0.82rem; white-space:nowrap;">'
            f'<span style="color:#1a1a1a; font-weight:600;">{safe_html(m["value"])}</span>'
            f'<span style="color:#A0AEC0; margin-left:4px;">{_ordinal(int(pct))}</span>'
            f'</div>'
            f'</div>'
        )

    bars_html = "".join(bar_rows)
    subtitle_html = (
        f'<div style="margin-top:3px; font-size:0.68rem; color:#A0AEC0; text-align:center; '
        f'max-width:120px; font-style:italic;">{safe_html(subtitle)}</div>'
        if subtitle else ""
    )

    html_block = (
        f'<div class="snapshot-container" style="display:flex; gap:20px; align-items:stretch; background:#FAFBFC; border-radius:12px; padding:20px; border:1px solid #EDF2F7; margin-bottom:4px;">'
        f'<div class="snapshot-badge" style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:110px; flex-shrink:0;">'
        f'<div style="width:88px; height:88px; border-radius:50%; border:4px solid {primary_color}; display:flex; align-items:center; justify-content:center; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
        f'<span style="font-size:2rem; font-weight:800; color:#1a1a1a;">{score_int}</span>'
        f'</div>'
        f'<div style="margin-top:6px; font-size:1.3rem; font-weight:700; color:{primary_color};">{grade}</div>'
        f'<div style="margin-top:2px; font-size:0.78rem; color:#718096; text-align:center; max-width:110px;">{safe_html(archetype_name)}</div>'
        f'{subtitle_html}'
        f'</div>'
        f'<div class="snapshot-bars" style="flex:1; min-width:0;">{bars_html}</div>'
        f'</div>'
    )
    st.markdown(html_block, unsafe_allow_html=True)


def render_highlights(highlights, primary_color):
    """Render 2-4 auto-generated highlight cards.

    Parameters
    ----------
    highlights : list of dict
        Each: {"bold": str, "text": str}
    primary_color : str
        Team primary hex color.
    """
    if not highlights:
        return

    cards = []
    for hl in highlights:
        cards.append(
            f'<div style="border-left:3px solid {primary_color}; background:white; '
            f'padding:8px 14px; border-radius:0 8px 8px 0; margin-bottom:6px; '
            f'box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
            f'<span style="font-weight:700; color:#1a1a1a;">{safe_html(hl["bold"])}</span> '
            f'<span style="color:#4A5568;">{safe_html(hl["text"])}</span>'
            f'</div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_true_talent_bar(observed_eb, true_talent_eb, league_avg_eb, deviation, primary_color, is_pitcher=False):
    """Render a horizontal bar showing observed vs true talent vs league average.

    Only call when |deviation| > 0.005.
    """
    all_vals = [observed_eb, true_talent_eb, league_avg_eb]
    d_min = min(all_vals) - 0.03
    d_max = max(all_vals) + 0.03
    d_range = d_max - d_min
    if d_range <= 0:
        return

    def _pos(v):
        return max(0, min(100, (v - d_min) / d_range * 100))

    obs_pos = _pos(observed_eb)
    tt_pos = _pos(true_talent_eb)
    lg_pos = _pos(league_avg_eb)

    if is_pitcher:
        arrow_color = "#38A169" if deviation < 0 else "#E53E3E"
        outlook = "likely to improve" if deviation < 0 else "may regress"
    else:
        arrow_color = "#38A169" if deviation > 0 else "#E53E3E"
        outlook = "likely to improve" if deviation > 0 else "may regress"

    caption = f"Season stats suggest {observed_eb:.3f} EB/PA, but true talent estimate is {true_talent_eb:.3f} \u2014 {outlook}."

    html_block = (
        f'<div style="background:#FAFBFC; border-radius:10px; padding:14px 16px; border:1px solid #EDF2F7; margin-bottom:4px;">'
        f'<div style="font-size:0.82rem; font-weight:600; color:#718096; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">True Talent vs. Observed</div>'
        f'<div style="position:relative; height:28px; margin:0 8px;">'
        f'<div style="position:absolute; top:11px; left:0; right:0; height:6px; background:#EDF2F7; border-radius:3px;"></div>'
        f'<div style="position:absolute; top:4px; left:{lg_pos:.1f}%; width:2px; height:20px; margin-left:-1px; background:rgba(160,160,160,0.5);"></div>'
        f'<div style="position:absolute; top:12px; left:{min(obs_pos, tt_pos):.1f}%; width:{abs(tt_pos - obs_pos):.1f}%; height:4px; background:{arrow_color}; border-radius:2px; opacity:0.6;"></div>'
        f'<div style="position:absolute; top:4px; left:{obs_pos:.1f}%; width:20px; height:20px; margin-left:-10px; background:white; border:3px solid {primary_color}; border-radius:50%; box-shadow:0 1px 3px rgba(0,0,0,0.1);"></div>'
        f'<div style="position:absolute; top:7px; left:{tt_pos:.1f}%; width:14px; height:14px; margin-left:-7px; background:{arrow_color}; transform:rotate(45deg); box-shadow:0 1px 3px rgba(0,0,0,0.1);"></div>'
        f'</div>'
        f'<div style="display:flex; gap:16px; justify-content:center; margin-top:8px; font-size:0.75rem; color:#718096;">'
        f'<span>&#9679; Observed</span><span>&#9670; True Talent</span><span>&#124; Lg Avg</span>'
        f'</div>'
        f'<div style="font-size:0.82rem; color:#4A5568; margin-top:6px; text-align:center;">{safe_html(caption)}</div>'
        f'</div>'
    )
    st.markdown(html_block, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison page components
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison_grades(
    name_1, score_1, archetype_1, color_1,
    name_2, score_2, archetype_2, color_2,
):
    """Render two grade badges side by side for comparison pages."""
    def _badge(name, score, archetype, color):
        s = int(round(score))
        grade = _score_letter_grade(score)
        return (
            f'<div style="display:flex; flex-direction:column; align-items:center; flex:1;">'
            f'<div style="font-weight:700; color:{color}; font-size:0.9rem; margin-bottom:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:160px;">{safe_html(name)}</div>'
            f'<div style="width:72px; height:72px; border-radius:50%; border:3px solid {color}; display:flex; align-items:center; justify-content:center; background:white; box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<span style="font-size:1.6rem; font-weight:800; color:#1a1a1a;">{s}</span>'
            f'</div>'
            f'<div style="margin-top:4px; font-size:1.1rem; font-weight:700; color:{color};">{grade}</div>'
            f'<div style="font-size:0.75rem; color:#718096; text-align:center;">{safe_html(archetype)}</div>'
            f'</div>'
        )

    html = (
        f'<div style="display:flex; align-items:flex-start; justify-content:center; gap:40px; '
        f'background:#FAFBFC; border-radius:12px; padding:16px; border:1px solid #EDF2F7; margin-bottom:8px;">'
        f'{_badge(name_1, score_1, archetype_1, color_1)}'
        f'<div style="display:flex; align-items:center; padding-top:28px; color:#A0AEC0; font-weight:700; font-size:0.9rem;">vs</div>'
        f'{_badge(name_2, score_2, archetype_2, color_2)}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_comparison_bars(metrics, name_1, color_1, name_2, color_2):
    """Render side-by-side filled percentile bars for two players.

    Parameters
    ----------
    metrics : list of dict
        Each: {"label": str, "v1": str, "v2": str, "pct1": float, "pct2": float,
               "num1": float, "num2": float, "higher_better": bool (default True)}
    name_1, name_2 : str
        Player last names for compact labels.
    color_1, color_2 : str
        Hex team colors.
    """
    h1 = color_1.lstrip("#")
    r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
    h2 = color_2.lstrip("#")
    r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)

    # Compact last names
    ln1 = name_1.split()[-1][:8] if " " in name_1 else name_1[:8]
    ln2 = name_2.split()[-1][:8] if " " in name_2 else name_2[:8]

    rows = []
    for m in metrics:
        pct1_raw = m.get("pct1")
        pct2_raw = m.get("pct2")
        has_pct1 = pct1_raw is not None
        has_pct2 = pct2_raw is not None
        pct1 = max(0, min(100, pct1_raw)) if has_pct1 else 0
        pct2 = max(0, min(100, pct2_raw)) if has_pct2 else 0
        higher_better = m.get("higher_better", True)

        # Bold the winner
        n1, n2 = m.get("num1"), m.get("num2")
        w1 = w2 = ""
        if n1 is not None and n2 is not None:
            if higher_better:
                w1 = "font-weight:700;" if n1 > n2 else ""
                w2 = "font-weight:700;" if n2 > n1 else ""
            else:
                w1 = "font-weight:700;" if n1 < n2 else ""
                w2 = "font-weight:700;" if n2 < n1 else ""

        op1 = (0.30 + 0.70 * (pct1 / 100)) if has_pct1 else 0
        op2 = (0.30 + 0.70 * (pct2 / 100)) if has_pct2 else 0

        pct1_label = _ordinal(int(pct1)) if has_pct1 else ""
        pct2_label = _ordinal(int(pct2)) if has_pct2 else ""
        median_bg = "rgba(0,0,0,0.25)" if (has_pct1 or has_pct2) else "transparent"

        rows.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:#718096; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:3px;">{safe_html(m["label"])}</div>'
            # Player 1 bar
            f'<div style="display:flex; align-items:center; height:26px; gap:6px; margin-bottom:2px;">'
            f'<div style="width:60px; text-align:right; font-size:0.78rem; color:{color_1}; font-weight:600; white-space:nowrap;">{safe_html(ln1)}</div>'
            f'<div style="flex:1; position:relative; height:16px; background:#EDF2F7; border-radius:8px; overflow:hidden;">'
            f'<div style="position:absolute; top:0; left:0; width:{pct1:.1f}%; height:100%; background:rgba({r1},{g1},{b1},{op1:.2f}); border-radius:8px;"></div>'
            f'<div style="position:absolute; top:0; left:50%; width:2px; height:100%; background:{median_bg}; transform:translateX(-1px);"></div>'
            f'</div>'
            f'<div style="width:90px; text-align:right; font-size:0.78rem; white-space:nowrap;">'
            f'<span style="{w1} color:#1a1a1a;">{safe_html(m["v1"])}</span>'
            f'<span style="color:#A0AEC0; margin-left:3px;">{pct1_label}</span>'
            f'</div>'
            f'</div>'
            # Player 2 bar
            f'<div style="display:flex; align-items:center; height:26px; gap:6px;">'
            f'<div style="width:60px; text-align:right; font-size:0.78rem; color:{color_2}; font-weight:600; white-space:nowrap;">{safe_html(ln2)}</div>'
            f'<div style="flex:1; position:relative; height:16px; background:#EDF2F7; border-radius:8px; overflow:hidden;">'
            f'<div style="position:absolute; top:0; left:0; width:{pct2:.1f}%; height:100%; background:rgba({r2},{g2},{b2},{op2:.2f}); border-radius:8px;"></div>'
            f'<div style="position:absolute; top:0; left:50%; width:2px; height:100%; background:{median_bg}; transform:translateX(-1px);"></div>'
            f'</div>'
            f'<div style="width:90px; text-align:right; font-size:0.78rem; white-space:nowrap;">'
            f'<span style="{w2} color:#1a1a1a;">{safe_html(m["v2"])}</span>'
            f'<span style="color:#A0AEC0; margin-left:3px;">{pct2_label}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

    block = (
        f'<div style="background:#FAFBFC; border-radius:12px; padding:16px 12px; border:1px solid #EDF2F7;">'
        + "".join(rows)
        + f'</div>'
    )
    st.markdown(block, unsafe_allow_html=True)
