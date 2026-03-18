"""
Shared player-related helpers used by Hitter Profile, Pitcher Profile, and Hitter Comparison pages.
"""

import html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
