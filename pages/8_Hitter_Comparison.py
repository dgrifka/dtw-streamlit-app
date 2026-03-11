"""
Hitter Comparison Page

Side-by-side comparison of two hitters: contact quality, distributions,
spray charts, luck reports, and Bayesian rankings.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_batted_balls, get_available_batted_ball_seasons,
    load_all_season_pa_rankings, compute_league_percentiles,
    load_player_evaluations_pa, load_player_metadata, load_pa_counts,
    load_player_projections, resolve_player_id,
)
from utils.team_mappings import get_team_color, get_team_logo_url
from utils.player_helpers import (
    normalize_name, build_headshot_url, categorize_launch_angle,
    is_barrel, is_barrel_vectorized,
    _ordinal, _percentile_color, render_percentile_bar, render_comparison_metric,
    luck_tier_label,
    TB_MAP, PLOTLY_CONFIG, PLOTLY_CONFIG_STATIC,
)
from utils.responsive import inject_responsive_css, render_home_link

st.set_page_config(
    page_title="Hitter Comparison | DTW Simulator",
    page_icon="⚾",
    layout="wide",
)

inject_responsive_css()

# Disable autocorrect/autocapitalize on selectbox search input
import streamlit.components.v1 as components
components.html("""
<script>
    const doc = window.parent.document;
    const observer = new MutationObserver(() => {
        doc.querySelectorAll('[data-testid="stSelectbox"] input').forEach(el => {
            el.setAttribute('autocorrect', 'off');
            el.setAttribute('autocapitalize', 'off');
            el.setAttribute('spellcheck', 'false');
            el.setAttribute('autocomplete', 'off');
        });
    });
    observer.observe(doc.body, {childList: true, subtree: true});
</script>
""", height=0)


# =============================================================================
# DATA LOADING
# =============================================================================

col_season, _ = st.columns([1, 3])
with col_season:
    available_seasons = get_available_batted_ball_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0, key="cmp_season")
    else:
        season = pd.Timestamp.now().year

bb_df = load_batted_balls(season)
if bb_df.empty:
    st.title("Hitter Comparison")
    st.info(f"No batted ball data available for {season}.")
    st.stop()

metadata_df = load_player_metadata(season)
pa_rankings = load_player_evaluations_pa(season, "hitter")

# Multi-season PA rankings for historical timeline
all_season_pa_rankings = load_all_season_pa_rankings("hitter")

_title_col, _logo_col = st.columns([5, 1])
with _title_col:
    st.title("Hitter Comparison")
with _logo_col:
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mlb_simulator_logo.png")
    if os.path.exists(_logo_path):
        st.image(_logo_path, width=85)

# =============================================================================
# PLAYER SELECTION
# =============================================================================

# Build player list
player_teams = (
    bb_df.groupby(["player", "team"]).size()
    .reset_index(name="count")
    .drop(columns="count")
)
player_teams["display"] = player_teams.apply(
    lambda r: f"{r['player']} ({r['team']})", axis=1
)
display_to_name = dict(zip(player_teams["display"], player_teams["player"]))
display_to_team = dict(zip(player_teams["display"], player_teams["team"]))
display_list = sorted(player_teams["display"].unique())

# Eligible players (50+ BBs) for random selection
eligible_names = bb_df.groupby(["player", "team"]).size()
eligible_names = eligible_names[eligible_names >= 50].index.tolist()
eligible_displays = [f"{n} ({t})" for n, t in eligible_names]
if not eligible_displays:
    eligible_displays = display_list

# Resolve defaults from query params or random
qp1 = st.query_params.get("p1", "")
qp2 = st.query_params.get("p2", "")


def _resolve_index(query_val, exclude_display=None):
    """Resolve a query param to a display_list index."""
    if query_val and query_val in display_list:
        return display_list.index(query_val)
    if query_val:
        qn = normalize_name(query_val)
        fuzzy = [i for i, d in enumerate(display_list) if qn in normalize_name(d)]
        if fuzzy:
            return fuzzy[0]
    # Random pick, try to avoid the excluded player
    pool = [d for d in eligible_displays if d != exclude_display] or eligible_displays
    pick = random.choice(pool)
    if pick in display_list:
        return display_list.index(pick)
    return 0


default_idx1 = _resolve_index(qp1)
default_pick1 = display_list[default_idx1]
default_idx2 = _resolve_index(qp2, exclude_display=default_pick1)

# Layout: Player 1 | Swap | Player 2 | Shuffle Both
sel1, swap_col, sel2, shuffle_col = st.columns([3, 0.5, 3, 1])

with sel1:
    selected_p1 = st.selectbox("Player 1", options=display_list, index=default_idx1, key="cmp_p1")
with swap_col:
    st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True)
    if st.button("Swap", width="stretch"):
        st.query_params["p1"] = st.session_state.get("cmp_p2", display_list[default_idx2])
        st.query_params["p2"] = st.session_state.get("cmp_p1", display_list[default_idx1])
        st.rerun()
with sel2:
    selected_p2 = st.selectbox("Player 2", options=display_list, index=default_idx2, key="cmp_p2")
with shuffle_col:
    st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True)
    if st.button("Shuffle Both", width="stretch"):
        picks = random.sample(eligible_displays, min(2, len(eligible_displays)))
        st.query_params["p1"] = picks[0]
        st.query_params["p2"] = picks[1] if len(picks) > 1 else picks[0]
        st.rerun()

# Update query params for bookmarking
st.query_params["p1"] = selected_p1
st.query_params["p2"] = selected_p2

if selected_p1 == selected_p2:
    st.info("Same player selected for both sides. Select different players to see a comparison.")


# =============================================================================
# RESOLVE PLAYER DATA
# =============================================================================

def resolve_player_data(display_label):
    """Resolve all data for a selected player display label."""
    name = display_to_name.get(display_label, display_label)
    team = display_to_team.get(display_label)
    pbb = bb_df[(bb_df["player"] == name) & (bb_df["team"] == team)].copy()
    if pbb.empty:
        return None
    pbb = pbb.sort_values("date_parsed")
    team_short = pbb["team"].iloc[-1]

    # Metadata
    meta = None
    if not metadata_df.empty:
        match = metadata_df[(metadata_df["player_name"] == name) & (metadata_df["team"] == team_short)]
        if match.empty:
            match = metadata_df[metadata_df["player_name"] == name]
        if match.empty:
            pn = normalize_name(name)
            match = metadata_df[metadata_df["player_name"].apply(normalize_name).str.contains(pn)]
        if not match.empty:
            meta = match.iloc[0]

    # Player ID
    pid = None
    if meta is not None and "player_id" in meta.index:
        pid = int(meta["player_id"])
    if pid is None:
        pid = resolve_player_id(name, metadata_df, pa_rankings)

    # PA ranking
    ranking = None
    if not pa_rankings.empty:
        rmatch = pa_rankings[(pa_rankings["player"] == name) & (pa_rankings["team"] == team_short)]
        if rmatch.empty:
            rmatch = pa_rankings[pa_rankings["player"] == name]
        if not rmatch.empty:
            ranking = rmatch.iloc[0]

    # Team colors
    primary, secondary = get_team_color(team_short)

    # Computed stats
    pbb["actual_tb"] = pbb["actual_result"].map(TB_MAP).fillna(0)
    pbb["is_barrel"] = is_barrel_vectorized(pbb["launch_speed"], pbb["launch_angle"])
    pbb["bb_type"] = pbb["launch_angle"].apply(categorize_launch_angle)

    return {
        "name": name, "team": team_short, "bb": pbb, "meta": meta,
        "player_id": pid, "ranking": ranking,
        "primary_color": primary, "secondary_color": secondary,
        "avg_ev": pbb["launch_speed"].mean(),
        "avg_eb": pbb["estimated_bases"].mean(),
        "n_bb": len(pbb),
        "barrel_rate": pbb["is_barrel"].mean() * 100,
        "total_actual_tb": pbb["actual_tb"].sum(),
        "total_expected_tb": pbb["estimated_bases"].sum(),
    }


p1 = resolve_player_data(selected_p1)
p2 = resolve_player_data(selected_p2)

if p1 is None or p2 is None:
    st.warning("Could not load data for one or both players.")
    st.stop()

# Handle same-team color collision: use secondary color for player 2
p2_color = p2["primary_color"]
if p1["team"] == p2["team"]:
    p2_color = p2["secondary_color"]
    # If secondary is too dark/close to primary, use a fallback
    if p2_color == p2["primary_color"]:
        p2_color = "#E67E22"
p1_color = p1["primary_color"]


# =============================================================================
# HEAD-TO-HEAD HERO
# =============================================================================

st.divider()

hero1, hero2 = st.columns(2)

for col, p, color in [(hero1, p1, p1_color), (hero2, p2, p2_color)]:
    with col:
        img_col, info_col = st.columns([1, 2])
        with img_col:
            if p["player_id"]:
                try:
                    st.image(build_headshot_url(p["player_id"]), width=130)
                except Exception:
                    logo = get_team_logo_url(p["team"])
                    if logo:
                        st.image(logo, width=100)
            else:
                logo = get_team_logo_url(p["team"])
                if logo:
                    st.image(logo, width=100)
        with info_col:
            st.markdown(f"### {p['name']}")
            pos_str = ""
            age_str = ""
            if p["meta"] is not None:
                pos = p["meta"].get("position", "")
                if pos:
                    pos_str = f" | {pos}"
                bd = p["meta"].get("birth_date", "")
                if bd:
                    try:
                        birth = pd.to_datetime(bd)
                        age = (pd.Timestamp.now() - birth).days // 365
                        age_str = f" | Age {age}"
                    except Exception:
                        pass
            st.markdown(f"**{p['team']}**{pos_str}{age_str}")

# Comparison metrics
st.markdown("#### Head-to-Head")
st.caption(f"{season} Season")

# Compute percentiles for EV, barrel rate, avg EB
league_pcts = compute_league_percentiles(season, "player")
if league_pcts:
    ev_pct_1 = (league_pcts["ev_by_player"] < p1["avg_ev"]).mean() * 100
    ev_pct_2 = (league_pcts["ev_by_player"] < p2["avg_ev"]).mean() * 100
    barrel_pct_1 = (league_pcts["barrel_rates"] < p1["barrel_rate"]).mean() * 100
    barrel_pct_2 = (league_pcts["barrel_rates"] < p2["barrel_rate"]).mean() * 100
    avg_eb_pct_1 = (league_pcts["avg_eb_by_player"] < p1["avg_eb"]).mean() * 100
    avg_eb_pct_2 = (league_pcts["avg_eb_by_player"] < p2["avg_eb"]).mean() * 100
else:
    ev_pct_1 = ev_pct_2 = barrel_pct_1 = barrel_pct_2 = avg_eb_pct_1 = avg_eb_pct_2 = 50

# EB/PA percentile
eb_pa_pct_1 = eb_pa_pct_2 = None
if p1["ranking"] is not None and p2["ranking"] is not None and not pa_rankings.empty:
    all_eb_pa = pa_rankings["posterior_mean"]
    eb_pa_pct_1 = (all_eb_pa < p1["ranking"]["posterior_mean"]).mean() * 100
    eb_pa_pct_2 = (all_eb_pa < p2["ranking"]["posterior_mean"]).mean() * 100

luck_1 = p1["total_actual_tb"] - p1["total_expected_tb"]
luck_2 = p2["total_actual_tb"] - p2["total_expected_tb"]

# Luck percentiles (computed early for Head-to-Head)
luck_pct_1 = luck_pct_2 = None
actual_pct_1 = actual_pct_2 = None
expected_pct_1 = expected_pct_2 = None
if len(bb_df) > 1000:
    all_luck = bb_df.copy()
    all_luck["actual_tb"] = all_luck["actual_result"].map(TB_MAP).fillna(0)
    player_luck = all_luck.groupby("player").agg(
        actual=("actual_tb", "sum"), expected=("estimated_bases", "sum"),
        n=("estimated_bases", "count"),
    )
    player_luck = player_luck[player_luck["n"] >= 30]
    player_luck["luck"] = player_luck["actual"] - player_luck["expected"]
    if p1["name"] in player_luck.index:
        luck_pct_1 = (player_luck["luck"] < luck_1).mean() * 100
        actual_pct_1 = (player_luck["actual"] < p1["total_actual_tb"]).mean() * 100
        expected_pct_1 = (player_luck["expected"] < p1["total_expected_tb"]).mean() * 100
    if p2["name"] in player_luck.index:
        luck_pct_2 = (player_luck["luck"] < luck_2).mean() * 100
        actual_pct_2 = (player_luck["actual"] < p2["total_actual_tb"]).mean() * 100
        expected_pct_2 = (player_luck["expected"] < p2["total_expected_tb"]).mean() * 100

# Build comparison rows using render_comparison_metric
h2h_rows = ""
h2h_rows += render_comparison_metric("Batted Balls", f"{p1['n_bb']:,}", f"{p2['n_bb']:,}", p1['n_bb'], p2['n_bb'], bold_higher=False)
h2h_rows += render_comparison_metric("Avg EV", f"{p1['avg_ev']:.1f} mph", f"{p2['avg_ev']:.1f} mph", p1['avg_ev'], p2['avg_ev'], ev_pct_1, ev_pct_2)
h2h_rows += render_comparison_metric("Barrel Rate", f"{p1['barrel_rate']:.1f}%", f"{p2['barrel_rate']:.1f}%", p1['barrel_rate'], p2['barrel_rate'], barrel_pct_1, barrel_pct_2)
h2h_rows += render_comparison_metric("Avg EB/BB", f"{p1['avg_eb']:.3f}", f"{p2['avg_eb']:.3f}", p1['avg_eb'], p2['avg_eb'], avg_eb_pct_1, avg_eb_pct_2)

if p1["ranking"] is not None and p2["ranking"] is not None:
    eb1 = p1["ranking"]["posterior_mean"]
    eb2 = p2["ranking"]["posterior_mean"]
    h2h_rows += render_comparison_metric("EB/PA", f"{eb1:.3f}", f"{eb2:.3f}", eb1, eb2, eb_pa_pct_1, eb_pa_pct_2)

h2h_rows += render_comparison_metric("Net Lucky Bases", f"{luck_1:+.1f}", f"{luck_2:+.1f}", luck_1, luck_2, luck_pct_1, luck_pct_2, bold_higher=False)

st.markdown(f'<div style="background:#F7FAFC; border-radius:8px; padding:8px 4px;">{h2h_rows}</div>', unsafe_allow_html=True)


# =============================================================================
# BAYESIAN EB/PA COMPARISON BAR
# =============================================================================

if p1["ranking"] is not None and p2["ranking"] is not None:
    st.divider()
    st.subheader("Est. Bases / PA Comparison")
    st.caption(f"{season} Season")

    eb1 = p1["ranking"]["posterior_mean"]
    hdi1_low = p1["ranking"]["hdi_low"]
    hdi1_high = p1["ranking"]["hdi_high"]
    eb2 = p2["ranking"]["posterior_mean"]
    hdi2_low = p2["ranking"]["hdi_low"]
    hdi2_high = p2["ranking"]["hdi_high"]

    # Best player
    best_idx = pa_rankings["posterior_mean"].idxmax()
    best_row = pa_rankings.loc[best_idx]
    best_eb = best_row["posterior_mean"]
    best_hdi_low = best_row["hdi_low"]
    best_hdi_high = best_row["hdi_high"]
    best_name = best_row["player"]
    best_team = best_row.get("team", "")
    best_color, _ = get_team_color(best_team) if best_team else ("#DAA520", "#DAA520")

    # League average
    lg_mean = pa_rankings["posterior_mean"].mean()
    lg_sd = pa_rankings["posterior_mean"].std()
    lg_low = lg_mean - lg_sd
    lg_high = lg_mean + lg_sd

    display_min = min(hdi1_low, hdi2_low, best_hdi_low, lg_low) - 0.03
    display_max = max(hdi1_high, hdi2_high, best_hdi_high, lg_high) + 0.03
    display_range = display_max - display_min

    def _pct_pos(val):
        return max(0, min(100, (val - display_min) / display_range * 100))

    def _bar_row(label, color, mean_val, hdi_lo, hdi_hi, marker_style="circle"):
        left = _pct_pos(hdi_lo)
        width = _pct_pos(hdi_hi) - left
        marker = _pct_pos(mean_val)
        label_short = label.split(" ")[-1][:10] if " " in label else label[:10]

        if marker_style == "diamond":
            marker_html = (
                f'<div style="position:absolute; top:1px; left:{marker:.1f}%; '
                f'width:14px; height:14px; margin-left:-7px; '
                f'background:{color}; transform:rotate(45deg);"></div>'
            )
        else:
            marker_html = (
                f'<div style="position:absolute; top:0px; left:{marker:.1f}%; '
                f'width:16px; height:16px; margin-left:-8px; '
                f'background:white; border:3px solid {color}; '
                f'border-radius:50%;"></div>'
            )

        return (
            f'<div style="display:flex; align-items:center; height:26px; margin-bottom:4px;">'
            f'<div style="width:80px; text-align:right; padding-right:8px; font-weight:600; color:{color}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{label}">{label_short}</div>'
            f'<div style="flex:1; position:relative; height:16px;">'
            f'<div style="position:absolute; top:2px; left:{left:.1f}%; width:{width:.1f}%; height:12px; background:{color}; opacity:0.5; border-radius:6px;"></div>'
            f'{marker_html}'
            f'</div>'
            f'<div style="width:50px; padding-left:6px; font-size:11px; color:{color}; font-weight:600;">{mean_val:.3f}</div>'
            f'</div>'
        )

    # League avg row
    lg_left = _pct_pos(lg_low)
    lg_width = _pct_pos(lg_high) - lg_left
    lg_marker = _pct_pos(lg_mean)
    lg_row = (
        f'<div style="display:flex; align-items:center; height:26px;">'
        f'<div style="width:80px; text-align:right; padding-right:8px; color:rgba(120,120,120,0.9);">Lg Avg</div>'
        f'<div style="flex:1; position:relative; height:16px;">'
        f'<div style="position:absolute; top:2px; left:{lg_left:.1f}%; width:{lg_width:.1f}%; height:12px; background:rgba(160,160,160,0.35); border-radius:6px;"></div>'
        f'<div style="position:absolute; top:2px; left:{lg_marker:.1f}%; width:12px; height:12px; margin-left:-6px; background:rgba(150,150,150,0.7); border-radius:50%;"></div>'
        f'</div>'
        f'<div style="width:50px; padding-left:6px; font-size:11px; color:rgba(120,120,120,0.9);">{lg_mean:.3f}</div>'
        f'</div>'
    )

    bar_html = (
        '<div style="font-size:12px; margin:8px 0 2px 0;">'
        + _bar_row(p1["name"], p1_color, eb1, hdi1_low, hdi1_high)
        + _bar_row(p2["name"], p2_color, eb2, hdi2_low, hdi2_high)
        + _bar_row(best_name, best_color, best_eb, best_hdi_low, best_hdi_high, "diamond")
        + lg_row
        + '</div>'
    )
    st.markdown(bar_html, unsafe_allow_html=True)
    st.caption("Bayesian estimate of true production per plate appearance with 89% HDI credible intervals.")


# =============================================================================
# HISTORICAL EB/PA TIMELINE (moved up to follow current-season EB/PA)
# =============================================================================

if len(all_season_pa_rankings) > 0 and (p1["player_id"] is not None or p2["player_id"] is not None):
    # Collect data for both players
    tl_data = {1: [], 2: []}
    league_avg_data = []
    best_player_data = []

    for s in sorted(all_season_pa_rankings.keys()):
        pa_df = all_season_pa_rankings[s]

        lg_mean_s = pa_df["posterior_mean"].mean()
        league_avg_data.append({"season": s, "value": lg_mean_s})

        best_idx_s = pa_df["posterior_mean"].idxmax()
        best_row_s = pa_df.loc[best_idx_s]
        best_team_s = best_row_s.get("team", "")
        best_color_s, _ = get_team_color(best_team_s) if best_team_s else ("#DAA520", "#DAA520")
        best_player_data.append({
            "season": s, "value": best_row_s["posterior_mean"],
            "name": best_row_s["player"], "color": best_color_s,
        })

        for pnum, pdata in [(1, p1), (2, p2)]:
            match = pd.DataFrame()
            if pdata["player_id"] and "player_id" in pa_df.columns:
                match = pa_df[pa_df["player_id"] == pdata["player_id"]]
            if match.empty:
                match = pa_df[pa_df["player"] == pdata["name"]]
            if not match.empty:
                row = match.iloc[0]
                tl_data[pnum].append({
                    "season": s, "value": row["posterior_mean"],
                    "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"],
                })

    if tl_data[1] or tl_data[2]:
        st.divider()
        st.subheader("Est. Bases / PA")

        lg_df = pd.DataFrame(league_avg_data)
        best_df = pd.DataFrame(best_player_data)

        # Trim to union of both players' season ranges
        all_player_seasons = [d["season"] for d in tl_data[1]] + [d["season"] for d in tl_data[2]]
        if all_player_seasons:
            min_s, max_s = min(all_player_seasons), max(all_player_seasons)
            lg_df = lg_df[(lg_df["season"] >= min_s) & (lg_df["season"] <= max_s)]
            best_df = best_df[(best_df["season"] >= min_s) & (best_df["season"] <= max_s)]

        fig_tl = go.Figure()

        # League average
        fig_tl.add_trace(go.Scatter(
            x=lg_df["season"], y=lg_df["value"],
            mode="markers", name="Lg Avg",
            marker=dict(color="rgba(160,160,160,0.8)", size=11),
            hovertemplate="Season: %{x}<br>Lg Avg: %{y:.3f}<extra></extra>",
        ))

        # Best player — gold diamonds per season
        if not best_df.empty:
            first_best = best_df["season"].iloc[0]
            for _, brow in best_df.iterrows():
                fig_tl.add_trace(go.Scatter(
                    x=[brow["season"]], y=[brow["value"]],
                    mode="markers",
                    name=brow["name"],
                    marker=dict(color="#DAA520", size=12, symbol="diamond",
                                line=dict(width=1, color="white")),
                    hovertemplate=f"Season: %{{x}}<br>{brow['name']}: %{{y:.3f}}<extra></extra>",
                    legendgroup="best",
                    showlegend=bool(brow["season"] == first_best),
                ))
            fig_tl.data[-len(best_df)].name = "Best Hitter"

        # Player lines with HDI
        for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
            if not tl_data[pnum]:
                continue
            tl_df = pd.DataFrame(tl_data[pnum])

            # Parse hex color for rgba fill
            c = color.lstrip("#")
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

            if len(tl_df) == 1:
                # Single point: use error bars for HDI
                row = tl_df.iloc[0]
                fig_tl.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["value"],
                    mode="markers", name=pdata["name"],
                    marker=dict(color=color, size=12),
                    error_y=dict(
                        type="data",
                        array=[row["hdi_high"] - row["value"]],
                        arrayminus=[row["value"] - row["hdi_low"]],
                        color=f"rgba({r},{g},{b},0.5)",
                        thickness=2, width=6,
                    ),
                    hovertemplate="Season: %{x}<br>EB/PA: %{y:.3f}<extra></extra>",
                ))
            else:
                # Multiple points: fill band + line
                fig_tl.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["hdi_high"],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["hdi_low"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.15)",
                    showlegend=False, hoverinfo="skip",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["value"],
                    mode="lines+markers", name=pdata["name"],
                    line=dict(color=color, width=2.5),
                    marker=dict(color=color, size=12),
                    hovertemplate="Season: %{x}<br>EB/PA: %{y:.3f}<extra></extra>",
                ))

        # Add last-name annotations at the last data point of each player's line
        for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
            if tl_data[pnum]:
                last_pt = tl_data[pnum][-1]
                last_name = pdata["name"].split()[-1]
                fig_tl.add_annotation(
                    x=last_pt["season"], y=last_pt["value"],
                    text=last_name, showarrow=False,
                    xshift=10, font=dict(color=color, size=11),
                    xanchor="left",
                )

        # Multi-year projections for both players
        x_max = max_s
        proj_data = {}
        for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
            if pdata["player_id"] is None:
                continue
            points = []
            for proj_season in range(max_s + 1, max_s + 4):
                proj_df = load_player_projections(proj_season, "hitter")
                if proj_df.empty:
                    continue
                proj_match = proj_df[proj_df["player_id"] == pdata["player_id"]] if "player_id" in proj_df.columns else pd.DataFrame()
                if proj_match.empty:
                    proj_match = proj_df[proj_df["player"] == pdata["name"]]
                if not proj_match.empty:
                    points.append(proj_match.iloc[0].to_dict() | {"season": proj_season})
            if points:
                proj_data[pnum] = (points, color, pdata)

        if proj_data:
            all_proj_seasons = [p["season"] for pts, _, _ in proj_data.values() for p in pts]
            x_max = max(all_proj_seasons)

            # Shared projection zone
            fig_tl.add_vrect(
                x0=max_s + 0.5, x1=max(all_proj_seasons) + 0.5,
                fillcolor="rgba(180,180,220,0.10)", line_width=0,
                layer="below",
            )
            fig_tl.add_annotation(
                x=(max_s + 0.5 + max(all_proj_seasons) + 0.5) / 2,
                y=1.0, yref="paper", yanchor="bottom",
                text="Projected", showarrow=False,
                font=dict(size=13, color="rgba(120,120,160,0.7)"),
            )

            for pnum, (points, color, pdata) in proj_data.items():
                c = color.lstrip("#")
                pr, pg, pb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

                p_seasons = [p["season"] for p in points]
                p_values = [p["projected_eb_pa"] for p in points]
                p_hdi_hi = [p["projected_hdi_high"] for p in points]
                p_hdi_lo = [p["projected_hdi_low"] for p in points]

                # HDI ribbon
                fig_tl.add_trace(go.Scatter(
                    x=p_seasons, y=p_hdi_hi,
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_tl.add_trace(go.Scatter(
                    x=p_seasons, y=p_hdi_lo,
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=f"rgba({pr},{pg},{pb},0.10)",
                    showlegend=False, hoverinfo="skip",
                ))

                # Dashed connection from last actual to first projection
                if tl_data[pnum]:
                    last_actual_pt = tl_data[pnum][-1]
                    fig_tl.add_trace(go.Scatter(
                        x=[last_actual_pt["season"], p_seasons[0]],
                        y=[last_actual_pt["value"], p_values[0]],
                        mode="lines",
                        line=dict(color=f"rgba({pr},{pg},{pb},0.4)", width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Projection trajectory line
                if len(p_seasons) > 1:
                    fig_tl.add_trace(go.Scatter(
                        x=p_seasons, y=p_values,
                        mode="lines",
                        line=dict(color=f"rgba({pr},{pg},{pb},0.5)", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Projection markers (open diamonds)
                fig_tl.add_trace(go.Scatter(
                    x=p_seasons, y=p_values,
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color="rgba(255,255,255,0)", size=12,
                        symbol="diamond-open",
                        line=dict(width=2.5, color=color),
                    ),
                    customdata=[
                        [p.get("aging_effect", 0), p["projected_hdi_low"], p["projected_hdi_high"]]
                        for p in points
                    ],
                    hovertemplate=(
                        "Projection %{x}<br>"
                        "EB/PA: %{y:.3f}<br>"
                        "89% HDI: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>"
                        "Aging effect: %{customdata[0]:+.3f}"
                        "<extra></extra>"
                    ),
                ))

        fig_tl.update_layout(
            xaxis=dict(title="Season", title_font_size=14, tickfont_size=13, dtick=1, tickformat="d",
                       range=[min_s - 0.5, x_max + 0.5]),
            yaxis=dict(title="Est. Bases per PA", title_font_size=14, tickfont_size=13),
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
            dragmode=False,
        )
        st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CONFIG)


# =============================================================================
# OVERLAID DISTRIBUTION CHARTS
# =============================================================================

st.divider()
st.subheader("Contact Quality Distributions")
st.caption(f"{season} Season")

col_ev, col_la, col_eb = st.columns(3)

with col_ev:
    st.markdown("#### Exit Velocity")
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Histogram(
        x=p1["bb"]["launch_speed"], name=p1["name"], opacity=0.6,
        marker_color=p1_color, histnorm="probability density", nbinsx=30,
    ))
    fig_ev.add_trace(go.Histogram(
        x=p2["bb"]["launch_speed"], name=p2["name"], opacity=0.6,
        marker_color=p2_color, histnorm="probability density", nbinsx=30,
    ))
    fig_ev.update_layout(
        barmode="overlay", xaxis_title="Exit Velocity (mph)", yaxis_title="Density",
        height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        dragmode=False,
    )
    st.plotly_chart(fig_ev, width="stretch", config=PLOTLY_CONFIG)

with col_la:
    st.markdown("#### Launch Angle")
    fig_la = go.Figure()
    fig_la.add_trace(go.Histogram(
        x=p1["bb"]["launch_angle"], name=p1["name"], opacity=0.6,
        marker_color=p1_color, histnorm="probability density", nbinsx=30,
    ))
    fig_la.add_trace(go.Histogram(
        x=p2["bb"]["launch_angle"], name=p2["name"], opacity=0.6,
        marker_color=p2_color, histnorm="probability density", nbinsx=30,
    ))
    fig_la.update_layout(
        barmode="overlay", xaxis_title="Launch Angle (deg)", yaxis_title="Density",
        height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        dragmode=False,
    )
    st.plotly_chart(fig_la, width="stretch", config=PLOTLY_CONFIG)

with col_eb:
    st.markdown("#### Estimated Bases")
    eb_bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, float("inf")]
    eb_labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]

    p1_eb_dist = pd.cut(p1["bb"]["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False).value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100
    p2_eb_dist = pd.cut(p2["bb"]["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False).value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100

    fig_eb = go.Figure()
    fig_eb.add_trace(go.Bar(x=eb_labels, y=p1_eb_dist.values, name=p1["name"], marker_color=p1_color, opacity=0.85))
    fig_eb.add_trace(go.Bar(x=eb_labels, y=p2_eb_dist.values, name=p2["name"], marker_color=p2_color, opacity=0.85))
    fig_eb.update_layout(
        barmode="group", xaxis_title="Estimated Bases", yaxis_title="% of Batted Balls",
        height=350, template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(t=30), dragmode=False,
    )
    st.plotly_chart(fig_eb, width="stretch", config=PLOTLY_CONFIG)


# =============================================================================
# SIDE-BY-SIDE SPRAY CHARTS
# =============================================================================

if "coord_x" in p1["bb"].columns and "coord_x" in p2["bb"].columns:
    st.divider()
    st.subheader("Spray Charts")
    st.caption(f"{season} Season")

    spray1, spray2 = st.columns(2)

    for col, p, color in [(spray1, p1, p1_color), (spray2, p2, p2_color)]:
        with col:
            st.markdown(f"#### {p['name']}")
            spray_data = p["bb"].dropna(subset=["coord_x", "coord_y"])
            if not spray_data.empty:
                HP_X, HP_Y = 125.42, 199.02
                FT = 0.5
                fig_spray = go.Figure()
                line_color = "rgba(0,0,0,0.15)"

                foul_len = 350 * FT
                for angle_deg in [-45, 45]:
                    rad = np.radians(angle_deg)
                    fig_spray.add_trace(go.Scatter(
                        x=[HP_X, HP_X + foul_len * np.sin(rad)],
                        y=[HP_Y, HP_Y - foul_len * np.cos(rad)],
                        mode="lines", line=dict(color=line_color, width=1.5),
                        showlegend=False, hoverinfo="skip",
                    ))

                arc_angles = np.linspace(-np.pi / 4, np.pi / 4, 60)
                arc_r = 95 * FT
                fig_spray.add_trace(go.Scatter(
                    x=HP_X + arc_r * np.sin(arc_angles),
                    y=HP_Y - arc_r * np.cos(arc_angles),
                    mode="lines", line=dict(color=line_color, width=1),
                    showlegend=False, hoverinfo="skip",
                ))

                b = 90 * FT * np.sin(np.pi / 4)
                bases_x = [HP_X, HP_X + b, HP_X, HP_X - b, HP_X]
                bases_y = [HP_Y, HP_Y - b, HP_Y - 2 * b, HP_Y - b, HP_Y]
                fig_spray.add_trace(go.Scatter(
                    x=bases_x, y=bases_y,
                    mode="lines", line=dict(color=line_color, width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                fig_spray.add_trace(go.Scatter(
                    x=spray_data["coord_x"], y=spray_data["coord_y"],
                    mode="markers",
                    marker=dict(color=spray_data["estimated_bases"], colorscale="RdYlGn", size=6,
                                colorbar=dict(title="Est.<br>Bases")),
                    customdata=np.stack([
                        spray_data["estimated_bases"],
                        spray_data["launch_speed"],
                        spray_data["actual_result"],
                    ], axis=-1),
                    hovertemplate=(
                        "Est. Bases: %{customdata[0]:.2f}<br>"
                        "Exit Velo: %{customdata[1]:.1f}<br>"
                        "Result: %{customdata[2]}<extra></extra>"
                    ),
                    showlegend=False,
                ))

                fig_spray.update_layout(
                    height=450, template="plotly_white",
                    xaxis=dict(visible=False, scaleanchor="y"),
                    yaxis=dict(visible=False, autorange="reversed"),
                    dragmode=False,
                )
                st.plotly_chart(fig_spray, width="stretch", config=PLOTLY_CONFIG_STATIC)

                # Spray direction breakdown
                if "spray_direction" in spray_data.columns:
                    dirs = spray_data["spray_direction"].value_counts(normalize=True) * 100
                    parts = [f"{d}: {dirs.get(d, 0):.0f}%" for d in ["Pull", "Center", "Oppo"]]
                    st.caption(" · ".join(parts))
            else:
                st.info("No spray chart data available.")


# =============================================================================
# COMBINED CONTACT TYPE TABLE
# =============================================================================

st.divider()
st.subheader("Contact Type Breakdown")
st.caption(f"{season} Season")

type_order = ["Ground Ball", "Line Drive", "Fly Ball", "Pop Up"]

# League averages
bb_df_typed = bb_df.copy()
bb_df_typed["bb_type"] = bb_df_typed["launch_angle"].apply(categorize_launch_angle)
lg_type = bb_df_typed.groupby("bb_type").agg(lg_avg_eb=("estimated_bases", "mean")).reset_index()


def _contact_stats(pdata, prefix):
    ts = pdata["bb"].groupby("bb_type").agg(
        count=("estimated_bases", "count"),
        avg_eb=("estimated_bases", "mean"),
    ).reset_index()
    ts["pct"] = (ts["count"] / ts["count"].sum() * 100).round(1)
    ts = ts.rename(columns={"count": f"{prefix} Count", "pct": f"{prefix} %", "avg_eb": f"{prefix} Avg EB"})
    return ts


t1 = _contact_stats(p1, "P1")
t2 = _contact_stats(p2, "P2")

merged = t1.merge(t2, on="bb_type", how="outer").merge(lg_type, on="bb_type", how="left")
merged["bb_type"] = pd.Categorical(merged["bb_type"], categories=type_order, ordered=True)
merged = merged.sort_values("bb_type").reset_index(drop=True)
merged = merged.rename(columns={"bb_type": "Type", "lg_avg_eb": "Lg Avg EB"})

ct_col_config = {
    "P1 Avg EB": st.column_config.NumberColumn(format="%.3f"),
    "P2 Avg EB": st.column_config.NumberColumn(format="%.3f"),
    "Lg Avg EB": st.column_config.NumberColumn(format="%.3f"),
    "P1 %": st.column_config.NumberColumn(format="%.1f%%"),
    "P2 %": st.column_config.NumberColumn(format="%.1f%%"),
}
# Rename P1/P2 to player last names for readability
p1_last = p1["name"].split()[-1]
p2_last = p2["name"].split()[-1]
merged = merged.rename(columns={
    "P1 Count": f"{p1_last} #", "P1 %": f"{p1_last} %", "P1 Avg EB": f"{p1_last} EB",
    "P2 Count": f"{p2_last} #", "P2 %": f"{p2_last} %", "P2 Avg EB": f"{p2_last} EB",
})
ct_col_config = {
    f"{p1_last} EB": st.column_config.NumberColumn(format="%.3f"),
    f"{p2_last} EB": st.column_config.NumberColumn(format="%.3f"),
    "Lg Avg EB": st.column_config.NumberColumn(format="%.3f"),
    f"{p1_last} %": st.column_config.NumberColumn(format="%.1f%%"),
    f"{p2_last} %": st.column_config.NumberColumn(format="%.1f%%"),
}

st.dataframe(merged, hide_index=True, width="stretch", column_config=ct_col_config)


# =============================================================================
# LUCK COMPARISON
# =============================================================================

st.divider()
st.subheader("Luck Comparison")
st.caption(f"{season} Season")

lc1, lc2 = st.columns(2)
for col, p, luck_val, luck_pct, actual_pct, expected_pct, color in [
    (lc1, p1, luck_1, luck_pct_1, actual_pct_1, expected_pct_1, p1_color),
    (lc2, p2, luck_2, luck_pct_2, actual_pct_2, expected_pct_2, p2_color),
]:
    with col:
        st.markdown(f"#### {p['name']}")
        m1, m2 = st.columns(2)
        m1.metric("Actual TB", f"{p['total_actual_tb']:.0f}")
        render_percentile_bar(actual_pct, container=m1)
        m2.metric("Expected TB", f"{p['total_expected_tb']:.1f}")
        render_percentile_bar(expected_pct, container=m2)
        m3, m4 = st.columns(2)
        delta_color = "normal" if luck_val >= 0 else "inverse"
        tier = luck_tier_label(luck_pct)
        delta_text = tier if tier else ("Lucky" if luck_val > 0 else "Unlucky")
        m3.metric("Net Lucky Bases", f"{luck_val:+.1f}", delta=delta_text, delta_color=delta_color)
        render_percentile_bar(luck_pct, container=m4)

# Overlaid cumulative luck chart
st.markdown("#### Luck Over Time")
st.caption("Cumulative actual minus expected total bases. Rising = lucky, falling = unlucky.")

fig_luck = go.Figure()

luck_last_points = []
for p, color, name in [(p1, p1_color, p1["name"]), (p2, p2_color, p2["name"])]:
    luck_ts = p["bb"].sort_values("date_parsed").copy()
    luck_ts["cum_luck"] = luck_ts["actual_tb"].cumsum() - luck_ts["estimated_bases"].cumsum()
    luck_ts["bb_num"] = range(1, len(luck_ts) + 1)
    fig_luck.add_trace(go.Scatter(
        x=luck_ts["bb_num"], y=luck_ts["cum_luck"],
        mode="lines", name=name,
        line=dict(color=color, width=2.5),
        hovertemplate="BB #%{x}<br>Cumulative Luck: %{y:.1f}<extra></extra>",
    ))
    luck_last_points.append((luck_ts["bb_num"].iloc[-1], luck_ts["cum_luck"].iloc[-1], name, color))

fig_luck.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

# Add last-name annotations at end of each line
for x, y, name, color in luck_last_points:
    last_name = name.split()[-1]
    fig_luck.add_annotation(
        x=x, y=y, text=last_name, showarrow=False,
        xshift=10, font=dict(color=color, size=11), xanchor="left",
    )

fig_luck.update_layout(
    xaxis_title="Batted Ball #", yaxis_title="Cumulative Luck (TB)",
    height=400, template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    dragmode=False,
)
st.plotly_chart(fig_luck, width="stretch", config=PLOTLY_CONFIG)


# =============================================================================
# FOOTER — LINKS TO INDIVIDUAL PROFILES
# =============================================================================

st.divider()
p1_display = f"{p1['name']} ({p1['team']})"
p2_display = f"{p2['name']} ({p2['team']})"
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(f"[View {p1['name']}'s full profile](Hitter_Profile?player={p1_display.replace(' ', '+')})")
with fc2:
    st.markdown(f"[View {p2['name']}'s full profile](Hitter_Profile?player={p2_display.replace(' ', '+')})")

render_home_link()
