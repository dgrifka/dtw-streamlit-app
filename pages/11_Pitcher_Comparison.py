"""
Pitcher Comparison Page

Side-by-side comparison of two pitchers: contact quality allowed, distributions,
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
    load_player_projections, load_rate_stat_projections, resolve_player_id,
    build_player_display_list, get_cached_radar_data,
)
from utils.team_mappings import get_team_color, get_team_logo_url
from utils.player_helpers import (
    normalize_name, build_headshot_url, categorize_launch_angle,
    is_barrel, is_barrel_vectorized,
    _ordinal, _percentile_color, render_percentile_bar, render_comparison_metric,
    luck_tier_label, safe_html, render_comparison_radar_chart,
    render_sticky_comparison_bar,
    render_comparison_grades, render_comparison_bars,
    TB_MAP, PLOTLY_CONFIG, PLOTLY_CONFIG_STATIC,
)
from utils.player_analytics import (
    get_player_radar_percentiles, PITCHER_ARCHETYPE_DESC,
    compute_player_grade, compute_projected_grade,
)
from utils.responsive import inject_responsive_css, render_home_link

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

available_seasons = get_available_batted_ball_seasons()
_default_bb_season = available_seasons[0] if available_seasons else pd.Timestamp.now().year
season = st.session_state.get("pcmp_bb_season", _default_bb_season)

bb_df = load_batted_balls(season)
if bb_df.empty:
    st.title("Pitcher Comparison")
    st.info(f"No batted ball data available for {season}.")
    st.stop()

if "pitcher" not in bb_df.columns:
    st.title("Pitcher Comparison")
    st.warning("Pitcher data not available in batted ball dataset.")
    st.stop()

metadata_df = load_player_metadata(season)
pa_rankings = load_player_evaluations_pa(season, "pitcher")

# Multi-season PA rankings for historical timeline
all_season_pa_rankings = load_all_season_pa_rankings("pitcher")

# Compute archetypes for hero badge display (cached)
_cmp_archetype_map = {}
_cmp_radar_df = get_cached_radar_data(season, player_type="pitcher", min_pa=30)
if not _cmp_radar_df.empty:
    for _, row in _cmp_radar_df[["player", "team", "archetype"]].iterrows():
        _cmp_archetype_map[f"{row['player']}|{row['team']}"] = row["archetype"]

st.title("Pitcher Comparison")

# =============================================================================
# PLAYER SELECTION
# =============================================================================

# Build deduplicated pitcher list (pitcher column, opponent = pitcher's team)
display_list, display_to_name, display_to_team, multi_id_names = build_player_display_list(
    bb_df, metadata_df, name_col="pitcher", team_col="opponent"
)

# Eligible pitchers (50+ BBs faced) for random selection
eligible_displays = [d for d in display_list
                     if bb_df[bb_df["pitcher"] == display_to_name[d]].shape[0] >= 50]
if not eligible_displays:
    eligible_displays = display_list

# Resolve defaults from query params or session state or random
qp1 = st.query_params.get("p1", "")
qp2 = st.query_params.get("p2", "")
if not qp1:
    qp1 = st.session_state.get("pitcher_comparison_last_p1", "")
if not qp2:
    qp2 = st.session_state.get("pitcher_comparison_last_p2", "")


def _resolve_index(query_val, exclude_display=None):
    """Resolve a query param to a display_list index."""
    if query_val and query_val in display_list:
        return display_list.index(query_val)
    if query_val:
        qn = normalize_name(query_val)
        fuzzy = [i for i, d in enumerate(display_list) if qn in normalize_name(d)]
        if fuzzy:
            return fuzzy[0]
    pool = [d for d in eligible_displays if d != exclude_display] or eligible_displays
    pick = random.choice(pool)
    if pick in display_list:
        return display_list.index(pick)
    return 0


default_idx1 = _resolve_index(qp1)
default_pick1 = display_list[default_idx1]
default_idx2 = _resolve_index(qp2, exclude_display=default_pick1)

# Layout: Player 1 | Swap | Player 2 | Shuffle Both
sel1, swap_col, sel2, shuffle_col = st.columns([3, 0.5, 3, 1], vertical_alignment="bottom")

with sel1:
    selected_p1 = st.selectbox("Pitcher 1", options=display_list, index=default_idx1, key="pcmp_p1")
with swap_col:
    if st.button("Swap", width="stretch"):
        st.query_params["p1"] = st.session_state.get("pcmp_p2", display_list[default_idx2])
        st.query_params["p2"] = st.session_state.get("pcmp_p1", display_list[default_idx1])
        st.rerun()
with sel2:
    selected_p2 = st.selectbox("Pitcher 2", options=display_list, index=default_idx2, key="pcmp_p2")
with shuffle_col:
    if st.button("Shuffle Both", width="stretch"):
        picks = random.sample(eligible_displays, min(2, len(eligible_displays)))
        st.query_params["p1"] = picks[0]
        st.query_params["p2"] = picks[1] if len(picks) > 1 else picks[0]
        st.rerun()

# Update query params for bookmarking + cache in session state
st.query_params["p1"] = selected_p1
st.query_params["p2"] = selected_p2
st.session_state["pitcher_comparison_last_p1"] = selected_p1
st.session_state["pitcher_comparison_last_p2"] = selected_p2

if selected_p1 == selected_p2:
    st.warning("Same pitcher selected for both sides. Select different pitchers to see a comparison.")


# =============================================================================
# RESOLVE PLAYER DATA
# =============================================================================

def resolve_player_data(display_label):
    """Resolve all data for a selected pitcher display label."""
    name = display_to_name.get(display_label, display_label)
    team = display_to_team.get(display_label)
    # Pitcher batted balls: pitcher column, opponent = pitcher's team
    pbb = bb_df[bb_df["pitcher"] == name].copy()
    if name in multi_id_names:
        pbb = pbb[pbb["opponent"] == team]
    if pbb.empty:
        return None
    pbb = pbb.sort_values("date_parsed")
    team_short = pbb["opponent"].iloc[-1]

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

    # Archetype lookup
    archetype = _cmp_archetype_map.get(f"{name}|{team_short}")

    return {
        "name": name, "team": team_short, "bb": pbb, "meta": meta,
        "player_id": pid, "ranking": ranking, "archetype": archetype,
        "primary_color": primary, "secondary_color": secondary,
        "avg_ev": pbb["launch_speed"].mean(),
        "avg_eb": pbb["estimated_bases"].mean(),
        "n_bb": len(pbb),
        "barrel_rate": pbb["is_barrel"].mean() * 100,
        "total_actual_tb": pbb["actual_tb"].sum(),
        "total_expected_tb": pbb["estimated_bases"].sum(),
        "n_pa": int(ranking["n_batted_balls"]) if ranking is not None and "n_batted_balls" in ranking.index else None,
    }


p1 = resolve_player_data(selected_p1)
p2 = resolve_player_data(selected_p2)

if p1 is None or p2 is None:
    st.warning("Could not load data for one or both pitchers.")
    st.stop()

# Handle same-team color collision: use secondary color for pitcher 2
p2_color = p2["primary_color"]
if p1["team"] == p2["team"]:
    p2_color = p2["secondary_color"]
    if p2_color == p2["primary_color"]:
        p2_color = "#E67E22"
p1_color = p1["primary_color"]


# =============================================================================
# HEAD-TO-HEAD HERO
# =============================================================================

st.divider()

# Sentinel for sticky bar IntersectionObserver
st.markdown('<div id="hero-section-sentinel"></div>', unsafe_allow_html=True)

hero1, hero2 = st.columns(2)

_hero_headshots = {}

for col, p, color in [(hero1, p1, p1_color), (hero2, p2, p2_color)]:
    with col:
        pos_str = ""
        age_str = ""
        throws_str = ""
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
            th = p["meta"].get("throw_hand", "")
            if th:
                throws_label = {"L": "Throws L", "R": "Throws R"}.get(th, f"Throws {th}")
                throws_str = f" | {throws_label}"

        pa_str = f" | {p['n_pa']:,} BF" if p.get("n_pa") else ""

        arch_html = ""
        if p.get("archetype"):
            arch_html = (
                f'<div style="margin-top:4px;">'
                f'<span style="display:inline-block; background:{color}; color:white; '
                f'padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.8rem; '
                f'letter-spacing:0.3px;">{p["archetype"]}</span>'
                f'</div>'
            )

        img_url = ""
        if p["player_id"]:
            img_url = build_headshot_url(p["player_id"])
        else:
            img_url = get_team_logo_url(p["team"]) or ""
        _hero_headshots[p["name"]] = img_url

        img_html = ""
        if img_url:
            img_html = (
                f'<img src="{img_url}" '
                f'style="width:100px; height:100px; object-fit:contain; border-radius:8px; flex-shrink:0;" '
                f'onerror="this.style.display=\'none\'">'
            )

        st.markdown(
            f'<div style="display:flex; align-items:center; gap:16px; '
            f'background:#f8f9fa; border-radius:10px; padding:16px; '
            f'border-left:4px solid {color};">'
            f'{img_html}'
            f'<div>'
            f'<div style="font-size:1.3rem; font-weight:700; margin-bottom:2px;">{safe_html(p["name"])}</div>'
            f'<div style="color:#4A5568; font-size:0.95rem;"><b>{safe_html(p["team"])}</b>{pos_str}{age_str}{throws_str}{pa_str}</div>'
            f'{arch_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

render_sticky_comparison_bar(
    name_1=p1["name"], team_1=p1["team"], color_1=p1_color,
    headshot_url_1=_hero_headshots.get(p1["name"], ""),
    name_2=p2["name"], team_2=p2["team"], color_2=p2_color,
    headshot_url_2=_hero_headshots.get(p2["name"], ""),
)


# =============================================================================
# OVERLAPPING RADAR CHART
# =============================================================================

if not _cmp_radar_df.empty:
    _radar_pcts_1 = get_player_radar_percentiles(_cmp_radar_df, p1["name"], p1["team"], "pitcher")
    _radar_pcts_2 = get_player_radar_percentiles(_cmp_radar_df, p2["name"], p2["team"], "pitcher")

    if _radar_pcts_1 and _radar_pcts_2:
        st.divider()
        _radar_fig = render_comparison_radar_chart(
            _radar_pcts_1, _radar_pcts_2,
            p1["name"], p2["name"],
            p1_color, p2_color,
        )
        st.plotly_chart(_radar_fig, width="stretch", config=PLOTLY_CONFIG)
    elif _radar_pcts_1:
        st.divider()
        from utils.player_helpers import render_radar_chart
        st.caption(f"Only {p1['name']} has enough data for the radar chart.")
        _radar_fig = render_radar_chart(_radar_pcts_1, p1_color)
        st.plotly_chart(_radar_fig, width="stretch", config=PLOTLY_CONFIG)
    elif _radar_pcts_2:
        st.divider()
        from utils.player_helpers import render_radar_chart
        st.caption(f"Only {p2['name']} has enough data for the radar chart.")
        _radar_fig = render_radar_chart(_radar_pcts_2, p2_color)
        st.plotly_chart(_radar_fig, width="stretch", config=PLOTLY_CONFIG)

    if _radar_pcts_1 or _radar_pcts_2:
        with st.expander("How does this work?"):
            archetype_lines = "\n".join(
                f"- **{name}**: {desc}" for name, desc in PITCHER_ARCHETYPE_DESC.items()
            )
            st.markdown(
                "**Radar Chart:** Shows how each pitcher compares to every pitcher with 30+ batters faced "
                "this season. Each spoke is a different skill, measured as a percentile (0 to 100). All axes "
                "are oriented so that bigger = better (e.g., \"Command\" = low walk rate, \"HR Prevention\" = "
                "low HR rate).\n\n"
                "**The 6 Axes:**\n"
                "- **Run Prevention** — Bayesian EB/PA allowed (inverted: lower allowed = higher percentile)\n"
                "- **Strikeout Ability** — Strikeout rate (K%)\n"
                "- **Command** — 1 minus walk rate (higher = fewer walks)\n"
                "- **HR Prevention** — 1 minus HR rate (higher = fewer HR allowed)\n"
                "- **Weak Contact** — 1 minus hard-hit rate (higher = softer contact induced)\n"
                "- **Ground Balls** — Ground ball rate (launch angle < 10 degrees)\n\n"
                "**Archetypes:** Pitchers are grouped by their radar shape using a clustering algorithm (K-Means). "
                "Pitchers in the same archetype tend to have similar strengths and weaknesses.\n\n"
                f"{archetype_lines}\n\n"
                "**Data Note:** The radar uses the best available estimate of each pitcher's true skill level. "
                "When preseason projections are available, they are combined with in-season performance using "
                "inverse-variance weighting for a more stable \"true talent\" estimate. Otherwise, the Bayesian "
                "posterior from the current season is used, which already shrinks small samples toward the league mean."
            )


# =============================================================================
# PROJECTION HELPER + BUILD COMPARISON OPTIONS
# =============================================================================

def _lookup_proj_row(proj_df, pdata):
    """Find a player's full projection row in a projection DataFrame."""
    pid = pdata.get("player_id")
    match = pd.DataFrame()
    if pid and "player_id" in proj_df.columns:
        match = proj_df[proj_df["player_id"] == pid]
    if match.empty:
        match = proj_df[proj_df["player"] == pdata["name"]]
    return match.iloc[0] if not match.empty else None

def _lookup_eval_row(pa_df, pdata):
    """Find a player's row in an evaluation rankings DataFrame."""
    pid = pdata.get("player_id")
    match = pd.DataFrame()
    if pid and "player_id" in pa_df.columns:
        match = pa_df[pa_df["player_id"] == pid]
    if match.empty:
        match = pa_df[pa_df["player"] == pdata["name"]]
    return match.iloc[0] if not match.empty else None

# Build list of seasons/projections where BOTH pitchers have data
_ebpa_options = []

# Projection years
for _proj_yr in range(season, season + 4):
    _pjdf = load_player_projections(_proj_yr, "pitcher")
    if _pjdf.empty:
        continue
    if _lookup_proj_row(_pjdf, p1) is not None and _lookup_proj_row(_pjdf, p2) is not None:
        _ebpa_options.append(f"{_proj_yr} Projected")

# Current season actuals
if p1["ranking"] is not None and p2["ranking"] is not None:
    _ebpa_options.append(f"{season} Actual")

# Past season actuals
for _s in sorted(all_season_pa_rankings.keys(), reverse=True):
    if _s == season:
        continue
    _pa_s = all_season_pa_rankings[_s]
    if _lookup_eval_row(_pa_s, p1) is not None and _lookup_eval_row(_pa_s, p2) is not None:
        _ebpa_options.append(f"{_s} Actual")

# =============================================================================
# HEAD-TO-HEAD
# =============================================================================

st.markdown("#### Head-to-Head")

# Season/projection selector
if len(_ebpa_options) > 1:
    _ebpa_choice = st.selectbox("Compare", options=_ebpa_options, index=0,
                                 key="pcmp_ebpa_view", label_visibility="collapsed")
else:
    _ebpa_choice = _ebpa_options[0] if _ebpa_options else f"{season} Actual"

_ebpa_year = int(_ebpa_choice.split()[0])
_ebpa_is_proj = "Projected" in _ebpa_choice

# Resolve EB/PA data for the selected view
_sel_eb1 = _sel_eb2 = None
_sel_pct1 = _sel_pct2 = None
_sel_ref_rankings = None

if _ebpa_is_proj:
    _sel_pjdf = load_player_projections(_ebpa_year, "pitcher")
    _sel_pr1 = _lookup_proj_row(_sel_pjdf, p1)
    _sel_pr2 = _lookup_proj_row(_sel_pjdf, p2)
    if _sel_pr1 is not None and _sel_pr2 is not None:
        _sel_eb1 = _sel_pr1["projected_eb_pa"]
        _sel_eb2 = _sel_pr2["projected_eb_pa"]
        # For pitchers, lower EB/PA = better, so invert percentile direction
        _sel_pct1 = (_sel_pjdf["projected_eb_pa"] > _sel_eb1).mean() * 100
        _sel_pct2 = (_sel_pjdf["projected_eb_pa"] > _sel_eb2).mean() * 100
else:
    if _ebpa_year == season and p1["ranking"] is not None and p2["ranking"] is not None:
        _sel_ref_rankings = pa_rankings
        _sel_eb1 = p1["ranking"]["posterior_mean"]
        _sel_eb2 = p2["ranking"]["posterior_mean"]
    elif _ebpa_year in all_season_pa_rankings:
        _sel_ref_rankings = all_season_pa_rankings[_ebpa_year]
        _er1 = _lookup_eval_row(_sel_ref_rankings, p1)
        _er2 = _lookup_eval_row(_sel_ref_rankings, p2)
        if _er1 is not None and _er2 is not None:
            _sel_eb1 = _er1["posterior_mean"]
            _sel_eb2 = _er2["posterior_mean"]
    if _sel_eb1 is not None and _sel_ref_rankings is not None:
        # Invert for pitchers: lower EB/PA allowed = better
        _sel_pct1 = (_sel_ref_rankings["posterior_mean"] > _sel_eb1).mean() * 100
        _sel_pct2 = (_sel_ref_rankings["posterior_mean"] > _sel_eb2).mean() * 100

# Compute percentiles for EV, barrel rate, avg EB (pitcher = inverted)
league_pcts = compute_league_percentiles(season, "pitcher")
if league_pcts:
    # For pitchers: lower values = better, invert percentiles
    ev_pct_1 = (league_pcts["ev_by_player"] > p1["avg_ev"]).mean() * 100
    ev_pct_2 = (league_pcts["ev_by_player"] > p2["avg_ev"]).mean() * 100
    barrel_pct_1 = (league_pcts["barrel_rates"] > p1["barrel_rate"]).mean() * 100
    barrel_pct_2 = (league_pcts["barrel_rates"] > p2["barrel_rate"]).mean() * 100
    avg_eb_pct_1 = (league_pcts["avg_eb_by_player"] > p1["avg_eb"]).mean() * 100
    avg_eb_pct_2 = (league_pcts["avg_eb_by_player"] > p2["avg_eb"]).mean() * 100
else:
    ev_pct_1 = ev_pct_2 = barrel_pct_1 = barrel_pct_2 = avg_eb_pct_1 = avg_eb_pct_2 = 50

# Luck for pitchers: positive = pitcher got lucky (expected > actual)
luck_1 = p1["total_expected_tb"] - p1["total_actual_tb"]
luck_2 = p2["total_expected_tb"] - p2["total_actual_tb"]

# Luck percentiles
luck_pct_1 = luck_pct_2 = None
actual_pct_1 = actual_pct_2 = None
expected_pct_1 = expected_pct_2 = None
if len(bb_df) > 1000 and "pitcher" in bb_df.columns:
    all_luck = bb_df.copy()
    all_luck["actual_tb"] = all_luck["actual_result"].map(TB_MAP).fillna(0)
    pitcher_luck = all_luck.groupby("pitcher").agg(
        actual=("actual_tb", "sum"), expected=("estimated_bases", "sum"),
        n=("estimated_bases", "count"),
    )
    pitcher_luck = pitcher_luck[pitcher_luck["n"] >= 30]
    pitcher_luck["luck"] = pitcher_luck["expected"] - pitcher_luck["actual"]
    if p1["name"] in pitcher_luck.index:
        luck_pct_1 = (pitcher_luck["luck"] < luck_1).mean() * 100
        # For pitchers: lower actual TB allowed = better
        actual_pct_1 = (pitcher_luck["actual"] > p1["total_actual_tb"]).mean() * 100
        expected_pct_1 = (pitcher_luck["expected"] > p1["total_expected_tb"]).mean() * 100
    if p2["name"] in pitcher_luck.index:
        luck_pct_2 = (pitcher_luck["luck"] < luck_2).mean() * 100
        actual_pct_2 = (pitcher_luck["actual"] > p2["total_actual_tb"]).mean() * 100
        expected_pct_2 = (pitcher_luck["expected"] > p2["total_expected_tb"]).mean() * 100

# --- Grade badges ---
_score_1 = _score_2 = None
_arch_1 = p1.get("archetype", "Unknown") or "Unknown"
_arch_2 = p2.get("archetype", "Unknown") or "Unknown"
_grade_caption = None

# Preseason detection: evaluation season < current year and projections exist
_current_year = pd.Timestamp.now().year
_cmp_proj_df = load_player_projections(season + 1, "pitcher") if season < _current_year else pd.DataFrame()
_cmp_proj_active = _cmp_proj_df[_cmp_proj_df["p_active_next_season"] > 0.3] if not _cmp_proj_df.empty and "p_active_next_season" in _cmp_proj_df.columns else _cmp_proj_df
_cmp_is_preseason = season < _current_year and not _cmp_proj_active.empty

if _cmp_is_preseason:
    _pg1 = compute_projected_grade(p1["name"], p1["team"], _cmp_proj_active, player_type="pitcher")
    _pg2 = compute_projected_grade(p2["name"], p2["team"], _cmp_proj_active, player_type="pitcher")
    if _pg1 is not None:
        _score_1 = _pg1
    if _pg2 is not None:
        _score_2 = _pg2
    if _pg1 is not None or _pg2 is not None:
        _target_season = int(_cmp_proj_df["target_season"].iloc[0]) if "target_season" in _cmp_proj_df.columns else _current_year
        _grade_caption = f"Grades based on {_target_season} projections"

# Fill any missing scores from radar percentiles
if not _cmp_radar_df.empty:
    _rp1 = get_player_radar_percentiles(_cmp_radar_df, p1["name"], p1["team"], "pitcher")
    _rp2 = get_player_radar_percentiles(_cmp_radar_df, p2["name"], p2["team"], "pitcher")
    if _score_1 is None and _rp1:
        _score_1 = compute_player_grade(_rp1, player_type="pitcher")
        if _score_1 is None:
            _score_1 = sum(_rp1.values()) / len(_rp1)
    if _score_2 is None and _rp2:
        _score_2 = compute_player_grade(_rp2, player_type="pitcher")
        if _score_2 is None:
            _score_2 = sum(_rp2.values()) / len(_rp2)

if _score_1 is not None and _score_2 is not None:
    render_comparison_grades(
        p1["name"], _score_1, _arch_1, p1_color,
        p2["name"], _score_2, _arch_2, p2_color,
    )
    if _grade_caption:
        st.caption(_grade_caption)

# --- Build comparison bar metrics ---
_h2h_metrics = []

# Season actuals — for pitchers, lower = better, percentiles are already inverted above
_h2h_metrics.append({"label": "EV Allowed", "v1": f"{p1['avg_ev']:.1f} mph", "v2": f"{p2['avg_ev']:.1f} mph",
                      "pct1": ev_pct_1, "pct2": ev_pct_2, "num1": p1["avg_ev"], "num2": p2["avg_ev"],
                      "higher_better": False})
_h2h_metrics.append({"label": "Barrel Rate", "v1": f"{p1['barrel_rate']:.1f}%", "v2": f"{p2['barrel_rate']:.1f}%",
                      "pct1": barrel_pct_1, "pct2": barrel_pct_2, "num1": p1["barrel_rate"], "num2": p2["barrel_rate"],
                      "higher_better": False})
_h2h_metrics.append({"label": "Avg EB/BB Allowed", "v1": f"{p1['avg_eb']:.3f}", "v2": f"{p2['avg_eb']:.3f}",
                      "pct1": avg_eb_pct_1, "pct2": avg_eb_pct_2, "num1": p1["avg_eb"], "num2": p2["avg_eb"],
                      "higher_better": False})

# EB/PA (Bayesian or projected) — lower = better for pitchers
if _sel_eb1 is not None and _sel_eb2 is not None:
    _ebpa_label = f"EB/PA ({_ebpa_year})" if _ebpa_year != season else "EB/PA Allowed"
    if _ebpa_is_proj:
        _ebpa_label = f"Proj. EB/PA ({_ebpa_year})"
    _h2h_metrics.append({"label": _ebpa_label, "v1": f"{_sel_eb1:.3f}", "v2": f"{_sel_eb2:.3f}",
                          "pct1": _sel_pct1, "pct2": _sel_pct2, "num1": _sel_eb1, "num2": _sel_eb2,
                          "higher_better": False})

# True talent
if p1["ranking"] is not None and p2["ranking"] is not None:
    _tt1 = p1["ranking"].get("true_talent_eb_pa") if "true_talent_eb_pa" in p1["ranking"].index else None
    _tt2 = p2["ranking"].get("true_talent_eb_pa") if "true_talent_eb_pa" in p2["ranking"].index else None
    if _tt1 is not None and _tt2 is not None and pd.notna(_tt1) and pd.notna(_tt2):
        _h2h_metrics.append({"label": "True Talent EB/PA", "v1": f"{_tt1:.3f}", "v2": f"{_tt2:.3f}",
                              "pct1": None, "pct2": None, "num1": _tt1, "num2": _tt2,
                              "higher_better": False})

# Rate stats (K%, BB%, HR%) — pitchers: K% higher = better, BB% lower = better, HR% lower = better
if p1["ranking"] is not None and p2["ranking"] is not None:
    _r1, _r2 = p1["ranking"], p2["ranking"]
    for _rc, _rlabel, _higher_better in [("k_rate_posterior", "K Rate", True), ("bb_rate_posterior", "BB Rate", False), ("hr_rate_posterior", "HR Rate", False)]:
        if _rc in _r1.index and _rc in _r2.index and pd.notna(_r1.get(_rc)) and pd.notna(_r2.get(_rc)):
            if _higher_better:
                # K%: higher = better for pitchers
                _rpct1 = (_r1[_rc] > pa_rankings[_rc].dropna()).mean() * 100 if _rc in pa_rankings.columns else None
                _rpct2 = (_r2[_rc] > pa_rankings[_rc].dropna()).mean() * 100 if _rc in pa_rankings.columns else None
            else:
                # BB%, HR%: lower = better for pitchers, invert percentile
                _rpct1 = (_r1[_rc] < pa_rankings[_rc].dropna()).mean() * 100 if _rc in pa_rankings.columns else None
                _rpct2 = (_r2[_rc] < pa_rankings[_rc].dropna()).mean() * 100 if _rc in pa_rankings.columns else None
            _h2h_metrics.append({"label": _rlabel, "v1": f"{_r1[_rc]*100:.1f}%", "v2": f"{_r2[_rc]*100:.1f}%",
                                  "pct1": _rpct1, "pct2": _rpct2, "num1": _r1[_rc], "num2": _r2[_rc],
                                  "higher_better": _higher_better})

# Traditional stats — no percentile bars, just values with winner bolded
if p1["ranking"] is not None and p2["ranking"] is not None:
    _r1, _r2 = p1["ranking"], p2["ranking"]
    trad_stats = [
        ("ERA", "era", False, ".2f"),
        ("WHIP", "whip", False, ".2f"),
        ("IP", "innings_pitched", True, ".1f"),
    ]
    for label, col, higher_better, fmt in trad_stats:
        if col and col in _r1.index and col in _r2.index:
            v1 = _r1.get(col)
            v2 = _r2.get(col)
            if pd.notna(v1) and pd.notna(v2):
                _pct1, _pct2 = None, None
                if not pa_rankings.empty and col in pa_rankings.columns:
                    league_vals = pa_rankings[col].dropna()
                    if len(league_vals) >= 10:
                        if higher_better:
                            _pct1 = (v1 > league_vals).mean() * 100
                            _pct2 = (v2 > league_vals).mean() * 100
                        else:
                            _pct1 = (v1 < league_vals).mean() * 100
                            _pct2 = (v2 < league_vals).mean() * 100
                _h2h_metrics.append({"label": label, "v1": format(v1, fmt), "v2": format(v2, fmt),
                                      "pct1": _pct1, "pct2": _pct2, "num1": v1, "num2": v2,
                                      "higher_better": higher_better})

render_comparison_bars(_h2h_metrics, p1["name"], p1_color, p2["name"], p2_color)


# =============================================================================
# BAYESIAN EB/PA COMPARISON BAR
# =============================================================================

if _sel_eb1 is not None and _sel_eb2 is not None:
    st.divider()
    st.subheader("Est. Bases / PA Allowed")
    st.caption(_ebpa_choice)

    _bar_valid = False

    if _ebpa_is_proj:
        _sel_pjdf = load_player_projections(_ebpa_year, "pitcher")
        _pr1 = _lookup_proj_row(_sel_pjdf, p1)
        _pr2 = _lookup_proj_row(_sel_pjdf, p2)
        if _pr1 is not None and _pr2 is not None:
            eb1 = _pr1["projected_eb_pa"]
            hdi1_low = _pr1["projected_hdi_low"]
            hdi1_high = _pr1["projected_hdi_high"]
            eb2 = _pr2["projected_eb_pa"]
            hdi2_low = _pr2["projected_hdi_low"]
            hdi2_high = _pr2["projected_hdi_high"]

            # For pitchers, "best" = lowest EB/PA
            best_idx_p = _sel_pjdf["projected_eb_pa"].idxmin()
            best_row_p = _sel_pjdf.loc[best_idx_p]
            best_eb = best_row_p["projected_eb_pa"]
            best_hdi_low = best_row_p["projected_hdi_low"]
            best_hdi_high = best_row_p["projected_hdi_high"]
            best_name = best_row_p["player"]
            best_team = best_row_p.get("team", "")
            best_color, _ = get_team_color(best_team) if best_team else ("#DAA520", "#DAA520")

            lg_mean = _sel_pjdf["projected_eb_pa"].mean()
            lg_sd = _sel_pjdf["projected_eb_pa"].std()
            lg_low = lg_mean - lg_sd
            lg_high = lg_mean + lg_sd
            _bar_valid = True
    else:
        _bar_ref = pa_rankings if _ebpa_year == season else all_season_pa_rankings.get(_ebpa_year)
        if _bar_ref is not None:
            _er1 = _lookup_eval_row(_bar_ref, p1)
            _er2 = _lookup_eval_row(_bar_ref, p2)
            if _er1 is not None and _er2 is not None:
                eb1 = _er1["posterior_mean"]
                hdi1_low = _er1["hdi_low"]
                hdi1_high = _er1["hdi_high"]
                eb2 = _er2["posterior_mean"]
                hdi2_low = _er2["hdi_low"]
                hdi2_high = _er2["hdi_high"]

                # For pitchers, "best" = lowest
                best_idx_e = _bar_ref["posterior_mean"].idxmin()
                best_row_e = _bar_ref.loc[best_idx_e]
                best_eb = best_row_e["posterior_mean"]
                best_hdi_low = best_row_e["hdi_low"]
                best_hdi_high = best_row_e["hdi_high"]
                best_name = best_row_e["player"]
                best_team = best_row_e.get("team", "")
                best_color, _ = get_team_color(best_team) if best_team else ("#DAA520", "#DAA520")

                lg_mean = _bar_ref["posterior_mean"].mean()
                lg_sd = _bar_ref["posterior_mean"].std()
                lg_low = lg_mean - lg_sd
                lg_high = lg_mean + lg_sd
                _bar_valid = True

    if _bar_valid:
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
        _caption = "Projected EB/PA allowed with uncertainty ranges. Lower = better." if _ebpa_is_proj else "Estimated true EB/PA allowed with uncertainty ranges. Lower = better."
        st.caption(_caption)


# =============================================================================
# HISTORICAL EB/PA TIMELINE + RATE STAT TABS
# =============================================================================

def _add_player_annotations(fig, anno_pts):
    """Add collision-aware last-name annotations to a comparison chart."""
    if len(anno_pts) == 2:
        all_y = []
        for trace in fig.data:
            if hasattr(trace, "y") and trace.y is not None:
                all_y.extend([v for v in trace.y if v is not None])
        y_range = (max(all_y) - min(all_y)) if all_y else 1
        threshold = y_range * 0.06

        a, b = anno_pts[0], anno_pts[1]
        close = abs(a["y"] - b["y"]) < threshold
        for i, pt in enumerate(anno_pts):
            yshift = 0
            if close:
                yshift = 12 if i == 0 else -12
            last_name = pt["name"].split()[-1]
            fig.add_annotation(
                x=pt["x"], y=pt["y"],
                text=last_name, showarrow=False,
                xshift=10, yshift=yshift,
                font=dict(color=pt["color"], size=11),
                xanchor="left",
            )
    else:
        for pt in anno_pts:
            last_name = pt["name"].split()[-1]
            fig.add_annotation(
                x=pt["x"], y=pt["y"],
                text=last_name, showarrow=False,
                xshift=10, font=dict(color=pt["color"], size=11),
                xanchor="left",
            )


if len(all_season_pa_rankings) > 0 and (p1["player_id"] is not None or p2["player_id"] is not None):
    tl_data = {1: [], 2: []}
    rate_tl_data = {1: [], 2: []}
    league_avg_data = []
    rate_league_avg = []
    best_player_data = []

    for s in sorted(all_season_pa_rankings.keys()):
        pa_df = all_season_pa_rankings[s]

        lg_mean_s = pa_df["posterior_mean"].mean()
        league_avg_data.append({"season": s, "value": lg_mean_s})

        # Rate stat league averages
        _rate_lg = {"season": s}
        for _rc in ["k_rate_posterior", "bb_rate_posterior", "hr_rate_posterior"]:
            if _rc in pa_df.columns:
                _rate_lg[_rc] = pa_df[_rc].dropna().mean()
        rate_league_avg.append(_rate_lg)

        # For pitchers, "best" = lowest EB/PA
        best_idx_s = pa_df["posterior_mean"].idxmin()
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
                # Rate stats
                _rate_row = {"season": s}
                for _rc in ["k_rate_posterior", "k_rate_hdi_low", "k_rate_hdi_high",
                             "bb_rate_posterior", "bb_rate_hdi_low", "bb_rate_hdi_high",
                             "hr_rate_posterior", "hr_rate_hdi_low", "hr_rate_hdi_high"]:
                    if _rc in row.index and pd.notna(row.get(_rc)):
                        _rate_row[_rc] = row[_rc]
                rate_tl_data[pnum].append(_rate_row)

    _has_rate_timeline = any("k_rate_posterior" in t for t in rate_tl_data[1]) or any("k_rate_posterior" in t for t in rate_tl_data[2])

    if tl_data[1] or tl_data[2]:
        st.divider()
        st.subheader("Season History")

        lg_df = pd.DataFrame(league_avg_data)
        best_df = pd.DataFrame(best_player_data)
        rate_lg_df = pd.DataFrame(rate_league_avg)

        all_player_seasons = [d["season"] for d in tl_data[1]] + [d["season"] for d in tl_data[2]]
        if all_player_seasons:
            min_s, max_s = min(all_player_seasons), max(all_player_seasons)
            lg_df = lg_df[(lg_df["season"] >= min_s) & (lg_df["season"] <= max_s)]
            best_df = best_df[(best_df["season"] >= min_s) & (best_df["season"] <= max_s)]
            rate_lg_df = rate_lg_df[(rate_lg_df["season"] >= min_s) & (rate_lg_df["season"] <= max_s)]

        # --- Projections (shared across tabs) ---
        proj_data = {}
        for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
            if pdata["player_id"] is None:
                continue
            points = []
            for proj_season in range(max_s + 1, max_s + 4):
                proj_df = load_player_projections(proj_season, "pitcher")
                if proj_df.empty:
                    continue
                proj_match = proj_df[proj_df["player_id"] == pdata["player_id"]] if "player_id" in proj_df.columns else pd.DataFrame()
                if proj_match.empty:
                    proj_match = proj_df[proj_df["player"] == pdata["name"]]
                if not proj_match.empty:
                    points.append(proj_match.iloc[0].to_dict() | {"season": proj_season})
            if points:
                proj_data[pnum] = (points, color, pdata)

        # --- Rate stat comparison chart builder (pitcher directions) ---
        def _build_comparison_rate_chart(rate_prefix, y_label, higher_is_better):
            mean_col = f"{rate_prefix}_posterior"
            lo_col = f"{rate_prefix}_hdi_low"
            hi_col = f"{rate_prefix}_hdi_high"

            has_data = False
            for pnum in [1, 2]:
                if any(mean_col in t for t in rate_tl_data[pnum]):
                    has_data = True
            if not has_data:
                return None

            fig = go.Figure()

            # League average markers
            if not rate_lg_df.empty and mean_col in rate_lg_df.columns:
                lg_vals = rate_lg_df.dropna(subset=[mean_col])
                if not lg_vals.empty:
                    fig.add_trace(go.Scatter(
                        x=lg_vals["season"], y=lg_vals[mean_col] * 100,
                        mode="markers", name="Lg Avg",
                        marker=dict(color="rgba(160,160,160,0.8)", size=11),
                        hovertemplate="Season: %{x}<br>Lg Avg: %{y:.1f}%<extra></extra>",
                    ))

            # Player lines with HDI
            for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
                rtl = [t for t in rate_tl_data[pnum] if mean_col in t]
                if not rtl:
                    continue
                vals = pd.DataFrame(rtl)
                c = color.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

                if len(vals) == 1:
                    row_r = vals.iloc[0]
                    err_hi = [(row_r[hi_col] - row_r[mean_col]) * 100] if hi_col in row_r.index and pd.notna(row_r.get(hi_col)) else None
                    err_lo = [(row_r[mean_col] - row_r[lo_col]) * 100] if lo_col in row_r.index and pd.notna(row_r.get(lo_col)) else None
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[mean_col] * 100,
                        mode="markers", name=pdata["name"],
                        marker=dict(color=color, size=12),
                        error_y=dict(
                            type="data", array=err_hi, arrayminus=err_lo,
                            color=f"rgba({r},{g},{b},0.5)", thickness=2, width=6,
                        ) if err_hi else None,
                        hovertemplate=f"Season: %{{x}}<br>{y_label}: %{{y:.1f}}%<extra></extra>",
                    ))
                else:
                    if hi_col in vals.columns and lo_col in vals.columns:
                        fig.add_trace(go.Scatter(
                            x=vals["season"], y=vals[hi_col] * 100,
                            mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip",
                        ))
                        fig.add_trace(go.Scatter(
                            x=vals["season"], y=vals[lo_col] * 100,
                            mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.15)",
                            showlegend=False, hoverinfo="skip",
                        ))
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[mean_col] * 100,
                        mode="lines+markers", name=pdata["name"],
                        line=dict(color=color, width=2.5),
                        marker=dict(color=color, size=12),
                        hovertemplate=f"Season: %{{x}}<br>{y_label}: %{{y:.1f}}%<extra></extra>",
                    ))

            # Rate stat projections
            rate_x_max = max_s
            rate_proj_data = {}
            proj_col = f"projected_{rate_prefix}"
            proj_lo_col = f"projected_{rate_prefix}_hdi_low"
            proj_hi_col = f"projected_{rate_prefix}_hdi_high"
            for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
                if pdata["player_id"] is None:
                    continue
                rp_points = []
                for proj_s in range(max_s + 1, max_s + 4):
                    rp_df = load_rate_stat_projections(proj_s, "pitcher", rate_prefix)
                    if rp_df.empty or proj_col not in rp_df.columns:
                        continue
                    rp_match = rp_df[rp_df["player_id"] == pdata["player_id"]] if "player_id" in rp_df.columns else pd.DataFrame()
                    if rp_match.empty:
                        rp_match = rp_df[rp_df["player"] == pdata["name"]]
                    if not rp_match.empty:
                        rp_points.append(rp_match.iloc[0].to_dict() | {"season": proj_s})
                if rp_points:
                    rate_proj_data[pnum] = (rp_points, color, pdata)

            if rate_proj_data:
                all_rp_seasons = [p["season"] for pts, _, _ in rate_proj_data.values() for p in pts]
                rate_x_max = max(all_rp_seasons)
                fig.add_vrect(
                    x0=max_s + 0.5, x1=max(all_rp_seasons) + 0.5,
                    fillcolor="rgba(180,180,220,0.10)", line_width=0, layer="below",
                )
                fig.add_annotation(
                    x=(max_s + 0.5 + max(all_rp_seasons) + 0.5) / 2,
                    y=1.0, yref="paper", yanchor="bottom",
                    text="Projected", showarrow=False,
                    font=dict(size=13, color="rgba(120,120,160,0.7)"),
                )

                for pnum, (rp_points, color, pdata) in rate_proj_data.items():
                    c = color.lstrip("#")
                    pr, pg, pb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                    rp_seasons = [p["season"] for p in rp_points]
                    rp_vals = [p[proj_col] * 100 for p in rp_points]
                    rp_hi = [p.get(proj_hi_col, p[proj_col]) * 100 for p in rp_points]
                    rp_lo = [p.get(proj_lo_col, p[proj_col]) * 100 for p in rp_points]

                    fig.add_trace(go.Scatter(
                        x=rp_seasons, y=rp_hi, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=rp_seasons, y=rp_lo, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=f"rgba({pr},{pg},{pb},0.10)",
                        showlegend=False, hoverinfo="skip",
                    ))

                    rtl = [t for t in rate_tl_data[pnum] if mean_col in t]
                    if rtl:
                        last_actual_val = rtl[-1][mean_col] * 100
                        last_actual_season = rtl[-1]["season"]
                        fig.add_trace(go.Scatter(
                            x=[last_actual_season, rp_seasons[0]],
                            y=[last_actual_val, rp_vals[0]],
                            mode="lines",
                            line=dict(color=f"rgba({pr},{pg},{pb},0.4)", width=1.5, dash="dot"),
                            showlegend=False, hoverinfo="skip",
                        ))

                    if len(rp_seasons) > 1:
                        fig.add_trace(go.Scatter(
                            x=rp_seasons, y=rp_vals, mode="lines",
                            line=dict(color=f"rgba({pr},{pg},{pb},0.5)", width=2, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        ))

                    fig.add_trace(go.Scatter(
                        x=rp_seasons, y=rp_vals, mode="markers",
                        showlegend=False,
                        marker=dict(color="rgba(255,255,255,0)", size=12, symbol="diamond-open",
                                    line=dict(width=2.5, color=color)),
                        customdata=[
                            [p.get("aging_effect", 0),
                             p.get(proj_lo_col, p[proj_col]) * 100,
                             p.get(proj_hi_col, p[proj_col]) * 100]
                            for p in rp_points
                        ],
                        hovertemplate=(
                            "Projection %{x}<br>"
                            f"{y_label}: %{{y:.1f}}%<br>"
                            "89% HDI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
                            "Aging effect: %{customdata[0]:+.4f}<extra></extra>"
                        ),
                    ))

            # Player name annotations
            anno_pts = []
            for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
                if pnum in rate_proj_data:
                    rp_points, _, _ = rate_proj_data[pnum]
                    last_p = rp_points[-1]
                    anno_pts.append({"name": pdata["name"], "x": last_p["season"],
                                     "y": last_p[proj_col] * 100, "color": color})
                else:
                    rtl = [t for t in rate_tl_data[pnum] if mean_col in t]
                    if rtl:
                        last_t = rtl[-1]
                        anno_pts.append({"name": pdata["name"], "x": last_t["season"],
                                         "y": last_t[mean_col] * 100, "color": color})
            _add_player_annotations(fig, anno_pts)

            all_tick_s = list(range(min_s, rate_x_max + 1))
            direction = "" if higher_is_better else " (lower is better)"
            fig.update_layout(
                xaxis=dict(title="Season", title_font_size=14, tickfont_size=13,
                           tickvals=all_tick_s, ticktext=[str(s) for s in all_tick_s],
                           range=[min(all_tick_s) - 0.5, max(all_tick_s) + 0.5]),
                yaxis=dict(title=f"{y_label}{direction}", title_font_size=14,
                           tickfont_size=13, ticksuffix="%"),
                height=400, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, font=dict(size=13)),
                dragmode=False,
            )
            # Direction indicator annotation
            if higher_is_better:
                fig.add_annotation(
                    x=0.01, y=0.98, xref="paper", yref="paper",
                    text="↑ Higher = Better", showarrow=False,
                    font=dict(size=12, color="rgba(40, 140, 70, 0.7)"),
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)",
                )
            else:
                fig.add_annotation(
                    x=0.01, y=0.98, xref="paper", yref="paper",
                    text="↓ Lower = Better", showarrow=False,
                    font=dict(size=12, color="rgba(180, 60, 60, 0.7)"),
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)",
                )
            return fig

        # --- Lazy tab rendering via segmented control ---
        if _has_rate_timeline:
            _history_view = st.segmented_control(
                "View", ["EB/PA", "K%", "BB%", "HR%"], default="EB/PA",
                key="pcmp_history_view",
            )
        else:
            _history_view = "EB/PA"

        if _history_view == "EB/PA":
            if _has_rate_timeline:
                st.caption("Bayesian model estimate with projections")

            fig_tl = go.Figure()

            fig_tl.add_trace(go.Scatter(
                x=lg_df["season"], y=lg_df["value"],
                mode="markers", name="Lg Avg",
                marker=dict(color="rgba(160,160,160,0.8)", size=11),
                hovertemplate="Season: %{x}<br>Lg Avg: %{y:.3f}<extra></extra>",
            ))

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
                fig_tl.data[-len(best_df)].name = "Best Pitcher"

            for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
                if not tl_data[pnum]:
                    continue
                tl_df = pd.DataFrame(tl_data[pnum])

                c = color.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

                if len(tl_df) == 1:
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

            # Multi-year projections
            x_max = max_s

            if proj_data:
                all_proj_seasons = [p["season"] for pts, _, _ in proj_data.values() for p in pts]
                x_max = max(all_proj_seasons)

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

                    if tl_data[pnum]:
                        last_actual_pt = tl_data[pnum][-1]
                        fig_tl.add_trace(go.Scatter(
                            x=[last_actual_pt["season"], p_seasons[0]],
                            y=[last_actual_pt["value"], p_values[0]],
                            mode="lines",
                            line=dict(color=f"rgba({pr},{pg},{pb},0.4)", width=1.5, dash="dot"),
                            showlegend=False, hoverinfo="skip",
                        ))

                    if len(p_seasons) > 1:
                        fig_tl.add_trace(go.Scatter(
                            x=p_seasons, y=p_values,
                            mode="lines",
                            line=dict(color=f"rgba({pr},{pg},{pb},0.5)", width=2, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        ))

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

            # Player name annotations — prefer projection endpoint, fallback to last actual
            anno_pts = []
            for pnum, pdata, color in [(1, p1, p1_color), (2, p2, p2_color)]:
                if pnum in proj_data:
                    points, _, _ = proj_data[pnum]
                    last_p = points[-1]
                    anno_pts.append({"name": pdata["name"], "x": last_p["season"],
                                     "y": last_p["projected_eb_pa"], "color": color})
                elif tl_data[pnum]:
                    last_pt = tl_data[pnum][-1]
                    anno_pts.append({"name": pdata["name"], "x": last_pt["season"],
                                     "y": last_pt["value"], "color": color})
            _add_player_annotations(fig_tl, anno_pts)

            fig_tl.update_layout(
                xaxis=dict(title="Season", title_font_size=14, tickfont_size=13, dtick=1, tickformat="d",
                           range=[min_s - 0.5, x_max + 0.5]),
                yaxis=dict(title="Est. Bases per PA Allowed  (↓ better)", title_font_size=14, tickfont_size=13),
                height=400, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
                dragmode=False,
            )
            fig_tl.add_annotation(
                x=0.01, y=0.98, xref="paper", yref="paper",
                text="↓ Lower = Better", showarrow=False,
                font=dict(size=12, color="rgba(180, 60, 60, 0.7)"),
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
            )
            st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CONFIG)

        # --- Rate stat views (lazy — only rendered when selected) ---
        if _has_rate_timeline and _history_view == "K%":
            st.caption("Bayesian strikeout rate with 89% credible interval and projections")
            _fig_k = _build_comparison_rate_chart("k_rate", "K%", higher_is_better=True)
            if _fig_k:
                st.plotly_chart(_fig_k, width="stretch", config=PLOTLY_CONFIG)

        if _has_rate_timeline and _history_view == "BB%":
            st.caption("Bayesian walk rate with 89% credible interval and projections")
            _fig_bb = _build_comparison_rate_chart("bb_rate", "BB%", higher_is_better=False)
            if _fig_bb:
                st.plotly_chart(_fig_bb, width="stretch", config=PLOTLY_CONFIG)

        if _has_rate_timeline and _history_view == "HR%":
            st.caption("Bayesian home run rate with 89% credible interval and projections")
            _fig_hr = _build_comparison_rate_chart("hr_rate", "HR%", higher_is_better=False)
            if _fig_hr:
                st.plotly_chart(_fig_hr, width="stretch", config=PLOTLY_CONFIG)


# =============================================================================
# OVERLAID DISTRIBUTION CHARTS
# =============================================================================

st.divider()
_bb_banner_cols = st.columns([3, 1])
with _bb_banner_cols[0]:
    st.markdown(
        '<span style="font-size:0.85rem; color:#4A5568;">'
        'The sections below use batted ball data from the selected season.</span>',
        unsafe_allow_html=True,
    )
with _bb_banner_cols[1]:
    if available_seasons:
        season = st.selectbox("Batted Ball Season", options=available_seasons, index=0,
                              key="pcmp_bb_season", label_visibility="collapsed")
st.subheader("Contact Quality Allowed")

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
    st.subheader("Spray Charts Allowed")
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
# SPLITS: vs LHH / vs RHH
# =============================================================================

if "bat_side" in p1["bb"].columns:
    st.divider()
    st.subheader("Splits: vs LHH / vs RHH")
    st.caption(f"{season} Season")

    splits_1, splits_2 = st.columns(2)

    for col, p, color in [(splits_1, p1, p1_color), (splits_2, p2, p2_color)]:
        with col:
            st.markdown(f"#### {p['name']}")
            for side, label in [("L", "vs LHH"), ("R", "vs RHH")]:
                side_bb = p["bb"][p["bb"]["bat_side"] == side]
                if side_bb.empty:
                    st.caption(f"{label}: No data")
                    continue
                n = len(side_bb)
                avg_eb = side_bb["estimated_bases"].mean()
                avg_ev = side_bb["launch_speed"].mean()
                barrel_r = side_bb["is_barrel"].mean() * 100
                st.markdown(
                    f'<div style="background:#f8f9fa; border-radius:6px; padding:8px 12px; margin-bottom:6px; border-left:3px solid {color};">'
                    f'<span style="font-weight:600;">{label}</span> · {n} BB · '
                    f'Avg EB: {avg_eb:.3f} · Avg EV: {avg_ev:.1f} · Barrel: {barrel_r:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# LUCK COMPARISON
# =============================================================================

st.divider()
st.subheader("Luck Comparison")
st.caption(f"{season} Season — Positive = pitcher got lucky (allowed fewer bases than expected)")

lc1, lc2 = st.columns(2)
for col, p, luck_val, luck_pct, actual_pct, expected_pct, color in [
    (lc1, p1, luck_1, luck_pct_1, actual_pct_1, expected_pct_1, p1_color),
    (lc2, p2, luck_2, luck_pct_2, actual_pct_2, expected_pct_2, p2_color),
]:
    with col:
        st.markdown(f"#### {p['name']}")
        m1, m2 = st.columns(2)
        m1.metric("Actual TB Allowed", f"{p['total_actual_tb']:.0f}")
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
st.caption("Cumulative expected minus actual total bases allowed. Rising = pitcher getting lucky.")

fig_luck = go.Figure()

luck_last_points = []
for p, color, name in [(p1, p1_color, p1["name"]), (p2, p2_color, p2["name"])]:
    luck_ts = p["bb"].sort_values("date_parsed").copy()
    # Pitcher luck: expected - actual (positive = pitcher lucky)
    luck_ts["cum_luck"] = luck_ts["estimated_bases"].cumsum() - luck_ts["actual_tb"].cumsum()
    luck_ts["bb_num"] = range(1, len(luck_ts) + 1)
    fig_luck.add_trace(go.Scatter(
        x=luck_ts["bb_num"], y=luck_ts["cum_luck"],
        mode="lines", name=name,
        line=dict(color=color, width=2.5),
        hovertemplate="BB #%{x}<br>Cumulative Luck: %{y:.1f}<extra></extra>",
    ))
    luck_last_points.append((luck_ts["bb_num"].iloc[-1], luck_ts["cum_luck"].iloc[-1], name, color))

fig_luck.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

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
    st.markdown(f"[View {p1['name']}'s full profile](Pitcher_Profile?player={p1_display.replace(' ', '+')})")
with fc2:
    st.markdown(f"[View {p2['name']}'s full profile](Pitcher_Profile?player={p2_display.replace(' ', '+')})")

render_home_link()
