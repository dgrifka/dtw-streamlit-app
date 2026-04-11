"""
Pitcher Profile Page

Individual pitcher deep-dive: contact quality allowed, luck report, and batted ball
visualizations with Bayesian uncertainty from the DTW Simulator model.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_batted_balls, get_available_batted_ball_seasons,
    load_all_season_pa_rankings, compute_league_percentiles,
    load_player_evaluations_pa, load_player_metadata, load_pa_counts,
    load_player_projections, load_rate_stat_projections, load_pqs_projections,
    resolve_player_id, build_player_display_list, get_cached_radar_data,
)
from utils.team_mappings import (
    get_team_color, get_team_logo_url, get_short_name,
    TEAM_NAME_MAPPING,
)
from utils.player_helpers import (
    normalize_name, build_headshot_url, build_video_url,
    categorize_launch_angle, is_barrel, is_barrel_vectorized,
    _ordinal, _percentile_color,
    render_percentile_bar, render_comparison_bar_html,
    plotly_download_config, _player_filename_slug,
    luck_tier_label, pqs_tier_label, safe_html,
    render_sticky_player_bar,
    TB_MAP, PLOTLY_CONFIG, PLOTLY_CONFIG_STATIC,
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

# Season selector
col_season, _ = st.columns([1, 3])
with col_season:
    available_seasons = get_available_batted_ball_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0, key="pp_season")
    else:
        season = pd.Timestamp.now().year

# Load data
bb_df = load_batted_balls(season)
if bb_df.empty:
    st.title("Pitcher Profile")
    st.info(f"No batted ball data available for {season}.")
    st.stop()

# Verify pitcher column exists
if "pitcher" not in bb_df.columns:
    st.title("Pitcher Profile")
    st.warning("Pitcher data not available in batted ball dataset.")
    st.stop()

# Load supplementary data
metadata_df = load_player_metadata(season)
pa_rankings = load_player_evaluations_pa(season, "pitcher")

# Load multi-season PA rankings for historical timeline
all_season_pa_rankings = load_all_season_pa_rankings("pitcher")


# =============================================================================
# PITCHER SEARCH
# =============================================================================

st.title("Pitcher Profile")

# Check if we arrived via query param (cross-page linking), fall back to session state
query_player = st.query_params.get("player", "")
if not query_player:
    query_player = st.session_state.get("pitcher_profile_last_player", "")

# Build deduplicated pitcher list (merges traded pitchers, keeps same-name-different-person separate)
display_list, display_to_name, display_to_team, multi_id_names = build_player_display_list(
    bb_df, metadata_df, name_col="pitcher", team_col="opponent"
)

# Resolve default selection (from query param or random)
default_index = 0
if query_player and query_player in display_list:
    default_index = display_list.index(query_player)
elif query_player:
    qn = normalize_name(query_player)
    fuzzy = [i for i, d in enumerate(display_list) if qn in normalize_name(d)]
    if fuzzy:
        default_index = fuzzy[0]
else:
    # No query param — pick a random pitcher with 50+ batted balls faced
    eligible_displays = [d for d in display_list
                         if bb_df[bb_df["pitcher"] == display_to_name[d]].shape[0] >= 50]
    if not eligible_displays:
        eligible_displays = display_list
    random_pick = random.choice(eligible_displays)
    if random_pick in display_list:
        default_index = display_list.index(random_pick)

# Pitcher selector dropdown
search_col, shuffle_col = st.columns([3, 1], vertical_alignment="bottom")
with search_col:
    selected_display = st.selectbox(
        "Select pitcher",
        options=display_list,
        index=default_index,
        key="pp_pitcher_select",
    )
with shuffle_col:
    if st.button("Shuffle", width="stretch"):
        eligible_displays = [d for d in display_list
                             if bb_df[bb_df["pitcher"] == display_to_name[d]].shape[0] >= 50]
        if not eligible_displays:
            eligible_displays = display_list
        st.query_params["player"] = random.choice(eligible_displays)
        st.rerun()

st.session_state["pitcher_profile_last_player"] = selected_display
selected_pitcher = display_to_name.get(selected_display, selected_display)
selected_team_hint = display_to_team.get(selected_display)

# Filter batted ball data — show full season for traded pitchers, filter by team for same-name collisions
pitcher_bb = bb_df[bb_df["pitcher"] == selected_pitcher].copy()
if selected_pitcher in multi_id_names:
    pitcher_bb = pitcher_bb[pitcher_bb["opponent"] == selected_team_hint]
if pitcher_bb.empty:
    st.warning(f"No batted ball data found for {selected_pitcher}.")
    st.stop()

# Resolve team (opponent column = pitcher's team)
pitcher_bb = pitcher_bb.sort_values("date_parsed")
pitcher_team_short = pitcher_bb["opponent"].iloc[-1]

# Update URL with selected pitcher for bookmarking/sharing
st.query_params["player"] = selected_display

# League averages for context
league_avg_eb = bb_df["estimated_bases"].mean()
league_avg_ev = bb_df["launch_speed"].mean()


# =============================================================================
# HERO SECTION
# =============================================================================

st.divider()

# Resolve player_id from metadata
pitcher_meta = None
if not metadata_df.empty:
    meta_match = metadata_df[
        (metadata_df["player_name"] == selected_pitcher) &
        (metadata_df["team"] == pitcher_team_short)
    ]
    if meta_match.empty:
        meta_match = metadata_df[metadata_df["player_name"] == selected_pitcher]
    if meta_match.empty:
        player_norm = normalize_name(selected_pitcher)
        meta_match = metadata_df[
            metadata_df["player_name"].apply(normalize_name).str.contains(player_norm)
        ]
    if not meta_match.empty:
        pitcher_meta = meta_match.iloc[0]

player_id = None
if pitcher_meta is not None and "player_id" in pitcher_meta.index:
    player_id = int(pitcher_meta["player_id"])
if player_id is None:
    player_id = resolve_player_id(selected_pitcher, metadata_df, pa_rankings, pitcher_team_short)

# Look up Bayesian ranking
pitcher_ranking = None
if not pa_rankings.empty:
    rank_match = pa_rankings[
        (pa_rankings["player"] == selected_pitcher) &
        (pa_rankings["team"] == pitcher_team_short)
    ]
    if rank_match.empty:
        rank_match = pa_rankings[pa_rankings["player"] == selected_pitcher]
    if not rank_match.empty:
        pitcher_ranking = rank_match.iloc[0]

# Team color
primary_color, secondary_color = get_team_color(pitcher_team_short)

# Compute base stats
avg_eb = pitcher_bb["estimated_bases"].mean()
avg_ev = pitcher_bb["launch_speed"].mean()
n_bb = len(pitcher_bb)
pitcher_bb["is_barrel"] = is_barrel_vectorized(pitcher_bb["launch_speed"], pitcher_bb["launch_angle"])
barrel_rate_raw = pitcher_bb["is_barrel"].mean() * 100

# Prefer true talent → Bayesian posterior → raw for barrel rate
barrel_rate = barrel_rate_raw
_barrel_bayesian = False
if pitcher_ranking is not None:
    if "true_talent_barrel_rate" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("true_talent_barrel_rate")):
        barrel_rate = pitcher_ranking["true_talent_barrel_rate"] * 100
        _barrel_bayesian = True
    elif "barrel_rate_posterior" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("barrel_rate_posterior")):
        barrel_rate = pitcher_ranking["barrel_rate_posterior"] * 100
        _barrel_bayesian = True

# Hero identity card — stays cohesive on mobile
pos_str = ""
age_str = ""
throw_str = ""
if pitcher_meta is not None:
    pos = pitcher_meta.get("position", "")
    if pos:
        pos_str = f" | {pos}"
    bd = pitcher_meta.get("birth_date", "")
    if bd:
        try:
            birth = pd.to_datetime(bd)
            age = (pd.Timestamp.now() - birth).days // 365
            age_str = f" | Age {age}"
        except Exception:
            pass
    th = pitcher_meta.get("throw_hand", "")
    if th:
        throw_str = f" | Throws {th}"

# PA count from rankings
n_pa = int(pitcher_ranking["n_batted_balls"]) if pitcher_ranking is not None and "n_batted_balls" in pitcher_ranking.index else None
pa_str = f" | {n_pa:,} PA" if n_pa else ""
bb_str = f" | {n_bb:,} BB"

img_url = build_headshot_url(player_id) if player_id else (get_team_logo_url(pitcher_team_short) or "")
img_html = ""
if img_url:
    img_html = (
        f'<img src="{img_url}" '
        f'style="width:120px; height:120px; object-fit:contain; border-radius:8px; flex-shrink:0;" '
        f'onerror="this.style.display=\'none\'">'
    )

st.markdown(
    f'<div id="hero-section-sentinel" style="display:flex; align-items:center; gap:20px; '
    f'background:#f8f9fa; border-radius:10px; padding:16px; '
    f'border-left:4px solid {primary_color};">'
    f'{img_html}'
    f'<div>'
    f'<div style="font-size:1.5rem; font-weight:700; margin-bottom:2px;">{safe_html(selected_pitcher)}</div>'
    f'<div style="color:#4A5568; font-size:1rem;"><b>{safe_html(pitcher_team_short)}</b>{pos_str}{throw_str}{age_str}{pa_str}{bb_str}</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

render_sticky_player_bar(
    player_name=selected_pitcher,
    team_short=pitcher_team_short,
    primary_color=primary_color,
    headshot_url=img_url,
    subtitle=f"{pos_str}{throw_str}{age_str}".lstrip(" |"),
)

# Key stats — inverted percentiles (lower = better for pitcher)
league_pcts = compute_league_percentiles(season, "pitcher")
if league_pcts:
    ev_pct_raw = (league_pcts["ev_by_player"] < avg_ev).mean() * 100
    ev_pct = 100 - ev_pct_raw  # invert: low EV allowed = high percentile
    # Prefer Bayesian percentile ranking when available (inverted: lower = better)
    if _barrel_bayesian and not pa_rankings.empty and "barrel_rate_posterior" in pa_rankings.columns:
        barrel_pct_raw = (pa_rankings["barrel_rate_posterior"].dropna() < pitcher_ranking["barrel_rate_posterior"]).mean() * 100
    else:
        barrel_pct_raw = (league_pcts["barrel_rates"] < barrel_rate).mean() * 100
    barrel_pct = 100 - barrel_pct_raw  # invert: low barrel rate = high percentile
else:
    ev_pct = barrel_pct = 50

# --- Pre-compute EV comparison bar data (pitcher: lower = better) ---
_ev_bar_html = ""
if league_pcts:
    _ev_series = league_pcts["ev_by_player"]
    _bb_counts = league_pcts["bb_counts"]
    _qualified_ev = _ev_series[_bb_counts >= 50]
    if not _qualified_ev.empty:
        _best_ev_name = _qualified_ev.idxmin()  # lowest EV = best pitcher
        _best_ev_val = _qualified_ev.min()
        _best_ev_match = pa_rankings[pa_rankings["player"] == _best_ev_name] if not pa_rankings.empty else pd.DataFrame()
        _best_ev_team = _best_ev_match.iloc[0]["team"] if not _best_ev_match.empty else ""
        _best_ev_color, _ = get_team_color(_best_ev_team) if _best_ev_team else ("#DAA520", "#DAA520")
        _best_ev_bb = bb_df[bb_df["pitcher"] == _best_ev_name]
        _best_ev_se = _best_ev_bb["launch_speed"].std() / np.sqrt(len(_best_ev_bb)) if len(_best_ev_bb) > 1 else 1.0
        _player_ev_se = pitcher_bb["launch_speed"].std() / np.sqrt(n_bb) if n_bb > 1 else 1.0
        _ev_bar_html = render_comparison_bar_html(
            selected_pitcher, avg_ev, avg_ev - 1.6 * _player_ev_se, avg_ev + 1.6 * _player_ev_se, primary_color,
            _best_ev_name, _best_ev_val, _best_ev_val - 1.6 * _best_ev_se, _best_ev_val + 1.6 * _best_ev_se, _best_ev_color,
            _qualified_ev.mean(), _qualified_ev.std(),
            value_fmt=".1f", caption="lower = better",
        )

# --- Pre-compute barrel rate comparison bar data (pitcher: lower = better) ---
_barrel_bar_html = ""
if league_pcts:
    _barrel_series = league_pcts["barrel_rates"]
    _qualified_barrel = _barrel_series[_bb_counts >= 50]
    if not _qualified_barrel.empty:
        _best_brl_name = _qualified_barrel.idxmin()  # lowest barrel rate = best pitcher
        _best_brl_val = _qualified_barrel.min()
        _best_brl_match = pa_rankings[pa_rankings["player"] == _best_brl_name] if not pa_rankings.empty else pd.DataFrame()
        _best_brl_team = _best_brl_match.iloc[0]["team"] if not _best_brl_match.empty else ""
        _best_brl_color, _ = get_team_color(_best_brl_team) if _best_brl_team else ("#DAA520", "#DAA520")
        _best_brl_bb = bb_df[bb_df["pitcher"] == _best_brl_name]
        _brl_p_best = max(_best_brl_val / 100, 0.01)
        _best_brl_se = np.sqrt(_brl_p_best * (1 - _brl_p_best) / len(_best_brl_bb)) * 100 if len(_best_brl_bb) > 1 else 1.0
        _brl_p_player = max(barrel_rate / 100, 0.01)
        _player_brl_se = np.sqrt(_brl_p_player * (1 - _brl_p_player) / n_bb) * 100 if n_bb > 1 else 1.0
        _barrel_bar_html = render_comparison_bar_html(
            selected_pitcher, barrel_rate, barrel_rate - 1.6 * _player_brl_se, barrel_rate + 1.6 * _player_brl_se, primary_color,
            _best_brl_name, _best_brl_val, _best_brl_val - 1.6 * _best_brl_se, _best_brl_val + 1.6 * _best_brl_se, _best_brl_color,
            _qualified_barrel.mean(), _qualified_barrel.std(),
            value_fmt=".1f", value_suffix="%", caption="lower = better",
        )

# --- Pre-compute EB/PA comparison bar data (pitcher: lower = better) ---
# Hierarchy: true_talent_eb_pa > projected_eb_pa > posterior_mean
_proj_df = load_player_projections(season, "pitcher")
if _proj_df.empty:
    _proj_df = load_player_projections(season + 1, "pitcher")
_proj_active = _proj_df[_proj_df["p_active_next_season"] > 0.3] if not _proj_df.empty and "p_active_next_season" in _proj_df.columns else _proj_df

_eb_bar_html = ""
bayesian_eb = None
eb_pct = None
_eb_label = "Est. Bases Allowed/PA"
_eb_help = "Estimated true production allowed per plate appearance. Lower is better for pitchers. Small samples are adjusted toward league average."

# Level 1: true_talent_eb_pa (projection + in-season blend)
_has_tt = (
    pitcher_ranking is not None
    and "true_talent_eb_pa" in pa_rankings.columns
    and pa_rankings["true_talent_eb_pa"].notna().any()
    and pd.notna(pitcher_ranking.get("true_talent_eb_pa"))
)
if _has_tt:
    _eb_label = "True Talent EB/PA"
    _eb_help = "True talent estimate combining preseason projections with in-season Bayesian evaluation. Lower is better for pitchers."
    bayesian_eb = pitcher_ranking["true_talent_eb_pa"]
    hdi_low = pitcher_ranking["hdi_low"]
    hdi_high = pitcher_ranking["hdi_high"]
    _tt_vals = pa_rankings["true_talent_eb_pa"].dropna()
    eb_pct = 100 - (_tt_vals < bayesian_eb).mean() * 100  # inverted: lower = better
    _best_idx = _tt_vals.idxmin()
    _best_row = pa_rankings.loc[_best_idx]
    _best_name = _best_row["player"]
    _best_team = _best_row.get("team", "")
    _best_color, _ = get_team_color(_best_team) if _best_team else ("#DAA520", "#DAA520")
    _eb_bar_html = render_comparison_bar_html(
        selected_pitcher, bayesian_eb, hdi_low, hdi_high, primary_color,
        _best_name, _best_row["true_talent_eb_pa"], _best_row["hdi_low"], _best_row["hdi_high"], _best_color,
        _tt_vals.mean(), _tt_vals.std(),
        caption="lower = better",
    )

# Level 2: projections (pre-season / early-season)
if bayesian_eb is None and not _proj_active.empty:
    _pm = _proj_active[_proj_active["player"] == selected_pitcher]
    _team_pm = _pm[_pm["team"] == pitcher_team_short]
    _pm = _team_pm if not _team_pm.empty else _pm
    if not _pm.empty:
        _eb_label = "Projected EB/PA"
        _eb_help = "Bayesian projection based on multi-year historical performance and aging curves. Lower is better for pitchers."
        _p = _pm.iloc[0]
        bayesian_eb = _p["projected_eb_pa"]
        hdi_low = _p["projected_hdi_low"]
        hdi_high = _p["projected_hdi_high"]
        eb_pct = 100 - (_proj_active["projected_eb_pa"] < bayesian_eb).mean() * 100  # inverted
        _best_idx = _proj_active["projected_eb_pa"].idxmin()
        _best_proj = _proj_active.loc[_best_idx]
        _best_name = _best_proj["player"]
        _best_team = _best_proj.get("team", "")
        _best_color, _ = get_team_color(_best_team) if _best_team else ("#DAA520", "#DAA520")
        _eb_bar_html = render_comparison_bar_html(
            selected_pitcher, bayesian_eb, hdi_low, hdi_high, primary_color,
            _best_name, _best_proj["projected_eb_pa"], _best_proj["projected_hdi_low"], _best_proj["projected_hdi_high"], _best_color,
            _proj_active["projected_eb_pa"].mean(), _proj_active["projected_eb_pa"].std(),
            caption="lower = better",
        )

# Level 3: raw posterior (historical seasons without projections)
if bayesian_eb is None and pitcher_ranking is not None:
    bayesian_eb = pitcher_ranking["posterior_mean"]
    hdi_low = pitcher_ranking["hdi_low"]
    hdi_high = pitcher_ranking["hdi_high"]
    eb_pct_raw = (pa_rankings["posterior_mean"] < bayesian_eb).mean() * 100
    eb_pct = 100 - eb_pct_raw
    _best_idx = pa_rankings["posterior_mean"].idxmin()
    _best_row = pa_rankings.loc[_best_idx]
    _best_name = _best_row["player"]
    _best_team = _best_row.get("team", "")
    _best_color, _ = get_team_color(_best_team) if _best_team else ("#DAA520", "#DAA520")
    _eb_bar_html = render_comparison_bar_html(
        selected_pitcher, bayesian_eb, hdi_low, hdi_high, primary_color,
        _best_name, _best_row["posterior_mean"], _best_row["hdi_low"], _best_row["hdi_high"], _best_color,
        pa_rankings["posterior_mean"].mean(), pa_rankings["posterior_mean"].std(),
        caption="lower = better",
    )

hero_col1, hero_col2, hero_col3 = st.columns(3)

with hero_col1:
    st.metric("Avg EV Allowed", f"{avg_ev:.1f} mph",
              help="Average exit velocity allowed on all batted balls (mph). Lower is better.")
    render_percentile_bar(ev_pct, label=f"{_ordinal(int(ev_pct))} pct (lower EV = better)")
    if _ev_bar_html:
        st.markdown(_ev_bar_html, unsafe_allow_html=True)

with hero_col2:
    _barrel_help = "Barrel rate allowed. Barrels: EV >= 98 mph + launch angle in the sweet spot zone. Lower is better."
    if _barrel_bayesian:
        _barrel_help += " Adjusted for sample size using a Bayesian model."
    st.metric("Barrel Rate Allowed", f"{barrel_rate:.1f}%", help=_barrel_help)
    render_percentile_bar(barrel_pct, label=f"{_ordinal(int(barrel_pct))} pct (fewer barrels = better)")
    if _barrel_bar_html:
        st.markdown(_barrel_bar_html, unsafe_allow_html=True)
    if _barrel_bayesian and "barrel_rate_hdi_low" in pitcher_ranking.index:
        st.caption(
            f"89% CI: {pitcher_ranking['barrel_rate_hdi_low']*100:.1f}% – "
            f"{pitcher_ranking['barrel_rate_hdi_high']*100:.1f}%"
        )

with hero_col3:
    _hero_has_k = (
        pitcher_ranking is not None
        and "k_rate_posterior" in pitcher_ranking.index
        and pd.notna(pitcher_ranking.get("k_rate_posterior"))
    )
    if _hero_has_k:
        _hero_k = pitcher_ranking["k_rate_posterior"]
        st.metric(
            "K%", f"{_hero_k * 100:.1f}%",
            help="Bayesian strikeout rate per batter faced. **Higher is better** for pitchers — "
                 "more strikeouts means more missed bats. Adjusted for sample size.",
        )
        if not pa_rankings.empty and "k_rate_posterior" in pa_rankings.columns:
            _hero_k_pct = (pa_rankings["k_rate_posterior"].dropna() < _hero_k).mean() * 100
            render_percentile_bar(_hero_k_pct, label=f"{_ordinal(int(_hero_k_pct))} pct (higher K% = better)")
        # Comparison bar vs best + league avg
        if not pa_rankings.empty and "k_rate_posterior" in pa_rankings.columns:
            _k_pop = pa_rankings["k_rate_posterior"].dropna()
            _best_k_idx = _k_pop.idxmax()  # highest K% = best pitcher
            _best_k_row = pa_rankings.loc[_best_k_idx]
            _best_k_name = _best_k_row["player"]
            _best_k_team = _best_k_row.get("team", "")
            _best_k_color, _ = get_team_color(_best_k_team) if _best_k_team else ("#DAA520", "#DAA520")
            _k_hdi_lo = pitcher_ranking.get("k_rate_hdi_low", _hero_k)
            _k_hdi_hi = pitcher_ranking.get("k_rate_hdi_high", _hero_k)
            _best_k_hdi_lo = _best_k_row.get("k_rate_hdi_low", _best_k_row["k_rate_posterior"])
            _best_k_hdi_hi = _best_k_row.get("k_rate_hdi_high", _best_k_row["k_rate_posterior"])
            _k_bar_html = render_comparison_bar_html(
                selected_pitcher, _hero_k, _k_hdi_lo, _k_hdi_hi, primary_color,
                _best_k_name, _best_k_row["k_rate_posterior"], _best_k_hdi_lo, _best_k_hdi_hi, _best_k_color,
                _k_pop.mean(), _k_pop.std(),
                caption="higher = better",
            )
            if _k_bar_html:
                st.markdown(_k_bar_html, unsafe_allow_html=True)
        if "k_rate_hdi_low" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("k_rate_hdi_low")):
            st.caption(
                f"89% CI: {pitcher_ranking['k_rate_hdi_low']*100:.1f}% – "
                f"{pitcher_ranking['k_rate_hdi_high']*100:.1f}%"
            )
    elif bayesian_eb is not None:
        st.metric(_eb_label, f"{bayesian_eb:.3f}", help=_eb_help)
        render_percentile_bar(eb_pct, label=f"{_ordinal(int(eb_pct))} pct (lower EB/PA = better)")
        if _eb_bar_html:
            st.markdown(_eb_bar_html, unsafe_allow_html=True)
    else:
        st.metric("Avg Est. Bases Allowed/BB", f"{avg_eb:.3f}")
        st.caption("Full ranking not available (need player evaluation data)")

# PQS+ highlight below hero
if (pitcher_ranking is not None
        and "pitcher_quality_score" in pitcher_ranking.index
        and not pd.isna(pitcher_ranking.get("pitcher_quality_score", float("nan")))):
    _hero_pqs = pitcher_ranking["pitcher_quality_score"]
    _pqs_hero_col1, _pqs_hero_col2 = st.columns([1, 3])
    with _pqs_hero_col1:
        st.metric(
            "PQS+", f"{_hero_pqs:.0f}",
            help="Pitcher Quality Score (70% K% + 30% contact quality). **Higher is better.** "
                 "100 = league average. Every 15 points is roughly one standard deviation.",
        )
        if not pa_rankings.empty and "pitcher_quality_score" in pa_rankings.columns:
            _hero_pqs_pct = (pa_rankings["pitcher_quality_score"].dropna() < _hero_pqs).mean() * 100
            render_percentile_bar(_hero_pqs_pct, label=f"{_ordinal(int(_hero_pqs_pct))} pct (higher PQS+ = better)")
        _hero_pqs_has_hdi = (
            "pqs_hdi_low" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("pqs_hdi_low"))
            and "pqs_hdi_high" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("pqs_hdi_high"))
        )
        if _hero_pqs_has_hdi:
            st.caption(f"89% CI: {pitcher_ranking['pqs_hdi_low']:.0f} – {pitcher_ranking['pqs_hdi_high']:.0f}")
        _hero_tier = pqs_tier_label(_hero_pqs)
        if _hero_tier:
            _hero_tier_color = {
                "Elite": "green", "Above Avg": "green",
                "Average": "gray", "Below Avg": "orange", "Poor": "red",
            }.get(_hero_tier, "gray")
            st.caption(f"**:{_hero_tier_color}[{_hero_tier}]**")

# Pre-compute actual_tb (used by Quick Stats and Luck Report)
pitcher_bb["actual_tb"] = pitcher_bb["actual_result"].map(TB_MAP).fillna(0)


# =============================================================================
# RADAR + LUCK COMPUTATION (moved earlier for Snapshot section)
# =============================================================================

from utils.player_analytics import (
    find_similar_players, get_player_radar_percentiles,
    PITCHER_ARCHETYPE_DESC, generate_player_highlights,
    compute_player_grade, compute_projected_grade,
)
from utils.player_helpers import (
    render_radar_chart, render_archetype_badge, render_similar_players,
    render_snapshot_section, render_highlights,
)

_radar_df = get_cached_radar_data(season, player_type="pitcher", min_pa=100)
_using_projected_radar = False

if not _radar_df.empty:
    _player_pcts = get_player_radar_percentiles(_radar_df, selected_pitcher, pitcher_team_short, "pitcher")
else:
    _player_pcts = None

# Fall back to projected radar if player not in actual radar_df
if _player_pcts is None:
    from utils.data_loader import get_cached_projected_radar_data
    _proj_radar_df = get_cached_projected_radar_data(season, player_type="pitcher")
    if not _proj_radar_df.empty:
        _player_pcts = get_player_radar_percentiles(_proj_radar_df, selected_pitcher, pitcher_team_short, "pitcher")
        if _player_pcts is not None:
            _using_projected_radar = True
            _radar_df = _proj_radar_df

# Look up archetype for this pitcher
_archetype = "Unknown"
_arch_desc = ""
if _player_pcts is not None and not _radar_df.empty:
    _player_match = _radar_df[
        (_radar_df["player"] == selected_pitcher) & (_radar_df["team"] == pitcher_team_short)
    ]
    if _player_match.empty:
        _player_match = _radar_df[_radar_df["player"] == selected_pitcher]
    if not _player_match.empty:
        _archetype = _player_match.iloc[0]["archetype"]
        _arch_base = _archetype.replace(" II", "") if _archetype.endswith(" II") else _archetype
        _arch_desc = PITCHER_ARCHETYPE_DESC.get(_archetype, PITCHER_ARCHETYPE_DESC.get(_arch_base, ""))

# Compute Quick Stats (luck, trend, platoon)
_pitcher_luck = pitcher_bb["estimated_bases"].sum() - pitcher_bb["actual_tb"].sum()

_league_pcts_pitcher = compute_league_percentiles(season, group_col="pitcher")
# Pitcher luck is inverted: expected - actual (positive = lucky), so negate hitter luck
_league_luck_per_pitcher = -_league_pcts_pitcher.get("luck_per_player", pd.Series(dtype=float))
_luck_pct = (_league_luck_per_pitcher < _pitcher_luck).mean() * 100 if not _league_luck_per_pitcher.empty else 50.0

_lucky_hits_allowed = pitcher_bb[
    (pitcher_bb["actual_tb"] > 0) & (pitcher_bb["estimated_bases"] < 0.5)
]
_n_lucky_hits = len(_lucky_hits_allowed)

_recent_eb_p = None
_recent_vs_season_p = 0.0
if "date_parsed" in pitcher_bb.columns and len(pitcher_bb) > 0:
    _max_date_p = pitcher_bb["date_parsed"].max()
    _recent_p = pitcher_bb[pitcher_bb["date_parsed"] > _max_date_p - pd.Timedelta(days=14)]
    if len(_recent_p) > 5:
        _recent_eb_p = _recent_p["estimated_bases"].mean()
        _recent_vs_season_p = _recent_eb_p - avg_eb

_platoon_str_p = None
if not metadata_df.empty and "player" in pitcher_bb.columns:
    _bsm = metadata_df.set_index("player_name")["bat_side"].to_dict()
    pitcher_bb["_bat_side"] = pitcher_bb["player"].map(_bsm)
    _vs_l = pitcher_bb[pitcher_bb["_bat_side"] == "L"]
    _vs_r = pitcher_bb[pitcher_bb["_bat_side"] == "R"]
    if len(_vs_l) >= 10 and len(_vs_r) >= 10:
        _gap_p = _vs_l["estimated_bases"].mean() - _vs_r["estimated_bases"].mean()
        if abs(_gap_p) > 0.03:
            _weak_side = "vs LHH" if _gap_p > 0 else "vs RHH"
            _platoon_str_p = f"+{abs(_gap_p):.3f} EB/BB {_weak_side}"
    pitcher_bb.drop(columns=["_bat_side"], inplace=True, errors="ignore")


# =============================================================================
# PLAYER SNAPSHOT (Score + Percentile Bars + Highlights + True Talent)
# =============================================================================

if _player_pcts is not None and not _radar_df.empty:
    st.divider()

    _pm = _radar_df[
        (_radar_df["player"] == selected_pitcher) & (_radar_df["team"] == pitcher_team_short)
    ]
    if _pm.empty:
        _pm = _radar_df[_radar_df["player"] == selected_pitcher]
    _pr = _pm.iloc[0] if not _pm.empty else None

    if _pr is not None:
        # Grade — preseason mode uses projections; in-season uses radar percentiles
        _current_year = pd.Timestamp.now().year
        _is_preseason = season < _current_year and not _proj_active.empty

        if _is_preseason:
            _proj_grade = compute_projected_grade(
                selected_pitcher, pitcher_team_short, _proj_active, player_type="pitcher"
            )
            if _proj_grade is not None:
                _composite = _proj_grade
                _target_season = int(_proj_df["target_season"].iloc[0]) if "target_season" in _proj_df.columns else _current_year
                _snap_subtitle = f"Based on {_target_season} projections"
            else:
                _composite = compute_player_grade(_player_pcts, player_type="pitcher")
                if _composite is None:
                    _composite = sum(_player_pcts.values()) / len(_player_pcts)
                _snap_subtitle = None
        else:
            _composite = compute_player_grade(_player_pcts, player_type="pitcher")
            if _composite is None:
                _composite = sum(_player_pcts.values()) / len(_player_pcts)
            _has_projections = "true_talent_eb_pa" in _pr.index and pd.notna(_pr.get("true_talent_eb_pa"))
            _snap_subtitle = f"Includes {season} projections" if _has_projections else None

        # Build 8 metrics for pitcher — all oriented higher=better
        _snapshot_metrics = []

        # EB/PA Allowed: use posterior_mean + hero's eb_pct for consistency
        if pitcher_ranking is not None:
            _snapshot_metrics.append({"label": "EB/PA Allowed", "pct": eb_pct, "value": f"{pitcher_ranking['posterior_mean']:.3f}", "group": 0})

        # EV Allowed (inverted percentile from hero)
        _snapshot_metrics.append({"label": "EV Allowed", "pct": ev_pct, "value": f"{avg_ev:.1f} mph", "group": 0})

        # Weak Contact (inverted hard hit rate — already oriented higher=better)
        _wc_val = _pr.get("weak_contact", None)
        _wc_pct = _pr.get("weak_contact_pct", 50)
        if _wc_val is not None:
            _snapshot_metrics.append({"label": "Weak Contact", "pct": _wc_pct, "value": f"{_wc_val * 100:.1f}%", "group": 0})

        # Barrel Rate Allowed (inverted percentile from hero)
        _snapshot_metrics.append({"label": "Barrel Rate", "pct": barrel_pct, "value": f"{barrel_rate:.1f}%", "group": 0})

        # --- Discipline group ---
        # Strikeout Ability
        _ka_val = _pr.get("k_ability", None)
        _ka_pct = _pr.get("k_ability_pct", 50)
        if _ka_val is not None:
            _snapshot_metrics.append({"label": "K Rate", "pct": _ka_pct, "value": f"{_ka_val * 100:.1f}%", "group": 1})

        # Command (1 - BB rate)
        _cmd_val = _pr.get("command", None)
        _cmd_pct = _pr.get("command_pct", 50)
        if _cmd_val is not None:
            _bb_rate_display = (1 - _cmd_val) * 100
            _snapshot_metrics.append({"label": "Command", "pct": _cmd_pct, "value": f"{_bb_rate_display:.1f}% BB", "group": 1})

        # HR Prevention (1 - HR rate)
        _hrp_val = _pr.get("hr_prevention", None)
        _hrp_pct = _pr.get("hr_prevention_pct", 50)
        if _hrp_val is not None:
            _hr_rate_display = (1 - _hrp_val) * 100
            _snapshot_metrics.append({"label": "HR Prevention", "pct": _hrp_pct, "value": f"{_hr_rate_display:.1f}% HR", "group": 1})

        # Ground Ball Rate
        _gb_val = _pr.get("gb_rate", None)
        _gb_pct = _pr.get("gb_rate_pct", 50)
        if _gb_val is not None:
            _snapshot_metrics.append({"label": "Ground Balls", "pct": _gb_pct, "value": f"{_gb_val * 100:.1f}%", "group": 1})

        render_snapshot_section(_snapshot_metrics, _composite, _archetype, primary_color, subtitle=_snap_subtitle)

    # Auto-generated highlights
    _deviation = None
    if pitcher_ranking is not None and "deviation" in pitcher_ranking.index:
        _deviation = pitcher_ranking.get("deviation")
        if pd.isna(_deviation):
            _deviation = None

    _highlights = generate_player_highlights(
        radar_pcts=_player_pcts,
        archetype_name=_archetype,
        player_type="pitcher",
        deviation=_deviation,
        luck_pct=_luck_pct,
        recent_vs_season=_recent_vs_season_p if _recent_eb_p is not None else None,
        platoon_str=_platoon_str_p,
        archetype_desc=_arch_desc,
    )
    render_highlights(_highlights, primary_color)

    with st.expander("How does this work?"):
        st.markdown(
            "**Overall Score (0-100):** Based entirely on Run Prevention, the Bayesian model's estimate of "
            "how many estimated bases per PA a pitcher allows. A pitcher who limits hard contact and walks "
            "will grade well regardless of their strikeout rate or ground ball tendencies.\n\n"
            "When preseason projections are available (based on multi-year historical data), "
            "the score uses a \"true talent\" estimate that blends this season's results with projections "
            "for a more stable read, especially early in the year.\n\n"
            "**Letter Grades:** A+ (90+), A (80-89), B+ (70-79), B (60-69), C+ (50-59), C (40-49), D (30-39), F (<30).\n\n"
            "**Percentile Bars:** Each bar shows where this pitcher ranks among all qualified pitchers. "
            "The filled portion represents the pitcher's percentile (team color), with a vertical line at the 50th percentile "
            "(league median). All metrics are oriented so that higher = better "
            "(e.g., a high \"Command\" percentile means a low walk rate).\n\n"
            "**Radar Chart:** Shows the pitcher's skill profile across 6 dimensions. "
            "The grade is intentionally decoupled from the radar, so the grade answers \"how good at preventing runs?\" "
            "while the radar answers \"what kind of pitcher?\"\n\n"
            "**Highlights:** Auto-generated callouts based on the pitcher's most notable traits, "
            "including standout skills, luck context, recent trends, and platoon splits."
        )


# =============================================================================
# SEASON STATS (traditional pitching stats for context)
# =============================================================================

if pitcher_ranking is not None and "era" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("era")):
    st.divider()
    st.subheader(f"{season} Season Stats")

    _era = pitcher_ranking["era"]
    _whip = pitcher_ranking["whip"]
    _ip = pitcher_ranking["innings_pitched"]
    _ip_str = f"{_ip:.1f}" if isinstance(_ip, float) else str(_ip)
    _w = int(pitcher_ranking["wins"])
    _l = int(pitcher_ranking["losses"])
    _k = int(pitcher_ranking["strikeouts"])
    _bb_p = int(pitcher_ranking["walks"])
    _hr_a = int(pitcher_ranking["home_runs_allowed"])
    _sv = int(pitcher_ranking["saves"])

    _count_items = (
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">K </span><span style="font-weight:600; font-size:0.95rem;">{_k}</span></div>'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">BB </span><span style="font-weight:600; font-size:0.95rem;">{_bb_p}</span></div>'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">HR </span><span style="font-weight:600; font-size:0.95rem;">{_hr_a}</span></div>'
    )
    if _sv > 0:
        _count_items += f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">SV </span><span style="font-weight:600; font-size:0.95rem;">{_sv}</span></div>'

    st.markdown(
        f'<div style="background:#F7FAFC; border-radius:10px; padding:16px 20px; '
        f'border-left:4px solid {primary_color};">'
        # Rate stats row
        f'<div style="display:flex; justify-content:center; gap:32px; flex-wrap:wrap; margin-bottom:10px;">'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">ERA</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_era:.2f}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">WHIP</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_whip:.2f}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">IP</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_ip_str}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">W-L</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_w}-{_l}</div></div>'
        f'</div>'
        # Counting stats row
        f'<div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; padding-top:8px; border-top:1px solid #E2E8F0;">'
        f'{_count_items}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("Traditional stats via MLB Stats API — actual results, not model estimates. Totals may vary slightly from other sources if Statcast data was unavailable for a game or plate appearance.")


# =============================================================================
# PLAYER PROFILE (Radar Chart + Archetype + Similar Players + Quick Stats)
# =============================================================================

def _render_pitcher_quick_stats():
    """Render pitcher Quick Stats HTML block."""
    st.markdown('<div style="font-weight:600; margin-top:16px; margin-bottom:6px; font-size:0.85rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">Quick Stats</div>', unsafe_allow_html=True)
    _luck_str = f"{_pitcher_luck:+.1f} net lucky bases ({_luck_pct:.0f}th pct)"
    _quick_parts = [f"Luck: {_luck_str}", f"{_n_lucky_hits} lucky hits allowed (EB < 0.5)"]
    if _recent_eb_p is not None:
        _arrow = "+" if _recent_vs_season_p > 0.005 else ("" if _recent_vs_season_p < -0.005 else "")
        _trend_note = "worse" if _recent_vs_season_p > 0.005 else ("better" if _recent_vs_season_p < -0.005 else "same")
        _quick_parts.append(f"Last 14d: {_recent_eb_p:.3f} EB/BB ({_arrow}{_recent_vs_season_p:.3f} vs season — {_trend_note})")
    if _platoon_str_p:
        _quick_parts.append(f"Weakness: {_platoon_str_p}")
    st.markdown(
        "".join(f'<div style="color:#4A5568; font-size:0.88rem; padding:1px 0;">{p}</div>' for p in _quick_parts),
        unsafe_allow_html=True,
    )

if _player_pcts is not None:
    st.divider()
    st.subheader("Player Profile")

    _col_radar, _col_info = st.columns([3, 2])

    with _col_radar:
        _radar_fig = render_radar_chart(_player_pcts, primary_color)
        _slug = _player_filename_slug(selected_pitcher)
        st.plotly_chart(_radar_fig, use_container_width=True, config=plotly_download_config(f"{_slug}_radar_{season}", width=800, height=800))
        st.caption("Tap the camera icon above any chart to save as PNG.")
        if _using_projected_radar:
            st.caption(f"Based on preseason projections ({n_bb} batters faced so far). Will update to actual data at 100+ batters faced.")
        elif n_bb < 50:
            st.caption(f"Based on {n_bb} batted balls faced — profile may shift as more data accumulates.")

    with _col_info:
        _archetype_display = f"{_archetype} (Projected)" if _using_projected_radar and _archetype != "Unknown" else _archetype
        st.markdown(render_archetype_badge(_archetype_display, _arch_desc, primary_color), unsafe_allow_html=True)

        st.markdown('<div style="font-weight:600; margin-top:16px; margin-bottom:6px; font-size:0.85rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">Similar Pitchers</div>', unsafe_allow_html=True)
        _similar = find_similar_players(_radar_df, selected_pitcher, pitcher_team_short, "pitcher", n=5)
        st.markdown(render_similar_players(_similar), unsafe_allow_html=True)

        _render_pitcher_quick_stats()

    with st.expander("How does this work?"):
        st.markdown(
            "**Radar Chart:** Shows how this pitcher compares to every pitcher with 100+ batters faced this season. "
            "Each spoke is a different skill, measured as a percentile (0 to 100). All axes are oriented so that bigger = better "
            "(e.g., \"Command\" = low walk rate, \"HR Prevention\" = low HR rate).\n\n"
            "**Archetype:** Pitchers are grouped by their radar shape using a clustering algorithm (K-Means). "
            "Pitchers in the same archetype tend to have similar strengths and weaknesses.\n\n"
            "**Current Pitcher Archetypes:**\n"
            "- **Elite Command**: Exceptional command and run prevention with well-rounded skills. Consistently locates pitches and limits damage.\n"
            "- **Strikeout Artist**: Misses bats at an elite rate with strong overall run prevention. Overpowers hitters with swing-and-miss stuff.\n"
            "- **Ground Ball Machine**: Keeps the ball on the ground and limits home runs effectively. Relies on inducing weak ground-ball contact.\n"
            "- **Pitch-to-Contact**: Good command and induces weak contact, but doesn't miss many bats. Relies on location and movement over velocity.\n"
            "- **Finesse Pitcher**: Decent command and gets ground balls, but lacks swing-and-miss ability. Relies on guile and location over pure stuff.\n"
            "- **Volatile**: Has some swing-and-miss ability but walks too many batters. Results are inconsistent due to poor command.\n"
            "- **Below Average**: Below-average production across most skill dimensions this season.\n\n"
            "**Similar Pitchers:** The 5 pitchers whose overall skill profile most closely matches this pitcher's, "
            "based on the Euclidean distance between their radar shapes in 6-dimensional percentile space.\n\n"
            "**Quick Stats:** Luck = expected TB minus actual TB allowed (positive = pitcher got lucky). "
            "Lucky hits allowed = batted balls that became hits despite low expected bases (EB < 0.5). "
            "14-day trend shows recent EB/BB vs season average (higher = allowing harder contact). "
            "Platoon weakness shows which handedness the pitcher struggles against more.\n\n"
            "**Data Note:** The radar uses the best available estimate of each pitcher's true skill level. "
            "When preseason projections are available, they are combined with in-season performance using "
            "inverse-variance weighting for a more stable \"true talent\" estimate. Otherwise, the Bayesian "
            "posterior from the current season is used, which already shrinks small samples toward the league mean."
        )

else:
    # Fallback: no radar data — still show Quick Stats
    st.divider()
    st.subheader("Player Profile")
    _render_pitcher_quick_stats()


# =============================================================================
# HISTORICAL EST. BASES / PA TIMELINE
# =============================================================================

if len(all_season_pa_rankings) > 0 and player_id is not None:
    timeline_data = []
    best_player_data = []
    league_avg_data = []
    rate_timeline = []
    rate_league_avg = []

    for s in sorted(all_season_pa_rankings.keys()):
        pa_df = all_season_pa_rankings[s]

        lg_mean = pa_df["posterior_mean"].mean()
        lg_sd = pa_df["posterior_mean"].std()
        league_avg_data.append({"season": s, "value": lg_mean, "sd": lg_sd})

        # Rate stat league averages + SD
        _rate_lg = {"season": s}
        for _rc in ["k_rate_posterior", "bb_rate_posterior", "hr_rate_posterior"]:
            if _rc in pa_df.columns:
                _vals = pa_df[_rc].dropna()
                _rate_lg[_rc] = _vals.mean()
                _rate_lg[f"{_rc}_sd"] = _vals.std()
        rate_league_avg.append(_rate_lg)

        # Best pitcher = lowest posterior_mean
        best_idx_s = pa_df["posterior_mean"].idxmin()
        best_row_s = pa_df.loc[best_idx_s]
        best_team_s = best_row_s.get("team", "")
        best_color_s, _ = get_team_color(best_team_s) if best_team_s else ("#DAA520", "#DAA520")
        best_player_data.append({
            "season": s,
            "value": best_row_s["posterior_mean"],
            "name": best_row_s["player"],
            "team": best_team_s,
            "color": best_color_s,
        })

        match = pd.DataFrame()
        if "player_id" in pa_df.columns:
            match = pa_df[pa_df["player_id"] == player_id]
        if match.empty:
            match = pa_df[pa_df["player"] == selected_pitcher]
        if not match.empty:
            row = match.iloc[0]
            timeline_data.append({
                "season": s,
                "value": row["posterior_mean"],
                "hdi_low": row["hdi_low"],
                "hdi_high": row["hdi_high"],
            })
            _rate_row = {"season": s}
            for _rc in ["k_rate_posterior", "k_rate_hdi_low", "k_rate_hdi_high",
                         "bb_rate_posterior", "bb_rate_hdi_low", "bb_rate_hdi_high",
                         "hr_rate_posterior", "hr_rate_hdi_low", "hr_rate_hdi_high",
                         "pitcher_quality_score", "pqs_hdi_low", "pqs_hdi_high"]:
                if _rc in row.index and pd.notna(row.get(_rc)):
                    _rate_row[_rc] = row[_rc]
            rate_timeline.append(_rate_row)

    _has_rate_timeline = any("k_rate_posterior" in t for t in rate_timeline)
    _has_pqs_timeline = any("pitcher_quality_score" in t for t in rate_timeline)

    if timeline_data:
        st.divider()
        st.subheader("Season History")

        tl_df = pd.DataFrame(timeline_data)
        best_df = pd.DataFrame(best_player_data)
        lg_df = pd.DataFrame(league_avg_data)
        rate_tl_df = pd.DataFrame(rate_timeline) if rate_timeline else pd.DataFrame()
        rate_lg_df = pd.DataFrame(rate_league_avg) if rate_league_avg else pd.DataFrame()

        # Trim to player's season range
        player_seasons = tl_df["season"].tolist()
        min_s, max_s = min(player_seasons), max(player_seasons)
        lg_df = lg_df[(lg_df["season"] >= min_s) & (lg_df["season"] <= max_s)]
        best_df = best_df[(best_df["season"] >= min_s) & (best_df["season"] <= max_s)]
        if not rate_lg_df.empty:
            rate_lg_df = rate_lg_df[(rate_lg_df["season"] >= min_s) & (rate_lg_df["season"] <= max_s)]

        # --- Helper: build a rate stat chart with HDI bands ---
        def _build_rate_chart(rate_prefix, y_label, higher_is_better=True):
            mean_col = f"{rate_prefix}_posterior"
            lo_col = f"{rate_prefix}_hdi_low"
            hi_col = f"{rate_prefix}_hdi_high"
            if rate_tl_df.empty or mean_col not in rate_tl_df.columns:
                return None
            vals = rate_tl_df.dropna(subset=[mean_col])
            if vals.empty:
                return None
            fig = go.Figure()
            sd_col = f"{mean_col}_sd"
            if not rate_lg_df.empty and mean_col in rate_lg_df.columns:
                lg_vals = rate_lg_df.dropna(subset=[mean_col])
                if not lg_vals.empty:
                    _sd_arr = (lg_vals[sd_col] * 100).tolist() if sd_col in lg_vals.columns else [0] * len(lg_vals)
                    fig.add_trace(go.Scatter(
                        x=lg_vals["season"] + 0.15, y=lg_vals[mean_col] * 100,
                        mode="markers", name="Lg Avg",
                        marker=dict(color="rgba(160,160,160,0.8)", size=11),
                        error_y=dict(
                            type="data",
                            array=_sd_arr,
                            arrayminus=_sd_arr,
                            color="rgba(160,160,160,0.4)",
                            thickness=1.5,
                            width=4,
                        ),
                        hovertemplate="Season: %{x:.0f}<br>Lg Avg: %{y:.1f}%<br>±1 SD: %{error_y.array:.1f}%<extra></extra>",
                    ))
            c = primary_color.lstrip("#")
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            if len(vals) == 1:
                row_r = vals.iloc[0]
                err_hi = [(row_r[hi_col] - row_r[mean_col]) * 100] if hi_col in row_r.index and pd.notna(row_r.get(hi_col)) else None
                err_lo = [(row_r[mean_col] - row_r[lo_col]) * 100] if lo_col in row_r.index and pd.notna(row_r.get(lo_col)) else None
                fig.add_trace(go.Scatter(
                    x=vals["season"], y=vals[mean_col] * 100,
                    mode="markers", name=selected_pitcher,
                    marker=dict(color=primary_color, size=12),
                    error_y=dict(
                        type="data", array=err_hi, arrayminus=err_lo,
                        color=f"rgba({r},{g},{b},0.5)", thickness=2, width=8,
                    ) if err_hi else None,
                    hovertemplate=f"Season: %{{x}}<br>{y_label}: %{{y:.1f}}%<extra></extra>",
                ))
            else:
                if hi_col in vals.columns and lo_col in vals.columns:
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[hi_col] * 100,
                        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[lo_col] * 100,
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=f"rgba({r},{g},{b},0.15)", showlegend=False, hoverinfo="skip",
                    ))
                fig.add_trace(go.Scatter(
                    x=vals["season"], y=vals[mean_col] * 100,
                    mode="lines+markers", name=selected_pitcher,
                    line=dict(color=primary_color, width=2.5),
                    marker=dict(color=primary_color, size=12),
                    hovertemplate=f"Season: %{{x}}<br>{y_label}: %{{y:.1f}}%<extra></extra>",
                ))

            # --- Rate stat projections ---
            max_actual_s = int(vals["season"].max())
            proj_col = f"projected_{rate_prefix}"
            proj_lo_col = f"projected_{rate_prefix}_hdi_low"
            proj_hi_col = f"projected_{rate_prefix}_hdi_high"
            rate_proj_points = []
            for proj_s in range(max_actual_s + 1, max_actual_s + 4):
                rp_df = load_rate_stat_projections(proj_s, "pitcher", rate_prefix)
                if rp_df.empty or proj_col not in rp_df.columns:
                    continue
                rp_match = rp_df[rp_df["player_id"] == player_id] if "player_id" in rp_df.columns else pd.DataFrame()
                if rp_match.empty:
                    rp_match = rp_df[rp_df["player"] == selected_pitcher]
                if not rp_match.empty:
                    rate_proj_points.append(rp_match.iloc[0].to_dict() | {"season": proj_s})

            if rate_proj_points:
                rp_seasons = [p["season"] for p in rate_proj_points]
                rp_vals = [p[proj_col] * 100 for p in rate_proj_points]
                rp_hi = [p.get(proj_hi_col, p[proj_col]) * 100 for p in rate_proj_points]
                rp_lo = [p.get(proj_lo_col, p[proj_col]) * 100 for p in rate_proj_points]

                fig.add_vrect(
                    x0=max_actual_s + 0.5, x1=max(rp_seasons) + 0.5,
                    fillcolor="rgba(180,180,220,0.10)", line_width=0, layer="below",
                )
                fig.add_annotation(
                    x=(max_actual_s + 0.5 + max(rp_seasons) + 0.5) / 2,
                    y=1.0, yref="paper", yanchor="bottom",
                    text="Projected", showarrow=False,
                    font=dict(size=13, color="rgba(120,120,160,0.7)"),
                )
                fig.add_trace(go.Scatter(
                    x=rp_seasons, y=rp_hi, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=rp_seasons, y=rp_lo, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.10)",
                    showlegend=False, hoverinfo="skip",
                ))
                last_val = float(vals.iloc[-1][mean_col]) * 100
                fig.add_trace(go.Scatter(
                    x=[max_actual_s, rp_seasons[0]], y=[last_val, rp_vals[0]],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.4)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                if len(rp_seasons) > 1:
                    fig.add_trace(go.Scatter(
                        x=rp_seasons, y=rp_vals, mode="lines",
                        line=dict(color=f"rgba({r},{g},{b},0.5)", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig.add_trace(go.Scatter(
                    x=rp_seasons, y=rp_vals, mode="markers",
                    name="Projection", showlegend=False,
                    marker=dict(color="rgba(255,255,255,0)", size=12,
                                symbol="diamond-open",
                                line=dict(width=2.5, color=primary_color)),
                    customdata=[
                        [p.get("aging_effect", 0),
                         p.get(proj_lo_col, p[proj_col]) * 100,
                         p.get(proj_hi_col, p[proj_col]) * 100]
                        for p in rate_proj_points
                    ],
                    hovertemplate=(
                        "Projection %{x}<br>"
                        f"{y_label}: %{{y:.1f}}%<br>"
                        "89% HDI: [%{customdata[1]:.1f}%, %{customdata[2]:.1f}%]<br>"
                        "Aging effect: %{customdata[0]:+.4f}"
                        "<extra></extra>"
                    ),
                ))

            # Adjusted projection diamond marker for rate stats
            tt_col = f"true_talent_{rate_prefix}"
            if (pitcher_ranking is not None
                    and tt_col in pitcher_ranking.index
                    and pd.notna(pitcher_ranking.get(tt_col))):
                tt_val = pitcher_ranking[tt_col] * 100
                _hw_r = pitcher_ranking.get(f"{rate_prefix}_history_weight")
                if pd.isna(_hw_r):
                    _hw_r = pitcher_ranking.get("history_weight")
                _dev_r = pitcher_ranking.get(f"{rate_prefix}_deviation")
                c = primary_color.lstrip("#")
                _r, _g, _b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                _hover_parts = [f"Adjusted Projection {max_actual_s}", f"{y_label}: {tt_val:.1f}%"]
                if pd.notna(_hw_r):
                    _hover_parts.append(f"Prior weight: {_hw_r * 100:.0f}%")
                if pd.notna(_dev_r):
                    _hover_parts.append(f"vs Observed: {_dev_r * 100:+.1f}%")
                fig.add_trace(go.Scatter(
                    x=[max_actual_s + 0.2], y=[tt_val],
                    mode="markers", name="Adjusted Projection",
                    marker=dict(
                        symbol="diamond-open", size=12,
                        color=f"rgba({_r},{_g},{_b},0.6)",
                        line=dict(width=2.5, color=f"rgba({_r},{_g},{_b},0.6)"),
                    ),
                    hovertemplate="<br>".join(_hover_parts) + "<extra></extra>",
                ))

            x_max = max(rp_seasons) if rate_proj_points else max_actual_s
            all_tick_s = list(range(int(vals["season"].min()), x_max + 1))
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
        _history_options = []
        if _has_pqs_timeline:
            _history_options.append("PQS+")
        _history_options.append("EB/PA")
        if _has_rate_timeline:
            _history_options += ["K%", "BB%", "HR%"]
        _history_default = "PQS+" if _has_pqs_timeline else "EB/PA"
        if len(_history_options) > 1:
            _history_view = st.segmented_control(
                "View", _history_options, default=_history_default,
                key="pitcher_history_view",
            )
        else:
            _history_view = "EB/PA"

        if _history_view == "EB/PA":
            if _has_rate_timeline:
                st.caption("Bayesian model estimate with projections")

            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=lg_df["season"] + 0.15, y=lg_df["value"],
                mode="markers", name="Lg Avg",
                marker=dict(color="rgba(160,160,160,0.8)", size=11),
                error_y=dict(
                    type="data",
                    array=lg_df["sd"].tolist(),
                    arrayminus=lg_df["sd"].tolist(),
                    color="rgba(160,160,160,0.4)",
                    thickness=1.5,
                    width=4,
                ),
                hovertemplate="Season: %{x:.0f}<br>Lg Avg: %{y:.3f}<br>±1 SD: %{error_y.array:.3f}<extra></extra>",
            ))

            if not best_df.empty:
                first_best_season = best_df["season"].iloc[0]
                for _, brow in best_df.iterrows():
                    fig_timeline.add_trace(go.Scatter(
                        x=[brow["season"]], y=[brow["value"]],
                        mode="markers", name=brow["name"],
                        marker=dict(color="#DAA520", size=12, symbol="diamond",
                                    line=dict(width=1, color="white")),
                        hovertemplate=f"Season: %{{x}}<br>{brow['name']}: %{{y:.3f}}<extra></extra>",
                        legendgroup="best", showlegend=bool(brow["season"] == first_best_season),
                    ))
                fig_timeline.data[-len(best_df)].name = "Best Pitcher"

            if len(tl_df) == 1:
                row = tl_df.iloc[0]
                c = primary_color.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["value"],
                    mode="markers", name=selected_pitcher,
                    marker=dict(color=primary_color, size=12),
                    error_y=dict(
                        type="data",
                        array=[row["hdi_high"] - row["value"]],
                        arrayminus=[row["value"] - row["hdi_low"]],
                        color=f"rgba({r},{g},{b},0.5)",
                        thickness=2, width=6,
                    ),
                    hovertemplate="Season: %{x}<br>EB/PA Allowed: %{y:.3f}<extra></extra>",
                ))
            else:
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["hdi_high"],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["hdi_low"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({int(primary_color[1:3], 16)},{int(primary_color[3:5], 16)},{int(primary_color[5:7], 16)},0.15)",
                    showlegend=False, hoverinfo="skip",
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"], y=tl_df["value"],
                    mode="lines+markers", name=selected_pitcher,
                    line=dict(color=primary_color, width=2.5),
                    marker=dict(color=primary_color, size=12),
                    hovertemplate="Season: %{x}<br>EB/PA Allowed: %{y:.3f}<extra></extra>",
                ))

            # Multi-year projections
            proj_points = []
            for proj_season in range(max_s + 1, max_s + 4):
                proj_df = load_player_projections(proj_season, "pitcher")
                if proj_df.empty:
                    continue
                proj_match = proj_df[proj_df["player_id"] == player_id] if "player_id" in proj_df.columns else pd.DataFrame()
                if proj_match.empty:
                    proj_match = proj_df[proj_df["player"] == selected_pitcher]
                if not proj_match.empty:
                    proj_points.append(proj_match.iloc[0].to_dict() | {"season": proj_season})

            if proj_points:
                c = primary_color.lstrip("#")
                pr, pg, pb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                proj_seasons = [p["season"] for p in proj_points]
                proj_values = [p["projected_eb_pa"] for p in proj_points]
                proj_hdi_hi = [p["projected_hdi_high"] for p in proj_points]
                proj_hdi_lo = [p["projected_hdi_low"] for p in proj_points]

                fig_timeline.add_vrect(
                    x0=max_s + 0.5, x1=max(proj_seasons) + 0.5,
                    fillcolor="rgba(180,180,220,0.10)", line_width=0, layer="below",
                )
                fig_timeline.add_annotation(
                    x=(max_s + 0.5 + max(proj_seasons) + 0.5) / 2,
                    y=1.0, yref="paper", yanchor="bottom",
                    text="Projected", showarrow=False,
                    font=dict(size=13, color="rgba(120,120,160,0.7)"),
                )
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_hdi_hi,
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_hdi_lo,
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=f"rgba({pr},{pg},{pb},0.10)", showlegend=False, hoverinfo="skip",
                ))
                last_actual = tl_df[tl_df["season"] == max_s].iloc[0]
                fig_timeline.add_trace(go.Scatter(
                    x=[max_s, proj_seasons[0]],
                    y=[last_actual["value"], proj_values[0]],
                    mode="lines",
                    line=dict(color=f"rgba({pr},{pg},{pb},0.4)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
                if len(proj_seasons) > 1:
                    fig_timeline.add_trace(go.Scatter(
                        x=proj_seasons, y=proj_values, mode="lines",
                        line=dict(color=f"rgba({pr},{pg},{pb},0.5)", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_values, mode="markers",
                    name="Projection", showlegend=False,
                    marker=dict(color="rgba(255,255,255,0)", size=12, symbol="diamond-open",
                                line=dict(width=2.5, color=primary_color)),
                    customdata=[[p.get("aging_effect", 0), p["projected_hdi_low"], p["projected_hdi_high"]] for p in proj_points],
                    hovertemplate="Projection %{x}<br>EB/PA Allowed: %{y:.3f}<br>89% HDI: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>Aging effect: %{customdata[0]:+.3f}<extra></extra>",
                ))

            # Adjusted projection diamond marker (preseason + in-season combined)
            if (pitcher_ranking is not None
                    and "true_talent_eb_pa" in pitcher_ranking.index
                    and pd.notna(pitcher_ranking.get("true_talent_eb_pa"))):
                tt_val = pitcher_ranking["true_talent_eb_pa"]
                hw = pitcher_ranking.get("history_weight")
                dev = pitcher_ranking.get("deviation")
                c = primary_color.lstrip("#")
                _r, _g, _b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                _hover_parts = [f"Adjusted Projection {max_s}", f"EB/PA Allowed: {tt_val:.3f}"]
                if pd.notna(hw):
                    _hover_parts.append(f"Prior weight: {hw * 100:.0f}%")
                if pd.notna(dev):
                    _hover_parts.append(f"vs Observed: {dev:+.3f}")
                fig_timeline.add_trace(go.Scatter(
                    x=[max_s + 0.2], y=[tt_val],
                    mode="markers", name="Adjusted Projection",
                    marker=dict(
                        symbol="diamond-open", size=12,
                        color=f"rgba({_r},{_g},{_b},0.6)",
                        line=dict(width=2.5, color=f"rgba({_r},{_g},{_b},0.6)"),
                    ),
                    hovertemplate="<br>".join(_hover_parts) + "<extra></extra>",
                ))

            x_max = max(proj_seasons) if proj_points else max_s
            fig_timeline.update_layout(
                xaxis=dict(title="Season", title_font_size=14, tickfont_size=13, dtick=1, tickformat="d",
                           range=[min_s - 0.5, x_max + 0.5]),
                yaxis=dict(title="Est. Bases Allowed per PA  (↓ better)", title_font_size=14, tickfont_size=13),
                height=400, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=13)),
                dragmode=False,
            )
            fig_timeline.add_annotation(
                x=0.01, y=0.98, xref="paper", yref="paper",
                text="↓ Lower = Better", showarrow=False,
                font=dict(size=12, color="rgba(180, 60, 60, 0.7)"),
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.7)",
            )
            st.plotly_chart(fig_timeline, width="stretch", config=plotly_download_config(f"{_player_filename_slug(selected_pitcher)}_timeline_{season}", height=600))

            if len(timeline_data) == 1:
                st.caption("Only one season of data available. More history will accumulate over time.")

        # --- Rate stat views (lazy — only rendered when selected) ---
        if _has_rate_timeline and _history_view == "K%":
            st.caption("Bayesian strikeout rate with 89% credible interval and projections")
            _fig_k = _build_rate_chart("k_rate", "K%", higher_is_better=True)
            if _fig_k:
                st.plotly_chart(_fig_k, width="stretch", config=PLOTLY_CONFIG)

        if _has_rate_timeline and _history_view == "BB%":
            st.caption("Bayesian walk rate with 89% credible interval and projections")
            _fig_bb = _build_rate_chart("bb_rate", "BB%", higher_is_better=False)
            if _fig_bb:
                st.plotly_chart(_fig_bb, width="stretch", config=PLOTLY_CONFIG)

        if _has_rate_timeline and _history_view == "HR%":
            st.caption("Bayesian home run rate with 89% credible interval and projections")
            _fig_hr = _build_rate_chart("hr_rate", "HR%", higher_is_better=False)
            if _fig_hr:
                st.plotly_chart(_fig_hr, width="stretch", config=PLOTLY_CONFIG)

        # --- PQS+ trajectory (100-index scale, higher = better) ---
        if _has_pqs_timeline and _history_view == "PQS+":
            st.caption("Pitcher Quality Score (70% K% + 30% contact quality) with 89% credible interval — 100 = league avg, higher = better")
            pqs_col = "pitcher_quality_score"
            pqs_lo_col = "pqs_hdi_low"
            pqs_hi_col = "pqs_hdi_high"
            pqs_vals = rate_tl_df.dropna(subset=[pqs_col]) if not rate_tl_df.empty and pqs_col in rate_tl_df.columns else pd.DataFrame()
            if not pqs_vals.empty:
                fig_pqs = go.Figure()
                c = primary_color.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

                # League average reference line at PQS+ = 100
                fig_pqs.add_hline(
                    y=100, line_dash="dot",
                    line_color="rgba(160,160,160,0.6)", line_width=1.5,
                )
                fig_pqs.add_annotation(
                    x=0.99, y=100, xref="paper",
                    text="Lg Avg (100)", showarrow=False,
                    font=dict(size=11, color="rgba(160,160,160,0.8)"),
                    xanchor="right", yshift=12,
                )

                if len(pqs_vals) == 1:
                    row_p = pqs_vals.iloc[0]
                    err_hi = [row_p[pqs_hi_col] - row_p[pqs_col]] if pqs_hi_col in row_p.index and pd.notna(row_p.get(pqs_hi_col)) else None
                    err_lo = [row_p[pqs_col] - row_p[pqs_lo_col]] if pqs_lo_col in row_p.index and pd.notna(row_p.get(pqs_lo_col)) else None
                    fig_pqs.add_trace(go.Scatter(
                        x=pqs_vals["season"], y=pqs_vals[pqs_col],
                        mode="markers", name=selected_pitcher,
                        marker=dict(color=primary_color, size=12),
                        error_y=dict(
                            type="data", array=err_hi, arrayminus=err_lo,
                            color=f"rgba({r},{g},{b},0.5)", thickness=2, width=8,
                        ) if err_hi else None,
                        hovertemplate="Season: %{x}<br>PQS+: %{y:.0f}<extra></extra>",
                    ))
                else:
                    if pqs_hi_col in pqs_vals.columns and pqs_lo_col in pqs_vals.columns:
                        fig_pqs.add_trace(go.Scatter(
                            x=pqs_vals["season"], y=pqs_vals[pqs_hi_col],
                            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))
                        fig_pqs.add_trace(go.Scatter(
                            x=pqs_vals["season"], y=pqs_vals[pqs_lo_col],
                            mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor=f"rgba({r},{g},{b},0.15)", showlegend=False, hoverinfo="skip",
                        ))
                    fig_pqs.add_trace(go.Scatter(
                        x=pqs_vals["season"], y=pqs_vals[pqs_col],
                        mode="lines+markers", name=selected_pitcher,
                        line=dict(color=primary_color, width=2.5),
                        marker=dict(color=primary_color, size=12),
                        hovertemplate="Season: %{x}<br>PQS+: %{y:.0f}<extra></extra>",
                    ))

                # PQS projections
                max_actual_pqs_s = int(pqs_vals["season"].max())
                pqs_proj_points = []
                for proj_s in range(max_actual_pqs_s + 1, max_actual_pqs_s + 4):
                    pqs_proj_df = load_pqs_projections(proj_s)
                    if pqs_proj_df.empty or "projected_pqs" not in pqs_proj_df.columns:
                        continue
                    pqs_match = pqs_proj_df[pqs_proj_df["player_id"] == player_id] if "player_id" in pqs_proj_df.columns else pd.DataFrame()
                    if pqs_match.empty:
                        pqs_match = pqs_proj_df[pqs_proj_df["player"] == selected_pitcher]
                    if not pqs_match.empty:
                        pqs_proj_points.append(pqs_match.iloc[0].to_dict() | {"season": proj_s})

                if pqs_proj_points:
                    pp_seasons = [p["season"] for p in pqs_proj_points]
                    pp_vals = [p["projected_pqs"] for p in pqs_proj_points]
                    pp_hi = [p.get("projected_pqs_hdi_high", p["projected_pqs"]) for p in pqs_proj_points]
                    pp_lo = [p.get("projected_pqs_hdi_low", p["projected_pqs"]) for p in pqs_proj_points]

                    fig_pqs.add_vrect(
                        x0=max_actual_pqs_s + 0.5, x1=max(pp_seasons) + 0.5,
                        fillcolor="rgba(180,180,220,0.10)", line_width=0, layer="below",
                    )
                    fig_pqs.add_annotation(
                        x=(max_actual_pqs_s + 0.5 + max(pp_seasons) + 0.5) / 2,
                        y=1.0, yref="paper", yanchor="bottom",
                        text="Projected", showarrow=False,
                        font=dict(size=13, color="rgba(120,120,160,0.7)"),
                    )
                    fig_pqs.add_trace(go.Scatter(
                        x=pp_seasons, y=pp_hi, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_pqs.add_trace(go.Scatter(
                        x=pp_seasons, y=pp_lo, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.10)",
                        showlegend=False, hoverinfo="skip",
                    ))
                    last_pqs_val = float(pqs_vals.iloc[-1][pqs_col])
                    fig_pqs.add_trace(go.Scatter(
                        x=[max_actual_pqs_s, pp_seasons[0]], y=[last_pqs_val, pp_vals[0]],
                        mode="lines",
                        line=dict(color=f"rgba({r},{g},{b},0.4)", width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    if len(pp_seasons) > 1:
                        fig_pqs.add_trace(go.Scatter(
                            x=pp_seasons, y=pp_vals, mode="lines",
                            line=dict(color=f"rgba({r},{g},{b},0.5)", width=2, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        ))
                    fig_pqs.add_trace(go.Scatter(
                        x=pp_seasons, y=pp_vals, mode="markers",
                        name="Projection", showlegend=False,
                        marker=dict(color="rgba(255,255,255,0)", size=12,
                                    symbol="diamond-open",
                                    line=dict(width=2.5, color=primary_color)),
                        customdata=[
                            [p.get("projected_pqs_hdi_low", p["projected_pqs"]),
                             p.get("projected_pqs_hdi_high", p["projected_pqs"])]
                            for p in pqs_proj_points
                        ],
                        hovertemplate=(
                            "Projection %{x}<br>"
                            "PQS+: %{y:.0f}<br>"
                            "89% HDI: [%{customdata[0]:.0f}, %{customdata[1]:.0f}]"
                            "<extra></extra>"
                        ),
                    ))

                x_max_pqs = max(pp_seasons) if pqs_proj_points else max_actual_pqs_s
                all_tick_pqs = list(range(int(pqs_vals["season"].min()), x_max_pqs + 1))
                fig_pqs.update_layout(
                    xaxis=dict(title="Season", title_font_size=14, tickfont_size=13,
                               tickvals=all_tick_pqs, ticktext=[str(s) for s in all_tick_pqs],
                               range=[min(all_tick_pqs) - 0.5, max(all_tick_pqs) + 0.5]),
                    yaxis=dict(title="PQS+ (higher is better)", title_font_size=14, tickfont_size=13),
                    height=400, template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="center", x=0.5, font=dict(size=13)),
                    dragmode=False,
                )
                fig_pqs.add_annotation(
                    x=0.01, y=0.98, xref="paper", yref="paper",
                    text="↑ Higher = Better", showarrow=False,
                    font=dict(size=12, color="rgba(60, 140, 60, 0.7)"),
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)",
                )
                st.plotly_chart(fig_pqs, width="stretch", config=PLOTLY_CONFIG)


# =============================================================================
# LUCK REPORT (PITCHER PERSPECTIVE)
# =============================================================================

st.divider()
st.subheader("Luck Report")
st.caption(f"{season} Season — from the pitcher's perspective")

# actual_tb already computed after Hero section
total_actual_tb = pitcher_bb["actual_tb"].sum()
total_expected_tb = pitcher_bb["estimated_bases"].sum()
# Pitcher luck: expected - actual (positive = pitcher got lucky, allowed fewer bases than expected)
luck_score = total_expected_tb - total_actual_tb

# Compute league-wide percentiles (reuse cached league data)
actual_pct = expected_pct = luck_pct = None
_lp_p = _league_pcts_pitcher  # already loaded above
if _lp_p and not _lp_p.get("luck_per_player", pd.Series(dtype=float)).empty:
    _lp_counts = _lp_p["bb_counts"]
    _min30 = _lp_counts >= 30
    _lp_actual = _lp_p["actual_tb_sums"][_min30]
    _lp_expected = _lp_p["expected_tb_sums"][_min30]
    # Pitcher luck = expected - actual (negate the hitter-perspective cached values)
    _lp_pitcher_luck = -_lp_p["luck_per_player"][_min30]
    if selected_pitcher in _lp_pitcher_luck.index:
        # Actual TB allowed: lower = better, so invert percentile
        actual_pct = 100 - (_lp_actual < total_actual_tb).mean() * 100
        # Expected TB: lower = better, so invert percentile
        expected_pct = 100 - (_lp_expected < total_expected_tb).mean() * 100
        luck_pct = (_lp_pitcher_luck < luck_score).mean() * 100

# Luck metrics
lm1, lm2, lm3 = st.columns(3)
with lm1:
    st.metric("Actual TB Allowed", f"{total_actual_tb:.0f}")
    _actual_label = f"{_ordinal(int(actual_pct))} pct (fewer TB = better)" if actual_pct is not None else None
    render_percentile_bar(actual_pct, label=_actual_label)
with lm2:
    st.metric("Expected TB Allowed", f"{total_expected_tb:.1f}")
    _expected_label = f"{_ordinal(int(expected_pct))} pct (lower expected = better)" if expected_pct is not None else None
    render_percentile_bar(expected_pct, label=_expected_label)
with lm3:
    delta_color = "normal" if luck_score >= 0 else "inverse"
    tier = luck_tier_label(luck_pct)
    delta_text = tier if tier else ("Lucky" if luck_score > 0 else "Unlucky")
    st.metric("Net Lucky Bases", f"{luck_score:+.1f}", delta=delta_text, delta_color=delta_color,
              help="Expected TB minus actual TB allowed. Positive = lucky (allowed fewer bases than expected from contact quality). Negative = unlucky.")
    render_percentile_bar(luck_pct)

# --- Luck Accumulation Chart ---
st.markdown("#### Luck Over Time")
st.caption("Cumulative expected minus actual total bases allowed. Rising = getting lucky. Falling = unlucky.")

luck_ts = pitcher_bb.sort_values("date_parsed").copy()
luck_ts["cum_actual"] = luck_ts["actual_tb"].cumsum()
luck_ts["cum_expected"] = luck_ts["estimated_bases"].cumsum()
luck_ts["cum_luck"] = luck_ts["cum_expected"] - luck_ts["cum_actual"]
luck_ts["bb_num"] = range(1, len(luck_ts) + 1)

fig_luck = go.Figure()
fig_luck.add_trace(go.Scatter(
    x=luck_ts["bb_num"], y=luck_ts["cum_luck"],
    mode="lines", name="Cumulative Luck",
    line=dict(color=primary_color, width=2),
    hovertemplate="BB #%{x}<br>Cumulative Luck: %{y:.1f}<extra></extra>",
))
fig_luck.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_luck.update_layout(
    xaxis_title="Batted Ball #", yaxis_title="Cumulative Luck (TB)",
    height=400, template="plotly_white", dragmode=False,
)
st.plotly_chart(fig_luck, width="stretch", config=plotly_download_config(f"{_player_filename_slug(selected_pitcher)}_luck_{season}", height=600))


# =============================================================================
# PITCHING COMMAND (K% / BB% / HR%)
# =============================================================================

# K%/BB%/HR% — prefer Bayesian posterior from rankings, fall back to raw pa_counts
_k_rate_p = 20.0
_bb_rate_p = 8.0
_hr_rate_p = 3.0
_k_rate_bayesian_p = False
_bb_rate_bayesian_p = False
_hr_rate_bayesian_p = False
if pitcher_ranking is not None:
    if "k_rate_posterior" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("k_rate_posterior")):
        _k_rate_p = pitcher_ranking["k_rate_posterior"] * 100
        _k_rate_bayesian_p = True
    if "bb_rate_posterior" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("bb_rate_posterior")):
        _bb_rate_p = pitcher_ranking["bb_rate_posterior"] * 100
        _bb_rate_bayesian_p = True
    if "hr_rate_posterior" in pitcher_ranking.index and pd.notna(pitcher_ranking.get("hr_rate_posterior")):
        _hr_rate_p = pitcher_ranking["hr_rate_posterior"] * 100
        _hr_rate_bayesian_p = True

if _k_rate_bayesian_p or _bb_rate_bayesian_p or _hr_rate_bayesian_p:
    st.divider()
    st.subheader("Pitching Command")
    st.caption(
        "For pitchers: higher K% and lower BB%/HR% indicate better command. "
        "The Bayesian model adjusts for sample size — pitchers with fewer batters faced get "
        "pulled toward the league average, while large samples stay close to the raw number."
    )

    _pc1, _pc2, _pc3 = st.columns(3)
    with _pc1:
        _k_suffix_p = "" if _k_rate_bayesian_p else " (raw)"
        st.metric(f"K%{_k_suffix_p}", f"{_k_rate_p:.1f}%",
                  help="Strikeout rate per batter faced this season. **Higher is better** for pitchers — "
                  "a high K% means the pitcher misses more bats. "
                  "The Bayesian model estimates the pitcher's true strikeout rate by adjusting for sample size.")
        if _k_rate_bayesian_p and not pa_rankings.empty and "k_rate_posterior" in pa_rankings.columns:
            # K%: higher is better for pitchers
            _k_pct_p = (pa_rankings["k_rate_posterior"].dropna() < pitcher_ranking["k_rate_posterior"]).mean() * 100
            render_percentile_bar(_k_pct_p, container=_pc1)
            if "k_rate_hdi_low" in pitcher_ranking.index:
                st.caption(
                    f"89% CI: {pitcher_ranking['k_rate_hdi_low']*100:.1f}% – {pitcher_ranking['k_rate_hdi_high']*100:.1f}%"
                )
    with _pc2:
        _bb_suffix_p = "" if _bb_rate_bayesian_p else " (raw)"
        st.metric(f"BB%{_bb_suffix_p}", f"{_bb_rate_p:.1f}%",
                  help="Walk rate per batter faced this season. **Lower is better** for pitchers — "
                  "fewer walks means better command. "
                  "The Bayesian model estimates the pitcher's true walk rate by adjusting for sample size.")
        if _bb_rate_bayesian_p and not pa_rankings.empty and "bb_rate_posterior" in pa_rankings.columns:
            # BB%: lower is better for pitchers, so invert
            _bb_pct_p = (pa_rankings["bb_rate_posterior"].dropna() > pitcher_ranking["bb_rate_posterior"]).mean() * 100
            render_percentile_bar(_bb_pct_p, container=_pc2)
            st.caption("↓ Lower is better")
            if "bb_rate_hdi_low" in pitcher_ranking.index:
                st.caption(
                    f"89% CI: {pitcher_ranking['bb_rate_hdi_low']*100:.1f}% – {pitcher_ranking['bb_rate_hdi_high']*100:.1f}%"
                )
    with _pc3:
        _hr_suffix_p = "" if _hr_rate_bayesian_p else " (raw)"
        st.metric(f"HR%{_hr_suffix_p}", f"{_hr_rate_p:.1f}%",
                  help="Home run rate per batter faced this season. **Lower is better** for pitchers — "
                  "fewer home runs allowed means better pitch quality. "
                  "The Bayesian model estimates the pitcher's true HR rate by adjusting for sample size.")
        if _hr_rate_bayesian_p and not pa_rankings.empty and "hr_rate_posterior" in pa_rankings.columns:
            # HR%: lower is better for pitchers, so invert
            _hr_pct_p = (pa_rankings["hr_rate_posterior"].dropna() > pitcher_ranking["hr_rate_posterior"]).mean() * 100
            render_percentile_bar(_hr_pct_p, container=_pc3)
            st.caption("↓ Lower is better")
            if "hr_rate_hdi_low" in pitcher_ranking.index:
                st.caption(
                    f"89% CI: {pitcher_ranking['hr_rate_hdi_low']*100:.1f}% – {pitcher_ranking['hr_rate_hdi_high']*100:.1f}%"
                )

    # PQS+ distribution chart — capstone after K%/BB%/HR% rate stats
    if (not pa_rankings.empty and "pitcher_quality_score" in pa_rankings.columns
            and "pitcher_quality_score" in pitcher_ranking.index
            and not pd.isna(pitcher_ranking.get("pitcher_quality_score", float("nan")))):
        _pqs_pop = pa_rankings["pitcher_quality_score"].dropna()
        if len(_pqs_pop) > 10:
            _pqs_val_dist = pitcher_ranking["pitcher_quality_score"]
            _pqs_pct_dist = (_pqs_pop < _pqs_val_dist).mean() * 100

            fig_dist = go.Figure()

            # Tier shading bands
            _x_lo = max(40, _pqs_pop.min() - 5)
            _x_hi = min(170, _pqs_pop.max() + 5)
            for _t_lo, _t_hi, _t_color in [
                (123, _x_hi, "rgba(40,167,69,0.07)"),
                (108, 123, "rgba(40,167,69,0.035)"),
                (77, 92, "rgba(255,165,0,0.04)"),
                (_x_lo, 77, "rgba(220,53,69,0.06)"),
            ]:
                fig_dist.add_vrect(x0=_t_lo, x1=_t_hi, fillcolor=_t_color,
                                   line_width=0, layer="below")

            # League distribution
            fig_dist.add_trace(go.Histogram(
                x=_pqs_pop, nbinsx=30, histnorm="probability density",
                marker_color="rgba(160,160,160,0.45)", name="All Pitchers",
                hoverinfo="skip", showlegend=False,
            ))

            # League average reference
            fig_dist.add_vline(x=100, line_dash="dot",
                               line_color="rgba(140,140,140,0.6)", line_width=1.5)
            fig_dist.add_annotation(
                x=100, y=1.0, yref="paper", yanchor="bottom",
                text="Lg Avg", showarrow=False,
                font=dict(size=10, color="rgba(140,140,140,0.8)"), yshift=2,
            )

            # Player marker
            fig_dist.add_vline(x=_pqs_val_dist, line_color=primary_color, line_width=2.5)
            _arrow_side = "left" if _pqs_val_dist > 110 else "right"
            fig_dist.add_annotation(
                x=_pqs_val_dist, y=0.92, yref="paper",
                text=f"<b>{selected_pitcher}</b><br>PQS+ {_pqs_val_dist:.0f} ({_pqs_pct_dist:.0f}th pct)",
                showarrow=True, arrowhead=0, arrowwidth=1.5,
                arrowcolor=primary_color,
                ax=-80 if _arrow_side == "left" else 80, ay=-25,
                font=dict(size=12, color=primary_color),
                bgcolor="rgba(255,255,255,0.85)", borderpad=4,
                xanchor="right" if _arrow_side == "left" else "left",
            )

            # Tier labels at top
            for _lbl, _lx in [("Poor", (_x_lo + 77) / 2), ("Below\nAvg", 84.5),
                               ("Avg", 100), ("Above\nAvg", 115.5), ("Elite", min((_x_hi + 123) / 2, 140))]:
                fig_dist.add_annotation(
                    x=_lx, y=1.0, yref="paper", yanchor="bottom",
                    text=f"<span style='color:rgba(120,120,120,0.6)'>{_lbl}</span>",
                    showarrow=False, font=dict(size=9), yshift=14,
                )

            fig_dist.update_layout(
                xaxis=dict(title="PQS+", range=[_x_lo, _x_hi], title_font_size=13, tickfont_size=12),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                height=250, template="plotly_white",
                margin=dict(t=40, b=40, l=20, r=20),
                dragmode=False,
            )
            st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)

    with st.expander("What is PQS+?"):
        st.markdown("""
**Pitcher Quality Score (PQS+)** is a composite metric that combines a pitcher's
strikeout ability (K%) with their batted ball quality allowed (EB/PA) into a single number
on a 100-index scale.

**How to read it:** 100 is league average, **higher is better**. Think of it like Stuff+:
every 15 points is roughly one standard deviation. A PQS+ of 115 means the pitcher is
about one standard deviation above average, not "15% better" (that's how OPS+/ERA+ work,
which are ratio-based). PQS+ is z-score-based, so the numbers represent distance from
the pack rather than a percentage difference.

**Formula:** 70% K% + 30% EB/PA (z-scored, then scaled to 100-index)

| PQS+ Range | Tier |
|------------|------|
| 123+ | Elite |
| 108 to 123 | Above Avg |
| 92 to 108 | Average |
| 77 to 92 | Below Avg |
| Below 77 | Poor |

**Why these weights?** K% gets 70% because strikeouts are the most stable and
predictive pitcher skill. Contact quality (EB/PA) gets 30% — it matters, but is noisier.
The composite is validated at r=0.45 predicting next-year wOBA allowed, beating
K% alone (r=0.44).
        """)


# =============================================================================
# CONTACT QUALITY ALLOWED
# =============================================================================

st.divider()
st.subheader("Contact Quality Allowed")
st.caption(f"{season} Season")

_pla = pitcher_bb["launch_angle"]
pitcher_bb["bb_type"] = np.select(
    [_pla.isna(), _pla < 10, _pla < 25, _pla < 50],
    ["Unknown", "Ground Ball", "Line Drive", "Fly Ball"],
    default="Pop Up",
)

# --- Distribution charts row ---
col_ev, col_la = st.columns(2)

with col_ev:
    st.markdown("#### Exit Velocity Allowed")
    st.caption("← Leftward shift = better for pitcher")
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Histogram(
        x=bb_df["launch_speed"], name="League", opacity=0.3,
        marker_color="gray", histnorm="probability density", nbinsx=40,
    ))
    fig_ev.add_trace(go.Histogram(
        x=pitcher_bb["launch_speed"], name=selected_pitcher, opacity=0.6,
        marker_color=primary_color, histnorm="probability density", nbinsx=30,
    ))
    fig_ev.update_layout(
        barmode="overlay", xaxis_title="Exit Velocity (mph)", yaxis_title="Density",
        height=350, template="plotly_white",
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        dragmode=False,
    )
    st.plotly_chart(fig_ev, width="stretch", config=PLOTLY_CONFIG)

with col_la:
    st.markdown("#### Launch Angle Allowed")
    st.caption("More ground balls (low LA) and pop ups (high LA) = better")
    fig_la = go.Figure()
    fig_la.add_trace(go.Histogram(
        x=bb_df["launch_angle"], name="League", opacity=0.3,
        marker_color="gray", histnorm="probability density", nbinsx=40,
    ))
    fig_la.add_trace(go.Histogram(
        x=pitcher_bb["launch_angle"], name=selected_pitcher, opacity=0.6,
        marker_color=primary_color, histnorm="probability density", nbinsx=30,
    ))
    fig_la.update_layout(
        barmode="overlay", xaxis_title="Launch Angle (deg)", yaxis_title="Density",
        height=350, template="plotly_white",
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        dragmode=False,
    )
    st.plotly_chart(fig_la, width="stretch", config=PLOTLY_CONFIG)

# --- EV x LA Scatter + Spray Chart ---
col_evla, col_spray = st.columns(2)

with col_evla:
    st.markdown("#### Exit Velo x Launch Angle Allowed")
    st.caption("Each batted ball faced, colored by estimated bases.")
    evla_data = pitcher_bb.dropna(subset=["launch_speed", "launch_angle"])
    if not evla_data.empty:
        fig_evla = go.Figure()
        fig_evla.add_trace(go.Scatter(
            x=evla_data["launch_speed"], y=evla_data["launch_angle"],
            mode="markers",
            marker=dict(
                color=evla_data["estimated_bases"], colorscale="RdYlGn", size=6,
                colorbar=dict(title="Est.<br>Bases"),
            ),
            customdata=np.stack([
                evla_data["estimated_bases"], evla_data["actual_result"],
            ], axis=-1),
            hovertemplate=(
                "EV: %{x:.1f} mph<br>LA: %{y:.0f}&deg;<br>"
                "Est. Bases: %{customdata[0]:.2f}<br>Result: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_evla.add_hline(y=10, line_dash="dash", line_color="gray", opacity=0.3,
                           annotation_text="GB/LD", annotation_position="bottom right",
                           annotation_font_color="gray", annotation_font_size=10)
        fig_evla.add_hline(y=25, line_dash="dash", line_color="gray", opacity=0.3,
                           annotation_text="LD/FB", annotation_position="bottom right",
                           annotation_font_color="gray", annotation_font_size=10)
        fig_evla.add_vline(x=95, line_dash="dash", line_color="gray", opacity=0.3,
                           annotation_text="Hard Hit", annotation_position="top left",
                           annotation_font_color="gray", annotation_font_size=10)
        fig_evla.update_layout(
            xaxis_title="Exit Velocity (mph)", yaxis_title="Launch Angle (&deg;)",
            height=500, template="plotly_white", dragmode=False,
        )
        st.plotly_chart(fig_evla, width="stretch", config=plotly_download_config(f"{_player_filename_slug(selected_pitcher)}_ev_la_{season}"))

with col_spray:
    st.markdown("#### Spray Chart")
    st.caption("Where batters hit the ball against this pitcher.")
    if "coord_x" in pitcher_bb.columns and "coord_y" in pitcher_bb.columns:
        spray_data = pitcher_bb.dropna(subset=["coord_x", "coord_y"])
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
                marker=dict(
                    color=spray_data["estimated_bases"], colorscale="RdYlGn", size=6,
                    colorbar=dict(title="Est.<br>Bases"),
                ),
                customdata=np.stack([
                    spray_data["estimated_bases"], spray_data["launch_speed"],
                    spray_data["actual_result"],
                ], axis=-1),
                hovertemplate=(
                    "Est. Bases: %{customdata[0]:.2f}<br>Exit Velo: %{customdata[1]:.1f}<br>"
                    "Result: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ))

            fig_spray.update_layout(
                height=500, template="plotly_white",
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False, autorange="reversed"),
                dragmode=False,
            )
            st.plotly_chart(fig_spray, width="stretch", config=plotly_download_config(f"{_player_filename_slug(selected_pitcher)}_spray_{season}", width=800, height=800))

            if "spray_direction" in spray_data.columns:
                player_dirs = spray_data["spray_direction"].value_counts(normalize=True) * 100
                league_spray = bb_df.dropna(subset=["coord_x", "coord_y"])
                if "spray_direction" in league_spray.columns:
                    lg_dirs = league_spray["spray_direction"].value_counts(normalize=True) * 100
                    parts = []
                    for d in ["Pull", "Center", "Oppo"]:
                        p = player_dirs.get(d, 0)
                        lg = lg_dirs.get(d, 0)
                        parts.append(f"{d}: {p:.0f}% (Lg: {lg:.0f}%)")
                    st.caption(" · ".join(parts))

# --- Estimated Bases Distribution Chart ---
st.markdown("#### Estimated Bases Distribution")
st.caption("← More weight in lower bins = better for pitcher")

eb_bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, float("inf")]
eb_labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
pitcher_eb_cats = pd.cut(pitcher_bb["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False)
league_eb_cats = pd.cut(bb_df["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False)

pitcher_eb_dist = pitcher_eb_cats.value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100
league_eb_dist = league_eb_cats.value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100

fig_eb_dist = go.Figure()
fig_eb_dist.add_trace(go.Bar(
    x=eb_labels, y=pitcher_eb_dist.values,
    name=selected_pitcher, marker_color=primary_color, opacity=0.85,
))
fig_eb_dist.add_trace(go.Bar(
    x=eb_labels, y=league_eb_dist.values,
    name="League", marker_color="gray", opacity=0.5,
))
fig_eb_dist.update_layout(
    barmode="group", xaxis_title="Estimated Bases", yaxis_title="% of Batted Balls Faced",
    height=350, template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    margin=dict(t=30), dragmode=False,
)
st.plotly_chart(fig_eb_dist, width="stretch", config=PLOTLY_CONFIG)

# --- Contact Type Breakdown ---
st.markdown("#### Contact Type Breakdown")

type_order = ["Ground Ball", "Line Drive", "Fly Ball", "Pop Up"]

type_stats = pitcher_bb.groupby("bb_type").agg(
    count=("estimated_bases", "count"),
    avg_eb=("estimated_bases", "mean"),
    avg_ev=("launch_speed", "mean"),
    avg_la=("launch_angle", "mean"),
).reset_index()
type_stats["pct"] = (type_stats["count"] / type_stats["count"].sum() * 100).round(1)

# Pull% per type
if "spray_direction" in pitcher_bb.columns:
    pull_by_type = pitcher_bb.groupby("bb_type").apply(
        lambda g: (g["spray_direction"] == "Pull").mean() * 100
    ).rename("pull_pct")
    type_stats = type_stats.merge(pull_by_type, left_on="bb_type", right_index=True, how="left")
else:
    type_stats["pull_pct"] = float("nan")

# League type stats for comparison (vectorized, on copy to avoid mutating cache)
_lg_la = bb_df["launch_angle"]
_lg_bb_type = np.select(
    [_lg_la.isna(), _lg_la < 10, _lg_la < 25, _lg_la < 50],
    ["Unknown", "Ground Ball", "Line Drive", "Fly Ball"],
    default="Pop Up",
)
league_type_stats = bb_df.assign(bb_type=_lg_bb_type).groupby("bb_type").agg(
    lg_avg_eb=("estimated_bases", "mean"),
).reset_index()
type_stats = type_stats.merge(league_type_stats, on="bb_type", how="left")

type_stats["bb_type"] = pd.Categorical(type_stats["bb_type"], categories=type_order, ordered=True)
type_stats = type_stats.sort_values("bb_type").reset_index(drop=True)

ct_display = type_stats.rename(columns={
    "bb_type": "Type", "count": "Count", "pct": "%",
    "avg_ev": "Avg EV", "avg_la": "Avg LA",
    "avg_eb": "Avg Est. Bases", "lg_avg_eb": "Lg Avg EB",
    "pull_pct": "Pull%",
})
ct_col_config = {
    "Avg EV": st.column_config.NumberColumn(format="%.1f"),
    "Avg LA": st.column_config.NumberColumn(format="%.1f°"),
    "Avg Est. Bases": st.column_config.NumberColumn(format="%.3f"),
    "Lg Avg EB": st.column_config.NumberColumn(format="%.3f"),
    "Pull%": st.column_config.NumberColumn(format="%.1f%%"),
    "%": st.column_config.NumberColumn(format="%.1f%%"),
}
st.dataframe(ct_display, hide_index=True, width="stretch", column_config=ct_col_config)

# --- vs LHH / vs RHH Splits ---
if not metadata_df.empty and "player" in pitcher_bb.columns:
    # Map batter names to bat side
    bat_side_map = metadata_df.set_index("player_name")["bat_side"].to_dict()
    pitcher_bb["batter_hand"] = pitcher_bb["player"].map(bat_side_map)

    splits_available = pitcher_bb["batter_hand"].dropna()
    if len(splits_available) > 0:
        st.markdown("#### vs LHH / vs RHH")

        split_col_l, split_col_r = st.columns(2)
        for hand_label, hand_val, col in [("vs LHH", "L", split_col_l), ("vs RHH", "R", split_col_r)]:
            with col:
                with st.container(border=True):
                    subset = pitcher_bb[pitcher_bb["batter_hand"] == hand_val]
                    n = len(subset)
                    if n == 0:
                        st.markdown(f"**{hand_label}**: No data")
                    else:
                        st.markdown(f"**{hand_label}**")
                        s_ev = subset["launch_speed"].mean()
                        s_eb = subset["estimated_bases"].mean()
                        s_barrel = is_barrel_vectorized(subset["launch_speed"], subset["launch_angle"]).mean() * 100
                        s1, s2 = st.columns(2)
                        s1.metric("Count", f"{n}")
                        s2.metric("Avg EV", f"{s_ev:.1f}")
                        s3, s4 = st.columns(2)
                        s3.metric("Avg Est. Bases", f"{s_eb:.3f}")
                        s4.metric("Barrel Rate", f"{s_barrel:.1f}%")


# =============================================================================
# BATTED BALL LOG (PITCHER PERSPECTIVE)
# =============================================================================

st.divider()
st.subheader("Batted Ball Log")
st.caption(f"{season} Season — batted balls faced")

# --- Luckiest Outs (pitcher got lucky — high xBA outs) ---
st.markdown("#### Luckiest Outs")
st.caption("Outs with the highest expected batting average — the pitcher got lucky on these.")

outs = pitcher_bb[pitcher_bb["actual_tb"] == 0].copy()
if not outs.empty:
    lucky_outs = outs.nlargest(10, "xba")

    out_display_cols = ["date", "player", "launch_speed", "launch_angle",
                        "spray_direction", "estimated_bases", "xba"]
    if "play_id" in lucky_outs.columns:
        lucky_outs["video"] = lucky_outs["play_id"].apply(build_video_url)
        out_display_cols.append("video")

    available_cols = [c for c in out_display_cols if c in lucky_outs.columns]
    out_display = lucky_outs[available_cols].copy()

    col_rename = {
        "date": "Date", "player": "Batter", "launch_speed": "Exit Velo",
        "launch_angle": "Launch Angle", "spray_direction": "Spray",
        "estimated_bases": "Est. Bases", "xba": "xBA", "video": "Video",
    }
    out_display = out_display.rename(columns=col_rename)

    col_config = {
        "Exit Velo": st.column_config.NumberColumn(format="%.1f mph"),
        "Launch Angle": st.column_config.NumberColumn(format="%d°"),
        "Est. Bases": st.column_config.NumberColumn(format="%.2f"),
        "xBA": st.column_config.NumberColumn(format="%.3f"),
    }
    if "Video" in out_display.columns:
        col_config["Video"] = st.column_config.LinkColumn(display_text="Watch")

    st.dataframe(out_display, hide_index=True, width="stretch",
                  column_config=col_config)
else:
    st.info("No outs recorded.")

# --- Unluckiest Hits Allowed (low xBA hits — pitcher got unlucky) ---
st.markdown("#### Unluckiest Hits Allowed")
st.caption("Hits with the lowest expected batting average — the pitcher got unlucky on these.")

hits = pitcher_bb[pitcher_bb["actual_tb"] > 0].copy()
if not hits.empty:
    unlucky_hits = hits.nsmallest(10, "xba")

    hit_display_cols = ["date", "player", "launch_speed", "launch_angle",
                        "spray_direction", "actual_result", "estimated_bases", "xba"]
    if "play_id" in unlucky_hits.columns:
        unlucky_hits["video"] = unlucky_hits["play_id"].apply(build_video_url)
        hit_display_cols.append("video")

    available_cols = [c for c in hit_display_cols if c in unlucky_hits.columns]
    hit_display = unlucky_hits[available_cols].copy()

    col_rename_hits = {
        "date": "Date", "player": "Batter", "launch_speed": "Exit Velo",
        "launch_angle": "Launch Angle", "spray_direction": "Spray",
        "actual_result": "Result", "estimated_bases": "Est. Bases",
        "xba": "xBA", "video": "Video",
    }
    hit_display = hit_display.rename(columns=col_rename_hits)

    hit_col_config = {
        "Exit Velo": st.column_config.NumberColumn(format="%.1f mph"),
        "Launch Angle": st.column_config.NumberColumn(format="%d°"),
        "Est. Bases": st.column_config.NumberColumn(format="%.2f"),
        "xBA": st.column_config.NumberColumn(format="%.3f"),
    }
    if "Video" in hit_display.columns:
        hit_col_config["Video"] = st.column_config.LinkColumn(display_text="Watch")

    st.dataframe(hit_display, hide_index=True, width="stretch",
                  column_config=hit_col_config)
else:
    st.info("No hits allowed.")

# --- All Batted Balls Faced ---
st.markdown("#### All Batted Balls Faced")
st.caption("Full season log. Luck = expected bases minus actual bases (positive = pitcher got lucky).")

all_bb_display = pitcher_bb.sort_values("date_parsed", ascending=False).copy()
all_bb_display["luck"] = all_bb_display["estimated_bases"] - all_bb_display["actual_tb"]

all_bb_cols = ["date", "player", "launch_speed", "launch_angle",
               "spray_direction", "actual_result", "estimated_bases", "xba", "luck"]
if "play_id" in all_bb_display.columns:
    all_bb_display["video"] = all_bb_display["play_id"].apply(build_video_url)
    all_bb_cols.append("video")

available_cols = [c for c in all_bb_cols if c in all_bb_display.columns]
all_bb_show = all_bb_display[available_cols].copy()

all_bb_rename = {
    "date": "Date", "player": "Batter", "launch_speed": "Exit Velo",
    "launch_angle": "Launch Angle", "spray_direction": "Spray",
    "actual_result": "Result", "estimated_bases": "Est. Bases",
    "xba": "xBA", "luck": "Luck", "video": "Video",
}
all_bb_show = all_bb_show.rename(columns=all_bb_rename)

all_bb_col_config = {
    "Exit Velo": st.column_config.NumberColumn(format="%.1f mph"),
    "Launch Angle": st.column_config.NumberColumn(format="%d°"),
    "Est. Bases": st.column_config.NumberColumn(format="%.2f"),
    "xBA": st.column_config.NumberColumn(format="%.3f"),
    "Luck": st.column_config.NumberColumn(format="%+.2f"),
}
if "Video" in all_bb_show.columns:
    all_bb_col_config["Video"] = st.column_config.LinkColumn(display_text="Watch")

st.dataframe(all_bb_show, hide_index=True, width="stretch",
              column_config=all_bb_col_config, height=400)

# Download CSV
csv_data = all_bb_show.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv_data,
    file_name=f"{selected_pitcher.replace(' ', '_')}_batted_balls_faced_{season}.csv",
    mime="text/csv",
)


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()
with st.expander("Methodology"):
    st.markdown(f"""
**Estimated Bases** is the model-predicted expected bases for each batted ball,
based on the probability of each outcome (single, double, triple, home run)
given the exit velocity, launch angle, spray angle, and ballpark.

**Est. Bases Allowed per PA** is a statistical estimate of true production allowed per
plate appearance that accounts for walks (1 base), HBP (1 base), and strikeouts (0 bases)
alongside batted ball quality. Pitchers with fewer plate appearances are adjusted toward the
league average. **Lower is better** — the best pitchers allow the fewest estimated bases per PA.
The comparison bar shows the pitcher vs. the best pitcher and league average, with uncertainty ranges.

**Historical Timeline** tracks a pitcher's Est. Bases Allowed / PA across seasons with
uncertainty bars. The diamond shows the best pitcher each season for reference.

**Net Lucky Bases** (pitcher perspective) = expected total bases minus actual total bases allowed.
Positive = lucky (allowed fewer bases than contact quality suggested).
Negative = unlucky. Over a full season, extreme values tend to regress toward zero.
Tier labels: Very Unlucky (0-10th pct), Unlucky (10-30th), Luck Neutral (30-70th),
Lucky (70-90th), Very Lucky (90-100th).

**Barrel**: Exit velocity >= 98 mph with a launch angle in the "sweet spot" zone (26-30+
degrees, expanding with higher EV). Barrels produce the highest expected bases.

**Contact types**: Ground Ball (LA < 10°), Line Drive (10-25°), Fly Ball (25-50°),
Pop Up (50°+).

**Pitcher Quality Score (PQS+)** combines 70% K% and 30% EB/PA into a single composite on a
100-index scale (100 = league average, higher = better). Like Stuff+, every 15 points is roughly
one standard deviation. Validated at r=0.45 predicting next-year wOBA allowed.

**vs LHH / vs RHH**: Splits based on batter handedness — how the pitcher performs against
left-handed hitters vs. right-handed hitters.

{"**Video links**: Click to watch the play on Baseball Savant." if "play_id" in bb_df.columns else ""}
    """)

render_home_link()
