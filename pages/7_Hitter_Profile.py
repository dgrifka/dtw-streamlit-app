"""
Hitter Profile Page

Individual hitter deep-dive: contact quality, luck report, and batted ball
visualizations with Bayesian uncertainty from the DTW Simulator model.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    load_player_projections, load_rate_stat_projections, resolve_player_id,
)
from utils.team_mappings import (
    get_team_color, get_team_logo_url, get_short_name,
    TEAM_NAME_MAPPING,
)
from utils.player_helpers import (
    normalize_name, build_headshot_url, build_video_url,
    categorize_launch_angle, is_barrel, is_barrel_vectorized,
    _ordinal, _percentile_color,
    render_percentile_bar, luck_tier_label,
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
        season = st.selectbox("Season", options=available_seasons, index=0, key="pd_season")
    else:
        season = pd.Timestamp.now().year

# Load data
bb_df = load_batted_balls(season)
if bb_df.empty:
    st.title("Hitter Profile")
    st.info(f"No batted ball data available for {season}.")
    st.stop()

# Load supplementary data (may be empty — degrade gracefully)
metadata_df = load_player_metadata(season)
pa_rankings = load_player_evaluations_pa(season, "hitter")
pa_counts_df = load_pa_counts(season)

# Load multi-season PA rankings for historical timeline
all_season_pa_rankings = load_all_season_pa_rankings("hitter")


# =============================================================================
# PLAYER SEARCH
# =============================================================================

st.title("Hitter Profile")

# Check if we arrived via query param (cross-page linking)
query_player = st.query_params.get("player", "")

# Build unique player list — group by (player, team) to keep same-name players separate
player_teams = (
    bb_df.groupby(["player", "team"]).size()
    .reset_index(name="count")
    .drop(columns="count")
)

# Always show "Name (TEAM)" for every player
player_teams["display"] = player_teams.apply(
    lambda r: f"{r['player']} ({r['team']})", axis=1
)
display_to_name = dict(zip(player_teams["display"], player_teams["player"]))
display_to_team = dict(zip(player_teams["display"], player_teams["team"]))
display_list = sorted(player_teams["display"].unique())

# Resolve default selection (from query param or random)
default_index = 0
if query_player and query_player in display_list:
    default_index = display_list.index(query_player)
elif query_player:
    # Try fuzzy match on query param
    qn = normalize_name(query_player)
    fuzzy = [i for i, d in enumerate(display_list) if qn in normalize_name(d)]
    if fuzzy:
        default_index = fuzzy[0]
else:
    # No query param — pick a random player with 50+ batted balls
    eligible_names = bb_df.groupby(["player", "team"]).size()
    eligible_names = eligible_names[eligible_names >= 50].index.tolist()
    eligible_displays = [f"{n} ({t})" for n, t in eligible_names]
    if not eligible_displays:
        eligible_displays = display_list
    random_pick = random.choice(eligible_displays)
    if random_pick in display_list:
        default_index = display_list.index(random_pick)

# Player selector dropdown
search_col, shuffle_col = st.columns([3, 1], vertical_alignment="bottom")
with search_col:
    selected_display = st.selectbox(
        "Select player",
        options=display_list,
        index=default_index,
        key="pd_player_select",
    )
with shuffle_col:
    if st.button("Shuffle", width="stretch"):
        # Eligible: players with 50+ BBs, mapped to display labels
        eligible_names = bb_df.groupby(["player", "team"]).size()
        eligible_names = eligible_names[eligible_names >= 50].index.tolist()
        eligible_displays = [f"{n} ({t})" for n, t in eligible_names]
        if not eligible_displays:
            eligible_displays = display_list
        st.query_params["player"] = random.choice(eligible_displays)
        st.rerun()

selected_player = display_to_name.get(selected_display, selected_display)
selected_team_hint = display_to_team.get(selected_display)

# Filter batted ball data to this player + team
player_bb = bb_df[(bb_df["player"] == selected_player) & (bb_df["team"] == selected_team_hint)].copy()
if player_bb.empty:
    st.warning(f"No batted ball data found for {selected_player}.")
    st.stop()

# Resolve team (most recent)
player_bb = player_bb.sort_values("date_parsed")
player_team_short = player_bb["team"].iloc[-1]

# Update URL with selected player for bookmarking/sharing
st.query_params["player"] = selected_display

# League averages for context
league_avg_eb = bb_df["estimated_bases"].mean()
league_avg_ev = bb_df["launch_speed"].mean()


# =============================================================================
# HERO SECTION
# =============================================================================

st.divider()

# Resolve player_id (metadata → rankings fallback)
# Look up metadata for demographics (filter by name + team to disambiguate)
player_meta = None
if not metadata_df.empty:
    meta_match = metadata_df[
        (metadata_df["player_name"] == selected_player) &
        (metadata_df["team"] == player_team_short)
    ]
    if meta_match.empty:
        # Fallback: name only
        meta_match = metadata_df[metadata_df["player_name"] == selected_player]
    if meta_match.empty:
        player_norm = normalize_name(selected_player)
        meta_match = metadata_df[
            metadata_df["player_name"].apply(normalize_name).str.contains(player_norm)
        ]
    if not meta_match.empty:
        player_meta = meta_match.iloc[0]

# Resolve player_id from metadata (already team-filtered) or rankings
player_id = None
if player_meta is not None and "player_id" in player_meta.index:
    player_id = int(player_meta["player_id"])
if player_id is None:
    player_id = resolve_player_id(selected_player, metadata_df, pa_rankings)

# Look up Bayesian ranking (filter by name + team to disambiguate)
player_ranking = None
if not pa_rankings.empty:
    rank_match = pa_rankings[
        (pa_rankings["player"] == selected_player) &
        (pa_rankings["team"] == player_team_short)
    ]
    if rank_match.empty:
        # Fallback: name only
        rank_match = pa_rankings[pa_rankings["player"] == selected_player]
    if not rank_match.empty:
        player_ranking = rank_match.iloc[0]

# Team color
primary_color, secondary_color = get_team_color(player_team_short)

# Compute base stats
avg_eb = player_bb["estimated_bases"].mean()
avg_ev = player_bb["launch_speed"].mean()
n_bb = len(player_bb)
player_bb["is_barrel"] = is_barrel_vectorized(player_bb["launch_speed"], player_bb["launch_angle"])
barrel_rate = player_bb["is_barrel"].mean() * 100

# Hero identity card — stays cohesive on mobile
pos_str = ""
age_str = ""
if player_meta is not None:
    pos = player_meta.get("position", "")
    if pos:
        pos_str = f" | {pos}"
    bd = player_meta.get("birth_date", "")
    if bd:
        try:
            birth = pd.to_datetime(bd)
            age = (pd.Timestamp.now() - birth).days // 365
            age_str = f" | Age {age}"
        except Exception:
            pass

# PA count from rankings
n_pa = int(player_ranking["n_batted_balls"]) if player_ranking is not None and "n_batted_balls" in player_ranking.index else None
pa_str = f" | {n_pa:,} PA" if n_pa else ""

img_url = build_headshot_url(player_id) if player_id else (get_team_logo_url(player_team_short) or "")
img_html = ""
if img_url:
    img_html = (
        f'<img src="{img_url}" '
        f'style="width:120px; height:120px; object-fit:contain; border-radius:8px; flex-shrink:0;" '
        f'onerror="this.style.display=\'none\'">'
    )

st.markdown(
    f'<div style="display:flex; align-items:center; gap:20px; '
    f'background:#f8f9fa; border-radius:10px; padding:16px; '
    f'border-left:4px solid {primary_color};">'
    f'{img_html}'
    f'<div>'
    f'<div style="font-size:1.5rem; font-weight:700; margin-bottom:2px;">{selected_player}</div>'
    f'<div style="color:#4A5568; font-size:1rem;"><b>{player_team_short}</b>{pos_str}{age_str}{pa_str}</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# Key stats row — 3 metrics with percentiles
league_pcts = compute_league_percentiles(season, "player")
ev_pct = (league_pcts["ev_by_player"] < avg_ev).mean() * 100 if league_pcts else 50
barrel_pct = (league_pcts["barrel_rates"] < barrel_rate).mean() * 100 if league_pcts else 50

hero_stats, hero_bayesian = st.columns([3, 2])

with hero_stats:
    qs1, qs2, qs3 = st.columns(3)
    qs1.metric("Batted Balls", f"{n_bb:,}")
    qs2.metric("Avg EV", f"{avg_ev:.1f} mph",
               help="Average exit velocity on all batted balls (mph)")
    render_percentile_bar(ev_pct, container=qs2)
    qs3.metric("Barrel Rate", f"{barrel_rate:.1f}%",
               help="Barrels: EV >= 98 mph + launch angle in the sweet spot zone. Barrels produce the highest expected bases.")
    render_percentile_bar(barrel_pct, container=qs3)

with hero_bayesian:
    if player_ranking is not None:
        bayesian_eb = player_ranking["posterior_mean"]
        hdi_low = player_ranking["hdi_low"]
        hdi_high = player_ranking["hdi_high"]

        # Percentile rank
        eb_pct = (pa_rankings["posterior_mean"] < bayesian_eb).mean() * 100

        st.metric("Est. Bases/PA", f"{bayesian_eb:.3f}",
                  help="Estimated true production per plate appearance. Accounts for walks, HBP, strikeouts, and batted ball quality. Small samples are adjusted toward league average.")
        render_percentile_bar(eb_pct)

        # --- 3-Row Comparison Bar ---
        # Best player in current season
        best_idx = pa_rankings["posterior_mean"].idxmax()
        best_row = pa_rankings.loc[best_idx]
        best_eb = best_row["posterior_mean"]
        best_hdi_low = best_row["hdi_low"]
        best_hdi_high = best_row["hdi_high"]
        best_name = best_row["player"]
        best_team = best_row.get("team", "")
        best_color, _ = get_team_color(best_team) if best_team else ("#DAA520", "#DAA520")

        league_mean_eb = pa_rankings["posterior_mean"].mean()
        league_sd_eb = pa_rankings["posterior_mean"].std()
        lg_low = league_mean_eb - league_sd_eb
        lg_high = league_mean_eb + league_sd_eb

        # Display range — encompass all three reference points
        display_min = min(hdi_low, best_hdi_low, lg_low) - 0.03
        display_max = max(hdi_high, best_hdi_high, lg_high) + 0.03
        display_range = display_max - display_min

        def _pct_pos(val):
            return max(0, min(100, (val - display_min) / display_range * 100))

        # Player bar positions
        p_left = _pct_pos(hdi_low)
        p_width = _pct_pos(hdi_high) - p_left
        p_marker = _pct_pos(bayesian_eb)

        # Best player bar positions
        b_left = _pct_pos(best_hdi_low)
        b_width = _pct_pos(best_hdi_high) - b_left
        b_marker = _pct_pos(best_eb)

        # League average ±1 SD positions
        lg_left = _pct_pos(lg_low)
        lg_width = _pct_pos(lg_high) - lg_left
        lg_marker = _pct_pos(league_mean_eb)

        # Truncate best player name for label
        best_label = best_name.split(" ")[-1][:10] if " " in best_name else best_name[:10]

        st.markdown(f"""
        <div style="font-size:13px; margin:8px 0 2px 0;">
            <!-- Row 1: Player -->
            <div style="display:flex; align-items:center; height:26px; margin-bottom:4px;">
                <div style="width:70px; text-align:right; padding-right:8px; font-weight:600; color:{primary_color}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{selected_player.split(' ')[-1][:10]}</div>
                <div style="flex:1; position:relative; height:16px;">
                    <div style="position:absolute; top:2px; left:{p_left:.1f}%; width:{p_width:.1f}%;
                                height:12px; background:{primary_color}; opacity:0.7; border-radius:6px;"></div>
                    <div style="position:absolute; top:0px; left:{p_marker:.1f}%;
                                width:16px; height:16px; margin-left:-8px;
                                background:white; border:3px solid {primary_color};
                                border-radius:50%;"></div>
                </div>
                <div style="width:50px; padding-left:6px; font-size:12px; color:{primary_color}; font-weight:600;">{bayesian_eb:.3f}</div>
            </div>
            <!-- Row 2: Best Player -->
            <div style="display:flex; align-items:center; height:26px; margin-bottom:4px;">
                <div style="width:70px; text-align:right; padding-right:8px; color:{best_color}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{best_name}">{best_label}</div>
                <div style="flex:1; position:relative; height:16px;">
                    <div style="position:absolute; top:2px; left:{b_left:.1f}%; width:{b_width:.1f}%;
                                height:12px; background:{best_color}; opacity:0.5; border-radius:6px;"></div>
                    <div style="position:absolute; top:1px; left:{b_marker:.1f}%;
                                width:14px; height:14px; margin-left:-7px;
                                background:{best_color}; transform:rotate(45deg);"></div>
                </div>
                <div style="width:50px; padding-left:6px; font-size:12px; color:{best_color};">{best_eb:.3f}</div>
            </div>
            <!-- Row 3: League Average ±1 SD -->
            <div style="display:flex; align-items:center; height:26px;">
                <div style="width:70px; text-align:right; padding-right:8px; color:rgba(120,120,120,0.9);">Lg Avg</div>
                <div style="flex:1; position:relative; height:16px;">
                    <div style="position:absolute; top:2px; left:{lg_left:.1f}%; width:{lg_width:.1f}%;
                                height:12px; background:rgba(160,160,160,0.35); border-radius:6px;"></div>
                    <div style="position:absolute; top:2px; left:{lg_marker:.1f}%;
                                width:12px; height:12px; margin-left:-6px;
                                background:rgba(150,150,150,0.7); border-radius:50%;"></div>
                </div>
                <div style="width:50px; padding-left:6px; font-size:12px; color:rgba(120,120,120,0.9);">{league_mean_eb:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.metric("Avg Est. Bases/BB", f"{avg_eb:.3f}")
        st.caption("Full ranking not available (need player evaluation data)")


# =============================================================================
# PLATE DISCIPLINE (K% / BB%)
# =============================================================================

# Compute K% and BB% early (needed by both this section and Fantasy Context)
# K%/BB%/HR% — prefer Bayesian posterior from rankings, fall back to raw pa_counts
_k_rate = 20.0
_bb_rate = 8.0
_hr_rate = 3.0
_k_rate_bayesian = False
_bb_rate_bayesian = False
_hr_rate_bayesian = False
if player_ranking is not None:
    if "k_rate_posterior" in player_ranking.index and pd.notna(player_ranking.get("k_rate_posterior")):
        _k_rate = player_ranking["k_rate_posterior"] * 100
        _k_rate_bayesian = True
    if "bb_rate_posterior" in player_ranking.index and pd.notna(player_ranking.get("bb_rate_posterior")):
        _bb_rate = player_ranking["bb_rate_posterior"] * 100
        _bb_rate_bayesian = True
    if "hr_rate_posterior" in player_ranking.index and pd.notna(player_ranking.get("hr_rate_posterior")):
        _hr_rate = player_ranking["hr_rate_posterior"] * 100
        _hr_rate_bayesian = True

# Fall back to raw computation from pa_counts
if not _k_rate_bayesian or not _bb_rate_bayesian:
    if not pa_counts_df.empty:
        _pc = pa_counts_df[pa_counts_df["player"] == selected_player]
        if not _pc.empty:
            _total_k = _pc["strikeouts"].sum()
            _total_bb = _pc["walks"].sum()
            _pa_for_rate = n_pa if n_pa else n_bb
            if _pa_for_rate > 0:
                if not _k_rate_bayesian:
                    _k_rate = _total_k / _pa_for_rate * 100
                if not _bb_rate_bayesian:
                    _bb_rate = _total_bb / _pa_for_rate * 100

# Display K%, BB%, HR% metrics with percentile context
if _k_rate_bayesian or _bb_rate_bayesian or _hr_rate_bayesian:
    st.divider()
    st.subheader("Plate Discipline")
    st.caption(
        "Current-season estimates based on actual plate appearances this year. "
        "The Bayesian model adjusts for sample size — players with fewer PA get "
        "pulled toward the league average, while large samples stay close to the raw number."
    )

    _kbb1, _kbb2, _kbb3 = st.columns(3)
    with _kbb1:
        _k_suffix = "" if _k_rate_bayesian else " (raw)"
        st.metric(f"K%{_k_suffix}", f"{_k_rate:.1f}%",
                  help="Strikeout rate per plate appearance this season. **Lower is better** for hitters — "
                  "a low K% means the hitter makes contact more often. "
                  "The Bayesian model estimates the player's true strikeout rate by adjusting for sample size.")
        if _k_rate_bayesian and not pa_rankings.empty and "k_rate_posterior" in pa_rankings.columns:
            # Percentile: lower K% is better for hitters, so invert
            _k_pct = (pa_rankings["k_rate_posterior"].dropna() > player_ranking["k_rate_posterior"]).mean() * 100
            render_percentile_bar(_k_pct, container=_kbb1)
            st.caption("↓ Lower is better")
            if "k_rate_hdi_low" in player_ranking.index:
                st.caption(
                    f"89% CI: {player_ranking['k_rate_hdi_low']*100:.1f}% – {player_ranking['k_rate_hdi_high']*100:.1f}%"
                )
    with _kbb2:
        _bb_suffix = "" if _bb_rate_bayesian else " (raw)"
        st.metric(f"BB%{_bb_suffix}", f"{_bb_rate:.1f}%",
                  help="Walk rate per plate appearance this season. **Higher is better** for hitters — "
                  "walks are free bases. "
                  "The Bayesian model estimates the player's true walk rate by adjusting for sample size.")
        if _bb_rate_bayesian and not pa_rankings.empty and "bb_rate_posterior" in pa_rankings.columns:
            _bb_pct = (pa_rankings["bb_rate_posterior"].dropna() < player_ranking["bb_rate_posterior"]).mean() * 100
            render_percentile_bar(_bb_pct, container=_kbb2)
            if "bb_rate_hdi_low" in player_ranking.index:
                st.caption(
                    f"89% CI: {player_ranking['bb_rate_hdi_low']*100:.1f}% – {player_ranking['bb_rate_hdi_high']*100:.1f}%"
                )
    with _kbb3:
        _hr_suffix = "" if _hr_rate_bayesian else " (raw)"
        st.metric(f"HR%{_hr_suffix}", f"{_hr_rate:.1f}%",
                  help="Home run rate per plate appearance this season. "
                  "The Bayesian model estimates the player's true HR rate by adjusting for sample size.")
        if _hr_rate_bayesian and not pa_rankings.empty and "hr_rate_posterior" in pa_rankings.columns:
            _hr_pct = (pa_rankings["hr_rate_posterior"].dropna() < player_ranking["hr_rate_posterior"]).mean() * 100
            render_percentile_bar(_hr_pct, container=_kbb3)
            if "hr_rate_hdi_low" in player_ranking.index:
                st.caption(
                    f"89% CI: {player_ranking['hr_rate_hdi_low']*100:.1f}% – {player_ranking['hr_rate_hdi_high']*100:.1f}%"
                )


# =============================================================================
# SEASON STATS (traditional counting stats for context)
# =============================================================================

if player_ranking is not None and "avg" in player_ranking.index and pd.notna(player_ranking.get("avg")):
    st.divider()
    st.subheader(f"{season} Season Stats")

    _avg = player_ranking["avg"]
    _obp = player_ranking["obp"]
    _slg = player_ranking["slg"]
    _ops = _obp + _slg
    _hr = int(player_ranking["home_runs"])
    _r = int(player_ranking["runs"])
    _rbi = int(player_ranking["rbi"])
    _sb = int(player_ranking["stolen_bases"])
    _n_pa_trad = int(player_ranking["n_batted_balls"]) if "n_batted_balls" in player_ranking.index else None

    st.markdown(
        f'<div style="background:#F7FAFC; border-radius:10px; padding:16px 20px; '
        f'border-left:4px solid {primary_color};">'
        # Slash line row
        f'<div style="display:flex; justify-content:center; gap:32px; flex-wrap:wrap; margin-bottom:10px;">'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">AVG</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_avg:.3f}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">OBP</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_obp:.3f}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">SLG</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_slg:.3f}</div></div>'
        f'<div style="text-align:center;"><div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">OPS</div><div style="font-size:1.6rem; font-weight:700; color:#1a1a1a;">{_ops:.3f}</div></div>'
        f'</div>'
        # Counting stats row
        f'<div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; padding-top:8px; border-top:1px solid #E2E8F0;">'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">HR </span><span style="font-weight:600; font-size:0.95rem;">{_hr}</span></div>'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">R </span><span style="font-weight:600; font-size:0.95rem;">{_r}</span></div>'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">RBI </span><span style="font-weight:600; font-size:0.95rem;">{_rbi}</span></div>'
        f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">SB </span><span style="font-weight:600; font-size:0.95rem;">{_sb}</span></div>'
        + (f'<div style="text-align:center;"><span style="color:#718096; font-size:0.8rem;">PA </span><span style="font-weight:600; font-size:0.95rem;">{_n_pa_trad:,}</span></div>' if _n_pa_trad else '')
        + f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption("Traditional stats via MLB Stats API — actual results, not model estimates. Totals may vary slightly from other sources if Statcast data was unavailable for a game or plate appearance.")


# =============================================================================
# PLAYER PROFILE (Radar Chart + Archetype + Similar Players)
# =============================================================================

from utils.player_analytics import (
    compute_hitter_radar_metrics, cluster_player_archetypes,
    find_similar_players, get_player_radar_percentiles,
    HITTER_ARCHETYPE_DESC,
)
from utils.player_helpers import render_radar_chart, render_archetype_badge, render_similar_players

# Compute radar metrics for all hitters (cached via Streamlit's data flow)
_radar_df = compute_hitter_radar_metrics(pa_rankings, bb_df, min_pa=30)

if not _radar_df.empty:
    _radar_df = cluster_player_archetypes(_radar_df, player_type="hitter")
    _player_pcts = get_player_radar_percentiles(_radar_df, selected_player, player_team_short, "hitter")
else:
    _player_pcts = None

# Compute luck/trend stats (preserved from old Fantasy Context)
player_bb["_actual_tb"] = player_bb["actual_result"].map(TB_MAP).fillna(0)
_player_luck = player_bb["_actual_tb"].sum() - player_bb["estimated_bases"].sum()

_league_luck = bb_df.copy()
_league_luck["_actual_tb"] = _league_luck["actual_result"].map(TB_MAP).fillna(0)
_league_luck_per_player = _league_luck.groupby("player").apply(
    lambda x: x["_actual_tb"].sum() - x["estimated_bases"].sum(),
    include_groups=False,
)
_luck_pct = (_league_luck_per_player < _player_luck).mean() * 100

_unlucky_outs = player_bb[
    (player_bb["_actual_tb"] == 0) & (player_bb["estimated_bases"] > 0.5)
]
_n_unlucky = len(_unlucky_outs)

_recent_eb = None
_recent_vs_season = 0.0
if "date_parsed" in player_bb.columns and len(player_bb) > 0:
    _max_date = player_bb["date_parsed"].max()
    _recent = player_bb[player_bb["date_parsed"] > _max_date - pd.Timedelta(days=14)]
    if len(_recent) > 5:
        _recent_eb = _recent["estimated_bases"].mean()
        _recent_vs_season = _recent_eb - avg_eb

_platoon_str = None
if not metadata_df.empty and "pitcher" in player_bb.columns:
    _thm = metadata_df.set_index("player_name")["throw_hand"].to_dict()
    player_bb["_pitcher_hand"] = player_bb["pitcher"].map(_thm)
    _vs_l = player_bb[player_bb["_pitcher_hand"] == "L"]
    _vs_r = player_bb[player_bb["_pitcher_hand"] == "R"]
    if len(_vs_l) >= 10 and len(_vs_r) >= 10:
        _gap = _vs_l["estimated_bases"].mean() - _vs_r["estimated_bases"].mean()
        if abs(_gap) > 0.03:
            _better = "vs LHP" if _gap > 0 else "vs RHP"
            _platoon_str = f"+{abs(_gap):.3f} EB/BB {_better}"
    player_bb.drop(columns=["_pitcher_hand"], inplace=True, errors="ignore")

# Display
st.divider()
st.subheader("Player Profile")

if _player_pcts is not None:
    _col_radar, _col_info = st.columns([3, 2])

    with _col_radar:
        _radar_fig = render_radar_chart(_player_pcts, primary_color)
        st.plotly_chart(_radar_fig, use_container_width=True, config=PLOTLY_CONFIG)
        if n_bb < 50:
            st.caption(f"Based on {n_bb} batted balls — profile may shift as more data accumulates.")

    with _col_info:
        # Archetype badge
        _player_match = _radar_df[
            (_radar_df["player"] == selected_player) & (_radar_df["team"] == player_team_short)
        ]
        if _player_match.empty:
            _player_match = _radar_df[_radar_df["player"] == selected_player]
        _archetype = _player_match.iloc[0]["archetype"] if not _player_match.empty else "Unknown"
        _arch_desc = HITTER_ARCHETYPE_DESC.get(_archetype, "")

        st.markdown(render_archetype_badge(_archetype, _arch_desc, primary_color), unsafe_allow_html=True)

        # Similar players
        st.markdown('<div style="font-weight:600; margin-top:16px; margin-bottom:6px; font-size:0.85rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">Similar Players</div>', unsafe_allow_html=True)
        _similar = find_similar_players(_radar_df, selected_player, player_team_short, "hitter", n=5)
        st.markdown(render_similar_players(_similar), unsafe_allow_html=True)

        # Quick stats (luck, trends, platoon)
        st.markdown('<div style="font-weight:600; margin-top:16px; margin-bottom:6px; font-size:0.85rem; color:#718096; text-transform:uppercase; letter-spacing:0.5px;">Quick Stats</div>', unsafe_allow_html=True)

        _luck_str = f"{_player_luck:+.1f} net lucky bases ({_luck_pct:.0f}th pct)"
        _quick_parts = [f"Luck: {_luck_str}", f"{_n_unlucky} unlucky outs (EB > 0.5)"]
        if _recent_eb is not None:
            _arrow_sym = "+" if _recent_vs_season > 0.005 else ("" if _recent_vs_season < -0.005 else "")
            _quick_parts.append(f"Last 14d: {_recent_eb:.3f} EB/BB ({_arrow_sym}{_recent_vs_season:.3f} vs season)")
        if _platoon_str:
            _quick_parts.append(f"Platoon: {_platoon_str}")

        st.markdown(
            "".join(f'<div style="color:#4A5568; font-size:0.88rem; padding:1px 0;">{p}</div>' for p in _quick_parts),
            unsafe_allow_html=True,
        )

    with st.expander("How does this work?"):
        st.markdown(
            "**Radar Chart:** Shows how this player compares to every hitter with 30+ plate appearances this season. "
            "Each spoke is a different skill, measured as a percentile (0 to 100). A score of 80 means the player is better "
            "than 80% of hitters in that skill. All axes are oriented so that bigger = better.\n\n"
            "**Archetype:** Players are grouped by their radar shape using a clustering algorithm (K-Means). "
            "Players in the same archetype tend to have similar strengths and weaknesses.\n\n"
            "**Current Hitter Archetypes:**\n"
            "- **Elite All-Around**: Top-tier production across all skill dimensions. These are the most complete hitters in the league.\n"
            "- **Power Hitter**: Combines power with plate discipline to consistently drive the ball.\n"
            "- **Power Slugger**: Drives the ball hard with elite power and hard-hit rate, but doesn't draw many walks. Produces through raw hitting ability.\n"
            "- **Contact-Speed**: Puts the ball in play consistently and uses above-average speed to create value. A well-rounded offensive contributor.\n"
            "- **Power-Speed**: Combines speed and emerging power but strikes out frequently. High-ceiling profile with boom-or-bust at-bats.\n"
            "- **Speed Threat**: Gets on base through contact and creates havoc on the basepaths. Limited power but hard to keep off base.\n"
            "- **Patient Hitter**: Works counts and earns walks with solid contact ability. Lacks power and speed but contributes through plate discipline.\n"
            "- **Below Average**: Below-average production across most skill dimensions this season.\n\n"
            "**Similar Players:** The 5 hitters whose overall skill profile most closely matches this player's, "
            "based on the Euclidean distance between their radar shapes in 6-dimensional percentile space.\n\n"
            "**Data Note:** The radar uses the best available estimate of each player's true skill level. "
            "When preseason projections are available, they are combined with in-season performance using "
            "inverse-variance weighting for a more stable \"true talent\" estimate. Otherwise, the Bayesian "
            "posterior from the current season is used, which already shrinks small samples toward the league mean."
        )

else:
    # Fallback: no radar data available (missing rate stat columns, etc.)
    _quick_parts = []
    _luck_str = f"{_player_luck:+.1f} net lucky bases ({_luck_pct:.0f}th pct)"
    _quick_parts.append(f"Luck: {_luck_str}")
    _quick_parts.append(f"{_n_unlucky} unlucky outs (EB > 0.5)")
    if _recent_eb is not None:
        _arrow_sym = "+" if _recent_vs_season > 0.005 else ""
        _quick_parts.append(f"Last 14d: {_recent_eb:.3f} EB/BB ({_arrow_sym}{_recent_vs_season:.3f} vs season)")
    if _platoon_str:
        _quick_parts.append(f"Platoon: {_platoon_str}")

    st.markdown(
        "".join(f'<div style="color:#4A5568; font-size:0.9rem; padding:2px 0;">{p}</div>' for p in _quick_parts),
        unsafe_allow_html=True,
    )

# Clean up temp columns
player_bb.drop(columns=["_actual_tb"], inplace=True, errors="ignore")


# =============================================================================
# HISTORICAL EST. BASES / PA TIMELINE
# =============================================================================

if len(all_season_pa_rankings) > 0 and player_id is not None:
    # Collect player data across seasons
    timeline_data = []
    best_player_data = []
    league_avg_data = []
    # Rate stat timelines (K%, BB%, HR% — Bayesian posteriors with HDI)
    rate_timeline = []  # per-season rate stats for this player
    rate_league_avg = []  # per-season league averages

    for s in sorted(all_season_pa_rankings.keys()):
        pa_df = all_season_pa_rankings[s]

        # League average
        lg_mean = pa_df["posterior_mean"].mean()
        league_avg_data.append({"season": s, "value": lg_mean})

        # Rate stat league averages
        _rate_lg = {"season": s}
        for _rc in ["k_rate_posterior", "bb_rate_posterior", "hr_rate_posterior"]:
            if _rc in pa_df.columns:
                _rate_lg[_rc] = pa_df[_rc].dropna().mean()
        rate_league_avg.append(_rate_lg)

        # Best player this season
        best_idx_s = pa_df["posterior_mean"].idxmax()
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

        # Find player by player_id, fallback to name
        match = pd.DataFrame()
        if "player_id" in pa_df.columns:
            match = pa_df[pa_df["player_id"] == player_id]
        if match.empty:
            match = pa_df[pa_df["player"] == selected_player]
        if not match.empty:
            row = match.iloc[0]
            n_pa_val = int(row["n_batted_balls"]) if "n_batted_balls" in row.index else None
            timeline_data.append({
                "season": s,
                "value": row["posterior_mean"],
                "hdi_low": row["hdi_low"],
                "hdi_high": row["hdi_high"],
                "n_pa": n_pa_val,
            })
            # Collect rate stats for this player (K%, BB%, HR% posteriors + HDI)
            _rate_row = {"season": s}
            for _rc in ["k_rate_posterior", "k_rate_hdi_low", "k_rate_hdi_high",
                         "bb_rate_posterior", "bb_rate_hdi_low", "bb_rate_hdi_high",
                         "hr_rate_posterior", "hr_rate_hdi_low", "hr_rate_hdi_high"]:
                if _rc in row.index and pd.notna(row.get(_rc)):
                    _rate_row[_rc] = row[_rc]
            rate_timeline.append(_rate_row)

    # Check if we have rate stats for tabs
    _has_rate_timeline = any("k_rate_posterior" in t for t in rate_timeline)

    if timeline_data:
        st.divider()
        st.subheader("Season History")

        tl_df = pd.DataFrame(timeline_data)
        best_df = pd.DataFrame(best_player_data)
        lg_df = pd.DataFrame(league_avg_data)
        rate_tl_df = pd.DataFrame(rate_timeline) if rate_timeline else pd.DataFrame()
        rate_lg_df = pd.DataFrame(rate_league_avg) if rate_league_avg else pd.DataFrame()

        # Build PA count lookup for custom tick labels
        pa_by_season = {
            row["season"]: row["n_pa"]
            for row in timeline_data
            if row.get("n_pa") is not None
        }

        # Trim to player's season range
        player_seasons = tl_df["season"].tolist()
        min_s, max_s = min(player_seasons), max(player_seasons)
        lg_df = lg_df[(lg_df["season"] >= min_s) & (lg_df["season"] <= max_s)]
        best_df = best_df[(best_df["season"] >= min_s) & (best_df["season"] <= max_s)]
        if not rate_lg_df.empty:
            rate_lg_df = rate_lg_df[(rate_lg_df["season"] >= min_s) & (rate_lg_df["season"] <= max_s)]

        # --- Helper: build a rate stat chart with HDI bands ---
        def _build_rate_chart(rate_prefix, y_label, higher_is_better=True):
            """Build a rate stat chart with HDI bands, projections, and true talent."""
            mean_col = f"{rate_prefix}_posterior"
            lo_col = f"{rate_prefix}_hdi_low"
            hi_col = f"{rate_prefix}_hdi_high"
            if rate_tl_df.empty or mean_col not in rate_tl_df.columns:
                return None
            vals = rate_tl_df.dropna(subset=[mean_col])
            if vals.empty:
                return None

            fig = go.Figure()
            # League average
            if not rate_lg_df.empty and mean_col in rate_lg_df.columns:
                lg_vals = rate_lg_df.dropna(subset=[mean_col])
                if not lg_vals.empty:
                    fig.add_trace(go.Scatter(
                        x=lg_vals["season"], y=lg_vals[mean_col] * 100,
                        mode="markers", name="Lg Avg",
                        marker=dict(color="rgba(160,160,160,0.8)", size=11),
                        hovertemplate="Season: %{x}<br>Lg Avg: %{y:.1f}%<extra></extra>",
                    ))

            c = primary_color.lstrip("#")
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

            if len(vals) == 1:
                row_r = vals.iloc[0]
                err_hi = [(row_r[hi_col] - row_r[mean_col]) * 100] if hi_col in row_r.index and pd.notna(row_r.get(hi_col)) else None
                err_lo = [(row_r[mean_col] - row_r[lo_col]) * 100] if lo_col in row_r.index and pd.notna(row_r.get(lo_col)) else None
                fig.add_trace(go.Scatter(
                    x=vals["season"], y=vals[mean_col] * 100,
                    mode="markers", name=selected_player,
                    marker=dict(color=primary_color, size=12),
                    error_y=dict(
                        type="data", array=err_hi, arrayminus=err_lo,
                        color=f"rgba({r},{g},{b},0.5)", thickness=2, width=6,
                    ) if err_hi else None,
                    hovertemplate=f"Season: %{{x}}<br>{y_label}: %{{y:.1f}}%<extra></extra>",
                ))
            else:
                # HDI band
                if hi_col in vals.columns and lo_col in vals.columns:
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[hi_col] * 100,
                        mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=vals["season"], y=vals[lo_col] * 100,
                        mode="lines", line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba({r},{g},{b},0.15)",
                        showlegend=False, hoverinfo="skip",
                    ))
                # Player line
                fig.add_trace(go.Scatter(
                    x=vals["season"], y=vals[mean_col] * 100,
                    mode="lines+markers", name=selected_player,
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
                rp_df = load_rate_stat_projections(proj_s, "hitter", rate_prefix)
                if rp_df.empty or proj_col not in rp_df.columns:
                    continue
                rp_match = rp_df[rp_df["player_id"] == player_id] if "player_id" in rp_df.columns else pd.DataFrame()
                if rp_match.empty:
                    rp_match = rp_df[rp_df["player"] == selected_player]
                if not rp_match.empty:
                    rate_proj_points.append(rp_match.iloc[0].to_dict() | {"season": proj_s})

            if rate_proj_points:
                rp_seasons = [p["season"] for p in rate_proj_points]
                rp_vals = [p[proj_col] * 100 for p in rate_proj_points]
                rp_hi = [p.get(proj_hi_col, p[proj_col]) * 100 for p in rate_proj_points]
                rp_lo = [p.get(proj_lo_col, p[proj_col]) * 100 for p in rate_proj_points]

                # Shaded projection zone
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

                # HDI ribbon
                fig.add_trace(go.Scatter(
                    x=rp_seasons, y=rp_hi, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=rp_seasons, y=rp_lo, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.10)",
                    showlegend=False, hoverinfo="skip",
                ))

                # Dashed connector from last actual to first projection
                last_val = float(vals.iloc[-1][mean_col]) * 100
                fig.add_trace(go.Scatter(
                    x=[max_actual_s, rp_seasons[0]], y=[last_val, rp_vals[0]],
                    mode="lines",
                    line=dict(color=f"rgba({r},{g},{b},0.4)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                # Projection trajectory line
                if len(rp_seasons) > 1:
                    fig.add_trace(go.Scatter(
                        x=rp_seasons, y=rp_vals, mode="lines",
                        line=dict(color=f"rgba({r},{g},{b},0.5)", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Projection markers (open diamonds)
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

            # True talent reference line
            tt_col = f"true_talent_{rate_prefix}"
            if (player_ranking is not None
                    and tt_col in player_ranking.index
                    and pd.notna(player_ranking.get(tt_col))):
                tt_val = player_ranking[tt_col] * 100
                fig.add_hline(
                    y=tt_val, line_dash="dash",
                    line_color="rgba(124, 58, 237, 0.5)", line_width=1.5,
                )
                fig.add_annotation(
                    x=max_actual_s, y=tt_val,
                    text=f"True Talent: {tt_val:.1f}%",
                    showarrow=False, xshift=8, yshift=12,
                    font=dict(size=11, color="rgba(124, 58, 237, 0.8)"),
                    xanchor="left",
                )

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
            return fig

        # --- Tabs ---
        if _has_rate_timeline:
            _tab_labels = ["EB/PA", "K%", "BB%", "HR%"]
            _timeline_tabs = st.tabs(_tab_labels)
            _eb_container = _timeline_tabs[0]
        else:
            _eb_container = st.container()

        with _eb_container:
            if _has_rate_timeline:
                st.caption("Bayesian model estimate with projections")

            fig_timeline = go.Figure()

            # League average — gray dots only (no line)
            fig_timeline.add_trace(go.Scatter(
                x=lg_df["season"],
                y=lg_df["value"],
                mode="markers",
                name="Lg Avg",
                marker=dict(color="rgba(160,160,160,0.8)", size=11),
                hovertemplate="Season: %{x}<br>Lg Avg: %{y:.3f}<extra></extra>",
            ))

            # Best player — gold diamonds per season
            if not best_df.empty:
                first_best_season = best_df["season"].iloc[0]
                for _, brow in best_df.iterrows():
                    fig_timeline.add_trace(go.Scatter(
                        x=[brow["season"]],
                        y=[brow["value"]],
                        mode="markers",
                        name=brow["name"],
                        marker=dict(color="#DAA520", size=12, symbol="diamond",
                                    line=dict(width=1, color="white")),
                        hovertemplate=f"Season: %{{x}}<br>{brow['name']}: %{{y:.3f}}<extra></extra>",
                        legendgroup="best",
                        showlegend=bool(brow["season"] == first_best_season),
                    ))
                # Override legend entry for the group
                fig_timeline.data[-len(best_df)].name = "Best Hitter"

            # Player — line with HDI
            if len(tl_df) == 1:
                # Single point: use error bars for HDI
                row = tl_df.iloc[0]
                c = primary_color.lstrip("#")
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"],
                    y=tl_df["value"],
                    mode="markers",
                    name=selected_player,
                    marker=dict(color=primary_color, size=12),
                    error_y=dict(
                        type="data",
                        array=[row["hdi_high"] - row["value"]],
                        arrayminus=[row["value"] - row["hdi_low"]],
                        color=f"rgba({r},{g},{b},0.5)",
                        thickness=2,
                        width=6,
                    ),
                    hovertemplate=(
                        "Season: %{x}<br>"
                        "EB/PA: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                ))
            else:
                # Multiple points: fill band + line
                # Upper bound (invisible, for fill)
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"],
                    y=tl_df["hdi_high"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Lower bound with fill to upper
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"],
                    y=tl_df["hdi_low"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({int(primary_color[1:3], 16)},{int(primary_color[3:5], 16)},{int(primary_color[5:7], 16)},0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Player line + markers
                fig_timeline.add_trace(go.Scatter(
                    x=tl_df["season"],
                    y=tl_df["value"],
                    mode="lines+markers",
                    name=selected_player,
                    line=dict(color=primary_color, width=2.5),
                    marker=dict(color=primary_color, size=12),
                    hovertemplate=(
                        "Season: %{x}<br>"
                        "EB/PA: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                ))

            # Multi-year projections (shaded zone with trajectory)
            proj_points = []
            for proj_season in range(max_s + 1, max_s + 4):
                proj_df = load_player_projections(proj_season, "hitter")
                if proj_df.empty:
                    continue
                proj_match = proj_df[proj_df["player_id"] == player_id] if "player_id" in proj_df.columns else pd.DataFrame()
                if proj_match.empty:
                    proj_match = proj_df[proj_df["player"] == selected_player]
                if not proj_match.empty:
                    proj_points.append(proj_match.iloc[0].to_dict() | {"season": proj_season})

            if proj_points:
                c = primary_color.lstrip("#")
                pr, pg, pb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                proj_seasons = [p["season"] for p in proj_points]
                proj_values = [p["projected_eb_pa"] for p in proj_points]
                proj_hdi_hi = [p["projected_hdi_high"] for p in proj_points]
                proj_hdi_lo = [p["projected_hdi_low"] for p in proj_points]

                # Shaded background for projection zone
                fig_timeline.add_vrect(
                    x0=max_s + 0.5, x1=max(proj_seasons) + 0.5,
                    fillcolor="rgba(180,180,220,0.10)", line_width=0,
                    layer="below",
                )
                # "Projected" label
                fig_timeline.add_annotation(
                    x=(max_s + 0.5 + max(proj_seasons) + 0.5) / 2,
                    y=1.0, yref="paper", yanchor="bottom",
                    text="Projected", showarrow=False,
                    font=dict(size=13, color="rgba(120,120,160,0.7)"),
                )

                # HDI ribbon for projections
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_hdi_hi,
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_hdi_lo,
                    mode="lines", line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({pr},{pg},{pb},0.10)",
                    showlegend=False, hoverinfo="skip",
                ))

                # Dashed connection from last actual to first projection
                last_actual = tl_df[tl_df["season"] == max_s].iloc[0]
                fig_timeline.add_trace(go.Scatter(
                    x=[max_s, proj_seasons[0]],
                    y=[last_actual["value"], proj_values[0]],
                    mode="lines",
                    line=dict(color=f"rgba({pr},{pg},{pb},0.4)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

                # Projection trajectory line
                if len(proj_seasons) > 1:
                    fig_timeline.add_trace(go.Scatter(
                        x=proj_seasons, y=proj_values,
                        mode="lines",
                        line=dict(color=f"rgba({pr},{pg},{pb},0.5)", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Projection markers (open diamonds)
                fig_timeline.add_trace(go.Scatter(
                    x=proj_seasons, y=proj_values,
                    mode="markers",
                    name="Projection",
                    showlegend=False,
                    marker=dict(
                        color="rgba(255,255,255,0)", size=12,
                        symbol="diamond-open",
                        line=dict(width=2.5, color=primary_color),
                    ),
                    customdata=[
                        [p.get("aging_effect", 0), p["projected_hdi_low"], p["projected_hdi_high"]]
                        for p in proj_points
                    ],
                    hovertemplate=(
                        "Projection %{x}<br>"
                        "EB/PA: %{y:.3f}<br>"
                        "89% HDI: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>"
                        "Aging effect: %{customdata[0]:+.3f}"
                        "<extra></extra>"
                    ),
                ))

            # True talent reference line (when available from combined projection + evaluation)
            if (player_ranking is not None
                    and "true_talent_eb_pa" in player_ranking.index
                    and pd.notna(player_ranking.get("true_talent_eb_pa"))):
                tt_val = player_ranking["true_talent_eb_pa"]
                fig_timeline.add_hline(
                    y=tt_val,
                    line_dash="dash",
                    line_color="rgba(124, 58, 237, 0.5)",
                    line_width=1.5,
                )
                fig_timeline.add_annotation(
                    x=max_s, y=tt_val,
                    text=f"True Talent: {tt_val:.3f}",
                    showarrow=False,
                    xshift=8,
                    yshift=12,
                    font=dict(size=11, color="rgba(124, 58, 237, 0.8)"),
                    xanchor="left",
                )

            # Compute x-axis range including projections
            x_max = max(proj_seasons) if proj_points else max_s

            # Build custom tick labels with PA counts
            all_tick_seasons = list(range(min_s, x_max + 1))
            tick_labels = []
            for s in all_tick_seasons:
                if s in pa_by_season:
                    tick_labels.append(f"{s}<br><sub>{pa_by_season[s]:,} PA</sub>")
                else:
                    tick_labels.append(str(s))

            fig_timeline.update_layout(
                xaxis=dict(
                    title="Season",
                    title_font_size=14,
                    tickfont_size=13,
                    tickvals=all_tick_seasons,
                    ticktext=tick_labels,
                    range=[min_s - 0.5, x_max + 0.5],
                ),
                yaxis=dict(
                    title="Est. Bases per PA",
                    title_font_size=14,
                    tickfont_size=13,
                ),
                height=400,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=13),
                ),
                dragmode=False,
            )
            st.plotly_chart(fig_timeline, width="stretch", config=PLOTLY_CONFIG)

            if len(timeline_data) == 1:
                st.caption("Only one season of data available. More history will accumulate over time.")

        # --- Rate stat tabs ---
        if _has_rate_timeline:
            with _timeline_tabs[1]:
                st.caption("Bayesian strikeout rate with 89% credible interval and projections")
                _fig_k = _build_rate_chart("k_rate", "K%", higher_is_better=False)
                if _fig_k:
                    st.plotly_chart(_fig_k, width="stretch", config=PLOTLY_CONFIG)

            with _timeline_tabs[2]:
                st.caption("Bayesian walk rate with 89% credible interval and projections")
                _fig_bb = _build_rate_chart("bb_rate", "BB%")
                if _fig_bb:
                    st.plotly_chart(_fig_bb, width="stretch", config=PLOTLY_CONFIG)

            with _timeline_tabs[3]:
                st.caption("Bayesian home run rate with 89% credible interval and projections")
                _fig_hr = _build_rate_chart("hr_rate", "HR%")
                if _fig_hr:
                    st.plotly_chart(_fig_hr, width="stretch", config=PLOTLY_CONFIG)

    elif len(all_season_pa_rankings) > 0:
        # Player not found in any season rankings
        pass


# =============================================================================
# LUCK REPORT
# =============================================================================

st.divider()
st.subheader("Luck Report")
st.caption(f"{season} Season")

player_bb["actual_tb"] = player_bb["actual_result"].map(TB_MAP).fillna(0)

total_actual_tb = player_bb["actual_tb"].sum()
total_expected_tb = player_bb["estimated_bases"].sum()
luck_score = total_actual_tb - total_expected_tb

# Compute league-wide percentiles
actual_pct = expected_pct = luck_pct = None
if len(bb_df) > 1000:
    all_luck = bb_df.copy()
    all_luck["actual_tb"] = all_luck["actual_result"].map(TB_MAP).fillna(0)
    player_luck = all_luck.groupby("player").agg(
        actual=("actual_tb", "sum"),
        expected=("estimated_bases", "sum"),
        n=("estimated_bases", "count"),
    )
    player_luck = player_luck[player_luck["n"] >= 30]
    player_luck["luck"] = player_luck["actual"] - player_luck["expected"]
    if selected_player in player_luck.index:
        actual_pct = (player_luck["actual"] < total_actual_tb).mean() * 100
        expected_pct = (player_luck["expected"] < total_expected_tb).mean() * 100
        luck_pct = (player_luck["luck"] < luck_score).mean() * 100

# Luck metrics
lm1, lm2, lm3 = st.columns(3)
with lm1:
    st.metric("Actual Total Bases", f"{total_actual_tb:.0f}")
    render_percentile_bar(actual_pct)
with lm2:
    st.metric("Expected Total Bases", f"{total_expected_tb:.1f}")
    render_percentile_bar(expected_pct)
with lm3:
    delta_color = "normal" if luck_score >= 0 else "inverse"
    tier = luck_tier_label(luck_pct)
    delta_text = tier if tier else ("Lucky" if luck_score > 0 else "Unlucky")
    st.metric("Net Lucky Bases", f"{luck_score:+.1f}", delta=delta_text, delta_color=delta_color,
              help="Actual total bases minus expected total bases. Positive = lucky (more bases than expected from contact quality). Tends to regress toward zero over a full season.")
    render_percentile_bar(luck_pct)

# --- Luck Accumulation Chart ---
st.markdown("#### Luck Over Time")
st.caption("Cumulative actual total bases minus expected. Rising = getting lucky. Falling = unlucky.")

luck_ts = player_bb.sort_values("date_parsed").copy()
luck_ts["cum_actual"] = luck_ts["actual_tb"].cumsum()
luck_ts["cum_expected"] = luck_ts["estimated_bases"].cumsum()
luck_ts["cum_luck"] = luck_ts["cum_actual"] - luck_ts["cum_expected"]
luck_ts["bb_num"] = range(1, len(luck_ts) + 1)

fig_luck = go.Figure()
fig_luck.add_trace(go.Scatter(
    x=luck_ts["bb_num"],
    y=luck_ts["cum_luck"],
    mode="lines",
    name="Cumulative Luck",
    line=dict(color=primary_color, width=2),
    hovertemplate=(
        "BB #%{x}<br>"
        "Cumulative Luck: %{y:.1f}<br>"
        "<extra></extra>"
    ),
))
fig_luck.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig_luck.update_layout(
    xaxis_title="Batted Ball #",
    yaxis_title="Cumulative Luck (TB)",
    height=400,
    template="plotly_white",
    dragmode=False,
)
st.plotly_chart(fig_luck, width="stretch", config=PLOTLY_CONFIG)



# =============================================================================
# CONTACT QUALITY
# =============================================================================

st.divider()
st.subheader("Contact Quality Profile")
st.caption(f"{season} Season")

# Contact quality summary metrics
_hard_hit = (player_bb["launch_speed"] >= 95).mean() * 100 if len(player_bb) > 0 else 0
_sweet_spot = ((player_bb["launch_angle"] >= 8) & (player_bb["launch_angle"] <= 32)).mean() * 100 if len(player_bb) > 0 else 0
_barrel_count = int(player_bb["is_barrel"].sum()) if "is_barrel" in player_bb.columns else 0
_max_ev = player_bb["launch_speed"].max() if len(player_bb) > 0 else 0

# League percentiles for contact quality
_league_hh_pct = _league_ss_pct = None
if len(bb_df) > 1000:
    _lg_hh = bb_df.groupby("player")["launch_speed"].apply(lambda x: (x >= 95).mean() * 100)
    _league_hh_pct = (_lg_hh < _hard_hit).mean() * 100
    _lg_ss = bb_df.groupby("player")["launch_angle"].apply(lambda x: ((x >= 8) & (x <= 32)).mean() * 100)
    _league_ss_pct = (_lg_ss < _sweet_spot).mean() * 100

_cq1, _cq2, _cq3, _cq4 = st.columns(4)
with _cq1:
    st.metric("Hard Hit %", f"{_hard_hit:.1f}%", help="Exit velocity >= 95 mph")
    if _league_hh_pct is not None:
        render_percentile_bar(_league_hh_pct, container=_cq1)
with _cq2:
    st.metric("Sweet Spot %", f"{_sweet_spot:.1f}%", help="Launch angle 8-32 degrees")
    if _league_ss_pct is not None:
        render_percentile_bar(_league_ss_pct, container=_cq2)
with _cq3:
    st.metric("Barrels", f"{_barrel_count}")
with _cq4:
    st.metric("Max EV", f"{_max_ev:.1f} mph")

# Add derived columns
player_bb["bb_type"] = player_bb["launch_angle"].apply(categorize_launch_angle)

# --- Distribution charts row ---
col_ev, col_la = st.columns(2)

with col_ev:
    st.markdown("#### Exit Velocity Distribution")
    fig_ev = go.Figure()
    fig_ev.add_trace(go.Histogram(
        x=bb_df["launch_speed"],
        name="League",
        opacity=0.3,
        marker_color="gray",
        histnorm="probability density",
        nbinsx=40,
    ))
    fig_ev.add_trace(go.Histogram(
        x=player_bb["launch_speed"],
        name=selected_player,
        opacity=0.6,
        marker_color=primary_color,
        histnorm="probability density",
        nbinsx=30,
    ))
    fig_ev.update_layout(
        barmode="overlay",
        xaxis_title="Exit Velocity (mph)",
        yaxis_title="Density",
        height=350,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        dragmode=False,
    )
    st.plotly_chart(fig_ev, width="stretch", config=PLOTLY_CONFIG)

with col_la:
    st.markdown("#### Launch Angle Distribution")
    fig_la = go.Figure()
    fig_la.add_trace(go.Histogram(
        x=bb_df["launch_angle"],
        name="League",
        opacity=0.3,
        marker_color="gray",
        histnorm="probability density",
        nbinsx=40,
    ))
    fig_la.add_trace(go.Histogram(
        x=player_bb["launch_angle"],
        name=selected_player,
        opacity=0.6,
        marker_color=primary_color,
        histnorm="probability density",
        nbinsx=30,
    ))
    fig_la.update_layout(
        barmode="overlay",
        xaxis_title="Launch Angle (deg)",
        yaxis_title="Density",
        height=350,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        dragmode=False,
    )
    st.plotly_chart(fig_la, width="stretch", config=PLOTLY_CONFIG)

# --- EV x LA Scatter + Spray Chart ---
import numpy as np

col_evla, col_spray = st.columns(2)

with col_evla:
    st.markdown("#### Exit Velo x Launch Angle")
    st.caption("Each batted ball colored by estimated bases.")
    evla_data = player_bb.dropna(subset=["launch_speed", "launch_angle"])
    if not evla_data.empty:
        fig_evla = go.Figure()
        fig_evla.add_trace(go.Scatter(
            x=evla_data["launch_speed"],
            y=evla_data["launch_angle"],
            mode="markers",
            marker=dict(
                color=evla_data["estimated_bases"],
                colorscale="RdYlGn",
                size=6,
                colorbar=dict(title="Est.<br>Bases"),
            ),
            customdata=np.stack([
                evla_data["estimated_bases"],
                evla_data["actual_result"],
            ], axis=-1),
            hovertemplate=(
                "EV: %{x:.1f} mph<br>"
                "LA: %{y:.0f}&deg;<br>"
                "Est. Bases: %{customdata[0]:.2f}<br>"
                "Result: %{customdata[1]}"
                "<extra></extra>"
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
            xaxis_title="Exit Velocity (mph)",
            yaxis_title="Launch Angle (&deg;)",
            height=500,
            template="plotly_white",
            dragmode=False,
        )
        st.plotly_chart(fig_evla, width="stretch", config=PLOTLY_CONFIG)

with col_spray:
    st.markdown("#### Spray Chart")
    st.caption("Batted ball locations colored by estimated bases.")
    if "coord_x" in player_bb.columns and "coord_y" in player_bb.columns:
        spray_data = player_bb.dropna(subset=["coord_x", "coord_y"])
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
                x=spray_data["coord_x"],
                y=spray_data["coord_y"],
                mode="markers",
                marker=dict(
                    color=spray_data["estimated_bases"],
                    colorscale="RdYlGn",
                    size=6,
                    colorbar=dict(title="Est.<br>Bases"),
                ),
                customdata=np.stack([
                    spray_data["estimated_bases"],
                    spray_data["launch_speed"],
                    spray_data["actual_result"],
                ], axis=-1),
                hovertemplate=(
                    "Est. Bases: %{customdata[0]:.2f}<br>"
                    "Exit Velo: %{customdata[1]:.1f}<br>"
                    "Result: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            fig_spray.update_layout(
                height=500,
                template="plotly_white",
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False, autorange="reversed"),
                dragmode=False,
            )
            st.plotly_chart(fig_spray, width="stretch", config=PLOTLY_CONFIG_STATIC)

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

eb_bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, float("inf")]
eb_labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
player_eb_cats = pd.cut(player_bb["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False)
league_eb_cats = pd.cut(bb_df["estimated_bases"], bins=eb_bins, labels=eb_labels, right=False)

player_eb_dist = player_eb_cats.value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100
league_eb_dist = league_eb_cats.value_counts(normalize=True).reindex(eb_labels, fill_value=0) * 100

fig_eb_dist = go.Figure()
fig_eb_dist.add_trace(go.Bar(
    x=eb_labels, y=player_eb_dist.values,
    name=selected_player, marker_color=primary_color, opacity=0.85,
))
fig_eb_dist.add_trace(go.Bar(
    x=eb_labels, y=league_eb_dist.values,
    name="League", marker_color="gray", opacity=0.5,
))
fig_eb_dist.update_layout(
    barmode="group",
    xaxis_title="Estimated Bases",
    yaxis_title="% of Batted Balls",
    height=350,
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    margin=dict(t=30),
    dragmode=False,
)
st.plotly_chart(fig_eb_dist, width="stretch", config=PLOTLY_CONFIG)

# --- Contact Type Breakdown ---
st.markdown("#### Contact Type Breakdown")

type_order = ["Ground Ball", "Line Drive", "Fly Ball", "Pop Up"]

# Player type stats
type_stats = player_bb.groupby("bb_type").agg(
    count=("estimated_bases", "count"),
    avg_eb=("estimated_bases", "mean"),
    avg_ev=("launch_speed", "mean"),
    avg_la=("launch_angle", "mean"),
).reset_index()
type_stats["pct"] = (type_stats["count"] / type_stats["count"].sum() * 100).round(1)

# Pull% per type
if "spray_direction" in player_bb.columns:
    pull_by_type = player_bb.groupby("bb_type").apply(
        lambda g: (g["spray_direction"] == "Pull").mean() * 100
    ).rename("pull_pct")
    type_stats = type_stats.merge(pull_by_type, left_on="bb_type", right_index=True, how="left")
else:
    type_stats["pull_pct"] = float("nan")

# League type stats for comparison
bb_df["bb_type"] = bb_df["launch_angle"].apply(categorize_launch_angle)
league_type_stats = bb_df.groupby("bb_type").agg(
    lg_avg_eb=("estimated_bases", "mean"),
).reset_index()
type_stats = type_stats.merge(league_type_stats, on="bb_type", how="left")

type_stats["bb_type"] = pd.Categorical(type_stats["bb_type"], categories=type_order, ordered=True)
type_stats = type_stats.sort_values("bb_type").reset_index(drop=True)

# Display table
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

# --- vs LHP / vs RHP Splits ---
if not metadata_df.empty and "pitcher" in player_bb.columns:
    throw_hand_map = metadata_df.set_index("player_name")["throw_hand"].to_dict()
    player_bb["pitcher_hand"] = player_bb["pitcher"].map(throw_hand_map)

    splits_available = player_bb["pitcher_hand"].dropna()
    if len(splits_available) > 0:
        st.markdown("#### vs LHP / vs RHP")

        split_col_l, split_col_r = st.columns(2)
        for hand_label, hand_val, col in [("vs LHP", "L", split_col_l), ("vs RHP", "R", split_col_r)]:
            with col:
                with st.container(border=True):
                    subset = player_bb[player_bb["pitcher_hand"] == hand_val]
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
# BATTED BALL LOG
# =============================================================================

st.divider()
st.subheader("Batted Ball Log")
st.caption(f"{season} Season")

# --- Unluckiest Outs ---
st.markdown("#### Unluckiest Outs")
st.caption("Outs with the highest expected batting average — balls that should have been hits.")

outs = player_bb[player_bb["actual_tb"] == 0].copy()
if not outs.empty:
    unlucky_outs = outs.nlargest(10, "xba")

    out_display_cols = ["date", "opponent", "launch_speed", "launch_angle",
                        "spray_direction", "estimated_bases", "xba"]
    if "play_id" in unlucky_outs.columns:
        unlucky_outs["video"] = unlucky_outs["play_id"].apply(build_video_url)
        out_display_cols.append("video")

    available_cols = [c for c in out_display_cols if c in unlucky_outs.columns]
    out_display = unlucky_outs[available_cols].copy()

    col_rename = {
        "date": "Date", "opponent": "Opponent", "launch_speed": "Exit Velo",
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

# --- Luckiest Hits ---
st.markdown("#### Luckiest Hits")
st.caption("Hits with the lowest expected batting average — balls that shouldn't have been hits.")

hits = player_bb[player_bb["actual_tb"] > 0].copy()
if not hits.empty:
    lucky_hits = hits.nsmallest(10, "xba")

    hit_display_cols = ["date", "opponent", "launch_speed", "launch_angle",
                        "spray_direction", "actual_result", "estimated_bases", "xba"]
    if "play_id" in lucky_hits.columns:
        lucky_hits["video"] = lucky_hits["play_id"].apply(build_video_url)
        hit_display_cols.append("video")

    available_cols = [c for c in hit_display_cols if c in lucky_hits.columns]
    hit_display = lucky_hits[available_cols].copy()

    col_rename_hits = {
        "date": "Date", "opponent": "Opponent", "launch_speed": "Exit Velo",
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
    st.info("No hits recorded.")

# --- All Batted Balls Table ---
st.markdown("#### All Batted Balls")
st.caption("Full season batted ball log. Luck = actual total bases minus estimated bases per batted ball.")

all_bb_display = player_bb.sort_values("date_parsed", ascending=False).copy()
all_bb_display["luck"] = all_bb_display["actual_tb"] - all_bb_display["estimated_bases"]

all_bb_cols = ["date", "opponent", "launch_speed", "launch_angle",
               "spray_direction", "actual_result", "estimated_bases", "xba", "luck"]
if "play_id" in all_bb_display.columns:
    all_bb_display["video"] = all_bb_display["play_id"].apply(build_video_url)
    all_bb_cols.append("video")

available_cols = [c for c in all_bb_cols if c in all_bb_display.columns]
all_bb_show = all_bb_display[available_cols].copy()

all_bb_rename = {
    "date": "Date", "opponent": "Opponent", "launch_speed": "Exit Velo",
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
    file_name=f"{selected_player.replace(' ', '_')}_batted_balls_{season}.csv",
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

**Est. Bases per PA** is a statistical estimate of true production per plate appearance
that accounts for walks (1 base), HBP (1 base), and strikeouts (0 bases) alongside
batted ball quality. Players with fewer plate appearances are adjusted toward the
league average to prevent small-sample flukes. The comparison bar shows the player
vs. the best hitter and league average, with uncertainty ranges.

**Historical Timeline** tracks a player's Est. Bases / PA across seasons with uncertainty
bars. The gold diamond shows the best hitter each season for reference.

**Net Lucky Bases** = actual total bases minus expected total bases. Positive = lucky
(got more bases than contact quality suggested). Negative = unlucky. Over a full season,
extreme values tend to regress toward zero.
Tier labels: Very Unlucky (0-10th pct), Unlucky (10-30th), Luck Neutral (30-70th),
Lucky (70-90th), Very Lucky (90-100th).

**Barrel**: Exit velocity >= 98 mph with a launch angle in the "sweet spot" zone (26-30+
degrees, expanding with higher EV). Barrels produce the highest expected bases.

**Contact types**: Ground Ball (LA < 10°), Line Drive (10-25°), Fly Ball (25-50°),
Pop Up (50°+).

{"**Video links**: Click to watch the play on Baseball Savant." if "play_id" in bb_df.columns else ""}
    """)

render_home_link()
