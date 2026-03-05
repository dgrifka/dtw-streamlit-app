"""
Player Dashboard Page

Individual player deep-dive: contact quality, luck report, and batted ball
visualizations with Bayesian uncertainty from the DTW Simulator model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_batted_balls, get_available_batted_ball_seasons,
    load_player_evaluations_pa, load_player_metadata, load_pa_counts,
)
from utils.team_mappings import (
    get_team_color, get_team_logo_url, get_short_name,
    TEAM_NAME_MAPPING,
)

# Page config
st.set_page_config(
    page_title="Player Dashboard | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(name):
    """Strip accents for fuzzy search."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()


def build_headshot_url(player_id):
    """MLB CDN headshot URL."""
    return (
        f"https://img.mlb.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/"
        f"w_213,q_auto:best/v1/people/{player_id}/headshot/67/current"
    )


def build_video_url(play_id):
    """Baseball Savant video link from play_id."""
    if pd.isna(play_id) or play_id == "":
        return None
    return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


def categorize_launch_angle(la):
    """Categorize launch angle into batted ball type."""
    if pd.isna(la):
        return "Unknown"
    if la < -10:
        return "Ground Ball"
    elif la < 10:
        return "Ground Ball"
    elif la < 25:
        return "Line Drive"
    elif la < 50:
        return "Fly Ball"
    else:
        return "Pop Up"


def is_barrel(ev, la):
    """Check if a batted ball is a barrel (Statcast definition approximation)."""
    if pd.isna(ev) or pd.isna(la):
        return False
    return ev >= 98 and 26 <= la <= 30 + (ev - 98)


# Full name → short name lookup for batted ball data (which uses short names)
FULL_TO_SHORT = {v: k for k, v in {v: k for k, v in TEAM_NAME_MAPPING.items()}.items()}
# Actually just use get_short_name


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
    st.title("Player Dashboard")
    st.info(f"No batted ball data available for {season}.")
    st.stop()

# Load supplementary data (may be empty — degrade gracefully)
metadata_df = load_player_metadata(season)
pa_rankings = load_player_evaluations_pa(season, "hitter")
pa_counts_df = load_pa_counts(season)


# =============================================================================
# PLAYER SEARCH
# =============================================================================

st.title("Player Dashboard")

# Check if we arrived via query param (cross-page linking)
query_player = st.query_params.get("player", "")

# Build unique player list from batted ball data
player_teams = bb_df.groupby("player")["team"].first().reset_index()
player_list = sorted(player_teams["player"].unique())

# Search box
player_search = st.text_input(
    "Search player",
    value=query_player,
    placeholder="Type a player name...",
    key="pd_search",
)

if not player_search:
    # Show a landing state
    st.markdown("---")
    st.markdown("Search for any player to see their contact quality profile, luck report, and batted ball visualizations.")

    # Trending players: biggest movers in Bayesian EB
    if not pa_rankings.empty and len(pa_rankings) >= 10:
        st.subheader("Top Contact Quality (Bayesian)")
        top = pa_rankings.head(10)[["player", "team", "posterior_mean", "hdi_low", "hdi_high"]].copy()
        top.columns = ["Player", "Team", "EB/PA", "HDI Low", "HDI High"]
        st.dataframe(
            top,
            hide_index=True,
            use_container_width=True,
            column_config={
                "EB/PA": st.column_config.NumberColumn(format="%.3f"),
                "HDI Low": st.column_config.NumberColumn(format="%.3f"),
                "HDI High": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.stop()

# Fuzzy match
search_norm = normalize_name(player_search)
matches = [p for p in player_list if search_norm in normalize_name(p)]

if not matches:
    st.warning(f"No players found matching '{player_search}'.")
    st.stop()

if len(matches) > 1:
    selected_player = st.selectbox(
        "Select player",
        options=matches,
        index=0,
        key="pd_player_select",
    )
else:
    selected_player = matches[0]

# Filter batted ball data to this player
player_bb = bb_df[bb_df["player"] == selected_player].copy()
if player_bb.empty:
    st.warning(f"No batted ball data found for {selected_player}.")
    st.stop()

# Resolve team (most recent)
player_bb = player_bb.sort_values("date_parsed")
player_team_short = player_bb["team"].iloc[-1]

# League averages for context
league_avg_eb = bb_df["estimated_bases"].mean()
league_avg_ev = bb_df["launch_speed"].mean()
league_avg_xba = bb_df["xba"].mean()


# =============================================================================
# HERO SECTION
# =============================================================================

st.divider()

# Look up player metadata for headshot + demographics
player_meta = None
player_id = None
if not metadata_df.empty:
    # Match by name (metadata uses full name from MLB API)
    meta_match = metadata_df[metadata_df["player_name"] == selected_player]
    if meta_match.empty:
        # Try fuzzy
        meta_match = metadata_df[
            metadata_df["player_name"].apply(normalize_name).str.contains(search_norm)
        ]
    if not meta_match.empty:
        player_meta = meta_match.iloc[0]
        player_id = int(player_meta["player_id"])

# Look up Bayesian ranking
player_ranking = None
if not pa_rankings.empty:
    rank_match = pa_rankings[pa_rankings["player"] == selected_player]
    if not rank_match.empty:
        player_ranking = rank_match.iloc[0]

# Team color
primary_color, secondary_color = get_team_color(player_team_short)

# Hero layout
hero_left, hero_mid, hero_right = st.columns([1, 2, 2])

with hero_left:
    if player_id:
        st.image(build_headshot_url(player_id), width=160)
    else:
        logo_url = get_team_logo_url(player_team_short)
        if logo_url:
            st.image(logo_url, width=120)

with hero_mid:
    st.markdown(f"### {selected_player}")
    # Team + position line
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

    st.markdown(f"**{player_team_short}**{pos_str}{age_str}")

    # Quick stats row
    avg_eb = player_bb["estimated_bases"].mean()
    avg_ev = player_bb["launch_speed"].mean()
    n_bb = len(player_bb)

    qs1, qs2, qs3 = st.columns(3)
    qs1.metric("Batted Balls", f"{n_bb:,}")
    qs2.metric("Avg EV", f"{avg_ev:.1f} mph")
    qs3.metric("Avg EB/BB", f"{avg_eb:.3f}")

with hero_right:
    if player_ranking is not None:
        bayesian_eb = player_ranking["posterior_mean"]
        hdi_low = player_ranking["hdi_low"]
        hdi_high = player_ranking["hdi_high"]

        # Percentile rank
        if not pa_rankings.empty:
            pct = (pa_rankings["posterior_mean"] < bayesian_eb).mean() * 100
        else:
            pct = 50

        st.metric("Bayesian EB/PA", f"{bayesian_eb:.3f}")
        st.caption(f"89% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]")
        st.caption(f"League percentile: {pct:.0f}th")

        # Shrinkage
        raw_rate = player_ranking.get("raw_rate", avg_eb)
        shrinkage_pct = player_ranking.get("shrinkage", 0)
        if shrinkage_pct:
            st.caption(f"Shrinkage: {shrinkage_pct:.1%} toward league mean")
    else:
        # Fallback: raw stats
        st.metric("Raw EB/BB", f"{avg_eb:.3f}")
        st.caption("Bayesian estimate not available (need PA rankings data)")


# =============================================================================
# TABS
# =============================================================================

tab_contact, tab_luck = st.tabs(["Contact Quality", "Luck Report"])


# =============================================================================
# CONTACT QUALITY TAB
# =============================================================================

with tab_contact:
    st.subheader("Contact Quality Profile")

    # Add derived columns
    player_bb["bb_type"] = player_bb["launch_angle"].apply(categorize_launch_angle)
    player_bb["is_barrel"] = player_bb.apply(
        lambda r: is_barrel(r["launch_speed"], r["launch_angle"]), axis=1
    )

    # --- EV x LA Scatter (the showstopper chart) ---
    st.markdown("#### Exit Velocity vs Launch Angle")
    st.caption("Each point is a batted ball, colored by estimated bases. Hover for details.")

    fig_scatter = px.scatter(
        player_bb,
        x="launch_speed",
        y="launch_angle",
        color="estimated_bases",
        color_continuous_scale="RdYlGn",
        hover_data={
            "launch_speed": ":.1f",
            "launch_angle": ":.0f",
            "estimated_bases": ":.2f",
            "actual_result": True,
            "date": True,
            "opponent": True,
        },
        labels={
            "launch_speed": "Exit Velocity (mph)",
            "launch_angle": "Launch Angle (deg)",
            "estimated_bases": "Est. Bases",
        },
    )
    fig_scatter.update_layout(
        height=500,
        coloraxis_colorbar_title="Est.<br>Bases",
        template="plotly_white",
    )
    # Add reference lines
    fig_scatter.add_hline(y=10, line_dash="dot", line_color="gray", opacity=0.4,
                          annotation_text="Line Drive zone", annotation_position="top right")
    fig_scatter.add_hline(y=25, line_dash="dot", line_color="gray", opacity=0.4)
    fig_scatter.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4,
                          annotation_text="Fly Ball / Pop Up", annotation_position="top right")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Distribution charts row ---
    col_ev, col_la = st.columns(2)

    with col_ev:
        st.markdown("#### Exit Velocity Distribution")
        fig_ev = go.Figure()
        # League background
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
            opacity=0.7,
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
        )
        st.plotly_chart(fig_ev, use_container_width=True)

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
            opacity=0.7,
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
        )
        st.plotly_chart(fig_la, use_container_width=True)

    # --- Spray Chart ---
    if "coord_x" in player_bb.columns and "coord_y" in player_bb.columns:
        spray_data = player_bb.dropna(subset=["coord_x", "coord_y"])
        if not spray_data.empty:
            st.markdown("#### Spray Chart")
            st.caption("Batted ball locations colored by estimated bases.")

            fig_spray = px.scatter(
                spray_data,
                x="coord_x",
                y="coord_y",
                color="estimated_bases",
                color_continuous_scale="RdYlGn",
                hover_data={
                    "estimated_bases": ":.2f",
                    "launch_speed": ":.1f",
                    "actual_result": True,
                    "coord_x": False,
                    "coord_y": False,
                },
                labels={"estimated_bases": "Est. Bases"},
            )
            fig_spray.update_layout(
                height=500,
                width=500,
                template="plotly_white",
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False, autorange="reversed"),
                coloraxis_colorbar_title="Est.<br>Bases",
            )
            st.plotly_chart(fig_spray, use_container_width=False)

    # --- Contact Type Breakdown ---
    st.markdown("#### Contact Type Breakdown")

    type_stats = player_bb.groupby("bb_type").agg(
        count=("estimated_bases", "count"),
        avg_eb=("estimated_bases", "mean"),
        avg_ev=("launch_speed", "mean"),
    ).reset_index()
    type_stats["pct"] = (type_stats["count"] / type_stats["count"].sum() * 100).round(1)
    type_order = ["Ground Ball", "Line Drive", "Fly Ball", "Pop Up"]
    type_stats["bb_type"] = pd.Categorical(type_stats["bb_type"], categories=type_order, ordered=True)
    type_stats = type_stats.sort_values("bb_type").reset_index(drop=True)

    barrel_count = player_bb["is_barrel"].sum()
    barrel_rate = barrel_count / len(player_bb) * 100

    cs1, cs2, cs3, cs4, cs5 = st.columns(5)
    for i, (_, row) in enumerate(type_stats.iterrows()):
        col = [cs1, cs2, cs3, cs4][i] if i < 4 else cs4
        col.metric(row["bb_type"], f"{row['pct']:.1f}%", help=f"Avg EB: {row['avg_eb']:.3f}")
    cs5.metric("Barrel Rate", f"{barrel_rate:.1f}%", help=f"{barrel_count} barrels")


# =============================================================================
# LUCK REPORT TAB
# =============================================================================

with tab_luck:
    st.subheader("Luck Report")

    # Actual total bases mapping
    TB_MAP = {
        "Single": 1, "Double": 2, "Triple": 3, "Home Run": 4,
        "Out": 0, "Field Out": 0, "Grounded Into Dp": 0,
        "Fielders Choice": 0, "Fielders Choice Out": 0,
        "Sac Fly": 0, "Sac Bunt": 0, "Double Play": 0,
        "Force Out": 0, "Field Error": 0,
    }
    player_bb["actual_tb"] = player_bb["actual_result"].map(TB_MAP).fillna(0)

    total_actual_tb = player_bb["actual_tb"].sum()
    total_expected_tb = player_bb["estimated_bases"].sum()
    luck_score = total_actual_tb - total_expected_tb

    # Luck metrics
    lm1, lm2, lm3 = st.columns(3)
    lm1.metric("Actual Total Bases", f"{total_actual_tb:.0f}")
    lm2.metric("Expected Total Bases", f"{total_expected_tb:.1f}")
    delta_color = "normal" if luck_score >= 0 else "inverse"
    lm3.metric("Luck Score", f"{luck_score:+.1f}", delta=f"{'Lucky' if luck_score > 0 else 'Unlucky'}", delta_color=delta_color)

    # Luck percentile
    if len(bb_df) > 1000:
        # Compute luck score for all players with enough BBs
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
            luck_pct = (player_luck["luck"] < luck_score).mean() * 100
            st.caption(f"Luck percentile: {luck_pct:.0f}th (among players with 30+ batted balls)")

    # --- Luck Accumulation Chart ---
    st.markdown("#### Luck Accumulation Over Time")
    st.caption("Cumulative actual total bases minus expected. Rising = getting lucky. Falling = unlucky.")

    # Sort chronologically and compute cumulative
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
    )
    st.plotly_chart(fig_luck, use_container_width=True)

    # --- Unluckiest Outs ---
    st.markdown("#### Unluckiest Outs")
    st.caption("Outs with the highest expected batting average — balls that should have been hits.")

    outs = player_bb[player_bb["actual_tb"] == 0].copy()
    if not outs.empty:
        unlucky_outs = outs.nlargest(10, "xba")

        out_display_cols = ["date", "opponent", "launch_speed", "launch_angle",
                            "spray_direction", "estimated_bases", "xba"]
        # Add video link if play_id exists
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
            col_config["Video"] = st.column_config.LinkColumn(display_text="▶️")

        st.dataframe(out_display, hide_index=True, use_container_width=True,
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
            hit_col_config["Video"] = st.column_config.LinkColumn(display_text="▶️")

        st.dataframe(hit_display, hide_index=True, use_container_width=True,
                      column_config=hit_col_config)
    else:
        st.info("No hits recorded.")


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()
with st.expander("Methodology"):
    st.markdown(f"""
**Estimated Bases** is the model-predicted expected bases for each batted ball:
`P(1B)*1 + P(2B)*2 + P(3B)*3 + P(HR)*4`. The probabilities come from a Gradient
Boosting Classifier trained on Statcast data, accounting for exit velocity, launch angle,
spray angle, and ballpark.

**Bayesian EB/PA** is a hierarchical Bayesian estimate (NumPyro NUTS MCMC) that shrinks
small-sample players toward the league mean. The 89% HDI (Highest Density Interval) is the
range within which the player's true talent level lies with 89% probability. Players with
fewer plate appearances get more shrinkage — this is a feature, not a bug.

**Luck Score** = actual total bases minus expected total bases (sum of estimated bases for
each batted ball). Positive = lucky (got more bases than expected from contact quality).
Negative = unlucky. Over a full season, extreme luck scores tend to regress toward zero.

**Barrel**: Exit velocity >= 98 mph with a launch angle in the "sweet spot" zone (26-30+
degrees, expanding with higher EV). Barrels produce the highest expected bases.

**Contact types**: Ground Ball (LA < 10°), Line Drive (10-25°), Fly Ball (25-50°),
Pop Up (50°+).

{"**Video links**: Click ▶️ to watch the play on Baseball Savant." if "play_id" in bb_df.columns else ""}
    """)
