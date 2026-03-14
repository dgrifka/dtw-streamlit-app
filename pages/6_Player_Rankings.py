"""
Player Rankings

Bayesian hierarchical rankings of MLB hitters and pitchers by estimated
bases, with credible intervals and shrinkage for small sample sizes.
"""

import unicodedata
import urllib.parse

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_player_evaluations,
    load_player_evaluations_pa,
    load_player_metadata,
    load_batted_balls,
    get_player_evaluation_image_url,
    get_player_evaluation_team_image_url,
    get_available_player_evaluation_seasons,
)
from utils.team_mappings import TEAM_COLORS
from utils.player_analytics import compute_platoon_splits
from utils.responsive import inject_responsive_css, render_home_link

# Page config
st.set_page_config(
    page_title="Player Rankings | DTW Simulator",
    page_icon="⚾",
    layout="wide",
)

inject_responsive_css()

_logo_path = os.path.join(parent_dir, "assets", "mlb_simulator_logo.png")
if os.path.exists(_logo_path):
    st.logo(_logo_path)

st.title("Player Rankings")
st.markdown(
    "Statistical rankings of hitters and pitchers that account for sample size. "
    "Players with fewer at-bats are pulled toward the league average, while "
    "players with more data keep estimates closer to their observed performance."
)


def _normalize(text):
    """Strip accents and lowercase for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# =============================================================================
# CONTROLS (top-level — these affect the chart image)
# =============================================================================

col_season, col_type, col_metric, col_team = st.columns([1, 1, 1, 1])

with col_season:
    available_seasons = get_available_player_evaluation_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0)
    else:
        season = pd.Timestamp.now().year

with col_type:
    player_type = st.selectbox("Player type", options=["Hitter", "Pitcher"])

with col_metric:
    metric_mode = st.selectbox("Metric", options=["Per Plate Appearance", "Per Batted Ball"])

with col_team:
    all_teams = sorted(TEAM_COLORS.keys())
    selected_team = st.selectbox("Team", options=["All Teams"] + all_teams)

type_key = player_type.lower()
is_pa_mode = metric_mode == "Per Plate Appearance"
is_pitcher = type_key == "pitcher"
metric_short = "EB/PA" if is_pa_mode else "EB/BB"
count_label = "Plate Appearances" if is_pa_mode else "Batted Balls"

# Load rankings data
if is_pa_mode:
    df = load_player_evaluations_pa(season, type_key)
else:
    df = load_player_evaluations(season, type_key)

if df.empty:
    st.info(
        f"No {type_key} evaluation data available for {season}. "
        f"Data is generated weekly during the season."
    )
    st.stop()

# Merge position from metadata
metadata_df = load_player_metadata(season)
if not metadata_df.empty and "position" in metadata_df.columns:
    meta_slim = metadata_df[["player_name", "position"]].drop_duplicates(
        subset=["player_name"]
    ).rename(columns={"player_name": "player"})
    df = df.merge(meta_slim, on="player", how="left")
    df["position"] = df["position"].fillna("")
else:
    df["position"] = ""


# =============================================================================
# SECTION 1: HOW IT WORKS
# =============================================================================

with st.expander("How does this work?"):
    if is_pa_mode:
        st.markdown("""
**Per Plate Appearance mode** measures overall offensive production, not just contact quality.
Each plate appearance outcome is valued: walks = 1 base, HBP = 1 base, strikeouts = 0 bases,
and batted balls use the model's estimated bases (based on exit velocity, launch angle, and spray angle).

**How it works:**

Instead of treating each player independently, the model learns a league-wide baseline and
estimates how much each player deviates from it. Every player's estimate is informed by both
their own data and the overall population — so small samples get pulled toward the average,
while players with lots of data keep estimates close to their raw numbers.

**Reading the chart:**

- **Circle**: the model's best estimate of a player's true production
- **Thick line**: likely range (50% of the time, the true value falls here)
- **Thin line**: wider uncertainty range (89%)

**Validation (2024-2025 holdout):**

Predictions from end-of-2024 data, validated against 2025 outcomes (min 100 PA both seasons):
- EB/PA R² = 0.140 — modest but real year-over-year predictive power
- Bayesian posterior mean beats raw rate (+0.004 R²) — shrinkage helps
""")
    else:
        st.markdown("""
**Per Batted Ball mode** measures contact quality only — how well a player hits the ball when
they put it in play, based on exit velocity, launch angle, and spray angle.

**Why not just use raw averages?**

A player with 20 batted balls and a high average might just be on a hot streak. The model
recognizes the small sample and **pulls their estimate toward the league average**.
A player with 400+ batted balls keeps an estimate much closer to their raw numbers,
because there's enough data to trust it.

**Reading the chart:**

- **Circle**: the model's best estimate of a player's true contact quality
- **Thick line**: likely range (50% of the time, the true value falls here)
- **Thin line**: wider uncertainty range (89%)
- Players with fewer batted balls have wider ranges, reflecting greater uncertainty
""")

# =============================================================================
# SECTION 2: S3 MATPLOTLIB CHART + DOWNLOAD
# =============================================================================

chart_name = f"top_{type_key}s_pa" if is_pa_mode else f"top_{type_key}s"

if selected_team != "All Teams":
    chart_url = get_player_evaluation_team_image_url(season, selected_team, chart_name)
else:
    chart_url = get_player_evaluation_image_url(season, chart_name)

_chart_loaded = False
try:
    st.image(chart_url, width="stretch")
    _chart_loaded = True
except Exception:
    if selected_team != "All Teams":
        fallback_url = get_player_evaluation_image_url(season, chart_name)
        try:
            st.image(fallback_url, width="stretch")
            chart_url = fallback_url
            _chart_loaded = True
        except Exception:
            st.warning(f"Chart image not available for {season}.")
    else:
        st.warning(f"Chart image not available for {season}.")

if _chart_loaded:
    st.markdown(
        f"[Download chart image]({chart_url})",
        help="Right-click the link or the image above to save/copy.",
    )

# =============================================================================
# SECTION 3: RANKINGS TABLE
# =============================================================================

st.divider()
st.subheader(f"{player_type} rankings table")

# Filters for the table (search, position, min count)
col_search, col_pos, col_min_bb = st.columns([1, 1, 1])

with col_search:
    search_query = st.text_input(
        "Search player",
        placeholder="e.g. Ohtani, Juan, Suarez",
    )

with col_pos:
    position_options = ["All"]
    if type_key == "hitter":
        position_options += ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
    else:
        position_options += ["SP", "RP"]
    position_filter = st.selectbox("Position", position_options)

with col_min_bb:
    default_min = 100 if is_pa_mode else 30
    min_bb = st.slider(
        f"Min {count_label.lower()}",
        min_value=1,
        max_value=int(df["n_batted_balls"].max()),
        value=min(default_min, int(df["n_batted_balls"].max())),
        step=10,
    )

# Apply filters
filtered = df[df["n_batted_balls"] >= min_bb].copy()
if selected_team != "All Teams":
    filtered = filtered[filtered["team"] == selected_team]
if position_filter != "All":
    filtered = filtered[
        filtered["position"].str.contains(position_filter, case=False, na=False)
    ]
if search_query.strip():
    query_norm = _normalize(search_query.strip())
    filtered = filtered[
        filtered["player"].apply(lambda name: query_norm in _normalize(name))
    ]

filtered = filtered.sort_values(
    "posterior_mean", ascending=is_pitcher
).reset_index(drop=True)

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Total {player_type}s", f"{len(filtered):,}")
m2.metric("Avg est. bases", f"{filtered['posterior_mean'].mean():.3f}")
m3.metric("Avg raw rate", f"{filtered['raw_rate'].mean():.3f}")
m4.metric(f"Avg {count_label.lower()}", f"{filtered['n_batted_balls'].mean():.0f}")

display = filtered.copy()

# Drop chart-only columns before display
for col in ["hdi_50_low", "hdi_50_high"]:
    if col in display.columns:
        display = display.drop(columns=[col])

display.index = range(1, len(display) + 1)
display.index.name = "Rank"

n_col_name = count_label
display = display.rename(columns={
    "player": "Player",
    "team": "Team",
    "position": "Pos",
    "n_batted_balls": n_col_name,
    "posterior_mean": "Est. Bases (Model)",
    "raw_rate": "Est. Bases (Raw)",
    "hdi_low": "Range Low",
    "hdi_high": "Range High",
    "shrinkage": "Adjustment",
})

_profile_page = "Pitcher_Profile" if is_pitcher else "Hitter_Profile"
display["Profile"] = display["Player"].apply(
    lambda p: f"/{_profile_page}?player={urllib.parse.quote(p)}"
)

table_cols = [
    "Player", "Team", "Pos", "Est. Bases (Model)", "Est. Bases (Raw)",
    "Range Low", "Range High", "Adjustment", n_col_name, "Profile",
]
table_cols = [c for c in table_cols if c in display.columns]

COLUMN_CONFIG = {
    "Est. Bases (Model)": st.column_config.NumberColumn(format="%.4f"),
    "Est. Bases (Raw)": st.column_config.NumberColumn(format="%.4f"),
    "Range Low": st.column_config.NumberColumn(format="%.4f"),
    "Range High": st.column_config.NumberColumn(format="%.4f"),
    "Adjustment": st.column_config.NumberColumn(format="%+.4f"),
    n_col_name: st.column_config.NumberColumn(format="%d"),
    "Pos": st.column_config.TextColumn(width="small"),
    "Profile": st.column_config.LinkColumn(display_text="View"),
}

st.dataframe(
    display[table_cols],
    width="stretch",
    column_config=COLUMN_CONFIG,
)

csv_data = display[table_cols].to_csv(index=True)
st.download_button(
    "Download CSV",
    csv_data,
    f"player_rankings_{type_key}_{season}.csv",
    "text/csv",
)


# =============================================================================
# SECTION 4: SMALL SAMPLE STANDOUTS (Hitters, PA mode only)
# =============================================================================

if type_key == "hitter" and is_pa_mode and df["hdi_high"].notna().any():
    st.divider()
    st.subheader("Small sample standouts")

    # Use full df with a low floor (30 PA) to find small-sample players
    # whose wide HDI intervals give them elite ceilings
    standouts_pool = df[df["n_batted_balls"] >= 30].copy()
    if selected_team != "All Teams":
        standouts_pool = standouts_pool[standouts_pool["team"] == selected_team]

    col_upside, col_floor = st.columns(2)

    with col_upside:
        st.markdown("**High upside**")
        st.caption(
            "Players whose HDI ceiling is elite even though their point estimate "
            "is moderate — lottery tickets with breakout potential."
        )

        if standouts_pool["hdi_high"].notna().any():
            elite_threshold = standouts_pool["hdi_high"].quantile(0.80)
            moderate_threshold = standouts_pool["posterior_mean"].quantile(0.70)

            upside = standouts_pool[
                (standouts_pool["hdi_high"] >= elite_threshold) &
                (standouts_pool["posterior_mean"] < moderate_threshold)
            ].sort_values("hdi_high", ascending=False).head(15)

            if upside.empty:
                st.info("No high-upside players found with current filters.")
            else:
                up_display = upside[["player", "team", "posterior_mean",
                                      "hdi_low", "hdi_high", "n_batted_balls"]].copy()
                up_display["Profile"] = up_display["player"].apply(
                    lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                )
                up_display = up_display.rename(columns={
                    "player": "Player", "team": "Team",
                    "posterior_mean": metric_short, "hdi_low": "Floor",
                    "hdi_high": "Ceiling", "n_batted_balls": "PA",
                })
                st.dataframe(
                    up_display, hide_index=True, use_container_width=True,
                    column_config={
                        metric_short: st.column_config.NumberColumn(format="%.3f"),
                        "Floor": st.column_config.NumberColumn(format="%.3f"),
                        "Ceiling": st.column_config.NumberColumn(format="%.3f"),
                        "PA": st.column_config.NumberColumn(format="%d"),
                        "Profile": st.column_config.LinkColumn(display_text="View"),
                    },
                )
        else:
            st.info("Bayesian ranking data not available.")

    with col_floor:
        st.markdown("**Reliable floor**")
        st.caption(
            "Above-median players with the narrowest credible intervals — "
            "consistent production, less risk."
        )

        # Reliable floor uses the main filtered set (high-PA players)
        with_hdi = filtered[filtered["hdi_high"].notna()].copy()
        if not with_hdi.empty:
            with_hdi["hdi_width"] = with_hdi["hdi_high"] - with_hdi["hdi_low"]

            above_avg = with_hdi[
                with_hdi["posterior_mean"] >= with_hdi["posterior_mean"].median()
            ]
            safe_floor = above_avg.nsmallest(15, "hdi_width")

            if safe_floor.empty:
                st.info("No reliable-floor players found with current filters.")
            else:
                sf_display = safe_floor[["player", "team", "posterior_mean",
                                          "hdi_low", "hdi_high", "n_batted_balls"]].copy()
                sf_display["Profile"] = sf_display["player"].apply(
                    lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                )
                sf_display = sf_display.rename(columns={
                    "player": "Player", "team": "Team",
                    "posterior_mean": metric_short, "hdi_low": "Floor",
                    "hdi_high": "Ceiling", "n_batted_balls": "PA",
                })
                st.dataframe(
                    sf_display, hide_index=True, use_container_width=True,
                    column_config={
                        metric_short: st.column_config.NumberColumn(format="%.3f"),
                        "Floor": st.column_config.NumberColumn(format="%.3f"),
                        "Ceiling": st.column_config.NumberColumn(format="%.3f"),
                        "PA": st.column_config.NumberColumn(format="%d"),
                        "Profile": st.column_config.LinkColumn(display_text="View"),
                    },
                )
        else:
            st.info("Bayesian ranking data not available.")


# =============================================================================
# SECTION 5: PLATOON ADVANTAGE FINDER (hitters only)
# =============================================================================

if type_key == "hitter":
    st.divider()
    st.subheader("Platoon advantage finder")
    st.caption(
        "Hitters with the biggest performance gap vs left-handed or "
        "right-handed pitching. Based on batted ball data."
    )

    bb_df = load_batted_balls(season)

    if bb_df.empty:
        st.info(f"No batted ball data available for {season}.")
    elif metadata_df.empty:
        st.info("Player metadata not available (need pitcher throw hand).")
    else:
        col_plat_pos, col_plat_min = st.columns([1, 1])
        with col_plat_pos:
            plat_pos = st.selectbox(
                "Position (platoon)",
                ["All", "C", "1B", "2B", "SS", "3B", "OF", "DH"],
                key="platoon_pos",
            )
        with col_plat_min:
            plat_min_bb = st.slider(
                "Min BB per side", 10, 50, 15, step=5, key="platoon_min"
            )

        platoon_df = compute_platoon_splits(bb_df, metadata_df, min_bb=plat_min_bb)

        if platoon_df.empty:
            st.info("Platoon data not available (need pitcher metadata for throw hand).")
        else:
            # Filter by position
            if plat_pos != "All" and not metadata_df.empty:
                pos_players = metadata_df[
                    metadata_df["position"].str.contains(
                        plat_pos, case=False, na=False
                    )
                ]["player_name"].tolist()
                platoon_df = platoon_df[platoon_df["player"].isin(pos_players)]

            if platoon_df.empty:
                st.info("No players match the platoon filters.")
            else:
                # Dumbbell chart — top 15 by absolute platoon gap
                top_plat = platoon_df.head(15).copy()
                top_plat = top_plat.sort_values("platoon_gap", key=abs, ascending=True)

                fig_dumb = go.Figure()

                # Connecting lines
                for _, row in top_plat.iterrows():
                    fig_dumb.add_trace(go.Scatter(
                        x=[row["vs_lhp_eb"], row["vs_rhp_eb"]],
                        y=[row["player"], row["player"]],
                        mode="lines",
                        line=dict(color="#d1d5db", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # vs LHP dots
                fig_dumb.add_trace(go.Scatter(
                    x=top_plat["vs_lhp_eb"],
                    y=top_plat["player"],
                    mode="markers",
                    marker=dict(color="#2563eb", size=10),
                    name="vs LHP",
                    customdata=top_plat["vs_lhp_n"].astype(int).values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "vs LHP: %{x:.3f} EB/BB<br>"
                        "n=%{customdata}<extra></extra>"
                    ),
                ))

                # vs RHP dots
                fig_dumb.add_trace(go.Scatter(
                    x=top_plat["vs_rhp_eb"],
                    y=top_plat["player"],
                    mode="markers",
                    marker=dict(color="#dc2626", size=10),
                    name="vs RHP",
                    customdata=top_plat["vs_rhp_n"].astype(int).values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "vs RHP: %{x:.3f} EB/BB<br>"
                        "n=%{customdata}<extra></extra>"
                    ),
                ))

                fig_dumb.update_layout(
                    template="plotly_white",
                    xaxis_title="EB/BB",
                    height=max(400, len(top_plat) * 30),
                    margin=dict(l=140, r=20, t=20, b=40),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                    ),
                    yaxis=dict(tickfont=dict(size=11)),
                )
                st.plotly_chart(fig_dumb, use_container_width=True, theme=None)

                # Platoon table
                plat_display = platoon_df.head(30).copy()
                plat_display["Profile"] = plat_display["player"].apply(
                    lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                )
                plat_display = plat_display.rename(columns={
                    "player": "Player",
                    "team": "Team",
                    "vs_lhp_eb": "vs LHP (EB/BB)",
                    "vs_rhp_eb": "vs RHP (EB/BB)",
                    "vs_lhp_n": "vs LHP n",
                    "vs_rhp_n": "vs RHP n",
                    "vs_lhp_ev": "vs LHP EV",
                    "vs_rhp_ev": "vs RHP EV",
                    "vs_lhp_barrel": "vs LHP Barrel%",
                    "vs_rhp_barrel": "vs RHP Barrel%",
                    "platoon_gap": "Gap",
                    "platoon_pct_diff": "Gap %",
                })

                plat_cols = [
                    "Player", "Team", "vs LHP (EB/BB)", "vs RHP (EB/BB)", "Gap",
                    "vs LHP n", "vs RHP n", "vs LHP EV", "vs RHP EV",
                    "vs LHP Barrel%", "vs RHP Barrel%", "Profile",
                ]
                plat_cols = [c for c in plat_cols if c in plat_display.columns]

                st.dataframe(
                    plat_display[plat_cols],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "vs LHP (EB/BB)": st.column_config.NumberColumn(format="%.3f"),
                        "vs RHP (EB/BB)": st.column_config.NumberColumn(format="%.3f"),
                        "Gap": st.column_config.NumberColumn(format="%.3f"),
                        "vs LHP EV": st.column_config.NumberColumn(format="%.1f"),
                        "vs RHP EV": st.column_config.NumberColumn(format="%.1f"),
                        "vs LHP Barrel%": st.column_config.NumberColumn(format="%.1f%%"),
                        "vs RHP Barrel%": st.column_config.NumberColumn(format="%.1f%%"),
                        "Profile": st.column_config.LinkColumn(display_text="View"),
                    },
                )


# =============================================================================
# FOOTER
# =============================================================================

render_home_link()
