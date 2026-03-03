"""
Player Evaluations Page

Bayesian hierarchical rankings of MLB hitters and pitchers by estimated
bases per batted ball, with credible intervals and shrinkage for small
sample sizes.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_player_evaluations,
    get_player_evaluation_image_url,
    get_available_batted_ball_seasons,
)

# Page config
st.set_page_config(
    page_title="Player Evaluations | DTW Simulator",
    page_icon="⚾",
    layout="wide",
)

# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

st.title("Player Evaluations")
st.markdown(
    "Bayesian rankings of hitters and pitchers by contact quality "
    "(estimated bases per batted ball), with credible intervals."
)

# Season selector
col_season, col_type, _ = st.columns([1, 1, 2])
with col_season:
    available_seasons = get_available_batted_ball_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0)
    else:
        season = pd.Timestamp.now().year

with col_type:
    player_type = st.selectbox("Player Type", options=["Hitter", "Pitcher"])

type_key = player_type.lower()

# Load data
df = load_player_evaluations(season, type_key)

if df.empty:
    st.info(
        f"No {type_key} evaluation data available for {season}. "
        f"Data is generated daily during the season."
    )
    st.stop()

# -----------------------------------------------------------------------------
# CHART IMAGE FROM S3
# -----------------------------------------------------------------------------

st.divider()

chart_name = f"top_{type_key}s"
chart_url = get_player_evaluation_image_url(season, chart_name)

try:
    st.image(chart_url, use_container_width=True)
except Exception:
    st.warning(f"Chart image not available for {season}.")

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

st.divider()
st.subheader(f"{player_type} Rankings Table")

col_team, col_min_bb, _ = st.columns([1, 1, 2])

with col_team:
    teams = sorted(df["team"].unique())
    selected_team = st.selectbox("Filter by Team", options=["All Teams"] + teams)

with col_min_bb:
    min_bb = st.slider(
        "Min Batted Balls",
        min_value=1,
        max_value=int(df["n_batted_balls"].max()),
        value=30,
        step=10,
    )

# Apply filters
filtered = df[df["n_batted_balls"] >= min_bb].copy()
if selected_team != "All Teams":
    filtered = filtered[filtered["team"] == selected_team]

if type_key == "pitcher":
    filtered = filtered.sort_values("posterior_mean", ascending=True).reset_index(drop=True)
else:
    filtered = filtered.sort_values("posterior_mean", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# METRICS ROW
# -----------------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Total {player_type}s", f"{len(filtered):,}")
m2.metric("Avg Est. Bases", f"{filtered['posterior_mean'].mean():.3f}")
m3.metric("Avg Raw Rate", f"{filtered['raw_rate'].mean():.3f}")
m4.metric("Avg Batted Balls", f"{filtered['n_batted_balls'].mean():.0f}")

# -----------------------------------------------------------------------------
# RANKINGS TABLE
# -----------------------------------------------------------------------------

display = filtered.copy()
display.index = range(1, len(display) + 1)
display.index.name = "Rank"

# Rename for display
display = display.rename(columns={
    "player": "Player",
    "team": "Team",
    "n_batted_balls": "Batted Balls",
    "posterior_mean": "Est. Bases (Bayesian)",
    "raw_rate": "Est. Bases (Raw)",
    "hdi_low": "HDI Low",
    "hdi_high": "HDI High",
    "shrinkage": "Shrinkage",
})

COLUMN_CONFIG = {
    "Est. Bases (Bayesian)": st.column_config.NumberColumn(format="%.4f"),
    "Est. Bases (Raw)": st.column_config.NumberColumn(format="%.4f"),
    "HDI Low": st.column_config.NumberColumn(format="%.4f"),
    "HDI High": st.column_config.NumberColumn(format="%.4f"),
    "Shrinkage": st.column_config.NumberColumn(format="%+.4f"),
    "Batted Balls": st.column_config.NumberColumn(format="%d"),
}

st.dataframe(
    display,
    use_container_width=True,
    column_config=COLUMN_CONFIG,
)

# -----------------------------------------------------------------------------
# SHRINKAGE CHART
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Shrinkage Visualization")
st.caption(
    "Shows how the Bayesian model adjusts raw observed rates. "
    "Players with fewer batted balls are pulled more toward the league average."
)

shrinkage_name = f"top_{type_key}s_shrinkage"
shrinkage_url = get_player_evaluation_image_url(season, shrinkage_name)

try:
    st.image(shrinkage_url, use_container_width=True)
except Exception:
    st.info("Shrinkage chart not available.")

# -----------------------------------------------------------------------------
# METHODOLOGY
# -----------------------------------------------------------------------------

st.divider()
with st.expander("Methodology"):
    st.markdown("""
    **Bayesian Hierarchical Model**

    Each player's "true" contact quality is estimated using a Bayesian hierarchical
    model with partial pooling. Players with few batted balls are shrunk toward the
    league average; players with many batted balls retain their observed rate.

    **Model Details:**
    - **Metric**: Estimated bases per batted ball (model-predicted expected bases
      based on exit velocity, launch angle, spray angle, and ballpark factors)
    - **Non-centered parameterization** to avoid sampling issues with small samples
    - **89% Highest Density Interval (HDI)** credible intervals (ArviZ convention)
    - Population-level parameters estimated: league mean, between-player SD, within-player SD

    **Column Definitions:**
    - **Est. Bases (Bayesian)**: Posterior mean — the model's best estimate of true contact quality
    - **Est. Bases (Raw)**: Simple observed mean estimated bases per batted ball
    - **HDI Low / HDI High**: 89% credible interval bounds
    - **Shrinkage**: Difference between Bayesian and raw estimates (negative = shrunk downward)
    - **Batted Balls**: Number of balls put in play (excluding strikeouts, walks)

    **Interpretation:**
    - For **hitters**: higher = better quality contact
    - For **pitchers**: lower = better at suppressing quality contact
    """)
