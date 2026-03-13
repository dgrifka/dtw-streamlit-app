"""
Player Evaluations Page

Bayesian hierarchical rankings of MLB hitters and pitchers by estimated
bases per batted ball, with credible intervals and shrinkage for small
sample sizes.
"""

import unicodedata

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
    load_player_evaluations_pa,
    get_player_evaluation_image_url,
    get_player_evaluation_team_image_url,
    get_available_player_evaluation_seasons,
)
from utils.team_mappings import TEAM_COLORS
from utils.responsive import inject_responsive_css, render_home_link

# Page config
st.set_page_config(
    page_title="Player Rankings | DTW Simulator",
    page_icon="⚾",
    layout="wide",
)

inject_responsive_css()

# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mlb_simulator_logo.png")
st.logo(_logo_path)
st.title("Player Rankings")
st.markdown(
    "Bayesian rankings of hitters and pitchers with credible intervals. "
    "The model estimates each player's true underlying performance by pooling "
    "information across all players — players with fewer observations are pulled "
    "toward the league average, while players with more data retain estimates "
    "closer to their observed performance."
)

# Season selector
col_season, col_type, col_metric, col_team = st.columns([1, 1, 1, 1])
with col_season:
    available_seasons = get_available_player_evaluation_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0)
    else:
        season = pd.Timestamp.now().year

with col_type:
    player_type = st.selectbox("Player Type", options=["Hitter", "Pitcher"])

with col_metric:
    metric_mode = st.selectbox("Metric", options=["Per Plate Appearance", "Per Batted Ball"])

with col_team:
    all_teams = sorted(TEAM_COLORS.keys())
    selected_team = st.selectbox("Team", options=["All Teams"] + all_teams)

type_key = player_type.lower()
is_pa_mode = metric_mode == "Per Plate Appearance"

# Load data
if is_pa_mode:
    df = load_player_evaluations_pa(season, type_key)
else:
    df = load_player_evaluations(season, type_key)

if df.empty:
    st.info(
        f"No {type_key} evaluation data available for {season}. "
        f"Data is generated daily during the season."
    )
    st.stop()

# -----------------------------------------------------------------------------
# METHODOLOGY (above chart so readers understand what they're looking at)
# -----------------------------------------------------------------------------

st.divider()

with st.expander("How does this work?"):
    if is_pa_mode:
        st.markdown("""
**Per Plate Appearance mode** measures overall offensive production, not just contact quality.
Each plate appearance outcome is valued: walks = 1 base, HBP = 1 base, strikeouts = 0 bases,
and batted balls use the model's estimated bases (based on exit velocity, launch angle, and spray angle).

**What is a hierarchical model?**

Instead of treating each player independently, a hierarchical model learns a "league-wide"
baseline and estimates how much each player deviates from it. This is called **partial pooling** —
every player's estimate is informed by both their own data and the overall population.

**Reading the chart:**

- **Circle**: the model's best estimate of a player's true production (posterior mean)
- **Thick line**: 50% credible interval — there's a 50% chance the true value falls in this range
- **Thin line**: 89% credible interval — a wider range capturing more uncertainty

**Column definitions:**

- **Est. Bases (Bayesian)**: Posterior mean — the model's best estimate of true production per PA
- **Est. Bases (Raw)**: Simple observed mean estimated bases per plate appearance
- **Shrinkage**: Difference between Bayesian and raw estimates (negative = shrunk downward)
""")
    else:
        st.markdown("""
**Per Batted Ball mode** measures contact quality only — how well a player hits the ball when
they put it in play, based on exit velocity, launch angle, and spray angle.

**What is a hierarchical model?**

Instead of treating each player independently, a hierarchical model learns a "league-wide"
baseline and estimates how much each player deviates from it. This is called **partial pooling** —
every player's estimate is informed by both their own data and the overall population.

**Why does this matter?**

A player with 20 batted balls and a high raw average might just be on a hot streak. The model
recognizes the small sample and **pulls ("shrinks") their estimate toward the league average**.
A player with 400+ batted balls keeps an estimate much closer to their raw observed rate,
because there's enough data to trust it.

**Reading the chart:**

- **Circle**: the model's best estimate of a player's true contact quality (posterior mean)
- **Thick line**: 50% credible interval — there's a 50% chance the true value falls in this range
- **Thin line**: 89% credible interval — a wider range capturing more uncertainty
- Players with fewer batted balls have wider intervals, reflecting greater uncertainty

**Column definitions:**

- **Est. Bases (Bayesian)**: Posterior mean — the model's best estimate of true contact quality
- **Est. Bases (Raw)**: Simple observed mean estimated bases per batted ball
- **Shrinkage**: Difference between Bayesian and raw estimates (negative = shrunk downward)
""")

# -----------------------------------------------------------------------------
# CHART IMAGE FROM S3
# -----------------------------------------------------------------------------

chart_name = f"top_{type_key}s_pa" if is_pa_mode else f"top_{type_key}s"

if selected_team != "All Teams":
    chart_url = get_player_evaluation_team_image_url(season, selected_team, chart_name)
else:
    chart_url = get_player_evaluation_image_url(season, chart_name)

try:
    st.image(chart_url, width="stretch")
except Exception:
    if selected_team != "All Teams":
        # Fall back to all-teams chart if team chart unavailable
        fallback_url = get_player_evaluation_image_url(season, chart_name)
        try:
            st.image(fallback_url, width="stretch")
        except Exception:
            st.warning(f"Chart image not available for {season}.")
    else:
        st.warning(f"Chart image not available for {season}.")

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

st.divider()
st.subheader(f"{player_type} Rankings Table")


def _normalize(text):
    """Strip accents and lowercase for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


col_search, col_min_bb, _ = st.columns([1, 1, 2])

with col_search:
    search_query = st.text_input(
        "Search Player",
        placeholder="e.g. Ohtani, Juan, Suarez",
    )

with col_min_bb:
    count_label = "Plate Appearances" if is_pa_mode else "Batted Balls"
    default_min = 100 if is_pa_mode else 30
    min_bb = st.slider(
        f"Min {count_label}",
        min_value=1,
        max_value=int(df["n_batted_balls"].max()),
        value=min(default_min, int(df["n_batted_balls"].max())),
        step=10,
    )

# Apply filters
filtered = df[df["n_batted_balls"] >= min_bb].copy()
if selected_team != "All Teams":
    filtered = filtered[filtered["team"] == selected_team]
if search_query.strip():
    query_norm = _normalize(search_query.strip())
    filtered = filtered[
        filtered["player"].apply(lambda name: query_norm in _normalize(name))
    ]

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
m4.metric(f"Avg {count_label}", f"{filtered['n_batted_balls'].mean():.0f}")

# -----------------------------------------------------------------------------
# RANKINGS TABLE
# -----------------------------------------------------------------------------

display = filtered.copy()

# Drop chart-only columns before display
for col in ["hdi_50_low", "hdi_50_high"]:
    if col in display.columns:
        display = display.drop(columns=[col])

display.index = range(1, len(display) + 1)
display.index.name = "Rank"

# Rename for display
n_col_name = "Plate Appearances" if is_pa_mode else "Batted Balls"
display = display.rename(columns={
    "player": "Player",
    "team": "Team",
    "n_batted_balls": n_col_name,
    "posterior_mean": "Est. Bases (Bayesian)",
    "raw_rate": "Est. Bases (Raw)",
    "hdi_low": "HDI Low",
    "hdi_high": "HDI High",
    "shrinkage": "Shrinkage",
})

# Add player dashboard link
import urllib.parse
_profile_page = "Pitcher_Profile" if type_key == "pitcher" else "Hitter_Profile"
display["Profile"] = display["Player"].apply(
    lambda p: f"/{_profile_page}?player={urllib.parse.quote(p)}"
)

COLUMN_CONFIG = {
    "Est. Bases (Bayesian)": st.column_config.NumberColumn(format="%.4f"),
    "Est. Bases (Raw)": st.column_config.NumberColumn(format="%.4f"),
    "HDI Low": st.column_config.NumberColumn(format="%.4f"),
    "HDI High": st.column_config.NumberColumn(format="%.4f"),
    "Shrinkage": st.column_config.NumberColumn(format="%+.4f"),
    n_col_name: st.column_config.NumberColumn(format="%d"),
    "Profile": st.column_config.LinkColumn(display_text="View"),
}

st.dataframe(
    display,
    width="stretch",
    column_config=COLUMN_CONFIG,
)

render_home_link()

