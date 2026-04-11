"""
Team Rankings Page

Shows season-long luck metrics: which teams have over/under-performed
their expected wins based on the simulation model.

Charts are pre-rendered by the simulator pipeline and served as static
images from S3, eliminating the need for matplotlib/pillow/requests.
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

from utils.data_loader import load_game_summaries, _image_exists
from utils.luck_calculations import (
    filter_to_regular_season,
    calculate_luck_metrics,
    calculate_model_accuracy,
    get_extreme_teams,
)
from utils.responsive import inject_responsive_css, render_home_link

inject_responsive_css()

# S3 base URL for pre-rendered charts
S3_CHARTS_URL = "https://dtw-streamlit.s3.amazonaws.com/team-rankings"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_data_freshness_date(df: pd.DataFrame) -> str:
    """
    Get the most recent game date from the data for display purposes.

    Returns:
        Formatted date string like "Jan 15, 2025"
    """
    if df.empty:
        return "N/A"
    latest_date = df['date'].max()
    return latest_date.strftime("%b %d, %Y")


# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

st.title("Team Luck Rankings")
st.markdown("Which teams have been **lucky** (more wins than expected) or **unlucky** (fewer wins than expected)?")

# Load data
df = load_game_summaries()

if df.empty:
    st.error("No game data available. Please check the data source.")
    st.stop()

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

st.divider()

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Season filter - for now just 2025
    available_seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.selectbox(
        "Season",
        options=available_seasons,
        index=0  # Default to most recent
    )

with col2:
    # Regular season toggle
    regular_season_only = st.checkbox(
        "Regular Season Only",
        value=True,
        help="Exclude playoff games (playoffs started Sept 30, 2025)"
    )

# Apply filters
filtered_df = df[df['season'] == selected_season].copy()

if regular_season_only:
    filtered_df = filter_to_regular_season(filtered_df, selected_season)

# Check if we have data after filtering
if filtered_df.empty:
    st.warning(f"No games found for {selected_season}.")
    st.stop()

# Get data freshness date for chart labels
data_date = get_data_freshness_date(filtered_df)

# -----------------------------------------------------------------------------
# CALCULATE METRICS
# -----------------------------------------------------------------------------

luck_stats = calculate_luck_metrics(filtered_df)
model_accuracy = calculate_model_accuracy(filtered_df)
extreme_teams = get_extreme_teams(luck_stats)

# -----------------------------------------------------------------------------
# QUICK STATS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Quick Stats")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric(
        label="Games Analyzed",
        value=f"{model_accuracy['total']:,}"
    )

with stat_col2:
    st.metric(
        label="Model Accuracy",
        value=f"{model_accuracy['accuracy']:.1%}",
        help="How often the team with higher win probability actually won"
    )

with stat_col3:
    luckiest = extreme_teams['luckiest']
    st.metric(
        label="Luckiest Team",
        value=luckiest['team'],
        delta=f"+{luckiest['differential']:.1f} wins"
    )

with stat_col4:
    unluckiest = extreme_teams['unluckiest']
    st.metric(
        label="Unluckiest Team",
        value=unluckiest['team'],
        delta=f"{unluckiest['differential']:.1f} wins"
    )

# -----------------------------------------------------------------------------
# CHARTS (pre-rendered from S3)
# -----------------------------------------------------------------------------

# Determine chart suffix based on regular season toggle
chart_suffix = "" if regular_season_only else "_full"

st.divider()
st.subheader("Luck Overview")

luck_tab_net, luck_tab_wp = st.tabs(["Net Lucky Wins", "Wins - xWins"])

with luck_tab_net:
    st.caption("Net lucky wins = lucky wins minus unlucky losses. Teams on the right have more luck-driven wins.")
    net_lucky_url = f"{S3_CHARTS_URL}/{selected_season}/net_lucky_wins{chart_suffix}.png"
    if _image_exists(net_lucky_url):
        st.image(net_lucky_url, width="stretch")
    else:
        st.info("Net Lucky Wins chart not yet available. Check back when games are being played.")

with luck_tab_wp:
    st.caption("Teams on the right have won more games than expected; teams on the left have won fewer.")
    luck_diff_url = f"{S3_CHARTS_URL}/{selected_season}/luck_differential{chart_suffix}.png"
    if _image_exists(luck_diff_url):
        st.image(luck_diff_url, width="stretch")
    else:
        st.info("Charts are generated during the MLB season. Check back when games are being played.")

st.divider()
st.subheader("Actual Wins vs Expected Wins")
st.caption("Teams above the diagonal have won more than expected (lucky); teams below have won fewer (unlucky).")

wins_xwins_url = f"{S3_CHARTS_URL}/{selected_season}/wins_vs_xwins{chart_suffix}.png"
if _image_exists(wins_xwins_url):
    st.image(wins_xwins_url, width="stretch")
else:
    st.info("Charts are generated during the MLB season. Check back when games are being played.")

# -----------------------------------------------------------------------------
# DATA TABLE
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Full Rankings Table")
st.caption("Some games may be excluded due to incomplete tracking data or unsupported ballparks.")

# Prepare display dataframe
display_df = luck_stats[[
    'team', 'games_played', 'actual_wins', 'expected_wins',
    'luck_differential', 'lucky_wins', 'unlucky_losses', 'win_pct', 'expected_win_pct'
]].copy()

# Rename columns for display
display_df.columns = [
    'Team', 'Games', 'Actual Wins', 'Expected Wins',
    'Luck Diff', 'Lucky Wins', 'Unlucky Losses', 'Win %', 'Expected Win %'
]

# Format percentages
display_df['Win %'] = (display_df['Win %'] * 100).round(1).astype(str) + '%'
display_df['Expected Win %'] = (display_df['Expected Win %'] * 100).round(1).astype(str) + '%'

# Display table
st.dataframe(
    display_df,
    hide_index=True,
    width="stretch",
    column_config={
        'Luck Diff': st.column_config.NumberColumn(format="%.1f"),
    }
)

# Download button
csv_data = display_df.to_csv(index=False)
st.download_button(
    label="Download as CSV",
    data=csv_data,
    file_name=f"team_rankings_{selected_season}.csv",
    mime="text/csv"
)

# -----------------------------------------------------------------------------
# METHODOLOGY NOTE
# -----------------------------------------------------------------------------

st.divider()
with st.expander("Methodology"):
    st.markdown("""
**How luck is measured:**

Think of a team that wins a bunch of one-run games on bloop singles and errors. They've
got the same record as a team crushing the ball every night. Are they equally good? Probably not.

We measure the gap between what a team's plate appearances *should* have produced and what
actually happened:

1. **Expected Wins** — for each game, the simulator estimates a win probability based on
   batted ball quality (exit velocity, launch angle, spray angle), walks, strikeouts, and
   baserunning. Sum those probabilities across the season and you get expected wins.
2. **Luck Differential** — actual wins minus expected wins. Positive means the team has
   won more than their underlying performance suggests.
3. **Lucky Win** — won despite having less than a 50% deserve-to-win probability.
4. **Unlucky Loss** — lost despite having more than 50%.

This measures luck relative to *on-field performance quality*, not overall team talent. A
team can be genuinely good and still lucky — it just means they've won even more than their
already-strong play would predict.
    """)

render_home_link()
