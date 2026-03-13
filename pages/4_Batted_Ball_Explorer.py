"""
Batted Ball Explorer Page

Interactive explorer for individual batted balls — search by player,
exit velocity, launch angle, spray direction, and more.
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

from utils.data_loader import load_batted_balls, get_available_batted_ball_seasons
from utils.responsive import inject_responsive_css, render_home_link

# Page config
st.set_page_config(
    page_title="Batted Ball Explorer | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)

inject_responsive_css()

MAX_DISPLAY_ROWS = 500

# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mlb_simulator_logo.png")
st.logo(_logo_path)
st.title("Batted Ball Explorer")
st.markdown("Search and filter individual batted balls to see model-predicted outcome probabilities.")

# Season selector
col_season, _ = st.columns([1, 3])
with col_season:
    available_seasons = get_available_batted_ball_seasons()
    if available_seasons:
        season = st.selectbox("Season", options=available_seasons, index=0)
    else:
        season = pd.Timestamp.now().year

# Load data
df = load_batted_balls(season)

if df.empty:
    st.info(f"No batted ball data available for {season}. Data is collected as games are processed during the season.")
    st.stop()

# -----------------------------------------------------------------------------
# SUMMARY METRICS
# -----------------------------------------------------------------------------

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Batted Balls", f"{len(df):,}")
m2.metric("Games", f"{df['gamePk'].nunique():,}")
m3.metric("Players", f"{df['player'].nunique():,}")
m4.metric("Avg xBA", f"{df['xba'].mean():.3f}")

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Search & Filter")

# Row 1: Player, Team, Result
f1, f2, f3 = st.columns(3)

with f1:
    player_search = st.text_input("Player Name", placeholder="e.g. Ohtani")

with f2:
    teams = sorted(df['team'].dropna().unique())
    team_filter = st.selectbox("Team", options=["All"] + teams)

with f3:
    results = sorted(df['actual_result'].dropna().unique())
    result_filter = st.selectbox("Result", options=["All"] + results)

# Row 2: Exit Velo, Launch Angle, Estimated Bases sliders
f4, f5, f6 = st.columns(3)

ev_min, ev_max = float(df['launch_speed'].min()), float(df['launch_speed'].max())
la_min, la_max = float(df['launch_angle'].min()), float(df['launch_angle'].max())
eb_min, eb_max = float(df['estimated_bases'].min()), float(df['estimated_bases'].max())

with f4:
    ev_range = st.slider(
        "Exit Velocity (mph)",
        min_value=ev_min, max_value=ev_max,
        value=(ev_min, ev_max), step=0.5
    )

with f5:
    la_range = st.slider(
        "Launch Angle (°)",
        min_value=la_min, max_value=la_max,
        value=(la_min, la_max), step=1.0
    )

with f6:
    eb_range = st.slider(
        "Estimated Bases",
        min_value=eb_min, max_value=eb_max,
        value=(eb_min, eb_max), step=0.05
    )

# Row 3: Date Range, Spray Direction, Sort By
f7, f8, f9 = st.columns(3)

with f7:
    available_dates = sorted(df['date_parsed'].dropna().dt.date.unique())
    date_range = st.date_input(
        "Date Range",
        value=(available_dates[0], available_dates[-1]),
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )

with f8:
    spray_options = ["All"] + [d for d in ["Pull", "Center", "Oppo"] if d in df['spray_direction'].values]
    spray_filter = st.selectbox("Spray Direction", options=spray_options)

with f9:
    sort_options = {
        "Estimated Bases (High → Low)": ("estimated_bases", False),
        "Estimated Bases (Low → High)": ("estimated_bases", True),
        "Exit Velocity (High → Low)": ("launch_speed", False),
        "Exit Velocity (Low → High)": ("launch_speed", True),
        "Launch Angle (High → Low)": ("launch_angle", False),
        "xBA (High → Low)": ("xba", False),
        "Date (Recent First)": ("date_parsed", False),
    }
    sort_label = st.selectbox("Sort By", options=list(sort_options.keys()))

# -----------------------------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------------------------

filtered = df.copy()

if player_search:
    filtered = filtered[filtered['player'].str.contains(player_search, case=False, na=False)]

if team_filter != "All":
    filtered = filtered[filtered['team'] == team_filter]

if result_filter != "All":
    filtered = filtered[filtered['actual_result'] == result_filter]

filtered = filtered[
    (filtered['launch_speed'] >= ev_range[0]) & (filtered['launch_speed'] <= ev_range[1])
]
filtered = filtered[
    (filtered['launch_angle'] >= la_range[0]) & (filtered['launch_angle'] <= la_range[1])
]
filtered = filtered[
    (filtered['estimated_bases'] >= eb_range[0]) & (filtered['estimated_bases'] <= eb_range[1])
]

if spray_filter != "All":
    filtered = filtered[filtered['spray_direction'] == spray_filter]

if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered['date_parsed'].dt.date >= date_range[0]) &
        (filtered['date_parsed'].dt.date <= date_range[1])
    ]

# Sort
sort_col, sort_asc = sort_options[sort_label]
filtered = filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# -----------------------------------------------------------------------------
# RESULTS TABLE
# -----------------------------------------------------------------------------

st.divider()

total_results = len(filtered)
if total_results == 0:
    st.warning("No batted balls match your filters. Try broadening your search.")
    st.stop()

st.markdown(f"**{total_results:,} batted balls found**" +
            (f" (showing first {MAX_DISPLAY_ROWS:,})" if total_results > MAX_DISPLAY_ROWS else ""))

# Prepare display DataFrame
source_cols = [
    'team', 'player', 'launch_speed', 'launch_angle', 'spray_direction',
    'actual_result', 'estimated_bases', 'xba', 'hr_prob', 'date', 'opponent', 'venue'
]
# Add video link if play_id column exists
has_play_id = 'play_id' in filtered.columns
if has_play_id:
    source_cols.append('play_id')

available_cols = [c for c in source_cols if c in filtered.columns]
display = filtered.head(MAX_DISPLAY_ROWS)[available_cols].copy()

display['hr_prob'] = display['hr_prob'] * 100

# Build video URLs from play_id
if has_play_id:
    display['video'] = display['play_id'].apply(
        lambda pid: f"https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
        if pd.notna(pid) and pid != "" else None
    )
    display = display.drop(columns=['play_id'])

rename_map = {
    'team': 'Team', 'player': 'Player', 'launch_speed': 'Exit Velo',
    'launch_angle': 'Launch Angle', 'spray_direction': 'Spray',
    'actual_result': 'Result', 'estimated_bases': 'Est. Bases',
    'xba': 'xBA', 'hr_prob': 'HR%', 'date': 'Date',
    'opponent': 'Opponent', 'venue': 'Stadium', 'video': 'Video',
}
display = display.rename(columns=rename_map)

col_config = {
    'Exit Velo': st.column_config.NumberColumn(format="%.1f mph"),
    'Launch Angle': st.column_config.NumberColumn(format="%d°"),
    'Est. Bases': st.column_config.NumberColumn(format="%.2f"),
    'xBA': st.column_config.NumberColumn(format="%.3f"),
    'HR%': st.column_config.NumberColumn(format="%.2f%%"),
}
if 'Video' in display.columns:
    col_config['Video'] = st.column_config.LinkColumn(display_text="Watch")

st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    column_config=col_config,
)

# Download button
csv_data = filtered.to_csv(index=False)
st.download_button(
    label="📥 Download as CSV",
    data=csv_data,
    file_name=f"batted_balls_{season}.csv",
    mime="text/csv"
)

# -----------------------------------------------------------------------------
# PLAYER LEADERBOARD
# -----------------------------------------------------------------------------

st.divider()
st.subheader("📊 Player Leaderboard")
st.caption("Aggregated stats for players in your filtered results (min. 10 batted balls).")

if total_results > 0:
    agg = filtered.groupby(['player', 'team']).agg(
        batted_balls=('estimated_bases', 'count'),
        avg_ev=('launch_speed', 'mean'),
        max_ev=('launch_speed', 'max'),
        avg_la=('launch_angle', 'mean'),
        avg_est_bases=('estimated_bases', 'mean'),
        total_est_bases=('estimated_bases', 'sum'),
        avg_xba=('xba', 'mean'),
    ).reset_index()

    # Minimum 10 batted balls
    agg = agg[agg['batted_balls'] >= 10]

    if agg.empty:
        st.info("Not enough data for leaderboard (need at least 10 batted balls per player).")
    else:
        agg = agg.sort_values('total_est_bases', ascending=False).reset_index(drop=True)

        leader_display = agg.copy()
        leader_display.columns = [
            'Player', 'Team', 'BBs', 'Avg EV', 'Max EV', 'Avg LA',
            'Avg Est. Bases', 'Total Est. Bases', 'Avg xBA'
        ]

        # Add dashboard link
        import urllib.parse
        leader_display['Profile'] = leader_display['Player'].apply(
            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
        )

        st.dataframe(
            leader_display,
            hide_index=True,
            width="stretch",
            column_config={
                'Avg EV': st.column_config.NumberColumn(format="%.1f"),
                'Max EV': st.column_config.NumberColumn(format="%.1f"),
                'Avg LA': st.column_config.NumberColumn(format="%.1f°"),
                'Avg Est. Bases': st.column_config.NumberColumn(format="%.2f"),
                'Total Est. Bases': st.column_config.NumberColumn(format="%.1f"),
                'Avg xBA': st.column_config.NumberColumn(format="%.3f"),
                'Profile': st.column_config.LinkColumn(display_text="View"),
            }
        )

# -----------------------------------------------------------------------------
# METHODOLOGY
# -----------------------------------------------------------------------------

st.divider()
with st.expander("📖 Methodology"):
    st.markdown("""
    **Column Definitions:**

    - **Exit Velocity (EV)**: Speed of the ball off the bat in mph (from Statcast)
    - **Launch Angle (LA)**: Vertical angle of the ball off the bat in degrees (from Statcast)
    - **Spray Direction**: Pull, Center, or Oppo — computed from hit coordinates and batter handedness
    - **Estimated Bases**: Model-predicted expected bases for this batted ball
      (P(1B)×1 + P(2B)×2 + P(3B)×3 + P(HR)×4)
    - **xBA**: Expected batting average (1 - out probability) — the model's estimate of
      how often this batted ball would result in a hit
    - **HR%**: Model-predicted probability of a home run

    **How the model works:**

    A Gradient Boosting Classifier trained on historical Statcast data predicts the
    probability of each outcome (out, single, double, triple, home run) given the
    exit velocity, launch angle, spray angle, and ballpark. The model accounts for
    park-specific dimensions and effects.
    """)

render_home_link()
