"""
Daily Highlights Page

Surfaces the best and worst batted ball outcomes for each game day —
top estimated bases, unluckiest outs, and luckiest hits.
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
    page_title="Daily Highlights | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)

inject_responsive_css()

MAX_DISPLAY_ROWS = 15

# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mlb_simulator_logo.png")
st.logo(_logo_path)
st.title("Daily Highlights")
st.markdown("Best and worst batted ball outcomes for each game day.")

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

# Ensure date_parsed exists
if "date_parsed" not in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

# Get available dates (most recent first)
available_dates = sorted(df["date_parsed"].dropna().dt.date.unique(), reverse=True)

if not available_dates:
    st.info("No dates with batted ball data found.")
    st.stop()

# Date selector
col_date, col_spacer = st.columns([1, 3])
with col_date:
    selected_date = st.date_input(
        "Game Date",
        value=available_dates[0],
        min_value=available_dates[-1],
        max_value=available_dates[0],
    )

# Filter to selected date
day_df = df[df["date_parsed"].dt.date == selected_date].copy()

if day_df.empty:
    st.warning(f"No batted ball data for {selected_date.strftime('%B %d, %Y')}.")
    st.stop()

# Only include batted balls (with launch data)
day_df = day_df.dropna(subset=["launch_speed", "launch_angle"])

# -----------------------------------------------------------------------------
# DAY SUMMARY METRICS
# -----------------------------------------------------------------------------

st.divider()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Batted Balls", f"{len(day_df):,}")
m2.metric("Games", f"{day_df['gamePk'].nunique()}")
m3.metric("Avg Exit Velo", f"{day_df['launch_speed'].mean():.1f} mph")
m4.metric("Avg xBA", f"{day_df['xba'].mean():.3f}")

# Barrel rate: EV >= 95 and 25 <= LA <= 35
barrels = day_df[
    (day_df["launch_speed"] >= 95) &
    (day_df["launch_angle"] >= 25) &
    (day_df["launch_angle"] <= 35)
]
barrel_rate = len(barrels) / len(day_df) * 100 if len(day_df) > 0 else 0
m5.metric("Barrel Rate", f"{barrel_rate:.1f}%")

# Column config shared across tables
COLUMN_CONFIG = {
    "Exit Velo": st.column_config.NumberColumn(format="%.1f mph"),
    "Launch Angle": st.column_config.NumberColumn(format="%d°"),
    "Est. Bases": st.column_config.NumberColumn(format="%.2f"),
    "xBA": st.column_config.NumberColumn(format="%.3f"),
    "HR%": st.column_config.NumberColumn(format="%.2f%%"),
    "Out%": st.column_config.NumberColumn(format="%.2f%%"),
    "Video": st.column_config.LinkColumn(display_text="Watch"),
}


def format_table(data, cols):
    """Select and rename columns for display."""
    source_cols = [
        "player", "team", "launch_speed", "launch_angle", "spray_direction",
        "actual_result", "estimated_bases", "xba", "hr_prob", "out_prob", "opponent",
    ]
    # Include play_id if available (for video links)
    if "play_id" in data.columns:
        source_cols.append("play_id")
    available = [c for c in source_cols if c in data.columns]
    display = data.head(MAX_DISPLAY_ROWS)[available].copy()
    # Convert probabilities to percentage scale
    for col in ['hr_prob', 'out_prob']:
        if col in display.columns:
            display[col] = display[col] * 100
    # Build video URLs from play_id
    if "play_id" in display.columns:
        display["video"] = display["play_id"].apply(
            lambda pid: f"https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
            if pd.notna(pid) and pid != "" else None
        )
        display = display.drop(columns=["play_id"])
    rename_map = {
        "player": "Player",
        "team": "Team",
        "launch_speed": "Exit Velo",
        "launch_angle": "Launch Angle",
        "spray_direction": "Spray",
        "actual_result": "Result",
        "estimated_bases": "Est. Bases",
        "xba": "xBA",
        "hr_prob": "HR%",
        "out_prob": "Out%",
        "opponent": "Opponent",
        "video": "Video",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
    # Only keep requested columns that exist
    final_cols = [c for c in cols if c in display.columns]
    return display[final_cols]


# -----------------------------------------------------------------------------
# TOP ESTIMATED BASES
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Top Estimated Bases")
st.caption("Highest model-predicted expected bases — the day's best batted balls by outcome probability.")

top_eb = day_df.sort_values("estimated_bases", ascending=False)
display_cols = ["Player", "Team", "Exit Velo", "Launch Angle", "Spray",
                "Result", "Est. Bases", "xBA", "HR%", "Opponent", "Video"]
st.dataframe(
    format_table(top_eb, display_cols),
    hide_index=True,
    width="stretch",
    column_config=COLUMN_CONFIG,
)

# -----------------------------------------------------------------------------
# UNLUCKIEST OUTS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Unluckiest Outs")
st.caption("Outs with the highest expected batting average — balls that *should* have been hits.")

outs_df = day_df[day_df["actual_result"] == "Out"].copy()
if outs_df.empty:
    st.info("No outs recorded on this date.")
else:
    unlucky = outs_df.sort_values("xba", ascending=False)
    unlucky_cols = ["Player", "Team", "Exit Velo", "Launch Angle", "Spray",
                    "Est. Bases", "xBA", "HR%", "Opponent", "Video"]
    st.dataframe(
        format_table(unlucky, unlucky_cols),
        hide_index=True,
        width="stretch",
        column_config=COLUMN_CONFIG,
    )

# -----------------------------------------------------------------------------
# LUCKIEST HITS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Luckiest Hits")
st.caption("Hits with the lowest expected batting average — balls that probably *shouldn't* have been hits.")

hit_types = ["Single", "Double", "Triple", "Home Run"]
hits_df = day_df[day_df["actual_result"].isin(hit_types)].copy()
if hits_df.empty:
    st.info("No hits recorded on this date.")
else:
    lucky = hits_df.sort_values("xba", ascending=True)
    lucky_cols = ["Player", "Team", "Exit Velo", "Launch Angle", "Spray",
                  "Result", "Est. Bases", "xBA", "Out%", "Opponent", "Video"]
    st.dataframe(
        format_table(lucky, lucky_cols),
        hide_index=True,
        width="stretch",
        column_config=COLUMN_CONFIG,
    )

# -----------------------------------------------------------------------------
# METHODOLOGY
# -----------------------------------------------------------------------------

st.divider()
with st.expander("Methodology"):
    st.markdown("""
    **Column Definitions:**

    - **Exit Velocity (EV)**: Speed of the ball off the bat in mph
    - **Launch Angle (LA)**: Vertical angle of the ball off the bat in degrees
    - **Est. Bases**: Model-predicted expected bases (P(1B)×1 + P(2B)×2 + P(3B)×3 + P(HR)×4)
    - **xBA**: Expected batting average (1 - out probability)
    - **HR%**: Model-predicted home run probability
    - **Out%**: Model-predicted out probability
    - **Barrel Rate**: Percentage of batted balls with EV ≥ 95 mph and LA between 25° and 35°

    **How highlights are selected:**

    - **Top Estimated Bases**: Sorted by expected bases, highest first
    - **Unluckiest Outs**: Outs sorted by xBA, highest first (highest xBA = most unlucky)
    - **Luckiest Hits**: Hits sorted by xBA, lowest first (lowest xBA = most lucky)
    """)

render_home_link()
