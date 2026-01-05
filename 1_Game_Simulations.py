"""
Game Simulations Page
Browse, filter, and view game simulation results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the app root directory to Python path (parent of pages/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import utilities
from utils.data_loader import (
    load_game_summaries,
    get_game_images,
    get_deserved_winner,
    filter_games,
)
from utils.team_mappings import get_all_teams, get_short_name

st.set_page_config(
    page_title="Game Simulations | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .game-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .upset-badge {
        background-color: #ffc107;
        color: #000;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .deserved-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def format_wp_display(away_wp: float, home_wp: float, tie_wp: float) -> str:
    """Format win probabilities for display."""
    return f"{away_wp*100:.0f}% - {home_wp*100:.0f}% (Tie: {tie_wp*100:.0f}%)"


def display_game_card(row: pd.Series, expanded: bool = False):
    """Display a single game as an expandable card."""
    
    # Get deserved winner info
    winner_info = get_deserved_winner(row)
    
    # Build title
    date_str = row['date'].strftime("%m/%d/%Y")
    away_short = get_short_name(row['away'])
    home_short = get_short_name(row['home'])
    
    title = f"**{away_short}** {row['away_score']} @ **{home_short}** {row['home_score']} — {date_str}"
    
    # Add badges
    badges = []
    if winner_info['was_upset']:
        badges.append("🎲 Upset")
    
    if badges:
        title += f"  {'  '.join(badges)}"
    
    with st.expander(title, expanded=expanded):
        # Summary row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Deserve-to-Win",
                f"{get_short_name(winner_info['deserved_winner'])} ({winner_info['deserved_prob']*100:.0f}%)"
            )
        
        with col2:
            st.metric(
                "Actual Winner",
                winner_info['actual_winner']
            )
        
        with col3:
            wp_str = format_wp_display(row['away_wp'], row['home_wp'], row['tie_wp'])
            st.metric(f"{away_short} vs {home_short}", wp_str)
        
        st.divider()
        
        # Visualizations
        st.subheader("Visualizations")
        images = get_game_images(row)
        
        # Display in 2x2 grid
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.image(images['spray'], caption="Spray Chart", use_container_width=True)
            st.image(images['estimated_bases'], caption="Expected Bases by Player", use_container_width=True)
        
        with img_col2:
            st.image(images['rd'], caption="Run Distribution", use_container_width=True)
            st.image(images['player_contributions'], caption="Player Contributions", use_container_width=True)


def main():
    st.title("⚾ Game Simulations")
    st.markdown("Browse and filter simulation results. Click any game to see visualizations.")
    
    # Load data
    with st.spinner("Loading game data..."):
        df = load_game_summaries()
    
    if df.empty:
        st.error("Unable to load game data. Please try again later.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Team filter
    all_teams = get_all_teams()
    selected_teams = st.sidebar.multiselect(
        "Teams",
        options=all_teams,
        default=[],
        help="Filter games where any selected team played (home or away)"
    )
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(max_date - timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter games within this date range"
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    
    # Season filter
    seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.sidebar.selectbox(
        "Season",
        options=[None] + list(seasons),
        format_func=lambda x: "All Seasons" if x is None else str(x)
    )
    
    # Upsets only toggle
    upsets_only = st.sidebar.checkbox(
        "🎲 Upsets Only",
        value=False,
        help="Show only games where the actual winner was not the 'deserved' winner"
    )
    
    st.sidebar.divider()
    
    # Sort options
    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=["Date (Newest)", "Date (Oldest)", "Biggest Upset", "Closest Simulation"],
        index=0
    )
    
    # Apply filters
    filtered_df = filter_games(
        df,
        teams=selected_teams if selected_teams else None,
        start_date=start_date,
        end_date=end_date,
        season=selected_season,
        upsets_only=upsets_only
    )
    
    # Apply sorting
    if sort_by == "Date (Newest)":
        filtered_df = filtered_df.sort_values('date', ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered_df = filtered_df.sort_values('date', ascending=True)
    elif sort_by == "Biggest Upset":
        # Calculate upset magnitude (difference between deserved and actual)
        def upset_magnitude(row):
            info = get_deserved_winner(row)
            if info['was_upset']:
                return info['deserved_prob']
            return 0
        filtered_df['_upset_mag'] = filtered_df.apply(upset_magnitude, axis=1)
        filtered_df = filtered_df.sort_values('_upset_mag', ascending=False)
        filtered_df = filtered_df.drop(columns=['_upset_mag'])
    elif sort_by == "Closest Simulation":
        # Games where win probabilities were closest to 50-50
        filtered_df['_closeness'] = abs(filtered_df['home_wp'] - 0.5)
        filtered_df = filtered_df.sort_values('_closeness', ascending=True)
        filtered_df = filtered_df.drop(columns=['_closeness'])
    
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Results summary
    st.info(f"📊 Showing **{len(filtered_df)}** games")
    
    if len(filtered_df) == 0:
        st.warning("No games match your filters. Try adjusting the date range or team selection.")
        return
    
    # Pagination
    GAMES_PER_PAGE = 10
    total_pages = (len(filtered_df) - 1) // GAMES_PER_PAGE + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )
    
    start_idx = (page - 1) * GAMES_PER_PAGE
    end_idx = min(start_idx + GAMES_PER_PAGE, len(filtered_df))
    
    st.caption(f"Showing games {start_idx + 1}-{end_idx} of {len(filtered_df)}")
    
    # Display games
    for idx in range(start_idx, end_idx):
        row = filtered_df.iloc[idx]
        display_game_card(row, expanded=(idx == start_idx))  # Expand first game


if __name__ == "__main__":
    main()