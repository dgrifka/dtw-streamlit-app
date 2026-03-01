"""
Game Simulations Page
Browse, filter, and view game simulation results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the app root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_game_summaries, get_game_images, get_deserved_winner
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
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .filter-section {
        background: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_month_options(df):
    """Get unique months from the data for filtering."""
    months = df['date'].dt.to_period('M').unique()
    month_options = {}
    for m in sorted(months, reverse=True):
        label = m.strftime('%B %Y')
        month_options[label] = m
    return month_options


def main():
    # Load data
    df = load_game_summaries()
    
    st.title("📊 Game Simulations")
    st.markdown("Browse all games with deserve-to-win analysis. Click any game to see full visualizations.")
    
    if df.empty:
        st.error("Could not load game data. Please try again later.")
        st.stop()

    # ============ FILTERS SECTION ============
    st.markdown("### 🔍 Search & Filter")

    # Season filter
    available_seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.selectbox("Season", options=available_seasons, index=0)
    season_df = df[df['season'] == selected_season].copy()

    # Get all unique teams for the selected season
    all_teams = sorted(set(season_df['home'].tolist() + season_df['away'].tolist()))

    # Row 1: Team and Opponent
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_team = st.selectbox(
            "Team",
            options=["All Teams"] + all_teams,
            help="Filter games involving this team"
        )
    
    with col2:
        if selected_team != "All Teams":
            team_games = season_df[(season_df['home'] == selected_team) | (season_df['away'] == selected_team)]
            opponents = set()
            for _, row in team_games.iterrows():
                if row['home'] == selected_team:
                    opponents.add(row['away'])
                else:
                    opponents.add(row['home'])
            opponent_options = ["All Opponents"] + sorted(opponents)
            selected_opponent = st.selectbox(
                "Opponent",
                options=opponent_options,
                help="Filter by opponent"
            )
        else:
            selected_opponent = "All Opponents"
            st.selectbox("Opponent", options=["Select a team first"], disabled=True)
    
    with col3:
        month_options = get_month_options(season_df)
        selected_month = st.selectbox(
            "Month",
            options=["All Months"] + list(month_options.keys()),
            help="Quick filter by month"
        )
    
    # Row 2: Date Range and Toggles
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if selected_month == "All Months":
            min_date = season_df['date'].min().date()
            max_date = season_df['date'].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Custom date range filter"
            )
        else:
            st.date_input("Date Range", disabled=True, value=datetime.now().date(),
                         help="Clear month filter to use date range")
            date_range = None
    
    with col5:
        upsets_only = st.checkbox(
            "🎲 Upsets Only",
            help="Show only games where the 'wrong' team won"
        )
    
    with col6:
        sort_option = st.selectbox(
            "Sort By",
            options=["Most Recent", "Oldest First", "Biggest Upsets", "Closest Games"],
            help="How to order results"
        )
    
    st.divider()
    
    # ============ APPLY FILTERS ============
    filtered = season_df.copy()
    
    if selected_team != "All Teams":
        filtered = filtered[
            (filtered['home'] == selected_team) | 
            (filtered['away'] == selected_team)
        ]
    
    if selected_opponent != "All Opponents":
        filtered = filtered[
            (filtered['home'] == selected_opponent) | 
            (filtered['away'] == selected_opponent)
        ]
    
    if selected_month != "All Months":
        period = month_options[selected_month]
        filtered = filtered[filtered['date'].dt.to_period('M') == period]
    elif date_range is not None and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered['date'].dt.date >= start) & 
            (filtered['date'].dt.date <= end)
        ]
    
    if upsets_only:
        filtered['is_upset'] = filtered.apply(
            lambda r: get_deserved_winner(r)['was_upset'], axis=1
        )
        filtered = filtered[filtered['is_upset']]
    
    # ============ SORTING ============
    if sort_option == "Most Recent":
        filtered = filtered.sort_values('date', ascending=False)
    elif sort_option == "Oldest First":
        filtered = filtered.sort_values('date', ascending=True)
    elif sort_option == "Biggest Upsets":
        def upset_magnitude(row):
            winner_info = get_deserved_winner(row)
            if winner_info['was_upset']:
                return winner_info['deserved_prob']
            return 0
        filtered['upset_mag'] = filtered.apply(upset_magnitude, axis=1)
        filtered = filtered.sort_values('upset_mag', ascending=False)
    elif sort_option == "Closest Games":
        filtered['closeness'] = abs(filtered['away_wp'] - filtered['home_wp'])
        filtered = filtered.sort_values('closeness', ascending=True)
    
    # ============ DISPLAY RESULTS ============
    result_text = f"**{len(filtered)} games found**"
    if selected_team != "All Teams":
        result_text += f" for {selected_team}"
    if selected_opponent != "All Opponents":
        result_text += f" vs {selected_opponent}"
    if selected_month != "All Months":
        result_text += f" in {selected_month}"
    if upsets_only:
        result_text += " (upsets only)"
    
    st.markdown(result_text)
    
    if filtered.empty:
        st.info("No games match your filters. Try adjusting your search criteria.")
        st.stop()
    
    # Pagination
    GAMES_PER_PAGE = 12
    total_pages = (len(filtered) - 1) // GAMES_PER_PAGE + 1
    
    if total_pages > 1:
        page = st.selectbox(
            f"Page (1-{total_pages})",
            options=range(1, total_pages + 1),
            format_func=lambda x: f"Page {x} of {total_pages}"
        )
    else:
        page = 1
    
    start_idx = (page - 1) * GAMES_PER_PAGE
    end_idx = start_idx + GAMES_PER_PAGE
    page_games = filtered.iloc[start_idx:end_idx]
    
    # Display games in a grid (3 columns)
    for i in range(0, len(page_games), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(page_games):
                row = page_games.iloc[i + j]
                with col:
                    away_short = get_short_name(row['away'])
                    home_short = get_short_name(row['home'])
                    winner_info = get_deserved_winner(row)
                    upset_marker = " 🎲" if winner_info['was_upset'] else ""
                    
                    away_wp = int(round(row['away_wp'] * 100))
                    home_wp = int(round(row['home_wp'] * 100))
                    
                    # Highlight selected team if filtering
                    away_display = away_short
                    home_display = home_short
                    if selected_team != "All Teams":
                        if row['home'] == selected_team:
                            home_display = f"<b>{home_short}</b>"
                        else:
                            away_display = f"<b>{away_short}</b>"
                    
                    # Game card
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 1rem; 
                                margin-bottom: 0.5rem; border: 1px solid #e0e0e0;">
                        <div style="font-size: 0.8rem; color: #666;">
                            {row['date'].strftime('%A, %b %d, %Y')}
                        </div>
                        <div style="font-weight: 600; font-size: 1.1rem;">
                            {away_display} @ {home_display}{upset_marker}
                        </div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #1E3A5F;">
                            {row['away_score']} - {row['home_score']}
                        </div>
                        <div style="font-size: 0.85rem; color: #666;">
                            DTW: {away_wp}% - {home_wp}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # View button - USE SESSION STATE
                    if st.button("View Details", key=f"game_{row['gamePk']}", use_container_width=True):
                        st.session_state['selected_game_pk'] = int(row['gamePk'])
                        st.switch_page("pages/_Game_Detail.py")


if __name__ == "__main__":
    main()
