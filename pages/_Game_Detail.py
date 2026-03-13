"""
Game Detail Page
Shows all 4 visualizations for a selected game.
"""

import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_game_summaries, get_game_images, get_deserved_winner
from utils.team_mappings import get_short_name
from utils.responsive import inject_responsive_css, render_home_link, upset_badge_html

st.set_page_config(
    page_title="Game Detail | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .team-score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .team-name {
        font-size: 1rem;
        color: #4A5568;
    }
    .game-info-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .image-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .image-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.25rem;
    }
    .image-caption {
        font-size: 0.8rem;
        color: #718096;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

inject_responsive_css()


def main():
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "mlb_simulator_logo.png")
    st.logo(_logo_path)

    # Get gamePk from session state first, then query params as fallback
    game_pk = st.session_state.get('selected_game_pk', None)
    
    if game_pk is None:
        params = st.query_params
        game_pk = params.get("gamePk", None)
        if game_pk is not None:
            game_pk = int(game_pk)
    
    # Load all games
    df = load_game_summaries()
    
    if df.empty:
        st.error("Could not load game data.")
        return
    
    # If no game selected, show message and link
    if game_pk is None:
        st.title("Game Detail")
        st.warning("No game selected. Please select a game from Game Simulations.")
        st.page_link("pages/1_Game_Simulations.py", label="Browse All Games", width="stretch")
        return

    # Load specific game
    game_row = df[df['gamePk'] == game_pk]

    if game_row.empty:
        st.error(f"Game {game_pk} not found.")
        st.page_link("pages/1_Game_Simulations.py", label="Browse All Games", width="stretch")
        return
    
    row = game_row.iloc[0]
    
    # Get team info
    away_short = get_short_name(row['away'])
    home_short = get_short_name(row['home'])
    winner_info = get_deserved_winner(row)
    
    away_wp = int(round(row['away_wp'] * 100))
    home_wp = int(round(row['home_wp'] * 100))
    
    # ============ HEADER ============
    if st.button("← Back to Games"):
        st.switch_page("pages/1_Game_Simulations.py")
    
    st.divider()
    
    # Game title
    badge = upset_badge_html(winner_info['was_upset'])
    st.markdown(f"<h2>{row['away']} @ {row['home']}{badge}</h2>", unsafe_allow_html=True)
    
    # Score and info row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">{away_short}</div>
            <div class="team-score">{row['away_score']}</div>
            <div style="font-size: 0.85rem; color: #666;">DTW: {away_wp}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">{home_short}</div>
            <div class="team-score">{row['home_score']}</div>
            <div style="font-size: 0.85rem; color: #666;">DTW: {home_wp}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        date_str = row['date'].strftime('%-m/%d/%y')
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">Date</div>
            <div class="team-score" style="font-size: 1.2rem;">{date_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if winner_info['was_upset']:
            result_text = "Upset"
            result_detail = f"{get_short_name(winner_info['deserved_winner'])} deserved to win"
        else:
            result_text = "Deserved"
            result_detail = "Expected team won"
        
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">Result</div>
            <div class="team-score" style="font-size: 1.2rem;">{result_text}</div>
            <div style="font-size: 0.75rem; color: #666;">{result_detail}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ============ IMAGES IN COMPARTMENTS ============
    st.subheader("Simulation Visualizations")
    
    images = get_game_images(row)
    
    # Row 1: Spray Chart and Estimated Bases
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Spray Chart</div>
            <div class="image-caption">Where batted balls landed in the ballpark</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['spray'], width="stretch")
    
    with col2:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Estimated Bases</div>
            <div class="image-caption">Expected bases vs actual bases by player</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['estimated_bases'], width="stretch")
    
    # Separator
    st.markdown("<hr style='border: none; border-top: 1px solid #e0e0e0; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Row 2: Run Distribution and Player Contributions
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Run Distribution</div>
            <div class="image-caption">Simulated run outcomes from 10,000 simulations</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['rd'], width="stretch")
    
    with col4:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Player Contributions</div>
            <div class="image-caption">Individual player impact on win probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['player_contributions'], width="stretch")
    
    # ============ SIDEBAR ============
    st.sidebar.header("Find Another Game")
    
    teams = sorted(df['home'].unique().tolist())
    selected_team = st.sidebar.selectbox("Filter by Team", ["All Teams"] + teams)
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=row['date'].date(),
        min_value=min_date,
        max_value=max_date
    )
    
    filtered_df = df.copy()
    if selected_team != "All Teams":
        filtered_df = filtered_df[
            (filtered_df['home'] == selected_team) | 
            (filtered_df['away'] == selected_team)
        ]
    
    date_games = filtered_df[filtered_df['date'].dt.date == selected_date]
    
    if not date_games.empty:
        st.sidebar.markdown(f"**Games on {selected_date.strftime('%-m/%d/%y')}:**")
        for _, r in date_games.iterrows():
            away_s = get_short_name(r['away'])
            home_s = get_short_name(r['home'])
            label = f"{away_s} @ {home_s} ({r['away_score']}-{r['home_score']})"
            
            if r['gamePk'] == game_pk:
                st.sidebar.markdown(f"▶ **{label}** (current)")
            else:
                if st.sidebar.button(label, key=f"sidebar_{r['gamePk']}"):
                    st.session_state['selected_game_pk'] = int(r['gamePk'])
                    st.rerun()
    else:
        st.sidebar.info("No games found for this date/team.")
    
    st.sidebar.divider()
    st.sidebar.page_link("pages/1_Game_Simulations.py", label="Browse All Games")

    render_home_link()


if __name__ == "__main__":
    main()
