"""
MLB "Deserve to Win" Simulator - Streamlit App
Landing page with recent games ticker, feature cards, and about section.
"""

import streamlit as st
import os
import sys

# Add the app root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.data_loader import load_game_summaries, get_game_images, get_deserved_winner
from utils.team_mappings import get_short_name

# Page configuration
st.set_page_config(
    page_title="MLB Deserve-to-Win Simulator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #1E3A5F;
        margin-bottom: 0.25rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    
    .feature-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.4;
    }
    
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        color: #718096;
        font-size: 0.85rem;
        text-align: center;
    }
    
    .footer a {
        color: #2C5282;
    }
</style>
""", unsafe_allow_html=True)


def render_recent_games(df):
    """Render clickable recent games section."""
    st.markdown("### 📅 Recent Games")
    st.caption("Click any game to view full simulation details")
    
    recent = df.head(8)
    
    for row_start in [0, 4]:
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            game_idx = row_start + idx
            if game_idx < len(recent):
                row = recent.iloc[game_idx]
                with col:
                    away_short = get_short_name(row['away'])
                    home_short = get_short_name(row['home'])
                    winner_info = get_deserved_winner(row)
                    upset_marker = " 🎲" if winner_info['was_upset'] else ""
                    
                    away_wp = int(round(row['away_wp'] * 100))
                    home_wp = int(round(row['home_wp'] * 100))
                    date_str = row['date'].strftime("%b %d")
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%); 
                                padding: 0.75rem; border-radius: 8px; text-align: center; 
                                color: white; font-size: 0.85rem; margin-bottom: 0.5rem;">
                        <div style="color: #A0AEC0; font-size: 0.7rem;">{date_str}</div>
                        <div style="font-weight: 600;">{away_short} @ {home_short}{upset_marker}</div>
                        <div style="color: #FFD700; font-weight: 700;">{row['away_score']} - {row['home_score']}</div>
                        <div style="color: #A0AEC0; font-size: 0.75rem;">{away_wp}% - {home_wp}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use session state for navigation
                    if st.button("View →", key=f"home_game_{row['gamePk']}", use_container_width=True):
                        st.session_state['selected_game_pk'] = int(row['gamePk'])
                        st.switch_page("pages/_Game_Detail.py")


def render_feature_cards():
    """Render navigation cards for each section."""
    st.markdown("### Explore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/1_Game_Simulations.py", 
                     label="📊 **Game Simulations** - Browse all games", 
                     use_container_width=True)
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_rd.png",
                 width=300)
        st.caption("Browse all 2025 games with deserve-to-win analysis. Filter by team, date, or find the biggest upsets.")
    
    with col2:
        st.page_link("pages/2_Team_Rankings.py",
                     label="🏆 **Team Rankings** - Coming Soon",
                     use_container_width=True)
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_spray.png",
                 width=300)
        st.caption("Aggregate deserve-to-win percentages across the season. See which teams are lucky vs. unlucky.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.page_link("pages/3_Playoff_Probabilities.py",
                     label="🎯 **Playoff Probabilities** - Coming Soon",
                     use_container_width=True)
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_player_contributions.png",
                 width=300)
        st.caption("Monte Carlo simulations of the rest of the season using deserve-to-win team strengths.")
    
    with col4:
        st.page_link("pages/4_Batted_Ball_Explorer.py",
                     label="⚾ **Batted Ball Explorer** - Coming Soon",
                     use_container_width=True)
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_estimated_bases.png",
                 width=300)
        st.caption("Search individual batted balls. See outcome probabilities for any exit velocity, launch angle, and spray angle.")


def render_about_section():
    """Render the about/methodology section."""
    st.markdown("### About")
    
    tab1, tab2, tab3 = st.tabs(["How It Works", "Articles", "Links"])
    
    with tab1:
        st.markdown("""
        **The Deserve-to-Win Simulator** re-simulates every MLB game 10,000 times using actual batted ball data.
        
        For each batted ball in the game, we use:
        - **Exit Velocity** — How hard the ball was hit
        - **Launch Angle** — The vertical angle off the bat  
        - **Spray Angle** — The horizontal direction
        - **Ballpark Factors** — Stadium-specific effects
        
        A Gradient Boosting model (77% accuracy) predicts the probability distribution of outcomes 
        (single, double, out, etc.) for each batted ball. We then resample from these distributions 
        to simulate alternative game outcomes.
        """)
    
    with tab2:
        st.markdown("""
        📝 **[Who Deserved to Win? Building an MLB Game Outcome Simulator](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)**  
        Original methodology, motivation, and results from the 2024 season.
        
        📝 **[Applying Bayesian Hierarchical Methods to MLB Season Win Probabilities](https://medium.com/@dmgrifka_64770/applying-bayesian-hierarchical-methods-to-mlb-season-win-probabilties-with-pystan-468572abb932)**  
        Using deserve-to-win results to estimate true team strength with PyStan.
        """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Social Media")
            st.markdown("""
            🐦 **Twitter:** [@mlb_simulator](https://x.com/mlb_simulator)  
            🦋 **Bluesky:** [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)
            """)
        
        with col2:
            st.markdown("### Code & Portfolio")
            st.markdown("""
            💻 **Simulator Code:** [baseball_game_simulator](https://github.com/dgrifka/baseball_game_simulator)  
            🌐 **Portfolio:** [dgrifka.github.io](https://dgrifka.github.io)
            """)


def main():
    df = load_game_summaries()
    
    st.markdown('<p class="hero-title">⚾ MLB Deserve-to-Win Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Who <em>should</em> have won? 10,000 simulations per game using exit velocity, launch angle, and spray angle.</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.link_button("🐦 Twitter", "https://x.com/mlb_simulator", use_container_width=True)
    with col2:
        st.link_button("🦋 Bluesky", "https://bsky.app/profile/mlb-simulator.bsky.social", use_container_width=True)
    with col3:
        st.link_button("💻 GitHub", "https://github.com/dgrifka/baseball_game_simulator", use_container_width=True)
    with col4:
        st.link_button("🌐 Portfolio", "https://dgrifka.github.io", use_container_width=True)
    
    st.divider()
    
    if not df.empty:
        render_recent_games(df)
        st.divider()
    
    render_feature_cards()
    
    st.divider()
    
    render_about_section()
    
    st.markdown("""
    <div class="footer">
        Built by <a href="https://dgrifka.github.io">Derek Grifka</a> · 
        Data from MLB Stats API · 
        Model trained on Statcast data
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
