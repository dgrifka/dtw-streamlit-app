"""
MLB "Deserve to Win" Simulator - Streamlit App
Landing page with recent games ticker, feature cards, and about section.
"""

import streamlit as st

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_game_summaries,
    get_game_images,
    get_deserved_winner,
    get_short_name,
)

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
    /* Hide default hamburger menu for cleaner look */
    #MainMenu {visibility: hidden;}
    
    /* Ticker styling */
    .ticker-container {
        background: linear-gradient(90deg, #1E3A5F 0%, #2C5282 100%);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        overflow-x: auto;
        white-space: nowrap;
    }
    
    .ticker-label {
        color: #FFD700;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 1rem;
    }
    
    /* Game card in ticker */
    .ticker-game {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-right: 0.75rem;
        color: white;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: background 0.2s;
    }
    
    .ticker-game:hover {
        background: rgba(255,255,255,0.2);
    }
    
    .ticker-score {
        font-weight: 700;
        color: #FFD700;
    }
    
    .ticker-wp {
        font-size: 0.75rem;
        color: #A0AEC0;
    }
    
    /* Feature card styling */
    .feature-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
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
    
    /* Hero section */
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
    
    /* Social buttons */
    .social-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* About section */
    .about-card {
        background: #F7FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
    }
    
    /* Footer */
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
    
    /* Upset badge */
    .upset-badge {
        background: #F6AD55;
        color: #744210;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)


def render_ticker(df):
    """Render the recent games ticker at the top."""
    # Get most recent 8 games
    recent = df.head(8)
    
    st.markdown('<p style="color: #718096; font-size: 0.85rem; margin-bottom: 0.5rem;">📅 RECENT GAMES</p>', unsafe_allow_html=True)
    
    # Create columns for each game
    cols = st.columns(len(recent))
    
    for idx, (col, (_, row)) in enumerate(zip(cols, recent.iterrows())):
        with col:
            away_short = get_short_name(row['away'])
            home_short = get_short_name(row['home'])
            winner_info = get_deserved_winner(row)
            
            # Determine if upset
            upset_marker = " 🎲" if winner_info['was_upset'] else ""
            
            # Format win probability
            away_wp = int(round(row['away_wp'] * 100))
            home_wp = int(round(row['home_wp'] * 100))
            
            # Create a clickable container
            date_str = row['date'].strftime("%m/%d")
            
            with st.container():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%); 
                            padding: 0.75rem; border-radius: 8px; text-align: center; 
                            color: white; font-size: 0.85rem;">
                    <div style="color: #A0AEC0; font-size: 0.7rem;">{date_str}</div>
                    <div style="font-weight: 600;">{away_short} @ {home_short}{upset_marker}</div>
                    <div style="color: #FFD700; font-weight: 700;">{row['away_score']} - {row['home_score']}</div>
                    <div style="color: #A0AEC0; font-size: 0.75rem;">{away_wp}% - {home_wp}%</div>
                </div>
                """, unsafe_allow_html=True)


def render_feature_cards():
    """Render the main feature cards with sample images."""
    
    st.markdown("### Explore the Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spray Chart Card
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Spray Charts</div>
            <div class="feature-desc">
                Stadium-specific visualization showing batted ball locations 
                with predicted outcomes based on exit velocity, launch angle, and spray angle.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for sample image - user will provide
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_spray.png", 
                 use_container_width=True)
    
    with col2:
        # Run Distribution Card
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Run Distributions</div>
            <div class="feature-desc">
                See how runs were distributed across 10,000 simulations. 
                Was the actual score typical or a lucky/unlucky outlier?
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for sample image
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_rd.png",
                 use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Expected Bases Card
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📈 Expected Bases</div>
            <div class="feature-desc">
                Player rankings by expected bases — who made the best contact 
                regardless of whether they got lucky or unlucky?
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_estimated_bases.png",
                 use_container_width=True)
    
    with col4:
        # Player Contributions Card  
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">👥 Player Contributions</div>
            <div class="feature-desc">
                See each player's contribution to their team's expected run production,
                split by batted balls vs. walks.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_player_contributions.png",
                 use_container_width=True)
    
    # CTA Button
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.page_link("pages/1_Game_Simulations.py", label="🔍 Browse All Game Simulations", use_container_width=True)


def render_about_section():
    """Render the collapsible About section."""
    
    with st.expander("ℹ️ About This Project", expanded=False):
        tab1, tab2, tab3 = st.tabs(["How It Works", "Articles", "Links"])
        
        with tab1:
            st.markdown("""
            ### The Model
            
            For each batted ball, a **Gradient Boosting Classifier** predicts outcome probabilities:
            
            | Feature | Description |
            |---------|-------------|
            | Exit Velocity | How hard the ball was hit (mph) |
            | Launch Angle | Vertical angle off the bat (degrees) |
            | Spray Angle | Horizontal direction, adjusted for batter handedness |
            | Ballpark | Stadium-specific dimensions and factors |
            
            **Outcomes:** Out, Single, Double, Triple, Home Run  
            **Accuracy:** 82% on held-out test data
            
            ### The Simulation
            
            1. Get all batted ball events from the MLB Stats API
            2. Predict outcome probabilities for each batted ball
            3. Run **10,000 simulations** sampling outcomes based on probabilities
            4. Calculate win percentage for each team
            
            ### New in 2025: Spray Angle
            
            The model now incorporates **spray angle** — where the ball was hit on the field.
            Pull-side vs. opposite-field contact has different outcome distributions even with
            identical exit velocity and launch angle. This improved accuracy from 77% to 82%.
            """)
        
        with tab2:
            st.markdown("""
            ### Published Articles
            
            📝 **[Who "Deserved" to Win? Building an MLB Game Outcome Simulator](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)**  
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
    # Load data for ticker
    df = load_game_summaries()
    
    # Hero Section
    st.markdown('<p class="hero-title">⚾ MLB Deserve-to-Win Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Who <em>should</em> have won? 10,000 simulations per game using exit velocity, launch angle, and spray angle.</p>', unsafe_allow_html=True)
    
    # Social links row
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
    
    # Recent Games Ticker
    if not df.empty:
        render_ticker(df)
        st.divider()
    
    # Feature Cards with Sample Images
    render_feature_cards()
    
    st.divider()
    
    # About Section
    render_about_section()
    
    # Footer
    st.markdown("""
    <div class="footer">
        Built by <a href="https://dgrifka.github.io">Derek Grifka</a> · 
        Data from MLB Stats API · 
        Model trained on Statcast data
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()