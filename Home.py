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

import requests

from utils.data_loader import load_game_summaries, get_game_images, get_deserved_winner
from utils.team_mappings import get_short_name
from utils.responsive import inject_responsive_css

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
        margin-bottom: 0.75rem;
    }

    /* Uniform card height per row */
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
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

inject_responsive_css()


@st.cache_data(ttl=86400)
def _get_player_eval_image_url(current_season):
    """Return a working player evaluation image URL, falling back to prior year (cached 24h)."""
    from datetime import datetime
    current_year = datetime.now().year
    candidates = dict.fromkeys([current_season, current_year, current_season - 1, current_year - 1])
    for season in candidates:
        url = f"https://dtw-streamlit.s3.amazonaws.com/player-evaluations/{season}/latest/top_hitters.png"
        try:
            r = requests.head(url, timeout=3)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


@st.cache_data(ttl=86400)
def _get_playoff_image_url(current_season):
    """Return a working playoff image URL, falling back to prior/current year (cached 24h)."""
    from datetime import datetime
    current_year = datetime.now().year
    # Try current season, current calendar year, then one year back from each
    candidates = dict.fromkeys([current_season, current_year, current_season - 1, current_year - 1])
    for season in candidates:
        url = f"https://dtw-streamlit.s3.amazonaws.com/playoff-probabilities/{season}/latest/team_strength.png"
        try:
            r = requests.head(url, timeout=3)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def render_recent_games(df):
    """Render clickable recent games section."""
    st.markdown("### Recent Games")
    st.caption("Click any game to view full simulation details")
    
    recent = df.head(4)

    cols = st.columns(4)
    for idx, col in enumerate(cols):
        if idx < len(recent):
            row = recent.iloc[idx]
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

                if st.button("View →", key=f"home_game_{row['gamePk']}", use_container_width=True):
                    st.session_state['selected_game_pk'] = int(row['gamePk'])
                    st.switch_page("pages/_Game_Detail.py")


def _render_card(page, label, image_url, caption, fit="cover"):
    """Render a single feature card with border, image, and caption."""
    with st.container(border=True):
        st.page_link(page, label=f"**{label}**", use_container_width=True)
        if image_url:
            bg = "background:#F7FAFC;" if fit == "contain" else ""
            st.markdown(
                f'<img src="{image_url}" class="responsive-card-img" style="width:100%; height:220px; '
                f'object-fit:{fit}; object-position:top; border-radius:6px; {bg}">',
                unsafe_allow_html=True,
            )
        st.caption(caption)


def render_feature_cards(current_season):
    """Render navigation cards grouped by category."""
    S3 = "https://dtw-streamlit.s3.amazonaws.com"

    playoff_img = _get_playoff_image_url(current_season)
    player_eval_img = _get_player_eval_image_url(current_season)

    # --- Games & Standings ---
    st.markdown("#### Games & Standings")
    col1, col2, col3 = st.columns(3)
    with col1:
        _render_card(
            "pages/1_Game_Simulations.py", "Game Simulations",
            f"{S3}/sample-images/sample_rd.png",
            f"Browse all {current_season} games with deserve-to-win analysis. Filter by team, date, or find the biggest upsets.",
            fit="contain")
    with col2:
        _render_card(
            "pages/2_Team_Luck_Rankings.py", "Team Luck Rankings",
            f"{S3}/team-rankings/{current_season}/net_lucky_wins.png",
            "Aggregate deserve-to-win percentages across the season. See which teams are lucky vs. unlucky.")
    with col3:
        _render_card(
            "pages/3_Playoff_Probabilities.py", "Playoff Probabilities",
            playoff_img,
            "Monte Carlo simulations of the rest of the season using deserve-to-win team strengths.",
            fit="contain")

    # --- Batted Ball Data ---
    st.markdown("#### Batted Ball Data")
    col4, col5, col6 = st.columns(3)
    with col4:
        _render_card(
            "pages/4_Batted_Ball_Explorer.py", "Batted Ball Explorer",
            f"{S3}/sample-images/sample_batted_ball_explorer.png",
            "Search individual batted balls. See outcome probabilities for any exit velocity, launch angle, and spray angle.")
    with col5:
        _render_card(
            "pages/5_Daily_Highlights.py", "Daily Highlights",
            f"{S3}/sample-images/sample_daily_highlights.png",
            "Best and worst batted balls from each game day — top estimated bases, unluckiest outs, and luckiest hits.")
    with col6:
        _render_card(
            "pages/6_Player_Rankings.py", "Player Rankings",
            player_eval_img,
            "Bayesian rankings of hitters and pitchers by contact quality, with credible intervals and shrinkage.",
            fit="contain")

    # --- Player Profiles ---
    st.markdown("#### Player Profiles")
    col7, col8, col9 = st.columns(3)
    with col7:
        _render_card(
            "pages/7_Hitter_Profile.py", "Hitter Profile",
            f"{S3}/sample-images/sample_hitter_profile.png",
            "Individual hitter deep-dives: contact quality heatmaps, luck reports, platoon splits, and Bayesian rankings.")
    with col8:
        _render_card(
            "pages/9_Pitcher_Profile.py", "Pitcher Profile",
            f"{S3}/sample-images/sample_pitcher_profile.png",
            "Pitcher deep-dives: contact quality allowed, luck reports, batter splits, and Bayesian rankings.")
    with col9:
        _render_card(
            "pages/8_Hitter_Comparison.py", "Hitter Comparison",
            f"{S3}/sample-images/sample_hitter_comparison.png",
            "Compare two hitters side-by-side: distributions, spray charts, luck, and Bayesian rankings.")


def render_about_section():
    """Render about section with all content visible (no tabs)."""
    
    # -----------------------------
    # How It Works
    # -----------------------------
    st.subheader("How It Works")

    st.markdown("""
    1. **Get the data** — Pull all batted ball events from a completed MLB game via the MLB Stats API
    2. **Predict outcomes** — Use a gradient boosting model trained on Statcast data to predict hit probability for each batted ball
    3. **Simulate 10,000 times** — Resample each batted ball outcome based on predicted probabilities, including walks, strikeouts, and baserunning
    4. **Calculate win probability** — Count how often each team wins across all simulations
    """)
    
    st.divider()
    
    # -----------------------------
    # Read More
    # -----------------------------
    st.subheader("Read More")

    st.markdown("""
    **[Who Deserved to Win? Building an MLB Game Outcome Simulator](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)**
    Original methodology, motivation, and results from the 2024 season.

    **[Applying Bayesian Hierarchical Methods to MLB Season Win Probabilities](https://medium.com/@dmgrifka_64770/applying-bayesian-hierarchical-methods-to-mlb-season-win-probabilties-with-pystan-468572abb932)**
    Using deserve-to-win results to estimate true team strength with PyStan.
    """)
    
    st.divider()
    
    # -----------------------------
    # Connect
    # -----------------------------
    st.subheader("Connect")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Social Media**")
        st.markdown("""
        **Twitter:** [@mlb_simulator](https://x.com/mlb_simulator)
        **Bluesky:** [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)
        """)
    
    with col2:
        st.markdown("**Code & Portfolio**")
        st.markdown("""
        **Simulator Code:** [baseball_game_simulator](https://github.com/dgrifka/baseball_game_simulator)
        **Portfolio:** [dgrifka.github.io](https://dgrifka.github.io)
        """)


def main():
    df = load_game_summaries()
    
    st.markdown('<p class="hero-title">MLB Deserve-to-Win Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Who <em>should</em> have won? 10,000 simulations per game using exit velocity, launch angle, and spray angle.</p>', unsafe_allow_html=True)

    st.divider()

    current_season = int(df['season'].max()) if not df.empty else 2026

    if not df.empty:
        render_recent_games(df)
        st.divider()

    render_feature_cards(current_season)
    
    st.divider()
    
    render_about_section()
    
    st.markdown("""
    <div class="footer">
        Built by <a href="https://dgrifka.github.io">Derek Grifka</a> · 
        Model trained on Statcast data
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
