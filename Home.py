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

    /* Slightly narrower sidebar */
    [data-testid="stSidebar"] {
        min-width: 200px;
        max-width: 200px;
    }
    
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

    .hero-explainer {
        font-size: 1.0rem;
        color: #718096;
        margin-bottom: 1.5rem;
        line-height: 1.6;
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


def render_feature_cards(current_season):
    """Render navigation cards for each section in a 3+2 layout."""
    st.markdown("### Explore")

    # Row 1: 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.page_link("pages/1_Game_Simulations.py",
                     label="📊 **Game Simulations**",
                     use_container_width=True)
        st.image("https://dtw-streamlit.s3.amazonaws.com/sample-images/sample_rd.png",
                 width="stretch")
        st.caption(f"Browse all {current_season} games with deserve-to-win analysis. Filter by team, date, or find the biggest upsets.")

    with col2:
        st.page_link("pages/2_Team_Luck_Rankings.py",
                     label="🏆 **Team Luck Rankings**",
                     use_container_width=True)
        st.image(f"https://dtw-streamlit.s3.amazonaws.com/team-rankings/{current_season}/net_lucky_wins.png",
                 width="stretch")
        st.caption("Aggregate deserve-to-win percentages across the season. See which teams are lucky vs. unlucky.")

    with col3:
        st.page_link("pages/3_Playoff_Probabilities.py",
                     label="🎯 **Playoff Probabilities**",
                     use_container_width=True)
        playoff_img = _get_playoff_image_url(current_season)
        if playoff_img:
            st.image(playoff_img, width="stretch")
        st.caption("Monte Carlo simulations of the rest of the season using deserve-to-win team strengths.")

    # Row 2: 3 columns
    col4, col5, col6 = st.columns(3)

    with col4:
        st.page_link("pages/4_Batted_Ball_Explorer.py",
                     label="⚾ **Batted Ball Explorer**",
                     use_container_width=True)
        st.markdown("""
        <table style="width:100%; border-collapse:collapse; background:#F7FAFC; border-radius:8px; font-size:0.75rem; overflow:hidden;">
          <tr style="border-bottom:1px solid #E2E8F0;">
            <th style="padding:6px 8px; text-align:left; color:#1E3A5F;">Player</th>
            <th style="padding:6px 8px; text-align:right; color:#1E3A5F;">EV</th>
            <th style="padding:6px 8px; text-align:right; color:#1E3A5F;">LA</th>
            <th style="padding:6px 8px; text-align:left; color:#1E3A5F;">Result</th>
            <th style="padding:6px 8px; text-align:right; color:#1E3A5F;">Est. Bases</th>
          </tr>
          <tr style="border-bottom:1px solid #EDF2F7;">
            <td style="padding:4px 8px;">A. Judge</td><td style="padding:4px 8px; text-align:right;">112.3</td><td style="padding:4px 8px; text-align:right;">28°</td><td style="padding:4px 8px;">Home Run</td><td style="padding:4px 8px; text-align:right; font-weight:600;">3.45</td>
          </tr>
          <tr style="border-bottom:1px solid #EDF2F7;">
            <td style="padding:4px 8px;">S. Ohtani</td><td style="padding:4px 8px; text-align:right;">108.7</td><td style="padding:4px 8px; text-align:right;">15°</td><td style="padding:4px 8px;">Single</td><td style="padding:4px 8px; text-align:right; font-weight:600;">1.82</td>
          </tr>
          <tr style="border-bottom:1px solid #EDF2F7;">
            <td style="padding:4px 8px;">M. Betts</td><td style="padding:4px 8px; text-align:right;">105.1</td><td style="padding:4px 8px; text-align:right;">32°</td><td style="padding:4px 8px;">Double</td><td style="padding:4px 8px; text-align:right; font-weight:600;">2.14</td>
          </tr>
          <tr>
            <td style="padding:4px 8px;">J. Soto</td><td style="padding:4px 8px; text-align:right;">101.9</td><td style="padding:4px 8px; text-align:right;">-8°</td><td style="padding:4px 8px; color:#999;">Out</td><td style="padding:4px 8px; text-align:right; font-weight:600;">0.67</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)
        st.caption("Search individual batted balls. See outcome probabilities for any exit velocity, launch angle, and spray angle.")

    with col5:
        st.page_link("pages/5_Daily_Highlights.py",
                     label="⭐ **Daily Highlights**",
                     use_container_width=True)
        st.markdown("""
        <div style="background:#F7FAFC; border-radius:8px; padding:8px 10px; font-size:0.75rem;">
          <div style="font-weight:600; color:#1E3A5F; margin-bottom:4px;">🔥 Top Estimated Bases</div>
          <table style="width:100%; border-collapse:collapse; margin-bottom:8px;">
            <tr style="border-bottom:1px solid #EDF2F7;">
              <td style="padding:3px 4px;">A. Judge</td><td style="padding:3px 4px; text-align:right;">112.3 mph</td><td style="padding:3px 4px; text-align:right;">28°</td><td style="padding:3px 4px; text-align:right; font-weight:600;">3.45</td>
            </tr>
            <tr>
              <td style="padding:3px 4px;">S. Ohtani</td><td style="padding:3px 4px; text-align:right;">108.7 mph</td><td style="padding:3px 4px; text-align:right;">25°</td><td style="padding:3px 4px; text-align:right; font-weight:600;">3.12</td>
            </tr>
          </table>
          <div style="display:flex; gap:12px;">
            <div style="flex:1;">
              <div style="font-weight:600; color:#1E3A5F; margin-bottom:4px;">😤 Unluckiest Outs</div>
              <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:2px 4px;">R. Acuña Jr.</td><td style="padding:2px 4px; text-align:right; color:#999;">106.2 · .412 xBA</td></tr>
              </table>
            </div>
            <div style="flex:1;">
              <div style="font-weight:600; color:#1E3A5F; margin-bottom:4px;">🍀 Luckiest Hits</div>
              <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:2px 4px;">F. Freeman</td><td style="padding:2px 4px; text-align:right; color:#999;">78.3 · .089 xBA</td></tr>
              </table>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Best and worst batted balls from each game day — top estimated bases, unluckiest outs, and luckiest hits.")

    with col6:
        st.page_link("pages/6_Player_Rankings.py",
                     label="📈 **Player Rankings**",
                     use_container_width=True)
        player_eval_img = _get_player_eval_image_url(current_season)
        if player_eval_img:
            st.image(player_eval_img, width="stretch")
        st.caption("Bayesian rankings of hitters and pitchers by contact quality, with credible intervals and shrinkage.")

    # Row 3: 3 columns — Hitter Profile, Pitcher Profile, Hitter Comparison
    col8, col9, col10 = st.columns(3)

    with col8:
        st.page_link("pages/7_Hitter_Profile.py",
                     label="🧑 **Hitter Profile**",
                     use_container_width=True)
        st.markdown("""
        <div style="background:#F7FAFC; border-radius:8px; padding:10px; font-size:0.75rem;">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:40px; height:40px; background:#E2E8F0; border-radius:50%;"></div>
            <div>
              <div style="font-weight:700; color:#1E3A5F;">Player Name</div>
              <div style="color:#999; font-size:0.7rem;">Team | Pos | Age 27</div>
            </div>
          </div>
          <div style="display:flex; gap:6px; margin-bottom:6px;">
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Avg EV</div>
              <div style="font-weight:600; color:#1E3A5F;">91.2</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Barrel%</div>
              <div style="font-weight:600; color:#1E3A5F;">8.4%</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Luck</div>
              <div style="font-weight:600; color:#38A169;">+4.2</div>
            </div>
          </div>
          <div style="height:6px; background:#E2E8F0; border-radius:3px; overflow:hidden;">
            <div style="width:68%; height:100%; background:linear-gradient(90deg, #2C5282, #3182CE); border-radius:3px;"></div>
          </div>
          <div style="font-size:0.6rem; color:#999; margin-top:2px;">68th percentile — EB/PA</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Individual hitter deep-dives: contact quality heatmaps, luck reports, platoon splits, and Bayesian rankings.")

    with col9:
        st.page_link("pages/9_Pitcher_Profile.py",
                     label="⚾ **Pitcher Profile**",
                     use_container_width=True)
        st.markdown("""
        <div style="background:#F7FAFC; border-radius:8px; padding:10px; font-size:0.75rem;">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:40px; height:40px; background:#E2E8F0; border-radius:50%;"></div>
            <div>
              <div style="font-weight:700; color:#1E3A5F;">Pitcher Name</div>
              <div style="color:#999; font-size:0.7rem;">Team | RHP | Age 28</div>
            </div>
          </div>
          <div style="display:flex; gap:6px; margin-bottom:6px;">
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">EV Allowed</div>
              <div style="font-weight:600; color:#1E3A5F;">87.3</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Barrel%</div>
              <div style="font-weight:600; color:#1E3A5F;">5.2%</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Luck</div>
              <div style="font-weight:600; color:#E53E3E;">-3.1</div>
            </div>
          </div>
          <div style="height:6px; background:#E2E8F0; border-radius:3px; overflow:hidden;">
            <div style="width:24%; height:100%; background:linear-gradient(90deg, #2C5282, #3182CE); border-radius:3px;"></div>
          </div>
          <div style="font-size:0.6rem; color:#999; margin-top:2px;">24th percentile — EB/PA allowed (lower = better)</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Pitcher deep-dives: contact quality allowed, luck reports, batter splits, and Bayesian rankings.")

    with col10:
        st.page_link("pages/8_Hitter_Comparison.py",
                     label="⚔️ **Hitter Comparison**",
                     use_container_width=True)
        st.markdown("""
        <div style="background:#F7FAFC; border-radius:8px; padding:10px; font-size:0.75rem;">
          <div style="display:flex; gap:8px; align-items:center; justify-content:center; margin-bottom:8px;">
            <div style="width:36px; height:36px; background:#E2E8F0; border-radius:50%;"></div>
            <div style="font-weight:700; color:#1E3A5F;">vs</div>
            <div style="width:36px; height:36px; background:#E2E8F0; border-radius:50%;"></div>
          </div>
          <div style="display:flex; gap:6px; margin-bottom:6px;">
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">Avg EV</div>
              <div style="font-weight:600; color:#005A9C;">91.2</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">vs</div>
              <div style="font-weight:600; color:#C41E3A;">89.7</div>
            </div>
          </div>
          <div style="display:flex; gap:6px;">
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">EB/PA</div>
              <div style="font-weight:600; color:#005A9C;">0.412</div>
            </div>
            <div style="flex:1; text-align:center; background:#EDF2F7; border-radius:4px; padding:4px;">
              <div style="font-size:0.65rem; color:#999;">vs</div>
              <div style="font-weight:600; color:#C41E3A;">0.389</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Compare two hitters side-by-side: distributions, spray charts, luck, and Bayesian rankings.")


def render_about_section():
    """Render about section with all content visible (no tabs)."""
    
    # -----------------------------
    # How It Works
    # -----------------------------
    st.subheader("⚙️ How It Works")
    
    st.markdown("""
    1. 📡 **Get the data** — Pull all batted ball events from a completed MLB game via the MLB Stats API
    2. 🤖 **Predict outcomes** — Use a gradient boosting model trained on Statcast data to predict hit probability for each batted ball
    3. 🔄 **Simulate 10,000 times** — Resample each batted ball outcome based on predicted probabilities, including walks, strikeouts, and baserunning
    4. 📊 **Calculate win probability** — Count how often each team wins across all simulations
    """)
    
    st.divider()
    
    # -----------------------------
    # Read More
    # -----------------------------
    st.subheader("📚 Read More")
    
    st.markdown("""
    📝 **[Who Deserved to Win? Building an MLB Game Outcome Simulator](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)**  
    Original methodology, motivation, and results from the 2024 season.
    
    📝 **[Applying Bayesian Hierarchical Methods to MLB Season Win Probabilities](https://medium.com/@dmgrifka_64770/applying-bayesian-hierarchical-methods-to-mlb-season-win-probabilties-with-pystan-468572abb932)**  
    Using deserve-to-win results to estimate true team strength with PyStan.
    """)
    
    st.divider()
    
    # -----------------------------
    # Connect
    # -----------------------------
    st.subheader("🔗 Connect")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Social Media**")
        st.markdown("""
        🐦 **Twitter:** [@mlb_simulator](https://x.com/mlb_simulator)  
        🦋 **Bluesky:** [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)
        """)
    
    with col2:
        st.markdown("**Code & Portfolio**")
        st.markdown("""
        💻 **Simulator Code:** [baseball_game_simulator](https://github.com/dgrifka/baseball_game_simulator)  
        🌐 **Portfolio:** [dgrifka.github.io](https://dgrifka.github.io)
        """)


def main():
    df = load_game_summaries()
    
    st.markdown('<p class="hero-title">⚾ MLB Deserve-to-Win Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Who <em>should</em> have won? 10,000 simulations per game using exit velocity, launch angle, and spray angle.</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-explainer">Every batted ball has a probability of being a hit based on how hard it was hit, at what angle, and where on the field it went. We re-roll each batted ball 10,000 times using those probabilities to see how often each team <em>should</em> have won — separating skill from luck.</p>', unsafe_allow_html=True)

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
