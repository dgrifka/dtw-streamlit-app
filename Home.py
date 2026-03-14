"""
MLB "Deserve to Win" Simulator - Streamlit App
Navigation entrypoint with grouped sidebar and landing page.
"""

import streamlit as st
import os
import sys

# Add the app root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Page configuration (shared across all pages via st.navigation)
st.set_page_config(
    page_title="MLB Deserve-to-Win Simulator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto",
)

# Logo (applies to all pages via st.navigation)
_logo_path = os.path.join(current_dir, "assets", "mlb_simulator_logo.png")
if os.path.exists(_logo_path):
    st.logo(_logo_path)

# ── Navigation ──────────────────────────────────────────────────────────────

nav = st.navigation(
    {
        "": [
            st.Page("_home.py", title="Home", default=True),
            st.Page("pages/_Game_Detail.py", title="Game Detail", visibility="hidden"),
        ],
        "Games & Standings": [
            st.Page("pages/1_Game_Simulations.py", title="Game Simulations"),
            st.Page("pages/2_Team_Luck_Rankings.py", title="Team Luck Rankings"),
            st.Page("pages/3_Playoff_Probabilities.py", title="Playoff Probabilities"),
        ],
        "Batted Ball Data": [
            st.Page("pages/4_Batted_Ball_Explorer.py", title="Batted Ball Explorer"),
            st.Page("pages/6_Player_Rankings.py", title="Player Rankings"),
        ],
        "Player Profiles": [
            st.Page("pages/7_Hitter_Profile.py", title="Hitter Profile"),
            st.Page("pages/8_Hitter_Comparison.py", title="Hitter Comparison"),
            st.Page("pages/9_Pitcher_Profile.py", title="Pitcher Profile"),
            st.Page("pages/10_About.py", title="About"),
        ],
    },
)

nav.run()
