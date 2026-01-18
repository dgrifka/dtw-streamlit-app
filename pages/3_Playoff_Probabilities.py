"""
Playoff Probabilities Page (Coming Soon)
"""

import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_game_summaries

st.set_page_config(
    page_title="Playoff Probabilities | DTW Simulator",
    page_icon="🎯",
    layout="wide"
)

df = load_game_summaries()

st.title("🎯 Playoff Probabilities")
st.info("**Coming Soon!** This page will show Monte Carlo simulations of the rest of the season using deserve-to-win team strengths.")

st.markdown("""
### Planned Features
- Playoff odds for each team
- Division winner probabilities
- Wild card race projections
- AL/NL Rooting Guide
""")

if not df.empty:
    st.divider()
    st.subheader("📊 Quick Stats Preview")
    st.markdown(f"**Total Games in Database:** {len(df)}")
