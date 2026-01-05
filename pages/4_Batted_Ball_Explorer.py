"""
Batted Ball Explorer Page (Coming Soon)
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
    page_title="Batted Ball Explorer | DTW Simulator",
    page_icon="⚾",
    layout="wide"
)

df = load_game_summaries()

st.title("⚾ Batted Ball Explorer")
st.info("**Coming Soon!** This page will let you explore individual batted balls and see outcome probabilities.")

st.markdown("""
### Planned Features
- Search by exit velocity, launch angle, spray angle
- Outcome probability distributions
- Compare to league averages
- Historical similar batted balls
""")

if not df.empty:
    st.divider()
    st.subheader("📊 Quick Stats Preview")
    st.markdown(f"**Total Games in Database:** {len(df)}")
