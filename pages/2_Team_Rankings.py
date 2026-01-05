"""
Team Rankings Page (Coming Soon)
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
    page_title="Team Rankings | DTW Simulator",
    page_icon="🏆",
    layout="wide"
)

# Show recent games header (shared across pages)
df = load_game_summaries()

st.title("🏆 Team Rankings")
st.info("**Coming Soon!** This page will show aggregate deserve-to-win percentages across the season to identify which teams have been lucky vs. unlucky.")

st.markdown("""
### Planned Features
- Season-long deserve-to-win record vs actual record
- Luck factor rankings
- Win probability trends over time
- Division/League comparisons
""")

if not df.empty:
    st.divider()
    st.subheader("📊 Quick Stats Preview")
    st.markdown(f"**Total Games in Database:** {len(df)}")
    st.markdown(f"**Date Range:** {df['date'].min().strftime('%b %d, %Y')} - {df['date'].max().strftime('%b %d, %Y')}")
