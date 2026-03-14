"""
About Page
How it works, methodology overview, articles, and contact links.
"""

import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.responsive import inject_responsive_css, render_home_link

inject_responsive_css()

st.title("About")

# -----------------------------------------------------------------------------
# How It Works
# -----------------------------------------------------------------------------
st.subheader("How It Works")

st.markdown("""
After every MLB game, the simulator re-evaluates each batted ball to answer one question:
**did the right team win?**

1. **Collect every batted ball** from the completed game — exit velocity, launch angle, and spray angle
2. **Estimate each outcome** using a machine learning model trained on millions of historical batted balls to predict the probability of a hit, double, home run, etc.
3. **Simulate 10,000 games** by re-rolling each batted ball outcome based on those probabilities, along with walks, strikeouts, and baserunning
4. **Determine a win probability** for each team based on how often they win across all 10,000 simulations
""")

st.markdown("")

# -----------------------------------------------------------------------------
# Read More
# -----------------------------------------------------------------------------
st.subheader("Read More")

st.markdown("""
**[Who Deserved to Win? Building an MLB Game Outcome Simulator](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)**
Original methodology, motivation, and results from the 2024 season.

**[Applying Bayesian Hierarchical Methods to MLB Season Win Probabilities](https://medium.com/@dmgrifka_64770/applying-bayesian-hierarchical-methods-to-mlb-season-win-probabilties-with-pystan-468572abb932)**
Using deserve-to-win results to estimate true team strength.
""")

st.markdown("")

# -----------------------------------------------------------------------------
# Connect
# -----------------------------------------------------------------------------
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

render_home_link()
