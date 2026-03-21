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
After every MLB game, we ask one question: **did the right team win?**

A lot can go wrong between bat and scoreboard. A 110 mph line drive gets caught. A weak
grounder finds a hole. Our simulator strips away that randomness and re-simulates the game
10,000 times to find out what *should* have happened.

Here's how:

1. **Collect every plate appearance** from the game — batted balls (exit velocity, launch
   angle, spray angle), plus walks, strikeouts, hit-by-pitches, and stolen bases
2. **Estimate batted ball outcomes** using a model trained on millions of historical batted
   balls to predict the probability of a hit, double, home run, etc.
3. **Simulate 10,000 games** by re-rolling each batted ball outcome based on those
   probabilities, while preserving walks, strikeouts, and baserunning
4. **Calculate a win probability** — the percentage of simulations each team wins
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

# Explainer Threads
st.markdown("**Explainer Threads**")

THREADS = [
    {
        "tag": "Player Metrics",
        "tag_color": "#2B6CB0",
        "title": "What is EB/PA?",
        "description": "How we estimate a hitter's true production — and why sample size matters.",
        "url": "https://x.com/mlb_simulator/status/2034742985791803895",
    },
    {
        "tag": "Run Scoring",
        "tag_color": "#2F855A",
        "title": "Can a Simple Formula Predict Baseball Scores?",
        "description": "Why the Poisson distribution fails for MLB runs — and what works better.",
        "url": "https://x.com/mlb_simulator/status/2035183677861163433",
    },
    {
        "tag": "Projections",
        "tag_color": "#975A16",
        "title": "Understanding Player Projections",
        "description": "How multi-year Bayesian models handle aging, uncertainty, and small samples.",
        "url": "https://x.com/mlb_simulator/status/2035401562906894794",
    },
]

cols = st.columns(len(THREADS))
for col, thread in zip(cols, THREADS):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<span style="background:{thread["tag_color"]}; color:white; '
                f'font-size:0.7rem; font-weight:600; padding:2px 8px; '
                f'border-radius:10px;">{thread["tag"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'**{thread["title"]}**')
            st.caption(thread["description"])
            st.markdown(f'[Read thread &rarr;]({thread["url"]})')

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
