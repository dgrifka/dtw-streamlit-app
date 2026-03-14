"""
Playoff Probabilities Page

Displays daily playoff probability charts (pre-rendered by the simulator
pipeline) and a sortable probability table.  Data is loaded from S3.
"""

import streamlit as st
import pandas as pd
import os
import sys
import time

# Hourly cache-busting version — S3 ignores query params on unsigned requests,
# but Streamlit's st.image() caches by URL string, so rotating this forces re-fetch.
_cache_version = int(time.time() // 3600)

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_playoff_probabilities, _image_exists, S3_BASE_URL
from utils.responsive import inject_responsive_css, render_home_link

inject_responsive_css()

# S3 base URL for playoff probability charts
S3_PLAYOFF_URL = f"{S3_BASE_URL}/playoff-probabilities"


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def _get_available_playoff_seasons():
    """Auto-detect which seasons have playoff data on S3 (cached 24h)."""
    available = []
    for year in range(datetime.now().year, datetime.now().year - 3, -1):
        url = f"{S3_PLAYOFF_URL}/{year}/latest/results.parquet"
        if _image_exists(url):
            available.append(year)
    return available


# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

st.title("Playoff Probabilities")
st.markdown(
    "How likely is your team to make the playoffs? "
    "Powered by **50,000 season simulations**, updated daily."
)

# Determine season — auto-detect which years have data in S3
from datetime import datetime
current_year = datetime.now().year

available_seasons = _get_available_playoff_seasons()

if available_seasons:
    selected_season = st.selectbox(
        "Season", available_seasons,
        index=0,
        label_visibility="collapsed" if len(available_seasons) == 1 else "visible",
    )
else:
    selected_season = current_year  # fall through to pre-season message

# Load probability table from S3
results_df = load_playoff_probabilities(selected_season)

# Check for chart availability
prob_chart_url = f"{S3_PLAYOFF_URL}/{selected_season}/latest/playoff_probabilities.png?v={_cache_version}"
strength_chart_url = f"{S3_PLAYOFF_URL}/{selected_season}/latest/team_strength.png?v={_cache_version}"

has_prob_chart = _image_exists(prob_chart_url)
has_strength_chart = _image_exists(strength_chart_url)
has_data = results_df is not None and not results_df.empty

if not has_data and not has_prob_chart:
    st.info(
        f"{selected_season} playoff probabilities will appear here once the season begins. "
        "Pre-season projections based on prior-year team strength may also be available."
    )
    st.stop()


# -----------------------------------------------------------------------------
# QUICK STATS
# -----------------------------------------------------------------------------

if has_data:
    st.divider()
    st.subheader("Quick Stats")

    # Find notable teams
    top_al = results_df[results_df['League'] == 'AL'].nlargest(1, 'Playoff %')
    top_nl = results_df[results_df['League'] == 'NL'].nlargest(1, 'Playoff %')
    biggest_lock = results_df.nlargest(1, 'Playoff %').iloc[0]
    biggest_longshot = results_df[results_df['Playoff %'] > 0].nsmallest(1, 'Playoff %')

    stat_cols = st.columns(4)

    with stat_cols[0]:
        st.metric(
            label="Simulations",
            value="50,000",
            help="Number of simulated seasons"
        )

    with stat_cols[1]:
        st.metric(
            label="Top AL Team",
            value=top_al.iloc[0]['Team'] if not top_al.empty else "N/A",
            delta=f"{top_al.iloc[0]['Playoff %']:.0f}% playoff" if not top_al.empty else None,
        )

    with stat_cols[2]:
        st.metric(
            label="Top NL Team",
            value=top_nl.iloc[0]['Team'] if not top_nl.empty else "N/A",
            delta=f"{top_nl.iloc[0]['Playoff %']:.0f}% playoff" if not top_nl.empty else None,
        )

    with stat_cols[3]:
        if not biggest_longshot.empty:
            ls = biggest_longshot.iloc[0]
            st.metric(
                label="Longest Shot (>0%)",
                value=ls['Team'],
                delta=f"{ls['Playoff %']:.1f}% playoff",
            )


# -----------------------------------------------------------------------------
# CHARTS (pre-rendered from S3)
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Playoff Probabilities")
st.caption(
    "Stacked bars show wild card (blue), division winner (red), "
    "and first-round bye (orange) probabilities."
)

if has_prob_chart:
    st.image(prob_chart_url, width="stretch")
else:
    st.info("Playoff probability chart not yet available for this season.")

st.divider()
st.subheader("Team Strength")
st.caption(
    "Estimated probability of beating an average team at a neutral site. "
    "Thick bars show likely range; thin bars show wider uncertainty."
)

if has_strength_chart:
    st.image(strength_chart_url, width="stretch")
else:
    st.info("Team strength chart not yet available for this season.")


# -----------------------------------------------------------------------------
# ROOTING GUIDE (game days only)
# -----------------------------------------------------------------------------

rooting_al_url = f"{S3_PLAYOFF_URL}/{selected_season}/latest/rooting_guide_al.png?v={_cache_version}"
rooting_nl_url = f"{S3_PLAYOFF_URL}/{selected_season}/latest/rooting_guide_nl.png?v={_cache_version}"
has_rooting_al = _image_exists(rooting_al_url)
has_rooting_nl = _image_exists(rooting_nl_url)

if has_rooting_al or has_rooting_nl:
    st.divider()
    st.subheader("Today's Rooting Guide")
    st.caption(
        "Who should your team root for today? Each cell shows the team to root for "
        "and the impact on your playoff odds, based on 50,000 simulated seasons."
    )

    rooting_tab_al, rooting_tab_nl = st.tabs(["American League", "National League"])

    with rooting_tab_al:
        if has_rooting_al:
            _, center, _ = st.columns([1, 5, 1])
            with center:
                st.image(rooting_al_url, width="stretch")
        else:
            st.info("No AL rooting guide available today.")

    with rooting_tab_nl:
        if has_rooting_nl:
            _, center, _ = st.columns([1, 5, 1])
            with center:
                st.image(rooting_nl_url, width="stretch")
        else:
            st.info("No NL rooting guide available today.")

    with st.expander("How to read the rooting guide"):
        st.markdown("""
Each cell shows which team to root for in today's games and how much it matters for your
team's playoff chances, based on 50,000 simulated seasons.

- **Green (Your game)** — root for yourselves
- **Red (High, ≥5%)** — this game has a big impact on your playoff odds
- **Orange (Medium, ≥1.5%)** — meaningful impact
- **Yellow (Low, ≥0.75%)** — smaller but real
- **Grey with asterisk (Lesser evil)** — both outcomes hurt, this is the less bad one
- **Dash (—)** — negligible impact
- **BYE impacts** — for top contenders, shows impact on first-round bye probability
        """)


# -----------------------------------------------------------------------------
# DATA TABLE
# -----------------------------------------------------------------------------

if has_data:
    st.divider()
    st.subheader("Full Probability Table")

    # League filter
    league_filter = st.radio(
        "Filter by league",
        options=["Both", "AL", "NL"],
        horizontal=True,
    )

    display_df = results_df.copy()
    if league_filter != "Both":
        display_df = display_df[display_df['League'] == league_filter]

    # Sort by playoff %
    display_df = display_df.sort_values('Playoff %', ascending=False)

    # Format for display
    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            'Team': st.column_config.TextColumn('Team', width='medium'),
            'League': st.column_config.TextColumn('League', width='small'),
            'Division': st.column_config.TextColumn('Division', width='small'),
            'Current Wins': st.column_config.NumberColumn('W', width='small'),
            'Current Losses': st.column_config.NumberColumn('L', width='small'),
            'Playoff %': st.column_config.NumberColumn(
                'Playoff %', format="%.1f%%", width='small'),
            'Division Win %': st.column_config.NumberColumn(
                'Div Win %', format="%.1f%%", width='small'),
            'Bye %': st.column_config.NumberColumn(
                'Bye %', format="%.1f%%", width='small'),
        },
    )

    # Download button
    csv_data = display_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"playoff_probabilities_{selected_season}.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# METHODOLOGY
# -----------------------------------------------------------------------------

st.divider()
with st.expander("Methodology"):
    st.markdown("""
**How playoff probabilities work:**

Win-loss record tells you what happened. We want to know what's *likely to happen next*.

1. **Estimate team strength** — a Bayesian statistical model (one that updates its beliefs
   as new data comes in) looks at each team's deserve-to-win results, run differential,
   and outcomes to estimate how good they actually are, separate from luck.
2. **Simulate 50,000 remaining seasons** — for every unplayed game, the model generates
   a win probability and flips the coin. Repeat 50,000 times.
3. **Apply MLB's playoff rules** — 3 division winners + 3 wild cards per league, top 2
   seeds get first-round byes. Same structure MLB uses.
4. **Count the outcomes** — if a team makes the playoffs in 35,000 of 50,000 simulations,
   that's a 70% playoff probability.

These numbers reflect overall deserve-to-win performance (batted ball quality, walks,
strikeouts, baserunning), not just wins and losses. A 45-40 team that's been crushing
the ball and drawing walks will project better than a 47-38 team running on fumes and
lucky bounces.
    """)

render_home_link()
