"""
Team Rankings Page

Shows season-long luck metrics: which teams have over/under-performed
their expected wins based on the simulation model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
import requests
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_game_summaries
from utils.team_mappings import get_short_name, get_team_logo_url, TEAM_COLORS
from utils.luck_calculations import (
    filter_to_regular_season,
    calculate_luck_metrics,
    calculate_model_accuracy,
    get_extreme_teams,
)

# Page config
st.set_page_config(
    page_title="Team Rankings | DTW Simulator",
    page_icon="🏆",
    layout="wide"
)

# -----------------------------------------------------------------------------
# LOGO CACHING
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400)  # Cache logos for 24 hours
def load_team_logo(logo_url: str) -> np.ndarray | None:
    """
    Download and process a team logo image.
    
    Returns numpy array with transparent background, or None if failed.
    """
    if not logo_url:
        return None
    try:
        response = requests.get(logo_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGBA
        img = img.convert('RGBA')
        data = np.array(img)
        
        # Make white/near-white pixels transparent
        white_pixels = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
        data[white_pixels, 3] = 0
        
        return data
    except Exception as e:
        st.warning(f"Failed to load logo: {e}")
        return None


def get_logo_image(team_short_name: str, zoom: float = 0.055) -> OffsetImage | None:
    """Get an OffsetImage for a team logo, ready to add to a plot."""
    logo_url = get_team_logo_url(team_short_name)
    logo_data = load_team_logo(logo_url)
    if logo_data is not None:
        return OffsetImage(logo_data, zoom=zoom)
    return None


# -----------------------------------------------------------------------------
# CHART FUNCTIONS
# -----------------------------------------------------------------------------

def plot_luck_differential(luck_stats: pd.DataFrame) -> plt.Figure:
    """
    Create bar chart of luck differential with team logos.
    
    Positive values = more wins than expected (lucky)
    Negative values = fewer wins than expected (unlucky)
    """
    # Sort by luck differential (ascending so luckiest teams are on the right)
    df = luck_stats.sort_values('luck_differential', ascending=True).copy()
    df['short_name'] = df['team'].apply(get_short_name)
    
    # Get team colors
    df['color'] = df['short_name'].apply(lambda x: TEAM_COLORS.get(x, ('#333333', '#666666'))[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    # Create bars
    bars = ax.bar(
        range(len(df)),
        df['luck_differential'],
        color=df['color'],
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Calculate positioning
    max_abs_val = max(abs(df['luck_differential'].min()), df['luck_differential'].max())
    if max_abs_val == 0:
        max_abs_val = 1  # Avoid division by zero
    
    # Add team logos at the base of each bar
    for i, (idx, row) in enumerate(df.iterrows()):
        logo = get_logo_image(row['short_name'], zoom=0.045)
        if logo:
            # Position logo based on bar direction
            if row['luck_differential'] >= 0:
                alignment = (0.5, 1.2)  # Below x-axis for positive bars
            else:
                alignment = (0.5, -0.2)  # Above x-axis for negative bars
            
            ab = AnnotationBbox(
                logo,
                (i, 0),
                box_alignment=alignment,
                pad=0,
                frameon=False
            )
            ax.add_artist(ab)
    
    # Styling
    ax.set_title('Luck Differential: Actual Wins vs Expected Wins', fontsize=18, fontweight='bold', pad=15)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    
    # Y-axis
    y_max = int(max_abs_val) + 2
    y_ticks = np.arange(-y_max, y_max + 1, 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y):+d}" for y in y_ticks], fontsize=12, fontweight='bold')
    ax.set_ylim(-max_abs_val * 1.3, max_abs_val * 1.3)
    
    # Remove x-axis ticks (logos serve as labels)
    ax.set_xticks([])
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotations for lucky/unlucky directions
    ax.annotate(
        'More Wins\nThan Expected',
        xy=(len(df) * 0.82, max_abs_val * 0.75),
        fontsize=11,
        color='#008000',
        ha='center',
        va='center',
        fontweight='bold'
    )
    ax.annotate(
        'Fewer Wins\nThan Expected',
        xy=(len(df) * 0.18, -max_abs_val * 0.75),
        fontsize=11,
        color='#8B0000',
        ha='center',
        va='center',
        fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def plot_lucky_vs_unlucky(luck_stats: pd.DataFrame) -> plt.Figure:
    """
    Create scatter plot of lucky wins vs unlucky losses with team logos.
    """
    df = luck_stats.copy()
    df['short_name'] = df['team'].apply(get_short_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Calculate logo size based on data range
    x_range = df['lucky_wins'].max() - df['lucky_wins'].min()
    y_range = df['unlucky_losses'].max() - df['unlucky_losses'].min()
    
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1
    
    logo_size = min(x_range, y_range) * 0.08
    
    # Plot team logos
    for idx, row in df.iterrows():
        logo_url = get_team_logo_url(row['short_name'])
        logo_data = load_team_logo(logo_url)
        
        if logo_data is not None:
            x, y = row['lucky_wins'], row['unlucky_losses']
            x_min = max(0, x - logo_size / 2)
            x_max = x + logo_size / 2
            y_min = max(0, y - logo_size / 2)
            y_max = y + logo_size / 2
            
            # Ensure non-zero extent
            if abs(x_max - x_min) < 0.01:
                x_max += 0.01
            if abs(y_max - y_min) < 0.01:
                y_max += 0.01
            
            ax.imshow(logo_data, extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=10)
        else:
            # Fallback: plot text abbreviation
            ax.text(row['lucky_wins'], row['unlucky_losses'], row['short_name'][:3],
                   ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_title('Lucky Wins vs Unlucky Losses', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Lucky Wins (won as underdog)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Unlucky Losses (lost as favorite)', fontsize=14, fontweight='bold', labelpad=10)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set limits with padding
    ax.set_xlim(left=0, right=df['lucky_wins'].max() * 1.15 + 1)
    ax.set_ylim(bottom=0, top=df['unlucky_losses'].max() * 1.15 + 1)
    
    # Add mean lines
    mean_lucky = df['lucky_wins'].mean()
    mean_unlucky = df['unlucky_losses'].mean()
    ax.axvline(mean_lucky, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(mean_unlucky, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Integer ticks
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Quadrant labels
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.5)
    text_props = dict(fontsize=12, fontweight='bold', fontstyle='italic', color='black', alpha=0.9)
    
    quadrants = {
        (0.85, 0.85): 'Both',
        (0.85, 0.15): 'Just\nLucky',
        (0.15, 0.15): 'Neither',
        (0.15, 0.85): 'Just\nUnlucky'
    }
    
    for (x, y), text in quadrants.items():
        ax.text(x, y, text, transform=ax.transAxes, ha='center', va='center',
               bbox=box_props, **text_props)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------------------------

st.title("🏆 Team Rankings")
st.markdown("Which teams have been **lucky** (more wins than expected) or **unlucky** (fewer wins than expected)?")

# Load data
df = load_game_summaries()

if df.empty:
    st.error("No game data available. Please check the data source.")
    st.stop()

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

st.divider()

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Season filter - for now just 2025
    available_seasons = sorted(df['season'].unique(), reverse=True)
    selected_season = st.selectbox(
        "Season",
        options=available_seasons,
        index=0  # Default to most recent
    )

with col2:
    # Regular season toggle
    regular_season_only = st.checkbox(
        "Regular Season Only",
        value=True,
        help="Exclude playoff games (playoffs started Sept 30, 2025)"
    )

# Apply filters
filtered_df = df[df['season'] == selected_season].copy()

if regular_season_only:
    filtered_df = filter_to_regular_season(filtered_df, selected_season)

# Check if we have data after filtering
if filtered_df.empty:
    st.warning(f"No games found for {selected_season}.")
    st.stop()

# -----------------------------------------------------------------------------
# CALCULATE METRICS
# -----------------------------------------------------------------------------

luck_stats = calculate_luck_metrics(filtered_df)
model_accuracy = calculate_model_accuracy(filtered_df)
extreme_teams = get_extreme_teams(luck_stats)

# -----------------------------------------------------------------------------
# QUICK STATS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("📊 Quick Stats")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric(
        label="Games Analyzed",
        value=f"{model_accuracy['total']:,}"
    )

with stat_col2:
    st.metric(
        label="Model Accuracy",
        value=f"{model_accuracy['accuracy']:.1%}",
        help="How often the team with higher win probability actually won"
    )

with stat_col3:
    luckiest = extreme_teams['luckiest']
    st.metric(
        label="Luckiest Team",
        value=luckiest['team'],
        delta=f"+{luckiest['differential']:.1f} wins"
    )

with stat_col4:
    unluckiest = extreme_teams['unluckiest']
    st.metric(
        label="Unluckiest Team",
        value=unluckiest['team'],
        delta=f"{unluckiest['differential']:.1f} wins"
    )

# -----------------------------------------------------------------------------
# CHARTS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("📈 Luck Differential")
st.caption("Teams on the right have won more games than expected; teams on the left have won fewer.")

fig1 = plot_luck_differential(luck_stats)
st.pyplot(fig1)
plt.close(fig1)

st.divider()
st.subheader("📉 Lucky Wins vs Unlucky Losses")
st.caption("Lucky win = won when underdog (<50% win prob). Unlucky loss = lost when favorite (>50% win prob).")

fig2 = plot_lucky_vs_unlucky(luck_stats)
st.pyplot(fig2)
plt.close(fig2)

# -----------------------------------------------------------------------------
# DATA TABLE
# -----------------------------------------------------------------------------

st.divider()
st.subheader("📋 Full Rankings Table")

# Prepare display dataframe
display_df = luck_stats[[
    'team', 'games_played', 'actual_wins', 'expected_wins',
    'luck_differential', 'lucky_wins', 'unlucky_losses', 'win_pct', 'expected_win_pct'
]].copy()

# Rename columns for display
display_df.columns = [
    'Team', 'Games', 'Actual Wins', 'Expected Wins',
    'Luck Diff', 'Lucky Wins', 'Unlucky Losses', 'Win %', 'Expected Win %'
]

# Format percentages
display_df['Win %'] = (display_df['Win %'] * 100).round(1).astype(str) + '%'
display_df['Expected Win %'] = (display_df['Expected Win %'] * 100).round(1).astype(str) + '%'

# Display table
st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        'Luck Diff': st.column_config.NumberColumn(format="%.1f"),
    }
)

# Download button
csv_data = display_df.to_csv(index=False)
st.download_button(
    label="📥 Download as CSV",
    data=csv_data,
    file_name=f"team_rankings_{selected_season}.csv",
    mime="text/csv"
)

# -----------------------------------------------------------------------------
# METHODOLOGY NOTE
# -----------------------------------------------------------------------------

st.divider()
with st.expander("📖 Methodology"):
    st.markdown("""
    **How luck is calculated:**
    
    1. **Expected Wins**: For each game, our model calculates a win probability for each team 
       based on batted ball quality (launch angle, exit velocity, spray angle). We sum these 
       probabilities across the season to get "expected wins."
    
    2. **Luck Differential**: `Actual Wins - Expected Wins`. Positive means the team won more 
       games than their batted ball quality suggested; negative means they won fewer.
    
    3. **Lucky Win**: A game where the team won despite having <50% win probability.
    
    4. **Unlucky Loss**: A game where the team lost despite having >50% win probability.
    
    **Note**: This measures luck relative to *batted ball outcomes*, not overall team quality. 
    A team could be "lucky" here but still be genuinely good—it just means their wins exceeded 
    what their batted balls would typically produce.
    """)