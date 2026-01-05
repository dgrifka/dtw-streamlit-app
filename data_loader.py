"""
Data loading utilities for the Streamlit app.
Loads game summaries from public S3 bucket and constructs image URLs.
"""

import pandas as pd
import streamlit as st
from .team_mappings import get_short_name

# Public S3 URLs (no credentials needed)
S3_BASE_URL = "https://dtw-streamlit.s3.amazonaws.com"
GAME_SUMMARIES_URL = f"{S3_BASE_URL}/data/game_summaries.parquet"
IMAGES_BASE_URL = f"{S3_BASE_URL}/sim-images"


@st.cache_data(ttl=3600)  # Cache for 1 hour, then refresh
def load_game_summaries() -> pd.DataFrame:
    """
    Load game summaries from public S3 bucket.
    
    Caching:
        - Data is cached for 1 hour (ttl=3600 seconds)
        - During the season, games update after each game finishes
        - 1 hour refresh ensures reasonably fresh data without hammering S3
    
    Returns:
        DataFrame with columns: home, away, gamePk, date, home_score, away_score,
                               home_wp, away_wp, tie_wp, season
    """
    try:
        df = pd.read_parquet(GAME_SUMMARIES_URL)
        
        # Ensure date is parsed correctly
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        
        # Sort by date descending (most recent first)
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Failed to load game data: {e}")
        return pd.DataFrame()


def build_image_url(row: pd.Series, chart_type: str) -> str:
    """
    Build the S3 image URL for a specific game visualization.
    
    Filename pattern: {away}_{home}_{away_score}-{home_score}--{away_wp}-{home_wp}_{chart_type}.png
    
    Args:
        row: DataFrame row with game data
        chart_type: One of 'spray', 'rd', 'estimated_bases', 'player_contributions'
    
    Example:
        Yankees @ Red Sox, 5-3, 45%-55% win prob
        -> Yankees_Red Sox_5-3--45-55_spray.png
    """
    away_short = get_short_name(row['away'])
    home_short = get_short_name(row['home'])
    
    # Win percentages are stored as 0.0-1.0, convert to int percentage
    away_wp = int(round(row['away_wp'] * 100))
    home_wp = int(round(row['home_wp'] * 100))
    
    filename = f"{away_short}_{home_short}_{row['away_score']}-{row['home_score']}--{away_wp}-{home_wp}_{chart_type}.png"
    
    return f"{IMAGES_BASE_URL}/{row['gamePk']}/{filename}"


def get_game_images(row: pd.Series) -> dict:
    """
    Get all 4 visualization URLs for a game.
    
    Returns:
        Dict with keys: spray, rd, estimated_bases, player_contributions
    """
    chart_types = ['spray', 'rd', 'estimated_bases', 'player_contributions']
    return {ct: build_image_url(row, ct) for ct in chart_types}


def get_deserved_winner(row: pd.Series) -> dict:
    """
    Determine the "deserved winner" based on simulation.
    
    Returns:
        Dict with 'team', 'probability', and 'actual_winner'
    """
    away_wp = row['away_wp']
    home_wp = row['home_wp']
    
    # Deserved winner is team with higher win probability
    if away_wp > home_wp:
        deserved = row['away']
        deserved_prob = away_wp
    else:
        deserved = row['home']
        deserved_prob = home_wp
    
    # Actual winner based on score
    if row['away_score'] > row['home_score']:
        actual = row['away']
    elif row['home_score'] > row['away_score']:
        actual = row['home']
    else:
        actual = "Tie"
    
    return {
        'deserved_winner': deserved,
        'deserved_prob': deserved_prob,
        'actual_winner': actual,
        'was_upset': deserved != actual and actual != "Tie"
    }


def filter_games(
    df: pd.DataFrame,
    teams: list = None,
    start_date=None,
    end_date=None,
    season: int = None,
    upsets_only: bool = False
) -> pd.DataFrame:
    """
    Filter games by various criteria.
    
    Args:
        df: Game summaries DataFrame
        teams: List of team names to filter (matches home OR away)
        start_date: Minimum date
        end_date: Maximum date  
        season: Filter to specific season year
        upsets_only: Only show games where actual winner != deserved winner
    """
    filtered = df.copy()
    
    if teams:
        # Match if team is home OR away
        filtered = filtered[
            filtered['home'].isin(teams) | filtered['away'].isin(teams)
        ]
    
    if start_date:
        filtered = filtered[filtered['date'] >= pd.to_datetime(start_date)]
    
    if end_date:
        filtered = filtered[filtered['date'] <= pd.to_datetime(end_date)]
    
    if season:
        filtered = filtered[filtered['season'] == season]
    
    if upsets_only:
        # Add deserved winner info and filter
        def is_upset(row):
            info = get_deserved_winner(row)
            return info['was_upset']
        filtered = filtered[filtered.apply(is_upset, axis=1)]
    
    return filtered.reset_index(drop=True)
