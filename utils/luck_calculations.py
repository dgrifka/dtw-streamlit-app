"""
Luck calculation utilities for Team Rankings.

Calculates "luck" metrics by comparing actual wins to expected wins
based on the simulation's win probabilities.

Key concepts:
- Expected wins: Sum of win probabilities across all games
- Luck differential: Actual wins - Expected wins
- Lucky win: Won a game where win probability was < 50%
- Unlucky loss: Lost a game where win probability was > 50%
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Playoff start dates by season (regular season ends the day before)
PLAYOFF_START_DATES = {
    2025: datetime(2025, 9, 30),
    2026: datetime(2026, 9, 29),
}


def filter_to_regular_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Filter out playoff and non-regular-season games for a given season.

    Args:
        df: DataFrame with 'date' column
        season: Year to filter (e.g., 2025)

    Returns:
        DataFrame with only regular season games
    """
    result = df.copy()
    # Filter by game_type when available (excludes Spring Training, etc.)
    if 'game_type' in result.columns:
        result = result[result['game_type'] == 'R']
    if season in PLAYOFF_START_DATES:
        cutoff = PLAYOFF_START_DATES[season]
        result = result[result['date'] < cutoff]
    return result


def redistribute_tie_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistribute tie probability equally between home and away win probabilities.
    
    This ensures home_wp + away_wp = 1.0 for each game, which is necessary
    for luck differential to sum to zero across all teams.
    
    Args:
        df: DataFrame with home_wp, away_wp, and optionally tie_wp columns
        
    Returns:
        DataFrame with adjusted home_wp and away_wp columns
    """
    df = df.copy()
    
    # Calculate tie probability if not present
    if 'tie_wp' in df.columns:
        tie_wp = df['tie_wp']
    else:
        # Infer tie probability from the gap
        tie_wp = 1 - (df['home_wp'] + df['away_wp'])
        tie_wp = tie_wp.clip(lower=0)  # Handle floating point errors
    
    # Redistribute tie probability equally
    df['home_wp'] = df['home_wp'] + 0.5 * tie_wp
    df['away_wp'] = df['away_wp'] + 0.5 * tie_wp
    
    return df


def reshape_to_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape game-level data to team-game level.
    
    Each game produces two rows: one for the home team, one for the away team.
    Tie probability is redistributed before reshaping to ensure luck sums to zero.
    
    Args:
        df: Game summaries with home/away columns
        
    Returns:
        DataFrame with one row per team per game, columns:
        - team: Team name
        - gamePk: Game ID
        - date: Game date
        - team_score: Runs scored by this team
        - opponent_score: Runs scored by opponent
        - win_prob: This team's win probability (adjusted for ties)
        - won: 1 if team won, 0 if lost
    """
    # Redistribute tie probability first
    df = redistribute_tie_probability(df)
    
    # Home team perspective
    home_games = df[['gamePk', 'date', 'home', 'home_score', 'away_score', 'home_wp']].copy()
    home_games.columns = ['gamePk', 'date', 'team', 'team_score', 'opponent_score', 'win_prob']
    
    # Away team perspective
    away_games = df[['gamePk', 'date', 'away', 'away_score', 'home_score', 'away_wp']].copy()
    away_games.columns = ['gamePk', 'date', 'team', 'team_score', 'opponent_score', 'win_prob']
    
    # Combine
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    
    # Add win indicator
    all_games['won'] = (all_games['team_score'] > all_games['opponent_score']).astype(int)
    
    return all_games.sort_values(['team', 'date']).reset_index(drop=True)


def calculate_luck_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate luck metrics for each team.
    
    Args:
        df: Game summaries DataFrame with columns:
            home, away, home_wp, away_wp, home_score, away_score, date
            
    Returns:
        DataFrame with one row per team, columns:
        - team: Team name
        - games_played: Number of games
        - actual_wins: Games won
        - expected_wins: Sum of win probabilities
        - luck_differential: actual_wins - expected_wins
        - lucky_wins: Won when win_prob < 0.5
        - unlucky_losses: Lost when win_prob > 0.5
        - win_pct: Actual winning percentage
        - expected_win_pct: Expected winning percentage
    """
    # Reshape to team-game level
    team_games = reshape_to_team_games(df)
    
    # Calculate aggregates per team
    team_stats = team_games.groupby('team').agg(
        games_played=('gamePk', 'count'),
        actual_wins=('won', 'sum'),
        expected_wins=('win_prob', 'sum'),
    ).reset_index()
    
    # Calculate luck differential
    team_stats['luck_differential'] = team_stats['actual_wins'] - team_stats['expected_wins']
    
    # Calculate lucky wins (won as underdog) and unlucky losses (lost as favorite)
    team_games['lucky_win'] = ((team_games['won'] == 1) & (team_games['win_prob'] < 0.5)).astype(int)
    team_games['unlucky_loss'] = ((team_games['won'] == 0) & (team_games['win_prob'] > 0.5)).astype(int)
    
    luck_counts = team_games.groupby('team').agg(
        lucky_wins=('lucky_win', 'sum'),
        unlucky_losses=('unlucky_loss', 'sum'),
    ).reset_index()
    
    # Merge luck counts
    team_stats = team_stats.merge(luck_counts, on='team')
    
    # Add percentages
    team_stats['win_pct'] = team_stats['actual_wins'] / team_stats['games_played']
    team_stats['expected_win_pct'] = team_stats['expected_wins'] / team_stats['games_played']
    
    # Round for cleaner display
    team_stats['expected_wins'] = team_stats['expected_wins'].round(1)
    team_stats['luck_differential'] = team_stats['luck_differential'].round(1)
    
    # Sort by luck differential (luckiest first)
    team_stats = team_stats.sort_values('luck_differential', ascending=False).reset_index(drop=True)
    
    return team_stats


def calculate_model_accuracy(df: pd.DataFrame) -> dict:
    """
    Calculate how often the model's favorite actually won.
    
    Args:
        df: Game summaries DataFrame
        
    Returns:
        Dict with accuracy stats:
        - correct: Number of games where favorite won
        - total: Total games
        - accuracy: Percentage correct
    """
    # Determine predicted and actual winners
    home_favored = df['home_wp'] > df['away_wp']
    home_won = df['home_score'] > df['away_score']
    
    # Correct when: (home favored AND home won) OR (away favored AND away won)
    correct = ((home_favored & home_won) | (~home_favored & ~home_won)).sum()
    total = len(df)
    
    return {
        'correct': int(correct),
        'total': int(total),
        'accuracy': correct / total if total > 0 else 0
    }


def get_extreme_teams(luck_stats: pd.DataFrame) -> dict:
    """
    Get the luckiest and unluckiest teams.
    
    Args:
        luck_stats: Output from calculate_luck_metrics()
        
    Returns:
        Dict with luckiest/unluckiest team info
    """
    luckiest_idx = luck_stats['luck_differential'].idxmax()
    unluckiest_idx = luck_stats['luck_differential'].idxmin()
    
    return {
        'luckiest': {
            'team': luck_stats.loc[luckiest_idx, 'team'],
            'differential': luck_stats.loc[luckiest_idx, 'luck_differential'],
        },
        'unluckiest': {
            'team': luck_stats.loc[unluckiest_idx, 'team'],
            'differential': luck_stats.loc[unluckiest_idx, 'luck_differential'],
        }
    }