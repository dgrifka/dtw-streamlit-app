"""
Data loading utilities for the Streamlit app.
Loads game summaries from public S3 bucket and constructs image URLs.
"""

import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from .team_mappings import get_short_name

# Public S3 URLs (no credentials needed)
S3_BASE_URL = "https://dtw-streamlit.s3.amazonaws.com"
GAME_SUMMARIES_URL = f"{S3_BASE_URL}/data/game_summaries.parquet"
IMAGES_BASE_URL = f"{S3_BASE_URL}/sim-images"


@st.cache_data(ttl=3600)
def load_game_summaries() -> pd.DataFrame:
    """Load game summaries from public S3 bucket with 1-hour cache."""
    try:
        df = pd.read_parquet(GAME_SUMMARIES_URL)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load game data: {e}")
        return pd.DataFrame()


def build_image_url(row: pd.Series, chart_type: str) -> str:
    """Build the S3 image URL for a specific game visualization."""
    away_short = get_short_name(row['away'])
    home_short = get_short_name(row['home'])
    away_wp = int(round(row['away_wp'] * 100))
    home_wp = int(round(row['home_wp'] * 100))
    filename = f"{away_short}_{home_short}_{row['away_score']}-{row['home_score']}--{away_wp}-{home_wp}_{chart_type}.png"
    return f"{IMAGES_BASE_URL}/{row['gamePk']}/{filename}"


def get_game_images(row: pd.Series) -> dict:
    """Get all 4 visualization URLs for a game."""
    chart_types = ['spray', 'rd', 'estimated_bases', 'player_contributions']
    return {ct: build_image_url(row, ct) for ct in chart_types}


def get_deserved_winner(row: pd.Series) -> dict:
    """Determine the deserved winner based on simulation."""
    away_wp = row['away_wp']
    home_wp = row['home_wp']
    
    if away_wp > home_wp:
        deserved = row['away']
        deserved_prob = away_wp
    else:
        deserved = row['home']
        deserved_prob = home_wp
    
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


@st.cache_data(ttl=3600)
def load_playoff_probabilities(season: int):
    """Load playoff probability results from S3 (1-hour cache)."""
    url = f"{S3_BASE_URL}/playoff-probabilities/{season}/latest/results.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400)
def _image_exists(url: str) -> bool:
    """Check if a URL returns HTTP 200 (cached 24 hours)."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception:
        return False


@st.cache_data(ttl=86400)
def get_available_batted_ball_seasons() -> list[int]:
    """Auto-detect which seasons have batted ball data on S3 (cached 24h)."""
    current_year = pd.Timestamp.now().year
    available = []
    for year in range(current_year, current_year - 5, -1):
        url = f"{S3_BASE_URL}/data/batted_balls_{year}.parquet"
        if _image_exists(url):
            available.append(year)
    return available


@st.cache_data(ttl=86400)
def get_available_player_evaluation_seasons() -> list[int]:
    """Auto-detect which seasons have player evaluation data on S3 (cached 24h)."""
    current_year = pd.Timestamp.now().year
    available = []
    for year in range(current_year, current_year - 5, -1):
        url = f"{S3_BASE_URL}/player-evaluations/{year}/latest/hitter_rankings.parquet"
        if _image_exists(url):
            available.append(year)
    return available


@st.cache_data(ttl=3600)
def load_batted_balls(season: int) -> pd.DataFrame:
    """Load batted ball data from public S3 bucket with 1-hour cache."""
    url = f"{S3_BASE_URL}/data/batted_balls_{season}.parquet"
    try:
        df = pd.read_parquet(url)
        df['date_parsed'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        # Normalize actual_result to title case (MLB API returns lowercase)
        if 'actual_result' in df.columns:
            df['actual_result'] = (
                df['actual_result']
                .str.replace('_', ' ')
                .str.title()
            )
        # Clip extreme probabilities (GBC overfitting edge cases)
        prob_cols = ['out_prob', 'single_prob', 'double_prob', 'triple_prob', 'hr_prob']
        if all(col in df.columns for col in prob_cols):
            for col in prob_cols:
                df[col] = df[col].clip(0.001, 0.999)
            # Renormalize so probabilities sum to 1
            prob_sum = df[prob_cols].sum(axis=1)
            for col in prob_cols:
                df[col] = df[col] / prob_sum
            # Recalculate derived columns from clipped probabilities
            df['estimated_bases'] = (df['single_prob'] * 1 + df['double_prob'] * 2
                                     + df['triple_prob'] * 3 + df['hr_prob'] * 4)
            df['xba'] = (1 - df['out_prob']).round(3)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_available_projection_seasons() -> list[int]:
    """Auto-detect which seasons have projection data on S3 (cached 24h)."""
    current_year = pd.Timestamp.now().year
    available = []
    for year in range(current_year + 1, current_year - 3, -1):
        url = f"{S3_BASE_URL}/player-projections/{year}/latest/hitter_projections.parquet"
        if _image_exists(url):
            available.append(year)
    return available


@st.cache_data(ttl=3600)
def load_player_evaluations(season: int, player_type: str = "hitter") -> pd.DataFrame:
    """Load player evaluation rankings from S3 (1-hour cache).

    Args:
        season: MLB season year
        player_type: 'hitter' or 'pitcher'

    Returns:
        DataFrame with columns: player, team, n_batted_balls, raw_rate,
        posterior_mean, hdi_low, hdi_high, shrinkage
    """
    url = f"{S3_BASE_URL}/player-evaluations/{season}/latest/{player_type}_rankings.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_player_evaluations_pa(season: int, player_type: str = "hitter") -> pd.DataFrame:
    """Load per-plate-appearance player evaluation rankings from S3 (1-hour cache).

    Args:
        season: MLB season year
        player_type: 'hitter' or 'pitcher'

    Returns:
        DataFrame with PA-mode rankings (same schema as batted-ball rankings,
        but n_batted_balls represents plate appearances)
    """
    url = f"{S3_BASE_URL}/player-evaluations/{season}/latest/{player_type}_pa_rankings.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


def get_player_evaluation_image_url(season: int, chart_name: str) -> str:
    """Build URL for a player evaluation chart image on S3."""
    return f"{S3_BASE_URL}/player-evaluations/{season}/latest/{chart_name}.png"


def team_slug(name: str) -> str:
    """Convert team short name to URL-safe slug.

    Examples: 'Yankees' -> 'yankees', 'Blue Jays' -> 'blue_jays',
              'D-backs' -> 'dbacks'
    """
    return name.lower().replace(' ', '_').replace('-', '')


def get_player_evaluation_team_image_url(season: int, team: str, chart_name: str) -> str:
    """Build URL for a team-specific player evaluation chart on S3."""
    slug = team_slug(team)
    return f"{S3_BASE_URL}/player-evaluations/{season}/latest/teams/{slug}/{chart_name}.png"


@st.cache_data(ttl=3600)
def load_player_metadata(season: int) -> pd.DataFrame:
    """Load player metadata (ID, name, team, position, etc.) from S3."""
    url = f"{S3_BASE_URL}/data/player_metadata_{season}.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


def build_player_display_list(
    bb_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    name_col: str = "player",
    team_col: str = "team",
) -> tuple[list[str], dict[str, str], dict[str, str], set[str]]:
    """Build deduplicated player display list, merging traded players.

    For each player name:
    - 1 player_id in metadata → trade; merge to most recent team, one dropdown entry
    - Multiple player_ids → different people with same name; keep separate by team
    - Not in metadata → fall back to keeping separate by team

    Args:
        bb_df: Batted ball DataFrame
        metadata_df: Player metadata DataFrame (may be empty)
        name_col: Column name for player name (e.g. "player" or "pitcher")
        team_col: Column name for team (e.g. "team" or "opponent")

    Returns:
        (display_list, display_to_name, display_to_team, multi_id_names)
    """
    # Get most recent team per player name from batted ball data
    player_recent = (
        bb_df.sort_values("date_parsed")
        .groupby(name_col)[team_col].last()
        .reset_index()
        .rename(columns={name_col: "player", team_col: "recent_team"})
    )

    # Detect same-name collisions via metadata player_id
    multi_id_names: set[str] = set()
    if not metadata_df.empty and "player_id" in metadata_df.columns:
        name_to_ids = metadata_df.groupby("player_name")["player_id"].nunique()
        multi_id_names = set(name_to_ids[name_to_ids > 1].index)

    # For multi-id names, keep separate entries by team
    if multi_id_names:
        collision_entries = (
            bb_df[bb_df[name_col].isin(multi_id_names)]
            .groupby([name_col, team_col]).size()
            .reset_index(name="count").drop(columns="count")
            .rename(columns={name_col: "player", team_col: "recent_team"})
        )
    else:
        collision_entries = pd.DataFrame(columns=["player", "recent_team"])

    # Merge: non-collision players get most recent team; collision players keep per-team entries
    normal_players = player_recent[~player_recent["player"].isin(multi_id_names)]
    player_list = pd.concat([normal_players, collision_entries], ignore_index=True)
    player_list["display"] = player_list.apply(
        lambda r: f"{r['player']} ({r['recent_team']})", axis=1
    )

    display_to_name = dict(zip(player_list["display"], player_list["player"]))
    display_to_team = dict(zip(player_list["display"], player_list["recent_team"]))
    display_list_sorted = sorted(player_list["display"].unique())

    return display_list_sorted, display_to_name, display_to_team, multi_id_names


def resolve_player_id(player_name: str, metadata_df: pd.DataFrame, pa_rankings: pd.DataFrame) -> int | None:
    """Resolve a player_id from metadata or PA rankings parquet.

    Checks metadata first, then falls back to player_id column in rankings
    (available when rankings were generated with PA mode).

    Returns:
        int player_id, or None if not found.
    """
    # Try metadata first
    if not metadata_df.empty:
        match = metadata_df[metadata_df["player_name"] == player_name]
        if not match.empty:
            return int(match.iloc[0]["player_id"])

    # Fallback: player_id from rankings parquet (PA mode includes it)
    if not pa_rankings.empty and "player_id" in pa_rankings.columns:
        match = pa_rankings[pa_rankings["player"] == player_name]
        if not match.empty:
            pid = match.iloc[0]["player_id"]
            if pd.notna(pid):
                return int(pid)

    return None


@st.cache_data(ttl=3600)
def load_pa_counts(season: int) -> pd.DataFrame:
    """Load per-player walk/strikeout counts from S3."""
    url = f"{S3_BASE_URL}/data/pa_counts_{season}.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_player_projections(target_season: int, player_type: str = "hitter") -> pd.DataFrame:
    """Load preseason player projections from S3 (1-hour cache).

    Args:
        target_season: Season being projected (e.g. 2026)
        player_type: 'hitter' or 'pitcher'

    Returns:
        DataFrame with columns: player_id, player, team, projected_eb_pa,
        projected_hdi_low, projected_hdi_high, aging_effect, p_active_next_season, etc.
    """
    url = f"{S3_BASE_URL}/player-projections/{target_season}/latest/{player_type}_projections.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_rate_stat_projections(
    target_season: int, player_type: str = "hitter", rate_type: str = "k_rate"
) -> pd.DataFrame:
    """Load rate stat projections from S3 (1-hour cache).

    Args:
        target_season: Season being projected (e.g. 2026)
        player_type: 'hitter' or 'pitcher'
        rate_type: 'k_rate', 'bb_rate', or 'hr_rate'

    Returns:
        DataFrame with columns: player_id, player, team,
        projected_{rate_type}, projected_{rate_type}_hdi_low/high, etc.
    """
    s3_type = f"{player_type}_{rate_type}"
    url = f"{S3_BASE_URL}/player-projections/{target_season}/latest/{s3_type}_projections.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_season_pa_rankings(player_type: str = "hitter") -> dict:
    """Load PA rankings for all available seasons (cached 1h).

    Returns dict mapping season (int) to DataFrame.
    """
    result = {}
    for s in get_available_player_evaluation_seasons():
        pa_data = load_player_evaluations_pa(s, player_type)
        if not pa_data.empty:
            result[s] = pa_data
    return result


@st.cache_data(ttl=3600)
def compute_league_percentiles(season: int, group_col: str = "player") -> dict:
    """Compute league-wide EV, barrel rate, avg EB, luck, hard-hit%, and sweet-spot% per player (cached 1h).

    Returns dict with keys: ev_by_player, barrel_rates, avg_eb_by_player,
    luck_per_player, hard_hit_pcts, sweet_spot_pcts.
    Each value is a pandas Series indexed by player/pitcher name.
    Returns empty dict if no data.
    """
    from .player_helpers import is_barrel_vectorized, TB_MAP

    bb_df = load_batted_balls(season)
    if bb_df.empty:
        return {}
    is_barrel_col = is_barrel_vectorized(bb_df["launch_speed"], bb_df["launch_angle"])
    actual_tb = bb_df["actual_result"].map(TB_MAP).fillna(0)
    enriched = bb_df.assign(is_barrel=is_barrel_col, _actual_tb=actual_tb)
    grouped = enriched.groupby(group_col)
    return {
        "ev_by_player": grouped["launch_speed"].mean(),
        "barrel_rates": grouped["is_barrel"].mean() * 100,
        "avg_eb_by_player": grouped["estimated_bases"].mean(),
        "luck_per_player": grouped.apply(
            lambda x: x["_actual_tb"].sum() - x["estimated_bases"].sum(),
            include_groups=False,
        ),
        "actual_tb_sums": grouped["_actual_tb"].sum(),
        "expected_tb_sums": grouped["estimated_bases"].sum(),
        "bb_counts": grouped["estimated_bases"].count(),
        "hard_hit_pcts": grouped["launch_speed"].apply(lambda x: (x >= 95).mean() * 100),
        "sweet_spot_pcts": grouped["launch_angle"].apply(
            lambda x: ((x >= 8) & (x <= 32)).mean() * 100
        ),
    }


@st.cache_data(ttl=3600)
def get_cached_radar_data(season: int, player_type: str = "hitter", min_pa: int = 30) -> pd.DataFrame:
    """Compute radar metrics + clustering for all players (cached 1h).

    Returns DataFrame with radar metric columns, percentile columns, and 'archetype' column.
    Returns empty DataFrame if insufficient data.
    """
    from .player_analytics import (
        compute_hitter_radar_metrics, compute_pitcher_radar_metrics,
        cluster_player_archetypes,
    )

    pa_rankings = load_player_evaluations_pa(season, player_type)
    bb_df = load_batted_balls(season)
    if pa_rankings.empty:
        return pd.DataFrame()

    if player_type == "hitter":
        radar_df = compute_hitter_radar_metrics(pa_rankings, bb_df, min_pa=min_pa)
    else:
        radar_df = compute_pitcher_radar_metrics(pa_rankings, bb_df, min_pa=min_pa)

    if not radar_df.empty:
        radar_df = cluster_player_archetypes(radar_df, player_type=player_type)

    return radar_df


def filter_games(
    df: pd.DataFrame,
    teams: list = None,
    start_date=None,
    end_date=None,
    season: int = None,
    upsets_only: bool = False
) -> pd.DataFrame:
    """Filter games by various criteria."""
    filtered = df.copy()
    
    if teams:
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
        def is_upset(row):
            info = get_deserved_winner(row)
            return info['was_upset']
        filtered = filtered[filtered.apply(is_upset, axis=1)]
    
    return filtered.reset_index(drop=True)
