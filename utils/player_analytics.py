"""
Player analytics helper functions.

Pure computation — no Streamlit calls. Reuses existing data loaded by data_loader.py.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .player_helpers import is_barrel_vectorized, TB_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Radar chart axes
# ─────────────────────────────────────────────────────────────────────────────

HITTER_RADAR_AXES = [
    ("Contact Quality", "eb_pa"),
    ("Power", "power"),
    ("Plate Discipline", "discipline"),
    ("Contact Rate", "contact_rate"),
    ("Hard Hit", "hard_hit_rate"),
    ("Speed", "speed"),
]

PITCHER_RADAR_AXES = [
    ("Run Prevention", "run_prevention"),
    ("Strikeout Ability", "k_ability"),
    ("Command", "command"),
    ("HR Prevention", "hr_prevention"),
    ("Weak Contact", "weak_contact"),
    ("Ground Balls", "gb_rate"),
]

# K values determined by EDA (silhouette analysis on 2025 data)
HITTER_K = 7
PITCHER_K = 6


def compute_hitter_radar_metrics(pa_rankings, bb_df, min_pa=30):
    """
    Build the 6-axis radar metric DataFrame for all hitters.

    Returns DataFrame indexed by player with raw metric values + percentile columns.
    """
    df = pa_rankings.copy()
    df = df[df["n_batted_balls"] >= min_pa].copy()
    if df.empty:
        return pd.DataFrame()

    # Contact Quality: prefer true_talent, fallback posterior_mean
    if "true_talent_eb_pa" in df.columns:
        df["eb_pa"] = df["true_talent_eb_pa"].fillna(df["posterior_mean"])
    else:
        df["eb_pa"] = df["posterior_mean"]

    # Power: HR rate
    if "true_talent_hr_rate" in df.columns:
        df["power"] = df["true_talent_hr_rate"].fillna(df["hr_rate_posterior"])
    else:
        df["power"] = df.get("hr_rate_posterior", pd.Series(dtype=float))

    # Plate Discipline: BB rate
    if "true_talent_bb_rate" in df.columns:
        df["discipline"] = df["true_talent_bb_rate"].fillna(df["bb_rate_posterior"])
    else:
        df["discipline"] = df.get("bb_rate_posterior", pd.Series(dtype=float))

    # Contact Rate: 1 - K%
    if "true_talent_k_rate" in df.columns:
        df["contact_rate"] = 1 - df["true_talent_k_rate"].fillna(df["k_rate_posterior"])
    else:
        k_rate = df.get("k_rate_posterior", pd.Series(dtype=float))
        df["contact_rate"] = 1 - k_rate if not k_rate.empty else pd.Series(dtype=float)

    # Hard Hit rate: EV >= 95 mph from batted balls
    if not bb_df.empty:
        bb_df_temp = bb_df.copy()
        bb_df_temp["_is_hard_hit"] = bb_df_temp["launch_speed"] >= 95
        hhr = bb_df_temp.groupby("player")["_is_hard_hit"].mean()
        df["hard_hit_rate"] = df["player"].map(hhr)
    else:
        df["hard_hit_rate"] = np.nan
    df["hard_hit_rate"] = df["hard_hit_rate"].fillna(df["hard_hit_rate"].median())

    # Speed: SB per PA
    if "stolen_bases" in df.columns:
        df["speed"] = df["stolen_bases"].fillna(0) / df["n_batted_balls"]
    else:
        df["speed"] = 0.0

    # Drop rows missing required metrics
    metric_cols = [col for _, col in HITTER_RADAR_AXES]
    df = df.dropna(subset=metric_cols)
    if df.empty:
        return pd.DataFrame()

    # Compute percentiles
    for _, col in HITTER_RADAR_AXES:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Player grade (decoupled from radar chart)
# ─────────────────────────────────────────────────────────────────────────────

# Speed weight for hitter grade: empirically optimized to maximize Spearman
# correlation with OPS on 2025 data (r=0.843 at 10% vs 0.836 at 0%).
HITTER_SPEED_WEIGHT = 0.10


def compute_player_grade(radar_pcts, player_type="hitter"):
    """Compute 0-100 player grade from radar percentiles.

    Hitter: (1 - w) * eb_pa_pct + w * speed_pct  (w = HITTER_SPEED_WEIGHT)
    Pitcher: run_prevention_pct (lower EB/PA allowed = higher grade)

    Parameters
    ----------
    radar_pcts : dict
        {axis_label: percentile_value (0-100)} from get_player_radar_percentiles.
    player_type : str
        "hitter" or "pitcher"

    Returns
    -------
    float or None
        0-100 grade, or None if required axes are missing.
    """
    if not radar_pcts:
        return None

    if player_type == "pitcher":
        val = radar_pcts.get("Run Prevention")
        return val if val is not None else None

    # Hitter: EB/PA-weighted with speed component
    eb_pa = radar_pcts.get("Contact Quality")
    speed = radar_pcts.get("Speed")
    if eb_pa is None:
        return None
    if speed is None:
        return eb_pa
    w = HITTER_SPEED_WEIGHT
    return (1 - w) * eb_pa + w * speed


def compute_pitcher_radar_metrics(pa_rankings, bb_df, min_pa=30):
    """
    Build the 6-axis radar metric DataFrame for all pitchers.

    Returns DataFrame indexed by player with raw metric values + percentile columns.
    """
    df = pa_rankings.copy()
    df = df[df["n_batted_balls"] >= min_pa].copy()
    if df.empty:
        return pd.DataFrame()

    # Run Prevention: EB/PA allowed (lower = better, will invert at percentile stage)
    if "true_talent_eb_pa" in df.columns:
        df["run_prevention"] = df["true_talent_eb_pa"].fillna(df["posterior_mean"])
    else:
        df["run_prevention"] = df["posterior_mean"]

    # Strikeout Ability: K rate
    if "true_talent_k_rate" in df.columns:
        df["k_ability"] = df["true_talent_k_rate"].fillna(df["k_rate_posterior"])
    else:
        df["k_ability"] = df.get("k_rate_posterior", pd.Series(dtype=float))

    # Command: 1 - BB rate
    if "true_talent_bb_rate" in df.columns:
        df["command"] = 1 - df["true_talent_bb_rate"].fillna(df["bb_rate_posterior"])
    else:
        bb_rate = df.get("bb_rate_posterior", pd.Series(dtype=float))
        df["command"] = 1 - bb_rate if not bb_rate.empty else pd.Series(dtype=float)

    # HR Prevention: 1 - HR rate
    if "true_talent_hr_rate" in df.columns:
        df["hr_prevention"] = 1 - df["true_talent_hr_rate"].fillna(df["hr_rate_posterior"])
    else:
        hr_rate = df.get("hr_rate_posterior", pd.Series(dtype=float))
        df["hr_prevention"] = 1 - hr_rate if not hr_rate.empty else pd.Series(dtype=float)

    # Weak Contact: inverted hard hit rate from batted balls
    if not bb_df.empty and "pitcher" in bb_df.columns:
        bb_temp = bb_df.copy()
        bb_temp["_is_hard_hit"] = bb_temp["launch_speed"] >= 95
        hhr = bb_temp.groupby("pitcher")["_is_hard_hit"].mean()
        df["weak_contact"] = 1 - df["player"].map(hhr).fillna(hhr.median())
    else:
        df["weak_contact"] = 0.5

    # Ground Ball rate: LA < 10 degrees
    if not bb_df.empty and "pitcher" in bb_df.columns:
        bb_temp = bb_df.copy()
        bb_temp["_is_gb"] = bb_temp["launch_angle"] < 10
        gbr = bb_temp.groupby("pitcher")["_is_gb"].mean()
        df["gb_rate"] = df["player"].map(gbr).fillna(gbr.median())
    else:
        df["gb_rate"] = 0.5

    metric_cols = [col for _, col in PITCHER_RADAR_AXES]
    df = df.dropna(subset=metric_cols)
    if df.empty:
        return pd.DataFrame()

    # Compute percentiles — run_prevention is inverted (lower EB/PA = higher percentile)
    df["run_prevention_pct"] = (1 - df["run_prevention"].rank(pct=True)) * 100
    for _, col in PITCHER_RADAR_AXES:
        if col != "run_prevention":
            df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    return df


def _get_percentile_matrix(df, axes):
    """Extract percentile columns as numpy array for clustering."""
    pct_cols = [f"{col}_pct" for _, col in axes]
    return df[pct_cols].values


def _name_hitter_cluster(centroid):
    """Assign archetype name based on centroid percentile values."""
    eb, power, disc, contact, hh, speed = centroid
    overall = np.mean(centroid)

    if overall >= 62:
        if contact >= 65 and speed >= 60:
            return "Elite Contact-Speed"
        return "Elite All-Around"
    if power >= 60 and hh >= 60:
        if disc >= 50:
            return "Power Hitter"
        return "Power Slugger"
    if contact >= 60 and speed >= 60:
        if eb >= 45:
            return "Contact-Speed"
        return "Speed Threat"
    if speed >= 60:
        if power >= 40:
            return "Power-Speed"
        return "Speed Threat"
    if disc >= 55 and contact >= 55:
        return "Patient Hitter"
    if contact >= 60:
        return "Contact Hitter"
    if overall <= 30:
        return "Below Average"
    return "Below Average"


def _name_pitcher_cluster(centroid):
    """Assign archetype name based on centroid percentile values."""
    run_prev, k_ability, command, hr_prev, weak, gb = centroid
    overall = np.mean(centroid)

    if overall >= 62:
        if k_ability >= 65:
            return "Dominant"
        return "Elite Command"
    if k_ability >= 65 and run_prev >= 55:
        return "Strikeout Artist"
    if gb >= 65 and hr_prev >= 60:
        return "Ground Ball Machine"
    if command >= 60 and weak >= 55:
        return "Pitch-to-Contact"
    if command >= 55 and gb >= 55:
        return "Finesse Pitcher"
    if command >= 55 and run_prev >= 45:
        return "Command Pitcher"
    if k_ability >= 45 and overall <= 35:
        return "Volatile"
    if overall <= 30:
        return "Below Average"
    return "Below Average"


# Archetype descriptions keyed by name
HITTER_ARCHETYPE_DESC = {
    "Elite All-Around": "Top-tier production across all skill dimensions. These are the most complete hitters in the league.",
    "Elite Contact-Speed": "Elite production with outstanding contact skills and baserunning speed.",
    "Power Hitter": "Combines power with plate discipline to consistently drive the ball.",
    "Power Slugger": "Drives the ball hard with elite power and hard-hit rate, but doesn't draw many walks. Produces through raw hitting ability.",
    "Contact-Speed": "Puts the ball in play consistently and uses above-average speed to create value. A well-rounded offensive contributor.",
    "Power-Speed": "Combines speed and emerging power but strikes out frequently. High-ceiling profile with boom-or-bust at-bats.",
    "Speed Threat": "Gets on base through contact and creates havoc on the basepaths. Limited power but hard to keep off base.",
    "Patient Hitter": "Works counts and earns walks with solid contact ability. Lacks power and speed but contributes through plate discipline.",
    "Contact Hitter": "Puts the ball in play and avoids strikeouts, but limited impact tools.",
    "Below Average": "Below-average production across most skill dimensions this season.",
    "Unknown": "Not enough data to determine an archetype yet.",
}

PITCHER_ARCHETYPE_DESC = {
    "Dominant": "Elite across the board with overpowering stuff and pinpoint command.",
    "Elite Command": "Exceptional command and run prevention with well-rounded skills. Consistently locates pitches and limits damage.",
    "Strikeout Artist": "Misses bats at an elite rate with strong overall run prevention. Overpowers hitters with swing-and-miss stuff.",
    "Ground Ball Machine": "Keeps the ball on the ground and limits home runs effectively. Relies on inducing weak ground-ball contact.",
    "Pitch-to-Contact": "Good command and induces weak contact, but doesn't miss many bats. Relies on location and movement over velocity.",
    "Finesse Pitcher": "Decent command and gets ground balls, but lacks swing-and-miss ability. Relies on guile and location over pure stuff.",
    "Command Pitcher": "Relies on command and pitch-ability over pure stuff.",
    "Volatile": "Has some swing-and-miss ability but walks too many batters. Results are inconsistent due to poor command.",
    "Below Average": "Below-average production across most skill dimensions this season.",
    "Unknown": "Not enough data to determine an archetype yet.",
}


def cluster_player_archetypes(df, player_type="hitter"):
    """
    Run K-Means clustering on the 6D percentile space and assign archetype names.

    Parameters
    ----------
    df : DataFrame
        Output of compute_hitter/pitcher_radar_metrics (must have *_pct columns).
    player_type : str
        "hitter" or "pitcher"

    Returns
    -------
    df with added 'archetype' column.
    """
    if player_type == "hitter":
        axes = HITTER_RADAR_AXES
        k = HITTER_K
        name_fn = _name_hitter_cluster
    else:
        axes = PITCHER_RADAR_AXES
        k = PITCHER_K
        name_fn = _name_pitcher_cluster

    X = _get_percentile_matrix(df, axes)
    if len(X) < k:
        # Too few players for clustering — assign archetypes individually
        df = df.copy()
        df["archetype"] = [name_fn(row) for row in X]
        return df

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Name each cluster from its centroid
    cluster_names = {}
    used_names = set()
    for i, centroid in enumerate(km.cluster_centers_):
        name = name_fn(centroid)
        # Avoid duplicate names by appending a suffix
        if name in used_names:
            name = f"{name} II"
        used_names.add(name)
        cluster_names[i] = name

    df = df.copy()
    df["archetype"] = pd.Series(labels, index=df.index).map(cluster_names)
    return df


def find_similar_players(df, player_name, player_team, player_type="hitter", n=5):
    """
    Find the N most similar players by Euclidean distance in percentile space.

    Returns list of dicts: [{player, team, similarity, archetype}, ...]
    """
    axes = HITTER_RADAR_AXES if player_type == "hitter" else PITCHER_RADAR_AXES
    pct_cols = [f"{col}_pct" for _, col in axes]

    # Find the target player row
    match = df[(df["player"] == player_name) & (df["team"] == player_team)]
    if match.empty:
        match = df[df["player"] == player_name]
    if match.empty:
        return []

    target = match.iloc[0][pct_cols].values.astype(float)

    # Compute distances to all other players
    others = df[~((df["player"] == player_name) & (df["team"] == match.iloc[0]["team"]))].copy()
    if others.empty:
        return []

    other_vecs = others[pct_cols].values.astype(float)
    dists = np.sqrt(((other_vecs - target) ** 2).sum(axis=1))

    # Normalize to 0-100 similarity score (max possible distance = sqrt(6*100^2) ≈ 245)
    max_dist = np.sqrt(len(pct_cols) * 100**2)
    similarities = np.clip(100 - (dists / max_dist * 100), 0, 100)

    others = others.copy()
    others["_dist"] = dists
    others["_similarity"] = similarities
    top = others.nsmallest(n, "_dist")

    return [
        {
            "player": row["player"],
            "team": row["team"],
            "similarity": round(row["_similarity"], 0),
            "archetype": row.get("archetype", ""),
        }
        for _, row in top.iterrows()
    ]


def get_player_radar_percentiles(df, player_name, player_team, player_type="hitter"):
    """
    Get radar percentile values for a single player.

    Returns dict {axis_label: percentile_value} or None.
    """
    axes = HITTER_RADAR_AXES if player_type == "hitter" else PITCHER_RADAR_AXES

    match = df[(df["player"] == player_name) & (df["team"] == player_team)]
    if match.empty:
        match = df[df["player"] == player_name]
    if match.empty:
        return None

    row = match.iloc[0]
    return {label: round(row[f"{col}_pct"], 1) for label, col in axes}


def classify_contact_tier(posterior_mean_pct, barrel_rate, k_rate, bb_rate,
                          league_medians):
    """
    Classify player into fantasy-relevant tier based on Bayesian ranking
    percentile and batted ball profile.

    Tiers:
    - "Elite Masher"   — 95th+ EB/PA AND barrel_rate > 15%
    - "Power Hitter"   — 80th-95th EB/PA
    - "Barrel Hunter"  — 65th-80th EB/PA, above-avg barrels
    - "Contact Machine"— 50th-80th EB/PA, low K%, low barrel%
    - "Average"        — 30th-50th EB/PA
    - "Below Average"  — 10th-30th EB/PA
    - "Struggling"     — <10th EB/PA
    """
    if pd.isna(posterior_mean_pct):
        return "Unknown"

    barrel_rate = barrel_rate if not pd.isna(barrel_rate) else 0
    k_rate = k_rate if not pd.isna(k_rate) else 20
    barrel_med = league_medians.get("barrel_rate_median", 7)
    k_med = league_medians.get("k_rate_median", 20)

    if posterior_mean_pct >= 95 and barrel_rate > 15:
        return "Elite Masher"
    elif posterior_mean_pct >= 80:
        return "Power Hitter"
    elif posterior_mean_pct >= 65:
        return "Barrel Hunter"
    elif posterior_mean_pct >= 50:
        if k_rate < k_med and barrel_rate < barrel_med:
            return "Contact Machine"
        return "Average"
    elif posterior_mean_pct >= 30:
        return "Average"
    elif posterior_mean_pct >= 10:
        return "Below Average"
    else:
        return "Struggling"


def compute_platoon_splits(bb_df, metadata_df, min_bb=15):
    """
    League-wide platoon splits for all hitters.

    Returns DataFrame with per-player vs-LHP and vs-RHP stats,
    filtered to players with >= min_bb batted balls per side.
    """
    if bb_df.empty or metadata_df.empty:
        return pd.DataFrame()

    if "pitcher" not in bb_df.columns or "throw_hand" not in metadata_df.columns:
        return pd.DataFrame()

    bb = bb_df.copy()

    # Map pitcher name to throw hand via metadata
    throw_hand_map = metadata_df.set_index("player_name")["throw_hand"].to_dict()
    bb["pitcher_hand"] = bb["pitcher"].map(throw_hand_map)
    bb = bb.dropna(subset=["pitcher_hand"])
    bb["is_barrel"] = is_barrel_vectorized(bb["launch_speed"], bb["launch_angle"])

    sides = []
    for hand_val, label in [("L", "lhp"), ("R", "rhp")]:
        subset = bb[bb["pitcher_hand"] == hand_val]
        if subset.empty:
            continue
        agg = subset.groupby("player").agg(
            team=("team", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            n=("estimated_bases", "count"),
            eb=("estimated_bases", "mean"),
            ev=("launch_speed", "mean"),
            barrel=("is_barrel", "mean"),
        ).reset_index()
        agg = agg.rename(columns={
            "n": f"vs_{label}_n",
            "eb": f"vs_{label}_eb",
            "ev": f"vs_{label}_ev",
            "barrel": f"vs_{label}_barrel",
        })
        agg[f"vs_{label}_barrel"] = (agg[f"vs_{label}_barrel"] * 100).round(1)
        agg[f"vs_{label}_ev"] = agg[f"vs_{label}_ev"].round(1)
        agg[f"vs_{label}_eb"] = agg[f"vs_{label}_eb"].round(3)
        sides.append(agg)

    if len(sides) != 2:
        return pd.DataFrame()

    merged = sides[0].merge(sides[1], on=["player", "team"], how="outer")

    # Filter for minimum sample size on each side
    merged = merged[
        (merged["vs_lhp_n"].fillna(0) >= min_bb) &
        (merged["vs_rhp_n"].fillna(0) >= min_bb)
    ]

    merged["platoon_gap"] = merged["vs_lhp_eb"] - merged["vs_rhp_eb"]
    avg_eb = merged[["vs_lhp_eb", "vs_rhp_eb"]].mean(axis=1)
    merged["platoon_pct_diff"] = (
        (merged["platoon_gap"] / avg_eb) * 100
    ).round(1)

    return merged.sort_values(
        "platoon_gap", key=abs, ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-generated player highlights
# ─────────────────────────────────────────────────────────────────────────────

def generate_player_highlights(
    radar_pcts,
    archetype_name,
    player_type="hitter",
    deviation=None,
    luck_pct=None,
    recent_vs_season=None,
    platoon_str=None,
    archetype_desc=None,
):
    """Generate 2-4 short, data-driven highlight callouts for a player.

    Parameters
    ----------
    radar_pcts : dict
        {axis_label: percentile_value (0-100)} from get_player_radar_percentiles.
    archetype_name : str
        Player archetype label.
    player_type : str
        "hitter" or "pitcher".
    deviation : float or None
        true_talent_eb_pa - posterior_mean. Positive = true talent above observed.
    luck_pct : float or None
        Luck percentile (0-100).
    recent_vs_season : float or None
        14-day EB/BB minus season average.
    platoon_str : str or None
        Platoon advantage string (e.g., "+0.050 EB/BB vs LHP").
    archetype_desc : str or None
        Archetype description text.

    Returns
    -------
    list of dict
        Each: {"bold": str, "text": str}. Max 4 items.
    """
    if not radar_pcts:
        return []

    candidates = []  # (priority, {"bold": ..., "text": ...})

    # 1. Standout skill — any axis >= 85th percentile
    for label, pct in radar_pcts.items():
        if pct >= 85:
            ordinal = int(pct)
            if player_type == "pitcher":
                candidates.append((1, {
                    "bold": f"Elite {label}.",
                    "text": f"Ranks in the top {100 - ordinal}% of all pitchers in {label.lower()} this season.",
                }))
            else:
                candidates.append((1, {
                    "bold": f"Elite {label}.",
                    "text": f"Ranks in the top {100 - ordinal}% of all hitters in {label.lower()} this season.",
                }))

    # 2. True talent divergence
    if deviation is not None and abs(deviation) > 0.02:
        if player_type == "pitcher":
            if deviation < 0:
                candidates.append((2, {
                    "bold": "Regression candidate (favorable).",
                    "text": "True talent estimate is better than current stats suggest \u2014 expect improvement.",
                }))
            else:
                candidates.append((2, {
                    "bold": "Regression candidate (unfavorable).",
                    "text": "True talent estimate is worse than current stats suggest \u2014 performance may decline.",
                }))
        else:
            if deviation > 0:
                candidates.append((2, {
                    "bold": "Regression candidate (favorable).",
                    "text": "True talent estimate is above current stats \u2014 expect improvement.",
                }))
            else:
                candidates.append((2, {
                    "bold": "Regression candidate (unfavorable).",
                    "text": "True talent estimate is below current stats \u2014 performance may decline.",
                }))

    # 3. Luck context
    if luck_pct is not None:
        if luck_pct < 15:
            if player_type == "pitcher":
                candidates.append((3, {
                    "bold": "Running unlucky.",
                    "text": "Has been one of the unluckiest pitchers this season \u2014 results have been worse than the contact quality suggests.",
                }))
            else:
                candidates.append((3, {
                    "bold": "Running unlucky.",
                    "text": "Has been one of the unluckiest hitters this season \u2014 expect a rebound.",
                }))
        elif luck_pct > 85:
            if player_type == "pitcher":
                candidates.append((3, {
                    "bold": "Running lucky.",
                    "text": "Has been one of the luckiest pitchers this season \u2014 batted ball quality suggests results may worsen.",
                }))
            else:
                candidates.append((3, {
                    "bold": "Running lucky.",
                    "text": "Has been one of the luckiest hitters this season \u2014 results may cool off.",
                }))

    # 4. Hot/cold streak
    if recent_vs_season is not None:
        if player_type == "pitcher":
            # For pitchers, negative = recent is better (lower EB allowed)
            if recent_vs_season < -0.02:
                candidates.append((4, {
                    "bold": "Trending up.",
                    "text": f"Last 14 days: allowing {abs(recent_vs_season):.3f} fewer EB/BB than season average.",
                }))
            elif recent_vs_season > 0.02:
                candidates.append((4, {
                    "bold": "Recent struggles.",
                    "text": f"Last 14 days: allowing {recent_vs_season:.3f} more EB/BB than season average.",
                }))
        else:
            if recent_vs_season > 0.02:
                candidates.append((4, {
                    "bold": "Trending up.",
                    "text": f"Last 14 days: producing {recent_vs_season:.3f} more EB/BB than season average.",
                }))
            elif recent_vs_season < -0.02:
                candidates.append((4, {
                    "bold": "Recent slump.",
                    "text": f"Last 14 days: producing {abs(recent_vs_season):.3f} fewer EB/BB than season average.",
                }))

    # 5. Platoon advantage
    if platoon_str:
        candidates.append((5, {
            "bold": "Platoon advantage.",
            "text": platoon_str if player_type == "hitter" else f"Weakness: {platoon_str}",
        }))

    # Sort by priority, take top 3 (archetype always appended as #4)
    candidates.sort(key=lambda x: x[0])
    highlights = [c[1] for c in candidates[:3]]

    # Always include archetype as final highlight
    if archetype_name and archetype_name != "Unknown":
        desc = archetype_desc or ""
        highlights.append({
            "bold": f"{archetype_name}.",
            "text": desc,
        })

    return highlights[:4]
