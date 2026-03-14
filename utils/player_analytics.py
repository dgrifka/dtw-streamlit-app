"""
Player analytics helper functions.

Pure computation — no Streamlit calls. Reuses existing data loaded by data_loader.py.
"""

import numpy as np
import pandas as pd

from .player_helpers import is_barrel_vectorized, TB_MAP


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
