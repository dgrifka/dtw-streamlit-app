"""
Player Rankings

Bayesian hierarchical rankings of MLB hitters and pitchers by estimated
bases, with credible intervals and shrinkage for small sample sizes.
"""

import unicodedata
import urllib.parse

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (
    load_player_evaluations,
    load_player_evaluations_pa,
    load_player_metadata,
    load_player_projections,
    load_batted_balls,
    get_player_evaluation_image_url,
    get_player_evaluation_team_image_url,
    get_available_player_evaluation_seasons,
    get_available_projection_seasons,
)
from utils.team_mappings import TEAM_COLORS
from utils.player_analytics import compute_platoon_splits
from utils.player_helpers import PLOTLY_CONFIG
from utils.responsive import inject_responsive_css, render_home_link

inject_responsive_css()

st.title("Player Rankings")
st.markdown(
    "Bayesian statistical rankings that separate signal from noise. "
    "Players with small samples get pulled toward the league average; "
    "players with lots of data keep estimates close to their raw numbers."
)


def _normalize(text):
    """Strip accents and lowercase for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _build_forest_plot(plot_df, color, label, metric_short="EB/PA",
                       mean_col="posterior_mean", low_col="hdi_low",
                       high_col="hdi_high", low_50_col="hdi_50_low",
                       high_50_col="hdi_50_high", count_col="n_batted_balls",
                       count_label="PA"):
    """Build a Plotly forest plot showing credible intervals."""
    plot_df = plot_df.sort_values(mean_col, ascending=True).copy()

    # Disambiguate duplicate player names by appending team (or age if same team)
    dup_names = plot_df["player"].duplicated(keep=False)
    if dup_names.any() and "team" in plot_df.columns:
        plot_df.loc[dup_names, "player"] = (
            plot_df.loc[dup_names, "player"] + " (" + plot_df.loc[dup_names, "team"] + ")"
        )
        # If still duplicated (same name + same team), add age
        still_dup = plot_df["player"].duplicated(keep=False)
        if still_dup.any():
            age_col = "age_at_projection" if "age_at_projection" in plot_df.columns else "Age" if "Age" in plot_df.columns else None
            if age_col:
                plot_df.loc[still_dup, "player"] = (
                    plot_df.loc[still_dup, "player"].str.rstrip(")")
                    + ", age " + plot_df.loc[still_dup, age_col].astype(int).astype(str) + ")"
                )
    fig = go.Figure()

    has_50_hdi = low_50_col in plot_df.columns and high_50_col in plot_df.columns

    # 89% HDI thin lines
    for _, row in plot_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[low_col], row[high_col]],
            y=[row["player"], row["player"]],
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 50% HDI thick lines (if available)
    if has_50_hdi and plot_df[low_50_col].notna().any():
        for _, row in plot_df.iterrows():
            if pd.notna(row.get(low_50_col)) and pd.notna(row.get(high_50_col)):
                fig.add_trace(go.Scatter(
                    x=[row[low_50_col], row[high_50_col]],
                    y=[row["player"], row["player"]],
                    mode="lines",
                    line=dict(color=color, width=5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Mean dots
    _hover_cols = [low_col, high_col, count_col]
    _hover_team = "team" in plot_df.columns
    if _hover_team:
        _hover_cols = _hover_cols + ["team"]
    fig.add_trace(go.Scatter(
        x=plot_df[mean_col],
        y=plot_df["player"],
        mode="markers",
        marker=dict(color=color, size=8),
        name=label,
        customdata=plot_df[_hover_cols].values,
        hovertemplate=(
            "<b>%{y}</b>" + (" (%{customdata[3]})" if _hover_team else "") + "<br>"
            f"{metric_short}: %{{x:.3f}}<br>"
            "Range: %{customdata[0]:.3f} – %{customdata[1]:.3f}<br>"
            f"{count_label}: %{{customdata[2]:.0f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"Est. Bases / {metric_short.split('/')[-1]}",
        height=max(300, len(plot_df) * 28),
        margin=dict(l=140, r=20, t=10, b=40),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# =============================================================================
# CONTROLS (top-level — these affect the chart image)
# =============================================================================

col_season, col_type, col_metric, col_team = st.columns([1, 1, 1, 1])

with col_season:
    eval_seasons = get_available_player_evaluation_seasons()
    proj_seasons = get_available_projection_seasons()
    current_year = pd.Timestamp.now().year
    # Only include projection-only seasons for current year and next year
    # (further-out projections are useful on timeline charts, not as standalone pages)
    all_seasons_set = set(eval_seasons)
    for ps in proj_seasons:
        if ps <= current_year + 1:
            all_seasons_set.add(ps)
    available_seasons = sorted(all_seasons_set, reverse=True)

    # Build display labels: mark projection-only seasons
    eval_seasons_set = set(eval_seasons)
    season_labels = []
    for s in available_seasons:
        if s not in eval_seasons_set:
            season_labels.append(f"{s} (Projected)")
        else:
            season_labels.append(str(s))

    if available_seasons:
        # Default to current year if available, otherwise first in list
        default_idx = 0
        if current_year in available_seasons:
            default_idx = available_seasons.index(current_year)
        season_label = st.selectbox("Season", options=season_labels, index=default_idx)
        season = int(season_label.split()[0])
    else:
        season = current_year

with col_type:
    player_type = st.selectbox("Player type", options=["Hitter", "Pitcher"])

with col_metric:
    metric_mode = st.selectbox("Metric", options=["Per Plate Appearance", "Per Batted Ball"])

with col_team:
    all_teams = sorted(TEAM_COLORS.keys())
    selected_team = st.selectbox("Team", options=["All Teams"] + all_teams)

type_key = player_type.lower()
is_pa_mode = metric_mode == "Per Plate Appearance"
is_pitcher = type_key == "pitcher"
metric_short = "EB/PA" if is_pa_mode else "EB/BB"
count_label = "Plate Appearances" if is_pa_mode else "Batted Balls"

# Load rankings data
if is_pa_mode:
    df = load_player_evaluations_pa(season, type_key)
else:
    df = load_player_evaluations(season, type_key)

# Load projections (for PA mode, hitters only — projections are EB/PA-based)
proj_df = pd.DataFrame()
if is_pa_mode and type_key == "hitter":
    proj_df = load_player_projections(season, "hitter")

has_eval_data = not df.empty
has_proj_data = not proj_df.empty

if not has_eval_data and not has_proj_data:
    st.info(
        f"No {type_key} evaluation data available for {season}. "
        f"Data is generated weekly during the season."
    )
    st.stop()

# Load metadata (used by both tabs)
metadata_df = load_player_metadata(season)
# For projection-only seasons, try the most recent eval season's metadata
if metadata_df.empty and has_proj_data and not has_eval_data:
    for fallback_season in eval_seasons[:3]:
        metadata_df = load_player_metadata(fallback_season)
        if not metadata_df.empty:
            break

# Merge position into eval data
if has_eval_data:
    if not metadata_df.empty and "position" in metadata_df.columns:
        meta_slim = metadata_df[["player_name", "position"]].drop_duplicates(
            subset=["player_name"]
        ).rename(columns={"player_name": "player"})
        df = df.merge(meta_slim, on="player", how="left")
        df["position"] = df["position"].fillna("")
    else:
        df["position"] = ""

# Merge position + age into projection data
if has_proj_data and not metadata_df.empty:
    if "position" in metadata_df.columns:
        meta_pos = metadata_df[["player_name", "position"]].drop_duplicates(
            subset=["player_name"]
        ).rename(columns={"player_name": "player"})
        proj_df = proj_df.merge(meta_pos, on="player", how="left")
        proj_df["position"] = proj_df["position"].fillna("")
    else:
        proj_df["position"] = ""


# =============================================================================
# TAB LAYOUT — show projections tab when available
# =============================================================================

show_tabs = has_proj_data and is_pa_mode and type_key == "hitter"

if show_tabs:
    # Offseason (no eval data): projections first; in-season: rankings first
    if has_eval_data:
        tab_rankings, tab_projections = st.tabs(["Season Rankings", "Preseason Projections"])
    else:
        tab_projections, tab_rankings = st.tabs(["Preseason Projections", "Season Rankings"])
else:
    # No projections — everything renders at top level
    tab_rankings = st
    tab_projections = None


# =============================================================================
# SEASON RANKINGS TAB (existing content)
# =============================================================================

with tab_rankings:
    if not has_eval_data:
        st.info(
            f"No in-season evaluation data for {season} yet. "
            f"Check the Preseason Projections tab for projected rankings."
        )
    else:
        # Check if true talent data exists
        has_true_talent_data = "true_talent_eb_pa" in df.columns and df["true_talent_eb_pa"].notna().any()

        # SECTION 1: HOW IT WORKS
        with st.expander("How does this work?"):
            if is_pa_mode and has_true_talent_data:
                st.markdown("""
**True Talent Estimates** combine two independent sources of information:

1. **Preseason projection** — a Bayesian model trained on multiple years of historical
   performance, with an aging curve and selection bias correction
2. **In-season evaluation** — the current season's Bayesian hierarchical model

These are combined using inverse-variance weighting: early in the season, the preseason
projection dominates (it has more data behind it). As the season progresses and in-season
sample sizes grow, the current-season data takes over.

**Key columns:**
- **True Talent EB/PA** — the combined best estimate (used for ranking)
- **Est. Bases (Season)** — current-season model estimate only
- **Preseason Proj.** — preseason projection only
- **Deviation** — raw rate minus true talent (positive = overperforming, likely to regress)
""")
            elif is_pa_mode:
                st.markdown("""
**Per Plate Appearance mode** measures overall offensive production, not just contact quality.
Each plate appearance is valued: walks = 1 base, HBP = 1 base, strikeouts = 0 bases, and
batted balls use the model's estimated bases (based on exit velocity, launch angle, and
spray angle).

**Why not just use raw stats?**

A player hitting .350 through 80 plate appearances might be the real deal, or might be
riding a hot streak. The Bayesian model accounts for sample size by pulling small samples
toward the league average, while players with 400+ PA keep estimates close to their raw
numbers. Think of it as a confidence-weighted average: more data means more trust in the
individual player's numbers.

**Reading the chart:**

- **Circle** — the model's best estimate of true production
- **Thick line** — likely range (50% credible interval)
- **Thin line** — wider uncertainty (89% credible interval)

**Does shrinkage actually help?**

We tested it. Using end-of-2024 data to predict 2025 outcomes (min 100 PA both seasons):
the Bayesian estimate beat the raw rate at predicting next-year EB/PA (R² = 0.140 vs 0.136).
Modest, but that's the nature of year-over-year hitting — it's noisy, and every edge counts.
""")
            else:
                st.markdown("""
**Per Batted Ball mode** measures contact quality only — how well a player hits the ball
when they put it in play, based on exit velocity, launch angle, and spray angle.

**Why not just use raw averages?**

A player with 20 batted balls and a high average might just be on a hot streak. The model
recognizes the small sample and pulls the estimate toward the league average. A player with
400+ batted balls keeps an estimate much closer to their raw numbers, because there's enough
data to trust it.

**Reading the chart:**

- **Circle** — the model's best estimate of true contact quality
- **Thick line** — likely range (50% credible interval)
- **Thin line** — wider uncertainty (89% credible interval)
- Wider ranges = more uncertainty (fewer batted balls)
""")

        # SECTION 2: S3 MATPLOTLIB CHART + DOWNLOAD
        chart_name = f"top_{type_key}s_pa" if is_pa_mode else f"top_{type_key}s"

        if selected_team != "All Teams":
            chart_url = get_player_evaluation_team_image_url(season, selected_team, chart_name)
        else:
            chart_url = get_player_evaluation_image_url(season, chart_name)

        _chart_loaded = False
        try:
            st.image(chart_url, width="stretch")
            _chart_loaded = True
        except Exception:
            if selected_team != "All Teams":
                fallback_url = get_player_evaluation_image_url(season, chart_name)
                try:
                    st.image(fallback_url, width="stretch")
                    chart_url = fallback_url
                    _chart_loaded = True
                except Exception:
                    st.warning(f"Chart image not available for {season}.")
            else:
                st.warning(f"Chart image not available for {season}.")

        if _chart_loaded:
            st.markdown(
                f"[Download chart image]({chart_url})",
                help="Right-click the link or the image above to save/copy.",
            )

        # SECTION 3: RANKINGS TABLE
        st.divider()
        st.subheader(f"{player_type} rankings table")

        # Filters for the table (search, position, min count)
        col_search, col_pos, col_min_bb = st.columns([1, 1, 1])

        with col_search:
            search_query = st.text_input(
                "Search player",
                placeholder="e.g. Ohtani, Juan, Suarez",
                key="eval_search",
            )

        with col_pos:
            position_options = ["All"]
            if type_key == "hitter":
                position_options += ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
            else:
                position_options += ["SP", "RP"]
            position_filter = st.selectbox("Position", position_options, key="eval_pos")

        with col_min_bb:
            default_min = 100 if is_pa_mode else 30
            min_bb = st.slider(
                f"Min {count_label.lower()}",
                min_value=1,
                max_value=int(df["n_batted_balls"].max()),
                value=min(default_min, int(df["n_batted_balls"].max())),
                step=10,
                key="eval_min_count",
            )

        # Apply filters
        filtered = df[df["n_batted_balls"] >= min_bb].copy()
        if selected_team != "All Teams":
            filtered = filtered[filtered["team"] == selected_team]
        if position_filter != "All":
            filtered = filtered[
                filtered["position"].str.contains(position_filter, case=False, na=False)
            ]
        if search_query.strip():
            query_norm = _normalize(search_query.strip())
            filtered = filtered[
                filtered["player"].apply(lambda name: query_norm in _normalize(name))
            ]

        has_true_talent = "true_talent_eb_pa" in filtered.columns and filtered["true_talent_eb_pa"].notna().any()
        sort_col = "true_talent_eb_pa" if has_true_talent else "posterior_mean"

        filtered = filtered.sort_values(
            sort_col, ascending=is_pitcher
        ).reset_index(drop=True)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Total {player_type}s", f"{len(filtered):,}")
        if has_true_talent:
            m2.metric("Avg true talent", f"{filtered['true_talent_eb_pa'].mean():.3f}")
        else:
            m2.metric("Avg est. bases", f"{filtered['posterior_mean'].mean():.3f}")
        m3.metric("Avg raw rate", f"{filtered['raw_rate'].mean():.3f}")
        m4.metric(f"Avg {count_label.lower()}", f"{filtered['n_batted_balls'].mean():.0f}")

        display = filtered.copy()

        # Drop chart-only columns before display
        drop_cols = ["hdi_50_low", "hdi_50_high", "true_talent_hdi_low",
                     "true_talent_hdi_high", "history_weight"]
        for col in drop_cols:
            if col in display.columns:
                display = display.drop(columns=[col])

        display.index = range(1, len(display) + 1)
        display.index.name = "Rank"

        n_col_name = count_label
        rename_map = {
            "player": "Player",
            "team": "Team",
            "position": "Pos",
            "n_batted_balls": n_col_name,
            "posterior_mean": "Est. Bases (Season)",
            "raw_rate": "Est. Bases (Raw)",
            "hdi_low": "Range Low",
            "hdi_high": "Range High",
            "shrinkage": "Adjustment",
        }
        if has_true_talent:
            rename_map["true_talent_eb_pa"] = "True Talent EB/PA"
            rename_map["deviation"] = "Deviation"
            rename_map["preseason_eb_pa"] = "Preseason Proj."
        display = display.rename(columns=rename_map)

        _profile_page = "Pitcher_Profile" if is_pitcher else "Hitter_Profile"
        display["Profile"] = display["Player"].apply(
            lambda p: f"/{_profile_page}?player={urllib.parse.quote(p)}"
        )

        if has_true_talent:
            table_cols = [
                "Player", "Team", "Pos", "True Talent EB/PA", "Est. Bases (Season)",
                "Preseason Proj.", "Deviation", "Range Low", "Range High",
                n_col_name, "Profile",
            ]
        else:
            table_cols = [
                "Player", "Team", "Pos", "Est. Bases (Season)", "Est. Bases (Raw)",
                "Range Low", "Range High", "Adjustment", n_col_name, "Profile",
            ]
        table_cols = [c for c in table_cols if c in display.columns]

        COLUMN_CONFIG = {
            "True Talent EB/PA": st.column_config.NumberColumn(format="%.4f"),
            "Est. Bases (Season)": st.column_config.NumberColumn(format="%.4f"),
            "Est. Bases (Raw)": st.column_config.NumberColumn(format="%.4f"),
            "Preseason Proj.": st.column_config.NumberColumn(format="%.4f"),
            "Deviation": st.column_config.NumberColumn(format="%+.4f"),
            "Range Low": st.column_config.NumberColumn(format="%.4f"),
            "Range High": st.column_config.NumberColumn(format="%.4f"),
            "Adjustment": st.column_config.NumberColumn(format="%+.4f"),
            n_col_name: st.column_config.NumberColumn(format="%d"),
            "Pos": st.column_config.TextColumn(width="small"),
            "Profile": st.column_config.LinkColumn(display_text="View"),
        }

        st.dataframe(
            display[table_cols],
            width="stretch",
            column_config=COLUMN_CONFIG,
        )

        csv_data = display[table_cols].to_csv(index=True)
        st.download_button(
            "Download CSV",
            csv_data,
            f"player_rankings_{type_key}_{season}.csv",
            "text/csv",
            key="eval_csv",
        )

        # Regression candidates (when true talent available)
        if has_true_talent and type_key == "hitter" and is_pa_mode:
            with_dev = filtered[filtered["deviation"].notna()].copy()
            if len(with_dev) >= 20:
                st.divider()
                st.subheader("Regression candidates")
                st.caption(
                    "Players whose raw performance deviates most from their true talent "
                    "estimate. Positive deviation = overperforming (likely to regress down); "
                    "negative = underperforming (likely to bounce back)."
                )
                reg_up, reg_down = st.columns(2)
                with reg_up:
                    st.markdown("##### Due to cool down")
                    overperformers = with_dev.nlargest(10, "deviation")[
                        ["player", "team", "raw_rate", "true_talent_eb_pa", "deviation", "n_batted_balls"]
                    ].copy()
                    overperformers = overperformers.rename(columns={
                        "player": "Player", "team": "Team",
                        "raw_rate": "Raw", "true_talent_eb_pa": "True Talent",
                        "deviation": "Deviation", "n_batted_balls": "PA",
                    })
                    st.dataframe(
                        overperformers, hide_index=True, use_container_width=True,
                        column_config={
                            "Raw": st.column_config.NumberColumn(format="%.3f"),
                            "True Talent": st.column_config.NumberColumn(format="%.3f"),
                            "Deviation": st.column_config.NumberColumn(format="%+.3f"),
                            "PA": st.column_config.NumberColumn(format="%d"),
                        },
                    )
                with reg_down:
                    st.markdown("##### Due to bounce back")
                    underperformers = with_dev.nsmallest(10, "deviation")[
                        ["player", "team", "raw_rate", "true_talent_eb_pa", "deviation", "n_batted_balls"]
                    ].copy()
                    underperformers = underperformers.rename(columns={
                        "player": "Player", "team": "Team",
                        "raw_rate": "Raw", "true_talent_eb_pa": "True Talent",
                        "deviation": "Deviation", "n_batted_balls": "PA",
                    })
                    st.dataframe(
                        underperformers, hide_index=True, use_container_width=True,
                        column_config={
                            "Raw": st.column_config.NumberColumn(format="%.3f"),
                            "True Talent": st.column_config.NumberColumn(format="%.3f"),
                            "Deviation": st.column_config.NumberColumn(format="%+.3f"),
                            "PA": st.column_config.NumberColumn(format="%d"),
                        },
                    )

        # SECTION 4: PLAYER SPOTLIGHT (Hitters, PA mode only)
        if type_key == "hitter" and is_pa_mode and df["hdi_high"].notna().any():
            st.divider()
            st.subheader("Player spotlight")
            st.caption(
                "Estimated Bases per Plate Appearance (EB/PA) — accounts for all plate "
                "appearance outcomes including walks, strikeouts, and batted ball quality."
            )

            # Compute both standout groups
            standouts_pool = df[df["n_batted_balls"] >= 30].copy()
            if selected_team != "All Teams":
                standouts_pool = standouts_pool[standouts_pool["team"] == selected_team]

            upside = pd.DataFrame()
            if standouts_pool["hdi_high"].notna().any():
                elite_threshold = standouts_pool["hdi_high"].quantile(0.80)
                moderate_threshold = standouts_pool["posterior_mean"].quantile(0.70)
                upside = standouts_pool[
                    (standouts_pool["hdi_high"] >= elite_threshold) &
                    (standouts_pool["posterior_mean"] < moderate_threshold)
                ].sort_values("hdi_high", ascending=False).head(15)

            safe_floor = pd.DataFrame()
            with_hdi = filtered[filtered["hdi_high"].notna()].copy()
            if not with_hdi.empty:
                with_hdi["hdi_width"] = with_hdi["hdi_high"] - with_hdi["hdi_low"]
                above_avg = with_hdi[
                    with_hdi["posterior_mean"] >= with_hdi["posterior_mean"].median()
                ]
                safe_floor = above_avg.nsmallest(15, "hdi_width")

            has_50_hdi = "hdi_50_low" in df.columns and "hdi_50_high" in df.columns

            if not upside.empty or not safe_floor.empty:
                tab_upside, tab_floor = st.tabs(["High Upside (Small Samples)", "Reliable Floor (Established)"])

                with tab_upside:
                    st.caption(
                        "Players with limited plate appearances whose ceiling is elite even though "
                        "their current estimate is moderate. High variability means high risk — "
                        "but also breakout potential."
                    )
                    if not upside.empty:
                        fig_upside = _build_forest_plot(upside, "#2563eb", "High Upside",
                                                        metric_short=metric_short)
                        st.plotly_chart(fig_upside, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

                        up_display = upside[["player", "team", "posterior_mean",
                                              "hdi_low", "hdi_high", "n_batted_balls"]].copy()
                        up_display["Profile"] = up_display["player"].apply(
                            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                        )
                        up_display = up_display.rename(columns={
                            "player": "Player", "team": "Team",
                            "posterior_mean": metric_short, "hdi_low": "Floor",
                            "hdi_high": "Ceiling", "n_batted_balls": "PA",
                        })
                        st.dataframe(
                            up_display, hide_index=True, use_container_width=True,
                            column_config={
                                metric_short: st.column_config.NumberColumn(format="%.3f"),
                                "Floor": st.column_config.NumberColumn(format="%.3f"),
                                "Ceiling": st.column_config.NumberColumn(format="%.3f"),
                                "PA": st.column_config.NumberColumn(format="%d"),
                                "Profile": st.column_config.LinkColumn(display_text="View"),
                            },
                        )
                    else:
                        st.info("No high-upside players found with current filters.")

                with tab_floor:
                    st.caption(
                        "Above-average players with enough data that the model is confident in their "
                        "estimate. Narrow ranges mean consistent, predictable production."
                    )
                    if not safe_floor.empty:
                        fig_floor = _build_forest_plot(safe_floor, "#16a34a", "Reliable Floor",
                                                       metric_short=metric_short)
                        st.plotly_chart(fig_floor, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

                        sf_display = safe_floor[["player", "team", "posterior_mean",
                                                  "hdi_low", "hdi_high", "n_batted_balls"]].copy()
                        sf_display["Profile"] = sf_display["player"].apply(
                            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                        )
                        sf_display = sf_display.rename(columns={
                            "player": "Player", "team": "Team",
                            "posterior_mean": metric_short, "hdi_low": "Floor",
                            "hdi_high": "Ceiling", "n_batted_balls": "PA",
                        })
                        st.dataframe(
                            sf_display, hide_index=True, use_container_width=True,
                            column_config={
                                metric_short: st.column_config.NumberColumn(format="%.3f"),
                                "Floor": st.column_config.NumberColumn(format="%.3f"),
                                "Ceiling": st.column_config.NumberColumn(format="%.3f"),
                                "PA": st.column_config.NumberColumn(format="%d"),
                                "Profile": st.column_config.LinkColumn(display_text="View"),
                            },
                        )
                    else:
                        st.info("No reliable-floor players found with current filters.")

                hdi_caption = "Thin lines = possible range, dots = model estimate"
                if has_50_hdi:
                    hdi_caption = "Thin lines = possible range, thick lines = likely range, dots = model estimate"
                st.caption(hdi_caption)
            else:
                st.info("Bayesian ranking data not available.")

        # SECTION 5: PLATOON ADVANTAGE FINDER (hitters only)
        if type_key == "hitter":
            st.divider()
            st.subheader("Platoon advantage finder")
            st.caption(
                "Hitters with the biggest performance gap vs left-handed or "
                "right-handed pitching. Based on batted ball data."
            )

            bb_df = load_batted_balls(season)

            if bb_df.empty:
                st.info(f"No batted ball data available for {season}.")
            elif metadata_df.empty:
                st.info("Player metadata not available (need pitcher throw hand).")
            else:
                col_plat_pos, col_plat_min = st.columns([1, 1])
                with col_plat_pos:
                    plat_pos = st.selectbox(
                        "Position (platoon)",
                        ["All", "C", "1B", "2B", "SS", "3B", "OF", "DH"],
                        key="platoon_pos",
                    )
                with col_plat_min:
                    plat_min_bb = st.slider(
                        "Min BB per side", 10, 50, 15, step=5, key="platoon_min"
                    )

                platoon_df = compute_platoon_splits(bb_df, metadata_df, min_bb=plat_min_bb)

                if platoon_df.empty:
                    st.info("Platoon data not available (need pitcher metadata for throw hand).")
                else:
                    if plat_pos != "All" and not metadata_df.empty:
                        pos_players = metadata_df[
                            metadata_df["position"].str.contains(
                                plat_pos, case=False, na=False
                            )
                        ]["player_name"].tolist()
                        platoon_df = platoon_df[platoon_df["player"].isin(pos_players)]

                    if platoon_df.empty:
                        st.info("No players match the platoon filters.")
                    else:
                        top_plat = platoon_df.head(15).copy()
                        top_plat = top_plat.sort_values("platoon_gap", key=abs, ascending=True)

                        fig_dumb = go.Figure()

                        for _, row in top_plat.iterrows():
                            fig_dumb.add_trace(go.Scatter(
                                x=[row["vs_lhp_eb"], row["vs_rhp_eb"]],
                                y=[row["player"], row["player"]],
                                mode="lines",
                                line=dict(color="#d1d5db", width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            ))

                        fig_dumb.add_trace(go.Scatter(
                            x=top_plat["vs_lhp_eb"],
                            y=top_plat["player"],
                            mode="markers",
                            marker=dict(color="#2563eb", size=10),
                            name="vs LHP",
                            customdata=top_plat["vs_lhp_n"].astype(int).values,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "vs LHP: %{x:.3f} EB/BB<br>"
                                "n=%{customdata}<extra></extra>"
                            ),
                        ))

                        fig_dumb.add_trace(go.Scatter(
                            x=top_plat["vs_rhp_eb"],
                            y=top_plat["player"],
                            mode="markers",
                            marker=dict(color="#dc2626", size=10),
                            name="vs RHP",
                            customdata=top_plat["vs_rhp_n"].astype(int).values,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "vs RHP: %{x:.3f} EB/BB<br>"
                                "n=%{customdata}<extra></extra>"
                            ),
                        ))

                        fig_dumb.update_layout(
                            template="plotly_white",
                            xaxis_title="EB/BB",
                            height=max(400, len(top_plat) * 30),
                            margin=dict(l=140, r=20, t=20, b=40),
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1,
                            ),
                            yaxis=dict(tickfont=dict(size=11)),
                        )
                        st.plotly_chart(fig_dumb, use_container_width=True, theme=None)

                        plat_display = platoon_df.head(30).copy()
                        plat_display["Profile"] = plat_display["player"].apply(
                            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                        )
                        plat_display = plat_display.rename(columns={
                            "player": "Player",
                            "team": "Team",
                            "vs_lhp_eb": "vs LHP (EB/BB)",
                            "vs_rhp_eb": "vs RHP (EB/BB)",
                            "vs_lhp_n": "vs LHP n",
                            "vs_rhp_n": "vs RHP n",
                            "vs_lhp_ev": "vs LHP EV",
                            "vs_rhp_ev": "vs RHP EV",
                            "vs_lhp_barrel": "vs LHP Barrel%",
                            "vs_rhp_barrel": "vs RHP Barrel%",
                            "platoon_gap": "Gap",
                            "platoon_pct_diff": "Gap %",
                        })

                        plat_cols = [
                            "Player", "Team", "vs LHP (EB/BB)", "vs RHP (EB/BB)", "Gap",
                            "vs LHP n", "vs RHP n", "vs LHP EV", "vs RHP EV",
                            "vs LHP Barrel%", "vs RHP Barrel%", "Profile",
                        ]
                        plat_cols = [c for c in plat_cols if c in plat_display.columns]

                        st.dataframe(
                            plat_display[plat_cols],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "vs LHP (EB/BB)": st.column_config.NumberColumn(format="%.3f"),
                                "vs RHP (EB/BB)": st.column_config.NumberColumn(format="%.3f"),
                                "Gap": st.column_config.NumberColumn(format="%.3f"),
                                "vs LHP EV": st.column_config.NumberColumn(format="%.1f"),
                                "vs RHP EV": st.column_config.NumberColumn(format="%.1f"),
                                "vs LHP Barrel%": st.column_config.NumberColumn(format="%.1f%%"),
                                "vs RHP Barrel%": st.column_config.NumberColumn(format="%.1f%%"),
                                "Profile": st.column_config.LinkColumn(display_text="View"),
                            },
                        )


# =============================================================================
# PRESEASON PROJECTIONS TAB
# =============================================================================

if tab_projections is not None:
    with tab_projections:
        with st.expander("How do projections work?"):
            st.markdown(f"""
**Preseason projections** use a Bayesian hierarchical model trained on multiple years of
player evaluation data to project each hitter's estimated bases per plate appearance (EB/PA)
for {season}.

The model accounts for:
- **Historical performance** — each player's EB/PA across up to 5 prior seasons
- **Aging curve** — estimated from within-player year-over-year changes (not cross-sectional
  averages, which suffer from survivorship bias)
- **Sample size** — players with fewer seasons of data have wider uncertainty ranges
- **Selection bias** — the model estimates the probability each player is active

**Reading the chart:**
- **Circle** — projected EB/PA (best estimate)
- **Thin line** — 89% credible interval (possible range)
- **Thick line** — 50% credible interval (likely range)

Projections are most useful before the season starts. Once in-season data accumulates,
the Season Rankings tab combines projection priors with current performance for a
"true talent" estimate.
""")

        # Filters
        col_psearch, col_ppos, col_pteam, col_pmin = st.columns([1.5, 1, 1, 1])

        with col_psearch:
            proj_search = st.text_input(
                "Search player",
                placeholder="e.g. Ohtani, Juan, Suarez",
                key="proj_search",
            )

        with col_ppos:
            proj_pos_options = ["All"]
            if "position" in proj_df.columns:
                proj_pos_options += ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
            proj_pos_filter = st.selectbox("Position", proj_pos_options, key="proj_pos")

        with col_pteam:
            proj_team = selected_team  # reuse top-level team filter

        with col_pmin:
            max_seasons = int(proj_df["n_seasons"].max()) if "n_seasons" in proj_df.columns else 5
            proj_min_seasons = st.slider(
                "Min seasons of data",
                min_value=1,
                max_value=max(max_seasons, 1),
                value=1,
                step=1,
                key="proj_min_seasons",
            )

        # Apply projection filters
        proj_filtered = proj_df.copy()
        if "n_seasons" in proj_filtered.columns:
            proj_filtered = proj_filtered[proj_filtered["n_seasons"] >= proj_min_seasons]
        if proj_team != "All Teams":
            proj_filtered = proj_filtered[proj_filtered["team"] == proj_team]
        if proj_pos_filter != "All" and "position" in proj_filtered.columns:
            proj_filtered = proj_filtered[
                proj_filtered["position"].str.contains(proj_pos_filter, case=False, na=False)
            ]
        if proj_search.strip():
            pq = _normalize(proj_search.strip())
            proj_filtered = proj_filtered[
                proj_filtered["player"].apply(lambda n: pq in _normalize(n))
            ]

        proj_filtered = proj_filtered.sort_values(
            "projected_eb_pa", ascending=False
        ).reset_index(drop=True)

        # Summary metrics
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Players", f"{len(proj_filtered):,}")
        pm2.metric("Avg Projected EB/PA", f"{proj_filtered['projected_eb_pa'].mean():.3f}")
        if "age_at_projection" in proj_filtered.columns:
            pm3.metric("Avg Age", f"{proj_filtered['age_at_projection'].mean():.1f}")

        # Projections table
        proj_display = proj_filtered.copy()
        proj_display.index = range(1, len(proj_display) + 1)
        proj_display.index.name = "Rank"

        proj_rename = {
            "player": "Player",
            "team": "Team",
            "position": "Pos",
            "age_at_projection": "Age",
            "projected_eb_pa": "Projected EB/PA",
            "projected_hdi_low": "Range Low",
            "projected_hdi_high": "Range High",
            "most_recent_eb_pa": "Recent EB/PA",
            "aging_effect": "Aging Effect",
            "n_seasons": "Seasons",
            "p_active_next_season": "P(Active)",
        }
        proj_display = proj_display.rename(columns=proj_rename)

        # Convert P(Active) from 0-1 to percentage for display
        if "P(Active)" in proj_display.columns:
            proj_display["P(Active)"] = proj_display["P(Active)"] * 100

        proj_display["Profile"] = proj_display["Player"].apply(
            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
        )

        proj_table_cols = [
            "Player", "Team", "Pos", "Age", "Projected EB/PA",
            "Range Low", "Range High", "Recent EB/PA", "Aging Effect",
            "Seasons", "P(Active)", "Profile",
        ]
        proj_table_cols = [c for c in proj_table_cols if c in proj_display.columns]

        PROJ_COL_CONFIG = {
            "Projected EB/PA": st.column_config.NumberColumn(format="%.4f"),
            "Range Low": st.column_config.NumberColumn(format="%.4f"),
            "Range High": st.column_config.NumberColumn(format="%.4f"),
            "Recent EB/PA": st.column_config.NumberColumn(format="%.4f"),
            "Aging Effect": st.column_config.NumberColumn(format="%+.4f"),
            "Age": st.column_config.NumberColumn(format="%.1f"),
            "Seasons": st.column_config.NumberColumn(format="%d"),
            "P(Active)": st.column_config.NumberColumn(format="%.0f%%"),
            "Pos": st.column_config.TextColumn(width="small"),
            "Profile": st.column_config.LinkColumn(display_text="View"),
        }

        st.dataframe(
            proj_display[proj_table_cols],
            width="stretch",
            column_config=PROJ_COL_CONFIG,
        )

        proj_csv = proj_display[proj_table_cols].to_csv(index=True)
        st.download_button(
            "Download CSV",
            proj_csv,
            f"player_projections_{season}.csv",
            "text/csv",
            key="proj_csv",
        )

        # Forest plot of top 20 projections
        st.divider()
        st.subheader("Top 20 projected hitters")

        top_20_proj = proj_filtered.head(20).copy()
        if not top_20_proj.empty:
            fig_proj = _build_forest_plot(
                top_20_proj, "#7c3aed", f"Top {len(top_20_proj)} Projected",
                metric_short="EB/PA",
                mean_col="projected_eb_pa",
                low_col="projected_hdi_low",
                high_col="projected_hdi_high",
                low_50_col="projected_hdi_50_low",
                high_50_col="projected_hdi_50_high",
                count_col="n_seasons",
                count_label="Seasons",
            )
            st.plotly_chart(fig_proj, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

            has_50 = ("projected_hdi_50_low" in top_20_proj.columns
                      and top_20_proj["projected_hdi_50_low"].notna().any())
            caption = "Thin lines = possible range, dots = projected EB/PA"
            if has_50:
                caption = "Thin lines = possible range, thick lines = likely range, dots = projected EB/PA"
            st.caption(caption)


# =============================================================================
# FOOTER
# =============================================================================

render_home_link()
