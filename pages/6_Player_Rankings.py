"""
Player Rankings

Bayesian hierarchical rankings of MLB hitters and pitchers by estimated
bases, with credible intervals and shrinkage for small sample sizes.
"""

import contextlib
import unicodedata
import urllib.parse

import requests
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
    load_pa_counts,
    get_player_evaluation_image_url,
    get_player_evaluation_team_image_url,
    get_available_player_evaluation_seasons,
    get_available_projection_seasons,
    get_cached_radar_data,
)
from utils.team_mappings import TEAM_COLORS, get_team_logo_url
from utils.player_analytics import compute_platoon_splits
from utils.player_helpers import PLOTLY_CONFIG_FOREST
from utils.responsive import inject_responsive_css, render_home_link

MLB_LOGO_URL = "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png"

inject_responsive_css()

st.title("Player Rankings")
st.markdown(
    "Bayesian statistical rankings that separate signal from noise. "
    "Players with small samples get pulled toward the league average; "
    "players with lots of data keep estimates close to their raw numbers."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_image_bytes(url: str) -> bytes | None:
    """Fetch image bytes from URL, returning None on 404/error."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception:
        return None


def _normalize(text):
    """Strip accents and lowercase for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _build_forest_plot(plot_df, color, label, metric_short="EB/PA",
                       mean_col="posterior_mean", low_col="hdi_low",
                       high_col="hdi_high", low_50_col="hdi_50_low",
                       high_50_col="hdi_50_high", count_col="n_batted_balls",
                       count_label="PA", sort_ascending=True,
                       use_team_colors=False, league_mean=None,
                       title=None, secondary_col=None,
                       secondary_name=None, secondary_color="#f59e0b",
                       secondary_symbol="diamond", secondary_size=7):
    """Build a Plotly forest plot showing credible intervals."""
    plot_df = plot_df.sort_values(mean_col, ascending=sort_ascending).copy()

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

    # Truncate long names for compact display; full name stays in hover tooltip
    plot_df["_display_name"] = plot_df["player"].apply(
        lambda n: (n[:18] + "...") if len(n) > 18 else n
    )

    # Build per-player colors from team when requested
    if use_team_colors and "team" in plot_df.columns:
        _player_colors = [
            TEAM_COLORS.get(row["team"], ('#888888', '#888888'))[0]
            for _, row in plot_df.iterrows()
        ]
    else:
        _player_colors = None

    fig = go.Figure()

    has_50_hdi = low_50_col in plot_df.columns and high_50_col in plot_df.columns

    # 89% HDI thin lines
    for i, (_, row) in enumerate(plot_df.iterrows()):
        _line_color = _player_colors[i] if _player_colors else color
        fig.add_trace(go.Scatter(
            x=[row[low_col], row[high_col]],
            y=[row["_display_name"], row["_display_name"]],
            mode="lines",
            line=dict(color=_line_color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 50% HDI thick lines (if available)
    if has_50_hdi and plot_df[low_50_col].notna().any():
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if pd.notna(row.get(low_50_col)) and pd.notna(row.get(high_50_col)):
                _line_color = _player_colors[i] if _player_colors else color
                fig.add_trace(go.Scatter(
                    x=[row[low_50_col], row[high_50_col]],
                    y=[row["_display_name"], row["_display_name"]],
                    mode="lines",
                    line=dict(color=_line_color, width=5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Mean dots — include full name in customdata for hover tooltip
    _hover_cols = ["player", low_col, high_col, count_col]
    _hover_team = "team" in plot_df.columns
    if _hover_team:
        _hover_cols = _hover_cols + ["team"]
    _dot_colors = _player_colors if _player_colors else color
    fig.add_trace(go.Scatter(
        x=plot_df[mean_col],
        y=plot_df["_display_name"],
        mode="markers",
        marker=dict(color=_dot_colors, size=8),
        name=label,
        customdata=plot_df[_hover_cols].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b>" + (" (%{customdata[4]})" if _hover_team else "") + "<br>"
            f"{metric_short}: %{{x:.3f}}<br>"
            "Range: %{customdata[1]:.3f} – %{customdata[2]:.3f}<br>"
            f"{count_label}: %{{customdata[3]:.0f}}<extra></extra>"
        ),
    ))

    # Optional secondary markers (e.g. current EB/PA alongside projected)
    _has_secondary = (
        secondary_col is not None
        and secondary_col in plot_df.columns
        and plot_df[secondary_col].notna().any()
    )
    if _has_secondary:
        _sec_name = secondary_name or secondary_col
        fig.add_trace(go.Scatter(
            x=plot_df[secondary_col],
            y=plot_df["_display_name"],
            mode="markers",
            marker=dict(symbol=secondary_symbol, size=secondary_size,
                        color=secondary_color,
                        line=dict(width=1, color="#92400e")),
            name=_sec_name,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{_sec_name}: " + "%{x:.3f}<extra></extra>"
            ),
        ))

    # Compute x-axis range from all plotted values with padding
    _range_vals = pd.concat([
        plot_df[low_col].dropna(),
        plot_df[high_col].dropna(),
        plot_df[mean_col].dropna(),
    ])
    if _has_secondary:
        _range_vals = pd.concat([_range_vals, plot_df[secondary_col].dropna()])
    if has_50_hdi and plot_df[low_50_col].notna().any():
        _range_vals = pd.concat([_range_vals, plot_df[low_50_col].dropna(), plot_df[high_50_col].dropna()])
    x_min = _range_vals.min()
    x_max = _range_vals.max()
    if league_mean is not None:
        x_min = min(x_min, league_mean)
        x_max = max(x_max, league_mean)
    x_pad = (x_max - x_min) * 0.06
    _xaxis_cfg = dict(
        range=[x_min - x_pad, x_max + x_pad],
        tickfont=dict(size=12),
        gridcolor="rgba(0,0,0,0.12)",
        gridwidth=1,
    )

    _show_legend = bool(_has_secondary)
    _top_margin = 50 if (_has_secondary and title) else 40 if title else 25
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=15, color="#1a1a1a"), x=0.5, xanchor="center") if title else None,
        xaxis_title=f"Est. Bases / {metric_short.split('/')[-1]}",
        height=max(300, len(plot_df) * 30),
        margin=dict(l=10, r=20, t=_top_margin, b=40),
        showlegend=_show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ) if _show_legend else None,
        yaxis=dict(tickfont=dict(size=12), automargin=True),
        xaxis=_xaxis_cfg,
    )

    # League mean reference line
    if league_mean is not None:
        fig.add_vline(
            x=league_mean, line_dash="dot", line_color="#888888", line_width=2,
            annotation_text="Lg avg", annotation_position="top right",
            annotation_font_size=13, annotation_font_color="#888888",
        )

    return fig


def _build_contributors_chart(contrib_df, title="", subtitle=""):
    """Build a Plotly stacked horizontal bar chart of total offensive contributions."""
    contrib_df = contrib_df.sort_values("total_bases", ascending=True).copy()
    n = len(contrib_df)

    # Flat green batted ball bars — matches weekly contributors social post
    bb_color = "#2E8B57"

    display_names = (contrib_df["player"] + "  (" + contrib_df["team"] + ")").tolist()

    fig = go.Figure()

    # Batted ball bases segment
    fig.add_trace(go.Bar(
        x=contrib_df["batted_ball_bases"].values,
        y=display_names,
        orientation="h",
        name="Batted Ball Bases",
        marker=dict(color=bb_color),
        customdata=contrib_df[["player", "team", "batted_ball_bases", "walk_bases", "total_bases"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Batted Ball: %{customdata[2]:.1f}<br>"
            "Walks: %{customdata[3]:.0f}<br>"
            "Total: %{customdata[4]:.1f}<extra></extra>"
        ),
    ))

    # Walk bases segment
    fig.add_trace(go.Bar(
        x=contrib_df["walk_bases"].values,
        y=display_names,
        orientation="h",
        name="Walks",
        marker=dict(color="#4169E1"),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Batted Ball: %{customdata[2]:.1f}<br>"
            "Walks: %{customdata[3]:.0f}<br>"
            "Total: %{customdata[4]:.1f}<extra></extra>"
        ),
        customdata=contrib_df[["player", "team", "batted_ball_bases", "walk_bases", "total_bases"]].values,
    ))

    # Total labels at end of bars
    fig.add_trace(go.Scatter(
        x=contrib_df["total_bases"].values + (contrib_df["total_bases"].max() * 0.02),
        y=display_names,
        mode="text",
        text=[f"{v:.0f}" for v in contrib_df["total_bases"].values],
        textposition="middle right",
        textfont=dict(size=11, color="#555"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Build title with optional subtitle
    title_text = title
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#888'>{subtitle}</span>"

    top_margin = 60 if title else 10
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        title=dict(
            text=title_text, font=dict(size=15, color="#1a1a1a"),
            x=0.5, xanchor="center", y=0.98,
        ) if title else None,
        height=max(400, n * 28 + top_margin),
        margin=dict(l=10, r=50, t=top_margin, b=40),
        xaxis=dict(
            title="Total Estimated Bases",
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(tickfont=dict(size=11), automargin=True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
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

# Compute archetypes for PA mode rankings (cached)
_archetype_map = {}
if is_pa_mode and has_eval_data:
    _radar_df = get_cached_radar_data(season, player_type=type_key, min_pa=30)
    if not _radar_df.empty:
        _archetype_map = dict(zip(
            _radar_df["player"] + "|" + _radar_df["team"],
            _radar_df["archetype"],
        ))
    df["archetype"] = (df["player"] + "|" + df["team"]).map(_archetype_map).fillna("")

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

# Merge archetypes into projection data (reuse _archetype_map from PA mode eval)
if has_proj_data and _archetype_map:
    proj_df["archetype"] = (proj_df["player"] + "|" + proj_df["team"]).map(_archetype_map).fillna("")
elif has_proj_data:
    proj_df["archetype"] = ""


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
    # No projections — everything renders at top level (nullcontext for `with` block)
    tab_rankings = contextlib.nullcontext()
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
        # Logo badge floated over chart top-right (no extra whitespace)
        _logo_url = get_team_logo_url(selected_team) if selected_team != "All Teams" else MLB_LOGO_URL
        _logo_label = selected_team if selected_team != "All Teams" else "MLB"

        # Helper to display a chart image from S3
        def _show_chart_image(chart_name_key):
            st.markdown(
                f'<div style="position:relative;height:0;overflow:visible;z-index:1;">'
                f'<img src="{_logo_url}" alt="{_logo_label}" '
                f'style="position:absolute;right:8px;top:0;height:64px;width:64px;object-fit:contain;" '
                f'onerror="this.style.display=\'none\'">'
                f'</div>',
                unsafe_allow_html=True,
            )
            if selected_team != "All Teams":
                _url = get_player_evaluation_team_image_url(season, selected_team, chart_name_key)
            else:
                _url = get_player_evaluation_image_url(season, chart_name_key)

            _bytes = _fetch_image_bytes(_url)
            if _bytes is None and selected_team != "All Teams":
                _fb_url = get_player_evaluation_image_url(season, chart_name_key)
                _bytes = _fetch_image_bytes(_fb_url)
                if _bytes is not None:
                    _url = _fb_url

            if _bytes is not None:
                st.image(_bytes, use_container_width=True)
                st.markdown(
                    f"[Download chart image]({_url})",
                    help="Right-click the link or the image above to save/copy.",
                )
            else:
                st.info(f"Chart not yet available for {season}. Check back as more games are played.")

        chart_name = f"top_{type_key}s_pa" if is_pa_mode else f"top_{type_key}s"

        # Sub-tabs for performance vs projections (hitter PA mode with true talent data)
        if has_true_talent_data and is_pa_mode and type_key == "hitter":
            sub_performance, sub_projections = st.tabs([
                f"{season} Performance", "End-of-Season Projections",
            ])
            with sub_performance:
                _show_chart_image(chart_name)
            with sub_projections:
                _show_chart_image(f"top_{type_key}s_pa_projections")
        else:
            _show_chart_image(chart_name)

        # SECTION 3: RANKINGS TABLE
        st.divider()
        st.subheader(f"{player_type} rankings table")

        # Filters for the table (search, position, archetype, min count)
        _has_archetypes = "archetype" in df.columns and df["archetype"].str.len().gt(0).any()
        if _has_archetypes:
            col_search, col_pos, col_arch, col_min_bb = st.columns([1, 1, 1, 1])
        else:
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

        archetype_filter = "All"
        if _has_archetypes:
            with col_arch:
                _arch_vals = sorted(df.loc[df["archetype"].str.len() > 0, "archetype"].unique())
                archetype_filter = st.selectbox("Archetype", ["All"] + _arch_vals, key="eval_arch")

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
        if archetype_filter != "All":
            filtered = filtered[filtered["archetype"] == archetype_filter]
        if search_query.strip():
            query_norm = _normalize(search_query.strip())
            filtered = filtered[
                filtered["player"].apply(lambda name: query_norm in _normalize(name))
            ]

        has_true_talent = "true_talent_eb_pa" in filtered.columns and filtered["true_talent_eb_pa"].notna().any()
        has_k_rate = "k_rate_posterior" in filtered.columns and filtered["k_rate_posterior"].notna().any()
        has_bb_rate = "bb_rate_posterior" in filtered.columns and filtered["bb_rate_posterior"].notna().any()
        has_hr_rate = "hr_rate_posterior" in filtered.columns and filtered["hr_rate_posterior"].notna().any()
        has_pqs = is_pitcher and "pitcher_quality_score" in filtered.columns and filtered["pitcher_quality_score"].notna().any()

        # Sort options — PQS first for pitchers when available
        sort_options = []
        if has_pqs:
            sort_options.append("PQS+ (higher is better)")
        sort_options.append("EB/PA" if is_pa_mode else "EB/BB")
        if has_k_rate:
            sort_options.append("K% (low is better)" if not is_pitcher else "K% (high is better)")
        if has_bb_rate:
            sort_options.append("BB%")
        if has_hr_rate:
            sort_options.append("HR%")

        if len(sort_options) > 1:
            sort_by = st.radio("Sort by", sort_options, horizontal=True, key="eval_sort")
        else:
            sort_by = sort_options[0]

        if sort_by.startswith("PQS"):
            sort_col = "pitcher_quality_score"
            sort_asc = False  # Higher PQS+ = better pitcher
        elif sort_by.startswith("K%"):
            sort_col = "k_rate_posterior"
            # For hitters, low K% is good (ascending); for pitchers, high K% is good (descending)
            sort_asc = not is_pitcher
        elif sort_by == "BB%":
            sort_col = "bb_rate_posterior"
            # For hitters, high BB% is good (descending); for pitchers, low BB% is good (ascending)
            sort_asc = is_pitcher
        elif sort_by == "HR%":
            sort_col = "hr_rate_posterior"
            # For hitters, high HR% is good (descending); for pitchers, low HR% is good (ascending)
            sort_asc = is_pitcher
        elif has_true_talent:
            sort_col = "true_talent_eb_pa"
            sort_asc = is_pitcher
        else:
            sort_col = "posterior_mean"
            sort_asc = is_pitcher

        filtered = filtered.sort_values(
            sort_col, ascending=sort_asc
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
                     "true_talent_hdi_high", "history_weight",
                     "k_rate_hdi_low", "k_rate_hdi_high",
                     "bb_rate_hdi_low", "bb_rate_hdi_high",
                     "hr_rate_hdi_low", "hr_rate_hdi_high",
                     "pqs_hdi_low", "pqs_hdi_high"]
        for col in drop_cols:
            if col in display.columns:
                display = display.drop(columns=[col])

        # Convert rate stats to percentages for display
        for rc in ["k_rate_raw", "k_rate_posterior", "bb_rate_raw", "bb_rate_posterior",
                    "hr_rate_raw", "hr_rate_posterior"]:
            if rc in display.columns:
                display[rc] = display[rc] * 100

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
        if has_pqs:
            rename_map["pitcher_quality_score"] = "PQS+"
        if is_pitcher and "xeb_pa" in display.columns:
            rename_map["xeb_pa"] = "xEB/PA"
        if has_k_rate:
            rename_map["k_rate_posterior"] = "K%"
            rename_map["k_rate_raw"] = "K% (Raw)"
        if has_bb_rate:
            rename_map["bb_rate_posterior"] = "BB%"
            rename_map["bb_rate_raw"] = "BB% (Raw)"
        if has_hr_rate:
            rename_map["hr_rate_posterior"] = "HR%"
            rename_map["hr_rate_raw"] = "HR% (Raw)"
        # Archetype column
        if "archetype" in display.columns and display["archetype"].str.len().gt(0).any():
            rename_map["archetype"] = "Archetype"
        # Traditional stats (only show if columns exist in parquet)
        has_trad_stats = False
        if not is_pitcher and "avg" in display.columns:
            rename_map["avg"] = "AVG"
            rename_map["home_runs"] = "HR"
            rename_map["stolen_bases"] = "SB"
            has_trad_stats = True
        elif is_pitcher and "era" in display.columns:
            rename_map["era"] = "ERA"
            rename_map["whip"] = "WHIP"
            has_trad_stats = True
        display = display.rename(columns=rename_map)

        _profile_page = "Pitcher_Profile" if is_pitcher else "Hitter_Profile"
        display["Profile"] = display["Player"].apply(
            lambda p: f"/{_profile_page}?player={urllib.parse.quote(p)}"
        )

        if has_true_talent:
            table_cols = [
                "Player", "Team", "Pos", "Archetype", "True Talent EB/PA", "Est. Bases (Season)",
                "Preseason Proj.", "Deviation",
                "K%", "BB%", "HR%", "PQS+",
                n_col_name,
            ]
            if is_pitcher and "xEB/PA" in display.columns:
                table_cols.insert(table_cols.index("Est. Bases (Season)") + 1, "xEB/PA")
        else:
            table_cols = [
                "Player", "Team", "Pos", "Archetype", "Est. Bases (Season)", "Est. Bases (Raw)",
                "K%", "BB%", "HR%", "PQS+",
                "Adjustment", n_col_name,
            ]
            if is_pitcher and "xEB/PA" in display.columns:
                table_cols.insert(table_cols.index("Est. Bases (Season)") + 1, "xEB/PA")
        # Insert traditional stats before Profile link
        if has_trad_stats:
            if not is_pitcher:
                table_cols.extend(["AVG", "HR", "SB"])
            else:
                table_cols.extend(["ERA", "WHIP"])
        table_cols.append("Profile")
        table_cols = [c for c in table_cols if c in display.columns]

        COLUMN_CONFIG = {
            "True Talent EB/PA": st.column_config.NumberColumn(
                format="%.4f",
                help="Combined estimate from preseason projection + in-season performance, weighted by confidence in each",
            ),
            "Est. Bases (Season)": st.column_config.NumberColumn(
                format="%.4f",
                help="Bayesian estimate of true production this season — adjusted for sample size",
            ),
            "Est. Bases (Raw)": st.column_config.NumberColumn(
                format="%.4f",
                help="Unadjusted observed rate — what the raw numbers say before Bayesian shrinkage",
            ),
            "Preseason Proj.": st.column_config.NumberColumn(
                format="%.4f",
                help="Preseason projection based on multi-year historical performance and aging curve",
            ),
            "Deviation": st.column_config.NumberColumn(
                format="%+.4f",
                help="Raw rate minus true talent — positive means overperforming (likely to regress), negative means underperforming (likely to bounce back)",
            ),
            "Adjustment": st.column_config.NumberColumn(
                format="%+.4f",
                help="How much the Bayesian model adjusted the raw rate — larger adjustments indicate more shrinkage (smaller samples)",
            ),
            "K%": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Bayesian strikeout rate — lower is better for hitters, higher is better for pitchers. Adjusted for sample size.",
            ),
            "BB%": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Bayesian walk rate — higher is better for hitters (more free bases), lower is better for pitchers. Adjusted for sample size.",
            ),
            "HR%": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Bayesian home run rate — percentage of plate appearances resulting in a home run. Adjusted for sample size.",
            ),
            "PQS+": st.column_config.NumberColumn(
                format="%.0f",
                help="Pitcher Quality Score (70% K% + 30% contact quality). Higher is better. "
                     "100 = league average. Combines strikeout ability and batted ball quality. "
                     "Validated at r=0.45 predicting next-year wOBA allowed.",
            ),
            "xEB/PA": st.column_config.NumberColumn(
                format="%.4f",
                help="Expected bases per PA — composite of K%, BB%, HBP%, and EB-per-BIP "
                     "posteriors. (1 − K% − BB% − HBP%) × EB_per_BIP + BB% + HBP%. "
                     "Lower is better. Validated to tie PQS+ on next-year predictive r.",
            ),
            n_col_name: st.column_config.NumberColumn(format="%d"),
            "Pos": st.column_config.TextColumn(width="small"),
            "Profile": st.column_config.LinkColumn(display_text="View"),
            "AVG": st.column_config.NumberColumn(format="%.3f", help="Batting average (H/AB)"),
            "HR": st.column_config.NumberColumn(format="%d", help="Home runs"),
            "SB": st.column_config.NumberColumn(format="%d", help="Stolen bases"),
            "ERA": st.column_config.NumberColumn(format="%.2f", help="Earned run average"),
            "WHIP": st.column_config.NumberColumn(format="%.2f", help="Walks + hits per inning pitched"),
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

        if has_trad_stats:
            st.caption("Traditional stats via MLB Stats API. Totals may vary slightly from other sources if Statcast data was unavailable for a game or plate appearance.")

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

            # Check if projections are available for this pool
            _use_projections = has_true_talent and not proj_df.empty and "projected_eb_pa" in proj_df.columns

            # League average from all players (not filtered subset) for reference line
            _lg_mean_eb = df["posterior_mean"].mean()

            if _use_projections:
                # -- Projection-based spotlight --
                _min_pa_spotlight = 10  # Projections carry the signal, not in-season HDI
                standouts_pool = df[df["n_batted_balls"] >= _min_pa_spotlight].copy()
                if selected_team != "All Teams":
                    standouts_pool = standouts_pool[standouts_pool["team"] == selected_team]

                # Merge projection columns onto the pool
                _proj_cols = ["player", "team", "projected_eb_pa", "projected_hdi_low",
                              "projected_hdi_high", "n_seasons"]
                _proj_cols = [c for c in _proj_cols if c in proj_df.columns]
                standouts_pool = standouts_pool.merge(
                    proj_df[_proj_cols].drop_duplicates(subset=["player", "team"]),
                    on=["player", "team"], how="left",
                )

                # Potential Risers: projection above current (positive regression candidates)
                risers = pd.DataFrame()
                _with_proj = standouts_pool.dropna(subset=["projected_eb_pa"])
                if not _with_proj.empty:
                    _proj_75 = _with_proj["projected_eb_pa"].quantile(0.75)
                    _curr_60 = _with_proj["posterior_mean"].quantile(0.60)
                    risers = _with_proj[
                        (_with_proj["projected_eb_pa"] >= _proj_75) &
                        (_with_proj["posterior_mean"] < _curr_60)
                    ].sort_values("projected_eb_pa", ascending=False).head(15)

                # High Upside: small sample players with elite ceiling
                # Fixed PA ceiling so tab stays useful as the season progresses
                _pa_ceiling = 75
                upside = pd.DataFrame()
                _small_sample = standouts_pool[
                    standouts_pool["n_batted_balls"] <= _pa_ceiling
                ]
                if _small_sample["hdi_high"].notna().any():
                    # Prefer projection-informed ceiling when available
                    if "projected_hdi_high" in _small_sample.columns and _small_sample["projected_hdi_high"].notna().any():
                        _small_proj = _small_sample.dropna(subset=["projected_hdi_high"])
                        _ceiling_col = "projected_hdi_high"
                    else:
                        _small_proj = _small_sample
                        _ceiling_col = "hdi_high"
                    _ceil_70 = _small_proj[_ceiling_col].quantile(0.70)
                    upside = _small_proj[
                        _small_proj[_ceiling_col] >= _ceil_70
                    ].sort_values(_ceiling_col, ascending=False).head(15)

                # Reliable Floor: multi-year history, above-median true talent, narrow projection HDI
                safe_floor = pd.DataFrame()
                _with_tt = standouts_pool.dropna(subset=["true_talent_eb_pa"])
                if not _with_tt.empty and "n_seasons" in _with_tt.columns:
                    _tt_med = _with_tt["true_talent_eb_pa"].median()
                    _floor_pool = _with_tt[
                        (_with_tt["true_talent_eb_pa"] >= _tt_med) &
                        (_with_tt["n_seasons"] >= 2)
                    ].copy()
                    if not _floor_pool.empty and "projected_hdi_low" in _floor_pool.columns:
                        _floor_pool["_proj_hdi_width"] = (
                            _floor_pool["projected_hdi_high"] - _floor_pool["projected_hdi_low"]
                        )
                        safe_floor = _floor_pool.nsmallest(15, "_proj_hdi_width")
            else:
                # -- Fallback: HDI-based spotlight (no projections) --
                risers = pd.DataFrame()  # No projections → can't identify risers

                standouts_pool = df[df["n_batted_balls"] >= 20].copy()
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

            _any_spotlight = not risers.empty or not upside.empty or not safe_floor.empty
            if _any_spotlight:
                # Build tab list dynamically based on what data is available
                _tab_names = []
                if not risers.empty:
                    _tab_names.append("Potential Risers")
                _tab_names.append("High Upside")
                _tab_names.append("Reliable Floor")
                _tabs = st.tabs(_tab_names)
                _tab_idx = 0

                # -- Potential Risers tab (projection-based only) --
                if not risers.empty:
                    with _tabs[_tab_idx]:
                        st.caption(
                            "Players whose multi-year projection is well above their current-season "
                            "production — positive regression candidates likely to improve."
                        )
                        fig_risers = _build_forest_plot(
                            risers, "#2563eb", "Projected EB/PA",
                            metric_short=metric_short,
                            mean_col="projected_eb_pa",
                            low_col="projected_hdi_low",
                            high_col="projected_hdi_high",
                            league_mean=_lg_mean_eb,
                            title=f"Potential Risers — Projected {metric_short}",
                            secondary_col="posterior_mean",
                            secondary_name=f"Current {metric_short}",
                        )
                        st.plotly_chart(fig_risers, use_container_width=True, config=PLOTLY_CONFIG_FOREST, theme=None)

                        _up_cols = ["player", "team", "projected_eb_pa", "posterior_mean",
                                    "n_batted_balls"]
                        if "true_talent_eb_pa" in risers.columns:
                            _up_cols.insert(4, "true_talent_eb_pa")
                        up_display = risers[[c for c in _up_cols if c in risers.columns]].copy()
                        up_display["Profile"] = up_display["player"].apply(
                            lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                        )
                        _rename = {
                            "player": "Player", "team": "Team",
                            "projected_eb_pa": "Projected EB/PA",
                            "posterior_mean": f"Current {metric_short}",
                            "true_talent_eb_pa": "True Talent",
                            "n_batted_balls": "PA",
                        }
                        up_display = up_display.rename(columns=_rename)
                        st.dataframe(up_display, hide_index=True, use_container_width=True,
                                     column_config={
                                         "Projected EB/PA": st.column_config.NumberColumn(format="%.3f"),
                                         f"Current {metric_short}": st.column_config.NumberColumn(format="%.3f"),
                                         "True Talent": st.column_config.NumberColumn(format="%.3f"),
                                         "PA": st.column_config.NumberColumn(format="%d"),
                                         "Profile": st.column_config.LinkColumn(display_text="View"),
                                     })
                        st.caption("Thin lines = projected range, circles = projected EB/PA, diamonds = current season")
                    _tab_idx += 1

                # -- High Upside tab --
                with _tabs[_tab_idx]:
                    st.caption(
                        "Players with limited plate appearances whose ceiling is elite. "
                        "High variability means high risk — but also breakout potential."
                    )
                    if not upside.empty:
                        fig_upside = _build_forest_plot(upside, "#2563eb", "High Upside",
                                                        metric_short=metric_short,
                                                        league_mean=_lg_mean_eb,
                                                        title=f"High Upside {player_type}s — {metric_short}")
                        st.plotly_chart(fig_upside, use_container_width=True, config=PLOTLY_CONFIG_FOREST, theme=None)

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

                    hdi_caption = "Thin lines = possible range, dots = model estimate"
                    if has_50_hdi:
                        hdi_caption = "Thin lines = possible range, thick lines = likely range, dots = model estimate"
                    st.caption(hdi_caption)
                _tab_idx += 1

                # -- Reliable Floor tab --
                with _tabs[_tab_idx]:
                    if _use_projections:
                        st.caption(
                            "Above-average players with 2+ years of history and the narrowest projection "
                            "intervals — the model is most confident in their production level."
                        )
                    else:
                        st.caption(
                            "Above-average players with enough data that the model is confident in their "
                            "estimate. Narrow ranges mean consistent, predictable production."
                        )
                    if not safe_floor.empty:
                        if _use_projections:
                            fig_floor = _build_forest_plot(
                                safe_floor, "#16a34a", "True Talent EB/PA",
                                metric_short=metric_short,
                                mean_col="true_talent_eb_pa",
                                low_col="projected_hdi_low",
                                high_col="projected_hdi_high",
                                league_mean=_lg_mean_eb,
                                title=f"Reliable Floor {player_type}s — True Talent {metric_short}",
                            )
                        else:
                            fig_floor = _build_forest_plot(safe_floor, "#16a34a", "Reliable Floor",
                                                           metric_short=metric_short,
                                                           league_mean=_lg_mean_eb,
                                                           title=f"Reliable Floor {player_type}s — {metric_short}")
                        st.plotly_chart(fig_floor, use_container_width=True, config=PLOTLY_CONFIG_FOREST, theme=None)

                        if _use_projections:
                            _sf_cols = ["player", "team", "true_talent_eb_pa", "projected_eb_pa",
                                        "n_seasons", "n_batted_balls"]
                            sf_display = safe_floor[[c for c in _sf_cols if c in safe_floor.columns]].copy()
                            sf_display["Profile"] = sf_display["player"].apply(
                                lambda p: f"/Hitter_Profile?player={urllib.parse.quote(p)}"
                            )
                            _sf_rename = {
                                "player": "Player", "team": "Team",
                                "true_talent_eb_pa": "True Talent",
                                "projected_eb_pa": "Projected EB/PA",
                                "n_seasons": "Seasons", "n_batted_balls": "PA",
                            }
                            sf_display = sf_display.rename(columns=_sf_rename)
                            st.dataframe(
                                sf_display, hide_index=True, use_container_width=True,
                                column_config={
                                    "True Talent": st.column_config.NumberColumn(format="%.3f"),
                                    "Projected EB/PA": st.column_config.NumberColumn(format="%.3f"),
                                    "Seasons": st.column_config.NumberColumn(format="%d"),
                                    "PA": st.column_config.NumberColumn(format="%d"),
                                    "Profile": st.column_config.LinkColumn(display_text="View"),
                                },
                            )
                        else:
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

                    _floor_caption = "Thin lines = possible range, dots = model estimate"
                    if _use_projections:
                        _floor_caption = "Thin lines = projected range, dots = true talent estimate"
                    elif has_50_hdi:
                        _floor_caption = "Thin lines = possible range, thick lines = likely range, dots = model estimate"
                    st.caption(_floor_caption)
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
                st.info("Pitcher metadata not available — needed to determine throw hand for platoon splits.")
            else:
                # Adaptive slider defaults based on season progress
                _bb_dates = pd.to_datetime(bb_df["date"], format="mixed", errors="coerce")
                _season_days = max(1, (_bb_dates.max() - _bb_dates.min()).days) if not _bb_dates.isna().all() else 0
                if _season_days <= 14:
                    _plat_min, _plat_default, _plat_step = 3, 5, 1
                elif _season_days <= 30:
                    _plat_min, _plat_default, _plat_step = 5, 10, 5
                else:
                    _plat_min, _plat_default, _plat_step = 10, 15, 5

                col_plat_pos, col_plat_min = st.columns([1, 1])
                with col_plat_pos:
                    plat_pos = st.selectbox(
                        "Position (platoon)",
                        ["All", "C", "1B", "2B", "SS", "3B", "OF", "DH"],
                        key="platoon_pos",
                    )
                with col_plat_min:
                    plat_min_bb = st.slider(
                        "Min BB per side", _plat_min, 50, _plat_default, step=_plat_step, key="platoon_min"
                    )

                if plat_min_bb < 10:
                    st.caption(
                        ":orange[Small sample warning:] platoon splits with fewer than 10 batted balls "
                        "per side are noisy. Use these as directional signals only."
                    )

                platoon_df = compute_platoon_splits(bb_df, metadata_df, min_bb=plat_min_bb)

                if platoon_df.empty:
                    if _season_days <= 14:
                        st.info(
                            f"Not enough data yet — only {_season_days} days into the season. "
                            f"Try lowering the minimum BB threshold or check back as more games are played."
                        )
                    else:
                        st.info("No players have enough batted balls per pitcher hand to compute platoon splits.")
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

        # SECTION 6: TOTAL OFFENSIVE CONTRIBUTIONS (PA-mode hitters only)
        if is_pa_mode and type_key == "hitter":
            st.divider()
            st.subheader("Total offensive contributions")

            with st.expander("How does this work?"):
                st.markdown("""
This chart shows **cumulative** estimated bases produced by each hitter — not per-plate-appearance
rates. It answers: "Who has contributed the most total offense?"

- **Batted Ball Bases** (green) — the sum of estimated bases from all batted ball outcomes,
  based on exit velocity, launch angle, and spray angle
- **Walks** (blue) — each walk counts as 1 base (reaching first)

A hitter with a modest EB/PA rate but lots of plate appearances can out-produce a high-rate
hitter with fewer opportunities. This complements the per-PA rankings above by showing volume.
""")

            # Load data early so we can set date picker bounds
            contrib_bb = load_batted_balls(season)

            if contrib_bb.empty:
                st.info(f"No batted ball data available for {season}.")
            else:
                contrib_pa = load_pa_counts(season)

                # Parse dates for PA counts
                if not contrib_pa.empty and "date" in contrib_pa.columns:
                    contrib_pa["date_parsed"] = pd.to_datetime(
                        contrib_pa["date"], format="%m/%d/%Y", errors="coerce"
                    )

                data_min_date = contrib_bb["date_parsed"].min().date()
                data_max_date = contrib_bb["date_parsed"].max().date()

                col_date_preset, col_top_n = st.columns([2, 1])
                with col_date_preset:
                    date_preset = st.radio(
                        "Date range",
                        ["Full Season", "Last 30 Days", "Last 14 Days", "Last 7 Days", "Custom"],
                        horizontal=True,
                        key="contrib_date_preset",
                    )
                with col_top_n:
                    contrib_top_n = st.slider(
                        "Top N players", 10, 50, 20, step=5, key="contrib_top_n"
                    )

                # Resolve date range
                if date_preset == "Custom":
                    custom_range = st.date_input(
                        "Select date range",
                        value=(data_min_date, data_max_date),
                        min_value=data_min_date,
                        max_value=data_max_date,
                        key="contrib_custom_dates",
                    )
                    if isinstance(custom_range, tuple) and len(custom_range) == 2:
                        min_date = pd.Timestamp(custom_range[0])
                        max_date = pd.Timestamp(custom_range[1])
                    else:
                        st.info("Select both a start and end date.")
                        st.stop()
                elif date_preset == "Last 30 Days":
                    max_date = pd.Timestamp(data_max_date)
                    min_date = max_date - pd.Timedelta(days=29)
                elif date_preset == "Last 14 Days":
                    max_date = pd.Timestamp(data_max_date)
                    min_date = max_date - pd.Timedelta(days=13)
                elif date_preset == "Last 7 Days":
                    max_date = pd.Timestamp(data_max_date)
                    min_date = max_date - pd.Timedelta(days=6)
                else:
                    min_date = pd.Timestamp(data_min_date)
                    max_date = pd.Timestamp(data_max_date)

                bb_filtered = contrib_bb[
                    (contrib_bb["date_parsed"] >= min_date)
                    & (contrib_bb["date_parsed"] <= max_date)
                ]
                pa_filtered = pd.DataFrame()
                if not contrib_pa.empty and "date_parsed" in contrib_pa.columns:
                    pa_filtered = contrib_pa[
                        (contrib_pa["date_parsed"] >= min_date)
                        & (contrib_pa["date_parsed"] <= max_date)
                    ]

                # Apply team filter
                if selected_team != "All Teams":
                    bb_filtered = bb_filtered[bb_filtered["team"] == selected_team]
                    if not pa_filtered.empty:
                        pa_filtered = pa_filtered[pa_filtered["team"] == selected_team]

                if bb_filtered.empty:
                    st.info("No batted ball data for the selected filters.")
                else:
                    # Aggregate batted ball bases
                    bb_agg = (
                        bb_filtered.groupby(["player", "team"])
                        .agg(batted_ball_bases=("estimated_bases", "sum"))
                        .reset_index()
                    )

                    # Aggregate walk bases
                    if not pa_filtered.empty and "walks" in pa_filtered.columns:
                        pa_agg = (
                            pa_filtered.groupby(["player", "team"])
                            .agg(walk_bases=("walks", "sum"))
                            .reset_index()
                        )
                    else:
                        pa_agg = pd.DataFrame(columns=["player", "team", "walk_bases"])

                    # Merge
                    merged = bb_agg.merge(pa_agg, on=["player", "team"], how="outer")
                    merged["batted_ball_bases"] = pd.to_numeric(
                        merged["batted_ball_bases"].fillna(0)
                    )
                    merged["walk_bases"] = pd.to_numeric(
                        merged["walk_bases"].fillna(0)
                    )
                    merged["total_bases"] = (
                        merged["batted_ball_bases"] + merged["walk_bases"]
                    )

                    # Sort and take top N
                    merged = (
                        merged.sort_values("total_bases", ascending=False)
                        .head(contrib_top_n)
                        .reset_index(drop=True)
                    )

                    # Build title and subtitle for screenshot context
                    date_fmt = "%b %d"
                    date_range_str = (
                        f"{min_date.strftime(date_fmt)} - "
                        f"{max_date.strftime(f'{date_fmt}, %Y')}"
                    )
                    team_label = selected_team if selected_team != "All Teams" else "MLB"
                    chart_title = f"Top {len(merged)} offensive contributors"
                    chart_subtitle = (
                        f"{team_label} · {date_range_str} · "
                        f"Batted ball estimated bases + walks"
                    )

                    fig_contrib = _build_contributors_chart(
                        merged, title=chart_title, subtitle=chart_subtitle,
                    )
                    st.plotly_chart(
                        fig_contrib,
                        use_container_width=True,
                        config=PLOTLY_CONFIG_FOREST,
                        theme=None,
                    )

                    walk_note = ""
                    if pa_agg.empty:
                        walk_note = " · Walk data not yet available"
                    st.caption(
                        f"Batted ball bases from exit velocity, launch angle, and spray "
                        f"angle model. Walks count as 1 base each.{walk_note}"
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
        _proj_has_archetypes = "archetype" in proj_df.columns and proj_df["archetype"].str.len().gt(0).any()
        if _proj_has_archetypes:
            col_psearch, col_ppos, col_parch, col_pteam, col_pmin = st.columns([2, 1, 1, 1, 1])
        else:
            col_psearch, col_ppos, col_pteam, col_pmin = st.columns([2, 1, 1, 1])

        with col_psearch:
            _proj_player_options = sorted(proj_df["player"].unique())
            proj_search_selections = st.multiselect(
                "Search players",
                options=_proj_player_options,
                default=[],
                placeholder="Type to search players...",
                key="proj_multi_search",
            )

        with col_ppos:
            proj_pos_options = ["All"]
            if "position" in proj_df.columns:
                proj_pos_options += ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
            proj_pos_filter = st.selectbox("Position", proj_pos_options, key="proj_pos")

        proj_arch_filter = "All"
        if _proj_has_archetypes:
            with col_parch:
                _proj_arch_vals = sorted(proj_df.loc[proj_df["archetype"].str.len() > 0, "archetype"].unique())
                proj_arch_filter = st.selectbox("Archetype", ["All"] + _proj_arch_vals, key="proj_arch")

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
        if proj_arch_filter != "All" and "archetype" in proj_filtered.columns:
            proj_filtered = proj_filtered[proj_filtered["archetype"] == proj_arch_filter]
        if proj_search_selections:
            proj_filtered = proj_filtered[proj_filtered["player"].isin(proj_search_selections)]

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
            "archetype": "Archetype",
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
            "Player", "Team", "Pos", "Archetype", "Age", "Projected EB/PA",
            "Recent EB/PA",
            "Seasons", "P(Active)", "Profile",
        ]
        proj_table_cols = [c for c in proj_table_cols if c in proj_display.columns]

        PROJ_COL_CONFIG = {
            "Projected EB/PA": st.column_config.NumberColumn(
                format="%.4f",
                help="Bayesian projection of estimated bases per plate appearance for the upcoming season",
            ),
            "Recent EB/PA": st.column_config.NumberColumn(
                format="%.4f",
                help="Most recent season's Bayesian EB/PA estimate (the model's starting point before aging adjustment)",
            ),
            "Age": st.column_config.NumberColumn(format="%.1f"),
            "Seasons": st.column_config.NumberColumn(
                format="%d",
                help="Number of prior seasons of data used to build this player's projection",
            ),
            "P(Active)": st.column_config.NumberColumn(
                format="%.0f%%",
                help="Model-estimated probability the player will be active next season, based on age, performance, and historical patterns of player attrition",
            ),
            "Pos": st.column_config.TextColumn(width="small"),
            "Profile": st.column_config.LinkColumn(display_text="View"),
        }

        # Projection rankings chart
        st.subheader("Projection rankings")

        # Check if rate stat evaluation data is available to offer metric tabs.
        # For projection-only seasons (e.g. 2026), pull rate data from the most
        # recent evaluation season that has it.
        _eval_for_rates = pd.DataFrame()
        if has_eval_data:
            _eval_for_rates = load_player_evaluations_pa(season, "hitter")
        if (_eval_for_rates.empty or "k_rate_posterior" not in _eval_for_rates.columns):
            for _fallback_s in eval_seasons[:3]:
                _fb = load_player_evaluations_pa(_fallback_s, "hitter")
                if not _fb.empty and "k_rate_posterior" in _fb.columns:
                    _eval_for_rates = _fb
                    break
        _has_rate_data = (not _eval_for_rates.empty
                          and "k_rate_posterior" in _eval_for_rates.columns
                          and _eval_for_rates["k_rate_posterior"].notna().any())

        # Metric selector + Sort + Logo badge in one row
        _proj_logo_url = get_team_logo_url(proj_team) if proj_team != "All Teams" else MLB_LOGO_URL
        _proj_logo_label = proj_team if proj_team != "All Teams" else "MLB"
        _col_metric, _col_div, _col_sort, _col_logo = st.columns([3, 0.1, 2, 1])

        with _col_metric:
            _proj_metric_options = ["EB/PA"]
            if _has_rate_data:
                _proj_metric_options += ["K% (low is better)", "BB%", "HR%"]
            _proj_metric = st.radio(
                "Metric", _proj_metric_options,
                horizontal=True, key="proj_forest_metric",
                help="EB/PA shows projected overall production. Rate stats show current-season Bayesian estimates for the projected players.",
            )

        with _col_div:
            st.markdown(
                '<div style="border-left:1px solid #ccc;height:60px;margin-top:8px;"></div>',
                unsafe_allow_html=True,
            )

        with _col_sort:
            _proj_sort_options = ["Best first", "Worst first"]
            _proj_sort = st.radio(
                "Sort", _proj_sort_options,
                horizontal=True, key="proj_forest_sort",
            )

        with _col_logo:
            st.markdown(
                f'<div style="text-align:right;padding-top:12px;">'
                f'<img src="{_proj_logo_url}" alt="{_proj_logo_label}" '
                f'style="height:64px;width:64px;object-fit:contain;" '
                f'onerror="this.style.display=\'none\'">'
                f'</div>',
                unsafe_allow_html=True,
            )
        _user_wants_asc = (_proj_sort == "Worst first")

        top_20_proj = proj_filtered.copy()
        if not top_20_proj.empty:
            if _proj_metric == "EB/PA":
                # Sort projections
                _eb_asc = _user_wants_asc
                top_20_proj = top_20_proj.sort_values(
                    "projected_eb_pa", ascending=_eb_asc
                ).head(20)

                _eb_league_mean = proj_filtered["projected_eb_pa"].mean()
                fig_proj = _build_forest_plot(
                    top_20_proj, "#7c3aed", "Projected EB/PA",
                    metric_short="EB/PA",
                    mean_col="projected_eb_pa",
                    low_col="projected_hdi_low",
                    high_col="projected_hdi_high",
                    low_50_col="projected_hdi_50_low",
                    high_50_col="projected_hdi_50_high",
                    count_col="n_seasons",
                    count_label="Seasons",
                    sort_ascending=not _eb_asc,
                    use_team_colors=True,
                    league_mean=_eb_league_mean,
                    title=f"Projected {player_type}s — EB/PA",
                )
                st.plotly_chart(fig_proj, use_container_width=True, config=PLOTLY_CONFIG_FOREST, theme=None)
                st.caption("Tap the camera icon (top-right) to download a high-res image.")

                has_50 = ("projected_hdi_50_low" in top_20_proj.columns
                          and top_20_proj["projected_hdi_50_low"].notna().any())
                caption = "Thin lines = possible range, dots = projected EB/PA"
                if has_50:
                    caption = "Thin lines = possible range, thick lines = likely range, dots = projected EB/PA"
                st.caption(caption)
            else:
                # Show K%, BB%, or HR% for the projected hitters
                _rate_map = {
                    "K% (low is better)": ("k_rate", "#dc2626", "K%"),
                    "BB%": ("bb_rate", "#16a34a", "BB%"),
                    "HR%": ("hr_rate", "#7c3aed", "HR%"),
                }
                _rate_prefix, _rate_color, _rate_label = _rate_map[_proj_metric]

                # Merge rate data onto projected players
                _rate_cols = ["player_id", f"{_rate_prefix}_posterior",
                              f"{_rate_prefix}_hdi_low", f"{_rate_prefix}_hdi_high"]
                _rate_slim = _eval_for_rates[[c for c in _rate_cols if c in _eval_for_rates.columns]].copy()

                # Merge by player_id first, fall back to player name
                _proj_with_rates = pd.DataFrame()
                if "player_id" in top_20_proj.columns and "player_id" in _rate_slim.columns:
                    _proj_with_rates = top_20_proj.merge(_rate_slim, on="player_id", how="left")
                if _proj_with_rates.empty or f"{_rate_prefix}_posterior" not in _proj_with_rates.columns or _proj_with_rates[f"{_rate_prefix}_posterior"].isna().all():
                    _rate_by_name = _eval_for_rates.drop_duplicates(subset=["player"])[
                        ["player"] + [c for c in _rate_cols if c in _eval_for_rates.columns and c != "player_id"]
                    ]
                    _proj_with_rates = top_20_proj.merge(_rate_by_name, on="player", how="left")

                _valid_rates = _proj_with_rates[_proj_with_rates[f"{_rate_prefix}_posterior"].notna()]
                if not _valid_rates.empty:
                    _plot_data = _valid_rates.copy()
                    # Clamp negative HDI (safety for pre-logit-transform data)
                    _hdi_low_col = f"{_rate_prefix}_hdi_low"
                    if _hdi_low_col in _plot_data.columns:
                        _plot_data[_hdi_low_col] = _plot_data[_hdi_low_col].clip(lower=0)
                    for c in [f"{_rate_prefix}_posterior", f"{_rate_prefix}_hdi_low", f"{_rate_prefix}_hdi_high"]:
                        if c in _plot_data.columns:
                            _plot_data[c] = _plot_data[c] * 100
                    # K%: low is good for hitters, so "best first" = ascending
                    # BB%/HR%: high is good for hitters, so "best first" = descending
                    if _rate_prefix == "k_rate":
                        _sort_asc = not _user_wants_asc  # best first = ascending (low K%)
                    else:
                        _sort_asc = _user_wants_asc  # best first = descending (high BB%/HR%)

                    _plot_data = _plot_data.sort_values(
                        f"{_rate_prefix}_posterior", ascending=_sort_asc
                    ).head(20)

                    _rate_league_mean = _eval_for_rates[f"{_rate_prefix}_posterior"].mean() * 100
                    fig_rate = _build_forest_plot(
                        _plot_data, _rate_color, f"{_rate_label} (Projected {player_type}s)",
                        metric_short=_rate_label,
                        mean_col=f"{_rate_prefix}_posterior",
                        low_col=f"{_rate_prefix}_hdi_low",
                        high_col=f"{_rate_prefix}_hdi_high",
                        count_col="n_seasons",
                        count_label="Seasons",
                        sort_ascending=not _sort_asc,
                        use_team_colors=True,
                        league_mean=_rate_league_mean,
                        title=f"Projected {player_type}s — {_rate_label}",
                    )
                    fig_rate.update_layout(
                        xaxis_title=_rate_label,
                        xaxis=dict(ticksuffix="%"),
                    )
                    st.plotly_chart(fig_rate, use_container_width=True, config=PLOTLY_CONFIG_FOREST, theme=None)
                    st.caption("Tap the camera icon (top-right) to download a high-res image.")

                    _direction_note = " Lower K% is better for hitters." if _rate_prefix == "k_rate" else ""
                    st.caption(
                        f"Current-season Bayesian {_rate_label} for the top projected hitters.{_direction_note} "
                        f"Thin lines = 89% credible interval, dots = model estimate."
                    )
                else:
                    st.info(f"No {_rate_label} data available for the projected players yet.")

        # Projection table (below chart)
        st.divider()
        st.subheader("Projection table")

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


# =============================================================================
# FOOTER
# =============================================================================

render_home_link()
