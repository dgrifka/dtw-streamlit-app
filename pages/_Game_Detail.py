"""
Game Detail Page
Shows all 4 visualizations for a selected game.
"""

import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import (load_game_summaries, get_game_images, get_deserved_winner,
                                load_batted_balls, load_pa_counts, _image_exists)
from utils.team_mappings import get_short_name
from utils.player_helpers import safe_html, build_video_url
from utils.responsive import inject_responsive_css, render_home_link, upset_badge_html

# Custom CSS
st.markdown("""
<style>
    .team-score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .team-name {
        font-size: 1rem;
        color: #4A5568;
    }
    .game-info-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .image-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .image-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.25rem;
    }
    .image-caption {
        font-size: 0.8rem;
        color: #718096;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

inject_responsive_css()


def main():
    # Get gamePk from session state first, then query params as fallback
    game_pk = st.session_state.get('selected_game_pk', None)
    
    if game_pk is None:
        params = st.query_params
        game_pk = params.get("gamePk", None)
        if game_pk is not None:
            game_pk = int(game_pk)
    
    # Load all games
    df = load_game_summaries()
    
    if df.empty:
        st.error("Could not load game data.")
        return
    
    # If no game selected, show message and link
    if game_pk is None:
        st.title("Game Detail")
        st.warning("No game selected. Please select a game from Game Simulations.")
        st.page_link("pages/1_Game_Simulations.py", label="Browse All Games", width="stretch")
        return

    # Load specific game
    game_row = df[df['gamePk'] == game_pk]

    if game_row.empty:
        st.error(f"Game {game_pk} not found.")
        st.page_link("pages/1_Game_Simulations.py", label="Browse All Games", width="stretch")
        return
    
    row = game_row.iloc[0]
    
    # Get team info
    away_short = get_short_name(row['away'])
    home_short = get_short_name(row['home'])
    winner_info = get_deserved_winner(row)
    
    away_wp = int(round(row['away_wp'] * 100))
    home_wp = int(round(row['home_wp'] * 100))
    
    # ============ HEADER ============
    if st.button("← Back to Games"):
        st.switch_page("pages/1_Game_Simulations.py")
    
    st.divider()
    
    # Game title
    badge = upset_badge_html(winner_info['was_upset'])
    st.markdown(f"<h2>{safe_html(row['away'])} @ {safe_html(row['home'])}{badge}</h2>", unsafe_allow_html=True)
    
    # Score and info row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">{safe_html(away_short)}</div>
            <div class="team-score">{row['away_score']}</div>
            <div style="font-size: 0.85rem; color: #666;">DTW: {away_wp}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">{safe_html(home_short)}</div>
            <div class="team-score">{row['home_score']}</div>
            <div style="font-size: 0.85rem; color: #666;">DTW: {home_wp}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        date_str = row['date'].strftime('%-m/%d/%y')
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">Date</div>
            <div class="team-score" style="font-size: 1.2rem;">{date_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if winner_info['was_upset']:
            result_text = "Upset"
            result_detail = f"{safe_html(get_short_name(winner_info['deserved_winner']))} deserved to win"
        else:
            result_text = "Deserved"
            result_detail = "Expected team won"
        
        st.markdown(f"""
        <div class="game-info-box">
            <div class="team-name">Result</div>
            <div class="team-score" style="font-size: 1.2rem;">{result_text}</div>
            <div style="font-size: 0.75rem; color: #666;">{result_detail}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ============ IMAGES IN COMPARTMENTS ============
    st.subheader("Simulation Visualizations")
    
    images = get_game_images(row)
    
    # Row 1: Spray Chart and Luck Ledger (fallback to Estimated Bases for older games)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Spray Chart</div>
            <div class="image-caption">Where batted balls landed in the ballpark</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['spray'], width="stretch")

    with col2:
        luck_url = images.get('luck_ledger')
        if luck_url and _image_exists(luck_url):
            title, caption = "Luck Ledger", "Which plays drove the luck differential"
        else:
            luck_url = images['estimated_bases']
            title, caption = "Estimated Bases", "Expected bases vs actual bases by player"
        st.markdown(f"""
        <div class="image-container">
            <div class="image-title">{title}</div>
            <div class="image-caption">{caption}</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(luck_url, width="stretch")
    
    # Separator
    st.markdown("<hr style='border: none; border-top: 1px solid #e0e0e0; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Row 2: Run Distribution and Player Contributions
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Run Distribution</div>
            <div class="image-caption">Simulated run outcomes from 10,000 simulations</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['rd'], width="stretch")
    
    with col4:
        st.markdown("""
        <div class="image-container">
            <div class="image-title">Player Contributions</div>
            <div class="image-caption">Individual player impact on win probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(images['player_contributions'], width="stretch")

    # ============ BATTED BALL TABLE ============
    st.divider()
    st.subheader("All Batted Balls")

    season = row['date'].year
    bb_df = load_batted_balls(season)

    if not bb_df.empty:
        game_bb = bb_df[bb_df['gamePk'] == game_pk].copy()

        if not game_bb.empty:
            # Compute actual bases and luck
            BASES_MAP = {'Single': 1, 'Double': 2, 'Triple': 3, 'Home Run': 4}
            game_bb['actual_bases'] = game_bb['actual_result'].map(BASES_MAP).fillna(0).astype(int)
            game_bb['luck'] = game_bb['actual_bases'] - game_bb['estimated_bases']
            game_bb = game_bb.sort_values('luck', ascending=False)

            # Walk/strikeout summary from PA counts
            pa_df = load_pa_counts(season)
            if not pa_df.empty:
                game_pa = pa_df[pa_df['gamePk'] == game_pk]
                if not game_pa.empty:
                    away_pa = game_pa[game_pa['team'] == row['away']]
                    home_pa = game_pa[game_pa['team'] == row['home']]
                    away_walks = int(away_pa['walks'].sum()) if not away_pa.empty else 0
                    away_ks = int(away_pa['strikeouts'].sum()) if not away_pa.empty else 0
                    home_walks = int(home_pa['walks'].sum()) if not home_pa.empty else 0
                    home_ks = int(home_pa['strikeouts'].sum()) if not home_pa.empty else 0
                    st.caption(
                        f"Batted balls only \u2014 "
                        f"{safe_html(row['away'])}: {away_walks} BB, {away_ks} K | "
                        f"{safe_html(row['home'])}: {home_walks} BB, {home_ks} K"
                    )

            # Build display DataFrame
            display_cols = ['player', 'team', 'pitcher', 'launch_speed', 'launch_angle',
                           'actual_result', 'estimated_bases', 'actual_bases', 'luck', 'xba']
            if 'play_id' in game_bb.columns:
                game_bb['video'] = game_bb['play_id'].apply(build_video_url)
                display_cols.append('video')

            available = [c for c in display_cols if c in game_bb.columns]
            display = game_bb[available].copy()
            display = display.rename(columns={
                'player': 'Player', 'team': 'Team', 'pitcher': 'Pitcher',
                'launch_speed': 'Exit Velo', 'launch_angle': 'Launch Angle',
                'actual_result': 'Result', 'estimated_bases': 'Est. Bases',
                'actual_bases': 'Actual Bases', 'luck': 'Luck', 'xba': 'xBA',
                'video': 'Video',
            })

            col_config = {
                'Exit Velo': st.column_config.NumberColumn(format="%.1f mph"),
                'Launch Angle': st.column_config.NumberColumn(format="%d\u00b0"),
                'Est. Bases': st.column_config.NumberColumn(format="%.2f"),
                'Actual Bases': st.column_config.NumberColumn(format="%d"),
                'Luck': st.column_config.NumberColumn(format="%+.2f"),
                'xBA': st.column_config.NumberColumn(format="%.3f"),
            }
            if 'Video' in display.columns:
                col_config['Video'] = st.column_config.LinkColumn(display_text="Watch")

            st.dataframe(display, hide_index=True, use_container_width=True,
                         column_config=col_config)
        else:
            st.info("Batted ball data not yet available for this game.")
    else:
        st.info(f"No batted ball data available for the {season} season.")

    # ============ SIDEBAR ============
    st.sidebar.header("Find Another Game")
    
    teams = sorted(df['home'].unique().tolist())
    selected_team = st.sidebar.selectbox("Filter by Team", ["All Teams"] + teams)
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=row['date'].date(),
        min_value=min_date,
        max_value=max_date
    )
    
    filtered_df = df.copy()
    if selected_team != "All Teams":
        filtered_df = filtered_df[
            (filtered_df['home'] == selected_team) | 
            (filtered_df['away'] == selected_team)
        ]
    
    date_games = filtered_df[filtered_df['date'].dt.date == selected_date]
    
    if not date_games.empty:
        st.sidebar.markdown(f"**Games on {selected_date.strftime('%-m/%d/%y')}:**")
        for _, r in date_games.iterrows():
            away_s = get_short_name(r['away'])
            home_s = get_short_name(r['home'])
            label = f"{away_s} @ {home_s} ({r['away_score']}-{r['home_score']})"
            
            if r['gamePk'] == game_pk:
                st.sidebar.markdown(f"▶ **{label}** (current)")
            else:
                if st.sidebar.button(label, key=f"sidebar_{r['gamePk']}"):
                    st.session_state['selected_game_pk'] = int(r['gamePk'])
                    st.rerun()
    else:
        st.sidebar.info("No games found for this date/team.")
    
    st.sidebar.divider()
    st.sidebar.page_link("pages/1_Game_Simulations.py", label="Browse All Games")

    render_home_link()


if __name__ == "__main__":
    main()
