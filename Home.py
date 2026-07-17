"""
MLB "Deserve to Win" Simulator - Streamlit App (SUNSET)

This app moved to https://dtwbaseball.com on July 25, 2026. Every old page
URL still resolves and redirects to its closest equivalent on the new site.
The app stays deployed to hold the dtw-str subdomain and serve the redirect.
"""

import streamlit as st

SITE = "https://dtwbaseball.com"

st.set_page_config(
    page_title="Moved to dtwbaseball.com",
    page_icon="⚾",
    layout="centered",
)


def tombstone(target: str, note: str | None = None):
    """Render the moved-notice page and auto-redirect to `target`.

    Meta-refresh is the only auto-redirect that works on Community Cloud:
    st.markdown HTML lands in the main document (scripts don't execute, and
    components.html iframes are sandboxed without allow-top-navigation).
    """
    st.markdown(
        f"<meta http-equiv='refresh' content='3; url={target}'>",
        unsafe_allow_html=True,
    )
    st.title("⚾ Down to the Wire has moved")
    st.markdown(
        f"This app now lives at **[dtwbaseball.com]({SITE})** — same simulations, "
        "standings, and player pages, updated daily."
    )
    if note:
        st.info(note)
    st.markdown(f"Taking you to **{target}** in a few seconds…")
    st.link_button("Take me there →", target)


def _page(url_path: str, target: str, title: str, note: str | None = None, **kwargs):
    def _render():
        tombstone(target, note)

    return st.Page(_render, title=title, url_path=url_path, **kwargs)


def _home():
    tombstone(SITE)


def _game_detail():
    game_pk = st.query_params.get("gamePk")
    target = f"{SITE}/games/{game_pk}" if game_pk else f"{SITE}/games"
    tombstone(target)


COMPARISON_NOTE = (
    "Side-by-side player comparison isn't on the new site yet — it's on the "
    "roadmap. The player pages there cover the same profiles in more depth."
)

nav = st.navigation(
    [
        st.Page(_home, title="Home", default=True),
        st.Page(_game_detail, title="Game Detail", url_path="Game_Detail"),
        _page("Game_Simulations", f"{SITE}/games", "Game Simulations"),
        _page("Team_Luck_Rankings", f"{SITE}/teams", "Team Luck Rankings"),
        _page("Playoff_Probabilities", f"{SITE}/standings", "Playoff Probabilities"),
        _page("Batted_Ball_Explorer", f"{SITE}/tools/batted-ball-explorer", "Batted Ball Explorer"),
        _page("Player_Rankings", f"{SITE}/hitters", "Player Rankings"),
        _page("Hitter_Profile", f"{SITE}/hitters", "Hitter Profile"),
        _page("Hitter_Comparison", f"{SITE}/hitters", "Hitter Comparison", note=COMPARISON_NOTE),
        _page("Pitcher_Profile", f"{SITE}/pitchers", "Pitcher Profile"),
        _page("Pitcher_Comparison", f"{SITE}/pitchers", "Pitcher Comparison", note=COMPARISON_NOTE),
        _page("About", f"{SITE}/about", "About"),
    ],
    position="hidden",
)

nav.run()
