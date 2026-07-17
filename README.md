# MLB Deserve-to-Win Simulator - Streamlit App

> **⚠️ Sunset notice:** This app has moved to **[dtwbaseball.com](https://dtwbaseball.com)**.
> The Streamlit app shuts down on **July 25, 2026** and `dtw-str.streamlit.app` becomes a
> permanent redirect. This repo stays up to hold the subdomain and serve the redirect page.

Interactive web app for exploring MLB game simulation results.

## Live App

🔗 **[View the live app](https://dtw-str.streamlit.app)** (redirects to [dtwbaseball.com](https://dtwbaseball.com) after July 25, 2026)

## Features

- **Home Page**: Project overview, methodology, and links
- **Game Simulations**: Browse, filter, and view game visualizations
  - Filter by team, date range, season
  - Sort by date, biggest upsets, closest simulations
  - View all 4 visualizations for each game
- **Team Rankings**: Season-long luck metrics with pre-rendered charts
  - Luck differential bar chart (actual wins vs expected wins)
  - Lucky wins vs unlucky losses scatter plot
  - Full data table with CSV download

### Planned Features
- **Playoff Probabilities**: Monte Carlo rest-of-season projections
- **Batted Ball Explorer**: Search individual batted balls by EV/LA/spray angle

## Architecture

Charts on the Team Rankings page are pre-rendered by the [simulator pipeline](https://github.com/dgrifka/baseball_game_simulator) and served as static images from S3. This avoids downloading 60 team logos and running matplotlib on every page load.

## Local Development
```bash
# Clone the repo
git clone https://github.com/dgrifka/dtw-streamlit-app.git
cd dtw-streamlit-app

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run Home.py
```

## Related Projects

- [baseball_game_simulator](https://github.com/dgrifka/baseball_game_simulator) — Core simulation engine (public)

## Author

[Derek Grifka](https://dgrifka.github.io)

## Social

- Twitter: [@mlb_simulator](https://x.com/mlb_simulator)
- Bluesky: [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)
