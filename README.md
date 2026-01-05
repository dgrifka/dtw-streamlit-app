# MLB Deserve-to-Win Simulator - Streamlit App

Interactive web app for exploring MLB game simulation results.

## Live App

🔗 **[View the live app](https://your-app.streamlit.app)** *(Update with actual URL after deployment)*

## Features

- **Home Page**: Project overview, methodology, and links
- **Game Simulations**: Browse, filter, and view game visualizations
  - Filter by team, date range, season
  - Sort by date, biggest upsets, closest simulations
  - View all 4 visualizations for each game

## Data Sources

All data is loaded from a public S3 bucket — no credentials required:

- **Game Summaries**: `https://dtw-streamlit.s3.amazonaws.com/data/game_summaries.parquet`
- **Images**: `https://dtw-streamlit.s3.amazonaws.com/sim-images/{gamePk}/{filename}.png`

Data is automatically updated after each game during the MLB season.

## Local Development

```bash
# Clone the repo
git clone https://github.com/dgrifka/dtw-streamlit-app.git
cd dtw-streamlit-app

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select `app.py` as the main file
5. Deploy!

No secrets or credentials needed — all data is public.

## Project Structure

```
dtw-streamlit-app/
├── app.py                      # Home page
├── pages/
│   └── 1_Game_Simulations.py   # Game browser
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # S3 data loading & caching
│   └── team_mappings.py        # Team name conversions
├── .streamlit/
│   └── config.toml             # Theme configuration
├── requirements.txt
└── README.md
```

## Related Projects

- [baseball_game_simulator](https://github.com/dgrifka/baseball_game_simulator) — Core simulation engine (public)

## Author

[Derek Grifka](https://dgrifka.github.io)

## Social

- Twitter: [@mlb_simulator](https://x.com/mlb_simulator)
- Bluesky: [@mlb-simulator.bsky.social](https://bsky.app/profile/mlb-simulator.bsky.social)