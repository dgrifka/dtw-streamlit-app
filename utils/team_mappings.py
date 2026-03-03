"""
Team name mapping utilities for the Streamlit app.
Maps between full team names (from parquet) and short names (used in image filenames).
"""

# Full team name -> Short name (for image filenames)
TEAM_NAME_MAPPING = {
    'Arizona Diamondbacks': 'D-backs',
    'Atlanta Braves': 'Braves',
    'Baltimore Orioles': 'Orioles',
    'Boston Red Sox': 'Red Sox',
    'Chicago White Sox': 'White Sox',
    'Chicago Cubs': 'Cubs',
    'Cincinnati Reds': 'Reds',
    'Cleveland Guardians': 'Guardians',
    'Colorado Rockies': 'Rockies',
    'Detroit Tigers': 'Tigers',
    'Houston Astros': 'Astros',
    'Kansas City Royals': 'Royals',
    'Los Angeles Angels': 'Angels',
    'Los Angeles Dodgers': 'Dodgers',
    'Miami Marlins': 'Marlins',
    'Milwaukee Brewers': 'Brewers',
    'Minnesota Twins': 'Twins',
    'New York Yankees': 'Yankees',
    'New York Mets': 'Mets',
    'Oakland Athletics': 'Athletics',
    'Philadelphia Phillies': 'Phillies',
    'Pittsburgh Pirates': 'Pirates',
    'San Diego Padres': 'Padres',
    'San Francisco Giants': 'Giants',
    'Seattle Mariners': 'Mariners',
    'St. Louis Cardinals': 'Cardinals',
    'Tampa Bay Rays': 'Rays',
    'Texas Rangers': 'Rangers',
    'Toronto Blue Jays': 'Blue Jays',
    'Washington Nationals': 'Nationals',
}

# Reverse mapping: Short name -> Full name
SHORT_TO_FULL = {v: k for k, v in TEAM_NAME_MAPPING.items()}

# Team colors for UI styling (primary, secondary)
TEAM_COLORS = {
    'D-backs': ('#A71930', '#E3D4AD'),
    'Braves': ('#CE1141', '#13274F'),
    'Orioles': ('#DF4601', '#000000'),
    'Red Sox': ('#BD3039', '#0C2340'),
    'Cubs': ('#0E3386', '#CC3433'),
    'White Sox': ('#27251F', '#C4CED4'),
    'Reds': ('#C6011F', '#000000'),
    'Guardians': ('#00385D', '#E50022'),
    'Rockies': ('#33006F', '#C4CED4'),
    'Tigers': ('#0C2340', '#FA4616'),
    'Astros': ('#002D62', '#EB6E1F'),
    'Royals': ('#004687', '#BD9B60'),
    'Angels': ('#BA0021', '#003263'),
    'Dodgers': ('#005A9C', '#EF3E42'),
    'Marlins': ('#00A3E0', '#EF3340'),
    'Brewers': ('#0A2351', '#B6922E'),
    'Twins': ('#002B5C', '#D31145'),
    'Mets': ('#002D72', '#FF5910'),
    'Yankees': ('#003087', '#E4002C'),
    'Athletics': ('#003831', '#EFB21E'),
    'Phillies': ('#E81828', '#002D72'),
    'Pirates': ('#27251F', '#FDB827'),
    'Padres': ('#2F241D', '#FFC425'),
    'Giants': ('#FD5A1E', '#27251F'),
    'Mariners': ('#0C2C56', '#005C5C'),
    'Cardinals': ('#C41E3A', '#0C2340'),
    'Rays': ('#092C5C', '#8FBCE6'),
    'Rangers': ('#003278', '#C0111F'),
    'Blue Jays': ('#134A8E', '#1D2D5C'),
    'Nationals': ('#AB0003', '#14225A'),
}

# Team logo URLs (ESPN CDN - reliable and consistent sizing)
TEAM_LOGOS = {
    'D-backs': 'https://a.espncdn.com/i/teamlogos/mlb/500/ari.png',
    'Braves': 'https://a.espncdn.com/i/teamlogos/mlb/500/atl.png',
    'Orioles': 'https://a.espncdn.com/i/teamlogos/mlb/500/bal.png',
    'Red Sox': 'https://a.espncdn.com/i/teamlogos/mlb/500/bos.png',
    'Cubs': 'https://a.espncdn.com/i/teamlogos/mlb/500/chc.png',
    'White Sox': 'https://a.espncdn.com/i/teamlogos/mlb/500/chw.png',
    'Reds': 'https://a.espncdn.com/i/teamlogos/mlb/500/cin.png',
    'Guardians': 'https://a.espncdn.com/i/teamlogos/mlb/500/cle.png',
    'Rockies': 'https://a.espncdn.com/i/teamlogos/mlb/500/col.png',
    'Tigers': 'https://a.espncdn.com/i/teamlogos/mlb/500/det.png',
    'Astros': 'https://a.espncdn.com/i/teamlogos/mlb/500/hou.png',
    'Royals': 'https://a.espncdn.com/i/teamlogos/mlb/500/kc.png',
    'Angels': 'https://a.espncdn.com/i/teamlogos/mlb/500/laa.png',
    'Dodgers': 'https://a.espncdn.com/i/teamlogos/mlb/500/lad.png',
    'Marlins': 'https://a.espncdn.com/i/teamlogos/mlb/500/mia.png',
    'Brewers': 'https://a.espncdn.com/i/teamlogos/mlb/500/mil.png',
    'Twins': 'https://a.espncdn.com/i/teamlogos/mlb/500/min.png',
    'Mets': 'https://a.espncdn.com/i/teamlogos/mlb/500/nym.png',
    'Yankees': 'https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png',
    'Athletics': 'https://a.espncdn.com/i/teamlogos/mlb/500/oak.png',
    'Phillies': 'https://a.espncdn.com/i/teamlogos/mlb/500/phi.png',
    'Pirates': 'https://a.espncdn.com/i/teamlogos/mlb/500/pit.png',
    'Padres': 'https://a.espncdn.com/i/teamlogos/mlb/500/sd.png',
    'Giants': 'https://a.espncdn.com/i/teamlogos/mlb/500/sf.png',
    'Mariners': 'https://a.espncdn.com/i/teamlogos/mlb/500/sea.png',
    'Cardinals': 'https://a.espncdn.com/i/teamlogos/mlb/500/stl.png',
    'Rays': 'https://a.espncdn.com/i/teamlogos/mlb/500/tb.png',
    'Rangers': 'https://a.espncdn.com/i/teamlogos/mlb/500/tex.png',
    'Blue Jays': 'https://a.espncdn.com/i/teamlogos/mlb/500/tor.png',
    'Nationals': 'https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png',
}


def get_team_logo_url(team_name: str) -> str:
    """
    Get the logo URL for a team.
    Accepts either full name ('New York Yankees') or short name ('Yankees').
    
    Example:
        get_team_logo_url('Yankees') -> 'https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png'
        get_team_logo_url('New York Yankees') -> 'https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png'
    """
    # If it's a full name, convert to short name first
    short_name = TEAM_NAME_MAPPING.get(team_name, team_name)
    return TEAM_LOGOS.get(short_name, '')

def get_short_name(full_name: str) -> str:
    """Convert full team name to short name for image filenames."""
    return TEAM_NAME_MAPPING.get(full_name, full_name)


def get_full_name(short_name: str) -> str:
    """Convert short team name back to full name."""
    return SHORT_TO_FULL.get(short_name, short_name)


def get_all_teams() -> list:
    """Return sorted list of all full team names."""
    return sorted(TEAM_NAME_MAPPING.keys())


def get_team_color(team_name: str) -> tuple:
    """Get team colors (primary, secondary) for UI styling."""
    short = get_short_name(team_name)
    return TEAM_COLORS.get(short, ('#1f77b4', '#ffffff'))
