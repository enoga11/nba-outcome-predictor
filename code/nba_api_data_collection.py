import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


SEASONS = ["2022-23", "2023-24", "2024-25"]

RAW_CACHE_DIR = "../data/raw/nba_api_cache"
RAW_OUTPUT_FILE = "../data/raw/nba_team_games_combined.csv"

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]


def ensure_directories():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RAW_OUTPUT_FILE), exist_ok=True)


def clean_team_name(name):
    name = str(name).strip()
    replacements = {
        "Los Angeles Clippers": "Los Angeles Clippers",
        "LA Clippers": "Los Angeles Clippers",
        "L.A. Clippers": "Los Angeles Clippers"
    }
    return replacements.get(name, name)


def get_season_games(season):
    cache_file = os.path.join(RAW_CACHE_DIR, f"{season}_team_games.csv")

    try:
        if os.path.exists(cache_file):
            print(f"Using cached games for {season}")
            return pd.read_csv(cache_file)

        print(f"Downloading games for {season}...")
        games = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T",
            season_nullable=season
        )

        df = games.get_data_frames()[0]

        if df.empty:
            print(f"No games found for {season}")
            return pd.DataFrame()

        df["SEASON"] = season
        df.to_csv(cache_file, index=False)
        print(f"Saved cached games for {season} to {cache_file}")

        time.sleep(1.5)
        return df

    except Exception as e:
        print(f"Error fetching games for {season}: {e}")
        return pd.DataFrame()


def load_all_seasons():
    season_frames = [get_season_games(season) for season in SEASONS]
    season_frames = [df for df in season_frames if not df.empty]

    if not season_frames:
        return pd.DataFrame()

    return pd.concat(season_frames, ignore_index=True)


def clean_raw_data(df):
    df = df.copy()

    required_columns = [
        "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME",
        "MATCHUP", "WL", "PTS", "SEASON"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_NAME"] = df["TEAM_NAME"].apply(clean_team_name)
    df["MATCHUP"] = df["MATCHUP"].astype(str).str.strip()
    df["WL"] = df["WL"].astype(str).str.strip()

    numeric_candidates = [
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["TEAM_NAME"].isin(NBA_TEAMS)].copy()
    df = df.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP", "WL", "PTS", "SEASON"])
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"]).reset_index(drop=True)

    return df


def main():
    ensure_directories()

    print("Loading season data...")
    raw_df = load_all_seasons()

    if raw_df.empty:
        print("No season data was collected. Exiting.")
        return

    print("Cleaning raw data...")
    raw_df = clean_raw_data(raw_df)

    raw_df.to_csv(RAW_OUTPUT_FILE, index=False)
    print(f"Saved combined raw data to {RAW_OUTPUT_FILE}")

    print("\nRaw dataset preview:")
    print(raw_df.head())

    print("\nRaw dataset shape:")
    print(raw_df.shape)

    print("\nRaw dataset columns:")
    print(raw_df.columns.tolist())


if __name__ == "__main__":
    main()
