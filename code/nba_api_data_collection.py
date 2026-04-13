import os
import time
from collections import defaultdict, deque

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


SEASONS = ["2022-23", "2023-24", "2024-25"]

RAW_CACHE_DIR = "../data/raw/nba_api_cache"
RAW_OUTPUT_FILE = "../data/raw/nba_team_games_combined.csv"
PROCESSED_OUTPUT_FILE = "../data/processed/games_with_features.csv"

NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]


def ensure_directories():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RAW_OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)


def clean_team_name(name):
    name = str(name).strip()
    replacements = {
        "Los Angeles Clippers": "LA Clippers",
        "L.A. Clippers": "LA Clippers"
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
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")

    df = df[df["TEAM_NAME"].isin(NBA_TEAMS)].copy()
    df = df.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP", "WL", "PTS", "SEASON"])

    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"]).reset_index(drop=True)
    return df


def build_game_level_dataset(team_games_df):
    df = team_games_df.copy()

    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.", regex=True, na=False)
    df["IS_AWAY"] = df["MATCHUP"].str.contains("@", regex=False, na=False)

    home_df = df[df["IS_HOME"]].copy()
    away_df = df[df["IS_AWAY"]].copy()

    home_df = home_df.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_NAME": "HOME_TEAM_NAME",
        "PTS": "HOME_PTS",
        "WL": "HOME_WL",
        "MATCHUP": "HOME_MATCHUP"
    })

    away_df = away_df.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_NAME": "AWAY_TEAM_NAME",
        "PTS": "AWAY_PTS",
        "WL": "AWAY_WL",
        "MATCHUP": "AWAY_MATCHUP"
    })

    home_keep = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "HOME_TEAM_NAME", "HOME_PTS", "HOME_WL"
    ]
    away_keep = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "AWAY_TEAM_ID", "AWAY_TEAM_NAME", "AWAY_PTS", "AWAY_WL"
    ]

    games_df = pd.merge(
        home_df[home_keep],
        away_df[away_keep],
        on=["GAME_ID", "GAME_DATE", "SEASON"],
        how="inner"
    )

    games_df = games_df.drop_duplicates(subset=["GAME_ID"]).copy()
    games_df["HOME_PTS"] = pd.to_numeric(games_df["HOME_PTS"], errors="coerce")
    games_df["AWAY_PTS"] = pd.to_numeric(games_df["AWAY_PTS"], errors="coerce")
    games_df = games_df.dropna(subset=["HOME_PTS", "AWAY_PTS"]).copy()

    games_df["HOME_TEAM_WINS"] = (games_df["HOME_PTS"] > games_df["AWAY_PTS"]).astype(int)

    games_df = games_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return games_df


def add_last10_features(games_df):
    games_df = games_df.copy()
    games_df = games_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    team_history = defaultdict(lambda: deque(maxlen=10))
    team_home_history = defaultdict(lambda: deque(maxlen=10))
    team_away_history = defaultdict(lambda: deque(maxlen=10))

    home_last10_win_pct = []
    away_last10_win_pct = []

    home_last10_opp_avg_win_pct = []
    away_last10_opp_avg_win_pct = []

    home_last10_adjusted = []
    away_last10_adjusted = []

    home_last10_home_win_pct = []
    away_last10_away_win_pct = []

    for _, row in games_df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        home_hist = list(team_history[home_team])
        away_hist = list(team_history[away_team])

        home_home_hist = list(team_home_history[home_team])
        away_away_hist = list(team_away_history[away_team])

        if len(home_hist) > 0:
            home_win_pct = sum(game["win"] for game in home_hist) / len(home_hist)
            home_opp_avg = sum(game["opp_pregame_win_pct"] for game in home_hist) / len(home_hist)
        else:
            home_win_pct = 0.5
            home_opp_avg = 0.5

        if len(away_hist) > 0:
            away_win_pct = sum(game["win"] for game in away_hist) / len(away_hist)
            away_opp_avg = sum(game["opp_pregame_win_pct"] for game in away_hist) / len(away_hist)
        else:
            away_win_pct = 0.5
            away_opp_avg = 0.5

        if len(home_home_hist) > 0:
            current_home_last10_home_win_pct = (
                sum(game["win"] for game in home_home_hist) / len(home_home_hist)
            )
        else:
            current_home_last10_home_win_pct = 0.5

        if len(away_away_hist) > 0:
            current_away_last10_away_win_pct = (
                sum(game["win"] for game in away_away_hist) / len(away_away_hist)
            )
        else:
            current_away_last10_away_win_pct = 0.5

        home_adjusted = home_win_pct + (home_opp_avg - 0.5)
        away_adjusted = away_win_pct + (away_opp_avg - 0.5)

        home_last10_win_pct.append(home_win_pct)
        away_last10_win_pct.append(away_win_pct)
        home_last10_opp_avg_win_pct.append(home_opp_avg)
        away_last10_opp_avg_win_pct.append(away_opp_avg)
        home_last10_adjusted.append(home_adjusted)
        away_last10_adjusted.append(away_adjusted)

        home_last10_home_win_pct.append(current_home_last10_home_win_pct)
        away_last10_away_win_pct.append(current_away_last10_away_win_pct)

        home_won = int(row["HOME_TEAM_WINS"] == 1)
        away_won = 1 - home_won

        home_entry = {
            "win": home_won,
            "opp_pregame_win_pct": away_win_pct
        }
        away_entry = {
            "win": away_won,
            "opp_pregame_win_pct": home_win_pct
        }

        team_history[home_team].append(home_entry)
        team_history[away_team].append(away_entry)

        team_home_history[home_team].append({"win": home_won})
        team_away_history[away_team].append({"win": away_won})

    games_df["home_last10_win_pct"] = home_last10_win_pct
    games_df["away_last10_win_pct"] = away_last10_win_pct
    games_df["last10_win_pct_diff"] = (
        games_df["home_last10_win_pct"] - games_df["away_last10_win_pct"]
    )

    games_df["home_last10_opponent_avg_win_pct"] = home_last10_opp_avg_win_pct
    games_df["away_last10_opponent_avg_win_pct"] = away_last10_opp_avg_win_pct
    games_df["last10_opponent_avg_win_pct_diff"] = (
        games_df["home_last10_opponent_avg_win_pct"] -
        games_df["away_last10_opponent_avg_win_pct"]
    )

    games_df["home_last10_adjusted"] = home_last10_adjusted
    games_df["away_last10_adjusted"] = away_last10_adjusted
    games_df["last10_adjusted_diff"] = (
        games_df["home_last10_adjusted"] - games_df["away_last10_adjusted"]
    )

    games_df["home_last10_home_win_pct"] = home_last10_home_win_pct
    games_df["away_last10_away_win_pct"] = away_last10_away_win_pct
    games_df["home_away_form_diff"] = (
        games_df["home_last10_home_win_pct"] - games_df["away_last10_away_win_pct"]
    )

    return games_df

def validate_games_dataset(games_df):
    if games_df.empty:
        raise ValueError("Processed dataset is empty.")

    if games_df["GAME_ID"].duplicated().any():
        duplicate_ids = games_df.loc[games_df["GAME_ID"].duplicated(), "GAME_ID"].tolist()
        raise ValueError(f"Duplicate GAME_ID values found: {duplicate_ids[:10]}")

    required_columns = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_NAME",
        "HOME_PTS",
        "AWAY_PTS",
        "HOME_TEAM_WINS",
        "home_last10_win_pct",
        "away_last10_win_pct",
        "last10_win_pct_diff",
        "home_last10_opponent_avg_win_pct",
        "away_last10_opponent_avg_win_pct",
        "home_last10_adjusted",
        "away_last10_adjusted",
        "last10_adjusted_diff",
        "home_last10_home_win_pct",
        "away_last10_away_win_pct",
        "home_away_form_diff"
    ]

    missing = [col for col in required_columns if col not in games_df.columns]
    if missing:
        raise ValueError(f"Missing required processed columns: {missing}")

    if games_df[required_columns].isnull().any().any():
        null_counts = games_df[required_columns].isnull().sum()
        raise ValueError(f"Null values found in required processed columns:\n{null_counts}")


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

    print("Building one-row-per-game dataset...")
    games_df = build_game_level_dataset(raw_df)

    print("Adding last-10 features...")
    games_df = add_last10_features(games_df)

    print("Validating processed dataset...")
    validate_games_dataset(games_df)

    games_df.to_csv(PROCESSED_OUTPUT_FILE, index=False)
    print(f"Saved processed data to {PROCESSED_OUTPUT_FILE}")

    print("\nProcessed dataset preview:")
    print(games_df.head())

    print("\nProcessed dataset columns:")
    print(games_df.columns.tolist())

    print("\nProcessed dataset shape:")
    print(games_df.shape)

    print("\nMissing values per column:")
    print(games_df.isnull().sum())

    print("\nSample feature rows:")
    print(
        games_df[
            [
                "GAME_DATE",
                "HOME_TEAM_NAME",
                "AWAY_TEAM_NAME",
                "HOME_TEAM_WINS",
                "home_last10_win_pct",
                "away_last10_win_pct",
                "last10_win_pct_diff",
                "home_last10_home_win_pct",
                "away_last10_away_win_pct",
                "home_away_form_diff",
                "home_last10_adjusted",
                "away_last10_adjusted",
                "last10_adjusted_diff"
            ]
        ].head(10)
    )

if __name__ == "__main__":
    main()