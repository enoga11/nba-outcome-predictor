import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_DIR = os.path.join(ROOT_DIR, "api")
CODE_DIR = os.path.join(ROOT_DIR, "code")

sys.path.insert(0, API_DIR)
sys.path.insert(0, CODE_DIR)

from feature_engineering import (
    add_efg_features,
    add_last10_features,
    add_matchup_features,
    add_net_rating_and_turnover_features,
    add_rest_features,
    add_scoring_features,
    build_game_level_dataset,
    clean_team_name,
)

try:
    from nba_api.stats.endpoints import leaguegamefinder
except Exception as exc:
    print("REAL IMPORT ERROR:", repr(exc))
    raise

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

LIVE_DIR = os.path.join(ROOT_DIR, "data", "live")
LIVE_DATA_FILE = os.path.join(LIVE_DIR, "live_games_with_features.csv")
LIVE_METADATA_FILE = os.path.join(LIVE_DIR, "metadata.json")


def get_default_live_season():
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}-{end_year}"


def prepare_live_team_games(raw_df):
    df = raw_df.copy()
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

    df = df[df["TEAM_NAME"].isin(set(NBA_TEAMS))].copy()
    df = df.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP", "WL", "PTS", "SEASON"])
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"]).reset_index(drop=True)


def build_live_feature_dataset():
    season = get_default_live_season()
    raw_df = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season
    ).get_data_frames()[0]

    if raw_df.empty:
        raise RuntimeError(f"NBA API returned no games for season {season}.")
    if "SEASON" not in raw_df.columns:
        raw_df["SEASON"] = season

    team_games_df = prepare_live_team_games(raw_df)
    games_df = build_game_level_dataset(team_games_df)
    games_df = add_last10_features(games_df)
    games_df = add_rest_features(games_df)
    games_df = add_scoring_features(games_df)
    games_df = add_matchup_features(games_df)
    games_df = add_efg_features(games_df)
    games_df = add_net_rating_and_turnover_features(games_df)
    return games_df, season


def main():
    os.makedirs(LIVE_DIR, exist_ok=True)
    games_df, season = build_live_feature_dataset()
    games_df.to_csv(LIVE_DATA_FILE, index=False)

    metadata = {
        "source": "local live snapshot",
        "season": season,
        "row_count": int(len(games_df)),
        "latest_game_date": games_df["GAME_DATE"].max().strftime("%Y-%m-%d") if not games_df.empty else None,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "data_file": "data/live/live_games_with_features.csv",
    }
    with open(LIVE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {LIVE_DATA_FILE}")
    print(f"Wrote {LIVE_METADATA_FILE}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()