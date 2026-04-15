import os
import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    boxscoretraditionalv3
)

SEASONS = ["2022-23", "2023-24", "2024-25"]

RAW_CACHE_DIR = "../data/raw/nba_api_games_cache"
RAW_OUTPUT_FILE = "../data/raw/nba_player_games_combined.csv"


def ensure_directories():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RAW_OUTPUT_FILE), exist_ok=True)

def load_games():
    df = pd.read_csv("../data/raw/nba_team_games_combined.csv")
    df = df.dropna(subset=["GAME_ID", "TEAM_ID", "SEASON"])
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    return df[["GAME_ID", "TEAM_ID", "SEASON"]].drop_duplicates()

def get_top_players_by_season():
    top_players = {}

    for season in SEASONS:
        print(f"Fetching player stats for {season}...")

        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        for team_id in stats["TEAM_ID"].unique():
            team_players = stats[stats["TEAM_ID"] == team_id]

            if team_players.empty:
                continue

            top2 = (
                team_players
                .sort_values(by="PTS", ascending=False)
                .head(2)["PLAYER_ID"]
                .astype(str)
                .tolist()
            )

            top_players[(season, team_id)] = top2

        time.sleep(1.5)

    return top_players

def build_star_availability():
    games = load_games()
    top_players_map = get_top_players_by_season()

    game_cache = {}
    results = []

    print("Processing games...")

    for idx, row in games.iterrows():
        game_id = row["GAME_ID"]
        team_id = row["TEAM_ID"]
        season = row["SEASON"]

        key = (season, team_id)
        top_players = top_players_map.get(key, [])

        if game_id not in game_cache:
            try:
                box = boxscoretraditionalv3.BoxScoreTraditionalV3(
                    game_id=game_id
                ).get_data_frames()[0]

                game_cache[game_id] = box
                time.sleep(1.5)

            except Exception as e:
                print(f"Error on game {game_id}: {e}")
                game_cache[game_id] = pd.DataFrame()

        boxscore = game_cache[game_id]

        if boxscore.empty:
            available = 3
        elif not top_players:
            available = 4
        else:
            players_in_game = set(boxscore["personId"].astype(str))
            available = sum(p in players_in_game for p in top_players)


        results.append({
            "GAME_ID": game_id,
            "TEAM_ID": team_id,
            "STAR_AVAILABLE": available  # 0,1,2
        })

        if idx % 200 == 0:
            print(f"Processed {idx} games...")

    return pd.DataFrame(results)


def main():
    ensure_directories()

    df = build_star_availability()

    df.to_csv(RAW_OUTPUT_FILE, index=False)
    print(f"Saved star availability to {RAW_OUTPUT_FILE}")

    print("\nPreview:")
    print(df.head())

    print("\nShape:", df.shape)


if __name__ == "__main__":
    main()