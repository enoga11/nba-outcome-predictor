import pandas as pd
import os
import time
import random
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

seasons = ["2022-23", "2023-24", "2024-25"]

CACHE_DIR = "../data/raw/nba_api_cache"
OUTPUT_FILE = "../data/raw/nba_team_games_combined.csv"
os.makedirs(CACHE_DIR, exist_ok=True)
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    
def get_season_games(season):
    """
    Downloads one season of team game data
    """
    cache_file = os.path.join(CACHE_DIR, f"{season}_team_games.csv")

    try:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            print(f"Using cached games for {season}")
            return df
        
        games = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T",
            season_nullable=season,
        )

        df = games.get_data_frames()[0]

        if df.empty:
            print(f"No games found for {season}")
            return pd.DataFrame()
        
        df["SEASON"] = season

        # Save cache
        df.to_csv(cache_file, index=False)
        print(f"Downloaded games for {season}")

        return df

    except Exception as e:
        print(f"Error fetching games for {season}: {e}")
        return pd.DataFrame()

# Collect all season
season_frames = [get_season_games(season) for season in seasons]
#Remove any empty seasons
season_frames = [df for df in season_frames if not df.empty]

if not season_frames:
    print("No season data was collected. Exiting.")
else:
    df = pd.concat(season_frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Remove repeated header rows if they appear
    if "Date" in df.columns:
        df = df[df["Date"] != "Date"]

    # Clean team names
    if "TEAM_NAME" in df.columns:
        df["TEAM_NAME"] = df["TEAM_NAME"].astype(str).str.strip()

        df["TEAM_NAME"] = df["TEAM_NAME"].replace({
            "Los Angeles Clippers": "LA Clippers",
            "L.A. Clippers": "LA Clippers",
        })

    # Keep only completed games with both scores present
    if "PTS" in df.columns and "PTS.1" in df.columns:
        df = df.dropna(subset=["PTS", "PTS.1"])

    # get rid of some teams that are not relevant to our analysis (like "NBA Development League")
    relevant_teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
        "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
        "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
        "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",  
        "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
    ]   
    df = df[df["TEAM_NAME"].isin(relevant_teams)].reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)

print(df.head())
print(df.columns.tolist())

# print all the team names in the dataset
print("Teams in dataset:")
print(df["TEAM_NAME"].unique())
