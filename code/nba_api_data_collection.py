import pandas as pd
import os
import time
import random
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

seasons = ["2022-23", "2023-24", "2024-25"]

CACHE_DIR = "../data/raw/nba_api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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

#Collect all season
season_frames = [get_season_games(season) for season in seasons]
#Remove any empty seasons
season_frames = [df for df in season_frames if not df.empty]

df = pd.concat(season_frames, ignore_index=True)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# Remove repeated header rows if they appear
if "Date" in df.columns:
    df = df[df["Date"] != "Date"]

# Keep only completed games with both scores present
if "PTS" in df.columns and "PTS.1" in df.columns:
    df = df.dropna(subset=["PTS", "PTS.1"])

df.to_csv("../data/raw/nba_team_games_combined.csv", index=False)

print(df.head())
print(df.columns.tolist())

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

# print all the team names in the dataset
print("Teams in dataset:")
print(df["TEAM_NAME"].unique())
