import os
import time
import random
import pandas as pd
import requests

months = [
    "october", "november", "december",
    "january", "february", "march", "april", "may", "june"
]

# Basketball Reference uses season END year in the URL
# Example: 2022-23 season -> 2023
seasons = [2023, 2024, 2025]

# Folder to cache each monthly HTML page
CACHE_DIR = "../data/raw/month_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

def get_month_games(season_end_year, month):
    """
    Download one month of one season.
    If the month was already saved locally, reuse it instead of requesting again.
    """
    cache_file = os.path.join(CACHE_DIR, f"{season_end_year}_{month}.html")
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_games-{month}.html"

    try:
        # Use cached copy if it already exists
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                html = f.read()
            print(f"Using cached {month} {season_end_year}")
        else:
            response = session.get(url, timeout=20)

            if response.status_code == 429:
                print(f"Rate limited on {month} {season_end_year}")
                return pd.DataFrame()

            if response.status_code != 200:
                print(f"Failed {month} {season_end_year}: HTTP {response.status_code}")
                return pd.DataFrame()

            html = response.text

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"Downloaded {month} {season_end_year}")

            # Sleep so requests are not too fast
            time.sleep(random.uniform(4, 7))

        tables = pd.read_html(html)

        if not tables:
            print(f"No table found for {month} {season_end_year}")
            return pd.DataFrame()

        df = tables[0]
        df["season"] = season_end_year
        df["month"] = month
        return df

    except Exception as e:
        print(f"Error for {month} {season_end_year}: {e}")
        return pd.DataFrame()


def get_season_games(season_end_year):
    """
    Combine all months for one season into a single DataFrame.
    """
    all_games = []

    for month in months:
        df = get_month_games(season_end_year, month)
        if not df.empty:
            all_games.append(df)

    if all_games:
        return pd.concat(all_games, ignore_index=True)

    return pd.DataFrame()


season_frames = [get_season_games(year) for year in seasons]
season_frames = [df for df in season_frames if not df.empty]

if season_frames:
    df = pd.concat(season_frames, ignore_index=True)

    # Remove repeated header rows if they appear
    if "Date" in df.columns:
        df = df[df["Date"] != "Date"]

    # Keep only completed games with both scores present
    if "PTS" in df.columns and "PTS.1" in df.columns:
        df = df.dropna(subset=["PTS", "PTS.1"])

    df.to_csv("../data/raw/nba_games_combined.csv", index=False)
    print("Done! Shape:", df.shape)
else:
    print("No data collected.")