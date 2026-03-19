import pandas as pd

df = pd.read_csv("../data/processed/games_cleaned.csv")

# Make sure data is sorted
df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

# Dictionaries to track team history
team_games = {}
team_wins = {}

home_win_pct = []
away_win_pct = []

for _, row in df.iterrows():
    home = row["HOME_TEAM_ID"]
    away = row["VISITOR_TEAM_ID"]

    # Get current stats BEFORE this game
    home_games = team_games.get(home, 0)
    home_wins_count = team_wins.get(home, 0)

    away_games = team_games.get(away, 0)
    away_wins_count = team_wins.get(away, 0)

    # Compute win %
    home_pct = home_wins_count / home_games if home_games > 0 else 0.5
    away_pct = away_wins_count / away_games if away_games > 0 else 0.5

    home_win_pct.append(home_pct)
    away_win_pct.append(away_pct)

    # Update AFTER computing feature
    team_games[home] = home_games + 1
    team_games[away] = away_games + 1

    if row["HOME_TEAM_WINS"] == 1:
        team_wins[home] = home_wins_count + 1
        team_wins[away] = away_wins_count
    else:
        team_wins[away] = away_wins_count + 1
        team_wins[home] = home_wins_count

# Add new features
df["home_win_pct"] = home_win_pct
df["away_win_pct"] = away_win_pct

# Difference in win percentage (home - away)
df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]

# Save new dataset
df.to_csv("../data/processed/games_with_features.csv", index=False)

print("Feature engineering complete.")
print(df[[
    "GAME_DATE_EST",
    "home_win_pct",
    "away_win_pct",
    "HOME_TEAM_WINS"
]].head())

print(df[["home_win_pct", "away_win_pct", "win_pct_diff"]].iloc[100:110])