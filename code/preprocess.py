# code/preprocess.py

import pandas as pd


def main():
    # Load raw Kaggle files
    games = pd.read_csv("../data/raw/games.csv")
    teams = pd.read_csv("../data/raw/teams.csv")

    # Print columns once so you can verify the dataset structure
    print("games.csv columns:")
    print(games.columns.tolist())
    print()
    print("teams.csv columns:")
    print(teams.columns.tolist())
    print()

    # Keep only the columns we need for the initial project pipeline
    # These are the core game-level fields:
    # - game date
    # - home team id
    # - visitor team id
    # - home/away points
    # - home team win label
    games = games[
        [
            "GAME_DATE_EST",
            "HOME_TEAM_ID",
            "VISITOR_TEAM_ID",
            "PTS_home",
            "PTS_away",
            "HOME_TEAM_WINS",
        ]
    ].copy()

    # Drop rows with missing critical values
    games = games.dropna(
        subset=[
            "GAME_DATE_EST",
            "HOME_TEAM_ID",
            "VISITOR_TEAM_ID",
            "PTS_home",
            "PTS_away",
            "HOME_TEAM_WINS",
        ]
    )

    # Convert date column to datetime
    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])

    # Make sure IDs and label are integers
    games["HOME_TEAM_ID"] = games["HOME_TEAM_ID"].astype(int)
    games["VISITOR_TEAM_ID"] = games["VISITOR_TEAM_ID"].astype(int)
    games["HOME_TEAM_WINS"] = games["HOME_TEAM_WINS"].astype(int)

    # Keep only the needed team mapping columns
    teams = teams[["TEAM_ID", "NICKNAME", "CITY", "ABBREVIATION"]].copy()

    # Make sure TEAM_ID is integer
    teams["TEAM_ID"] = teams["TEAM_ID"].astype(int)

    # Merge home team information
    games = games.merge(
        teams,
        left_on="HOME_TEAM_ID",
        right_on="TEAM_ID",
        how="left"
    )

    games = games.rename(
        columns={
            "NICKNAME": "HOME_TEAM_NAME",
            "CITY": "HOME_TEAM_CITY",
            "ABBREVIATION": "HOME_TEAM_ABBREVIATION",
        }
    )

    games = games.drop(columns=["TEAM_ID"])

    # Merge away team information
    games = games.merge(
        teams,
        left_on="VISITOR_TEAM_ID",
        right_on="TEAM_ID",
        how="left"
    )

    games = games.rename(
        columns={
            "NICKNAME": "AWAY_TEAM_NAME",
            "CITY": "AWAY_TEAM_CITY",
            "ABBREVIATION": "AWAY_TEAM_ABBREVIATION",
        }
    )

    games = games.drop(columns=["TEAM_ID"])

    # Sort by date so future feature engineering only uses past games
    games = games.sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Save cleaned dataset
    output_path = "../data/processed/games_cleaned.csv"
    games.to_csv(output_path, index=False)

    print("Preprocessing complete.")
    print(f"Saved cleaned data to: {output_path}")
    print()
    print("Cleaned dataset shape:")
    print(games.shape)
    print()
    print("First 5 rows:")
    print(games.head())


if __name__ == "__main__":
    main()