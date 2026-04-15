import os
from collections import defaultdict, deque

import pandas as pd


RAW_INPUT_FILE = "../data/raw/nba_team_games_combined.csv"
PROCESSED_OUTPUT_FILE = "../data/processed/games_with_features.csv"

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
    os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)


def clean_team_name(name):
    name = str(name).strip()
    replacements = {
        "Los Angeles Clippers": "Los Angeles Clippers",
        "L.A. Clippers": "Los Angeles Clippers"
    }
    return replacements.get(name, name)


def load_raw_data():
    if not os.path.exists(RAW_INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {RAW_INPUT_FILE}. Run nba_api_data_collection.py first."
        )

    df = pd.read_csv(RAW_INPUT_FILE)
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


def build_game_level_dataset(team_games_df):
    df = team_games_df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.", regex=True, na=False)
    df["IS_AWAY"] = df["MATCHUP"].str.contains("@", regex=False, na=False)

    home_df = df[df["IS_HOME"]].copy()
    away_df = df[df["IS_AWAY"]].copy()

    home_rename = {
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_NAME": "HOME_TEAM_NAME",
        "PTS": "HOME_PTS",
        "WL": "HOME_WL"
    }
    away_rename = {
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_NAME": "AWAY_TEAM_NAME",
        "PTS": "AWAY_PTS",
        "WL": "AWAY_WL"
    }

    stats_to_keep = ["FGM", "FGA", "FG3M", "FG3A", "OREB", "DREB", "FTM", "FTA", "REB", "AST", "TOV", "PLUS_MINUS"]
    for col in stats_to_keep:
        if col in df.columns:
            home_rename[col] = f"HOME_{col}"
            away_rename[col] = f"AWAY_{col}"

    home_df = home_df.rename(columns=home_rename)
    away_df = away_df.rename(columns=away_rename)

    home_keep = ["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID", "HOME_TEAM_NAME", "HOME_PTS", "HOME_WL"]
    away_keep = ["GAME_ID", "GAME_DATE", "SEASON", "AWAY_TEAM_ID", "AWAY_TEAM_NAME", "AWAY_PTS", "AWAY_WL"]

    for col in stats_to_keep:
        home_col = f"HOME_{col}"
        away_col = f"AWAY_{col}"
        if home_col in home_df.columns:
            home_keep.append(home_col)
        if away_col in away_df.columns:
            away_keep.append(away_col)

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

        if home_hist:
            home_win_pct = sum(game["win"] for game in home_hist) / len(home_hist)
            home_opp_avg = sum(game["opp_pregame_win_pct"] for game in home_hist) / len(home_hist)
        else:
            home_win_pct = 0.5
            home_opp_avg = 0.5

        if away_hist:
            away_win_pct = sum(game["win"] for game in away_hist) / len(away_hist)
            away_opp_avg = sum(game["opp_pregame_win_pct"] for game in away_hist) / len(away_hist)
        else:
            away_win_pct = 0.5
            away_opp_avg = 0.5

        if home_home_hist:
            current_home_last10_home_win_pct = sum(game["win"] for game in home_home_hist) / len(home_home_hist)
        else:
            current_home_last10_home_win_pct = 0.5

        if away_away_hist:
            current_away_last10_away_win_pct = sum(game["win"] for game in away_away_hist) / len(away_away_hist)
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

        team_history[home_team].append({
            "win": home_won,
            "opp_pregame_win_pct": away_win_pct
        })
        team_history[away_team].append({
            "win": away_won,
            "opp_pregame_win_pct": home_win_pct
        })

        team_home_history[home_team].append({"win": home_won})
        team_away_history[away_team].append({"win": away_won})

    games_df["home_last10_win_pct"] = home_last10_win_pct
    games_df["away_last10_win_pct"] = away_last10_win_pct
    games_df["last10_win_pct_diff"] = games_df["home_last10_win_pct"] - games_df["away_last10_win_pct"]

    games_df["home_last10_opponent_avg_win_pct"] = home_last10_opp_avg_win_pct
    games_df["away_last10_opponent_avg_win_pct"] = away_last10_opp_avg_win_pct
    games_df["last10_opponent_avg_win_pct_diff"] = (
        games_df["home_last10_opponent_avg_win_pct"] - games_df["away_last10_opponent_avg_win_pct"]
    )

    games_df["home_last10_adjusted"] = home_last10_adjusted
    games_df["away_last10_adjusted"] = away_last10_adjusted
    games_df["last10_adjusted_diff"] = games_df["home_last10_adjusted"] - games_df["away_last10_adjusted"]

    games_df["home_last10_home_win_pct"] = home_last10_home_win_pct
    games_df["away_last10_away_win_pct"] = away_last10_away_win_pct
    games_df["home_away_form_diff"] = games_df["home_last10_home_win_pct"] - games_df["away_last10_away_win_pct"]

    return games_df


def add_rest_features(games_df):
    games_df = games_df.copy()
    games_df = games_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    last_game_date = {}
    home_rest_days = []
    away_rest_days = []

    for _, row in games_df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]
        game_date = row["GAME_DATE"]

        if home_team in last_game_date:
            home_days = (game_date - last_game_date[home_team]).days
        else:
            home_days = 7

        if away_team in last_game_date:
            away_days = (game_date - last_game_date[away_team]).days
        else:
            away_days = 7

        home_rest_days.append(home_days)
        away_rest_days.append(away_days)

        last_game_date[home_team] = game_date
        last_game_date[away_team] = game_date

    games_df["home_rest_days"] = home_rest_days
    games_df["away_rest_days"] = away_rest_days
    games_df["rest_diff"] = games_df["home_rest_days"] - games_df["away_rest_days"]

    return games_df


def add_scoring_features(games_df):
    games_df = games_df.copy()
    games_df = games_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    team_points_scored = defaultdict(lambda: deque(maxlen=10))
    team_points_allowed = defaultdict(lambda: deque(maxlen=10))

    home_last10_pts_scored = []
    away_last10_pts_scored = []
    home_last10_pts_allowed = []
    away_last10_pts_allowed = []

    for _, row in games_df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        home_scored_hist = list(team_points_scored[home_team])
        away_scored_hist = list(team_points_scored[away_team])
        home_allowed_hist = list(team_points_allowed[home_team])
        away_allowed_hist = list(team_points_allowed[away_team])

        home_scored_avg = sum(home_scored_hist) / len(home_scored_hist) if home_scored_hist else 100.0
        away_scored_avg = sum(away_scored_hist) / len(away_scored_hist) if away_scored_hist else 100.0
        home_allowed_avg = sum(home_allowed_hist) / len(home_allowed_hist) if home_allowed_hist else 100.0
        away_allowed_avg = sum(away_allowed_hist) / len(away_allowed_hist) if away_allowed_hist else 100.0

        home_last10_pts_scored.append(home_scored_avg)
        away_last10_pts_scored.append(away_scored_avg)
        home_last10_pts_allowed.append(home_allowed_avg)
        away_last10_pts_allowed.append(away_allowed_avg)

        home_pts = row["HOME_PTS"]
        away_pts = row["AWAY_PTS"]

        team_points_scored[home_team].append(home_pts)
        team_points_allowed[home_team].append(away_pts)

        team_points_scored[away_team].append(away_pts)
        team_points_allowed[away_team].append(home_pts)

    games_df["home_last10_pts_scored"] = home_last10_pts_scored
    games_df["away_last10_pts_scored"] = away_last10_pts_scored
    games_df["home_last10_pts_allowed"] = home_last10_pts_allowed
    games_df["away_last10_pts_allowed"] = away_last10_pts_allowed

    return games_df


def add_matchup_features(games_df):
    games_df = games_df.copy()

    games_df["home_off_vs_away_def"] = (
        games_df["home_last10_pts_scored"] - games_df["away_last10_pts_allowed"]
    )
    games_df["away_off_vs_home_def"] = (
        games_df["away_last10_pts_scored"] - games_df["home_last10_pts_allowed"]
    )
    games_df["off_def_matchup_diff"] = (
        games_df["home_off_vs_away_def"] - games_df["away_off_vs_home_def"]
    )

    return games_df


def add_efg_features(games_df):
    games_df = games_df.copy()

    required_box_columns = [
        "HOME_FGM", "HOME_FGA", "HOME_FG3M",
        "AWAY_FGM", "AWAY_FGA", "AWAY_FG3M"
    ]

    if not all(col in games_df.columns for col in required_box_columns):
        print("Skipping eFG features because required box score columns were not found.")
        return games_df

    team_efg_history = defaultdict(lambda: deque(maxlen=10))

    home_last10_efg = []
    away_last10_efg = []

    for _, row in games_df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        home_hist = list(team_efg_history[home_team])
        away_hist = list(team_efg_history[away_team])

        home_efg_avg = sum(home_hist) / len(home_hist) if home_hist else 0.5
        away_efg_avg = sum(away_hist) / len(away_hist) if away_hist else 0.5

        home_last10_efg.append(home_efg_avg)
        away_last10_efg.append(away_efg_avg)

        if row["HOME_FGA"] and row["HOME_FGA"] > 0:
            current_home_efg = (row["HOME_FGM"] + 0.5 * row["HOME_FG3M"]) / row["HOME_FGA"]
        else:
            current_home_efg = 0.5

        if row["AWAY_FGA"] and row["AWAY_FGA"] > 0:
            current_away_efg = (row["AWAY_FGM"] + 0.5 * row["AWAY_FG3M"]) / row["AWAY_FGA"]
        else:
            current_away_efg = 0.5

        team_efg_history[home_team].append(current_home_efg)
        team_efg_history[away_team].append(current_away_efg)

    games_df["home_last10_efg"] = home_last10_efg
    games_df["away_last10_efg"] = away_last10_efg
    games_df["efg_diff"] = games_df["home_last10_efg"] - games_df["away_last10_efg"]

    return games_df

def add_net_rating_and_turnover_features(games_df):
    games_df = games_df.copy()
    games_df = games_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    required_cols = [
        "HOME_PTS", "AWAY_PTS",
        "HOME_FGA", "AWAY_FGA",
        "HOME_FTA", "AWAY_FTA",
        "HOME_OREB", "AWAY_OREB",
        "HOME_TOV", "AWAY_TOV"
    ]

    missing = [col for col in required_cols if col not in games_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for net rating / turnover rate: {missing}")

    team_net_rating_history = defaultdict(lambda: deque(maxlen=10))
    team_turnover_rate_history = defaultdict(lambda: deque(maxlen=10))

    home_net_ratings = []
    away_net_ratings = []
    net_rating_diffs = []

    home_turnover_rates = []
    away_turnover_rates = []
    turnover_rate_diffs = []

    for _, row in games_df.iterrows():
        home_team = row["HOME_TEAM_NAME"]
        away_team = row["AWAY_TEAM_NAME"]

        home_net_hist = list(team_net_rating_history[home_team])
        away_net_hist = list(team_net_rating_history[away_team])

        home_tov_hist = list(team_turnover_rate_history[home_team])
        away_tov_hist = list(team_turnover_rate_history[away_team])

        if home_net_hist:
            home_net_rating = sum(home_net_hist) / len(home_net_hist)
        else:
            home_net_rating = 0.0

        if away_net_hist:
            away_net_rating = sum(away_net_hist) / len(away_net_hist)
        else:
            away_net_rating = 0.0

        if home_tov_hist:
            home_turnover_rate = sum(home_tov_hist) / len(home_tov_hist)
        else:
            home_turnover_rate = 13.0

        if away_tov_hist:
            away_turnover_rate = sum(away_tov_hist) / len(away_tov_hist)
        else:
            away_turnover_rate = 13.0

        home_net_ratings.append(home_net_rating)
        away_net_ratings.append(away_net_rating)
        net_rating_diffs.append(home_net_rating - away_net_rating)

        home_turnover_rates.append(home_turnover_rate)
        away_turnover_rates.append(away_turnover_rate)
        turnover_rate_diffs.append(home_turnover_rate - away_turnover_rate)

        home_possessions = (
            row["HOME_FGA"] - row["HOME_OREB"] +
            row["HOME_TOV"] + 0.44 * row["HOME_FTA"]
        )
        away_possessions = (
            row["AWAY_FGA"] - row["AWAY_OREB"] +
            row["AWAY_TOV"] + 0.44 * row["AWAY_FTA"]
        )

        if home_possessions <= 0:
            home_possessions = 1.0
        if away_possessions <= 0:
            away_possessions = 1.0

        current_home_net_rating = (
            100 * row["HOME_PTS"] / home_possessions -
            100 * row["AWAY_PTS"] / away_possessions
        )

        current_away_net_rating = (
            100 * row["AWAY_PTS"] / away_possessions -
            100 * row["HOME_PTS"] / home_possessions
        )

        current_home_turnover_rate = 100 * row["HOME_TOV"] / home_possessions
        current_away_turnover_rate = 100 * row["AWAY_TOV"] / away_possessions

        team_net_rating_history[home_team].append(current_home_net_rating)
        team_net_rating_history[away_team].append(current_away_net_rating)

        team_turnover_rate_history[home_team].append(current_home_turnover_rate)
        team_turnover_rate_history[away_team].append(current_away_turnover_rate)

    games_df["home_net_rating"] = home_net_ratings
    games_df["away_net_rating"] = away_net_ratings
    games_df["net_rating_diff"] = net_rating_diffs

    games_df["home_turnover_rate"] = home_turnover_rates
    games_df["away_turnover_rate"] = away_turnover_rates
    games_df["turnover_rate_diff"] = turnover_rate_diffs

    return games_df

def validate_games_dataset(games_df):
    if games_df.empty:
        raise ValueError("Processed dataset is empty.")

    if games_df["GAME_ID"].duplicated().any():
        duplicate_ids = games_df.loc[games_df["GAME_ID"].duplicated(), "GAME_ID"].tolist()
        raise ValueError(f"Duplicate GAME_ID values found: {duplicate_ids[:10]}")

    required_columns = [
        "home_net_rating",
        "away_net_rating",
        "net_rating_diff",
        "home_turnover_rate",
        "away_turnover_rate",
        "turnover_rate_diff",
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
        "home_last10_home_win_pct",
        "away_last10_away_win_pct",
        "home_away_form_diff",
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_last10_pts_scored",
        "away_last10_pts_scored",
        "home_last10_pts_allowed",
        "away_last10_pts_allowed",
        "home_off_vs_away_def",
        "away_off_vs_home_def",
        "off_def_matchup_diff"
    ]

    missing = [col for col in required_columns if col not in games_df.columns]
    if missing:
        raise ValueError(f"Missing required processed columns: {missing}")

    if games_df[required_columns].isnull().any().any():
        null_counts = games_df[required_columns].isnull().sum()
        raise ValueError(f"Null values found in required processed columns:\n{null_counts}")


def main():
    ensure_directories()

    print("Loading raw data...")
    raw_df = load_raw_data()

    print("Building one-row-per-game dataset...")
    games_df = build_game_level_dataset(raw_df)

    print("Adding last-10 features...")
    games_df = add_last10_features(games_df)

    print("Adding rest features...")
    games_df = add_rest_features(games_df)

    print("Adding scoring features...")
    games_df = add_scoring_features(games_df)

    print("Adding matchup features...")
    games_df = add_matchup_features(games_df)

    print("Adding eFG features if available...")
    games_df = add_efg_features(games_df)

    print("Adding net rating and turnover rate features...")
    games_df = add_net_rating_and_turnover_features(games_df)

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

    sample_cols = [
        "GAME_DATE",
        "HOME_TEAM_NAME",
        "AWAY_TEAM_NAME",
        "HOME_TEAM_WINS",
        "home_last10_win_pct",
        "away_last10_win_pct",
        "home_last10_home_win_pct",
        "away_last10_away_win_pct",
        "home_away_form_diff",
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "home_off_vs_away_def",
        "away_off_vs_home_def",
        "off_def_matchup_diff"
    ]
    if "efg_diff" in games_df.columns:
        sample_cols.extend(["home_last10_efg", "away_last10_efg", "efg_diff"])

    print("\nSample feature rows:")
    print(games_df[sample_cols].head(10))


if __name__ == "__main__":
    main()