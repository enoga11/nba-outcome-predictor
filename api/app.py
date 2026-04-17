from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "games_with_features.csv"
PUBLIC_DIR = BASE_DIR / "public"

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

BASE_FEATURE_COLUMNS = [
    "home_last10_win_pct",
    "away_last10_win_pct",
    "last10_win_pct_diff",
    "home_last10_home_win_pct",
    "away_last10_away_win_pct",
    "home_away_form_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_off_vs_away_def",
    "away_off_vs_home_def",
    "home_net_rating",
    "away_net_rating",
    "net_rating_diff",
    "home_turnover_rate",
    "away_turnover_rate",
    "turnover_rate_diff",
]

MATCHUP_FEATURE_ALIASES = [
    "offensive_defensive_matchup_diff",
    "off_def_matchup_diff",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_last10_efg",
    "away_last10_efg",
]

OPTIONAL_EFG_DIFF_ALIASES = [
    "effective_fg_pct_diff",
    "efg_diff",
]


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


app = FastAPI(title="NBA Game Predictor")


def get_feature_columns(df: pd.DataFrame):
    feature_columns = BASE_FEATURE_COLUMNS.copy()

    for col in MATCHUP_FEATURE_ALIASES:
        if col in df.columns:
            feature_columns.append(col)
            break

    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in df.columns:
            feature_columns.append(col)

    for col in OPTIONAL_EFG_DIFF_ALIASES:
        if col in df.columns:
            feature_columns.append(col)
            break

    return feature_columns


@lru_cache(maxsize=1)
def load_resources():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    feature_columns = get_feature_columns(df)

    X = df[feature_columns]
    y = df["HOME_TEAM_WINS"]

    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(X, y)

    return df, feature_columns, model


def get_latest_home_row(source_df: pd.DataFrame, team_name: str):
    rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    if rows.empty:
        return None
    return rows.iloc[-1]


def get_latest_away_row(source_df: pd.DataFrame, team_name: str):
    rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    if rows.empty:
        return None
    return rows.iloc[-1]


def get_latest_team_row(source_df: pd.DataFrame, team_name: str):
    home_rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].copy()
    if not home_rows.empty:
        home_rows["TEAM_ROLE"] = "home"

    away_rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].copy()
    if not away_rows.empty:
        away_rows["TEAM_ROLE"] = "away"

    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    if team_rows.empty:
        return None

    team_rows = team_rows.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return team_rows.iloc[-1]


def get_team_snapshot(latest_team_row, team_name: str):
    if latest_team_row is None:
        raise ValueError(f"No team data found for {team_name}.")

    team_role = latest_team_row["TEAM_ROLE"]

    if team_role == "home":
        snapshot = {
            "last10_win_pct": latest_team_row["home_last10_win_pct"],
            "rest_days": latest_team_row["home_rest_days"],
            "last10_pts_scored": latest_team_row["home_last10_pts_scored"],
            "last10_pts_allowed": latest_team_row["home_last10_pts_allowed"],
            "net_rating": latest_team_row["home_net_rating"],
            "turnover_rate": latest_team_row["home_turnover_rate"],
        }
        if "home_last10_efg" in latest_team_row.index:
            snapshot["last10_efg"] = latest_team_row["home_last10_efg"]
    else:
        snapshot = {
            "last10_win_pct": latest_team_row["away_last10_win_pct"],
            "rest_days": latest_team_row["away_rest_days"],
            "last10_pts_scored": latest_team_row["away_last10_pts_scored"],
            "last10_pts_allowed": latest_team_row["away_last10_pts_allowed"],
            "net_rating": latest_team_row["away_net_rating"],
            "turnover_rate": latest_team_row["away_turnover_rate"],
        }
        if "away_last10_efg" in latest_team_row.index:
            snapshot["last10_efg"] = latest_team_row["away_last10_efg"]

    return snapshot


def build_prediction_input(source_df: pd.DataFrame, feature_columns, home_team: str, away_team: str):
    latest_home_team_row = get_latest_team_row(source_df, home_team)
    latest_away_team_row = get_latest_team_row(source_df, away_team)
    latest_home_home_row = get_latest_home_row(source_df, home_team)
    latest_away_away_row = get_latest_away_row(source_df, away_team)

    if latest_home_team_row is None:
        raise ValueError(f"No team data found for {home_team}.")
    if latest_away_team_row is None:
        raise ValueError(f"No team data found for {away_team}.")
    if latest_home_home_row is None:
        raise ValueError(f"No home-game data found for {home_team}.")
    if latest_away_away_row is None:
        raise ValueError(f"No away-game data found for {away_team}.")

    home_snapshot = get_team_snapshot(latest_home_team_row, home_team)
    away_snapshot = get_team_snapshot(latest_away_team_row, away_team)

    home_last10_win_pct = home_snapshot["last10_win_pct"]
    away_last10_win_pct = away_snapshot["last10_win_pct"]

    home_last10_home_win_pct = latest_home_home_row["home_last10_home_win_pct"]
    away_last10_away_win_pct = latest_away_away_row["away_last10_away_win_pct"]

    home_rest_days = home_snapshot["rest_days"]
    away_rest_days = away_snapshot["rest_days"]

    home_off_vs_away_def = home_snapshot["last10_pts_scored"] - away_snapshot["last10_pts_allowed"]
    away_off_vs_home_def = away_snapshot["last10_pts_scored"] - home_snapshot["last10_pts_allowed"]

    home_net_rating = home_snapshot["net_rating"]
    away_net_rating = away_snapshot["net_rating"]

    home_turnover_rate = home_snapshot["turnover_rate"]
    away_turnover_rate = away_snapshot["turnover_rate"]

    feature_dict = {
        "home_last10_win_pct": home_last10_win_pct,
        "away_last10_win_pct": away_last10_win_pct,
        "last10_win_pct_diff": home_last10_win_pct - away_last10_win_pct,
        "home_last10_home_win_pct": home_last10_home_win_pct,
        "away_last10_away_win_pct": away_last10_away_win_pct,
        "home_away_form_diff": home_last10_home_win_pct - away_last10_away_win_pct,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_diff": home_rest_days - away_rest_days,
        "home_off_vs_away_def": home_off_vs_away_def,
        "away_off_vs_home_def": away_off_vs_home_def,
        "offensive_defensive_matchup_diff": home_off_vs_away_def - away_off_vs_home_def,
        "off_def_matchup_diff": home_off_vs_away_def - away_off_vs_home_def,
        "home_net_rating": home_net_rating,
        "away_net_rating": away_net_rating,
        "net_rating_diff": home_net_rating - away_net_rating,
        "home_turnover_rate": home_turnover_rate,
        "away_turnover_rate": away_turnover_rate,
        "turnover_rate_diff": home_turnover_rate - away_turnover_rate,
    }

    if all(col in source_df.columns for col in OPTIONAL_FEATURE_COLUMNS):
        home_last10_efg = home_snapshot["last10_efg"]
        away_last10_efg = away_snapshot["last10_efg"]
        feature_dict["home_last10_efg"] = home_last10_efg
        feature_dict["away_last10_efg"] = away_last10_efg
        feature_dict["effective_fg_pct_diff"] = home_last10_efg - away_last10_efg
        feature_dict["efg_diff"] = home_last10_efg - away_last10_efg

    return pd.DataFrame([feature_dict])[feature_columns]


@app.get("/")
def serve_index():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
def health():
    df, feature_columns, _ = load_resources()
    return {
        "status": "ok",
        "games_loaded": int(len(df)),
        "features_used": feature_columns,
    }


@app.get("/api/teams")
def get_teams():
    return {"teams": NBA_TEAMS}


@app.post("/api/predict")
def predict_game(payload: PredictRequest):
    home_team = payload.home_team
    away_team = payload.away_team

    if not home_team or not away_team:
        raise HTTPException(status_code=400, detail="Please select both teams.")

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home team and away team cannot be the same.")

    if home_team not in NBA_TEAMS or away_team not in NBA_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid team selection.")

    df, feature_columns, model = load_resources()

    try:
        input_df = build_prediction_input(df, feature_columns, home_team, away_team)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    away_prob = float(probabilities[0])
    home_prob = float(probabilities[1])
    predicted_winner = home_team if prediction == 1 else away_team

    return {
        "predicted_winner": predicted_winner,
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "data_source": "Bundled processed dataset",
    }
