import json
import os
import sys
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import clean_team_name

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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
MATCHUP_FEATURE_ALIASES = ["offensive_defensive_matchup_diff", "off_def_matchup_diff"]
OPTIONAL_FEATURE_COLUMNS = ["home_last10_efg", "away_last10_efg"]
OPTIONAL_EFG_DIFF_ALIASES = ["effective_fg_pct_diff", "efg_diff"]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT_DIR, "data", "processed", "games_with_features.csv")
LIVE_DATA_FILE = os.path.join(ROOT_DIR, "data", "live", "live_games_with_features.csv")
LIVE_METADATA_FILE = os.path.join(ROOT_DIR, "data", "live", "metadata.json")

_base_df = None
_live_df = None
_model = None
_feature_columns = None


def get_feature_columns(df):
    cols = BASE_FEATURE_COLUMNS.copy()
    for col in MATCHUP_FEATURE_ALIASES:
        if col in df.columns:
            cols.append(col)
            break
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in df.columns:
            cols.append(col)
    for col in OPTIONAL_EFG_DIFF_ALIASES:
        if col in df.columns:
            cols.append(col)
            break
    return cols


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for team_col in ["HOME_TEAM_NAME", "AWAY_TEAM_NAME"]:
        if team_col in df.columns:
            df[team_col] = df[team_col].apply(clean_team_name)
    return df


def get_live_metadata():
    if os.path.exists(LIVE_METADATA_FILE):
        try:
            with open(LIVE_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_prediction_dataset():
    global _live_df
    if _live_df is not None:
        return _live_df, "cached live snapshot"

    if os.path.exists(LIVE_DATA_FILE):
        _live_df = load_dataset(LIVE_DATA_FILE)
        metadata = get_live_metadata() or {}
        source = metadata.get("source", "local live snapshot")
        exported_at = metadata.get("exported_at")
        if exported_at:
            source = f"{source} ({exported_at})"
        return _live_df, source

    return _base_df, "historical CSV fallback"


_base_df = load_dataset(DATA_FILE)
_feature_columns = get_feature_columns(_base_df)
X = _base_df[_feature_columns]
y = _base_df["HOME_TEAM_WINS"]
split = int(len(_base_df) * 0.8)
_model = LogisticRegression(class_weight="balanced", max_iter=2000)
_model.fit(X.iloc[:split], y.iloc[:split])
print("Model trained on CSV data at startup.")


def get_latest_team_row(source_df, team_name):
    home_rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].copy()
    if not home_rows.empty:
        home_rows["TEAM_ROLE"] = "home"
    away_rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].copy()
    if not away_rows.empty:
        away_rows["TEAM_ROLE"] = "away"
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    if team_rows.empty:
        return None
    return team_rows.sort_values(["GAME_DATE", "GAME_ID"]).iloc[-1]


def get_latest_home_row(source_df, team_name):
    rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    return None if rows.empty else rows.iloc[-1]


def get_latest_away_row(source_df, team_name):
    rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    return None if rows.empty else rows.iloc[-1]


def get_team_snapshot(row, team_name):
    if row is None:
        raise ValueError(f"No data found for {team_name}.")
    prefix = "home" if row["TEAM_ROLE"] == "home" else "away"
    snap = {
        "last10_win_pct": row[f"{prefix}_last10_win_pct"],
        "rest_days": row[f"{prefix}_rest_days"],
        "last10_pts_scored": row[f"{prefix}_last10_pts_scored"],
        "last10_pts_allowed": row[f"{prefix}_last10_pts_allowed"],
        "net_rating": row[f"{prefix}_net_rating"],
        "turnover_rate": row[f"{prefix}_turnover_rate"],
    }
    if f"{prefix}_last10_efg" in row.index:
        snap["last10_efg"] = row[f"{prefix}_last10_efg"]
    return snap


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/api/status")
def status():
    source_df, data_source = get_prediction_dataset()
    latest_game_date = None
    if "GAME_DATE" in source_df.columns and not source_df.empty:
        latest_game_date = source_df["GAME_DATE"].max()
        if pd.notna(latest_game_date):
            latest_game_date = latest_game_date.strftime("%Y-%m-%d")
        else:
            latest_game_date = None
    return {
        "data_source": data_source,
        "using_live_snapshot": os.path.exists(LIVE_DATA_FILE),
        "latest_game_date": latest_game_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    home_team = req.home_team
    away_team = req.away_team

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")
    if home_team not in NBA_TEAMS or away_team not in NBA_TEAMS:
        raise HTTPException(status_code=400, detail="One or both team names are invalid.")

    source_df, data_source = get_prediction_dataset()

    try:
        h = get_team_snapshot(get_latest_team_row(source_df, home_team), home_team)
        a = get_team_snapshot(get_latest_team_row(source_df, away_team), away_team)
        hh = get_latest_home_row(source_df, home_team)
        aa = get_latest_away_row(source_df, away_team)

        if hh is None:
            raise ValueError(f"No home-game data found for {home_team}.")
        if aa is None:
            raise ValueError(f"No away-game data found for {away_team}.")

        feat = {
            "home_last10_win_pct": h["last10_win_pct"],
            "away_last10_win_pct": a["last10_win_pct"],
            "last10_win_pct_diff": h["last10_win_pct"] - a["last10_win_pct"],
            "home_last10_home_win_pct": hh["home_last10_home_win_pct"],
            "away_last10_away_win_pct": aa["away_last10_away_win_pct"],
            "home_away_form_diff": hh["home_last10_home_win_pct"] - aa["away_last10_away_win_pct"],
            "home_rest_days": h["rest_days"],
            "away_rest_days": a["rest_days"],
            "rest_diff": h["rest_days"] - a["rest_days"],
            "home_off_vs_away_def": h["last10_pts_scored"] - a["last10_pts_allowed"],
            "away_off_vs_home_def": a["last10_pts_scored"] - h["last10_pts_allowed"],
            "offensive_defensive_matchup_diff": (h["last10_pts_scored"] - a["last10_pts_allowed"]) - (a["last10_pts_scored"] - h["last10_pts_allowed"]),
            "off_def_matchup_diff": (h["last10_pts_scored"] - a["last10_pts_allowed"]) - (a["last10_pts_scored"] - h["last10_pts_allowed"]),
            "home_net_rating": h["net_rating"],
            "away_net_rating": a["net_rating"],
            "net_rating_diff": h["net_rating"] - a["net_rating"],
            "home_turnover_rate": h["turnover_rate"],
            "away_turnover_rate": a["turnover_rate"],
            "turnover_rate_diff": h["turnover_rate"] - a["turnover_rate"],
        }

        if "last10_efg" in h and "last10_efg" in a:
            feat["home_last10_efg"] = h["last10_efg"]
            feat["away_last10_efg"] = a["last10_efg"]
            feat["effective_fg_pct_diff"] = h["last10_efg"] - a["last10_efg"]
            feat["efg_diff"] = h["last10_efg"] - a["last10_efg"]

        input_df = pd.DataFrame([feat])[_feature_columns]

        pred = _model.predict(input_df)[0]
        probs = _model.predict_proba(input_df)[0]

        home_prob = float(probs[1])
        away_prob = float(probs[0])

        home_prob_rounded = round(home_prob, 3)
        away_prob_rounded = round(away_prob, 3)

        winner = home_team if home_prob_rounded >= away_prob_rounded else away_team

        return {
            "winner": winner,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(float(probs[1]), 4),
            "away_win_prob": round(float(probs[0]), 4),
            "data_source": data_source,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))