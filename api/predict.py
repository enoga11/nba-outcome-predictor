# api/predict.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import sys

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    add_efg_features, add_last10_features, add_matchup_features,
    add_net_rating_and_turnover_features, add_rest_features,
    add_scoring_features, build_game_level_dataset, clean_team_name,
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- same constants as your home.py ---
BASE_FEATURE_COLUMNS = [
    "home_last10_win_pct", "away_last10_win_pct", "last10_win_pct_diff",
    "home_last10_home_win_pct", "away_last10_away_win_pct", "home_away_form_diff",
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_off_vs_away_def", "away_off_vs_home_def",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "home_turnover_rate", "away_turnover_rate", "turnover_rate_diff",
]
MATCHUP_FEATURE_ALIASES = ["offensive_defensive_matchup_diff", "off_def_matchup_diff"]
OPTIONAL_FEATURE_COLUMNS = ["home_last10_efg", "away_last10_efg"]
OPTIONAL_EFG_DIFF_ALIASES = ["effective_fg_pct_diff", "efg_diff"]

# Load data and train model once at startup
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_with_features.csv")
df = pd.read_csv(DATA_FILE)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

def get_feature_columns(df):
    cols = BASE_FEATURE_COLUMNS.copy()
    for col in MATCHUP_FEATURE_ALIASES:
        if col in df.columns:
            cols.append(col); break
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in df.columns:
            cols.append(col)
    for col in OPTIONAL_EFG_DIFF_ALIASES:
        if col in df.columns:
            cols.append(col); break
    return cols

feature_columns = get_feature_columns(df)

X = df[feature_columns]
y = df["HOME_TEAM_WINS"]
split = int(len(df) * 0.8)
model = LogisticRegression(class_weight="balanced", max_iter=2000)
model.fit(X.iloc[:split], y.iloc[:split])

# --- same helper methods as your home.py ---
def get_latest_team_row(source_df, team_name):
    home_rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].copy()
    if not home_rows.empty: home_rows["TEAM_ROLE"] = "home"
    away_rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].copy()
    if not away_rows.empty: away_rows["TEAM_ROLE"] = "away"
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    if team_rows.empty: return None
    return team_rows.sort_values(["GAME_DATE", "GAME_ID"]).iloc[-1]

def get_latest_home_row(source_df, team_name):
    rows = source_df[source_df["HOME_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    return None if rows.empty else rows.iloc[-1]

def get_latest_away_row(source_df, team_name):
    rows = source_df[source_df["AWAY_TEAM_NAME"] == team_name].sort_values(["GAME_DATE", "GAME_ID"])
    return None if rows.empty else rows.iloc[-1]

def get_team_snapshot(row, team_name):
    if row is None: raise ValueError(f"No data for {team_name}")
    role = row["TEAM_ROLE"]
    prefix = "home" if role == "home" else "away"
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

@app.post("/api/predict")
def predict(req: PredictRequest):
    home_team, away_team = req.home_team, req.away_team
    if home_team == away_team:
        raise HTTPException(400, "Teams must be different")

    try:
        h = get_team_snapshot(get_latest_team_row(df, home_team), home_team)
        a = get_team_snapshot(get_latest_team_row(df, away_team), away_team)
        hh = get_latest_home_row(df, home_team)
        aa = get_latest_away_row(df, away_team)

        feat = {
            "home_last10_win_pct": h["last10_win_pct"],
            "away_last10_win_pct": a["last10_win_pct"],
            "last10_win_pct_diff": h["last10_win_pct"] - a["last10_win_pct"],
            "home_last10_home_win_pct": hh["home_last10_home_win_pct"],
            "away_last10_away_win_pct": aa["away_last10_away_win_pct"],
            "home_away_form_diff": hh["home_last10_home_win_pct"] - aa["away_last10_away_win_pct"],
            "home_rest_days": h["rest_days"], "away_rest_days": a["rest_days"],
            "rest_diff": h["rest_days"] - a["rest_days"],
            "home_off_vs_away_def": h["last10_pts_scored"] - a["last10_pts_allowed"],
            "away_off_vs_home_def": a["last10_pts_scored"] - h["last10_pts_allowed"],
            "offensive_defensive_matchup_diff": (h["last10_pts_scored"] - a["last10_pts_allowed"]) - (a["last10_pts_scored"] - h["last10_pts_allowed"]),
            "off_def_matchup_diff": (h["last10_pts_scored"] - a["last10_pts_allowed"]) - (a["last10_pts_scored"] - h["last10_pts_allowed"]),
            "home_net_rating": h["net_rating"], "away_net_rating": a["net_rating"],
            "net_rating_diff": h["net_rating"] - a["net_rating"],
            "home_turnover_rate": h["turnover_rate"], "away_turnover_rate": a["turnover_rate"],
            "turnover_rate_diff": h["turnover_rate"] - a["turnover_rate"],
        }
        if "last10_efg" in h and "last10_efg" in a:
            feat.update({
                "home_last10_efg": h["last10_efg"], "away_last10_efg": a["last10_efg"],
                "effective_fg_pct_diff": h["last10_efg"] - a["last10_efg"],
                "efg_diff": h["last10_efg"] - a["last10_efg"],
            })

        input_df = pd.DataFrame([feat])[feature_columns]
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]

        return {
            "winner": home_team if pred == 1 else away_team,
            "home_team": home_team, "away_team": away_team,
            "home_win_prob": round(float(probs[1]), 4),
            "away_win_prob": round(float(probs[0]), 4),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))