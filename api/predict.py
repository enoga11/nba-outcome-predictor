import os
import sys
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    add_efg_features,
    add_last10_features,
    add_matchup_features,
    add_net_rating_and_turnover_features,
    add_rest_features,
    add_scoring_features,
    build_game_level_dataset,
    clean_team_name,
)

try:
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.library.http import NBAStatsHTTP
    # Spoof a browser User-Agent so the NBA API doesn't block server-side requests
    NBAStatsHTTP.HEADERS["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
except ImportError:
    leaguegamefinder = None

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

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_with_features.csv")

# --- Module-level state ---
_base_df = None
_live_df = None
_model = None
_feature_columns = None


def get_default_live_season():
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}-{end_year}"


def prepare_live_team_games(raw_df):
    df = raw_df.copy()
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
    df = df[df["TEAM_NAME"].isin(set(NBA_TEAMS))].copy()
    df = df.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP", "WL", "PTS", "SEASON"])
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_NAME"]).reset_index(drop=True)


def build_live_feature_dataset():
    if leaguegamefinder is None:
        raise RuntimeError("nba_api is not installed.")
    season = get_default_live_season()
    raw_df = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season
    ).get_data_frames()[0]
    if raw_df.empty:
        raise RuntimeError(f"NBA API returned no games for season {season}.")
    if "SEASON" not in raw_df.columns:
        raw_df["SEASON"] = season
    team_games_df = prepare_live_team_games(raw_df)
    games_df = build_game_level_dataset(team_games_df)
    games_df = add_last10_features(games_df)
    games_df = add_rest_features(games_df)
    games_df = add_scoring_features(games_df)
    games_df = add_matchup_features(games_df)
    games_df = add_efg_features(games_df)
    games_df = add_net_rating_and_turnover_features(games_df)
    return games_df


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


def get_prediction_dataset():
    """Lazy live fetch, cached after first success. Falls back to CSV if it fails."""
    global _live_df
    if _live_df is not None:
        return _live_df
    try:
        _live_df = build_live_feature_dataset()
        print("Live NBA data loaded and cached.")
        return _live_df
    except Exception as e:
        print(f"Live data fetch failed, using CSV fallback: {e}")
        return _base_df


# Load CSV and train model at startup (fast — no API call yet)
_base_df = pd.read_csv(DATA_FILE)
_base_df["GAME_DATE"] = pd.to_datetime(_base_df["GAME_DATE"], errors="coerce")
_feature_columns = get_feature_columns(_base_df)
X = _base_df[_feature_columns]
y = _base_df["HOME_TEAM_WINS"]
split = int(len(_base_df) * 0.8)
_model = LogisticRegression(class_weight="balanced", max_iter=2000)
_model.fit(X.iloc[:split], y.iloc[:split])
print("Model trained on CSV data at startup.")


# --- Helper functions ---

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


# --- API endpoint ---

class PredictRequest(BaseModel):
    home_team: str
    away_team: str


@app.post("/api/predict")
def predict(req: PredictRequest):
    home_team = req.home_team
    away_team = req.away_team

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")

    source_df = get_prediction_dataset()

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

        return {
            "winner": home_team if pred == 1 else away_team,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(float(probs[1]), 4),
            "away_win_prob": round(float(probs[0]), 4),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))