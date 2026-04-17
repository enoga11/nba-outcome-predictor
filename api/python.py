import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

DATA_FILE = os.path.join(os.getcwd(), "data", "processed", "games_with_features.csv")

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
    "off_def_matchup_diff",
    "home_net_rating",
    "away_net_rating",
    "net_rating_diff",
    "home_turnover_rate",
    "away_turnover_rate",
    "turnover_rate_diff",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_last10_efg",
    "away_last10_efg",
    "efg_diff",
]


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
feature_columns = None
model = None


def get_feature_columns(local_df):
    cols = BASE_FEATURE_COLUMNS.copy()
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in local_df.columns:
            cols.append(col)
    return cols


def train_model(local_df, cols):
    X = local_df[cols]
    y = local_df["HOME_TEAM_WINS"]

    split_index = int(len(local_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    local_model = LogisticRegression(class_weight="balanced", max_iter=2000)
    local_model.fit(X_train, y_train)

    train_pred = local_model.predict(X_train)
    test_pred = local_model.predict(X_test)

    print("Training Accuracy:", accuracy_score(y_train, train_pred))
    print("Test Accuracy:", accuracy_score(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred))

    return local_model


def get_latest_home_row(team_name):
    rows = df[df["HOME_TEAM_NAME"] == team_name].sort_values("GAME_DATE")
    if rows.empty:
        return None
    return rows.iloc[-1]


def get_latest_away_row(team_name):
    rows = df[df["AWAY_TEAM_NAME"] == team_name].sort_values("GAME_DATE")
    if rows.empty:
        return None
    return rows.iloc[-1]


def build_prediction_input(home_team, away_team):
    latest_home_row = get_latest_home_row(home_team)
    latest_away_row = get_latest_away_row(away_team)

    if latest_home_row is None:
        raise ValueError(f"No home-team data found for {home_team}.")
    if latest_away_row is None:
        raise ValueError(f"No away-team data found for {away_team}.")

    home_last10_win_pct = latest_home_row["home_last10_win_pct"]
    away_last10_win_pct = latest_away_row["away_last10_win_pct"]

    home_last10_home_win_pct = latest_home_row["home_last10_home_win_pct"]
    away_last10_away_win_pct = latest_away_row["away_last10_away_win_pct"]

    home_rest_days = latest_home_row["home_rest_days"]
    away_rest_days = latest_away_row["away_rest_days"]

    home_off_vs_away_def = (
        latest_home_row["home_last10_pts_scored"] - latest_away_row["away_last10_pts_allowed"]
    )
    away_off_vs_home_def = (
        latest_away_row["away_last10_pts_scored"] - latest_home_row["home_last10_pts_allowed"]
    )

    home_net_rating = latest_home_row["home_net_rating"]
    away_net_rating = latest_away_row["away_net_rating"]

    home_turnover_rate = latest_home_row["home_turnover_rate"]
    away_turnover_rate = latest_away_row["away_turnover_rate"]

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
        "off_def_matchup_diff": home_off_vs_away_def - away_off_vs_home_def,
        "home_net_rating": home_net_rating,
        "away_net_rating": away_net_rating,
        "net_rating_diff": home_net_rating - away_net_rating,
        "home_turnover_rate": home_turnover_rate,
        "away_turnover_rate": away_turnover_rate,
        "turnover_rate_diff": home_turnover_rate - away_turnover_rate,
    }

    if all(col in df.columns for col in OPTIONAL_FEATURE_COLUMNS):
        home_last10_efg = latest_home_row["home_last10_efg"]
        away_last10_efg = latest_away_row["away_last10_efg"]
        feature_dict["home_last10_efg"] = home_last10_efg
        feature_dict["away_last10_efg"] = away_last10_efg
        feature_dict["efg_diff"] = home_last10_efg - away_last10_efg

    return pd.DataFrame([feature_dict])[feature_columns]


@app.on_event("startup")
def startup():
    global df, feature_columns, model

    if not os.path.exists(DATA_FILE):
        raise RuntimeError(f"Missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    feature_columns = get_feature_columns(df)
    model = train_model(df, feature_columns)


@app.get("/api/teams")
def get_teams():
    return {"teams": NBA_TEAMS}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Teams must be different.")
    if req.home_team not in NBA_TEAMS or req.away_team not in NBA_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid team selection.")

    try:
        input_df = build_prediction_input(req.home_team, req.away_team)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    away_prob = float(probabilities[0])
    home_prob = float(probabilities[1])

    winner = req.home_team if prediction == 1 else req.away_team

    return {
        "winner": winner,
        "home_team": req.home_team,
        "away_team": req.away_team,
        "home_probability": home_prob,
        "away_probability": away_prob,
    }