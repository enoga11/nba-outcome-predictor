import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
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

DATA_FILE = "../data/processed/games_with_features.csv"

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
    "efg_diff"
]


class NBAPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("NBA Game Predictor")
        self.geometry("650x450")

        if not os.path.exists(DATA_FILE):
            messagebox.showerror(
                "Missing Data",
                f"Could not find {DATA_FILE}.\nRun nba_api_data_collection.py and feature_engineering.py first."
            )
            self.destroy()
            return

        self.df = pd.read_csv(DATA_FILE)
        self.df["GAME_DATE"] = pd.to_datetime(self.df["GAME_DATE"], errors="coerce")
        self.feature_columns = self.get_feature_columns()
        self.model = self.train_model()
        self.create_widgets()

    def get_feature_columns(self):
        feature_columns = BASE_FEATURE_COLUMNS.copy()
        for col in OPTIONAL_FEATURE_COLUMNS:
            if col in self.df.columns:
                feature_columns.append(col)
        return feature_columns

    def train_model(self):
        X = self.df[self.feature_columns]
        y = self.df["HOME_TEAM_WINS"]

        split_index = int(len(self.df) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        model = LogisticRegression(class_weight="balanced", max_iter=2000)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print("Training Accuracy:", accuracy_score(y_train, train_pred))
        print("Baseline Model Accuracy:", accuracy_score(y_test, test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        return model

    def create_widgets(self):
        title_label = tk.Label(
            self,
            text="NBA Home Team Win Predictor",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=20)

        home_frame = tk.Frame(self)
        home_frame.pack(pady=10)

        home_label = tk.Label(home_frame, text="Home Team:", font=("Arial", 12))
        home_label.pack(side="left", padx=10)

        self.home_team_var = tk.StringVar()
        self.home_team_dropdown = ttk.Combobox(
            home_frame,
            textvariable=self.home_team_var,
            values=NBA_TEAMS,
            state="readonly",
            width=25
        )
        self.home_team_dropdown.pack(side="left")

        away_frame = tk.Frame(self)
        away_frame.pack(pady=10)

        away_label = tk.Label(away_frame, text="Away Team:", font=("Arial", 12))
        away_label.pack(side="left", padx=10)

        self.away_team_var = tk.StringVar()
        self.away_team_dropdown = ttk.Combobox(
            away_frame,
            textvariable=self.away_team_var,
            values=NBA_TEAMS,
            state="readonly",
            width=25
        )
        self.away_team_dropdown.pack(side="left")

        predict_button = tk.Button(
            self,
            text="Predict Winner",
            font=("Arial", 12),
            command=self.predict_game
        )
        predict_button.pack(pady=20)

        self.result_label = tk.Label(
            self,
            text="Select two teams to predict the game result.",
            font=("Arial", 12),
            wraplength=550,
            justify="center"
        )
        self.result_label.pack(pady=20)

    def get_latest_home_row(self, team_name):
        rows = self.df[self.df["HOME_TEAM_NAME"] == team_name].sort_values("GAME_DATE")
        if rows.empty:
            return None
        return rows.iloc[-1]

    def get_latest_away_row(self, team_name):
        rows = self.df[self.df["AWAY_TEAM_NAME"] == team_name].sort_values("GAME_DATE")
        if rows.empty:
            return None
        return rows.iloc[-1]

    def build_prediction_input(self, home_team, away_team):
        latest_home_row = self.get_latest_home_row(home_team)
        latest_away_row = self.get_latest_away_row(away_team)

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

        if all(col in self.df.columns for col in OPTIONAL_FEATURE_COLUMNS):
            home_last10_efg = latest_home_row["home_last10_efg"]
            away_last10_efg = latest_away_row["away_last10_efg"]

            feature_dict["home_last10_efg"] = home_last10_efg
            feature_dict["away_last10_efg"] = away_last10_efg
            feature_dict["efg_diff"] = home_last10_efg - away_last10_efg

        return pd.DataFrame([feature_dict])[self.feature_columns]

    def predict_game(self):
        home_team = self.home_team_var.get()
        away_team = self.away_team_var.get()

        if home_team == "" or away_team == "":
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        if home_team == away_team:
            messagebox.showerror("Input Error", "Home team and away team cannot be the same.")
            return

        try:
            input_df = self.build_prediction_input(home_team, away_team)
        except ValueError as e:
            messagebox.showerror("Data Error", str(e))
            return

        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]

        away_prob = probabilities[0]
        home_prob = probabilities[1]

        if prediction == 1:
            winner_text = f"Predicted Winner: {home_team}"
        else:
            winner_text = f"Predicted Winner: {away_team}"

        result_text = (
            f"{winner_text}\n\n"
            f"{home_team} win probability: {home_prob:.2%}\n"
            f"{away_team} win probability: {away_prob:.2%}"
        )

        self.result_label.config(text=result_text)


def main():
    app = NBAPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()