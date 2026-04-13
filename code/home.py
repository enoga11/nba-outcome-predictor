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
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]


DATA_FILE = "../data/processed/games_with_features.csv"


class NBAPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("NBA Game Predictor")
        self.geometry("600x400")

        if not os.path.exists(DATA_FILE):
            messagebox.showerror(
                "Missing Data",
                f"Could not find {DATA_FILE}.\nRun nba_api_data_collection.py first."
            )
            self.destroy()
            return

        self.df = pd.read_csv(DATA_FILE)
        self.model = self.train_model()
        self.create_widgets()

    def train_model(self):
        X = self.df[
            [
                "home_last10_win_pct",
                "away_last10_win_pct",
                "last10_win_pct_diff",
                "home_last10_home_win_pct",
                "away_last10_away_win_pct",
                "home_away_form_diff"
            ]
        ]
        y = self.df["HOME_TEAM_WINS"]

        split_index = int(len(self.df) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Baseline Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

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
            wraplength=500,
            justify="center"
        )
        self.result_label.pack(pady=20)

    def predict_game(self):
        home_team = self.home_team_var.get()
        away_team = self.away_team_var.get()

        if home_team == "" or away_team == "":
            messagebox.showerror("Input Error", "Please select both teams.")
            return

        if home_team == away_team:
            messagebox.showerror("Input Error", "Home team and away team cannot be the same.")
            return

        home_rows = self.df[self.df["HOME_TEAM_NAME"] == home_team]
        away_rows = self.df[self.df["AWAY_TEAM_NAME"] == away_team]

        if home_rows.empty:
            messagebox.showerror("Data Error", f"No home-team data found for {home_team}.")
            return

        if away_rows.empty:
            messagebox.showerror("Data Error", f"No away-team data found for {away_team}.")
            return

        home_last10_win_pct = home_rows["home_last10_win_pct"].mean()
        away_last10_win_pct = away_rows["away_last10_win_pct"].mean()
        last10_win_pct_diff = home_last10_win_pct - away_last10_win_pct

        home_last10_home_win_pct = home_rows["home_last10_home_win_pct"].mean()
        away_last10_away_win_pct = away_rows["away_last10_away_win_pct"].mean()
        home_away_form_diff = home_last10_home_win_pct - away_last10_away_win_pct

        input_df = pd.DataFrame({
            "home_last10_win_pct": [home_last10_win_pct],
            "away_last10_win_pct": [away_last10_win_pct],
            "last10_win_pct_diff": [last10_win_pct_diff],
            "home_last10_home_win_pct": [home_last10_home_win_pct],
            "away_last10_away_win_pct": [away_last10_away_win_pct],
            "home_away_form_diff": [home_away_form_diff]
        })

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