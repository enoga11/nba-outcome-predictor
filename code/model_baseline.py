import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load feature-engineered dataset
    df = pd.read_csv("../data/processed/games_with_features.csv")

    # Select features and target
    X = df[["home_win_pct", "away_win_pct", "win_pct_diff"]]
    y = df["HOME_TEAM_WINS"]

    # Time-based split: first 80% train, last 20% test
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Train logistic regression baseline
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print("Baseline Model Accuracy:", accuracy)
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()