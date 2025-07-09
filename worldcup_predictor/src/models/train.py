# Placeholder for model training routines 
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(MODELS_DIR, exist_ok=True)


def train_match_model():
    print("Training match outcome prediction model...")

    # Load team features
    team_path = os.path.join(PROCESSED_DIR, 'team_features.csv')
    df = pd.read_csv(team_path)

    # Simulate match outcomes using team features
    X = df[['avg_goals_for', 'avg_goals_against', 'win_rate', 'recent_form']]
    y = (df['win_rate'] > 0.5).astype(int)  # win_rate > 50% → "likely win"

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Match Model Accuracy (CV): {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR, 'match_model.pkl'))
    print("Saved match model to models/match_model.pkl")


def train_award_model(target_column, model_filename, filter_goalkeepers=False):
    print(f"Training player award prediction model for:{target_column}")

    # Load player features
    player_path = os.path.join(PROCESSED_DIR, 'player_features.csv')
    df = pd.read_csv(player_path)

    # filter for goalkeeper-specific models
    if filter_goalkeepers:
        df = df[df['position'].str.lower() == 'goalkeeper']

    # Drop rows missing target
    df = df.dropna(subset=[target_column])

    # Features
    feature_cols = ['assists_per_90', 'cards_per_90']
    if 'injury_status' in df.columns:
        # Encode injury_status (Healthy/Injured/etc.)
        le = LabelEncoder()
        df['injury_status_encoded'] = le.fit_transform(df['injury_status'])
        feature_cols.append('injury_status_encoded')

    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{target_column} RMSE: {rmse:.3f}")

    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"Saved {target_column} model to {model_path}")

def main():
    train_match_model()

    # Train multiple award models
    train_award_model('goals_per_90', 'award_model_goals.pkl')
    train_award_model('assists_per_90', 'award_model_assists.pkl')
    train_award_model('cards_per_90', 'award_model_cards.pkl')
    train_award_model('save_percentage', 'award_model_saves.pkl', filter_goalkeepers=True)


if __name__ == "__main__":
    main()

