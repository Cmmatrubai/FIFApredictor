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

    # Load matchup dataset
    matchup_path = os.path.join(PROCESSED_DIR, 'matchup_dataset.csv')
    df = pd.read_csv(matchup_path)

    feature_cols = [
        'diff_avg_goals_for',
        'diff_avg_goals_against',
        'diff_win_rate',
        'diff_recent_form'
    ]
    X = df[feature_cols]
    y = df['label']

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
        df = df[df['position'].str.lower() == 'goalkeeper'] if 'position' in df.columns else df

    # Drop rows missing target
    df = df.dropna(subset=[target_column])

    # Features
    if target_column == 'goals_scored':
        feature_cols = ['assists_provided', 'dribbles_per_90', 'interceptions_per_90', 'tackles_per_90', 'total_duels_won_per_90']
    elif target_column == 'assists_provided':
        feature_cols = ['goals_scored', 'dribbles_per_90', 'interceptions_per_90', 'tackles_per_90', 'total_duels_won_per_90']
    elif target_column == 'save_percentage':
        feature_cols = ['clean_sheets', 'interceptions_per_90', 'tackles_per_90', 'total_duels_won_per_90']
    else:
        feature_cols = [col for col in df.columns if col not in ['player_name', 'nationality', target_column]]

    # Drop rows with NaN in any feature column
    df = df.dropna(subset=feature_cols)

    # Check if we have enough samples after filtering
    if len(df) < 5:  # Need at least 5 samples for meaningful training
        print(f"⚠️  Insufficient data for {target_column} model. Only {len(df)} samples available after filtering. Skipping this model.")
        return

    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{target_column} RMSE: {rmse:.3f}")

    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    print(f"Saved {target_column} model to {model_path}")

def main():
    train_match_model()

    # Train available award models based on columns in player_features.csv
    train_award_model('goals_scored', 'award_model_goals.pkl')
    train_award_model('assists_provided', 'award_model_assists.pkl')
    # Skipping cards_per_90 as it does not exist in the CSV
    if 'save_percentage' in pd.read_csv(os.path.join(PROCESSED_DIR, 'player_features.csv')).columns:
        train_award_model('save_percentage', 'award_model_saves.pkl', filter_goalkeepers=True)


if __name__ == "__main__":
    main()

