import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '../../data/processed')
MODELS_DIR = os.path.join(BASE_DIR, '../../models')

# Load team features and trained model
team_features_path = os.path.join(PROCESSED_DIR, 'team_features.csv')
model_path = os.path.join(MODELS_DIR, 'match_model.pkl')
team_features_df = pd.read_csv(team_features_path)
match_model = joblib.load(model_path)

# Required features for prediction 
FEATURE_COLUMNS = ['avg_goals_for', 'avg_goals_against', 'win_rate', 'recent_form']


# Extracts a team's features from the DataFrame
def get_team_row(team_name):

    row = team_features_df[team_features_df['team'].str.lower() == team_name.lower()]
    if row.empty:
        raise ValueError(f"Team '{team_name}' not found in team_features.csv.")
    return row[FEATURE_COLUMNS].values[0]


# Predicts the outcome of a match and returns win probabilities based on stats
def predict_match(team_a, team_b):
    
    # Get input vectors
    features_a = get_team_row(team_a)
    features_b = get_team_row(team_b)

    # Build "matchup" vector as: team_a - team_b (relative strength)
    input_vector = features_a - features_b
    input_vector = input_vector.reshape(1, -1)

    # Predict win probability for team A
    prob = match_model.predict_proba(input_vector)[0]
    # Assumes class 1 = "team A wins"
    prob_team_a_win = prob[1]
    prob_team_b_win = 1 - prob_team_a_win

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_prob": round(prob_team_a_win, 3),
        "team_b_win_prob": round(prob_team_b_win, 3)
    }


if __name__ == "__main__":
    team_a = input("Enter Team A: ")
    team_b = input("Enter Team B: ")
    result = predict_match(team_a, team_b)
    print(result)