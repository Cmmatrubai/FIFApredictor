import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/match_model.pkl')
TEAM_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../data/processed/team_features.csv')

model = joblib.load(MODEL_PATH)
team_features = pd.read_csv(TEAM_FEATURES_PATH)

feature_cols = ['avg_goals_for', 'avg_goals_against', 'win_rate', 'recent_form']

def get_team_features(team_name):
    row = team_features[team_features['team'].str.lower() == team_name.lower()]
    if row.empty:
        raise ValueError(f"Team '{team_name}' not found in team features.")
    return row[feature_cols].values[0]

def symmetric_win_probability(team_a, team_b):
    features_a = get_team_features(team_a)
    features_b = get_team_features(team_b)
    input_ab = features_a - features_b
    input_ba = features_b - features_a
    prob_ab = model.predict_proba(input_ab.reshape(1, -1))[0][1]
    prob_ba = model.predict_proba(input_ba.reshape(1, -1))[0][1]
    prob_team_a_win = (prob_ab + (1 - prob_ba)) / 2
    prob_team_b_win = 1 - prob_team_a_win
    return prob_team_a_win, prob_team_b_win

if __name__ == "__main__":
    pairs = [
        ("Argentina", "France"),
        ("Brazil", "Germany"),
        ("England", "Spain"),
    ]
    for team_a, team_b in pairs:
        try:
            prob_a, prob_b = symmetric_win_probability(team_a, team_b)
            print(f"{team_a} vs {team_b}: {team_a} win probability: {prob_a:.2%}, {team_b} win probability: {prob_b:.2%}")
            # Test order invariance
            prob_b_rev, prob_a_rev = symmetric_win_probability(team_b, team_a)
            print(f"{team_b} vs {team_a}: {team_b} win probability: {prob_b_rev:.2%}, {team_a} win probability: {prob_a_rev:.2%}")
            assert np.isclose(prob_a, prob_a_rev, atol=1e-2), "Order invariance failed!"
        except Exception as e:
            print(f"Error for {team_a} vs {team_b}: {e}") 