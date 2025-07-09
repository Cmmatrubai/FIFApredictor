# Placeholder for award prediction model

import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, '../../data/processed')
MODELS_DIR = os.path.join(BASE_DIR, '../../models')

player_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'player_features.csv'))

model_goals = joblib.load(os.path.join(MODELS_DIR, 'award_model_goals.pkl'))
model_assists = joblib.load(os.path.join(MODELS_DIR, 'award_model_assists.pkl'))
model_cards = joblib.load(os.path.join(MODELS_DIR, 'award_model_cards.pkl'))
model_saves = joblib.load(os.path.join(MODELS_DIR, 'award_model_saves.pkl'))

feature_cols = ['assists_per_90', 'cards_per_90']
if 'injury_status_encoded' in player_df.columns:
    feature_cols.append('injury_status_encoded')

# Returns all award predictions for a single player
def predict_all_awards(player_name):
   
    row = player_df[player_df['player'].str.lower() == player_name.lower()]
    if row.empty:
        raise ValueError(f"Player '{player_name}' not found in player_features.csv")

    player_info = row.iloc[0]
    features = row[feature_cols]

    result = {
        'player': player_info['player'],
        'team': player_info['team'],
    }

    # General players
    result['predicted_goals_per_90'] = round(model_goals.predict(features)[0], 3)
    result['predicted_assists_per_90'] = round(model_assists.predict(features)[0], 3)
    result['predicted_cards_per_90'] = round(model_cards.predict(features)[0], 3)

    # Goalkeeper-specific
    if str(player_info.get('position', '')).lower() == 'goalkeeper':
        result['predicted_save_percentage'] = round(model_saves.predict(features)[0], 3)
    else:
        result['predicted_save_percentage'] = None

    return result


# Returns top N players based on specific metric
def get_top_players(metric='goals', top_n=10):
    
    if metric == 'goals':
        model = model_goals
        df = player_df.copy()
    elif metric == 'assists':
        model = model_assists
        df = player_df.copy()
    elif metric == 'cards':
        model = model_cards
        df = player_df.copy()
    elif metric == 'saves':
        model = model_saves
        df = player_df[player_df['position'].str.lower() == 'goalkeeper'].copy()
    else:
        raise ValueError("Invalid metric. Choose from: 'goals', 'assists', 'cards', 'saves'")

    # Predict
    df['prediction'] = model.predict(df[feature_cols])
    df = df[['player', 'team', 'position', 'prediction']]
    df = df.sort_values(by='prediction', ascending=(metric == 'cards'))  # lower cards = better
    return df.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    print("FIFA 2026 Award Predictor")

    player_input = input("Enter a player's name for prediction (or press Enter to skip): ").strip()
    if player_input:
        try:
            result = predict_all_awards(player_input)
            print("\nAward Predictions for Player:")
            for k, v in result.items():
                print(f"{k}: {v}")
        except ValueError as e:
            print(f"❌ {e}")

    # === Top N player rankings (interactive) ===
    metric_input = input("\nEnter metric to show top players (goals, assists, saves, cards): ").strip().lower()
    if metric_input in ['goals', 'assists', 'saves', 'cards']:
        try:
            top_n = input("How many top players would you like to see?: ").strip()
            top_n = int(top_n) if top_n else 10
            print(f"\nTop {top_n} Predicted Players for {metric_input.capitalize()}:")
            print(get_top_players(metric=metric_input, top_n=top_n))
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("No valid metric entered. Skipping top player rankings.")
