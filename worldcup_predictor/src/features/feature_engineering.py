# Placeholder for feature engineering functions 
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

def compute_team_features(matches_df):
    print("Generating team-level features...")

    # Goals for and against
    goals_for = matches_df.groupby('team')['goals'].mean().rename('avg_goals_for')
    goals_against = matches_df.groupby('opponent')['goals'].mean().rename('avg_goals_against')

    # Win rate (1 = win, 0 = loss or draw)
    matches_df['win'] = matches_df['result'].apply(lambda r: 1 if r == 'Win' else 0)
    win_rate = matches_df.groupby('team')['win'].mean().rename('win_rate')

    # Recent form - average goals in last 5 matches
    matches_df = matches_df.sort_values(['team', 'date'])
    matches_df['recent_goals'] = matches_df.groupby('team')['goals'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    recent_form = matches_df.groupby('team')['recent_goals'].last().rename('recent_form')

    # Combine
    team_features = pd.concat([goals_for, goals_against, win_rate, recent_form], axis=1).reset_index()
    return team_features
    

def compute_player_features(players_df):
    print("Generating player-level features...")

    # Minutes to normalize
    players_df['goals_per_90'] = players_df['goals'] / players_df['minutes_played'] * 90
    players_df['assists_per_90'] = players_df['assists'] / players_df['minutes_played'] * 90
    players_df['cards_per_90'] = (players_df['yellow_cards'] + players_df['red_cards']) / players_df['minutes_played'] * 90

    # Save % only for goalkeepers
    players_df['save_percentage'] = np.where(
        players_df['position'] == 'Goalkeeper',
        players_df['saves'] / (players_df['saves'] + players_df['goals_conceded']),
        np.nan
    )

    # Group by player and average over matches
    feature_cols = ['goals_per_90', 'assists_per_90', 'cards_per_90', 'save_percentage']
    player_features = players_df.groupby(['player', 'team'])[feature_cols].mean().reset_index()
    return player_features


def main():
    # Load data
    print("Loading cleaned data...")
    matches_path = os.path.join(PROCESSED_DIR, 'matches_cleaned.csv')
    players_path = os.path.join(PROCESSED_DIR, 'WorldCupPlayers_cleaned.csv')

    matches_df = pd.read_csv(matches_path)
    players_df = pd.read_csv(players_path)

    # Ensure proper types
    matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
    players_df['minutes_played'] = players_df['minutes_played'].replace(0, np.nan)

    # Compute features
    team_df = compute_team_features(matches_df)
    player_df = compute_player_features(players_df)

    # Save results
    team_df.to_csv(os.path.join(PROCESSED_DIR, 'team_features.csv'), index=False)
    player_df.to_csv(os.path.join(PROCESSED_DIR, 'player_features.csv'), index=False)

    print("Feature engineering complete. Outputs saved to data/processed/.")


if __name__ == "__main__":
    main()