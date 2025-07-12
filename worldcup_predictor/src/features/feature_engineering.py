# Placeholder for feature engineering functions 
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

def compute_team_features(matches_df):
    print("Generating team-level features...")

    # Stack home and away stats for per-team aggregation
    home = matches_df[[
        'home_team_name', 'away_team_name', 'home_team_goals', 'away_team_goals', 'datetime'
    ]].copy()
    home.rename(columns={
        'home_team_name': 'team',
        'away_team_name': 'opponent',
        'home_team_goals': 'goals_for',
        'away_team_goals': 'goals_against',
    }, inplace=True)
    home['is_home'] = 1

    away = matches_df[[
        'away_team_name', 'home_team_name', 'away_team_goals', 'home_team_goals', 'datetime'
    ]].copy()
    away.rename(columns={
        'away_team_name': 'team',
        'home_team_name': 'opponent',
        'away_team_goals': 'goals_for',
        'home_team_goals': 'goals_against',
    }, inplace=True)
    away['is_home'] = 0

    all_matches = pd.concat([home, away], ignore_index=True)
    all_matches['date'] = pd.to_datetime(all_matches['datetime'], errors='coerce')

    # Win/draw/loss
    all_matches['win'] = (all_matches['goals_for'] > all_matches['goals_against']).astype(int)
    all_matches['draw'] = (all_matches['goals_for'] == all_matches['goals_against']).astype(int)
    all_matches['loss'] = (all_matches['goals_for'] < all_matches['goals_against']).astype(int)

    # Aggregate features
    team_stats = all_matches.groupby('team').agg(
        avg_goals_for = ('goals_for', 'mean'),
        avg_goals_against = ('goals_against', 'mean'),
        win_rate = ('win', 'mean'),
        draw_rate = ('draw', 'mean'),
        loss_rate = ('loss', 'mean'),
        n_matches = ('win', 'count')
    )

    # Recent form: average goals in last 5 matches
    all_matches = all_matches.sort_values(['team', 'date'])
    all_matches['recent_goals'] = all_matches.groupby('team')['goals_for'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    recent_form = all_matches.groupby('team')['recent_goals'].last().rename('recent_form')

    team_features = team_stats.join(recent_form)
    team_features = team_features.reset_index()
    return team_features
    

def compute_player_features(players_df):
    print("Generating player-level features...")

    feature_cols = [
        'goals_scored',
        'assists_provided',
        'dribbles_per_90',
        'interceptions_per_90',
        'tackles_per_90',
        'total_duels_won_per_90',
        'save_percentage',
        'clean_sheets'
    ]
    # Replace 'N.A' and similar with np.nan, and convert to numeric
    for col in feature_cols:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col].replace(['N.A', 'NA', 'n/a', '', None], np.nan), errors='coerce')
    # Group by player and team, average over appearances if needed
    player_features = players_df.groupby(['player_name', 'nationality'])[feature_cols].mean().reset_index()
    return player_features


def main():
    # Load data
    print("Loading cleaned data...")
    matches_path = os.path.join(PROCESSED_DIR, 'WorldCupMatches_cleaned.csv')
    players_path = os.path.join(PROCESSED_DIR, 'FIFA WC 2022 Players Stats_cleaned.csv')

    matches_df = pd.read_csv(matches_path)
    players_df = pd.read_csv(players_path)

    # Ensure proper types
    matches_df['date'] = pd.to_datetime(matches_df['datetime'], errors='coerce')
    # Remove minutes_played reference
    # players_df['minutes_played'] = players_df['minutes_played'].replace(0, np.nan)

    # Compute features
    team_df = compute_team_features(matches_df)
    player_df = compute_player_features(players_df)

    # Save results
    team_df.to_csv(os.path.join(PROCESSED_DIR, 'team_features.csv'), index=False)
    player_df.to_csv(os.path.join(PROCESSED_DIR, 'player_features.csv'), index=False)

    print("Feature engineering complete. Outputs saved to data/processed/.")


if __name__ == "__main__":
    main()