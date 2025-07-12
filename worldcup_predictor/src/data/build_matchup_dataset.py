import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

# Load cleaned match data and team features
df_matches = pd.read_csv(os.path.join(PROCESSED_DIR, 'WorldCupMatches_cleaned.csv'))
df_teams = pd.read_csv(os.path.join(PROCESSED_DIR, 'team_features.csv'))

# Set up for fast lookup
team_stats = df_teams.set_index('team')
feature_cols = ['avg_goals_for', 'avg_goals_against', 'win_rate', 'recent_form']

rows = []
for _, row in df_matches.iterrows():
    home = row['home_team_name']
    away = row['away_team_name']
    if home not in team_stats.index or away not in team_stats.index:
        continue
    home_stats = team_stats.loc[home, feature_cols].values
    away_stats = team_stats.loc[away, feature_cols].values
    diff_home_away = home_stats - away_stats
    diff_away_home = away_stats - home_stats
    # Label: 1 if home win, 0 otherwise
    label_home = 1 if row['home_team_goals'] > row['away_team_goals'] else 0
    label_away = 1 if row['away_team_goals'] > row['home_team_goals'] else 0
    # Add both orders
    rows.append({
        'team_a': home,
        'team_b': away,
        **{f'diff_{col}': d for col, d in zip(feature_cols, diff_home_away)},
        'label': label_home
    })
    rows.append({
        'team_a': away,
        'team_b': home,
        **{f'diff_{col}': d for col, d in zip(feature_cols, diff_away_home)},
        'label': label_away
    })

matchup_df = pd.DataFrame(rows)
matchup_df.to_csv(os.path.join(PROCESSED_DIR, 'matchup_dataset.csv'), index=False)
print(f"Saved matchup dataset with {len(matchup_df)} rows.") 