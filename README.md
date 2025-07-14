# FIFA World Cup Predictor Dashboard

This dashboard allows you to predict match outcomes between teams using the trained machine learning model.

## How to Run

### Option 1: Using the run script
```bash
cd worldcup_predictor
python run_dashboard.py
```

### Option 2: Direct Streamlit command
```bash
cd worldcup_predictor
streamlit run src/dashboard/app.py
```

The dashboard will be available at: http://localhost:8501

## How to Use

1. **Upload the Model**: 
   - Click "Browse files" in the "Upload trained match model" section
   - Select `models/match_model.pkl`

2. **Upload Team Features**:
   - Click "Browse files" in the "Upload team features" section  
   - Select `data/processed/team_features.csv`

3. **Make Predictions**:
   - Select two teams from the dropdown menus
   - Click "Predict Match Outcome"
   - View the win probabilities for each team

## Files Required

- **Match Model**: `models/match_model.pkl` - The trained classification model for match predictions
- **Team Features**: `data/processed/team_features.csv` - Team statistics and features

**Important**: Make sure to upload the `match_model.pkl` file, not the award models (`award_model_*.pkl`).

## Troubleshooting

If you encounter errors:

1. **KeyError: 'team'**: Make sure you've uploaded the correct team features CSV file
2. **Missing columns**: The dashboard expects specific columns in the team features file
3. **Model loading errors**: Ensure the model file is not corrupted
4. **AttributeError: 'GradientBoostingRegressor' object has no attribute 'predict_proba'**: You uploaded an award model instead of the match model. Use `models/match_model.pkl` for match predictions.

## Features

- **Match Prediction**: Predict win probabilities between any two teams
- **Team Selection**: Dropdown menus with all available teams
- **Real-time Results**: Instant prediction results with percentage probabilities

## Technical Details

The dashboard uses:
- **Streamlit** for the web interface
- **Pandas** for data handling
- **Joblib** for model loading
- **Scikit-learn** for model predictions

The prediction model uses team features including:
- Average goals scored
- Average goals conceded  
- Win rate
- Recent form 
