import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import StringIO

st.set_page_config(page_title="FIFA World Cup Predictor Dashboard", layout="wide")
st.title("🏆 FIFA World Cup Predictor Dashboard")

st.sidebar.header("Model & Data Upload")
model_file = st.sidebar.file_uploader("Upload trained match model (joblib)", type=["pkl", "joblib"])
team_features_file = st.sidebar.file_uploader("Upload team features (CSV)", type=["csv"])

model = None
team_features_df = None
model_type = None

if model_file and team_features_file:
    st.success("Model and team features uploaded!")
    model = joblib.load(model_file)
    
    # Determine model type
    model_type = type(model).__name__
    st.write(f"Model loaded: {model_type}")
    
    # Check if it's a classification model (has predict_proba)
    if hasattr(model, 'predict_proba'):
        st.success("✅ Classification model detected - can predict win probabilities")
    else:
        st.error("❌ Regression model detected - cannot predict win probabilities")
        st.info("Please upload the match classification model (match_model.pkl), not the award regression models")
        model = None
    
    # Read CSV from uploaded file
    team_features_df = pd.read_csv(team_features_file)
    st.write("Team features loaded.")
else:
    st.info("Please upload both a trained model and team features to begin.")

# --- Match Prediction Section ---
st.header("Match Outcome Prediction")
if model is not None and team_features_df is not None and hasattr(model, 'predict_proba'):
    # Check if 'team' column exists
    if 'team' not in team_features_df.columns:
        st.error("Error: 'team' column not found in the uploaded team features file.")
        st.write("Available columns:", list(team_features_df.columns))
    else:
        teams = team_features_df['team'].unique().tolist()
        team_a = st.selectbox("Select Team A", teams, key="team_a")
        team_b = st.selectbox("Select Team B", teams, key="team_b")
        if st.button("Predict Match Outcome"):
            feature_cols = ['avg_goals_for', 'avg_goals_against', 'win_rate', 'recent_form']
            # Check if all required feature columns exist
            missing_cols = [col for col in feature_cols if col not in team_features_df.columns]
            if missing_cols:
                st.error(f"Error: Missing required columns: {missing_cols}")
                st.write("Available columns:", list(team_features_df.columns))
            else:
                def get_team_row(team_name):
                    row = team_features_df[team_features_df['team'].str.lower() == team_name.lower()]
                    if row.empty:
                        st.error(f"Team '{team_name}' not found in team features.")
                        return None
                    return row[feature_cols].values[0]
                features_a = get_team_row(team_a)
                features_b = get_team_row(team_b)
                if features_a is not None and features_b is not None:
                    input_ab = features_a - features_b
                    input_ba = features_b - features_a
                    prob_ab = model.predict_proba(input_ab.reshape(1, -1))[0][1]
                    prob_ba = model.predict_proba(input_ba.reshape(1, -1))[0][1]
                    prob_team_a_win = (prob_ab + (1 - prob_ba)) / 2
                    prob_team_b_win = 1 - prob_team_a_win
                    st.markdown(f"**{team_a} win probability:** {prob_team_a_win:.2%}")
                    st.markdown(f"**{team_b} win probability:** {prob_team_b_win:.2%}")
elif model is not None and not hasattr(model, 'predict_proba'):
    st.error("❌ Wrong model type uploaded!")
    st.info("""
    **Please upload the correct model:**
    - ✅ Use `models/match_model.pkl` (classification model for match predictions)
    - ❌ Don't use `models/award_model_*.pkl` (regression models for player awards)
    """)
else:
    st.info("Upload model and team features to enable match prediction.")

st.markdown("---")
st.caption("Built with Streamlit. Integrate your model and data for full functionality.") 