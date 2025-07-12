#!/usr/bin/env python3
"""
Simple script to run the FIFA World Cup Predictor Dashboard
"""
import subprocess
import sys
import os

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the working directory to the project root
    os.chdir(current_dir)
    
    # Run the Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    print("Starting FIFA World Cup Predictor Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("\nTo use the dashboard:")
    print("1. Upload the match model: models/match_model.pkl")
    print("2. Upload the team features: data/processed/team_features.csv")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 