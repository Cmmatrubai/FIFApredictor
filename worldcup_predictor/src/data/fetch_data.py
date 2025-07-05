# Placeholder for data ingestion functions 

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

def fetch_all_csvs():
    for fname in os.listdir(RAW_DIR):
        if fname.lower().endswith('.csv'):
            raw_path = os.path.join(RAW_DIR, fname)
            df = pd.read_csv(raw_path)
            out_path = os.path.join(PROCESSED_DIR, fname)
            df.to_csv(out_path, index=False)
            print(f"Copied {fname} to processed directory.")

if __name__ == "__main__":
    fetch_all_csvs() 