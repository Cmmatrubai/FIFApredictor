import os
import pandas as pd
import re
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

# Helper to standardize column names

def clean_column_names(columns):
    return [re.sub(r'\s+', '_', col.strip().lower()) for col in columns]

# Helper to standardize date columns

def try_parse_date(val):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(val), fmt).strftime('%Y-%m-%d')
        except Exception:
            continue
    return val

# Main cleaning function

def clean_csv_file(fname):
    path = os.path.join(PROCESSED_DIR, fname)
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = clean_column_names(df.columns)
    # Try to standardize date columns
    for col in df.columns:
        if 'date' in col:
            df[col] = df[col].apply(try_parse_date)
    # Remove duplicates
    df = df.drop_duplicates()
    # Drop rows missing critical fields (first col, or any with 'team'/'player' in name)
    crit_cols = [df.columns[0]] + [c for c in df.columns if 'team' in c or 'player' in c]
    df = df.dropna(subset=crit_cols, how='any')
    # Output cleaned file
    out_path = os.path.join(PROCESSED_DIR, fname.replace('.csv', '_cleaned.csv'))
    df.to_csv(out_path, index=False)
    print(f"Cleaned {fname} -> {os.path.basename(out_path)}")

if __name__ == "__main__":
    for fname in os.listdir(PROCESSED_DIR):
        if fname.lower().endswith('.csv') and not fname.endswith('_cleaned.csv'):
            clean_csv_file(fname) 