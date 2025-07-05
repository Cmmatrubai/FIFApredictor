# FIFA World Cup 2026 Winner Predictor

A data-driven tool to forecast the World Cup champion and individual award winners using historical and current player/team statistics.

## Project Structure

- `data/raw/` - Untouched CSV/JSON dumps
- `data/processed/` - Cleaned Parquet/CSV ready for features
- `src/data/` - Data ingestion & cleaning scripts
- `src/features/` - Feature engineering
- `src/models/` - Modeling scripts
- `src/simulation/` - Monte Carlo tournament logic
- `src/evaluation/` - Backtesting & metrics
- `src/dashboard/` - Web UI (Streamlit/FastAPI)
- `notebooks/` - EDA & prototyping
- `tests/` - Unit & integration tests

## Getting Started

### 1. Clone the repository

```bash
# If you haven't already, clone the repo and enter the directory
# git clone <repo-url>
cd worldcup_predictor
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Setup

- Place raw data files (e.g., `WorldCupPlayers.csv`, match and injury files) in `data/raw/`.
- Use the scripts in `src/data/` to ingest and clean data:

```bash
python src/data/fetch_data.py
python src/data/clean_data.py
```

### 4. Feature Engineering & Modeling

- Run feature engineering and model training scripts:

```bash
python src/features/feature_engineering.py
python src/models/train.py
```

### 5. Simulation & Evaluation

- Simulate tournaments and evaluate models:

```bash
python src/simulation/monte_carlo.py
python src/evaluation/backtest.py
```

### 6. Dashboard

- Launch the dashboard (Streamlit example):

```bash
streamlit run src/dashboard/app.py
```

## Setup

See `requirements.txt` for dependencies.
