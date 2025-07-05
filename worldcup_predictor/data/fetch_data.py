import requests
import pandas as pd

# Replace with your football-data.org API key
API_KEY = "81cf7bef380d41779cfacc9623c8aa7d"
HEADERS = {"X-Auth-Token": API_KEY}

def fetch_qualifiers(competition_id=2000, season=2026, output_csv="qualifiers_2026.csv"):
    """
    Fetches World Cup 2026 qualifying matches and writes to CSV.
    Uses football-data.org API:
      https://api.football-data.org/documentation/api
    competition_id=2000 is FIFA World Cup
    """
    url = f"https://api.football-data.org/v4/competitions/{competition_id}/matches?season={season}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json().get("matches", [])
    df = pd.json_normalize(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved qualifiers to {output_csv}")

def fetch_injuries(output_csv="injuries_2026.json.csv"):
    """
    Placeholder for injury data fetch.
    Replace INJURY_API_URL with a real endpoint that returns injury info in JSON.
    """
    INJURY_API_URL = "https://example.com/api/injuries?season=2026"
    r = requests.get(INJURY_API_URL, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    df = pd.json_normalize(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved injuries to {output_csv}")

if __name__ == "__main__":
    fetch_qualifiers()
    fetch_injuries()