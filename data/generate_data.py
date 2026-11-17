"""
data/generate_data.py
---------------------
Generates a synthetic crime dataset with 40,000+ rows and 14 columns,
incorporating realistic correlations between crime types, weapons, districts,
and time-of-day patterns.
"""

import os
import random
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_ROWS = 42_000

CRIME_TYPES = [
    "Theft", "Robbery", "Assault", "Burglary", "Vandalism",
    "Drug Offense", "Homicide", "Sexual Assault", "Fraud", "Arson",
]

LOCATION_TYPES = ["Street", "Residence", "Commercial", "Park", "Vehicle", "Transit"]

DISTRICTS = [f"D{i}" for i in range(1, 9)]  # D1 – D8

WEAPONS = [
    "Firearm", "Knife", "Blunt Object", "Hands/Fists", "Vehicle", "Other",
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Chicago area bounding box
LAT_CENTER, LON_CENTER = 41.8781, -87.6298
LAT_STD, LON_STD = 0.08, 0.10

# District crime-rate multipliers (D3 and D5 are hotspots)
DISTRICT_WEIGHTS = {
    "D1": 0.08, "D2": 0.09, "D3": 0.18, "D4": 0.10,
    "D5": 0.17, "D6": 0.11, "D7": 0.13, "D8": 0.14,
}

# Crime-type base probabilities
CRIME_WEIGHTS = [0.22, 0.12, 0.17, 0.11, 0.09, 0.09, 0.03, 0.05, 0.08, 0.04]

# Weapon probability per crime type (probability that a weapon IS used)
WEAPON_USE_PROB = {
    "Theft":          0.15,
    "Robbery":        0.72,
    "Assault":        0.60,
    "Burglary":       0.20,
    "Vandalism":      0.05,
    "Drug Offense":   0.10,
    "Homicide":       0.90,
    "Sexual Assault": 0.30,
    "Fraud":          0.00,
    "Arson":          0.05,
}

# Weapon distribution per crime type (conditional on weapon being used)
WEAPON_DIST = {
    "Theft":          [0.05, 0.10, 0.20, 0.50, 0.10, 0.05],
    "Robbery":        [0.50, 0.25, 0.10, 0.10, 0.03, 0.02],
    "Assault":        [0.30, 0.25, 0.20, 0.20, 0.03, 0.02],
    "Burglary":       [0.10, 0.20, 0.30, 0.30, 0.05, 0.05],
    "Vandalism":      [0.00, 0.10, 0.50, 0.30, 0.00, 0.10],
    "Drug Offense":   [0.20, 0.30, 0.10, 0.30, 0.00, 0.10],
    "Homicide":       [0.70, 0.15, 0.08, 0.05, 0.01, 0.01],
    "Sexual Assault": [0.10, 0.20, 0.10, 0.50, 0.00, 0.10],
    "Fraud":          [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # won't be used
    "Arson":          [0.00, 0.00, 0.10, 0.00, 0.00, 0.90],
}

# Severity distribution per crime type (mean, std) clipped to 1-5
SEVERITY_PARAMS = {
    "Theft":          (2.0, 0.8),
    "Robbery":        (3.5, 0.8),
    "Assault":        (3.2, 1.0),
    "Burglary":       (2.8, 0.9),
    "Vandalism":      (1.8, 0.7),
    "Drug Offense":   (2.5, 0.9),
    "Homicide":       (5.0, 0.3),
    "Sexual Assault": (4.2, 0.7),
    "Fraud":          (2.3, 0.8),
    "Arson":          (3.8, 0.9),
}

# Average days to close per crime type (for closure_date estimation)
CLOSURE_DAYS = {
    "Theft":          8,  "Robbery":        18, "Assault":        14,
    "Burglary":       12, "Vandalism":       6, "Drug Offense":   10,
    "Homicide":       42, "Sexual Assault":  28, "Fraud":          35,
    "Arson":          22,
}


# ── Helper functions ────────────────────────────────────────────────────────────

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta + 1)))


def night_biased_hour() -> int:
    """Return an hour with higher probability during 22:00-02:00."""
    base_weights = [1] * 24
    for h in [22, 23, 0, 1, 2]:
        base_weights[h] = 4
    for h in [18, 19, 20, 21]:
        base_weights[h] = 2
    total = sum(base_weights)
    probs = [w / total for w in base_weights]
    return rng.choice(24, p=probs)


def generate_coordinates(district: str) -> tuple[float, float]:
    """Generate lat/lon near Chicago with slight district offsets."""
    offset_map = {
        "D1": (0.04, -0.04), "D2": (0.04,  0.04),
        "D3": (-0.04, -0.04), "D4": (-0.04,  0.04),
        "D5": (0.00, -0.08), "D6": (0.00,  0.08),
        "D7": (0.08,  0.00), "D8": (-0.08,  0.00),
    }
    lat_off, lon_off = offset_map[district]
    lat = LAT_CENTER + lat_off + rng.normal(0, LAT_STD * 0.4)
    lon = LON_CENTER + lon_off + rng.normal(0, LON_STD * 0.4)
    return round(lat, 6), round(lon, 6)


# ── Main generation ────────────────────────────────────────────────────────────

def generate_crime_data(n: int = N_ROWS) -> pd.DataFrame:
    print(f"Generating {n:,} crime records …")

    start_date = date(2019, 1, 1)
    end_date   = date(2023, 12, 31)

    records = []
    for i in range(1, n + 1):
        # District (weighted toward D3/D5)
        district = rng.choice(
            DISTRICTS,
            p=[DISTRICT_WEIGHTS[d] for d in DISTRICTS],
        )

        # Crime type
        crime_type = rng.choice(CRIME_TYPES, p=CRIME_WEIGHTS)

        # Date and time
        incident_date = random_date(start_date, end_date)
        hour = night_biased_hour()
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        day_of_week = incident_date.strftime("%A")

        # Coordinates
        lat, lon = generate_coordinates(district)

        # Weapon
        use_weapon = rng.random() < WEAPON_USE_PROB[crime_type]
        if use_weapon:
            weapon = rng.choice(WEAPONS, p=WEAPON_DIST[crime_type])
        else:
            weapon = np.nan  # ~40% overall NaN rate

        # Arrest
        arrest_probs = {
            "Theft": 0.25, "Robbery": 0.45, "Assault": 0.40,
            "Burglary": 0.30, "Vandalism": 0.20, "Drug Offense": 0.58,
            "Homicide": 0.35, "Sexual Assault": 0.22, "Fraud": 0.12, "Arson": 0.14,
        }
        arrest_made = rng.random() < arrest_probs[crime_type]

        # Closure date (30% null for open cases)
        if rng.random() < 0.70:
            avg_days = CLOSURE_DAYS[crime_type]
            days_added = max(1, int(rng.normal(avg_days, avg_days * 0.4)))
            closure_dt = incident_date + timedelta(days=days_added)
            closure_date = closure_dt.strftime("%Y-%m-%d")
        else:
            closure_date = np.nan

        # Severity
        mean_sev, std_sev = SEVERITY_PARAMS[crime_type]
        severity = int(np.clip(round(rng.normal(mean_sev, std_sev)), 1, 5))

        # Victim count
        victim_count = int(rng.choice([1, 2, 3, 4, 5], p=[0.60, 0.20, 0.10, 0.06, 0.04]))

        # Location type
        loc_weights = {
            "Street":     0.30, "Residence": 0.25, "Commercial": 0.20,
            "Park":       0.08, "Vehicle":   0.10, "Transit":    0.07,
        }
        location_type = rng.choice(
            list(loc_weights.keys()),
            p=list(loc_weights.values()),
        )

        records.append({
            "incident_id":   i,
            "date":          incident_date.strftime("%Y-%m-%d"),
            "time":          time_str,
            "day_of_week":   day_of_week,
            "crime_type":    crime_type,
            "location_type": location_type,
            "district":      district,
            "latitude":      lat,
            "longitude":     lon,
            "weapon_used":   weapon,
            "arrest_made":   arrest_made,
            "closure_date":  closure_date,
            "severity":      severity,
            "victim_count":  victim_count,
        })

    df = pd.DataFrame(records)
    print(f"Dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "crime_data.csv")

    df = generate_crime_data(N_ROWS)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(df.head(3).to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
