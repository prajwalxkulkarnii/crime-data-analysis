"""
Unit tests for DataCleaner and CrimeAnalyzer.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_cleaner import DataCleaner
from analyzer import CrimeAnalyzer


# ------------------------------------------------------------------ #
#  Synthetic DataFrame factory                                         #
# ------------------------------------------------------------------ #

def make_crime_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crime_types = ["Theft","Robbery","Assault","Burglary","Vandalism",
                   "Drug Offense","Homicide","Sexual Assault","Fraud","Arson"]
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    districts = [f"D{i}" for i in range(1, 9)]

    df = pd.DataFrame({
        "incident_id":   range(1, n + 1),
        "date":          pd.date_range("2021-01-01", periods=n, freq="4h").date,
        "time":          ["10:30:00"] * n,
        "day_of_week":   rng.choice(days, n),
        "crime_type":    rng.choice(crime_types, n),
        "location_type": rng.choice(["Street","Residence","Commercial"], n),
        "district":      rng.choice(districts, n),
        "latitude":      rng.uniform(41.6, 42.0, n),
        "longitude":     rng.uniform(-87.9, -87.5, n),
        "weapon_used":   [None if rng.random() < 0.4 else "Handgun" for _ in range(n)],
        "arrest_made":   rng.choice([True, False], n),
        "closure_date":  [None if rng.random() < 0.3 else "2022-01-10" for _ in range(n)],
        "severity":      rng.integers(1, 6, n),
        "victim_count":  rng.integers(1, 5, n),
    })
    return df


# ------------------------------------------------------------------ #
#  DataCleaner tests                                                   #
# ------------------------------------------------------------------ #

class TestDataCleaner:

    def test_report_missing_has_expected_cols(self, tmp_path):
        df = make_crime_df(200)
        csv_path = tmp_path / "crime.csv"
        df.to_csv(csv_path, index=False)
        cleaner = DataCleaner(str(csv_path))
        cleaner.load_data()
        report = cleaner.report_missing()
        assert "column" in report.columns
        assert "missing_count" in report.columns
        assert "missing_pct" in report.columns

    def test_clean_weapon_fills_nan(self, tmp_path):
        df = make_crime_df(100)
        csv_path = tmp_path / "crime.csv"
        df.to_csv(csv_path, index=False)
        cleaner = DataCleaner(str(csv_path))
        cleaner.load_data()
        result = cleaner.clean_weapon_column()
        assert result["weapon_used"].isnull().sum() == 0
        assert "Unknown" in result["weapon_used"].values

    def test_clean_datetime_adds_columns(self, tmp_path):
        df = make_crime_df(50)
        csv_path = tmp_path / "crime.csv"
        df.to_csv(csv_path, index=False)
        cleaner = DataCleaner(str(csv_path))
        cleaner.load_data()
        result = cleaner.clean_datetime()
        for col in ["hour", "is_weekend", "is_night", "month", "year"]:
            assert col in result.columns, f"Expected column '{col}'"

    def test_remove_duplicates(self, tmp_path):
        df = make_crime_df(100)
        df_duped = pd.concat([df, df.head(10)], ignore_index=True)
        csv_path = tmp_path / "crime.csv"
        df_duped.to_csv(csv_path, index=False)
        cleaner = DataCleaner(str(csv_path))
        cleaner.load_data()
        result = cleaner.remove_duplicates()
        assert len(result) <= len(df_duped)

    def test_validate_coordinates(self, tmp_path):
        df = make_crime_df(50)
        # Inject bad rows
        df.loc[0, "latitude"] = 999.0
        df.loc[1, "longitude"] = -999.0
        csv_path = tmp_path / "crime.csv"
        df.to_csv(csv_path, index=False)
        cleaner = DataCleaner(str(csv_path))
        cleaner.load_data()
        result = cleaner.validate_coordinates()
        assert (result["latitude"].between(-90, 90)).all()
        assert (result["longitude"].between(-180, 180)).all()


# ------------------------------------------------------------------ #
#  CrimeAnalyzer tests                                                 #
# ------------------------------------------------------------------ #

class TestCrimeAnalyzer:

    @pytest.fixture
    def analyzer(self):
        df = make_crime_df(500)
        # Apply minimal cleaning manually so analyzer gets clean data
        df["weapon_used"] = df["weapon_used"].fillna("Unknown")
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = 10
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["is_weekend"] = df["day_of_week"].isin(["Saturday","Sunday"]).astype(int)
        df["is_night"] = 0
        return CrimeAnalyzer(df)

    def test_crime_type_distribution_pct_sums_100(self, analyzer):
        result = analyzer.crime_type_distribution()
        assert abs(result["percentage"].sum() - 100.0) < 0.5

    def test_district_hotspots_sorted(self, analyzer):
        result = analyzer.district_hotspots()
        assert list(result["crime_count"]) == sorted(result["crime_count"], reverse=True)

    def test_arrest_rate_between_0_and_100(self, analyzer):
        result = analyzer.arrest_rate_by_crime()
        assert (result["arrest_rate_pct"] >= 0).all()
        assert (result["arrest_rate_pct"] <= 100).all()

    def test_resource_allocation_has_all_districts(self, analyzer):
        result = analyzer.resource_allocation_score()
        df = make_crime_df(500)
        expected_districts = set(df["district"].unique())
        got_districts = set(result["district"].unique())
        assert expected_districts == got_districts

    def test_high_risk_zones_respects_top_n(self, analyzer):
        for n in [1, 2, 3]:
            result = analyzer.high_risk_zones(top_n=n)
            assert len(result) == n
