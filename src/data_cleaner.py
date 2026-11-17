"""
src/data_cleaner.py
--------------------
DataCleaner class: loads, validates, and cleans the raw crime CSV.

Steps
-----
1. load_data          – read CSV with explicit dtypes
2. report_missing     – audit missing values per column
3. clean_weapon_column – impute NaN weapon_used → "Unknown"
4. clean_closure_dates – impute NaN closure_date using per-crime-type medians
5. clean_datetime     – parse date + time, extract temporal features
6. remove_duplicates  – drop duplicate incident_ids
7. validate_coordinates – remove rows with implausible lat/lon
8. get_clean_data     – run all steps in order and return the clean DataFrame
"""

import logging
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Chicago bounding box (generous)
LAT_MIN, LAT_MAX = 41.60, 42.10
LON_MIN, LON_MAX = -87.90, -87.40


class DataCleaner:
    """
    Loads and cleans the raw crime CSV file through a series of well-defined
    transformation steps.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the raw crime_data.csv file.

    Attributes
    ----------
    filepath : str
    df : pd.DataFrame  – working copy of the data, mutated by each step
    """

    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.df: pd.DataFrame = self.load_data()
        logger.info("DataCleaner initialised  shape=%s  file=%s", self.df.shape, filepath)

    # ── 1. Load ────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """
        Read the CSV with appropriate dtypes.

        Returns
        -------
        pd.DataFrame
        """
        dtype_map = {
            "incident_id":   "int64",
            "day_of_week":   "category",
            "crime_type":    "category",
            "location_type": "category",
            "district":      "category",
            "weapon_used":   "object",
            "arrest_made":   "bool",
            "severity":      "int8",
            "victim_count":  "int8",
        }
        df = pd.read_csv(
            self.filepath,
            dtype=dtype_map,
            parse_dates=False,   # we do our own datetime handling
        )
        logger.info("Loaded %d rows × %d cols", *df.shape)
        return df

    # ── 2. Report missing ──────────────────────────────────────────────────────

    def report_missing(self) -> pd.DataFrame:
        """
        Summarise missing values for each column.

        Returns
        -------
        pd.DataFrame
            Columns: ['column', 'missing_count', 'missing_pct']
        """
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        report = pd.DataFrame({
            "column":        missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_pct":   missing_pct.values,
        })
        report = report[report["missing_count"] > 0].reset_index(drop=True)
        logger.info("Missing-value report:\n%s", report.to_string(index=False))
        return report

    # ── 3. Weapon column ───────────────────────────────────────────────────────

    def clean_weapon_column(self) -> pd.DataFrame:
        """
        Impute NaN in `weapon_used` with the string "Unknown" and
        standardise capitalisation (title-case).

        Returns
        -------
        pd.DataFrame
        """
        before = self.df["weapon_used"].isna().sum()
        self.df["weapon_used"] = (
            self.df["weapon_used"]
            .fillna("Unknown")
            .str.strip()
            .str.title()
        )
        logger.info("clean_weapon_column: filled %d NaN → 'Unknown'", before)
        return self.df

    # ── 4. Closure dates ───────────────────────────────────────────────────────

    def clean_closure_dates(self) -> pd.DataFrame:
        """
        Impute missing `closure_date` using the per-crime-type median
        closure duration (days from incident date).  A boolean flag column
        `closure_date_estimated` is added.

        Returns
        -------
        pd.DataFrame
        """
        # Parse what we have
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df["closure_date"] = pd.to_datetime(self.df["closure_date"], errors="coerce")

        # Compute median closure time per crime type (from known records)
        known = self.df.dropna(subset=["closure_date"])
        known = known.copy()
        known["days_to_close"] = (known["closure_date"] - known["date"]).dt.days.clip(lower=0)
        median_days = known.groupby("crime_type", observed=True)["days_to_close"].median()

        # Fallback median across all types
        global_median = known["days_to_close"].median()

        self.df["closure_date_estimated"] = False

        missing_mask = self.df["closure_date"].isna()
        logger.info("clean_closure_dates: imputing %d missing closure dates", missing_mask.sum())

        for crime_type, grp_idx in self.df[missing_mask].groupby("crime_type", observed=True).groups.items():
            days = int(median_days.get(crime_type, global_median))
            self.df.loc[grp_idx, "closure_date"] = (
                self.df.loc[grp_idx, "date"] + pd.Timedelta(days=days)
            )
            self.df.loc[grp_idx, "closure_date_estimated"] = True

        return self.df

    # ── 5. Datetime features ───────────────────────────────────────────────────

    def clean_datetime(self) -> pd.DataFrame:
        """
        Parse `date` and `time` columns into a single `datetime` column and
        derive the following features:

        - hour          : int 0-23
        - month         : int 1-12
        - year          : int
        - day_of_week   : already present (kept as category)
        - is_weekend    : bool (Saturday or Sunday)
        - is_night      : bool (22:00-05:59 inclusive)

        Returns
        -------
        pd.DataFrame
        """
        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        # Combine date + time strings
        date_time_str = self.df["date"].dt.strftime("%Y-%m-%d") + " " + self.df["time"].astype(str)
        self.df["datetime"] = pd.to_datetime(date_time_str, errors="coerce")

        self.df["hour"]      = self.df["datetime"].dt.hour
        self.df["month"]     = self.df["datetime"].dt.month
        self.df["year"]      = self.df["datetime"].dt.year
        self.df["is_weekend"] = self.df["day_of_week"].isin(["Saturday", "Sunday"])
        self.df["is_night"]   = self.df["hour"].apply(lambda h: h >= 22 or h < 6)

        logger.info("clean_datetime: parsed datetime and derived hour/month/year/is_weekend/is_night")
        return self.df

    # ── 6. Remove duplicates ───────────────────────────────────────────────────

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Drop rows with duplicate `incident_id`.

        Returns
        -------
        pd.DataFrame
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=["incident_id"], keep="first")
        dropped = before - len(self.df)
        logger.info("remove_duplicates: dropped %d duplicate rows", dropped)
        return self.df

    # ── 7. Validate coordinates ────────────────────────────────────────────────

    def validate_coordinates(self) -> pd.DataFrame:
        """
        Remove rows where `latitude` or `longitude` fall outside the expected
        Chicago bounding box (or are NaN).

        Valid range:
            Latitude  : [41.60, 42.10]
            Longitude : [-87.90, -87.40]

        Returns
        -------
        pd.DataFrame
        """
        before = len(self.df)
        lat_ok = self.df["latitude"].between(LAT_MIN, LAT_MAX)
        lon_ok = self.df["longitude"].between(LON_MIN, LON_MAX)
        self.df = self.df[lat_ok & lon_ok].reset_index(drop=True)
        dropped = before - len(self.df)
        logger.info("validate_coordinates: removed %d rows with invalid coordinates", dropped)
        return self.df

    # ── 8. Full pipeline ───────────────────────────────────────────────────────

    def get_clean_data(self) -> pd.DataFrame:
        """
        Execute all cleaning steps in order and return the cleaned DataFrame.

        Order
        -----
        1. remove_duplicates
        2. clean_weapon_column
        3. clean_closure_dates  (requires raw date strings)
        4. clean_datetime       (parses date strings)
        5. validate_coordinates

        Returns
        -------
        pd.DataFrame
        """
        logger.info("=== Starting full cleaning pipeline ===")
        self.remove_duplicates()
        self.clean_weapon_column()
        self.clean_closure_dates()
        self.clean_datetime()
        self.validate_coordinates()
        logger.info("=== Cleaning complete — final shape: %s ===", self.df.shape)
        return self.df


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found.  Run  python data/generate_data.py  first.")
        sys.exit(1)

    cleaner = DataCleaner(raw_path)
    missing_report = cleaner.report_missing()
    print("\nMissing-value report:\n", missing_report)

    clean_df = cleaner.get_clean_data()
    print(f"\nClean data shape: {clean_df.shape}")
    print(clean_df.dtypes)
