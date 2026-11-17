"""
src/analyzer.py
---------------
CrimeAnalyzer class: computes statistical summaries and rankings from the
cleaned crime DataFrame.

All methods are pure (they do not mutate self.df) and return new DataFrames
or dicts that can be consumed by visualisers, reporters, or notebooks.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrimeAnalyzer:
    """
    Analyses cleaned crime data to produce statistical summaries.

    Parameters
    ----------
    df : pd.DataFrame
        A fully cleaned crime DataFrame (output of DataCleaner.get_clean_data).
    """

    DISTRICTS = [f"D{i}" for i in range(1, 9)]

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()
        logger.info("CrimeAnalyzer initialised with %d rows", len(self.df))

    # ── 1. Crime type distribution ─────────────────────────────────────────────

    def crime_type_distribution(self) -> pd.DataFrame:
        """
        Count incidents and compute percentages per crime type.

        Returns
        -------
        pd.DataFrame
            Columns: ['crime_type', 'count', 'percentage']
            Sorted by count descending.
        """
        counts = (
            self.df["crime_type"]
            .value_counts()
            .rename_axis("crime_type")
            .reset_index(name="count")
        )
        counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
        logger.info("crime_type_distribution computed for %d types", len(counts))
        return counts

    # ── 2. Temporal analysis ───────────────────────────────────────────────────

    def temporal_analysis(self) -> dict[str, Any]:
        """
        Break down crime counts by hour, day-of-week, month, and year.

        Returns
        -------
        dict with keys:
            'by_hour'       : pd.Series (index = 0-23)
            'by_day'        : pd.Series (index = day names)
            'by_month'      : pd.Series (index = 1-12)
            'by_year'       : pd.Series (index = year ints)
            'hour_day_pivot': pd.DataFrame pivot (hour × day)
        """
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        by_hour = self.df["hour"].value_counts().sort_index()
        by_day  = (
            self.df["day_of_week"]
            .value_counts()
            .reindex(day_order, fill_value=0)
        )
        by_month = self.df["month"].value_counts().sort_index()
        by_year  = self.df["year"].value_counts().sort_index()

        # Hour × day pivot for heatmap
        pivot = (
            self.df.groupby(["hour", "day_of_week"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        # Reorder columns to Mon-Sun
        present_days = [d for d in day_order if d in pivot.columns]
        pivot = pivot[present_days]

        result = {
            "by_hour":        by_hour,
            "by_day":         by_day,
            "by_month":       by_month,
            "by_year":        by_year,
            "hour_day_pivot": pivot,
        }
        logger.info("temporal_analysis complete")
        return result

    # ── 3. District hotspots ───────────────────────────────────────────────────

    def district_hotspots(self) -> pd.DataFrame:
        """
        Count crimes per district and compute density rank.

        Returns
        -------
        pd.DataFrame
            Columns: ['district', 'crime_count', 'pct_of_total', 'rank']
            Sorted by crime_count descending.
        """
        counts = (
            self.df["district"]
            .value_counts()
            .rename_axis("district")
            .reset_index(name="crime_count")
        )
        counts["pct_of_total"] = (counts["crime_count"] / counts["crime_count"].sum() * 100).round(2)
        counts = counts.sort_values("crime_count", ascending=False).reset_index(drop=True)
        counts["rank"] = counts.index + 1
        logger.info("district_hotspots:\n%s", counts.to_string(index=False))
        return counts

    # ── 4. Arrest rate by crime ────────────────────────────────────────────────

    def arrest_rate_by_crime(self) -> pd.DataFrame:
        """
        Compute arrest rate percentage for each crime type.

        Returns
        -------
        pd.DataFrame
            Columns: ['crime_type', 'total', 'arrests', 'arrest_rate_pct']
            Sorted by arrest_rate_pct descending.
        """
        grp = self.df.groupby("crime_type", observed=True)
        result = grp["arrest_made"].agg(total="count", arrests="sum").reset_index()
        result["arrest_rate_pct"] = (result["arrests"] / result["total"] * 100).round(2)
        result = result.sort_values("arrest_rate_pct", ascending=False).reset_index(drop=True)
        logger.info("arrest_rate_by_crime computed")
        return result

    # ── 5. Severity analysis ───────────────────────────────────────────────────

    def severity_analysis(self) -> pd.DataFrame:
        """
        Average severity per crime type AND per district.

        Returns
        -------
        pd.DataFrame
            Index = crime_type, columns = district names + 'overall_avg'
        """
        # Per crime type overall
        overall = (
            self.df.groupby("crime_type", observed=True)["severity"]
            .mean()
            .round(2)
            .rename("overall_avg")
        )
        # Per crime type × district pivot
        pivot = (
            self.df.groupby(["crime_type", "district"], observed=True)["severity"]
            .mean()
            .round(2)
            .unstack(fill_value=np.nan)
        )
        result = pivot.join(overall)
        logger.info("severity_analysis pivot shape: %s", result.shape)
        return result

    # ── 6. Weapon usage patterns ───────────────────────────────────────────────

    def weapon_usage_patterns(self) -> pd.DataFrame:
        """
        Cross-tabulate weapon frequency by crime type.

        Returns
        -------
        pd.DataFrame
            Rows = crime_type, columns = weapon_used values, values = counts.
        """
        pivot = pd.crosstab(
            self.df["crime_type"],
            self.df["weapon_used"],
        )
        logger.info("weapon_usage_patterns pivot shape: %s", pivot.shape)
        return pivot

    # ── 7. Closure time analysis ───────────────────────────────────────────────

    def closure_time_analysis(self) -> pd.DataFrame:
        """
        Average days to close cases per crime type (excludes estimated closures
        if the flag column exists).

        Returns
        -------
        pd.DataFrame
            Columns: ['crime_type', 'avg_days_to_close', 'median_days_to_close', 'sample_size']
        """
        df = self.df.copy()

        # Exclude estimated closures if the flag is present
        if "closure_date_estimated" in df.columns:
            df = df[~df["closure_date_estimated"]]

        # Ensure datetime types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["closure_date"] = pd.to_datetime(df["closure_date"], errors="coerce")

        df = df.dropna(subset=["date", "closure_date"])
        df["days_to_close"] = (df["closure_date"] - df["date"]).dt.days.clip(lower=0)

        result = (
            df.groupby("crime_type", observed=True)["days_to_close"]
            .agg(avg_days_to_close="mean", median_days_to_close="median", sample_size="count")
            .round(1)
            .reset_index()
            .sort_values("avg_days_to_close", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("closure_time_analysis computed for %d crime types", len(result))
        return result

    # ── 8. Resource allocation score ───────────────────────────────────────────

    def resource_allocation_score(self) -> pd.DataFrame:
        """
        Compute a composite risk/resource score per district:

            score = crime_count × avg_severity

        The recommended patrol units are scaled relative to the max score.

        Returns
        -------
        pd.DataFrame
            Columns: ['district', 'crime_count', 'avg_severity', 'score',
                      'score_normalised', 'recommended_units']
            Sorted by score descending.
        """
        grp = self.df.groupby("district", observed=True)
        result = grp.agg(
            crime_count=("incident_id", "count"),
            avg_severity=("severity", "mean"),
        ).reset_index()
        result["score"] = (result["crime_count"] * result["avg_severity"]).round(1)
        result["score_normalised"] = (result["score"] / result["score"].max()).round(4)
        # Scale to 2-15 patrol units range
        result["recommended_units"] = (
            (result["score_normalised"] * 13 + 2).round().astype(int)
        )
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        logger.info("resource_allocation_score:\n%s", result.to_string(index=False))
        return result

    # ── 9. High-risk zones ─────────────────────────────────────────────────────

    def high_risk_zones(self, top_n: int = 3) -> pd.DataFrame:
        """
        Identify the top N highest-risk districts with brief reasoning.

        Parameters
        ----------
        top_n : int, default 3

        Returns
        -------
        pd.DataFrame
            Columns: ['rank', 'district', 'crime_count', 'avg_severity',
                      'score', 'dominant_crime', 'peak_hour', 'reasoning']
        """
        alloc = self.resource_allocation_score().head(top_n).copy()

        dominant_crime = (
            self.df.groupby(["district", "crime_type"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates("district")
            .set_index("district")["crime_type"]
        )

        peak_hour = (
            self.df.groupby(["district", "hour"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates("district")
            .set_index("district")["hour"]
        )

        alloc["dominant_crime"] = alloc["district"].map(dominant_crime)
        alloc["peak_hour"]      = alloc["district"].map(peak_hour)
        alloc["rank"]           = range(1, top_n + 1)

        alloc["reasoning"] = alloc.apply(
            lambda r: (
                f"District {r['district']} ranks #{r['rank']} with {r['crime_count']:,} incidents "
                f"(avg severity {r['avg_severity']:.2f}). "
                f"Dominant crime: {r['dominant_crime']}. "
                f"Peak activity at {r['peak_hour']:02d}:00."
            ),
            axis=1,
        )
        cols = ["rank", "district", "crime_count", "avg_severity",
                "score", "dominant_crime", "peak_hour", "reasoning"]
        return alloc[cols].reset_index(drop=True)
