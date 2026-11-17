"""
src/hotspot_detector.py
------------------------
HotspotDetector class: spatial and temporal hotspot analysis using
grid-based binning and DBSCAN clustering.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HotspotDetector:
    """
    Detects crime hotspots using multiple spatial and temporal methods.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned crime DataFrame with at minimum:
        'latitude', 'longitude', 'district', 'hour', 'crime_type', 'severity'
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()
        # Pre-filter invalid coordinates
        self.df = self.df.dropna(subset=["latitude", "longitude"])
        logger.info("HotspotDetector ready with %d records", len(self.df))

    # ── 1. Grid-based hotspots ─────────────────────────────────────────────────

    def grid_based_hotspots(self, grid_size: float = 0.01) -> pd.DataFrame:
        """
        Bin lat/lon into a regular grid and count incidents per cell.

        Parameters
        ----------
        grid_size : float
            Grid cell side length in decimal degrees (~1 km at Chicago's latitude).

        Returns
        -------
        pd.DataFrame
            Columns: ['lat_bin', 'lon_bin', 'lat_center', 'lon_center',
                      'crime_count', 'avg_severity']
            Sorted by crime_count descending.
        """
        df = self.df.copy()
        df["lat_bin"] = (df["latitude"]  / grid_size).apply(np.floor) * grid_size
        df["lon_bin"] = (df["longitude"] / grid_size).apply(np.floor) * grid_size

        grid = (
            df.groupby(["lat_bin", "lon_bin"])
            .agg(crime_count=("incident_id", "count"), avg_severity=("severity", "mean"))
            .reset_index()
        )
        grid["lat_center"] = grid["lat_bin"] + grid_size / 2
        grid["lon_center"] = grid["lon_bin"] + grid_size / 2
        grid["avg_severity"] = grid["avg_severity"].round(2)
        grid = grid.sort_values("crime_count", ascending=False).reset_index(drop=True)
        logger.info("grid_based_hotspots: %d cells | top cell count=%d",
                    len(grid), grid["crime_count"].iloc[0])
        return grid

    # ── 2. DBSCAN clusters ─────────────────────────────────────────────────────

    def dbscan_clusters(
        self,
        eps: float = 0.005,
        min_samples: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run DBSCAN spatial clustering on lat/lon coordinates.

        Parameters
        ----------
        eps : float
            Neighbourhood radius in decimal degrees.
        min_samples : int
            Minimum points to form a dense region.

        Returns
        -------
        (labeled_df, centroids_df) tuple
            labeled_df  – original df with 'cluster_label' column appended
            centroids_df – cluster id, centroid lat/lon, size, dominant crime
        """
        coords = self.df[["latitude", "longitude"]].values
        # Fit DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", metric="haversine",
                    n_jobs=-1)
        # haversine expects radians
        coords_rad = np.radians(coords)
        labels = db.fit_predict(coords_rad)

        labeled_df = self.df.copy()
        labeled_df["cluster_label"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).sum()
        logger.info("DBSCAN: %d clusters found, %d noise points", n_clusters, noise)

        # Build centroids summary
        core_df = labeled_df[labeled_df["cluster_label"] >= 0]
        centroids = (
            core_df.groupby("cluster_label")
            .agg(
                centroid_lat=("latitude", "mean"),
                centroid_lon=("longitude", "mean"),
                cluster_size=("incident_id", "count"),
            )
            .reset_index()
        )
        # Dominant crime per cluster
        dom_crime = (
            core_df.groupby(["cluster_label", "crime_type"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates("cluster_label")
            .set_index("cluster_label")["crime_type"]
        )
        centroids["dominant_crime"] = centroids["cluster_label"].map(dom_crime)
        centroids = centroids.sort_values("cluster_size", ascending=False).reset_index(drop=True)

        return labeled_df, centroids

    # ── 3. Time-window hotspots ────────────────────────────────────────────────

    def time_window_hotspots(
        self,
        district: str,
        window_hours: int = 2,
    ) -> pd.DataFrame:
        """
        Identify peak crime time windows within a specific district.

        Parameters
        ----------
        district : str
            e.g. "D3"
        window_hours : int
            Size of the rolling window in hours.

        Returns
        -------
        pd.DataFrame
            Columns: ['window_start', 'window_end', 'crime_count']
            Sorted by crime_count descending.
        """
        sub = self.df[self.df["district"] == district].copy()
        if sub.empty:
            logger.warning("time_window_hotspots: no data for district %s", district)
            return pd.DataFrame(columns=["window_start", "window_end", "crime_count"])

        hourly = sub["hour"].value_counts().reindex(range(24), fill_value=0)

        rows = []
        for start in range(24):
            hours = [(start + i) % 24 for i in range(window_hours)]
            count = int(hourly[hours].sum())
            rows.append({
                "window_start": start,
                "window_end":   (start + window_hours) % 24,
                "crime_count":  count,
            })
        result = pd.DataFrame(rows).sort_values("crime_count", ascending=False).reset_index(drop=True)
        logger.info("time_window_hotspots district=%s window=%dh top_window=%d–%d (%d crimes)",
                    district, window_hours,
                    result.iloc[0]["window_start"], result.iloc[0]["window_end"],
                    result.iloc[0]["crime_count"])
        return result

    # ── 4. Predict risk level ──────────────────────────────────────────────────

    def predict_risk_level(self, lat: float, lon: float, hour: int) -> str:
        """
        Rule-based risk estimation for a given location and time.

        Parameters
        ----------
        lat, lon : float
            Target coordinates.
        hour : int
            Hour of day (0-23).

        Returns
        -------
        str
            "HIGH", "MEDIUM", or "LOW"
        """
        # Find nearest grid cell crime count
        grid = self.grid_based_hotspots(grid_size=0.01)
        grid["dist"] = np.sqrt((grid["lat_center"] - lat) ** 2 + (grid["lon_center"] - lon) ** 2)
        nearest = grid.nsmallest(1, "dist").iloc[0]
        local_count = nearest["crime_count"]

        # Night bonus
        is_night = hour >= 22 or hour < 6

        # Thresholds derived from grid distribution
        p75 = grid["crime_count"].quantile(0.75)
        p50 = grid["crime_count"].quantile(0.50)

        if local_count >= p75 or (local_count >= p50 and is_night):
            return "HIGH"
        elif local_count >= p50 or is_night:
            return "MEDIUM"
        else:
            return "LOW"

    # ── 5. Full hotspot report ─────────────────────────────────────────────────

    def generate_hotspot_report(self) -> pd.DataFrame:
        """
        Produce a consolidated hotspot report per district.

        Returns
        -------
        pd.DataFrame
            Columns: ['district', 'crime_count', 'peak_hour', 'peak_crime_type',
                      'avg_severity', 'risk_level', 'recommended_patrol_units']
        """
        rows = []
        for district in sorted(self.df["district"].unique()):
            sub = self.df[self.df["district"] == district]
            crime_count = len(sub)
            peak_hour   = int(sub["hour"].mode().iloc[0]) if not sub.empty else 0
            peak_crime  = sub["crime_type"].mode().iloc[0] if not sub.empty else "Unknown"
            avg_sev     = round(sub["severity"].mean(), 2)

            # Representative centroid for risk level
            lat_c = sub["latitude"].mean()
            lon_c = sub["longitude"].mean()
            risk  = self.predict_risk_level(lat_c, lon_c, peak_hour)

            patrol_units = int(crime_count / 100) + 1

            rows.append({
                "district":                district,
                "crime_count":             crime_count,
                "peak_hour":               peak_hour,
                "peak_crime_type":         peak_crime,
                "avg_severity":            avg_sev,
                "risk_level":              risk,
                "recommended_patrol_units": patrol_units,
            })

        result = pd.DataFrame(rows).sort_values("crime_count", ascending=False).reset_index(drop=True)
        logger.info("generate_hotspot_report:\n%s", result.to_string(index=False))
        return result
