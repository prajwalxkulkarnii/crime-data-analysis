"""
Crime Data Visualizer
Generates all publication-ready plots for the crime analysis system.
Saves PNG files to the outputs/ directory.
"""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class CrimeVisualizer:
    """
    Generates static visualizations for crime data analysis.

    Parameters
    ----------
    output_dir : str  Directory where PNGs are saved (created if missing).
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    # ------------------------------------------------------------------
    def plot_crime_distribution(self, df: pd.DataFrame) -> str:
        """Horizontal bar chart: incident count per crime type."""
        counts = (
            df["crime_type"].value_counts().reset_index()
        )
        counts.columns = ["crime_type", "count"]
        counts["pct"] = counts["count"] / counts["count"].sum() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("husl", len(counts))
        bars = ax.barh(counts["crime_type"], counts["count"], color=palette)
        for bar, (_, row) in zip(bars, counts.iterrows()):
            ax.text(
                bar.get_width() + 50,
                bar.get_y() + bar.get_height() / 2,
                f"{row['pct']:.1f}%",
                va="center", fontsize=9
            )
        ax.set_title("Crime Type Distribution", fontsize=15, fontweight="bold")
        ax.set_xlabel("Number of Incidents")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return self._save(fig, "01_crime_distribution.png")

    # ------------------------------------------------------------------
    def plot_temporal_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap: hour-of-day × day-of-week crime frequency."""
        if "hour" not in df.columns:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.hour

        pivot = df.pivot_table(
            index="day_of_week", columns="hour",
            values="incident_id", aggfunc="count", fill_value=0
        )
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd",
            linewidths=0.3,
            cbar_kws={"label": "Incident Count"},
        )
        ax.set_title("Crime Frequency: Hour of Day × Day of Week", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hour of Day (0–23)")
        ax.set_ylabel("Day of Week")
        fig.tight_layout()
        return self._save(fig, "02_temporal_heatmap.png")

    # ------------------------------------------------------------------
    def plot_district_choropleth(self, df: pd.DataFrame) -> str:
        """Bar chart of total incident count per district."""
        dist_counts = df.groupby("district")["incident_id"].count().sort_values(ascending=False)
        palette = sns.color_palette("flare", len(dist_counts))

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(dist_counts.index, dist_counts.values, color=palette, edgecolor="white")
        ax.bar_label(bars, fmt="%d", fontsize=9, padding=3)
        ax.set_title("Total Incidents by District", fontsize=14, fontweight="bold")
        ax.set_xlabel("District")
        ax.set_ylabel("Incident Count")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return self._save(fig, "03_district_incidents.png")

    # ------------------------------------------------------------------
    def plot_monthly_trend(self, df: pd.DataFrame) -> str:
        """Multi-year monthly crime trend lines."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        monthly = df.groupby(["year","month"])["incident_id"].count().reset_index()
        monthly.rename(columns={"incident_id": "incidents"}, inplace=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        for yr, grp in monthly.groupby("year"):
            ax.plot(grp["month"], grp["incidents"], marker="o", label=str(yr), lw=2)

        ax.set_xticks(range(1,13))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_title("Monthly Crime Trend by Year", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Incidents")
        ax.legend(title="Year", loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return self._save(fig, "04_monthly_trend.png")

    # ------------------------------------------------------------------
    def plot_arrest_rates(self, df: pd.DataFrame) -> str:
        """Stacked bar: arrested vs not-arrested per crime type."""
        grp = df.groupby(["crime_type","arrest_made"]).size().unstack(fill_value=0)
        grp.columns = ["Not Arrested","Arrested"]
        grp = grp.loc[grp.sum(axis=1).sort_values(ascending=True).index]

        fig, ax = plt.subplots(figsize=(10, 7))
        grp.plot(kind="barh", stacked=True, ax=ax,
                 color=["#E74C3C","#2ECC71"], edgecolor="white")
        ax.set_title("Arrest Outcomes by Crime Type", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Incidents")
        ax.set_ylabel("")
        ax.legend(loc="lower right")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        return self._save(fig, "05_arrest_rates.png")

    # ------------------------------------------------------------------
    def plot_severity_distribution(self, df: pd.DataFrame) -> str:
        """Box plot of crime severity score per crime type."""
        order = df.groupby("crime_type")["severity"].median().sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df, x="crime_type", y="severity",
            order=order, palette="coolwarm", ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
        ax.set_title("Crime Severity Distribution by Type", fontsize=14, fontweight="bold")
        ax.set_ylabel("Severity Score (1–5)")
        ax.set_xlabel("")
        fig.tight_layout()
        return self._save(fig, "06_severity_distribution.png")

    # ------------------------------------------------------------------
    def plot_resource_allocation(self, allocation_df: pd.DataFrame) -> str:
        """Bar chart: recommended patrol weight per district."""
        fig, ax = plt.subplots(figsize=(9, 5))
        col = "allocation_weight" if "allocation_weight" in allocation_df.columns else allocation_df.select_dtypes("number").columns[-1]
        bars = ax.bar(
            allocation_df["district"],
            allocation_df[col],
            color=sns.color_palette("rocket", len(allocation_df)),
            edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title("Recommended Patrol Resource Allocation by District",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("District")
        ax.set_ylabel("Allocation Weight")
        fig.tight_layout()
        return self._save(fig, "07_resource_allocation.png")
