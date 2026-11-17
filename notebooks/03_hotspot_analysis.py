"""
Hotspot Analysis notebook-style script.
Run:  python notebooks/03_hotspot_analysis.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_cleaner import DataCleaner
from hotspot_detector import HotspotDetector
from analyzer import CrimeAnalyzer

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data_clean.csv")
RAW_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Crime Hotspot Analysis  (Notebook 03)")
print("=" * 60)

if os.path.exists(CLEAN_PATH):
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
else:
    cleaner = DataCleaner(RAW_PATH)
    df = cleaner.get_clean_data()
    df.to_csv(CLEAN_PATH, index=False)

# 1. Grid-based hotspots
print("\n--- Grid-Based Hotspot Cells ---")
detector = HotspotDetector(df)
grid = detector.grid_based_hotspots(grid_size=0.01)
print(grid.head(10).to_string(index=False))

# 2. DBSCAN clusters
print("\n--- DBSCAN Clustering ---")
try:
    labels, centroids = detector.dbscan_clusters(eps=0.01, min_samples=10)
    n_clusters = len(centroids)
    print(f"  Clusters found: {n_clusters}")
    print(centroids.to_string(index=False))

    # Scatter plot of clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_df = df[["latitude","longitude"]].copy()
    scatter_df["cluster"] = labels
    colors = sns.color_palette("tab20", n_colors=max(n_clusters, 1))
    for idx, (cid, grp) in enumerate(scatter_df.groupby("cluster")):
        label = f"Cluster {cid}" if cid >= 0 else "Noise"
        color = "lightgray" if cid < 0 else colors[idx % len(colors)]
        ax.scatter(grp["longitude"], grp["latitude"], c=[color], s=5, label=label, alpha=0.5)
    ax.set_title("Crime Spatial Clusters (DBSCAN)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if n_clusters <= 15:
        ax.legend(markerscale=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_dbscan_clusters.png"), dpi=140)
    plt.close(fig)
    print(f"  Cluster plot → {OUT_DIR}/03_dbscan_clusters.png")
except Exception as e:
    print(f"  DBSCAN skipped: {e}")

# 3. Patrol recommendation
print("\n--- Resource Allocation Report ---")
report = detector.generate_hotspot_report()
print(report.to_string(index=False))

# 4. Risk level demo
print("\n--- Sample Risk Predictions ---")
samples = [
    (41.85, -87.65, 23),   # night-time typical location
    (41.90, -87.70, 10),   # daytime
    (41.75, -87.55,  2),   # 2am
]
for lat, lon, hr in samples:
    risk = detector.predict_risk_level(lat, lon, hr)
    print(f"  ({lat}, {lon}) at {hr:02d}:00 → Risk: {risk}")

print("\nHotspot analysis complete.")
