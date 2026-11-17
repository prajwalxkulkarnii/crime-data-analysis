"""
Crime Data Analysis - Main Pipeline Runner
Run this end-to-end to clean data, analyze patterns, detect hotspots,
generate visualizations, and print a resource allocation report.

Usage
-----
    # First generate synthetic data:
    python data/generate_data.py

    # Then run the full pipeline:
    python src/main.py
"""

import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_cleaner import DataCleaner
from analyzer import CrimeAnalyzer
from visualizer import CrimeVisualizer
from hotspot_detector import HotspotDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def banner(msg: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {msg}")
    print("=" * 65)


def main() -> None:
    t0 = time.time()
    RAW_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
    CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data_clean.csv")
    OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  1. Load & Clean                                                     #
    # ------------------------------------------------------------------ #
    banner("1 — Loading & Cleaning Data")
    cleaner = DataCleaner(RAW_PATH)
    df_clean = cleaner.get_clean_data()
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"  Clean data saved → {CLEAN_PATH}")
    print(f"  Shape: {df_clean.shape}")

    missing_report = cleaner.report_missing()
    print("\n  Missing-value report (top 5):")
    print(missing_report.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    #  2. Analysis                                                         #
    # ------------------------------------------------------------------ #
    banner("2 — Crime Analysis")
    analyzer = CrimeAnalyzer(df_clean)

    dist = analyzer.crime_type_distribution()
    print("\n  Crime Type Distribution (top 5):")
    print(dist.head(5).to_string(index=False))

    hotspots = analyzer.district_hotspots()
    print("\n  District Hotspots (ranked):")
    print(hotspots.to_string(index=False))

    arrest_rates = analyzer.arrest_rate_by_crime()
    print("\n  Arrest Rates by Crime Type:")
    print(arrest_rates.to_string(index=False))

    allocation = analyzer.resource_allocation_score()
    print("\n  Resource Allocation Weights:")
    print(allocation.to_string(index=False))

    high_risk = analyzer.high_risk_zones(top_n=3)
    print("\n  ⚠️  Top 3 HIGH-RISK Zones:")
    print(high_risk.to_string(index=False))

    # ------------------------------------------------------------------ #
    #  3. Visualizations                                                   #
    # ------------------------------------------------------------------ #
    banner("3 — Generating Visualizations")
    viz = CrimeVisualizer(output_dir=OUT_DIR)
    paths = []
    paths.append(viz.plot_crime_distribution(df_clean))
    paths.append(viz.plot_temporal_heatmap(df_clean))
    paths.append(viz.plot_district_choropleth(df_clean))
    paths.append(viz.plot_monthly_trend(df_clean))
    paths.append(viz.plot_arrest_rates(df_clean))
    paths.append(viz.plot_severity_distribution(df_clean))
    paths.append(viz.plot_resource_allocation(allocation))
    for p in paths:
        if p:
            print(f"  ✅ {os.path.basename(p)}")

    # ------------------------------------------------------------------ #
    #  4. Hotspot Detection                                                #
    # ------------------------------------------------------------------ #
    banner("4 — Hotspot Detection")
    detector = HotspotDetector(df_clean)
    grid = detector.grid_based_hotspots()
    print(f"\n  Grid-based hotspot cells: {len(grid)}")
    print("  Top 5 crime grid cells:")
    print(grid.head(5).to_string(index=False))

    report = detector.generate_hotspot_report()
    print("\n  Patrol Recommendation by District:")
    print(report.to_string(index=False))

    # ------------------------------------------------------------------ #
    #  5. Summary                                                          #
    # ------------------------------------------------------------------ #
    banner(f"PIPELINE COMPLETE in {time.time()-t0:.1f}s")
    print(f"  Total incidents:  {len(df_clean):,}")
    print(f"  Date range:       {df_clean['date'].min()} → {df_clean['date'].max()}")
    print(f"  Districts:        {df_clean['district'].nunique()}")
    print(f"  Crime types:      {df_clean['crime_type'].nunique()}")
    print(f"  Overall arrest %, {df_clean['arrest_made'].mean()*100:.1f}%")
    print(f"\n  Outputs saved to: {OUT_DIR}")
    print()


if __name__ == "__main__":
    main()
