"""
EDA and Visualization notebook-style script for Crime Data Analysis.
Run:  python notebooks/02_eda_visualization.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from data_cleaner import DataCleaner
from visualizer import CrimeVisualizer

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data_clean.csv")
RAW_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Crime Data EDA — Visualizations  (Notebook 02)")
print("=" * 60)

# Load clean data (generate if missing)
if os.path.exists(CLEAN_PATH):
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
else:
    print("Clean data not found — running cleaner first...")
    cleaner = DataCleaner(RAW_PATH)
    df = cleaner.get_clean_data()
    df.to_csv(CLEAN_PATH, index=False)

print(f"\nLoaded: {df.shape[0]:,} records × {df.shape[1]} columns")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Districts:  {sorted(df['district'].unique())}")
print(f"Crime types ({df['crime_type'].nunique()}): {list(df['crime_type'].unique())}")

# Key stats
print("\n--- Summary Stats ---")
print(f"  Arrest rate:     {df['arrest_made'].mean()*100:.1f}%")
print(f"  Avg severity:    {df['severity'].mean():.2f}")
print(f"  Weapon unknown:  {(df['weapon_used']=='Unknown').mean()*100:.1f}%")
print(f"  Night crimes:    {df.get('is_night', pd.Series([0]*len(df))).mean()*100:.1f}%")

# Generate all plots
print("\n--- Generating Visualizations ---")
viz = CrimeVisualizer(output_dir=OUT_DIR)
plots = [
    viz.plot_crime_distribution(df),
    viz.plot_temporal_heatmap(df),
    viz.plot_district_choropleth(df),
    viz.plot_monthly_trend(df),
    viz.plot_arrest_rates(df),
    viz.plot_severity_distribution(df),
]
for p in plots:
    if p: print(f"  ✅ {os.path.basename(p)}")

print("\nAll visualizations saved to:", OUT_DIR)
