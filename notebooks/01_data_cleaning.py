"""
Notebook-style EDA script for Crime Data Analysis.
Run:  python notebooks/01_data_cleaning.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_cleaner import DataCleaner

RAW_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Crime Data — Cleaning & EDA  (Notebook 01)")
print("=" * 60)

# 1. Load
cleaner = DataCleaner(RAW_PATH)
raw = cleaner.load_data()
print(f"\nRaw shape: {raw.shape}")
print(raw.dtypes.to_string())

# 2. Missing-value report
print("\n--- Missing Values ---")
missing = cleaner.report_missing()
print(missing[missing["missing_count"] > 0].to_string(index=False))

# 3. Run all cleaning
clean = cleaner.get_clean_data()
print(f"\nClean shape: {clean.shape}")

# 4. Before/After comparison
print("\n--- Before vs After ---")
print(f"  Rows:    {raw.shape[0]:,} → {clean.shape[0]:,}")
print(f"  weapon_used nulls: {raw['weapon_used'].isnull().sum()} → {clean['weapon_used'].isnull().sum()}")

# 5. Missing-value bar chart
fig, ax = plt.subplots(figsize=(10, 4))
m = missing[missing["missing_count"] > 0]
ax.barh(m["column"], m["missing_pct"], color="tomato")
ax.set_xlabel("Missing %")
ax.set_title("Missing Values per Column (Before Cleaning)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_missing_values.png"), dpi=140)
plt.close(fig)
print(f"\nPlot saved → {OUT_DIR}/01_missing_values.png")
print("Done.")
