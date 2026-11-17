# 🔍 Crime Data Analysis System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive, end-to-end crime data analysis system that cleans, explores, visualizes, and models urban crime patterns from 40,000+ incident records across 8 city districts. The system identifies high-risk zones, uncovers temporal patterns, and generates data-driven patrol resource allocation recommendations.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard](#-power-bi-dashboard)
- [Results & Insights](#-results--insights)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Data Generation** | Synthetic 40k+ row crime dataset with realistic spatial, temporal, and categorical correlations |
| **Data Cleaning** | Automated pipeline: missing value imputation, datetime parsing, coordinate validation, deduplication |
| **Exploratory Analysis** | Crime type distributions, temporal heatmaps, district comparisons, weapon usage patterns |
| **Hotspot Detection** | Grid-based binning + DBSCAN spatial clustering to identify high-crime zones |
| **Resource Allocation** | Composite scoring model per district to suggest optimal patrol unit distribution |
| **Interactive Dashboard** | Power BI dashboard with 4 pages: Executive Summary, Crime Patterns, Hotspot Map, Resource Allocation |
| **Automated Reporting** | Console-printed summary of top risks, arrest rates, and closure time statistics |

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | ≥ 2.0.0 | Data manipulation |
| NumPy | ≥ 1.24.0 | Numerical computing |
| Matplotlib | ≥ 3.7.0 | Static visualizations |
| Seaborn | ≥ 0.12.0 | Statistical plots |
| Scikit-learn | ≥ 1.3.0 | DBSCAN clustering |
| Plotly | ≥ 5.15.0 | Interactive charts |
| GeoPandas | ≥ 0.14.0 | Geospatial analysis |
| Folium | ≥ 0.14.0 | Leaflet.js maps |
| Jupyter | ≥ 1.0.0 | Interactive notebooks |
| Power BI | Desktop | Business intelligence dashboard |
| Pytest | ≥ 7.4.0 | Unit testing |

---

## 📁 Directory Structure

```
crime-data-analysis/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── crime_data.csv            # Raw generated dataset (40k+ rows)
│   └── crime_data_clean.csv      # Cleaned dataset (output of pipeline)
│
├── src/
│   ├── data_cleaner.py           # DataCleaner class
│   ├── analyzer.py               # CrimeAnalyzer class
│   ├── visualizer.py             # CrimeVisualizer class
│   ├── hotspot_detector.py       # HotspotDetector class
│   └── main.py                   # End-to-end runner
│
├── notebooks/
│   ├── 01_data_cleaning.py       # Data cleaning walkthrough
│   ├── 02_eda_visualization.py   # EDA and visualization
│   ├── 03_hotspot_analysis.py    # Hotspot & clustering analysis
│   └── plots/                    # Saved notebook figures
│
├── dashboard/
│   └── README.md                 # Power BI setup instructions
│
├── outputs/                      # Generated plots from src/visualizer.py
│
└── tests/
    ├── test_data_cleaner.py      # Unit tests for DataCleaner
    └── test_analyzer.py          # Unit tests for CrimeAnalyzer
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prajwalxkulakrni/crime-data-analysis.git
cd crime-data-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python data/generate_data.py
```

This creates `data/crime_data.csv` with 40,000+ synthetic records.

---

## 🚀 Usage

### Run the full pipeline

```bash
python src/main.py
```

This will:
1. Load and clean the raw crime data
2. Run all analyses (temporal, district, weapon, closure time)
3. Generate and save all visualizations to `outputs/`
4. Detect spatial hotspots via DBSCAN
5. Print a summary report to the console
6. Save the cleaned dataset to `data/crime_data_clean.csv`

### Run individual scripts

```bash
# Data cleaning walkthrough
python notebooks/01_data_cleaning.py

# Full EDA
python notebooks/02_eda_visualization.py

# Hotspot & clustering analysis
python notebooks/03_hotspot_analysis.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## 📊 Power BI Dashboard

The dashboard connects to `data/crime_data_clean.csv` and provides four interactive pages:

### Page 1 — Executive Summary
- **KPI Cards**: Total Incidents, Arrest Rate %, Average Severity, Open Cases
- **Trend Line**: Monthly incident count over 2019–2023
- **District Map**: Color-coded crime density by district

### Page 2 — Crime Patterns
- **Temporal Heatmap**: Hour-of-day × Day-of-week crime frequency matrix
- **Crime Type Breakdown**: Donut chart of incident proportions
- **Year-over-Year Comparison**: Multi-line chart by crime type

### Page 3 — Hotspot Map
- **Bubble Map**: Geographic scatter using lat/lon; bubble size = severity, color = crime type
- **Top Hotspot Table**: District, peak hour, dominant crime type, risk level

### Page 4 — Resource Allocation
- **Bar Chart**: Recommended patrol units per district
- **Allocation Score Table**: District score, rank, and suggested staffing level
- **Trend Slicer**: Filter by year and crime type

---

## 📈 Results & Insights

Based on analysis of 40,000+ incident records (2019–2023):

| Insight | Finding |
|---|---|
| **Highest crime districts** | D3 and D5 account for ~35% of all incidents |
| **Peak crime hours** | 10 PM – 2 AM (Night window) accounts for 38% of crimes |
| **Most common crime** | Theft (~22%), followed by Assault (~17%) |
| **Lowest arrest rate** | Fraud (12%) and Arson (14%) |
| **Highest arrest rate** | Drug Offense (58%) and Robbery (45%) |
| **Avg closure time** | Homicide: 42 days | Theft: 8 days |
| **Weapon involvement** | 60% of Assaults and 72% of Robberies involve weapons |
| **Weekend spike** | Saturday sees 18% more incidents than the weekly average |
| **DBSCAN clusters** | 6 major spatial crime clusters identified |
| **Top patrol allocation** | D3 recommended 12 units, D5 recommended 10 units |

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

Tests cover:
- Missing value reporting format
- Weapon column NaN imputation
- Datetime feature extraction
- Coordinate validation
- Crime distribution percentage sums
- District hotspot sorting
- Arrest rate range validation

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**prajwalxkulakrni**  
GitHub: [@prajwalxkulakrni](https://github.com/prajwalxkulakrni)
