# Power BI Dashboard Setup Guide
## Crime Data Analysis System

> [!IMPORTANT]
> Import `data/crime_data_clean.csv` into Power BI Desktop for all visualizations.

---

## 1. Connecting the Data Source

1. Open **Power BI Desktop** → **Get Data** → **Text/CSV**
2. Select `data/crime_data_clean.csv`
3. In **Power Query Editor** set types:
   - `date` → Date, `time` → Time
   - `latitude`, `longitude`, `severity` → Decimal Number
   - `arrest_made` → True/False
   - All other text fields → Text

---

## 2. DAX Measures

```dax
-- Total Incidents
Total Incidents = COUNT(crime_data_clean[incident_id])

-- Arrest Rate %
Arrest Rate % = 
DIVIDE(
    COUNTROWS(FILTER(crime_data_clean, crime_data_clean[arrest_made] = TRUE())),
    [Total Incidents]
) * 100

-- Avg Severity
Avg Severity = AVERAGE(crime_data_clean[severity])

-- Open Cases (no closure date)
Open Cases = 
COUNTROWS(
    FILTER(crime_data_clean, ISBLANK(crime_data_clean[closure_date]))
)

-- Case Closure Rate %
Closure Rate % = 
DIVIDE([Total Incidents] - [Open Cases], [Total Incidents]) * 100

-- High Severity Incidents (severity >= 4)
High Severity Incidents = 
COUNTROWS(FILTER(crime_data_clean, crime_data_clean[severity] >= 4))

-- Incidents with Weapon
Armed Incidents = 
COUNTROWS(
    FILTER(crime_data_clean, crime_data_clean[weapon_used] <> "Unknown")
)

-- YoY Change %
YoY Change % = 
VAR this_year = CALCULATE([Total Incidents], YEAR(crime_data_clean[date]) = MAX(YEAR(crime_data_clean[date])))
VAR last_year = CALCULATE([Total Incidents], YEAR(crime_data_clean[date]) = MAX(YEAR(crime_data_clean[date])) - 1)
RETURN DIVIDE(this_year - last_year, last_year) * 100

-- Recommended Patrol Units (per district)
Patrol Units = 
INT(DIVIDE([Total Incidents], 100)) + 1

-- Night Crime %
Night Crime % = 
DIVIDE(
    COUNTROWS(FILTER(crime_data_clean, crime_data_clean[is_night] = 1)),
    [Total Incidents]
) * 100
```

---

## 3. Dashboard Pages

### Page 1 — Executive Summary
| Visual | Configuration |
|--------|--------------|
| KPI Card | Total Incidents |
| KPI Card | Arrest Rate % |
| KPI Card | Open Cases |
| KPI Card | Avg Severity |
| Line Chart | Date (Month) → Incidents count with YoY overlay |
| Bar Chart | District → Total Incidents (sorted descending) |
| Donut Chart | Crime Type breakdown |

### Page 2 — Crime Patterns 🕐
| Visual | Configuration |
|--------|--------------|
| Matrix Heatmap | Hour (0–23) × Day of Week → Incident count |
| Bar | Crime Type → Incident count + Arrest Rate % dual axis |
| Stacked Bar | Crime Type → Arrested vs Not Arrested |
| Bar | Weapon Type → Usage count (filter out "Unknown") |
| Slicer | Year, Crime Type, District |

### Page 3 — Hotspot Map 🗺️
| Visual | Configuration |
|--------|--------------|
| ArcGIS / Bubble Map | Latitude + Longitude, bubble size = severity |
| Table | District, Incidents, Avg Severity, Arrest Rate — sorted by risk |
| Bar | District → Recommended Patrol Units |
| Filter Card | Date Range, Crime Type, Severity ≥ threshold |

> [!TIP]
> Enable **ArcGIS Maps** or use **Azure Maps** visual for geographic heat maps. Set bubble color to `severity` with a Red-Yellow-Green scale.

### Page 4 — Resource Allocation 🚔
| Visual | Configuration |
|--------|--------------|
| Bar | District → Patrol Units (DAX measure) |
| Table | District, Total Incidents, High Severity, Open Cases, Patrol Units |
| KPI Trend | District patrol allocation vs benchmark |
| Gauge | Avg Closure Rate % for open cases |

---

## 4. Useful Power Query Steps

```m
// Extract hour from time column
= Table.AddColumn(Source, "Hour", each Time.Hour([time]), Int64.Type)

// Flag night crime (10pm–6am)
= Table.AddColumn(Source, "IsNight", 
    each if [Hour] >= 22 or [Hour] < 6 then 1 else 0, Int64.Type)

// Bin severity into labels
= Table.AddColumn(Source, "SeverityLabel",
    each if [severity] >= 5 then "Critical"
         else if [severity] >= 4 then "High"
         else if [severity] >= 3 then "Medium"
         else "Low", type text)

// Calculate days to closure
= Table.AddColumn(Source, "DaysToClose",
    each if [closure_date] = null then null
         else Duration.Days([closure_date] - [date]), Int64.Type)
```

---

## 5. Conditional Formatting Rules

| Column | Rule |
|--------|------|
| `severity` | 1–2 = Green, 3 = Yellow, 4–5 = Red |
| `arrest_made` | TRUE = ✅, FALSE = ❌ |
| District Incidents | Top 3 = Red background (high risk) |
| `weapon_used` ≠ Unknown | Bold text |
