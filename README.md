# Spotify Recommendation Analytics

Spotify Recommendation Analytics is a reproducible Python project for exploring
cleaned Spotify track data, comparing popularity regression models, and
demonstrating content-based music recommendations.

The pipeline includes:

- exploratory data analysis with Pandas and NumPy;
- grouped, pivot, and correlation analysis;
- Linear and Ridge regression using Audio Only and Extended feature sets;
- a nearest-neighbor recommendation demonstration;
- report-ready tables and visualizations;
- optional SQLite reference outputs.

## Project structure

```text
spotify-recommendation-analytics/
|-- cleaned_data/
|   |-- data_clean.csv
|   |-- data_by_artist_clean.csv
|   |-- data_by_genres_clean.csv
|   |-- data_by_year_clean.csv
|   |-- data_w_genres_clean.csv
|   |-- cleaning_report.json
|   `-- feature_selection_report.json
|-- week7_outputs/
|   |-- figures/
|   |-- tables/
|   `-- run_summary.json
|-- spotify_week7_analysis.py
|-- requirements.txt
|-- RUN_INSTRUCTIONS.md
`-- REPORT_WEEK7_DRAFT.md
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the analysis

Run the complete pipeline, including optional SQLite reference outputs:

```powershell
py spotify_week7_analysis.py --root .
```

Run the main Python analysis without SQLite:

```powershell
py spotify_week7_analysis.py --root . --skip-sql
```

Generated outputs are written to:

```text
week7_outputs/
```

## Important outputs

- `week7_outputs/tables/regression_metrics.csv`
- `week7_outputs/tables/regression_coefficients.csv`
- `week7_outputs/tables/decade_feature_summary.csv`
- `week7_outputs/tables/genre_decade_popularity_pivot.csv`
- `week7_outputs/figures/popularity_distribution.png`
- `week7_outputs/figures/popularity_by_decade_boxplot.png`
- `week7_outputs/figures/regression_actual_vs_predicted.png`
- `week7_outputs/figures/regression_residuals.png`
- `week7_outputs/figures/regression_coefficients.png`
- `week7_outputs/run_summary.json`

## Regression results

| Feature set | Model | R² |
|---|---|---:|
| Audio Only | Linear Regression | 0.4442 |
| Audio Only | Ridge Regression | 0.4442 |
| Extended | Linear Regression | 0.7594 |
| Extended | Ridge Regression | 0.7594 |

The Extended feature set substantially improves popularity prediction compared
with the Audio Only feature set.

## SQL note

The group's main SQL analysis is completed separately in SQL Server Management
Studio. SQLite and SQL files produced by the Python script are optional
reference outputs and are excluded from Git by default.
