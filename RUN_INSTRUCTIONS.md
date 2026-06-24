# Spotify Recommendation Analytics - Run Instructions

## 1. Open the project folder

```powershell
cd "D:\spotify-recommendation-analytics"
```

## 2. Create a virtual environment

Create the environment once:

```powershell
py -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks `Activate.ps1`, use Command Prompt activation:

```powershell
cmd /c ".venv\Scripts\activate.bat && py spotify_week7_analysis.py --root . --skip-sql"
```

You can also run the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe spotify_week7_analysis.py --root . --skip-sql
```

## 3. Install dependencies

After activating the environment:

```powershell
pip install -r requirements.txt
```

Without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Run the project

Complete pipeline with optional SQLite reference outputs:

```powershell
py spotify_week7_analysis.py --root .
```

Main Python analysis without SQLite:

```powershell
py spotify_week7_analysis.py --root . --skip-sql
```

Outputs are saved under:

```text
week7_outputs/
```

## 5. Important report files

Tables:

```text
week7_outputs/tables/regression_metrics.csv
week7_outputs/tables/regression_coefficients.csv
week7_outputs/tables/decade_feature_summary.csv
week7_outputs/tables/genre_decade_popularity_pivot.csv
week7_outputs/tables/correlation_matrix.csv
```

Figures:

```text
week7_outputs/figures/popularity_distribution.png
week7_outputs/figures/popularity_by_decade_boxplot.png
week7_outputs/figures/regression_actual_vs_predicted.png
week7_outputs/figures/regression_residuals.png
week7_outputs/figures/regression_coefficients.png
```

Run metadata:

```text
week7_outputs/run_summary.json
```

## 6. Common issues

### `ModuleNotFoundError`

The command is using a Python installation without the project packages.
Activate `.venv`, install `requirements.txt`, or use the `.venv` interpreter
directly.

### PowerShell blocks `Activate.ps1`

Run:

```powershell
cmd /c ".venv\Scripts\activate.bat && py spotify_week7_analysis.py --root . --skip-sql"
```

### Input file not found

Confirm that `cleaned_data/data_clean.csv` exists. The additional cleaned CSV
files provide artist, genre, year, and pivot outputs.

### SQL outputs are not needed

Use `--skip-sql`. SQLite is an optional reference; the main SQL work is
completed separately in SQL Server Management Studio.
