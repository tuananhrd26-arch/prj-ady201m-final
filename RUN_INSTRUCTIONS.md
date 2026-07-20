# Spotify Recommendation Analytics — Run Instructions

These commands target Windows PowerShell and the authoritative repository:

```text
D:\spotify-recommendation-analytics-cũ
```

## 1. Fresh-clone setup and environment activation

For a fresh clone, create a local environment and install runtime and test
dependencies:

```powershell
cd "<REPOSITORY_CLONE_PATH>"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
```

`pytest` is installed separately because it is required by the accepted test
suite but is not a runtime dependency. The accepted local workspace can be
activated directly with:

```powershell
cd "D:\spotify-recommendation-analytics-cũ"
D:\spotify-recommendation-analytics-cũ\.venv\Scripts\Activate.ps1
```

If `.venv` already exists, skip the environment-creation command. All commands
below use the activated environment's `python` command.

## 2. Run the accepted non-SQL pipeline

```powershell
python spotify_week7_analysis.py --root . --output week7_outputs --skip-sql
```

Final acceptance used `--skip-sql`. The team's final report SQL work is
performed separately in SQL Server.

To audit without touching the canonical snapshot, use an external absolute
output path:

```powershell
python spotify_week7_analysis.py --root . --output D:\spotify-acceptance-output --skip-sql
```

## 3. Optional SQLite reference analysis

The optional six-query SQLite reference module can be run by omitting
`--skip-sql`:

```powershell
python spotify_week7_analysis.py --root . --output D:\spotify-sql-reference-output
```

Use a separate output folder unless you deliberately want to create SQLite
reference files. SQLite output is not the final SQL Server deliverable, and the
current SQLite query set does not include a multi-table JOIN.

## 4. Query the persisted recommender

Model index, preferred:

```powershell
python -m scripts.recommend_song --root . --model-index 19611 --top-n 10
```

Exact track ID, preferred:

```powershell
python -m scripts.recommend_song --root . --track-id "<TRACK_ID>" --top-n 10
```

Exact name and raw artists value:

```powershell
python -m scripts.recommend_song `
    --root . `
    --name "<EXACT_NAME>" `
    --artists "<EXACT_RAW_ARTISTS>" `
    --top-n 10
```

Name-and-artists lookup can be ambiguous when duplicate catalog identity pairs
exist. Use track ID or model index when possible.

The consumer loads and validates existing artifacts; it never refits the
scaler or nearest-neighbor model. All five canonical model artifacts are
versioned under `week7_outputs/model_artifacts/`, so a fresh clone can query the
consumer without first running the training pipeline.

## 5. Write recommendation results to CSV

```powershell
python -m scripts.recommend_song `
    --root . `
    --model-index 19611 `
    --top-n 10 `
    --output .\recommendations.csv
```

Without `--output`, the consumer prints results and creates no file.

## 6. Help commands

```powershell
python spotify_week7_analysis.py --help
python -m scripts.recommend_song --help
python scripts\recommend_song.py --help
```

## 7. Tests

Pipeline-only acceptance:

```powershell
python -m pytest tests\test_pipeline.py -q
```

Complete discovered suite:

```powershell
python -m pytest -q
```

The accepted suite has 235 passing tests. Existing Joblib loads may display a
NumPy 2.5 deprecation warning; this upstream warning is known and does not
indicate a failed test or artifact mismatch.

## 8. Common issues

### `ModuleNotFoundError`

Confirm the repository environment is activated, or invoke the interpreter
directly:

```powershell
.\.venv\Scripts\python.exe spotify_week7_analysis.py --help
```

### PowerShell blocks activation

Use the interpreter directly instead of changing execution policy:

```powershell
.\.venv\Scripts\python.exe spotify_week7_analysis.py --root . --output week7_outputs --skip-sql
```

### Matplotlib cache warning

For automated acceptance commands, point Matplotlib at a writable directory:

```powershell
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = "D:\spotify-acceptance-output\matplotlib-config"
```

### Unicode output under redirected Windows consoles

If the repository path contains non-ASCII characters and output is redirected,
enable UTF-8 mode:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```
