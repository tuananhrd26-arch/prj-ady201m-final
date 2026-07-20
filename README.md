# Spotify Recommendation Analytics

Spotify Recommendation Analytics is a reproducible Python project for exploring
cleaned Spotify track data, comparing popularity-regression experiments, and
building and querying a persisted content-based music recommender.

## Datasets

The pipeline reads the following files from `cleaned_data/`:

- `data_clean.csv` — track-level data required by every run.
- `data_by_artist_clean.csv` — artist-level aggregates.
- `data_by_genres_clean.csv` — genre-level aggregates.
- `data_by_year_clean.csv` — year-level aggregates.
- `data_w_genres_clean.csv` — artist-to-genre data used by the genre pivot.
- `cleaning_report.json` and `feature_selection_report.json` — cleaning metadata.

Pipeline code never rewrites these inputs.

## Architecture

```text
spotify-recommendation-analytics/
|-- cleaned_data/                  # Read-only cleaned inputs
|-- src/
|   |-- config.py                  # Stable constants and ProjectPaths
|   |-- data_loader.py             # Dataset discovery and loading
|   |-- preprocessing.py           # Track preparation
|   |-- validation.py              # Shared validation contracts
|   |-- eda.py                     # Pure EDA calculations
|   |-- visualization.py           # Static and interactive figures
|   |-- regression.py              # Regression experiments and artifact
|   |-- recommender.py              # Recommender training and persistence
|   |-- recommender_consumer.py     # Read-only artifact loading and queries
|   |-- sql_analysis.py             # Optional SQLite reference analysis
|   `-- pipeline.py                 # Reusable end-to-end orchestration
|-- scripts/
|   `-- recommend_song.py           # Read-only recommender CLI
|-- tests/                          # Characterization and acceptance suites
|-- week7_outputs/                  # Canonical tables, figures, and artifacts
|-- requirements.txt                # Runtime dependencies
|-- spotify_week7_analysis.py       # Public pipeline CLI and compatibility API
|-- RUN_INSTRUCTIONS.md
`-- FINAL_ACCEPTANCE.md
```

`src.pipeline.run_pipeline` is the single reusable orchestration path. It
creates paths, loads data, produces EDA outputs, runs regression, builds the
recommender, optionally runs SQLite reference queries, writes the run summary,
and returns an acceptance-friendly result dictionary.

`spotify_week7_analysis.py` remains the public command and backward-compatible
import entry point. Analysis implementations remain in their focused modules.

## Fresh-clone setup

Create a local virtual environment and install the runtime and test
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
```

`pytest` is installed separately because it is a development/test dependency,
not a runtime requirement. The canonical model artifacts are versioned under
`week7_outputs/model_artifacts/`, so no training run is required before using
the read-only consumer or running the accepted tests.

## Run the accepted pipeline

Activate the repository environment, then run the accepted non-SQL pipeline:

```powershell
python spotify_week7_analysis.py --root . --output week7_outputs --skip-sql
```

For a preservation audit, use an absolute output path outside the repository:

```powershell
python spotify_week7_analysis.py --root . --output D:\spotify-acceptance-output --skip-sql
```

The final Python acceptance uses `--skip-sql`. See the SQL status below.

## Output structure

A current non-SQL run creates 31 files:

```text
<output>/
|-- tables/             # 15 CSV files: 10 EDA, 3 regression, 2 recommender
|-- figures/            # 9 PNG files and 1 Plotly HTML file
|-- model_artifacts/    # 5 persisted regression/recommender artifacts
|-- sql/                # Empty when --skip-sql is used
`-- run_summary.json
```

Important tables include:

- `regression_metrics.csv`
- `regression_coefficients.csv`
- `regression_actual_vs_predicted.csv`
- `recommendation_demo_results.csv`
- `recommendation_validation.csv`
- `genre_decade_popularity_pivot.csv`
- `correlation_matrix.csv`

## Regression experiments

| Feature set | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Audio Only | Linear Regression | 13.1118 | 16.3080 | 0.4442 |
| Audio Only | Ridge Regression | 13.1119 | 16.3080 | 0.4442 |
| Extended | Linear Regression | 7.9820 | 10.7308 | 0.7594 |
| Extended | Ridge Regression | 7.9821 | 10.7309 | 0.7594 |

The accepted best model is Extended Linear Regression. Its persisted
scikit-learn `Pipeline` contains `scaler` and `model` steps and uses the ordered
14-feature Extended contract.

## Recommender and persisted artifacts

The recommender uses nine ordered features:

1. `acousticness`
2. `danceability`
3. `energy`
4. `instrumentalness`
5. `liveness`
6. `loudness`
7. `speechiness`
8. `tempo`
9. `valence`

The model-artifact directory contains:

- `best_popularity_model.joblib`
- `recommender_scaler.joblib`
- `nearest_neighbors_recommender.joblib`
- `recommender_features.json`
- `recommender_catalog.csv`

The catalog contains 170,653 rows and preserves the fitted model index, track
identity, metadata, and feature order.

## Read-only recommendation consumer

`scripts.recommend_song` loads the persisted catalog, scaler, and neighbor model
and validates their alignment. It never calls `fit` or rewrites an artifact.

Prefer a unique track ID or model index:

```powershell
python -m scripts.recommend_song --root . --model-index 19611 --top-n 10
python -m scripts.recommend_song --root . --track-id "47EiUVwUp4C9fGccaPuUCS" --top-n 10
```

Exact name-and-artists selection is also supported:

```powershell
python -m scripts.recommend_song --root . --name "Dakiti" --artists "['Bad Bunny', 'Jhay Cortez']" --top-n 10
```

This selector can be ambiguous when the catalog contains duplicate identity
pairs. ID or model index is preferred. Add `--output recommendations.csv` only
when a CSV is wanted; without `--output`, the consumer is read-only.

## Tests

```powershell
python -m pytest tests/test_pipeline.py -q
python -m pytest -q
```

The accepted suite contains 235 passing tests. Joblib currently emits an
upstream NumPy 2.5 deprecation warning while loading persisted arrays; the
warning does not affect artifact validation or predictions and is intentionally
not suppressed.

## Reproducibility

- Feature order and artifact filenames are centralized in `src/config.py`.
- Random state is fixed where model training requires it.
- Dataframe inputs are copied before transformations that could mutate them.
- Acceptance output should be generated outside the repository when comparing
  against the canonical `week7_outputs/` snapshot.
- PNG, Plotly, and Joblib files are compared semantically rather than by binary
  hash; deterministic CSVs and model predictions are compared strictly.

## SQL status

`src/sql_analysis.py` provides an optional, tested SQLite reference analysis
containing the existing six-query contract. It does not contain a multi-table
JOIN query, and no JOIN was added during final acceptance.

Final report SQL exercises will be executed separately by the project team in
SQL Server. The final Python acceptance pipeline therefore uses `--skip-sql`.
The absence of canonical SQLite CSV files under `week7_outputs/sql/` is
intentional for this acceptance.
