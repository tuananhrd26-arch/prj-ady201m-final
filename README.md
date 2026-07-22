# Spotify Recommendation Analytics

Spotify Recommendation Analytics is a reproducible Python 3.12 project for
exploring cleaned Spotify data, comparing popularity-regression models, and
serving a persisted content-based track recommender through Streamlit.

The public repository is intentionally small: it contains the accepted cleaned
inputs, the Python pipeline, the focused Streamlit application, and the five
persisted model artifacts. Development tests, audit reports, duplicate run
instructions, and reproducible report outputs are not part of the release.

## Repository tree

```text
.
|-- .streamlit/
|   `-- config.toml
|-- cleaned_data/
|   |-- data_clean.csv
|   |-- data_by_artist_clean.csv
|   |-- data_by_genres_clean.csv
|   |-- data_by_year_clean.csv
|   `-- data_w_genres_clean.csv
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- eda.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- recommender.py
|   |-- recommender_consumer.py
|   |-- regression.py
|   |-- sql_analysis.py
|   |-- streamlit_support.py
|   |-- validation.py
|   `-- visualization.py
|-- week7_outputs/
|   `-- model_artifacts/
|       |-- best_popularity_model.joblib
|       |-- nearest_neighbors_recommender.joblib
|       |-- recommender_catalog.csv
|       |-- recommender_features.json
|       `-- recommender_scaler.joblib
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- spotify_week7_analysis.py
`-- streamlit_app.py
```

`src/__init__.py` is also retained so `src` remains an explicit Python
package.

## Fresh-clone installation

Python 3.12 is the accepted version. From a fresh clone:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `python3.12 -m venv .venv` and
`.venv/bin/python -m pip ...` instead.

## Launch the Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. The single focused page lets you:

- search by track or artist and select an exact catalog row;
- request 5 to 15 unique audio-feature neighbors;
- inspect rank, cosine similarity, year, and popularity;
- view the exact result table; and
- compare the seed profile with the recommendation average on a radar chart.

The app loads the persisted feature contract, catalog, scaler, and
`NearestNeighbors` model. Normal app use calls `transform` and `kneighbors`;
it never calls `fit`, retrains a model, or writes an artifact.

## Run the accepted analysis pipeline

Use a generated output folder rather than overwriting the accepted artifacts:

```powershell
.\.venv\Scripts\python.exe spotify_week7_analysis.py --root . --output demo_outputs --skip-sql
```

The non-SQL run loads the five retained cleaned CSV files and generates EDA
tables and figures, four regression experiments, a persisted best regression
model, recommender demo/validation outputs, recommender artifacts, and a JSON
run summary under `demo_outputs/`. That folder is ignored by Git and can be
deleted after inspection.

The five inputs are read-only:

- `data_clean.csv` is the required track-level dataset.
- `data_by_artist_clean.csv` supplies artist aggregates.
- `data_by_genres_clean.csv` supplies genre profiles.
- `data_by_year_clean.csv` supplies year aggregates.
- `data_w_genres_clean.csv` supplies artist-to-genre data for the genre pivot.

The pipeline never modifies these files. The auxiliary datasets are optional
in the loader, but they are retained so the complete accepted output set is
reproducible.

## Persisted artifacts and recommender behavior

`week7_outputs/model_artifacts/` is the only retained canonical output group:

- `best_popularity_model.joblib` is the accepted regression model.
- `recommender_scaler.joblib` is the fitted nine-feature `StandardScaler`.
- `nearest_neighbors_recommender.joblib` is the fitted cosine-distance model.
- `recommender_features.json` preserves the ordered feature contract.
- `recommender_catalog.csv` aligns 170,653 catalog rows with fitted model
  indexes and contains track identity, display metadata, and feature values.

The ordered recommender features are acousticness, danceability, energy,
instrumentalness, liveness, loudness, speechiness, tempo, and valence.
Similarity is `1 - cosine distance`. Popularity is display context only and is
not a recommender feature. Duplicate name-and-artist pairs and the seed track
are excluded from results.

## Accepted regression result

| Feature set | Model | MAE | RMSE | R-squared |
|---|---|---:|---:|---:|
| Audio Only | Linear Regression | 13.1118 | 16.3080 | 0.4442 |
| Audio Only | Ridge Regression | 13.1119 | 16.3080 | 0.4442 |
| Extended | Linear Regression | 7.9820 | 10.7308 | 0.7594 |
| Extended | Ridge Regression | 7.9821 | 10.7309 | 0.7594 |

The accepted final model is Extended Linear Regression. Its scikit-learn
pipeline contains a scaler followed by `LinearRegression` and uses the ordered
14-feature extended contract defined in `src/config.py`.

## SQL scope

The supported public acceptance command uses `--skip-sql`. The project team's
SQL Server report work remains a separate deliverable. `src/sql_analysis.py`
is retained because the public analysis module imports it at startup and
`src.pipeline` uses it when `--skip-sql` is omitted; it is an optional SQLite
reference implementation, not the SQL Server submission.

## Streamlit Community Cloud

Connect this GitHub repository in Streamlit Community Cloud and set the entry
point to `streamlit_app.py`. Select Python 3.12 in the deployment settings.
`requirements.txt` and `.streamlit/config.toml` are discovered automatically,
and the app requires no secrets. Keep all five files under
`week7_outputs/model_artifacts/` in the deployed revision.

## Troubleshooting

- `ModuleNotFoundError`: run commands from the repository root with the same
  interpreter used for `pip install -r requirements.txt`.
- Missing recommender artifacts: confirm the five artifact filenames above
  exist under `week7_outputs/model_artifacts/`; do not retrain them just to
  launch the app.
- Non-ASCII Windows path or redirected output errors: set
  `$env:PYTHONUTF8 = "1"` and `$env:PYTHONIOENCODING = "utf-8"` before running.
- Matplotlib cache warnings: set `$env:MPLCONFIGDIR` to a writable temporary
  directory when running the pipeline in a restricted environment.
- Plotly preview export errors: Kaleido is installed by the requirements, but
  current Kaleido releases may also require a compatible local Chrome/Chromium
  installation. The interactive HTML output is still generated if static
  preview export is unavailable.
