# Spotify Recommendation Analytics

This student data-analysis project explores cleaned Spotify track data,
compares popularity-regression models, and serves a persisted content-based
track recommender through Streamlit. The repository deliberately keeps one
analysis notebook and one application entry point so the complete workflow is
easy to inspect and demonstrate.

## Project structure

```text
.
|-- .streamlit/
|   `-- config.toml
|-- data/
|   |-- data_clean.csv
|   |-- data_by_artist_clean.csv
|   |-- data_by_genres_clean.csv
|   |-- data_by_year_clean.csv
|   `-- data_w_genres_clean.csv
|-- models/
|   |-- best_popularity_model.joblib
|   |-- nearest_neighbors_recommender.joblib
|   |-- recommender_catalog.csv
|   |-- recommender_features.json
|   `-- recommender_scaler.joblib
|-- python/
|   `-- spotify_analysis.ipynb
|-- sql/
|   `-- README.md
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- streamlit_app.py
```

## Datasets

The five accepted files under `data/` are cleaned, read-only inputs:

- `data_clean.csv` contains 170,653 track-level rows used for EDA, regression,
  and recommendation.
- `data_by_artist_clean.csv` contains artist-level audio summaries.
- `data_by_genres_clean.csv` contains genre-level profiles.
- `data_by_year_clean.csv` contains annual audio summaries.
- `data_w_genres_clean.csv` connects artist summaries with cleaned genres.

The notebook reads these files without changing their encoding, rows, columns,
or values. Any preparation happens only in memory.

## Analysis workflow

Open `python/spotify_analysis.ipynb` to follow the project from configuration
through dataset loading, validation, descriptive statistics, EDA, year and
decade trends, correlations, regression comparison, recommender construction,
validation, and final findings. Generated charts and tables are written to the
ignored `generated_outputs/` folder.

## Regression experiments

All experiments predict `popularity` with `test_size=0.2` and
`random_state=42`. Ridge Regression uses `alpha=10.0`.

| Feature set | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Audio Only | Linear Regression | 13.1118 | 16.3080 | 0.4442 |
| Audio Only | Ridge Regression | 13.1119 | 16.3080 | 0.4442 |
| Extended | Linear Regression | 7.9820 | 10.7308 | 0.7594 |
| Extended | Ridge Regression | 7.9821 | 10.7309 | 0.7594 |

The accepted best model is **Extended Linear Regression**.

## Recommendation approach

The content-based recommender represents every track with nine ordered audio
features: acousticness, danceability, energy, instrumentalness, liveness,
loudness, speechiness, tempo, and valence. A persisted `StandardScaler`
standardizes the rows, and a cosine `NearestNeighbors` model finds similar
profiles. The aligned catalog preserves a contiguous `_model_index` for all
170,653 fitted rows. Results exclude the seed track and repeated track/artist
pairs and return the exact requested Top N. Popularity is shown as context but
is not used as a recommendation feature.

## Streamlit application

`streamlit_app.py` is self-contained and loads the accepted artifacts without
training. It supports:

- track and artist search;
- exact track selection by aligned catalog index;
- 3 to 15 recommendations;
- cosine similarity, popularity, and year display;
- an exact result table; and
- selected-track versus recommendation-average audio profiles.

The app calls only persisted-model transformation and neighbor-query methods.
It never calls `fit`, retrains a model, or writes an artifact.

## Installation from a fresh clone

Python 3.12 is the accepted version. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No removed `src/` package is required.

## Run the notebook

```powershell
jupyter notebook python\spotify_analysis.ipynb
```

Run the cells in order. By default, the notebook loads accepted artifacts and
creates only ignored charts and tables.

## Run Streamlit

```powershell
python -m streamlit run streamlit_app.py
```

Open the local address printed by Streamlit, normally
`http://localhost:8501`.

## Model artifacts and rebuild safety

The `models/` directory is the accepted immutable snapshot:

- `best_popularity_model.joblib` is the Extended Linear Regression pipeline.
- `recommender_scaler.joblib` is the fitted nine-feature scaler.
- `nearest_neighbors_recommender.joblib` is the fitted cosine neighbor model.
- `recommender_features.json` preserves feature order.
- `recommender_catalog.csv` preserves fitted-row identity and alignment.

At the top of the notebook, `REBUILD_MODELS = False` is the safe default. In
this mode, accepted artifacts are loaded and nothing under `models/` is
overwritten. Setting it to `True` trains the four regression experiments and
the recommender, but writes new artifacts only under
`generated_outputs/model_artifacts/`. Writing to `models/` additionally
requires manually setting the separate `OVERWRITE_ACCEPTED_MODELS = True`
confirmation. Use a temporary copy when testing rebuild mode.

## SQL Server scope

The final SQL report is performed separately in Microsoft SQL Server. The
group will add SQL scripts and screenshots under `sql/`. The Python notebook
does not claim that the SQL Server portion is complete.

## Troubleshooting

- If a module is missing, activate `.venv` and reinstall with
  `python -m pip install -r requirements.txt`.
- Run notebook and Streamlit commands from the repository root so relative
  `data/` and `models/` paths resolve correctly.
- If Streamlit reports a missing artifact, confirm that all five documented
  files are present under `models/`; do not retrain just to launch the app.
- If PowerShell blocks activation, use
  `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.
- Matplotlib and notebook outputs belong under ignored `generated_outputs/`;
  deleting that folder does not remove accepted inputs or models.
