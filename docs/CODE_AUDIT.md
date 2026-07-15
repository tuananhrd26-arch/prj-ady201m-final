# Baseline Code Audit

Audit date: 2026-07-15 (Asia/Saigon)  
Scope: inspection and documentation only; the complete analysis pipeline was not run.

## 1. Environment and Git baseline

- Absolute working directory: `D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics`
- Repository-root check: **confirmed by required project contents**. The working directory contains `spotify_week7_analysis.py`, `requirements.txt`, `README.md`, `RUN_INSTRUCTIONS.md`, `cleaned_data/`, and `week7_outputs/`.
- Git repository: **no**. `git rev-parse --is-inside-work-tree` and all other Git queries returned `fatal: not a git repository`.
- Current branch: unavailable because `.git` metadata is absent.
- Current commit: unavailable because `.git` metadata is absent.
- Git status: unavailable because `.git` metadata is absent.
- Branch action: `final-code-refactor` was **not** created. A clean working tree could not be established, so automatic branch creation was not permitted.
- Python launcher: `C:\Windows\py.exe`
- Python interpreter: `C:\Program Files\Python312\python.exe`
- Python version: 3.12.6
- Syntax check: `py -m compileall spotify_week7_analysis.py` succeeded.
- CLI help check: failed before argument parsing at line 39 because `joblib` is not installed.
- Pipeline execution: deliberately not attempted because it overwrites files under `week7_outputs/`.

### Dependency/import baseline

| Import | Status | Version |
|---|---|---|
| pandas | missing | not installed |
| numpy | missing | not installed |
| sklearn / scikit-learn | missing | not installed |
| matplotlib | missing | not installed |
| seaborn | missing | not installed |
| plotly | missing | not installed |
| joblib | missing | not installed |

`requirements.txt` declares pandas, NumPy, matplotlib, seaborn, scikit-learn, and joblib. It does not declare Plotly, even though an interactive Plotly output is part of the target workflow. No dependency was installed during this audit.

## 2. Current folder tree

Baseline tree before creating this audit documentation:

```text
spotify-recommendation-analytics/
|-- .gitignore (446 bytes)
|-- README.md (2,686 bytes)
|-- REPORT_WEEK7_DRAFT.md (14,057 bytes)
|-- RUN_INSTRUCTIONS.md (2,475 bytes)
|-- requirements.txt (52 bytes)
|-- spotify_week7_analysis.py (31,188 bytes)
|-- cleaned_data/
|   |-- cleaning_report.json (2,974 bytes)
|   |-- data_by_artist_clean.csv (4,178,175 bytes)
|   |-- data_by_genres_clean.csv (617,396 bytes)
|   |-- data_by_year_clean.csv (21,105 bytes)
|   |-- data_clean.csv (30,654,264 bytes)
|   |-- data_w_genres_clean.csv (6,018,084 bytes)
|   `-- feature_selection_report.json (14,013 bytes)
`-- week7_outputs/
    |-- run_summary.json (2,508 bytes)
    |-- figures/
    |   |-- audio_feature_trends_by_year.png (196,823 bytes)
    |   |-- correlation_heatmap.png (185,166 bytes)
    |   |-- popularity_by_decade_boxplot.png (45,367 bytes)
    |   |-- popularity_distribution.png (40,677 bytes)
    |   |-- regression_actual_vs_predicted.png (220,457 bytes)
    |   |-- regression_coefficients.png (66,514 bytes)
    |   |-- regression_residuals.png (235,151 bytes)
    |   `-- tracks_by_decade.png (49,800 bytes)
    `-- tables/
        |-- audio_feature_trends_by_year.csv (3,297 bytes)
        |-- correlation_matrix.csv (927 bytes)
        |-- dataset_overview.csv (131 bytes)
        |-- decade_feature_summary.csv (599 bytes)
        |-- descriptive_statistics.csv (1,347 bytes)
        |-- genre_decade_popularity_pivot.csv (2,593 bytes)
        |-- missing_values_after_cleaning.csv (379 bytes)
        |-- recommendation_demo_results.csv (6,833 bytes)
        |-- regression_actual_vs_predicted.csv (359,541 bytes)
        |-- regression_coefficients.csv (2,520 bytes)
        |-- regression_metrics.csv (230 bytes)
        |-- top_genres_audio_profile.csv (1,142 bytes)
        `-- tracks_by_decade.csv (229 bytes)
```

There was no baseline `docs/`, `week7_outputs/model_artifacts/`, `week7_outputs/sql/`, HTML file, joblib file, or Git metadata directory.

During final verification, the managed workspace exposed empty `.git/` and `.agents/` directory placeholders, both timestamped 2026-07-15 22:36:02. Each contains zero children. They were absent from the initial top-level listing, contain no files, and do not change the Git result: `git status` and `git rev-parse` still report that this is not a repository. The current top level therefore consists of the baseline entries above, the new `docs/` directory, and those two empty environment placeholders.

## 3. Dataset inventory

The runtime packages needed to call `pandas.read_csv` are absent. The CSV audit therefore scanned every record with Python's standard-library CSV parser and inferred the default pandas-style dtypes from the full contents. The main dataset's inferred dtypes are independently corroborated by `cleaned_data/feature_selection_report.json`. Memory values are approximate deep payload estimates, not live pandas measurements.

### `data_clean.csv`

- Size: 30,654,264 bytes
- Shape: 170,653 rows x 20 columns
- Columns, inferred dtype, and missing count:

| Column | Dtype | Missing |
|---|---:|---:|
| valence | float64 | 0 |
| year | int64 | 0 |
| acousticness | float64 | 0 |
| artists | object/string | 0 |
| danceability | float64 | 0 |
| duration_ms | int64 | 0 |
| energy | float64 | 0 |
| explicit | int64 | 0 |
| id | object/string | 0 |
| instrumentalness | float64 | 0 |
| key | float64 | 0 |
| liveness | float64 | 0 |
| loudness | float64 | 0 |
| mode | float64 | 0 |
| name | object/string | 0 |
| popularity | int64 | 0 |
| release_date | object/string | 0 |
| speechiness | float64 | 0 |
| tempo | float64 | 0 |
| release_date_parsed | object/string | 119,798 |

- Duplicate full rows: 0
- Duplicate `id` values: 0
- Duplicate `name + artists` rows: 12,968
- Approximate in-memory payload: 42,506,406 bytes (40.54 MiB)

### `data_by_artist_clean.csv`

- Size: 4,178,175 bytes
- Shape: 28,680 rows x 15 columns
- Exact column order: `mode`, `count`, `acousticness`, `artists`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, `key`
- Inferred dtypes: `count` int64; `artists` object/string; every other column float64.
- Missing values: 0 in every column.
- Duplicate full rows: 0
- Duplicate `id`: not applicable; no `id` column.
- Duplicate `name + artists`: not applicable; no `name` column.
- Approximate in-memory payload: 4,070,062 bytes (3.88 MiB)

### `data_by_genres_clean.csv`

- Size: 617,396 bytes
- Shape: 2,973 rows x 15 columns
- Exact column order: `mode`, `genres`, `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, `key`, `genres_clean`
- Inferred dtypes: `genres` and `genres_clean` object/string; every other column float64.
- Missing values: `genres_clean` has 1; every other column has 0.
- Duplicate full rows: 0
- Duplicate `id`: not applicable.
- Duplicate `name + artists`: not applicable.
- Approximate in-memory payload: 461,380 bytes (0.44 MiB)

### `data_by_year_clean.csv`

- Size: 21,105 bytes
- Shape: 100 rows x 14 columns
- Exact column order: `mode`, `year`, `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, `key`
- Inferred dtypes: `year` int64; every other column float64.
- Missing values: 0 in every column.
- Duplicate full rows: 0
- Duplicate `id`: not applicable.
- Duplicate `name + artists`: not applicable.
- Approximate in-memory payload: 12,000 bytes (0.01 MiB)

### `data_w_genres_clean.csv`

- Size: 6,018,084 bytes
- Shape: 28,680 rows x 17 columns
- Exact column order: `genres`, `artists`, `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`, `popularity`, `key`, `mode`, `count`, `genres_clean`
- Inferred dtypes: `count` int64; `genres`, `artists`, and `genres_clean` object/string; every other column float64.
- Missing values: `genres_clean` has 9,857; every other column has 0.
- Duplicate full rows: 0
- Duplicate `id`: not applicable.
- Duplicate `name + artists`: not applicable; no `name` column.
- Approximate in-memory payload: 6,332,515 bytes (6.04 MiB)

### Main-track domain validation

All invalid counts are zero.

| Field | Required domain | Observed range/values | Invalid rows |
|---|---|---|---:|
| popularity | 0..100 | 0..100 | 0 |
| acousticness | 0..1 | 0..0.996 | 0 |
| danceability | 0..1 | 0..0.988 | 0 |
| energy | 0..1 | 0..1 | 0 |
| instrumentalness | 0..1 | 0..1 | 0 |
| liveness | 0..1 | 0..1 | 0 |
| speechiness | 0..1 | 0..0.97 | 0 |
| valence | 0..1 | 0..1 | 0 |
| key | 0..11 | 0..11 | 0 |
| explicit | only 0 or 1 | 0, 1 | 0 |
| mode | only 0 or 1 | 0.0, 1.0 | 0 |
| year | finite integer year (1000..2026 audit rule) | 1921..2020 | 0 |
| tempo | finite numeric | 0..243.507 | 0 |
| loudness | finite numeric | -60..3.855 | 0 |
| duration_ms | finite numeric | 5,108..5,403,500 | 0 |

No CSV was modified or rewritten.

## 4. Current output inventory

- Analytical tables: **13 CSV files** under `week7_outputs/tables/`.
- Static figures: **8 PNG files** under `week7_outputs/figures/`.
- Interactive outputs: **0** HTML files anywhere under `week7_outputs/`.
- Model artifacts: **0 actual files**. The `week7_outputs/model_artifacts/` directory is absent.
- SQL reference outputs: **0 actual files**. The `week7_outputs/sql/` directory is absent.

All existing output filenames and sizes are listed in the folder tree in section 2. Folders were not counted as files.

### `run_summary.json`

The file exists and records:

- run start `2026-06-24T15:16:39+07:00`, completion `2026-06-24T15:16:51+07:00`, and duration 12.16 seconds;
- absolute input paths rooted at `D:\spotify-recommendation-analytics`, which is not the current workspace path;
- 170,653 processed track rows and 21 processed columns (the script adds `decade` to the 20-column input);
- paths for tables, figures, SQL, and model artifact directories, also rooted at the old path;
- 13 table filenames and 8 figure filenames;
- four regression configurations trained;
- Extended Linear Regression as best by R2, R2 0.7594;
- Extended Ridge Regression as the diagnostic plot model;
- SQL reference execution set to false;
- a successful 30-genre pivot note and the SQL Server/SQLite note.

It does **not** record the command, Python/package versions, source/data hashes, split row IDs, complete model parameters, saved model filename, recommender artifact names, catalog mapping, seed identity validation, uniqueness checks, or output hashes. Its absolute paths are stale in the current copy, so it is not independently reproducible as current-workspace provenance.

## 5. Function-by-function code map

### Imports

- Standard library: `argparse`, `ast`, `json`, `sqlite3`, `dataclass`, `datetime`, `Path`, and typing names `Any`, `Dict`, `Iterable`, `List`, `Tuple`.
- Third party: `joblib`, `matplotlib.pyplot`, `numpy`, `pandas`, and scikit-learn components for cloning, linear models, metrics, splitting, nearest neighbors, pipelines, and standard scaling.
- `Iterable` is imported but not used.
- Seaborn and Plotly are not imported by the entry point.

### Constants and class

- Lines 52-53: `RANDOM_STATE = 42`, `TARGET = "popularity"`.
- Lines 55-65: nine Audio Only regression features.
- Lines 67-73: Extended features add `duration_ms`, `year`, `explicit`, `key`, and `mode`.
- Line 75: `REGRESSION_FEATURES` aliases the Extended list.
- Lines 77-87: nine recommender features.
- Line 89: four trend features.
- Lines 92-99: frozen `ProjectPaths` dataclass.
- Lines 692-773: mutable module-level `SQL_QUERIES` dictionary containing six queries. The code does not mutate it.

### Command-line arguments

Source inspection shows three options at lines 838-845:

- `--root PATH` (default `.`)
- `--output NAME` (default `week7_outputs`)
- `--skip-sql` (boolean flag)

The executable `--help` output cannot currently be reached because imports fail first.

### Functions in source order

| Lines | Function and parameters | Return | Reads | Writes / responsibility |
|---:|---|---|---|---|
| 102-114 | `make_paths(root, output_dir="week7_outputs")` | `ProjectPaths` | none | Creates output, tables, figures, SQL, and model directories. |
| 117-120 | `read_csv_if_exists(path)` | dataframe or `None` | supplied CSV when present | none |
| 123-151 | `load_project_data(root)` | `(dataframes, input_paths)` | required `cleaned_data/data_clean.csv` or fallback `data/data.csv`; four optional cleaned CSVs | none |
| 154-182 | `clean_tracks_for_analysis(df)` | reset-index dataframe copy | in-memory dataframe | none; numeric coercion, ID/name filtering, median imputation, `decade` derivation |
| 185-186 | `save_table(df, path)` | `None` | in-memory dataframe | supplied CSV, UTF-8 with BOM, overwritten if present |
| 189-201 | `plot_tracks_by_decade(tracks, fig_path)` | `None` | in-memory tracks | supplied PNG, overwritten if present |
| 204-212 | `plot_popularity_distribution(tracks, fig_path)` | `None` | in-memory tracks | supplied PNG |
| 215-233 | `plot_popularity_by_decade_boxplot(tracks, fig_path)` | `None` | in-memory tracks | supplied PNG |
| 236-250 | `plot_feature_trends(tracks, fig_path)` | yearly trend dataframe | in-memory tracks | supplied PNG |
| 253-271 | `plot_correlation_heatmap(tracks, features, fig_path)` | flattened correlation dataframe | in-memory tracks | supplied PNG |
| 274-288 | `parse_artist_names(value)` | list of artist strings | supplied value | none |
| 291-343 | `create_genre_decade_pivot(tracks, artist_genres, max_genres=30)` | `(pivot or None, note)` | two in-memory frames | none |
| 346-432 | `create_eda_outputs(data, paths)` | cleaned tracks | all loaded dataframes | 9 EDA CSVs and 5 PNGs; stores genre note in `tracks.attrs` |
| 435-619 | `regression_analysis(tracks, paths)` | regression summary dict | in-memory tracks | 3 CSVs, 3 PNGs, `ridge_popularity_model.joblib` |
| 622-668 | `build_recommender_demo(tracks, paths, n_examples=5, n_neighbors=10)` | `None` | in-memory tracks | 3 artifact files plus `recommendation_demo_results.csv` |
| 671-689 | `create_sqlite_database(data, paths)` | database path | loaded dataframes | deletes/replaces an existing `spotify_week7.sqlite`, then writes tables |
| 776-785 | `run_sql_outputs(db_path, paths)` | `None` | SQLite database and six global SQL queries | six SQL-result CSVs and `spotify_week7_queries.sql` |
| 788-833 | `write_run_summary(paths, input_files, tracks, regression_summary, sql_executed, run_started_at)` | `None` | current filesystem globs and in-memory summaries | overwrites `run_summary.json` |
| 836-878 | `main()` | `None` | CLI and all project inputs | orchestrates all outputs and prints progress |

`create_eda_outputs` writes `dataset_overview.csv`, `descriptive_statistics.csv`, `missing_values_after_cleaning.csv`, `tracks_by_decade.csv`, `decade_feature_summary.csv`, `audio_feature_trends_by_year.csv`, `correlation_matrix.csv`, optional `top_genres_audio_profile.csv`, and optional `genre_decade_popularity_pivot.csv`; it writes `tracks_by_decade.png`, `popularity_distribution.png`, `popularity_by_decade_boxplot.png`, `audio_feature_trends_by_year.png`, and `correlation_heatmap.png`.

### Main execution flow

```text
parse args -> resolve root -> create output directories -> load CSVs
-> clean tracks and create EDA outputs -> train/evaluate regression models
-> build recommender demo -> optionally recreate SQLite and SQL outputs
-> write run_summary.json
```

The full command therefore overwrites named CSV/PNG/JSON/joblib files. The SQL path additionally unlinks an existing database. This is why the pipeline was not executed during this audit.

### State, repeated logic, and responsibility boundaries

- There is no intentionally changing global runtime state. The global feature lists and SQL dictionary are mutable Python objects but are not mutated by current code.
- `tracks.attrs["genre_pivot_note"]` is in-memory metadata shared between EDA and run-summary code.
- Figure setup, layout, save, and close logic is repeated across eight plot blocks.
- Feature existence/drop-null logic is repeated in modeling paths.
- Output naming and direct writes are spread throughout analysis functions rather than centralized in a manifest.
- `create_eda_outputs` (87 lines) mixes cleaning, tabulation, orchestration, plotting, optional-data branching, and metadata creation.
- `regression_analysis` (185 lines) mixes experiment definition, splitting, fitting, metric calculation, coefficient extraction, model selection, persistence, charting, and summary creation.
- `build_recommender_demo` mixes catalog construction, training, persistence, seed selection, querying, presentation formatting, and saving.
- `main` is a reasonable orchestration size, but `make_paths` performs writes as a side effect and the public entry point cannot expose `--help` unless every heavy dependency imports successfully.

## 6. Regression audit

- Target: `popularity` (line 53).
- Audio Only features: `danceability`, `energy`, `acousticness`, `valence`, `tempo`, `loudness`, `instrumentalness`, `liveness`, `speechiness` (lines 55-65).
- Extended features: all Audio Only fields plus `duration_ms`, `year`, `explicit`, `key`, `mode` (lines 67-73).
- Per-feature-set model frame: rows missing target or any requested feature are dropped (line 462). The earlier cleaning step median-fills numeric missing values, so current main data supplies all model rows.
- Split: `train_test_split`, `test_size=0.2`, `random_state=42`, default `shuffle=True`, no stratification (lines 465-470). A split is recreated for each feature set.
- Pipelines: both estimators are cloned into `Pipeline(StandardScaler(), model)` (lines 472-479).
- Linear Regression: default `LinearRegression()` configuration (line 444).
- Ridge Regression: `Ridge(alpha=10.0, random_state=42)` with the default solver (line 445). The random state has no practical effect unless the chosen solver uses randomness.
- MAE: `mean_absolute_error(y_test, y_pred)` (line 482).
- RMSE: square root of `mean_squared_error(y_test, y_pred)` (line 483).
- R-squared: `r2_score(y_test, y_pred)` (line 484).
- Metric-best rule: maximum unrounded R2 across all four fitted results (line 603). Ties resolve to the first fitted result because Python `max` retains the first maximum.
- Diagnostic/saved selection rule: prefer Extended Ridge unconditionally when it exists; otherwise use max R2 (lines 534-544).
- Existing best model in `run_summary.json`: Extended Linear Regression, R2 0.7594.
- Existing diagnostic model: Extended Ridge Regression.
- Actually saved by source: the selected diagnostic pipeline, normally Extended Ridge Regression, under `ridge_popularity_model.joblib` (lines 552-555).
- Existing artifact: absent from this repository copy.

Existing metrics:

| Feature set | Model | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Audio Only | Linear Regression | 13.1118 | 16.3080 | 0.4442 |
| Audio Only | Ridge Regression | 13.1119 | 16.3080 | 0.4442 |
| Extended | Linear Regression | 7.9820 | 10.7308 | 0.7594 |
| Extended | Ridge Regression | 7.9821 | 10.7309 | 0.7594 |

The rounded R2 values tie, but Linear has slightly better MAE and RMSE in the saved metrics and was selected as best by the unrounded R2. The code intentionally plots and saves Extended Ridge instead. `run_summary.json` names the best and plot models separately but does not name the saved artifact/model. Consequently, the metric-best model, plotted model, saved model, and most prominent summary label are not one consistent model identity.

## 7. Recommender audit

### Implementation trace

- Features: `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence` (lines 77-87 and 623).
- Catalog: `tracks.dropna(subset=["name", "artists"] + features).copy().reset_index(drop=True)` (line 628).
- Catalog index: reset to a contiguous positional index at line 628.
- Scaling: a new `StandardScaler` is fit and transforms the full catalog feature matrix at lines 632-633.
- Neighbor model: `NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")`, fit on the scaled full catalog at lines 635-636.
- Seed selection: sort the catalog by descending popularity, drop duplicate `name + artists`, take `n_examples`, then call `reset_index(drop=True)` (lines 642-647).
- Seed index: `input_row.name` from the reset demo dataframe (line 651).
- Seed vector: `X[input_idx]` (line 652).
- Similarity: `1 - cosine_distance` (line 664).
- Exclusion: blindly discards the first returned neighbor with `[1:]` (line 653); it does not compare `neighbor_idx` with the true seed catalog index.
- Duplicate recommendation exclusion: none.
- Ranking: `enumerate(..., start=1)` after slicing (line 653).
- Results: one combined `recommendation_demo_results.csv` (line 668).
- Saved recommender artifacts: `recommender_scaler.joblib`, `nearest_neighbors_recommender.joblib`, and `recommender_features.json` (lines 638-640).
- Catalog mapping: not saved. No artifact maps nearest-neighbor row positions back to `id`, `name`, `artists`, or other metadata.

### Confirmed seed/vector mismatch

The potential index bug is present.

`model_df` and `X` share full-catalog positions after line 628. `demo_inputs` is independently sorted and then reset at lines 642-647. Therefore `input_row.name` at line 651 is only the demo row number, not the original model row number. Line 652 uses it as a model-matrix row.

The first five actual matrix rows are:

| X row | Actual vector song | Popularity |
|---:|---|---:|
| 0 | Piano Concerto No. 3 in D Minor, Op. 30: III. Finale. Alla breve | 4 |
| 1 | Clancy Lowered the Boom | 5 |
| 2 | Gati Bali | 5 |
| 3 | Danny Boy | 3 |
| 4 | When Irish Eyes Are Smiling | 2 |

The existing output labels the five query groups as `Dakiti`, `Mood (feat. iann dior)`, `Dynamite`, `Blinding Lights`, and `WAP (feat. Megan Thee Stallion)`. Thus the displayed seed and vector seed differ for all five demo groups. The output's recommendations and similarities describe matrix rows 0-4 while being labeled as those five popular songs.

### Guarantee assessment

| Requirement | Current guarantee | Evidence |
|---|---|---|
| Displayed seed equals vector seed | **No; confirmed false** | reset/sort index mismatch at lines 642-652 |
| Seed never recommended to itself | **No guarantee** | code drops the first result, not the result whose index equals the seed; equal-distance ties can reorder |
| Duplicate `name + artist` recommendations removed | **No** | no deduplication; current `Dakiti` group repeats two pairs twice |
| Exactly Top N unique results | **No** | current `Dakiti` group has 10 rows but only 8 unique song/artist pairs |
| Similarities finite | **No explicit guarantee** | code does not validate finite scaled vectors/distances; current output happens to contain 50 finite values, range 0.9447..1.0 |
| Reloaded model rows map to metadata | **No** | model catalog/mapping is not saved |

The current output has ranks 1 through 10 and 10 rows for each of five displayed seeds. It contains no displayed-seed self-recommendations, but that does not validate actual-vector self-exclusion because the displayed and actual seeds differ.

## 8. Verified bugs and risks

### Confirmed

1. **Recommender seed identity corruption:** line 651 obtains an index from a reset, sorted demo frame and line 652 applies it to the original catalog matrix.
2. **Non-unique recommendation results:** no output-level deduplication exists; one current seed returns only 8 unique pairs out of 10 rows.
3. **Model/catalog cannot be reloaded as a functional recommender:** scaler and nearest-neighbor objects are specified for persistence, but catalog row metadata is not.
4. **Regression identity inconsistency:** best-by-metric is Extended Linear, diagnostics and persistence prefer Extended Ridge, and the run summary does not identify the saved model.
5. **Expected interactive visualization is absent:** no HTML output, Plotly source code, Plotly import, or Plotly requirement exists.
6. **Expected output counts are not met:** 13 tables rather than 15 and 8 visuals rather than 10.
7. **Actual model artifacts are absent:** the source names four files, but none exists in this workspace. `.gitignore` excludes joblib files and `week7_outputs/model_artifacts/`, making artifact omission from a copied repository likely.
8. **Current command cannot start in this environment:** all checked third-party packages are missing; `--help` fails on the first third-party import.
9. **Run provenance is stale:** `run_summary.json` points to a different absolute project root and lacks artifact/source/data versioning.
10. **Output preservation risk:** normal pipeline execution overwrites known outputs; the SQL path explicitly deletes an existing SQLite file before rebuilding it.

### Risks requiring a provisioned runtime to settle

- Reproduce metrics and plots from the current source/data/package combination; existing files may have been generated elsewhere.
- Confirm the exact installed-pandas dtype representation for every auxiliary CSV. This audit's dtypes are full-file inferred values; only the main dataset has an existing pandas report for corroboration.
- Test cosine behavior for zero standardized vectors and equal-distance duplicate vectors.
- Determine whether scikit-learn's neighbor tie ordering ever leaves the actual catalog seed in results after blind first-row removal.
- Confirm source-to-artifact compatibility once the missing artifacts are regenerated in a protected output directory.

## 9. Code, report, and output inconsistencies

| Expected item | Classification | Verified state |
|---|---|---|
| 15 analytical tables | **PRESENT BUT INCOMPLETE** | 13 table CSVs exist and the `--skip-sql` Python path creates at most those 13 named table outputs. |
| 10 visual outputs including interactive Plotly | **PRESENT BUT INCOMPLETE** | 8 static PNGs exist. |
| Interactive Plotly output | **MISSING** | no HTML, dependency, import, or generation code. |
| Four or more model artifacts | **INCONSISTENT** | source writes four named files, report lists them, but zero artifacts exist. |
| Multiple-seed recommendation validation | **PRESENT BUT INCOMPLETE** | 5 demo labels and 50 rows exist, but there is no validation artifact and seed identity is wrong. |
| Reproducible `run_summary.json` | **PRESENT BUT INCOMPLETE** | file exists but has stale absolute paths and incomplete provenance/artifact metadata. |
| `py spotify_week7_analysis.py --root . --skip-sql` | **CANNOT VERIFY YET** | source accepts the options and compiles, but execution stops at missing `joblib`. |
| Regression comparison | **PRESENT AND VERIFIED (static evidence)** | source defines four experiments and matching metric rows exist; fresh runtime reproduction is pending. |
| Best/plot/saved model consistency | **INCONSISTENT** | Linear is best; Ridge is plotted/saved; summary omits saved identity. |
| Recommender catalog alignment | **INCONSISTENT** | output label and vector positions are demonstrably different. |

Additional documentation/output findings:

- `README.md` claims generated outputs under `week7_outputs/` but does not disclose that model artifacts are ignored/absent. Its `R²` header is visibly mojibaked as `RÂ²` in the current file.
- `REPORT_WEEK7_DRAFT.md` lists six SQL CSVs, a SQL script, and four model artifacts as files to use, but none is present in this workspace.
- The report interprets the recommendation table as audio similarity for its displayed inputs. Because seed identity is wrong, those specific interpretations are not valid.
- `run_summary.json` says SQL was skipped, which explains absent SQL outputs for that recorded run, but the report still names them without a presence qualification.
- `requirements.txt` includes seaborn even though the current entry point does not use it; it omits Plotly even though Plotly is a target output.

## 10. Questions and blockers requiring human input

1. Restore or identify the intended `.git` repository before implementation so a clean baseline, `final-code-refactor` branch, and exact Git diff can exist.
2. Confirm the authoritative package/environment strategy and approve installing only declared/needed dependencies when implementation begins.
3. Define the exact two missing analytical tables and exact two missing visuals, including the required Plotly deliverable, rather than deriving them from output-count targets alone.
4. Decide whether the best regression model must control plots and persistence, or whether Ridge is deliberately retained; in either case the summary and artifact naming must agree.
5. Confirm the desired persistent recommender catalog format (CSV, Parquet, or joblib metadata frame) and stable identifier (`id` is the strongest candidate).
6. Decide whether SQL reference outputs must be generated and versioned even when the required final command uses `--skip-sql`.
7. Approve an output-protection approach for later tests, such as a temporary output directory, before any end-to-end execution.

No refactoring or implementation-module files were created in this audit.
