# Final Acceptance Report

## Repository state

- Repository: `D:\spotify-recommendation-analytics-cũ`
- Branch: `final-code-refactor`
- Starting HEAD: `8f03b2ff43344a179d3ec31d765696376075fb9f`
- Final HEAD: `<populated by the final documentation commit>`
- Final target status: clean working tree on `final-code-refactor`

No merge or push is part of this acceptance.

## Final architecture

- `src/pipeline.py` owns reusable orchestration through `run_pipeline`.
- `spotify_week7_analysis.py` remains the public CLI and compatibility entry
  point.
- Data loading, preprocessing, validation, EDA, visualization, regression,
  recommender, consumer, and optional SQLite logic remain in focused modules.
- The recommendation consumer is read-only and never fits a model.

Pipeline order is: paths, data loading, EDA, regression, recommender, optional
SQLite reference output, run summary, completion reporting.

## Automated tests

| Run | Result | Duration | Warnings |
|---|---:|---:|---:|
| Pipeline-only | 16 passed | 10.92s | 0 |
| Architecture suite | 235 passed | 104.45s | 48 |
| Complete discovered suite | 235 passed | 109.62s | 48 |

The 48 warnings are the existing Joblib/NumPy 2.5 array-shape deprecation
warning and were not suppressed.

## External acceptance execution

Accepted command:

```powershell
D:\spotify-recommendation-analytics-cũ\.venv\Scripts\python.exe `
    D:\spotify-recommendation-analytics-cũ\spotify_week7_analysis.py `
    --root D:\spotify-recommendation-analytics-cũ `
    --output D:\spotify-final-acceptance-audit-20260716-utf8\week7_outputs `
    --skip-sql
```

- Audit root: `D:\spotify-final-acceptance-audit-20260716-utf8`
- Wall-clock duration: 49.46s
- Exit code: 0
- Standard error: empty
- SQL files: 0

The first redirected-console attempt stopped before data loading because the
Windows CP-1252 stream could not encode `ũ`. The accepted retry used
`PYTHONUTF8=1` and a new sibling audit directory; no application change or
canonical write was required.

## Audit manifest

- Tables: 15 CSV files — 10 EDA, 3 regression, 2 recommender.
- Figures: 10 files — 6 EDA PNG, 1 Plotly HTML, 1 Plotly preview PNG, and 3
  regression PNG.
- Model artifacts: 5.
- Root metadata: `run_summary.json`.
- Total output files: 31.
- SQL directory: present and empty.

Every output was nonempty. Every CSV loaded, JSON parsed, Joblib artifact
loaded, PNG signature validated, and Plotly HTML contained HTML and Plotly
markers.

## EDA and regression acceptance

All generated EDA CSVs matched the canonical snapshot in filename, schema,
row order, categorical identity, and strict numeric values.

| Feature set | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Audio Only | Linear Regression | 13.1118 | 16.3080 | 0.4442 |
| Audio Only | Ridge Regression | 13.1119 | 16.3080 | 0.4442 |
| Extended | Linear Regression | 7.9820 | 10.7308 | 0.7594 |
| Extended | Ridge Regression | 7.9821 | 10.7309 | 0.7594 |

Best model: Extended Linear Regression, R² 0.7594. The generated regression
artifact is a scikit-learn `Pipeline` with `scaler` and `model` steps, a
`LinearRegression` estimator, and the ordered 14-feature Extended contract.
Predictions on a fixed canonical sample matched the canonical artifact at
`rtol=0`, `atol=1e-12`.

## Recommender and consumer acceptance

- Catalog shape: 170,653 rows × 15 columns.
- Ordered recommender features: 9.
- Catalog dataframe fingerprint:
  `8f87b530ffa60f3de5b6fd45a933963600cc83c30da59ecbc4c26169eb20b6b2`.
- StandardScaler parameters and `mean_`, `scale_`, `var_` matched canonically.
- NearestNeighbors used cosine distance with fitted shape `(170653, 9)` and
  matched the canonical fitted matrix at `rtol=0`, `atol=1e-12`.

Canonical seeds:

| Model index | Track ID | Name | Artists |
|---:|---|---|---|
| 19611 | `47EiUVwUp4C9fGccaPuUCS` | Dakiti | `['Bad Bunny', 'Jhay Cortez']` |
| 19606 | `3tjFYV6RSFtuktYl3ZtYcq` | Mood (feat. iann dior) | `['24kGoldn', 'iann dior']` |
| 19618 | `0t1kP63rueHleOhQkYSXFY` | Dynamite | `['BTS']` |
| 19616 | `0VjIjW4GlUZAMYd2vXMi3b` | Blinding Lights | `['The Weeknd']` |
| 19608 | `4Oun2ylbjFKMPTiaSbbCih` | WAP (feat. Megan Thee Stallion) | `['Cardi B', 'Megan Thee Stallion']` |

For all five seeds, read-only Top-10 consumer results matched the audit demo in
model indexes, ranks, IDs, names, artists, order, distances, and similarities
at the CSV's persisted precision. Fitting methods were forbidden during this
check. The CLI smoke command exited 0 without `--output` and changed no file.

## SQL acceptance scope

Final acceptance used `--skip-sql`. `src/sql_analysis.py` remains an optional,
tested six-query SQLite reference module. No SQLite JOIN query was added, and
the empty canonical SQL directory is intentional. Final report SQL exercises
will be completed separately by the team in SQL Server.

## Preservation and historical differences

- All 7 `cleaned_data/` files retained their pre-task recursive SHA-256
  manifest.
- All 31 canonical `week7_outputs/` files retained their pre-task recursive
  SHA-256 manifest.
- No canonical output, model artifact, or database was modified or regenerated.
- No persistent database was added.
- No full pipeline ran inside canonical `week7_outputs/`.

Expected historical snapshot differences were confirmed:

- Current `recommendation_validation.csv` adds `exact_top_n`; it is true for
  all five seeds, while every shared column matches canonically.
- Current `run_summary.json` adds `catalog_file` and `catalog_rows`, and the
  artifact manifest adds `recommender_catalog.csv`; shared semantic fields
  match.
- Historical absolute paths and runtime timestamps naturally differ from the
  external audit location.

## Verdict

**PASS.** The final architecture, automated tests, external non-SQL pipeline,
artifact equivalence, read-only consumer, and repository-preservation checks
all satisfy the acceptance scope. SQL Server report queries remain a separate
team deliverable.
