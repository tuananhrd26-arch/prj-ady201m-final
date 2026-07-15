# Workspace Reconciliation Report

Comparison date: 2026-07-15 (Asia/Saigon)  
Scope: read-only comparison and planning. No reconciliation operation was executed.

## 1. Executive conclusion

Use `D:\spotify-recommendation-analytics-cũ` as the final working repository after a non-destructive preservation pass.

The existing Git repository contains the most complete and most recent source code. Its working-tree entry point is a 1,049-line evolution of the 882-line audited entry point and fixes the confirmed seed/vector mismatch, explicit self-exclusion, duplicate recommendations, the regression best/plot/saved-model mismatch, and the missing Plotly output. It also produces the expected 15 tables and 10 visuals and has four actual model/recommender artifacts.

The current audited workspace remains important because it contains the only copies of `docs/CODE_AUDIT.md`, `docs/IMPLEMENTATION_PLAN.md`, and `docs/ENVIRONMENT_BASELINE.md`, plus the prepared `.venv`. Its source is not newer: it is byte-identical to `spotify_week7_analysis_before_report_fix.py` in the repository and identical to the repository's committed `HEAD` version.

The cleaned datasets have different raw hashes only because one copy uses LF and the other CRLF line endings. Every cleaned file becomes byte-identical after newline normalization, and all CSV row counts, column counts, headers, and values are the same. Neither dataset copy is analytically newer.

No files should be overwritten yet. First preserve the repository history, dirty diff, untracked report packages, both output sets, and current audit documents in a new external backup directory. Then reconcile on a temporary preservation branch. Create `final-code-refactor` only after the reconciled repository is clean and the preservation commits are complete.

## 2. Two-folder overview

### Current audited workspace

Absolute path:

```text
D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics
```

- Expected Spotify structure: present.
- Valid Git repository: no. The exposed `.git/` directory is empty.
- Top-level files: 6 files, 50,904 bytes.

| Major folder | File count | Total bytes |
|---|---:|---:|
| `.agents/` | 0 | 0 |
| `.git/` | 0 | 0 |
| `.venv/` | 17,013 | 464,033,206 |
| `__pycache__/` | 1 | 42,700 |
| `cleaned_data/` | 7 | 41,506,011 |
| `docs/` before this report | 3 | 52,960 |
| `week7_outputs/` | 22 | 1,422,231 |

Top-level entries are `.agents/`, `.git/`, `.gitignore`, `.venv/`, `__pycache__/`, `cleaned_data/`, `docs/`, `README.md`, `REPORT_WEEK7_DRAFT.md`, `requirements.txt`, `RUN_INSTRUCTIONS.md`, `spotify_week7_analysis.py`, and `week7_outputs/`.

### Existing Git repository

Absolute path:

```text
D:\spotify-recommendation-analytics-cũ
```

- Expected Spotify structure: present.
- Valid Git repository: yes.
- Top-level files: 9 files, 12,451,384 bytes.

| Major folder | File count | Total bytes |
|---|---:|---:|
| `.git/` | 66 | 21,084,711 |
| `cleaned_data/` | 7 | 41,737,761 |
| `report_package/` | 35 | 1,553,372 |
| `report_package_updated/` | 40 | 15,569,549 |
| `report_tables/` | 9 | 13,320 |
| `week7_outputs/` | 30 | 15,463,276 |

Top-level entries are `.git/`, `.gitignore`, `cleaned_data/`, `README.md`, `report_package/`, `report_package_updated/`, `report_tables/`, `REPORT_WEEK7_DRAFT.md`, `requirements.txt`, `RUN_INSTRUCTIONS.md`, `spotify_report_package.zip`, `spotify_report_package_updated.zip`, `spotify_week7_analysis.py`, `spotify_week7_analysis_before_report_fix.py`, and `week7_outputs/`.

## 3. Existing Git status

- Branch: `main`
- Commit: `0ce731edaa93c5741368b29ff100c9187a2ebac9`
- Remote: `origin https://github.com/tuananhrd26-arch/spotify-recommendation-analytics.git`
- Upstream: `main...origin/main`; no ahead/behind count is shown.
- Staged files: none; `git diff --cached` is empty.
- Deleted tracked files: none.
- Modified tracked files: 14.
- Untracked entries shown by ordinary status: 10 top-level/path entries.
- Four model artifacts also exist but are ignored by `week7_outputs/model_artifacts/` in `.gitignore`, so ordinary status does not show them.

Full `git status --short`:

```text
 M requirements.txt
 M spotify_week7_analysis.py
 M week7_outputs/figures/audio_feature_trends_by_year.png
 M week7_outputs/figures/correlation_heatmap.png
 M week7_outputs/figures/popularity_by_decade_boxplot.png
 M week7_outputs/figures/popularity_distribution.png
 M week7_outputs/figures/regression_actual_vs_predicted.png
 M week7_outputs/figures/regression_coefficients.png
 M week7_outputs/figures/regression_residuals.png
 M week7_outputs/figures/tracks_by_decade.png
 M week7_outputs/run_summary.json
 M week7_outputs/tables/descriptive_statistics.csv
 M week7_outputs/tables/recommendation_demo_results.csv
 M week7_outputs/tables/regression_actual_vs_predicted.csv
?? report_package/
?? report_package_updated/
?? report_tables/
?? spotify_report_package.zip
?? spotify_report_package_updated.zip
?? spotify_week7_analysis_before_report_fix.py
?? week7_outputs/figures/interactive_energy_loudness.html
?? week7_outputs/figures/interactive_energy_loudness_preview.png
?? week7_outputs/tables/decade_explicit_multiindex_summary.csv
?? week7_outputs/tables/recommendation_validation.csv
```

### Modified-file diff inspection

Each modified tracked file was inspected with `git diff -- <file>`; no cached diff exists.

| Modified tracked file | Diff evidence and meaning |
|---|---|
| `requirements.txt` | +2/-0: adds `plotly` and `kaleido`. |
| `spotify_week7_analysis.py` | Git numstat +193/-26; working tree adds 167 net lines and the completed Plotly/recommender/regression changes described below. |
| Eight existing PNGs | Binary diffs. All retain the same dimensions. `tracks_by_decade.png` is pixel-identical despite different encoded bytes; four non-regression EDA plots have only negligible pixel/rendering differences; the three regression plots materially differ because they now use Extended Linear rather than Extended Ridge. |
| `run_summary.json` | +35/-4: July 14 run metadata, 15 tables, 10 visuals, aligned Linear model identity, recommendation validation, interactive plot note, and four artifact names. |
| `descriptive_statistics.csv` | +7/-7: only tiny last-decimal floating-point representation changes. |
| `recommendation_demo_results.csv` | +51/-51: replaces mislabeled recommendations with correctly indexed, unique results and adds model indexes and cosine distance. |
| `regression_actual_vs_predicted.csv` | +33,374/-33,374: same 34,131-row shape, but predictions change from the previously forced Ridge diagnostics to the metric-best Linear model. |

## 4. Source-code comparison

### File identity

| Property | Current workspace | Existing repository |
|---|---|---|
| Size | 31,188 bytes | 38,872 bytes |
| SHA-256 | `2a508a72947088d32b86de09730e71726b1b12dcba7b7fe90aff0fb7497647b5` | `efcb0772505302d07649c5306c70c7f9ba2655f858548ebb5fdbaf69c5b24d89` |
| Lines | 882 | 1,049 |
| Content assessment | older audited baseline | newer completed Week 7 pipeline |

The current source, the repository's committed `HEAD:spotify_week7_analysis.py`, and the untracked `spotify_week7_analysis_before_report_fix.py` all have the same Git blob/hash content. The repository working tree is therefore a direct uncommitted upgrade, not an unrelated fork.

### Functions

Both copies contain these 19 functions in the same overall order:

`make_paths`, `read_csv_if_exists`, `load_project_data`, `clean_tracks_for_analysis`, `save_table`, `plot_tracks_by_decade`, `plot_popularity_distribution`, `plot_popularity_by_decade_boxplot`, `plot_feature_trends`, `plot_correlation_heatmap`, `parse_artist_names`, `create_genre_decade_pivot`, `create_eda_outputs`, `regression_analysis`, `build_recommender_demo`, `create_sqlite_database`, `run_sql_outputs`, `write_run_summary`, and `main`.

The repository additionally defines `plot_interactive_energy_loudness` at lines 275-318. `build_recommender_demo` grows from 47 lines returning `None` to 145 lines returning a validation summary dictionary. `create_eda_outputs`, `write_run_summary`, and `main` are expanded to incorporate the new outputs and metadata.

### Constants, feature lists, and CLI

- `RANDOM_STATE`, `TARGET`, all regression feature lists, the recommender features, trend features, and all six SQL queries are identical.
- Both CLIs expose exactly `--root`, `--output`, and `--skip-sql` with the same meanings.
- The repository adds `import plotly.express as px`; no existing import is removed.

### Regression changes

- Current workspace: calculates `best_result`, but deliberately selects Extended Ridge for plots and persistence; saves `ridge_popularity_model.joblib`.
- Existing repository: assigns `selected_result = best_result`, uses that same model for predictions, plots, coefficients, persistence, and summary; saves `best_popularity_model.joblib`.
- The July run summary confirms Extended Linear Regression is both best and plotted. The saved filename is model-neutral and matches that policy.
- Conclusion: the existing repository fixes the best-model versus plotted/saved-model inconsistency.

### Recommender changes

- Current workspace: resets the sorted seed dataframe, uses its `0..4` display index against the full matrix, drops the first neighbor without checking identity, does not deduplicate results, and saves no catalog mapping.
- Existing repository: stores `_model_index` before seed sorting, does not reset it, queries `X[input_idx]` using the preserved model position, explicitly checks displayed label against the vector row, excludes `neighbor_idx == input_idx`, excludes the seed key, deduplicates `name + artists`, searches the full candidate set until 10 unique results are found, records model indexes/distances, and writes `recommendation_validation.csv`.
- The existing validation file reports 10 results and all checks true for five seeds.
- The repository still saves only the scaler, nearest-neighbor model, and feature list. It does **not** save the row-aligned `model_df` catalog or an ID/metadata mapping. A reloaded arbitrary neighbor index still cannot be mapped to song metadata without independently reconstructing the exact catalog. The missing recommender catalog problem is not fixed.
- The validation logic records `recommendations_returned` but does not include equality to the requested Top N in `validation_passed`; the current data/run does return exactly 10 for every seed.

### Output filename changes

Repository-only source outputs:

- `decade_explicit_multiindex_summary.csv`
- `recommendation_validation.csv`
- `interactive_energy_loudness.html`
- `interactive_energy_loudness_preview.png`

Regression artifact rename:

- Old: `ridge_popularity_model.joblib`
- New: `best_popularity_model.joblib`

### Known-issue disposition

| Audited issue | Current workspace | Existing repository |
|---|---|---|
| Seed/vector index mismatch | present | fixed |
| Explicit self-recommendation exclusion | not guaranteed | fixed by model index and song key |
| Duplicate recommendations | present | fixed by `name + artists` set |
| Exact Top N unique | not guaranteed; one output has 8 unique | achieved for the current catalog/run; validation should still assert count |
| Missing recommender catalog artifact | present | still present |
| Best/plot/saved model inconsistency | present | fixed |
| Missing Plotly output | present | fixed; HTML and preview exist |

## 5. Requirements comparison

| Property | Current workspace | Existing repository |
|---|---|---|
| Exists | yes | yes |
| Size | 52 bytes | 75 bytes |
| SHA-256 | `bfb3ba79b4be86b8b6acc0864387bd79abad9adb78f10956c70d8e78ad2645b3` | `e679199c5eb5e8cace43f29c9d53eab39a9d5fabdfb7871ad2c33c7778865d4e` |
| Lines | 6 | 8 |

Both contain pandas, NumPy, matplotlib, seaborn, scikit-learn, and joblib. The repository additionally declares Plotly and Kaleido, which its newer source uses for HTML and preview generation. The repository requirements are content-newer and should be authoritative, subject to later version-pinning decisions. `pytest` is installed in the current `.venv` as a development dependency but is not declared by either file.

## 6. Documentation comparison

| File | Current exists/size/SHA-256 | Repository exists/size/SHA-256 | Result and content authority |
|---|---|---|---|
| `.gitignore` | yes; 446; `ec4d00daffa3c89128856060fbdb84f54b3c268fae416f2ae31d8ae9f0fa0044` | yes; 446; same hash | identical |
| `README.md` | yes; 2,686; `f445c1109534fdc8bb053482e334f68841153f35377209e81df063008bbf102e` | yes; 2,686; same hash | identical; neither describes the new outputs fully |
| `RUN_INSTRUCTIONS.md` | yes; 2,475; `a573677f4a39b34fe875efe91b378a1949bc5af4e357432bfd20d36291552e8a` | yes; 2,475; same hash | identical |
| `REPORT_WEEK7_DRAFT.md` | yes; 14,057; `05018921bbdaa7c120392fb051079f8d819d1afc19dc91836b7e70453e6913de` | yes; 14,057; same hash | identical |
| `docs/CODE_AUDIT.md` | yes; 29,446; `0568177f250623855ba18c3a18776fd5a856abdd1658949e81ac4ee22a6ddadd` | no | current-only historical audit |
| `docs/IMPLEMENTATION_PLAN.md` | yes; 16,187; `8057dd4eb537096fcb5db9510298520fa37a586610bfd5acffab4422acf0c910` | no | current-only plan |
| `docs/ENVIRONMENT_BASELINE.md` | yes; 7,327; `e3a74cafe3e46471f95a37043d59447e8edf9a0723ed01252dbd1662127848e8` | no | current-only environment record |

The three audit documents should eventually be copied into the repository after backup. `CODE_AUDIT.md` must remain labeled as a point-in-time audit of the older committed source; it should not be silently rewritten to imply the newer fixes were present at audit time.

The repository also has unique untracked documentation under `report_package/`, `report_package_updated/`, and `report_tables/`. Those files are complementary report-delivery evidence and must be preserved pending a decision about tracking versus external archival.

## 7. Cleaned-data comparison

Every raw hash differs, but every file is byte-identical after replacing CRLF with LF. The repository adds exactly one carriage-return byte per line; no cell or JSON value differs. CSV shapes and headers are identical.

| File | Current bytes / SHA-256 | Repository bytes / SHA-256 | Rows x columns | Result |
|---|---|---|---:|---|
| `cleaning_report.json` | 2,974 / `f3575ca2517f5fe23f77e9bdf3f23acb2b2a87bbea76067cb406eee781090888` | 3,104 / `3be3355c4eeec09ff367810000ae667a1a5b6e390ebd0d2613d96020c717f35b` | JSON | semantically identical; newline-only difference |
| `data_by_artist_clean.csv` | 4,178,175 / `bf07120baf4cc044dfa3382eb17eef2866acfb85028fe89b724e25258a7f70df` | 4,206,856 / `8aed3ba6638aeb99efc94e4cee7a5bfd7741a8b17c66dee21f674fe68281bb55` | 28,680 x 15 | identical cells; newline-only difference |
| `data_by_genres_clean.csv` | 617,396 / `cc77e459f9dd96896bef4086d5efb170c166e7ee81ad966102cc652ec1bb5a2b` | 620,370 / `43e0ae6348145b25e9bcd5444169029afc4e50dbb81c4e132f71e378f6a93142` | 2,973 x 15 | identical cells; newline-only difference |
| `data_by_year_clean.csv` | 21,105 / `5b3c85aefc4a3f91d6f0ee6e6b4898b95deb5c5b944a86eab858611d03cdd815` | 21,206 / `8e50fbc2b211153b55e4125133ba674a5d812833f6c7db7ba65eac961043fb56` | 100 x 14 | identical cells; newline-only difference |
| `data_clean.csv` | 30,654,264 / `bd5ac9e831c9d4e7310cd3cffc90ff6d6d3bd1d5cb6f029c8cc5bcb3288d1802` | 30,824,918 / `05ceae39a36221b30393b5c60353158349de7d94d9e8ad26aaa52e6e82422659` | 170,653 x 20 | identical cells; newline-only difference |
| `data_w_genres_clean.csv` | 6,018,084 / `36195e1b90ef30c318936f05fc14fd2349f58868511279ab33a41cff46b97781` | 6,046,765 / `4800bf0c8466bfba32eb52278946868274aceaaae1a02e04ce4d9e4c5e0775d9` | 28,680 x 17 | identical cells; newline-only difference |
| `feature_selection_report.json` | 14,013 / `3b93b325913d24cbe83899a3bf8fcbc4dae88ee4adc84043b4442364114b60b2` | 14,542 / `2b4ba5120942fdd2d64663fe70e5537d4f298d6676ed14027255b95b67cc12b0` | JSON | semantically identical; newline-only difference |

No cleaned-data file needs to be copied during reconciliation. Preserve the repository working-tree encoding to avoid a noisy all-data Git diff.

## 8. Output comparison

### Summary

| Measure | Current workspace | Existing repository |
|---|---:|---:|
| Tables | 13 | 15 |
| Static PNG visuals | 8 | 9 |
| Interactive HTML visuals | 0 | 1 |
| Total visuals | 8 | 10 |
| Model/recommender artifacts | 0 | 4 |
| Recommendation validation | no | yes |
| SQL outputs | 0 | 0 |
| Run summary date | 2026-06-24 | 2026-07-14 |

The repository output set is content-newer, not merely timestamp-newer. Its run summary names the added files, the newer source generates them, the validation table demonstrates the corrected seed/index behavior, and the regression output reflects the corrected selected model.

No output exists only in the current workspace by relative path. Eight output files exist only in the repository. Every shared file has a different raw hash because repository text files use CRLF and all figures were regenerated. Ten shared tables are content-identical after newline normalization. The genuinely changed shared files are all eight encoded figures, `run_summary.json`, `descriptive_statistics.csv` (round-off only), `recommendation_demo_results.csv` (corrected), and `regression_actual_vs_predicted.csv` (selected-model change).

### Tables manifest

| Relative path | Current size / SHA-256 | Repository size / SHA-256 | Comparison |
|---|---|---|---|
| `tables/audio_feature_trends_by_year.csv` | 3,297 / `3bbeeb76c14b967b002a8f7f1f4372b40eb53016c588e58964afeb56b410dcee` | 3,398 / `e6ffe174584753e70342895c015d9f2d505b115635177043855d3ca7867224b9` | content-identical; CRLF only |
| `tables/correlation_matrix.csv` | 927 / `cfc049e57596c9623f74f86a084ec6b9223433abc4e56abe92a0537bc952d121` | 938 / `24cc0ec73cb4a490a0dfe500c1f5bffb14d144052471715f92b11a2f5f29b714` | content-identical; CRLF only |
| `tables/dataset_overview.csv` | 131 / `9438933a161cb676991725d3cc2560aa56b3a3e00ff1e0af4ca6cd79639d192d` | 137 / `314768e32ee042ba4eec44d78aaef461aaffacf49503093fd210517a7ecd3c84` | content-identical; CRLF only |
| `tables/decade_explicit_multiindex_summary.csv` | absent | 1,017 / `8cdee7e58204397a389b61aa19d9fe91a752b6991cfd731b5ff60d634ea16705` | repository only |
| `tables/decade_feature_summary.csv` | 599 / `16d55297a5738ff99e686bd08cbcdf65066d56f0eda647595df7e8e59626e132` | 611 / `0160c2c44436a9c8174cc60a2dbc96c516ad1c47fc1f48570b59c556fb89e916` | content-identical; CRLF only |
| `tables/descriptive_statistics.csv` | 1,347 / `1704bdb6127fffaeb310c7d4c001da4566aa14f2eaaf19d83586c7ff54c6be41` | 1,360 / `bb9cf850092dc6c1b2d3d2b9506a47263f2d82916e5b205dec01cefdb2ed43d1` | tiny final-decimal differences |
| `tables/genre_decade_popularity_pivot.csv` | 2,593 / `2ff046eff279e7340486155efd8bbfc0ebeb9afc809f343e4599f2655281b0ba` | 2,624 / `dc5ef4b2ccb64d7838efceb2bb3077fb55a0c8378403303ea6f981854f5032ae` | content-identical; CRLF only |
| `tables/missing_values_after_cleaning.csv` | 379 / `ad5ff4f13961243ba8603a1506787ca9aeb9e7acce815b264e0b22d4a872e3d4` | 401 / `e4a7f4aa5080bf36df2d28eb70b029675b932d02d22b1b6809ec779165266db4` | content-identical; CRLF only |
| `tables/recommendation_demo_results.csv` | 6,833 / `d6a2a4aa997a15138249d4937ce1737b7261b1f7056f77d321c8ba8ac920fdad` | 6,099 / `1a0bc95d331878522239df6d8e0e1c59e08044704e552dab7fcac56e220a3cf3` | corrected content in repository |
| `tables/recommendation_validation.csv` | absent | 656 / `6558374005258e161edbd6393cd2f84296e2967d53d4e3cd70136589d4fce019` | repository only |
| `tables/regression_actual_vs_predicted.csv` | 359,541 / `66b30e5c513bb71dcb65d090faf41bf62faa988834bce53e479589872b369166` | 393,717 / `1216c5533d4898ea4b7107c157f05b70d5631d081bb3813fc1aa252f79203d45` | selected-model predictions differ |
| `tables/regression_coefficients.csv` | 2,520 / `d679c8ddf2ee6760c96643ca135f1456bbe7bf6b450762f08927696b46fb9b79` | 2,567 / `aff2ef715d32cda2d4ccdf11f5e3fc2a5668169669dd80f37b4d12ad952d7f08` | content-identical; CRLF only |
| `tables/regression_metrics.csv` | 230 / `d4fdb89775baeea0b38a7b94e0bc19232ed0ec814912f1f37f96249aab99e8e2` | 235 / `79e5a255a0a95ca4aa40b77532424478a97b9f1ce6cfceadd39e9a1b7a141bd1` | content-identical; CRLF only |
| `tables/top_genres_audio_profile.csv` | 1,142 / `822f2df72473d6ea5806d3778317cce119b0cf1e21db6fcf0b574c8154febb0f` | 1,163 / `6f5ef0df864ed74e4bbc13cc5c397d9d1ce12d65f8417070e86ab5c65bdea656` | content-identical; CRLF only |
| `tables/tracks_by_decade.csv` | 229 / `effd9ac4053614cc1b9cd498e9f8e85fa1fac99af4d7d92fb368bc20f21253aa` | 241 / `464f17a81ba45648f8020dcb24ac7e3a4f28a6d38feaf8f9e0f954272b5c1e09` | content-identical; CRLF only |

### Figures and interactive manifest

| Relative path | Current size / SHA-256 | Repository size / SHA-256 | Comparison |
|---|---|---|---|
| `figures/audio_feature_trends_by_year.png` | 196,823 / `6df7815bdd9d69cf4dfbda0b77b5a34c29d22136b32131c5dfd888a01e98004b` | 196,704 / `3fb5471a2d0714e219f75109e129263e0d59a9645c7142f931da5a10dd8c7f12` | same geometry; negligible render difference |
| `figures/correlation_heatmap.png` | 185,166 / `62e83d2e8feef55f3817e69bd7ff154d706898264ab909f7c38b027a166e4d75` | 185,458 / `64d2af46e8e608a2db3fa7b8205ffb41f56bd3b898e7cf4decb0c231d0dd464e` | same geometry; negligible render difference |
| `figures/interactive_energy_loudness.html` | absent | 986,830 / `c6fc40265cd985ce4b9090260d695772d383283a8e0c945f0540f96ba0165313` | repository-only Plotly output |
| `figures/interactive_energy_loudness_preview.png` | absent | 728,012 / `b59603508d9cd5eb67b290bf356d1746bda0442b82d84aba81afac821a16ab45` | repository-only preview |
| `figures/popularity_by_decade_boxplot.png` | 45,367 / `9039f3c664dfc569ae23537d7667b3b1345bc14d62fa74c8a31e0c2ea13fa6fd` | 45,426 / `b458fa6afdb982455dd2955456153b28e861b5e10a1e89a6e5cf72bb28128170` | same geometry; negligible render difference |
| `figures/popularity_distribution.png` | 40,677 / `d13f9f04a96fdc9dadb2b791c3b38c4f157e7db5ec5aa4035f798c04d38e08b1` | 40,647 / `1cf42d52a0ff7e436b04723703f22271102b4c1625f0f22ead61aeebc391c603` | same geometry; negligible render difference |
| `figures/regression_actual_vs_predicted.png` | 220,457 / `b5a94cd8685db870554616a4b609b66f5fd4b342783f85a58ee3953ac0e093da` | 219,687 / `0aefe2fb196e3525734dacbaf5eea85d459596b42d1eb3548c70f0d4bf9a132d` | materially changed model/title/content |
| `figures/regression_coefficients.png` | 66,514 / `2887f8c4ab77cd0c4f2376452f9c7ba40616e45f778849133ed484299f4c592b` | 66,095 / `9e8bdbcdf49f52cb5228c54e15101f5769b286943f422a6b69ae199ed57dbbb1` | materially changed selected model/title |
| `figures/regression_residuals.png` | 235,151 / `bc36c9a85e5ddd40cadd30ccbdea076577ffdd8f20ab76a28eb0be72ec75af49` | 234,833 / `b8f19f19e2cb0082684c7209aa3e289da292f5675399d10a192358ef7854b7eb` | materially changed selected model/content |
| `figures/tracks_by_decade.png` | 49,800 / `f04309d0dd1f81c6079dba5f9fe55e85c2b477f9737ed5b3caeee0aff5f0a9e2` | 49,950 / `6c6a5f41d2fe89c58b9c0766600db57e4c57d36b5bc2948021c62fad2224f5b8` | decoded pixels identical; encoding/metadata differ |

### Model artifacts manifest

| Relative path | Current | Repository size / SHA-256 | Comparison |
|---|---|---|---|
| `model_artifacts/best_popularity_model.joblib` | absent | 1,969 / `2c92a06f8901d20cb05b24ddff272a5207b4918b0db548a83753d883efd93cde` | repository only; ignored by Git |
| `model_artifacts/nearest_neighbors_recommender.joblib` | absent | 12,287,598 / `c05e389e8d84b345786609689a843333ba7ae1f915ae92f78f46fe9bee0bcd0a` | repository only; ignored by Git |
| `model_artifacts/recommender_features.json` | absent | 151 / `b83476790576ae7a84740e1773afd855c03492544cd1ace3de89c60442854b4f` | repository only; feature list, not catalog |
| `model_artifacts/recommender_scaler.joblib` | absent | 1,151 / `2ab52ee3f0c9d9aa9b731e662098886e0d806e95469c943e30789e13f5a826bf` | repository only; ignored by Git |

### Run summary and other categories

| Relative path | Current size / SHA-256 | Repository size / SHA-256 | Comparison |
|---|---|---|---|
| `run_summary.json` | 2,508 / `fa78c9e569f62c691553842209a400008eb0c6b1cfbb0a9221d0675308a7c834` | 3,601 / `f8b1d97385740ffc6e7e8a5243838f36140b31e56f12d4067bcaa222c12aa5fc` | repository is the newer, more complete run |

Neither copy contains SQL files. There are no other files under either `week7_outputs/` beyond the table, figure, interactive, model-artifact, and run-summary entries above.

## 9. Authoritative-source matrix

| Component | Classification | Evidence |
|---|---|---|
| SOURCE CODE | **EXISTING GIT REPOSITORY** | Direct upgrade of committed/current baseline; fixes five audited gaps and adds required outputs. |
| REQUIREMENTS | **EXISTING GIT REPOSITORY** | Adds Plotly/Kaleido actually used by newer source. |
| DOCUMENTATION | **REQUIRES MANUAL DECISION** | Shared main docs are identical; current has unique audit docs; repository has unique report-package documentation. Preserve and merge both sets. |
| CLEANED DATA | **IDENTICAL** | All normalized hashes and parsed contents match; raw difference is only LF versus CRLF. |
| GENERATED TABLES | **EXISTING GIT REPOSITORY** | Complete 15-table run, corrected recommendations, added validation. |
| GENERATED FIGURES | **EXISTING GIT REPOSITORY** | Complete 10-visual run including Plotly; regression plots match corrected model policy. |
| MODEL ARTIFACTS | **EXISTING GIT REPOSITORY** | Only copy with four artifacts, though catalog mapping remains missing. |
| GIT HISTORY | **EXISTING GIT REPOSITORY** | Only valid repository and only copy connected to `origin/main`. |

## 10. Confirmed unique files in each copy

### Current workspace only

- `docs/CODE_AUDIT.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/ENVIRONMENT_BASELINE.md`
- This reconciliation report after creation
- `.venv/` with 17,013 environment files; reproducible and should not be migrated
- `__pycache__/spotify_week7_analysis.cpython-312.pyc`; generated and should not be migrated

The current source is not unique content because the repository contains the exact `spotify_week7_analysis_before_report_fix.py` backup and the same committed blob. No current output relative path is unique.

### Existing repository only

- Newer `spotify_week7_analysis.py` content and the two extra requirement lines
- `spotify_week7_analysis_before_report_fix.py` (content duplicate of current source, but a unique preservation path)
- `report_package/` (35 files)
- `report_package_updated/` (40 files)
- `report_tables/` (9 files)
- `spotify_report_package.zip` (1,173,997 bytes)
- `spotify_report_package_updated.zip` (11,187,588 bytes)
- `week7_outputs/tables/decade_explicit_multiindex_summary.csv`
- `week7_outputs/tables/recommendation_validation.csv`
- `week7_outputs/figures/interactive_energy_loudness.html`
- `week7_outputs/figures/interactive_energy_loudness_preview.png`
- Four ignored files under `week7_outputs/model_artifacts/`
- Valid `.git/` history and remote configuration

`report_package/01_python_code/spotify_week7_analysis.py` contains the old/current hash, while `report_package_updated/01_source/spotify_week7_analysis.py` contains the newer repository hash. The two packages are useful old/new snapshots and should be preserved until the final deliverable policy is decided.

## 11. Risks of choosing the wrong folder

### If the current workspace is chosen

- Git history and the configured remote would be abandoned or duplicated.
- The newer uncommitted source fixes would be lost unless manually reapplied.
- The complete 15-table/10-visual outputs, recommendation validation, model artifacts, and report packages could be omitted.
- Refactoring would start from known recommender and regression defects already fixed elsewhere.

### If the repository is chosen without reconciliation

- The three audit/environment documents and this report could be left behind.
- Dirty source/output changes might be accidentally overwritten or mixed into a refactor commit.
- Ignored model artifacts and untracked report packages could be lost because they are not protected by Git.
- The older June output baseline could be overwritten without an external snapshot.
- The absence of a saved recommender catalog could be overlooked because the newer validation passes only in-process.

## 12. Recommended final workspace

Final working repository:

```text
D:\spotify-recommendation-analytics-cũ
```

Continue in the existing Git repository rather than initializing the current workspace later. This preserves history, remote linkage, newer source, and the complete output set. Treat the current workspace as a read-only audit/reference copy until all audit documents and output snapshots have been backed up and migrated.

Do not copy the current `spotify_week7_analysis.py`, current `requirements.txt`, `cleaned_data/`, or current `week7_outputs/` over the repository. The current source is older, the requirements are incomplete for the newer source, the data is semantically identical, and the current outputs represent the older flawed run.

## 13. Non-destructive reconciliation procedure

1. **Obtain explicit approval and choose an external backup path.** Use a new, empty timestamped directory outside both project copies.
2. **Preserve Git history and dirty state.** Create a Git bundle, a binary working-tree patch, a status listing, and an untracked/ignored-file listing. These are additive backup operations only.
3. **Copy both evidence sets to the external backup.** Preserve both `week7_outputs/` folders under distinct run labels; preserve current `docs/`; preserve repository report packages, zip archives, source/requirements, backup source, model artifacts, and cleaned-data reports. Do not overwrite one snapshot with the other.
4. **Do not migrate `.venv/`.** Recreate `.venv` inside the final repository from Python 3.12 and the repository requirements; install pytest separately as a development dependency until dependency policy is decided.
5. **Create a temporary preservation branch in the repository.** `workspace-reconciliation` should be created from the current `main` commit while retaining the dirty working tree. This protects `main` from an accidental mixed commit.
6. **Commit the newer source separately.** Stage only `requirements.txt` and `spotify_week7_analysis.py`; review the staged diff; commit them as the completed pre-refactor pipeline.
7. **Copy the four reconciliation/audit docs into a new repository `docs/` folder.** Do this only after the external backup. Stage the exact doc paths and commit them separately. Keep `CODE_AUDIT.md` as historical evidence.
8. **Decide output versioning before staging outputs.** If report outputs are meant to be versioned, explicitly stage the 15 tables, 10 visuals, and `run_summary.json`—never `git add .`. If not, retain the July run as an external immutable snapshot and add narrowly scoped ignore rules. Preserve the June run only in the external backup; do not mix its files into the July output folder.
9. **Handle model artifacts explicitly.** They are currently ignored. Keep them in the external run snapshot or adopt Git LFS/release storage after a human decision. Do not force-add them automatically. The missing catalog artifact must be implemented later, not fabricated during reconciliation.
10. **Classify report packages and zip files.** Either commit selected human-readable manifests/notes, or add narrow ignore rules while keeping the files in both the repository working directory and external backup. Do not delete them to obtain a clean status.
11. **Reach a clean preservation baseline.** Every intentional source/doc/output change must be committed, and every retained but untracked deliverable must be deliberately ignored or tracked. Confirm `git status --short` is empty.
12. **Create `final-code-refactor` only now.** Branch from the clean reconciled preservation commit, then begin characterization tests. Do not run the canonical pipeline during reconciliation.

## 14. Exact commands proposed for the next step

These commands are proposals only and were **not** run. Replace the timestamp if needed and review every path before execution.

### A. Create external backups and Git evidence

```powershell
$repo = 'D:\spotify-recommendation-analytics-cũ'
$current = 'D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics'
$backup = 'D:\spotify-reconciliation-backup-20260715'

New-Item -ItemType Directory -Path $backup
New-Item -ItemType Directory -Path "$backup\current-audit-docs"
New-Item -ItemType Directory -Path "$backup\current-june-outputs"
New-Item -ItemType Directory -Path "$backup\repository-july-outputs"
New-Item -ItemType Directory -Path "$backup\repository-untracked"

git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo bundle create "$backup\repository-history.bundle" --all
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo diff --binary | Out-File -Encoding ascii "$backup\repository-working-tree.patch"
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo status --short --branch | Set-Content -Encoding utf8 "$backup\repository-status.txt"
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo ls-files --others --exclude-standard | Set-Content -Encoding utf8 "$backup\repository-untracked.txt"
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo status --ignored --short | Set-Content -Encoding utf8 "$backup\repository-ignored.txt"

Copy-Item -Path "$current\docs\*" -Destination "$backup\current-audit-docs" -Recurse
Copy-Item -Path "$current\week7_outputs\*" -Destination "$backup\current-june-outputs" -Recurse
Copy-Item -Path "$repo\week7_outputs\*" -Destination "$backup\repository-july-outputs" -Recurse
Copy-Item -LiteralPath "$repo\report_package" -Destination "$backup\repository-untracked\report_package" -Recurse
Copy-Item -LiteralPath "$repo\report_package_updated" -Destination "$backup\repository-untracked\report_package_updated" -Recurse
Copy-Item -LiteralPath "$repo\report_tables" -Destination "$backup\repository-untracked\report_tables" -Recurse
Copy-Item -LiteralPath "$repo\spotify_report_package.zip" -Destination "$backup\repository-untracked"
Copy-Item -LiteralPath "$repo\spotify_report_package_updated.zip" -Destination "$backup\repository-untracked"
Copy-Item -LiteralPath "$repo\spotify_week7_analysis.py" -Destination "$backup\repository-untracked"
Copy-Item -LiteralPath "$repo\spotify_week7_analysis_before_report_fix.py" -Destination "$backup\repository-untracked"
Copy-Item -LiteralPath "$repo\requirements.txt" -Destination "$backup\repository-untracked"
```

Before using those copy commands, create each destination only once so `Copy-Item` cannot merge with an old backup. Verify file counts and hashes afterward.

### B. Create a preservation branch and commit source

```powershell
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo switch -c workspace-reconciliation
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo add -- requirements.txt spotify_week7_analysis.py
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo diff --cached -- requirements.txt spotify_week7_analysis.py
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo commit -m "fix: preserve completed week 7 pipeline"
```

### C. Migrate the audit documents

```powershell
New-Item -ItemType Directory -Path "$repo\docs"
Copy-Item -LiteralPath "$current\docs\CODE_AUDIT.md" -Destination "$repo\docs\CODE_AUDIT.md"
Copy-Item -LiteralPath "$current\docs\IMPLEMENTATION_PLAN.md" -Destination "$repo\docs\IMPLEMENTATION_PLAN.md"
Copy-Item -LiteralPath "$current\docs\ENVIRONMENT_BASELINE.md" -Destination "$repo\docs\ENVIRONMENT_BASELINE.md"
Copy-Item -LiteralPath "$current\docs\WORKSPACE_RECONCILIATION.md" -Destination "$repo\docs\WORKSPACE_RECONCILIATION.md"

git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo add -- docs/CODE_AUDIT.md docs/IMPLEMENTATION_PLAN.md docs/ENVIRONMENT_BASELINE.md docs/WORKSPACE_RECONCILIATION.md
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo diff --cached -- docs
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo commit -m "docs: preserve audit and reconciliation baselines"
```

### D. Recreate the environment in the final repository

```powershell
py -3.12 -m venv "$repo\.venv"
& "$repo\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$repo\.venv\Scripts\python.exe" -m pip install -r "$repo\requirements.txt"
& "$repo\.venv\Scripts\python.exe" -m pip install pytest
& "$repo\.venv\Scripts\python.exe" -m compileall "$repo\spotify_week7_analysis.py"
& "$repo\.venv\Scripts\python.exe" "$repo\spotify_week7_analysis.py" --help
```

Do not copy the current `.venv`; environments embed absolute paths and are reproducible from requirements.

### E. Stage outputs only if the user approves versioning them

```powershell
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo add -- week7_outputs/run_summary.json week7_outputs/tables week7_outputs/figures
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo diff --cached --stat
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo diff --cached -- week7_outputs/run_summary.json week7_outputs/tables
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo commit -m "docs: preserve verified week 7 report outputs"
```

This command deliberately excludes `cleaned_data/` and ignored `model_artifacts/`. It must not be run until output-tracking policy is approved.

### F. Create the refactor branch only after reconciliation is clean

```powershell
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo status --short --branch
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo switch -c final-code-refactor
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo branch --show-current
git -c safe.directory='D:/spotify-recommendation-analytics-cũ' -C $repo rev-parse HEAD
```

Do not create `final-code-refactor` unless the status immediately before it is clean.

## 15. Human decisions still required

1. Approve `D:\spotify-recommendation-analytics-cũ` as the final working repository.
2. Approve the external backup location and whether full folder copies are also wanted in addition to the targeted evidence backup.
3. Decide whether the 15 tables, 10 visuals, and run summary should remain Git-tracked or live in immutable run archives/releases.
4. Decide whether joblib artifacts should remain ignored, use Git LFS, or be attached to a release. Do not force-add them by default.
5. Decide which report-package contents are source deliverables, which are duplicate archives, and which should be ignored after backup.
6. Decide whether `spotify_week7_analysis_before_report_fix.py` should be committed as historical evidence or retained only in the backup; it duplicates the committed/current baseline.
7. Decide whether to keep per-command `safe.directory` overrides or configure the existing repository as safe under the user's own account. No global configuration was changed during comparison.
8. Decide how to annotate historical audit findings that are already fixed in the newer uncommitted source.
9. Approve a future recommender catalog artifact format; the newer code still cannot map arbitrary reloaded neighbor indexes to metadata.
10. Decide dependency pinning and where pytest belongs before creating a lock file or modifying requirements further.

No implementation or refactoring should start until the reconciliation preservation commits are complete and `final-code-refactor` has been created from a clean state.
