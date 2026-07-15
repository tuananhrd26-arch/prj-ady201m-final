# Implementation Plan

Status: proposed only. No implementation in this plan has started. Every checklist item is intentionally unchecked.

## 1. Proposed final project tree

Only the requested implementation modules are proposed.

```text
spotify-recommendation-analytics/
|-- spotify_week7_analysis.py              # public compatibility entry point
|-- requirements.txt
|-- README.md
|-- RUN_INSTRUCTIONS.md
|-- REPORT_WEEK7_DRAFT.md
|-- cleaned_data/                          # unchanged source datasets/reports
|-- week7_outputs/                         # generated report artifacts
|-- docs/
|   |-- CODE_AUDIT.md
|   `-- IMPLEMENTATION_PLAN.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- eda.py
|   |-- visualization.py
|   |-- regression.py
|   |-- recommender.py
|   |-- validation.py
|   `-- pipeline.py
|-- scripts/
|   `-- recommend_song.py
`-- tests/
    |-- test_data_pipeline.py
    |-- test_regression.py
    `-- test_recommender.py
```

No other `src`, `scripts`, or `tests` modules are proposed.

## 2. Files that will remain unchanged

- All existing files in `cleaned_data/`; no dataset will be rewritten, moved, or renamed.
- Existing output artifacts will remain untouched until an explicitly approved, protected-output validation run.
- `REPORT_WEEK7_DRAFT.md` will remain unchanged during code migration. Any corrections should be a separate, evidence-backed documentation phase.
- `.gitignore` will remain unchanged during initial characterization. Later artifact/versioning decisions require explicit agreement.
- The required public command and filename will remain available: `py spotify_week7_analysis.py --root . --skip-sql`.

`README.md`, `RUN_INSTRUCTIONS.md`, and `requirements.txt` should change only in the final integration/documentation phase if tests prove a new dependency or command description is necessary.

## 3. Files that will be created

- `src/__init__.py`
- `src/config.py`
- `src/data_loader.py`
- `src/preprocessing.py`
- `src/eda.py`
- `src/visualization.py`
- `src/regression.py`
- `src/recommender.py`
- `src/validation.py`
- `src/pipeline.py`
- `scripts/recommend_song.py`
- `tests/test_data_pipeline.py`
- `tests/test_regression.py`
- `tests/test_recommender.py`

The two files under `docs/` were created by the audit, not by the future implementation.

## 4. Files that will later become compatibility wrappers

- `spotify_week7_analysis.py` will become a thin public compatibility wrapper that imports and calls `src.pipeline.main` (or an equivalently narrow public function) while preserving all current CLI options and the required command.
- No dataset, report, or output file will become a wrapper.

The entry point should not be converted until modular behavior is covered by characterization tests and command compatibility can be checked in a protected output location.

## 5. Migration phases

### Phase 0: Restore the engineering baseline

Purpose: make changes traceable and executions reproducible before implementation.

Actions:

- Restore the intended `.git` metadata or re-enter the actual Git clone.
- Confirm a clean/non-clean status without resetting, cleaning, stashing, or discarding files.
- Create `final-code-refactor` only if the restored working tree is clean.
- Create/activate the approved environment and install only reviewed project dependencies.
- Decide the exact 15-table and 10-visual manifests and model-selection policy.

Acceptance criteria:

- Git branch, commit, and status are available.
- Python/package versions are captured.
- All required imports pass, including Plotly if the interactive output is approved.
- `--help` exits zero and lists `--root`, `--output`, and `--skip-sql`.
- No existing output was overwritten.

Test commands:

```powershell
git status --short --branch
git branch --show-current
git rev-parse HEAD
py --version
py -c "import pandas, numpy, sklearn, matplotlib, seaborn, plotly, joblib"
py spotify_week7_analysis.py --help
```

### Phase 1: Add characterization and validation tests

Purpose: capture current valid behavior and explicitly encode corrected invariants before moving code.

Actions:

- Add dataset schema/domain tests in `tests/test_data_pipeline.py`.
- Add deterministic split/metric/model-identity tests in `tests/test_regression.py`.
- Add seed/vector identity, self-exclusion, uniqueness, exact Top N, finite similarity, and reloadable catalog mapping tests in `tests/test_recommender.py`.
- Put reusable validation rules in `src/validation.py` only when the tests need production validation behavior.
- Use temporary output directories in tests.

Acceptance criteria:

- Tests reproduce the current recommender mismatch as a failing characterization or encode the intended corrected behavior as initially failing tests.
- Data tests make no writes to `cleaned_data/`.
- Tests make no writes to `week7_outputs/`.
- Regression tests state one unambiguous best/plot/saved model policy.

Test commands:

```powershell
py -m pytest tests/test_data_pipeline.py -q
py -m pytest tests/test_regression.py -q
py -m pytest tests/test_recommender.py -q
```

### Phase 2: Extract configuration, loading, and preprocessing

Purpose: establish pure, testable data boundaries without changing analytical results.

Actions:

- Move constants and path configuration to `src/config.py`.
- Move CSV discovery/loading to `src/data_loader.py`.
- Move non-destructive track preparation to `src/preprocessing.py`.
- Keep dataset names, feature order, median behavior, and derived `decade` compatible unless a test-approved correction is needed.

Acceptance criteria:

- Loaded row/column counts match the audit.
- Input data hashes and mtimes do not change.
- Processed track schema matches current behavior.
- Missing and domain validation results match the audit.

Test commands:

```powershell
py -m pytest tests/test_data_pipeline.py -q
py -m compileall src
```

### Phase 3: Extract EDA and visualization

Purpose: separate calculations from rendering and establish explicit output manifests.

Actions:

- Move table calculations to `src/eda.py`.
- Move static and interactive rendering to `src/visualization.py`.
- Define and test exactly 15 analytical tables and 10 visuals, including at least one Plotly HTML output.
- Make output paths injectable so tests use temporary directories.

Acceptance criteria:

- The approved manifest contains exactly 15 table files and 10 visual files.
- Every output is nonempty and can be parsed/opened.
- The interactive HTML contains a Plotly visualization and is usable offline or has a documented external-JS policy.
- Calculated legacy tables match approved baseline values within stated numeric tolerances.

Test commands:

```powershell
py -m pytest tests/test_data_pipeline.py -q
py -m pytest -q
py spotify_week7_analysis.py --root . --output .audit_outputs --skip-sql
```

The `.audit_outputs` command is only permitted after output-protection and cleanup procedures are approved.

### Phase 4: Extract and reconcile regression

Purpose: make model evaluation, selection, charts, persistence, and metadata use one identity policy.

Actions:

- Move regression experiments to `src/regression.py`.
- Preserve feature sets, test size, random state, pipelines, and metrics unless tests justify a change.
- Select the best model with a documented deterministic tie-break across unrounded metrics.
- Use the selected model consistently for diagnostics and persistence, or explicitly name a deliberate alternate everywhere.
- Persist the pipeline plus feature order and selection metadata.

Acceptance criteria:

- Four expected experiment rows are produced.
- MAE, RMSE, and R2 calculations are tested.
- Best, plotted, saved, artifact-named, and summary-named model identities agree.
- Reloaded model predictions match pre-save predictions within tolerance.

Test commands:

```powershell
py -m pytest tests/test_regression.py -q
py -m pytest -q
```

### Phase 5: Extract and correct recommender

Purpose: make recommendation identity and persistence correct by construction.

Actions:

- Move catalog construction/training/querying to `src/recommender.py`.
- Retain the full catalog index or stable `id` through seed selection; never use a reset presentation index as a matrix index.
- Exclude the actual seed by stable identity/index, not by dropping the first result.
- Over-fetch neighbors, deduplicate on normalized `name + artists`, and continue until exactly Top N unique results or return an explicit insufficient-catalog status.
- Validate finite features, distances, and similarities.
- Save a catalog mapping aligned one-to-one with the fitted matrix.
- Implement `scripts/recommend_song.py` as the reload-and-query interface.

Acceptance criteria:

- Displayed seed and vector seed are identical for multiple tested seeds.
- The seed never appears in its own results.
- Every successful query returns exactly Top N unique pairs with contiguous ranks.
- Similarities are finite and within the documented cosine-derived range.
- Reloaded artifacts map every neighbor row to the correct metadata.
- Repeated calls with the same inputs are deterministic.

Test commands:

```powershell
py -m pytest tests/test_recommender.py -q
py scripts/recommend_song.py --help
py -m pytest -q
```

### Phase 6: Build pipeline orchestration and preserve the entry point

Purpose: wire modules together while keeping the academic command stable.

Actions:

- Put orchestration and CLI construction in `src/pipeline.py`.
- Convert `spotify_week7_analysis.py` to the thin compatibility wrapper only after modular parity passes.
- Generate `run_summary.json` from the current-run manifest, not directory globs that may include stale files.
- Record command/options, versions, source/data identifiers, model identity/artifacts, recommendation validation, and output hashes.

Acceptance criteria:

- `py spotify_week7_analysis.py --root . --skip-sql` remains valid.
- `--help` does not require analysis execution.
- The run summary corresponds only to the current run and uses portable relative paths where possible.
- The approved table, visual, model, and validation counts are met.
- Existing outputs are not touched during tests; the final canonical run requires separate approval.

Test commands:

```powershell
py spotify_week7_analysis.py --help
py spotify_week7_analysis.py --root . --output .audit_outputs --skip-sql
py -m pytest -q
```

### Phase 7: Documentation and final protected run

Purpose: reconcile instructions/report claims with verified source and artifacts.

Actions:

- Update `requirements.txt` only with dependencies actually used.
- Update `README.md` and `RUN_INSTRUCTIONS.md` for the preserved public command and artifact manifest.
- Correct report claims only from regenerated verified outputs.
- Run the canonical pipeline only after explicit approval to overwrite/version `week7_outputs/`.

Acceptance criteria:

- Fresh-environment installation and test commands succeed.
- Documentation names only files that exist or clearly labels optional outputs.
- Report metrics, figure labels, summary identities, and saved artifacts agree.
- Git diff contains no dataset rewrite and no unintended output deletion.

Test commands:

```powershell
py -m pip install -r requirements.txt
py -m pytest -q
py spotify_week7_analysis.py --root . --skip-sql
git status --short
git diff --check
```

The canonical command is deferred until overwrite approval is explicit.

## 6. Acceptance criteria by phase

| Phase | Required gate before continuing |
|---|---|
| 0 | Git and dependency baselines are available; CLI help works. |
| 1 | Tests safely encode schemas, regression identity, and recommender invariants. |
| 2 | Loading/preprocessing parity passes without input mutations. |
| 3 | Approved 15-table/10-visual manifests pass in a temporary output. |
| 4 | Regression best/plot/save/summary identity and reload checks pass. |
| 5 | Seed identity, uniqueness, Top N, finiteness, and catalog reload checks pass. |
| 6 | Public command compatibility and reproducible run summary pass. |
| 7 | Clean-environment test, documentation consistency, and protected canonical run pass. |

## 7. Command matrix

```powershell
# Static checks
py -m compileall spotify_week7_analysis.py src scripts tests

# Focused tests
py -m pytest tests/test_data_pipeline.py -q
py -m pytest tests/test_regression.py -q
py -m pytest tests/test_recommender.py -q

# Full tests
py -m pytest -q

# Public CLI compatibility
py spotify_week7_analysis.py --help
py spotify_week7_analysis.py --root . --output .audit_outputs --skip-sql

# Final command, only after canonical-output overwrite approval
py spotify_week7_analysis.py --root . --skip-sql
```

## 8. Rollback strategy

Git rollback is not currently available because this workspace has no `.git` metadata. Phase 0 must resolve that before implementation.

After Git is restored:

1. Start from a recorded commit and create `final-code-refactor` only from a clean tree.
2. Commit each accepted phase separately; do not combine dataset/output changes with source refactoring.
3. Run all experimental pipelines against a newly named temporary output directory outside `week7_outputs/`.
4. If a phase fails, stop using that phase's code and return to the last accepted commit through a new corrective commit or a user-approved, non-destructive Git operation. Never use `git reset --hard`, `git clean`, or automatic discard/stash commands.
5. Before a canonical run, copy/version the existing output manifest through an explicitly approved process. Do not silently delete or replace outputs.
6. Because cleaned inputs are immutable, validate their hashes before and after every phase that runs data code.

## 9. Unchecked implementation checklist

### Baseline

- [ ] Restore or locate valid Git metadata.
- [ ] Record branch, commit, and status.
- [ ] Create `final-code-refactor` only from a clean tree.
- [ ] Provision the approved dependency environment.
- [ ] Make all required imports pass.
- [ ] Approve the exact 15-table manifest.
- [ ] Approve the exact 10-visual manifest.
- [ ] Approve the regression selection policy.
- [ ] Approve the recommender catalog artifact format.
- [ ] Approve protected temporary and canonical output policies.

### Tests and validation

- [ ] Create `tests/test_data_pipeline.py`.
- [ ] Create `tests/test_regression.py`.
- [ ] Create `tests/test_recommender.py`.
- [ ] Test every audited dataset schema and domain.
- [ ] Test that inputs are never rewritten.
- [ ] Test regression split reproducibility and metrics.
- [ ] Test consistent regression model identity.
- [ ] Test displayed-seed/vector-seed equality.
- [ ] Test actual seed exclusion.
- [ ] Test recommendation deduplication and exact Top N.
- [ ] Test finite similarities.
- [ ] Test artifact reload and catalog mapping.

### Modules

- [ ] Create `src/__init__.py`.
- [ ] Create `src/config.py`.
- [ ] Create `src/data_loader.py`.
- [ ] Create `src/preprocessing.py`.
- [ ] Create `src/eda.py`.
- [ ] Create `src/visualization.py`.
- [ ] Create `src/regression.py`.
- [ ] Create `src/recommender.py`.
- [ ] Create `src/validation.py`.
- [ ] Create `src/pipeline.py`.
- [ ] Create `scripts/recommend_song.py`.

### Outputs and integration

- [ ] Produce exactly 15 approved analytical tables.
- [ ] Produce exactly 10 approved visuals.
- [ ] Produce and verify an interactive Plotly HTML output.
- [ ] Persist four or more complete, reloadable model/recommender artifacts.
- [ ] Persist a row-aligned recommender catalog mapping.
- [ ] Validate recommendations for multiple real seed songs.
- [ ] Make `run_summary.json` portable and current-run reproducible.
- [ ] Convert `spotify_week7_analysis.py` to a compatibility wrapper.
- [ ] Preserve `py spotify_week7_analysis.py --root . --skip-sql`.
- [ ] Update dependencies and instructions from verified behavior.
- [ ] Run the complete test suite.
- [ ] Obtain approval before the canonical output run.
- [ ] Run the final command successfully.
- [ ] Verify the final Git diff contains no unintended changes.

No phase should begin until the audit is reviewed and approved.
