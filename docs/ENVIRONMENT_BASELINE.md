# Environment Baseline

Baseline date: 2026-07-15 (Asia/Saigon)  
Scope: environment and version-control discovery/setup only. The Spotify analysis pipeline was not run.

## 1. Project path

Absolute project path:

```text
D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics
```

All required workspace paths were present:

- `spotify_week7_analysis.py`
- `requirements.txt`
- `README.md`
- `RUN_INSTRUCTIONS.md`
- `cleaned_data/`
- `week7_outputs/`
- `docs/CODE_AUDIT.md`
- `docs/IMPLEMENTATION_PLAN.md`

## 2. Existing Git repository discovery

A valid existing repository for the same project was found at:

```text
D:\spotify-recommendation-analytics-cũ
```

Evidence that it is the same project:

- Origin URL: `https://github.com/tuananhrd26-arch/spotify-recommendation-analytics.git`
- It contains the same Spotify project entry point, documentation, cleaned-data structure, and output structure.
- `.gitignore`, `README.md`, `REPORT_WEEK7_DRAFT.md`, and `RUN_INSTRUCTIONS.md` are byte-identical between the current copy and that repository.
- `requirements.txt` and `spotify_week7_analysis.py` differ because the existing repository has uncommitted changes.

The repository initially triggered Git's dubious-ownership protection because the sandbox user differs from the owner. It was inspected read-only by passing `safe.directory` for individual commands; global or repository Git configuration was not changed.

## 3. Git initialization decision

No new Git repository was initialized in the current project copy. The instruction required Git setup to stop when a valid repository for this exact project was found elsewhere.

The current project path therefore still has no valid Git branch, commit, index, or status. Its exposed `.git/` directory is an empty managed-workspace placeholder, not a repository.

## 4. Branch, commit, and repository state

Current project copy:

- Branch: unavailable
- Commit: unavailable
- `git status --short --branch`: `fatal: not a git repository (or any of the parent directories): .git`

Existing repository at `D:\spotify-recommendation-analytics-cũ`:

- Branch: `main`
- Commit: `0ce731edaa93c5741368b29ff100c9187a2ebac9`
- Upstream status: `main...origin/main`
- Working tree: not clean; it contains modified tracked outputs/source and untracked report/output files.

No baseline commit was created and no `final-code-refactor` branch was created in either directory.

## 5. Git remote

The existing repository has this remote:

```text
origin  https://github.com/tuananhrd26-arch/spotify-recommendation-analytics.git
```

The current project copy has no Git remote because it is not a repository.

## 6. Git identity

Git identity is configured and was not changed:

- `user.name`: `tuananhrd26-arch`
- `user.email`: `tuananhrd26@gmail.com`

Identity was available both as the effective/global identity and when querying the existing repository.

## 7. Python environment

The project virtual environment was created at:

```text
D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics\.venv
```

Runtime details:

- Python executable: `D:\spotify-recommendation-analytics-main (1)\spotify-recommendation-analytics\.venv\Scripts\python.exe`
- Python version: `3.12.6`
- Architecture/runtime: CPython, MSC v.1940, 64 bit (AMD64)
- pip version: `26.1.2`

All Python and pip verification commands after virtual-environment creation used `.venv\Scripts\python.exe`.

## 8. Installed dependency versions

Only the packages declared by `requirements.txt`, their resolver-selected transitive dependencies, and the two approved additions `pytest` and `plotly` were installed.

| Import/package | Version | Import result |
|---|---:|---|
| pandas | 3.0.3 | success |
| numpy | 2.5.1 | success |
| sklearn / scikit-learn | 1.9.0 | success |
| matplotlib | 3.11.0 | success |
| seaborn | 0.13.2 | success |
| plotly | 6.9.0 | success |
| joblib | 1.5.3 | success |
| pytest | 9.1.1 | success |

`requirements.txt` was inspected but not modified. No lock file was created.

## 9. Compile check

Command:

```powershell
.venv\Scripts\python.exe -m compileall spotify_week7_analysis.py
```

Result: **success**. The command compiled `spotify_week7_analysis.py` without a syntax error and created the ignored cache file `__pycache__/spotify_week7_analysis.cpython-312.pyc`.

## 10. CLI help check

Command:

```powershell
.venv\Scripts\python.exe spotify_week7_analysis.py --help
```

Result: **success**, exit code 0. The help output exposes all required options:

- `--root ROOT`
- `--output OUTPUT`
- `--skip-sql`

The full `--root . --skip-sql` analysis command was not run.

Matplotlib emitted a non-blocking warning because the sandbox could not write a font-manager lock file under `C:\Users\Tuananh\.matplotlib`. This did not prevent imports or CLI help from succeeding and did not write inside the project outputs.

## 11. Files staged and committed

None.

No file was staged, no commit was created, and no branch was created. `git add .` was not used. This is the required outcome after discovering the valid repository elsewhere.

The approved baseline file list was therefore not staged or committed in the current copy:

- `.gitignore`
- `README.md`
- `REPORT_WEEK7_DRAFT.md`
- `RUN_INSTRUCTIONS.md`
- `requirements.txt`
- `spotify_week7_analysis.py`
- `docs/CODE_AUDIT.md`
- `docs/IMPLEMENTATION_PLAN.md`

## 12. Cleaned-data preservation

Confirmed: no file under `cleaned_data/` was modified, rewritten, staged, committed, moved, or deleted.

The seven audited files remain present, and all seven sizes match the prior code-audit inventory exactly. Preservation check mismatch count: **0**.

## 13. Existing-output preservation

Confirmed: no file under `week7_outputs/` was modified, overwritten, staged, committed, moved, or deleted.

The 22 audited output files remain present, and all 22 sizes match the prior code-audit inventory exactly. Preservation check mismatch count: **0**.

No analysis, regression training, recommender fitting, SQL generation, or output rendering was run.

## 14. Errors and blockers

1. The current project copy is still not version-controlled because an existing repository for the same project was found elsewhere and initialization here was prohibited.
2. The existing repository is not clean. Its modified/untracked files must be reviewed and preserved before creating `final-code-refactor`; no destructive Git operation should be used.
3. The current and existing-repository versions of `requirements.txt` and `spotify_week7_analysis.py` differ, so the authoritative copy must be chosen or reconciled by the user before implementation.
4. No baseline commit or refactor branch exists yet.
5. The current `.gitignore` was not edited because the Git setup portion stopped. It already ignores `.venv/`, `__pycache__/`, individual Python cache extensions, and `*.joblib`, but does not contain the requested consolidated `*.py[cod]`, `.pytest_cache/`, or `.audit_outputs/` patterns.
6. The Matplotlib font-cache warning described above is non-blocking. If later rendering needs a writable cache, set `MPLCONFIGDIR` to an approved project-temporary location rather than writing outside the workspace.

Environment setup itself has no remaining package, import, compile, or CLI-help blocker.
