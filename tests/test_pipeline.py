"""Acceptance tests for reusable pipeline orchestration and the thin CLI."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Generator
from datetime import datetime as RealDateTime
from datetime import timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from src.config import DEFAULT_OUTPUT_DIRNAME, ProjectPaths
from src.data_loader import load_project_data
from src.preprocessing import clean_tracks_for_analysis
from src.recommender import build_recommender_demo
from src.regression import regression_analysis
from src.sql_analysis import SQL_QUERIES, create_sqlite_database, run_sql_outputs


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def repository_preservation(project_root: Path) -> Generator[None, None, None]:
    protected = [project_root / "cleaned_data", project_root / "week7_outputs"]
    before = {path.name: _tree_manifest(path) for path in protected}
    yield
    assert {path.name: _tree_manifest(path) for path in protected} == before


@pytest.fixture(scope="session")
def pipeline_module(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ModuleType, None, None]:
    previous_backend = os.environ.get("MPLBACKEND")
    previous_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(tmp_path_factory.mktemp("pipeline-matplotlib"))
    try:
        yield importlib.import_module("src.pipeline")
    finally:
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend
        if previous_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_config


@pytest.fixture(scope="session")
def project_module(pipeline_module: ModuleType) -> ModuleType:
    return importlib.import_module("spotify_week7_analysis")


def test_pipeline_import_has_no_side_effects(project_root: Path, tmp_path: Path) -> None:
    mpl_config = tmp_path / "matplotlib-config"
    mpl_config.mkdir()
    script = """
import importlib,json,joblib,sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
calls=[]
pd.read_csv=lambda *a,**k: calls.append('read_csv')
joblib.dump=lambda *a,**k: calls.append('joblib_dump')
sqlite3.connect=lambda *a,**k: calls.append('sqlite_connect')
plt.figure=lambda *a,**k: calls.append('figure')
StandardScaler.fit=lambda *a,**k: calls.append('scaler_fit')
NearestNeighbors.fit=lambda *a,**k: calls.append('neighbors_fit')
before_files=sorted(p.relative_to(Path.cwd()).as_posix() for p in Path.cwd().rglob('*') if p.is_file())
before_dirs=sorted(p.relative_to(Path.cwd()).as_posix() for p in Path.cwd().rglob('*') if p.is_dir())
importlib.import_module('src.pipeline')
after_files=sorted(p.relative_to(Path.cwd()).as_posix() for p in Path.cwd().rglob('*') if p.is_file())
after_dirs=sorted(p.relative_to(Path.cwd()).as_posix() for p in Path.cwd().rglob('*') if p.is_dir())
print(json.dumps({'files_equal':before_files==after_files,'dirs_equal':before_dirs==after_dirs,'calls':calls}))
"""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_config)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "files_equal": True,
        "dirs_equal": True,
        "calls": [],
    }


def test_public_compatibility_identities(
    pipeline_module: ModuleType,
    project_module: ModuleType,
) -> None:
    expected = {
        "make_paths": pipeline_module.make_paths,
        "create_eda_outputs": pipeline_module.create_eda_outputs,
        "write_run_summary": pipeline_module.write_run_summary,
        "run_pipeline": pipeline_module.run_pipeline,
        "load_project_data": load_project_data,
        "clean_tracks_for_analysis": clean_tracks_for_analysis,
        "regression_analysis": regression_analysis,
        "build_recommender_demo": build_recommender_demo,
        "create_sqlite_database": create_sqlite_database,
        "run_sql_outputs": run_sql_outputs,
        "SQL_QUERIES": SQL_QUERIES,
    }
    for name, value in expected.items():
        assert getattr(project_module, name) is value


def test_make_paths_preserves_relative_absolute_and_existing_behavior(
    pipeline_module: ModuleType,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    relative = pipeline_module.make_paths(root, "relative-output")
    assert relative == ProjectPaths(
        root=root,
        output=root / "relative-output",
        tables=root / "relative-output" / "tables",
        figures=root / "relative-output" / "figures",
        sql=root / "relative-output" / "sql",
        model_artifacts=root / "relative-output" / "model_artifacts",
    )
    for path in relative.__dict__.values():
        if path != root:
            assert path.is_dir()

    marker = relative.tables / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    assert pipeline_module.make_paths(root, "relative-output") == relative
    assert marker.read_text(encoding="utf-8") == "keep"

    absolute_output = tmp_path / "absolute-output"
    absolute = pipeline_module.make_paths(tmp_path / "ignored", absolute_output)
    assert absolute.output == absolute_output
    assert absolute.tables == absolute_output / "tables"
    assert absolute.figures == absolute_output / "figures"
    assert absolute.sql == absolute_output / "sql"
    assert absolute.model_artifacts == absolute_output / "model_artifacts"


def test_write_run_summary_matches_pre_extraction_contract(
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = pipeline_module.make_paths(tmp_path, "outputs")
    (paths.tables / "z.csv").write_text("z\n1\n", encoding="utf-8")
    (paths.figures / "a.png").write_bytes(b"png")
    (paths.model_artifacts / "model.joblib").write_bytes(b"model")

    started = RealDateTime(
        2026, 7, 16, 9, 59, 58, 750000, tzinfo=timezone(timedelta(hours=7))
    )
    completed = RealDateTime(
        2026, 7, 16, 10, 0, 0, tzinfo=timezone(timedelta(hours=7))
    )

    class FixedDateTime:
        @classmethod
        def now(cls) -> RealDateTime:
            return completed

    monkeypatch.setattr(pipeline_module, "datetime", FixedDateTime)
    input_files = {
        "tracks": tmp_path / "cleaned_data" / "data_clean.csv",
        "genres": tmp_path / "cleaned_data" / "genres.csv",
    }
    tracks = pd.DataFrame({"id": ["a", "b"], "popularity": [1, 2]})
    tracks.attrs = {
        "interactive_plot_note": "interactive-note",
        "genre_pivot_note": "genre-note",
    }
    regression = {
        "models_trained": ["m1"],
        "best_model": {"name": "m1"},
        "plot_model": {"name": "m1"},
    }
    recommendation = {"demo_rows": 5, "validation_passed": True}
    tracks_before = tracks.copy(deep=True)
    regression_before = json.loads(json.dumps(regression))
    recommendation_before = recommendation.copy()
    input_before = input_files.copy()

    returned = pipeline_module.write_run_summary(
        paths,
        input_files,
        tracks,
        regression,
        recommendation,
        False,
        started,
    )

    expected = {
        "run_started_at": "2026-07-16T09:59:58+07:00",
        "run_completed_at": "2026-07-16T10:00:00+07:00",
        "run_duration_seconds": 1.25,
        "input_files": {name: str(path.resolve()) for name, path in input_files.items()},
        "main_dataset_rows": 2,
        "main_dataset_columns": 2,
        "tables_folder": str(paths.tables),
        "figures_folder": str(paths.figures),
        "sql_folder": str(paths.sql),
        "model_artifacts_folder": str(paths.model_artifacts),
        "tables_created": ["z.csv"],
        "figures_created": ["a.png"],
        "models_trained": ["m1"],
        "best_regression_model_by_R2": {"name": "m1"},
        "regression_plot_model": {"name": "m1"},
        "recommendation_summary": recommendation,
        "interactive_plot_note": "interactive-note",
        "model_artifacts_created": ["model.joblib"],
        "sql_reference_executed": False,
        "notes": [
            "genre-note",
            (
                "SQLite/SQL outputs are optional reference outputs. "
                "The group's main SQL work is completed separately in "
                "SQL Server Management Studio."
            ),
        ],
    }
    content = (paths.output / "run_summary.json").read_text(encoding="utf-8")
    assert returned is None
    assert list(json.loads(content)) == list(expected)
    assert json.loads(content) == expected
    assert content == json.dumps(expected, indent=2)
    assert not content.endswith("\n")
    assert not (paths.output / "run_summary.json").read_bytes().startswith(b"\xef\xbb\xbf")
    assert input_files == input_before
    assert regression == regression_before
    assert recommendation == recommendation_before
    pd.testing.assert_frame_equal(tracks, tracks_before)


def _install_lightweight_dependencies(
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[tuple[Any, ...]],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, ProjectPaths]:
    paths = pipeline_module.make_paths(tmp_path, "outputs")
    data = {"tracks": pd.DataFrame({"id": ["raw"]})}
    tracks = pd.DataFrame({"id": ["clean"]})
    input_files = {"tracks": tmp_path / "input.csv"}
    regression = {
        "models_trained": ["model"],
        "best_model": {"model": "model"},
        "plot_model": {"model": "model"},
    }
    recommendation = {"validation_passed": True}

    monkeypatch.setattr(
        pipeline_module,
        "make_paths",
        lambda root, output: calls.append(("make_paths", root, output)) or paths,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_project_data",
        lambda root: calls.append(("load_project_data", root)) or (data, input_files),
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_eda_outputs",
        lambda received, received_paths: calls.append(
            ("create_eda_outputs", received, received_paths)
        ) or tracks,
    )
    monkeypatch.setattr(
        pipeline_module,
        "regression_analysis",
        lambda received, received_paths: calls.append(
            ("regression_analysis", received, received_paths)
        ) or regression,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_recommender_demo",
        lambda received, received_paths: calls.append(
            ("build_recommender_demo", received, received_paths)
        ) or recommendation,
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_run_summary",
        lambda **kwargs: calls.append(("write_run_summary", kwargs)),
    )
    return data, tracks, paths


def test_run_pipeline_skip_sql_preserves_order_arguments_and_return(
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Any, ...]] = []
    data, tracks, paths = _install_lightweight_dependencies(
        pipeline_module, monkeypatch, tmp_path, calls
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_sqlite_database",
        lambda *args: pytest.fail("SQL database creation ran with skip_sql=True"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_sql_outputs",
        lambda *args: pytest.fail("SQL output execution ran with skip_sql=True"),
    )

    result = pipeline_module.run_pipeline(tmp_path, "ignored", skip_sql=True)

    assert [call[0] for call in calls] == [
        "make_paths",
        "load_project_data",
        "create_eda_outputs",
        "regression_analysis",
        "build_recommender_demo",
        "write_run_summary",
    ]
    assert calls[0][1:] == (tmp_path.resolve(), "ignored")
    assert calls[2][1:] == (data, paths)
    assert calls[3][1:] == (tracks, paths)
    assert calls[4][1:] == (tracks, paths)
    summary_arguments = calls[5][1]
    assert summary_arguments["sql_executed"] is False
    assert summary_arguments["paths"] is paths
    assert summary_arguments["tracks"] is tracks
    assert result["paths"] is paths
    assert result["tracks"] is tracks
    assert result["sql_reference_executed"] is False
    assert result["run_summary_path"] == paths.output / "run_summary.json"
    assert list(paths.sql.iterdir()) == []
    output = capsys.readouterr().out
    assert "Skipping optional SQLite/SQL reference outputs..." in output
    assert f"Done. Outputs saved to: {paths.output}" in output


def test_run_pipeline_executes_optional_sql_in_order(
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Any, ...]] = []
    data, _, paths = _install_lightweight_dependencies(
        pipeline_module, monkeypatch, tmp_path, calls
    )
    database = paths.sql / "spotify_week7.sqlite"
    monkeypatch.setattr(
        pipeline_module,
        "create_sqlite_database",
        lambda received, received_paths: calls.append(
            ("create_sqlite_database", received, received_paths)
        ) or database,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_sql_outputs",
        lambda received_db, received_paths: calls.append(
            ("run_sql_outputs", received_db, received_paths)
        ),
    )

    result = pipeline_module.run_pipeline(tmp_path, "ignored", skip_sql=False)

    assert [call[0] for call in calls] == [
        "make_paths",
        "load_project_data",
        "create_eda_outputs",
        "regression_analysis",
        "build_recommender_demo",
        "create_sqlite_database",
        "run_sql_outputs",
        "write_run_summary",
    ]
    assert calls[5][1:] == (data, paths)
    assert calls[6][1:] == (database, paths)
    assert calls[7][1]["sql_executed"] is True
    assert result["sql_reference_executed"] is True


@pytest.mark.parametrize(
    ("stage", "skip_sql"),
    [
        ("load_project_data", True),
        ("create_eda_outputs", True),
        ("regression_analysis", True),
        ("build_recommender_demo", True),
        ("create_sqlite_database", False),
        ("run_sql_outputs", False),
    ],
)
def test_run_pipeline_propagates_stage_failures(
    stage: str,
    skip_sql: bool,
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Any, ...]] = []
    _install_lightweight_dependencies(pipeline_module, monkeypatch, tmp_path, calls)
    database = tmp_path / "outputs" / "sql" / "spotify_week7.sqlite"
    monkeypatch.setattr(
        pipeline_module,
        "create_sqlite_database",
        lambda *args: database,
    )
    monkeypatch.setattr(pipeline_module, "run_sql_outputs", lambda *args: None)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"failure:{stage}")

    monkeypatch.setattr(pipeline_module, stage, fail)
    with pytest.raises(RuntimeError, match=f"failure:{stage}"):
        pipeline_module.run_pipeline(tmp_path, "outputs", skip_sql=skip_sql)
    assert not any(call[0] == "write_run_summary" for call in calls)


def test_monolith_is_a_thin_entry_point(project_root: Path) -> None:
    source_path = project_root / "spotify_week7_analysis.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert local_functions == {"main"}
    forbidden = {
        "make_paths",
        "create_eda_outputs",
        "regression_analysis",
        "build_recommender_demo",
        "create_sqlite_database",
        "run_sql_outputs",
    }
    assert local_functions.isdisjoint(forbidden)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    calls = {
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_pipeline" in calls


def test_cli_help_retains_options(project_root: Path, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-config")
    completed = subprocess.run(
        [sys.executable, str(project_root / "spotify_week7_analysis.py"), "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--root" in completed.stdout
    assert "--output" in completed.stdout
    assert "--skip-sql" in completed.stdout
    assert completed.stderr == ""


def test_cli_main_delegates_arguments(
    project_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        project_module,
        "run_pipeline",
        lambda root, output, *, skip_sql: calls.append((root, output, skip_sql)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spotify_week7_analysis.py",
            "--root",
            str(tmp_path),
            "--output",
            "audit-output",
            "--skip-sql",
        ],
    )
    assert project_module.main() is None
    assert calls == [(tmp_path, "audit-output", True)]


def test_run_pipeline_default_output_contract(pipeline_module: ModuleType) -> None:
    assert pipeline_module.run_pipeline.__defaults__ == (DEFAULT_OUTPUT_DIRNAME,)
