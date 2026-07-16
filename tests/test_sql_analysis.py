"""Characterization tests for extracted SQLite reference analysis."""

from __future__ import annotations

import hashlib
import importlib
import os
import sqlite3
import subprocess
import sys
from collections.abc import Generator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import ProjectPaths
from src.data_loader import load_project_data
from src.preprocessing import clean_tracks_for_analysis


QUERY_NAMES = [
    "01_tracks_by_decade",
    "02_audio_features_by_year",
    "03_top_tracks_by_decade_window",
    "04_above_average_popularity_subquery",
    "05_top_artists_by_popularity",
    "06_top_genre_audio_profiles",
]
BASELINE = {
    "01_tracks_by_decade.csv": ((11, 5), "81f654c7e3c3d863975927d4933b339506b7b011dcf971dfbfeb3bdf67f4faeb", "e791b3dc4603b70cf23c408b749558485cecd839ff3b952dc48d910c397276c5"),
    "02_audio_features_by_year.csv": ((100, 6), "1326aa61d76f7a6516d8d7cc9d8eb13654670493fdf2f5eaabcc1f9c498b055c", "fa9d1e06872737570de637b23c4786549828812c9d6c7d809ffbe75962a0199e"),
    "03_top_tracks_by_decade_window.csv": ((55, 6), "2de49b04e4d937614f524e105b9a95fdce322929ca592af428eaac6d85d8b513", "0e5358ddc146c74def109f2ecfec64cb1de18daa992c8112c869a72e9f40330a"),
    "04_above_average_popularity_subquery.csv": ((20, 5), "e1f0eb31e2f5349cfac59ff0bd029cb26db37e87601a6329dcfde22803b747cb", "e3731b1018861ac7600780637271b8e6e08f573ae4aec402eb4e98abf9f67d7e"),
    "05_top_artists_by_popularity.csv": ((20, 5), "992778112c48dfcc6c49f8789fa2cfd5b220f1ea380339fd9c07446cfe98e98b", "a2bea09767680258f4c8a95abcbad511076ccedc69c3993ce89863949398a2b0"),
    "06_top_genre_audio_profiles.csv": ((20, 7), "5ba16004f5008e6cec6be737d36b7713dcb2fe182d6c9a38608902b23ae5929e", "a9f99d4cd052563edd5900f8d3b29f4feeaa86ab539b3d839a8ffacf4d35fe21"),
}


@dataclass(frozen=True)
class SqlRun:
    paths: ProjectPaths
    database_path: Path
    create_return: Path
    run_return: None
    results: Mapping[str, pd.DataFrame]
    inputs_unchanged: bool


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sql_module() -> ModuleType:
    return importlib.import_module("src.sql_analysis")


@pytest.fixture(scope="session")
def pipeline_module() -> ModuleType:
    return importlib.import_module("src.pipeline")


@pytest.fixture(scope="session")
def project_module(
    project_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ModuleType, None, None]:
    previous_backend = os.environ.get("MPLBACKEND")
    previous_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(tmp_path_factory.mktemp("matplotlib-config"))
    try:
        module = importlib.import_module("spotify_week7_analysis")
        assert Path(module.__file__).resolve() == project_root / "spotify_week7_analysis.py"
        yield module
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
def canonical_project_data(project_root: Path) -> dict[str, pd.DataFrame]:
    data, _ = load_project_data(project_root)
    return data


@pytest.fixture(scope="session")
def canonical_processed_tracks(
    canonical_project_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    return clean_tracks_for_analysis(canonical_project_data["tracks"])


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sha256_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def repository_preservation(project_root: Path) -> Generator[None, None, None]:
    cleaned = project_root / "cleaned_data"
    outputs = project_root / "week7_outputs"
    before_cleaned = _sha256_manifest(cleaned)
    before_outputs = _sha256_manifest(outputs)
    yield
    assert _sha256_manifest(cleaned) == before_cleaned
    assert _sha256_manifest(outputs) == before_outputs


@pytest.fixture(scope="session")
def temporary_sql_run(
    project_module: ModuleType,
    sql_module: ModuleType,
    canonical_project_data: Mapping[str, pd.DataFrame],
    tmp_path_factory: pytest.TempPathFactory,
) -> SqlRun:
    root = tmp_path_factory.mktemp("canonical-sql-run")
    paths = project_module.make_paths(root, "outputs")
    before = {name: frame.copy(deep=True) for name, frame in canonical_project_data.items()}
    create_return = sql_module.create_sqlite_database(dict(canonical_project_data), paths)
    run_return = sql_module.run_sql_outputs(create_return, paths)
    results = {
        path.name: pd.read_csv(path) for path in sorted(paths.sql.glob("*.csv"))
    }
    inputs_unchanged = all(
        frame.equals(before[name]) for name, frame in canonical_project_data.items()
    )
    moved = create_return.with_suffix(".closed-check")
    create_return.rename(moved)
    moved.rename(create_return)
    return SqlRun(
        paths=paths,
        database_path=create_return,
        create_return=create_return,
        run_return=run_return,
        results=results,
        inputs_unchanged=inputs_unchanged,
    )


@pytest.fixture()
def synthetic_data() -> dict[str, pd.DataFrame]:
    tracks = pd.DataFrame(
        {
            "name": ["Echo", "Alpha", "Bravo", "Charlie", "Delta", "Foxtrot", "Golf", "Hotel", "Null Pop"],
            "artists": ["A", "B", "C", "D", "E", "F", "G", "H", "N"],
            "year": [1991, 1991, 1992, 1993, 1994, 1995, 2001, 2002, 2003],
            "popularity": [90.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, np.nan],
            "energy": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "danceability": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "acousticness": [0.2] * 9,
            "valence": [0.3] * 9,
        }
    )
    artists = pd.DataFrame(
        {
            "artists": ["Low Count", "Top Artist", "Second Artist"],
            "count": [4, 8, 5],
            "popularity": [99.0, 88.0, 77.0],
            "energy": [0.1, 0.9, 0.8],
            "danceability": [0.2, 0.7, 0.6],
        }
    )
    genres = pd.DataFrame(
        {
            "genres_clean": [None, "genre b", "genre a"],
            "popularity": [100.0, 80.0, 90.0],
            "energy": [0.1, 0.8, 0.9],
            "danceability": [0.1, 0.7, 0.8],
            "acousticness": [0.9, 0.2, 0.1],
            "valence": [0.1, 0.6, 0.7],
            "tempo": [80.0, 120.0, 130.0],
        }
    )
    return {"tracks": tracks, "artist_features": artists, "genre_features": genres}


def _run_synthetic(
    project_module: ModuleType,
    data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> tuple[ProjectPaths, dict[str, pd.DataFrame]]:
    paths = project_module.make_paths(tmp_path, "outputs")
    database = project_module.create_sqlite_database(data, paths)
    project_module.run_sql_outputs(database, paths)
    return paths, {
        path.name: pd.read_csv(path) for path in sorted(paths.sql.glob("*.csv"))
    }


def test_public_sql_function_identity(
    project_module: ModuleType,
    sql_module: ModuleType,
) -> None:
    assert project_module.create_sqlite_database is sql_module.create_sqlite_database
    assert project_module.run_sql_outputs is sql_module.run_sql_outputs
    assert project_module.SQL_QUERIES is sql_module.SQL_QUERIES


def test_sql_import_has_no_side_effects(project_root: Path, tmp_path: Path) -> None:
    script = """
import importlib, json, sqlite3
from pathlib import Path
calls=[]
sqlite3.connect=lambda *args, **kwargs: calls.append(args)
before=sorted(path.relative_to(Path.cwd()).as_posix() for path in Path.cwd().rglob('*'))
importlib.import_module('src.sql_analysis')
after=sorted(path.relative_to(Path.cwd()).as_posix() for path in Path.cwd().rglob('*'))
print(json.dumps({'before':before,'after':after,'calls':calls}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == '{"before": [], "after": [], "calls": []}'


def test_sql_dependency_boundary(project_root: Path, tmp_path: Path) -> None:
    script = """
import importlib,json,sys
before=set(sys.modules); importlib.import_module('src.sql_analysis'); introduced=set(sys.modules)-before
forbidden=['matplotlib','seaborn','plotly','sklearn','joblib','src.eda','src.visualization','src.regression','src.recommender','src.recommender_consumer','spotify_week7_analysis']
print(json.dumps(sorted(m for m in introduced if any(m==n or m.startswith(n+'.') for n in forbidden))))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "[]"


def test_query_manifest_and_baseline(
    temporary_sql_run: SqlRun,
    sql_module: ModuleType,
) -> None:
    assert list(sql_module.SQL_QUERIES) == QUERY_NAMES
    assert temporary_sql_run.create_return == temporary_sql_run.database_path
    assert temporary_sql_run.run_return is None
    assert temporary_sql_run.inputs_unchanged
    assert set(temporary_sql_run.results) == set(BASELINE)
    for filename, (shape, fingerprint, file_hash) in BASELINE.items():
        frame = temporary_sql_run.results[filename]
        assert frame.shape == shape
        assert _frame_fingerprint(frame) == fingerprint
        assert hashlib.sha256(
            (temporary_sql_run.paths.sql / filename).read_bytes()
        ).hexdigest() == file_hash
    assert hashlib.sha256(
        (temporary_sql_run.paths.sql / "spotify_week7_queries.sql").read_bytes()
    ).hexdigest() == "206b22aed89960ddd7002143af10a892d5c59fa277c4eea765690aa14df5de68"
    assert temporary_sql_run.database_path.is_file()


def test_canonical_sql_directory_is_historically_empty(project_root: Path) -> None:
    assert list((project_root / "week7_outputs" / "sql").iterdir()) == []


def test_canonical_data_preparation_contract(
    canonical_project_data: Mapping[str, pd.DataFrame],
    canonical_processed_tracks: pd.DataFrame,
) -> None:
    assert set(canonical_project_data) == {
        "tracks", "artist_features", "genre_features", "year_features", "artist_genres"
    }
    assert canonical_processed_tracks is not canonical_project_data["tracks"]
    assert len(canonical_processed_tracks) == len(canonical_project_data["tracks"])


def test_database_tables_and_loading(
    temporary_sql_run: SqlRun,
    canonical_project_data: Mapping[str, pd.DataFrame],
) -> None:
    with closing(sqlite3.connect(temporary_sql_run.database_path)) as conn:
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        assert tables == ["artist_features", "artist_genres", "genre_features", "tracks", "year_features"]
        for table in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            assert count == len(canonical_project_data[table])


def test_filter_group_aggregate_and_order(
    project_module: ModuleType,
    synthetic_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    _, results = _run_synthetic(project_module, synthetic_data, tmp_path)
    decades = results["01_tracks_by_decade.csv"]
    assert decades["decade"].tolist() == [1990, 2000]
    assert decades["total_tracks"].tolist() == [6, 3]
    assert decades.loc[0, "avg_popularity"] == pytest.approx(73.33)
    years = results["02_audio_features_by_year.csv"]
    assert years["year"].tolist() == sorted(
        synthetic_data["tracks"]["year"].unique().tolist()
    )
    artists = results["05_top_artists_by_popularity.csv"]
    assert artists["artist_name"].tolist() == ["Top Artist", "Second Artist"]
    genres = results["06_top_genre_audio_profiles.csv"]
    assert genres["genre_name"].tolist() == ["genre a", "genre b"]


def test_window_function_and_subquery(
    project_module: ModuleType,
    synthetic_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    _, results = _run_synthetic(project_module, synthetic_data, tmp_path)
    ranked = results["03_top_tracks_by_decade_window.csv"]
    group = ranked.loc[ranked["decade"] == 1990]
    assert group["track_name"].tolist() == ["Alpha", "Echo", "Bravo", "Charlie", "Delta"]
    assert group["rank_in_decade"].tolist() == [1, 2, 3, 4, 5]
    above = results["04_above_average_popularity_subquery.csv"]
    assert above["track_name"].tolist() == ["Alpha", "Echo", "Bravo", "Charlie"]
    assert "Null Pop" not in above["track_name"].tolist()
    assert above["popularity_gap"].tolist() == pytest.approx([26.25, 26.25, 16.25, 6.25])


def test_no_join_exists_in_current_query_contract(sql_module: ModuleType) -> None:
    assert all(" JOIN " not in f" {query.upper()} " for query in sql_module.SQL_QUERIES.values())


def test_optional_auxiliary_absence_preserves_failure(
    project_module: ModuleType,
    synthetic_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    paths = project_module.make_paths(tmp_path, "outputs")
    database = project_module.create_sqlite_database({"tracks": synthetic_data["tracks"]}, paths)
    with pytest.raises(pd.errors.DatabaseError, match="no such table: artist_features"):
        project_module.run_sql_outputs(database, paths)
    assert sorted(path.name for path in paths.sql.glob("*.csv")) == [
        f"{name}.csv" for name in QUERY_NAMES[:4]
    ]
    assert not (paths.sql / "spotify_week7_queries.sql").exists()


def test_empty_filtered_results_keep_schema(
    project_module: ModuleType,
    synthetic_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    data = dict(synthetic_data)
    data["artist_features"] = data["artist_features"].assign(count=0)
    data["genre_features"] = data["genre_features"].assign(genres_clean=None)
    _, results = _run_synthetic(project_module, data, tmp_path)
    assert results["05_top_artists_by_popularity.csv"].empty
    assert results["05_top_artists_by_popularity.csv"].columns.tolist() == [
        "artist_name", "total_tracks", "avg_popularity", "avg_energy", "avg_danceability"
    ]
    assert results["06_top_genre_audio_profiles.csv"].empty
    assert len(results["06_top_genre_audio_profiles.csv"].columns) == 7


def test_synthetic_inputs_are_immutable(
    project_module: ModuleType,
    synthetic_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    before = {name: frame.copy(deep=True) for name, frame in synthetic_data.items()}
    _run_synthetic(project_module, synthetic_data, tmp_path)
    for name, frame in synthetic_data.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_skip_sql_compatibility(
    project_module: ModuleType,
    pipeline_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracks = pd.DataFrame({"name": ["x"]})
    monkeypatch.setattr(pipeline_module, "load_project_data", lambda root: ({"tracks": tracks}, {}))
    monkeypatch.setattr(pipeline_module, "create_eda_outputs", lambda data, paths: tracks)
    monkeypatch.setattr(pipeline_module, "regression_analysis", lambda tracks, paths: {"models_trained": [], "best_model": {}, "plot_model": {}})
    monkeypatch.setattr(pipeline_module, "build_recommender_demo", lambda tracks, paths: {})
    monkeypatch.setattr(pipeline_module, "write_run_summary", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "create_sqlite_database",
        lambda *args, **kwargs: pytest.fail("SQL executed despite --skip-sql"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["spotify_week7_analysis.py", "--root", str(tmp_path), "--output", "out", "--skip-sql"],
    )
    project_module.main()
    assert "Skipping optional SQLite/SQL reference outputs" in capsys.readouterr().out
    assert list((tmp_path / "out" / "sql").iterdir()) == []
