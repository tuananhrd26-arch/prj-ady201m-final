"""Read-only characterization tests for persisted recommender consumption."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (
    RECOMMENDER_CATALOG_FILENAME,
    RECOMMENDER_FEATURES,
    RECOMMENDER_FEATURES_FILENAME,
    RECOMMENDER_NEIGHBORS_FILENAME,
    RECOMMENDER_SCALER_FILENAME,
)
from src.recommender_consumer import (
    CATALOG_COLUMNS,
    RECOMMENDATION_COLUMNS,
    RecommenderArtifacts,
    load_recommender_artifacts,
    recommend_from_artifacts,
    resolve_catalog_track,
)


ARTIFACT_FILENAMES = [
    RECOMMENDER_SCALER_FILENAME,
    RECOMMENDER_NEIGHBORS_FILENAME,
    RECOMMENDER_FEATURES_FILENAME,
    RECOMMENDER_CATALOG_FILENAME,
]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return manifest


def _repository_file_set(root: Path) -> set[str]:
    excluded = {".git", ".venv", ".pytest_cache", "__pycache__"}
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
    }


@pytest.fixture(scope="session", autouse=True)
def repository_preservation(project_root: Path) -> Generator[None, None, None]:
    cleaned = project_root / "cleaned_data"
    outputs = project_root / "week7_outputs"
    cleaned_before = _sha256_manifest(cleaned)
    outputs_before = _sha256_manifest(outputs)
    yield
    assert _sha256_manifest(cleaned) == cleaned_before
    assert _sha256_manifest(outputs) == outputs_before


@pytest.fixture(scope="session")
def canonical_artifacts(project_root: Path) -> RecommenderArtifacts:
    return load_recommender_artifacts(
        project_root / "week7_outputs" / "model_artifacts"
    )


@pytest.fixture(scope="session")
def temporary_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("consumer-project")
    artifact_dir = root / "week7_outputs" / "model_artifacts"
    artifact_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260716)
    rows = 36
    catalog = pd.DataFrame(
        {
            "_model_index": np.arange(rows),
            "id": [f"consumer-{index:03d}" for index in range(rows)],
            "name": [f"Consumer Track {index:03d}" for index in range(rows)],
            "artists": [f"['Consumer Artist {index:03d}']" for index in range(rows)],
            "year": rng.integers(1950, 2021, rows),
            "popularity": rng.uniform(0, 100, rows),
            "acousticness": rng.uniform(0, 1, rows),
            "danceability": rng.uniform(0, 1, rows),
            "energy": rng.uniform(0, 1, rows),
            "instrumentalness": rng.uniform(0, 1, rows),
            "liveness": rng.uniform(0, 1, rows),
            "loudness": rng.uniform(-35, -1, rows),
            "speechiness": rng.uniform(0, 0.8, rows),
            "tempo": rng.uniform(50, 210, rows),
            "valence": rng.uniform(0, 1, rows),
        }
    )
    catalog.loc[30, ["name", "artists"]] = catalog.loc[
        10, ["name", "artists"]
    ].to_numpy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(catalog[RECOMMENDER_FEATURES])
    neighbors = NearestNeighbors(n_neighbors=rows, metric="cosine").fit(matrix)
    joblib.dump(scaler, artifact_dir / RECOMMENDER_SCALER_FILENAME)
    joblib.dump(neighbors, artifact_dir / RECOMMENDER_NEIGHBORS_FILENAME)
    (artifact_dir / RECOMMENDER_FEATURES_FILENAME).write_text(
        json.dumps(RECOMMENDER_FEATURES, indent=2), encoding="utf-8"
    )
    catalog.to_csv(
        artifact_dir / RECOMMENDER_CATALOG_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )
    return root


@pytest.fixture(scope="session")
def temporary_artifacts(temporary_project: Path) -> RecommenderArtifacts:
    return load_recommender_artifacts(
        temporary_project / "week7_outputs" / "model_artifacts"
    )


def _run_cli(
    project_root: Path,
    arguments: list[str],
    *,
    module: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if module:
        command += ["-m", "scripts.recommend_song"]
    else:
        command += [str(project_root / "scripts" / "recommend_song.py")]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    return subprocess.run(
        command + arguments,
        cwd=cwd or project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_imports_have_no_side_effects(project_root: Path, tmp_path: Path) -> None:
    script = """
import importlib, json, sys
from pathlib import Path
before = sorted(path.relative_to(Path.cwd()).as_posix() for path in Path.cwd().rglob("*"))
importlib.import_module("src.recommender_consumer")
importlib.import_module("scripts.recommend_song")
after = sorted(path.relative_to(Path.cwd()).as_posix() for path in Path.cwd().rglob("*"))
print(json.dumps({"before": before, "after": after, "matplotlib": "matplotlib" in sys.modules}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "before": [],
        "after": [],
        "matplotlib": False,
    }


def test_consumer_dependency_boundary(project_root: Path, tmp_path: Path) -> None:
    script = """
import importlib, json, sys
before = set(sys.modules)
importlib.import_module("src.recommender_consumer")
importlib.import_module("scripts.recommend_song")
introduced = set(sys.modules) - before
forbidden = ["matplotlib", "seaborn", "plotly", "sqlite3", "src.eda", "src.visualization", "src.regression", "spotify_week7_analysis"]
loaded = sorted(module for module in introduced if any(module == name or module.startswith(name + ".") for name in forbidden))
print(json.dumps(loaded))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(project_root), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == []


def test_missing_artifact_reporting(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        load_recommender_artifacts(tmp_path)
    for filename in ARTIFACT_FILENAMES:
        assert filename in str(error.value)


def test_canonical_artifact_loading(canonical_artifacts: RecommenderArtifacts) -> None:
    artifacts = canonical_artifacts
    assert artifacts.features == tuple(RECOMMENDER_FEATURES)
    assert artifacts.catalog.shape == (170653, 15)
    assert artifacts.catalog.columns.tolist() == CATALOG_COLUMNS
    np.testing.assert_array_equal(
        artifacts.catalog["_model_index"], np.arange(170653)
    )
    assert artifacts.catalog["id"].is_unique
    assert np.isfinite(artifacts.catalog[RECOMMENDER_FEATURES].to_numpy()).all()
    assert artifacts.scaler.n_features_in_ == 9
    assert artifacts.neighbors.n_samples_fit_ == 170653
    assert artifacts.neighbors.n_features_in_ == 9
    assert artifacts.neighbors._fit_X.shape == (170653, 9)


def test_loading_and_querying_never_fit(
    temporary_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fit method called by consumer")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit_transform", forbidden)
    monkeypatch.setattr(NearestNeighbors, "fit", forbidden)
    artifacts = load_recommender_artifacts(
        temporary_project / "week7_outputs" / "model_artifacts"
    )
    result = recommend_from_artifacts(artifacts, model_index=0, top_n=5)
    assert len(result) == 5


def test_selector_validation(temporary_artifacts: RecommenderArtifacts) -> None:
    catalog = temporary_artifacts.catalog
    expected = catalog.iloc[0]
    assert resolve_catalog_track(catalog, track_id=expected["id"])["_model_index"] == 0
    assert resolve_catalog_track(catalog, model_index=0)["id"] == expected["id"]
    assert resolve_catalog_track(
        catalog, name=expected["name"], artists=expected["artists"]
    )["id"] == expected["id"]
    invalid = [
        {},
        {"track_id": expected["id"], "model_index": 0},
        {"name": expected["name"]},
        {"artists": expected["artists"]},
        {"model_index": True},
        {"model_index": -1},
    ]
    for selector in invalid:
        with pytest.raises(ValueError):
            resolve_catalog_track(catalog, **selector)
    for selector in [
        {"track_id": "unknown"},
        {"model_index": 9999},
        {"name": "unknown", "artists": "unknown"},
    ]:
        with pytest.raises(LookupError):
            resolve_catalog_track(catalog, **selector)
    duplicate = catalog.iloc[10]
    with pytest.raises(LookupError, match="ambiguous"):
        resolve_catalog_track(
            catalog, name=duplicate["name"], artists=duplicate["artists"]
        )


def test_selector_equivalence(temporary_artifacts: RecommenderArtifacts) -> None:
    row = temporary_artifacts.catalog.iloc[0]
    by_id = recommend_from_artifacts(
        temporary_artifacts, track_id=row["id"], top_n=10
    )
    by_index = recommend_from_artifacts(
        temporary_artifacts, model_index=0, top_n=10
    )
    by_pair = recommend_from_artifacts(
        temporary_artifacts,
        name=row["name"],
        artists=row["artists"],
        top_n=10,
    )
    pd.testing.assert_frame_equal(by_id, by_index)
    pd.testing.assert_frame_equal(by_id, by_pair)


def test_canonical_recommendation_equivalence(
    project_root: Path,
    canonical_artifacts: RecommenderArtifacts,
) -> None:
    expected = pd.read_csv(
        project_root / "week7_outputs" / "tables" / "recommendation_demo_results.csv"
    )
    for seed_index, group in expected.groupby("input_model_index", sort=False):
        actual = recommend_from_artifacts(
            canonical_artifacts, model_index=int(seed_index), top_n=10
        )
        np.testing.assert_array_equal(actual["rank"], group["rank"])
        np.testing.assert_array_equal(
            actual["recommended_model_index"], group["recommended_model_index"]
        )
        assert actual["recommended_name"].tolist() == group["recommended_song"].tolist()
        assert actual["recommended_artists"].tolist() == group["recommended_artists"].tolist()
        np.testing.assert_allclose(
            actual["cosine_distance"], group["cosine_distance"], rtol=0, atol=0.000051
        )
        np.testing.assert_allclose(
            actual["similarity"], group["similarity"], rtol=0, atol=0.000051
        )
        expected_ids = canonical_artifacts.catalog.iloc[
            group["recommended_model_index"].astype(int).to_numpy()
        ]["id"].astype(str).tolist()
        assert actual["recommended_id"].tolist() == expected_ids


def test_query_output_contract(temporary_artifacts: RecommenderArtifacts) -> None:
    result = recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=10)
    query = temporary_artifacts.catalog.iloc[0]
    assert result.columns.tolist() == RECOMMENDATION_COLUMNS
    assert result["rank"].tolist() == list(range(1, len(result) + 1))
    assert result["query_model_index"].eq(0).all()
    assert not result["recommended_model_index"].eq(0).any()
    assert not result["recommended_id"].eq(str(query["id"])).any()
    assert not (
        result["recommended_name"].eq(query["name"])
        & result["recommended_artists"].eq(query["artists"])
    ).any()
    assert result["recommended_model_index"].is_unique
    assert not result.duplicated(["recommended_name", "recommended_artists"]).any()
    assert np.isfinite(result[["cosine_distance", "similarity"]]).all().all()
    np.testing.assert_allclose(
        result["similarity"], 1 - result["cosine_distance"], rtol=0, atol=0
    )


@pytest.mark.parametrize("top_n", [1, 5, 10])
def test_top_n_counts(temporary_artifacts: RecommenderArtifacts, top_n: int) -> None:
    assert len(recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=top_n)) == top_n


@pytest.mark.parametrize("top_n", [0, -1, True, 1.5])
def test_invalid_top_n(temporary_artifacts: RecommenderArtifacts, top_n: Any) -> None:
    with pytest.raises(ValueError):
        recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=top_n)


def test_top_n_candidate_exhaustion(temporary_artifacts: RecommenderArtifacts) -> None:
    result = recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=100)
    assert 0 < len(result) < 100


def test_artifact_and_catalog_immutability(
    temporary_project: Path,
    temporary_artifacts: RecommenderArtifacts,
) -> None:
    artifact_dir = temporary_project / "week7_outputs" / "model_artifacts"
    before_files = _sha256_manifest(artifact_dir)
    before_catalog = temporary_artifacts.catalog.copy(deep=True)
    recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=10)
    assert _sha256_manifest(artifact_dir) == before_files
    pd.testing.assert_frame_equal(temporary_artifacts.catalog, before_catalog)


@pytest.mark.parametrize("module", [True, False], ids=["module", "direct-file"])
def test_cli_success(
    project_root: Path,
    temporary_project: Path,
    temporary_artifacts: RecommenderArtifacts,
    tmp_path: Path,
    module: bool,
) -> None:
    output = tmp_path / "created" / "recommendations.csv"
    before = _sha256_manifest(temporary_project)
    completed = _run_cli(
        project_root,
        [
            "--root", str(temporary_project), "--model-index", "0",
            "--top-n", "5", "--output", str(output),
            "--no-validate-alignment",
        ],
        module=module,
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr
    assert "Query:" in completed.stdout and "Output:" in completed.stdout
    assert output.is_file()
    written = pd.read_csv(output)
    expected = recommend_from_artifacts(temporary_artifacts, model_index=0, top_n=5)
    pd.testing.assert_frame_equal(written, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    after = _sha256_manifest(temporary_project)
    assert after == before
    assert list((tmp_path / "created").iterdir()) == [output]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--track-id", "consumer-000", "--model-index", "0"],
        ["--track-id", "unknown"],
        ["--model-index", "0", "--top-n", "0"],
    ],
)
def test_cli_expected_errors(
    project_root: Path,
    temporary_project: Path,
    tmp_path: Path,
    arguments: list[str],
) -> None:
    completed = _run_cli(
        project_root,
        ["--root", str(temporary_project), *arguments, "--no-validate-alignment"],
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert completed.stderr.strip()
    assert "Traceback" not in completed.stderr


def test_cli_missing_artifacts(project_root: Path, tmp_path: Path) -> None:
    completed = _run_cli(
        project_root,
        ["--root", str(tmp_path), "--model-index", "0"],
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "Missing recommender artifact files" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_canonical_cli_is_read_only(project_root: Path, tmp_path: Path) -> None:
    outputs = project_root / "week7_outputs"
    before = _sha256_manifest(outputs)
    repository_before = _repository_file_set(project_root)
    completed = _run_cli(
        project_root,
        [
            "--root", str(project_root), "--model-index", "19611",
            "--top-n", "1", "--no-validate-alignment",
        ],
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert "Query:" in completed.stdout and "Dakiti" in completed.stdout
    assert "Traceback" not in completed.stderr
    assert _sha256_manifest(outputs) == before
    assert _repository_file_set(project_root) == repository_before
