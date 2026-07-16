"""Deterministic characterization tests for the content recommender."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]
TOP_N = 10
SEED_COLUMNS = ["input_song", "input_artists", "input_model_index"]
RESULT_COLUMNS = [
    "input_song",
    "input_artists",
    "input_model_index",
    "rank",
    "recommended_song",
    "recommended_artists",
    "recommended_model_index",
    "year",
    "popularity",
    "cosine_distance",
    "similarity",
]
VALIDATION_BOOLEAN_COLUMNS = [
    "label_matches_query_vector",
    "input_track_absent",
    "no_duplicate_recommendations",
    "similarity_formula_valid",
    "finite_similarity",
    "rank_consecutive",
    "exact_top_n",
    "validation_passed",
]
ARTIFACT_NAMES = {
    "scaler": "recommender_scaler.joblib",
    "neighbors": "nearest_neighbors_recommender.joblib",
    "features": "recommender_features.json",
    "catalog": "recommender_catalog.csv",
}
CATALOG_COLUMNS = [
    "_model_index",
    "id",
    "name",
    "artists",
    "year",
    "popularity",
    *FEATURES,
]


@dataclass(frozen=True)
class RecommenderRun:
    summary: dict[str, Any]
    paths: Any
    results: pd.DataFrame
    validation: pd.DataFrame
    scaler: StandardScaler
    neighbors: NearestNeighbors
    features: list[str]
    catalog: pd.DataFrame


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_module(
    project_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ModuleType, None, None]:
    """Import the project without invoking main or using a user cache directory."""
    mpl_config = tmp_path_factory.mktemp("recommender-matplotlib-config")
    previous = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    module = importlib.import_module("spotify_week7_analysis")
    assert Path(module.__file__).resolve() == project_root / "spotify_week7_analysis.py"
    yield module
    if previous is None:
        os.environ.pop("MPLCONFIGDIR", None)
    else:
        os.environ["MPLCONFIGDIR"] = previous


@pytest.fixture(scope="session")
def canonical_recommender_outputs(project_root: Path) -> Mapping[str, Any]:
    output = project_root / "week7_outputs"
    tables = output / "tables"
    artifacts = output / "model_artifacts"
    with (output / "run_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (artifacts / ARTIFACT_NAMES["features"]).open(encoding="utf-8") as handle:
        features = json.load(handle)
    return {
        "results": pd.read_csv(tables / "recommendation_demo_results.csv"),
        "validation": pd.read_csv(tables / "recommendation_validation.csv"),
        "summary": summary,
        "scaler": joblib.load(artifacts / ARTIFACT_NAMES["scaler"]),
        "neighbors": joblib.load(artifacts / ARTIFACT_NAMES["neighbors"]),
        "features": features,
        "catalog": pd.read_csv(artifacts / ARTIFACT_NAMES["catalog"]),
        "artifact_dir": artifacts,
    }


@pytest.fixture(scope="session")
def synthetic_tracks() -> pd.DataFrame:
    """Create a varied catalog with deterministic identity and vector duplicates."""
    rng = np.random.default_rng(20260716)
    rows = 160
    tracks = pd.DataFrame(
        {
            "id": [f"synthetic-{index:03d}" for index in range(rows)],
            "name": [f"Track {index:03d}" for index in range(rows)],
            "artists": [f"['Artist {index:03d}']" for index in range(rows)],
            "popularity": rng.uniform(0.0, 75.0, rows),
            "year": rng.integers(1921, 2021, rows),
            "acousticness": rng.uniform(0.0, 1.0, rows),
            "danceability": rng.uniform(0.0, 1.0, rows),
            "energy": rng.uniform(0.0, 1.0, rows),
            "instrumentalness": rng.uniform(0.0, 1.0, rows),
            "liveness": rng.uniform(0.0, 1.0, rows),
            "loudness": rng.uniform(-35.0, -1.0, rows),
            "speechiness": rng.uniform(0.0, 0.8, rows),
            "tempo": rng.uniform(50.0, 210.0, rows),
            "valence": rng.uniform(0.0, 1.0, rows),
        }
    )
    tracks.loc[:24, "popularity"] = 100.0 - np.arange(25) * 0.5

    # Preserve unique ids while adding repeated metadata identities.
    for offset, row_index in enumerate(range(130, 140)):
        source_index = 40 + offset // 2
        tracks.loc[row_index, ["name", "artists"]] = tracks.loc[
            source_index, ["name", "artists"]
        ].to_numpy()

    # Identical feature vectors exercise tie handling without reducing identity supply.
    tracks.loc[100:109, FEATURES] = tracks.loc[60:69, FEATURES].to_numpy()

    assert tracks["id"].is_unique
    assert np.isfinite(tracks[FEATURES].to_numpy()).all()
    assert not tracks[["id", "name", "artists", "popularity", "year"] + FEATURES].isna().any().any()
    assert tracks[["name", "artists"]].drop_duplicates().shape[0] >= 140
    assert tracks.nlargest(20, "popularity")[["name", "artists"]].drop_duplicates().shape[0] == 20
    return tracks


@pytest.fixture(scope="session")
def insufficient_tracks(synthetic_tracks: pd.DataFrame) -> pd.DataFrame:
    """Retain four identities so Top 10 is impossible but inputs stay valid."""
    tracks = synthetic_tracks.iloc[:12].copy(deep=True).reset_index(drop=True)
    for index in range(len(tracks)):
        identity = index % 4
        tracks.loc[index, "name"] = f"Small Track {identity}"
        tracks.loc[index, "artists"] = f"['Small Artist {identity}']"
        tracks.loc[index, "popularity"] = 100.0 - index
    assert tracks[["name", "artists"]].drop_duplicates().shape[0] == 4
    assert tracks["id"].is_unique
    assert np.isfinite(tracks[FEATURES].to_numpy()).all()
    return tracks


def _sha256_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[path.relative_to(root).as_posix()] = digest.hexdigest()
    return manifest


@pytest.fixture(scope="session", autouse=True)
def repository_preservation(project_root: Path) -> Generator[None, None, None]:
    cleaned = project_root / "cleaned_data"
    outputs = project_root / "week7_outputs"
    cleaned_before = _sha256_manifest(cleaned)
    outputs_before = _sha256_manifest(outputs)
    yield
    assert _sha256_manifest(cleaned) == cleaned_before, "cleaned_data changed"
    assert _sha256_manifest(outputs) == outputs_before, "week7_outputs changed"


def _load_run(paths: Any, summary: dict[str, Any]) -> RecommenderRun:
    with (paths.model_artifacts / ARTIFACT_NAMES["features"]).open(
        encoding="utf-8"
    ) as handle:
        features = json.load(handle)
    return RecommenderRun(
        summary=summary,
        paths=paths,
        results=pd.read_csv(paths.tables / "recommendation_demo_results.csv"),
        validation=pd.read_csv(paths.tables / "recommendation_validation.csv"),
        scaler=joblib.load(paths.model_artifacts / ARTIFACT_NAMES["scaler"]),
        neighbors=joblib.load(paths.model_artifacts / ARTIFACT_NAMES["neighbors"]),
        features=features,
        catalog=pd.read_csv(paths.model_artifacts / ARTIFACT_NAMES["catalog"]),
    )


@pytest.fixture(scope="session")
def temporary_recommender_run(
    project_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> RecommenderRun:
    run_root = tmp_path_factory.mktemp("recommender-run")
    paths = project_module.make_paths(run_root, "outputs")
    summary = project_module.build_recommender_demo(
        synthetic_tracks.copy(deep=True),
        paths,
        n_examples=5,
        n_neighbors=TOP_N,
    )
    return _load_run(paths, summary)


@pytest.fixture(scope="session")
def insufficient_recommender_run(
    project_module: ModuleType,
    insufficient_tracks: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> RecommenderRun:
    run_root = tmp_path_factory.mktemp("insufficient-recommender-run")
    paths = project_module.make_paths(run_root, "outputs")
    summary = project_module.build_recommender_demo(
        insufficient_tracks.copy(deep=True),
        paths,
        n_examples=2,
        n_neighbors=TOP_N,
    )
    return _load_run(paths, summary)


def _catalog(tracks: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    required = ["name", "artists"] + features
    catalog = tracks.dropna(subset=required).copy().reset_index(drop=True)
    catalog["_model_index"] = catalog.index
    return catalog


def _normalize(value: Any) -> str:
    return str(value).strip().casefold()


def _identity(name: Any, artists: Any) -> tuple[str, str]:
    return _normalize(name), _normalize(artists)


def _seed_groups(results: pd.DataFrame) -> list[tuple[tuple[Any, ...], pd.DataFrame]]:
    return list(results.groupby(SEED_COLUMNS, sort=False, dropna=False))


def _assert_seed_alignment(
    results: pd.DataFrame,
    catalog: pd.DataFrame,
    scaler: StandardScaler,
    neighbors: NearestNeighbors,
    features: list[str],
) -> None:
    seeds = results[SEED_COLUMNS].drop_duplicates().reset_index(drop=True)
    indexes = seeds["input_model_index"].astype(int).to_numpy()
    assert ((indexes >= 0) & (indexes < len(catalog))).all()
    catalog_seeds = catalog.iloc[indexes]
    assert catalog_seeds["name"].astype(str).tolist() == seeds["input_song"].astype(str).tolist()
    assert catalog_seeds["artists"].astype(str).tolist() == seeds["input_artists"].astype(str).tolist()
    seed_vectors = scaler.transform(catalog_seeds[features])
    np.testing.assert_allclose(
        neighbors._fit_X[indexes],
        seed_vectors,
        rtol=0,
        atol=1e-12,
    )


def _assert_exclusion_and_uniqueness(
    results: pd.DataFrame,
    catalog: pd.DataFrame,
    expected_count: int,
) -> None:
    for (seed_name, seed_artists, seed_index), group in _seed_groups(results):
        seed_index = int(seed_index)
        seed_id = catalog.iloc[seed_index]["id"]
        seed_identity = _identity(seed_name, seed_artists)
        recommended_indexes = group["recommended_model_index"].astype(int)
        identities = [
            _identity(row.recommended_song, row.recommended_artists)
            for row in group.itertuples(index=False)
        ]
        assert len(group) == expected_count
        assert seed_index not in set(recommended_indexes)
        assert all(catalog.iloc[index]["id"] != seed_id for index in recommended_indexes)
        assert seed_identity not in identities
        assert len(set(identities)) == expected_count
        assert recommended_indexes.nunique() == expected_count


def _filtered_query(
    catalog: pd.DataFrame,
    scaler: StandardScaler,
    neighbors: NearestNeighbors,
    features: list[str],
    seed_index: int,
    count: int,
) -> tuple[list[int], list[float]]:
    matrix = scaler.transform(catalog[features])
    query = matrix[seed_index].reshape(1, -1)
    first_distances, first_indexes = neighbors.kneighbors(
        query, n_neighbors=len(catalog)
    )
    second_distances, second_indexes = neighbors.kneighbors(
        query, n_neighbors=len(catalog)
    )
    np.testing.assert_array_equal(first_indexes, second_indexes)
    np.testing.assert_allclose(first_distances, second_distances, rtol=0, atol=0)

    seed_pair = (str(catalog.iloc[seed_index]["name"]), str(catalog.iloc[seed_index]["artists"]))
    seen: set[tuple[str, str]] = set()
    selected_indexes: list[int] = []
    selected_distances: list[float] = []
    for distance, neighbor_index in zip(first_distances[0], first_indexes[0]):
        neighbor_index = int(neighbor_index)
        if neighbor_index == seed_index:
            continue
        pair = (
            str(catalog.iloc[neighbor_index]["name"]),
            str(catalog.iloc[neighbor_index]["artists"]),
        )
        if pair == seed_pair or pair in seen:
            continue
        seen.add(pair)
        selected_indexes.append(neighbor_index)
        selected_distances.append(float(distance))
        if len(selected_indexes) >= count:
            break
    return selected_indexes, selected_distances


def test_recommender_constants(project_module: ModuleType) -> None:
    assert project_module.RECOMMENDER_FEATURES == FEATURES


def test_canonical_recommendation_manifest(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    results = canonical_recommender_outputs["results"]
    assert not results.empty
    assert results.columns.tolist() == RESULT_COLUMNS
    assert results[SEED_COLUMNS].drop_duplicates().shape[0] == 5
    for _, group in _seed_groups(results):
        assert len(group) == TOP_N
        assert group["rank"].tolist() == list(range(1, TOP_N + 1))
    required = [
        "input_song",
        "input_artists",
        "input_model_index",
        "recommended_song",
        "recommended_artists",
        "recommended_model_index",
        "cosine_distance",
        "similarity",
    ]
    assert not results[required].isna().any().any()


def test_canonical_validation_manifest(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    results = canonical_recommender_outputs["results"]
    validation = canonical_recommender_outputs["validation"]
    assert len(validation) == 5
    assert validation[SEED_COLUMNS].drop_duplicates().shape[0] == 5
    existing_booleans = [column for column in validation if is_bool_dtype(validation[column])]
    assert existing_booleans
    assert validation[existing_booleans].all().all()
    assert validation["recommendations_returned"].eq(TOP_N).all()
    counts = results.groupby(SEED_COLUMNS).size()
    assert counts.eq(TOP_N).all()


def test_canonical_seed_vector_alignment(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    _assert_seed_alignment(
        canonical_recommender_outputs["results"],
        canonical_recommender_outputs["catalog"],
        canonical_recommender_outputs["scaler"],
        canonical_recommender_outputs["neighbors"],
        canonical_recommender_outputs["features"],
    )


def test_canonical_self_exclusion_and_uniqueness(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    _assert_exclusion_and_uniqueness(
        canonical_recommender_outputs["results"],
        canonical_recommender_outputs["catalog"],
        TOP_N,
    )


def test_canonical_similarity_relationship(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    results = canonical_recommender_outputs["results"]
    assert is_numeric_dtype(results["cosine_distance"])
    assert is_numeric_dtype(results["similarity"])
    assert np.isfinite(results[["cosine_distance", "similarity"]].to_numpy()).all()
    np.testing.assert_allclose(
        results["similarity"],
        1.0 - results["cosine_distance"],
        rtol=0,
        atol=1e-12,
    )
    assert results["similarity"].between(-1.0, 1.0, inclusive="both").all()


def test_canonical_artifact_structures(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    scaler = canonical_recommender_outputs["scaler"]
    neighbors = canonical_recommender_outputs["neighbors"]
    features = canonical_recommender_outputs["features"]
    catalog = canonical_recommender_outputs["catalog"]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(neighbors, NearestNeighbors)
    assert neighbors.metric == "cosine"
    assert scaler.n_features_in_ == len(FEATURES)
    assert neighbors.n_features_in_ == len(FEATURES)
    assert features == FEATURES
    if hasattr(scaler, "feature_names_in_"):
        assert scaler.feature_names_in_.tolist() == FEATURES
    assert neighbors.n_samples_fit_ == len(catalog)
    assert neighbors._fit_X.shape == (len(catalog), len(FEATURES))
    assert np.isfinite(neighbors._fit_X).all()


def test_row_aligned_catalog_artifact_exists(
    canonical_recommender_outputs: Mapping[str, Any],
) -> None:
    artifact_dir = canonical_recommender_outputs["artifact_dir"]
    path = artifact_dir / ARTIFACT_NAMES["catalog"]
    assert path.is_file()

    catalog = canonical_recommender_outputs["catalog"]
    scaler = canonical_recommender_outputs["scaler"]
    neighbors = canonical_recommender_outputs["neighbors"]
    features = canonical_recommender_outputs["features"]
    results = canonical_recommender_outputs["results"]
    required = ["id", "name", "artists", *FEATURES]

    assert not catalog.empty
    assert catalog.columns.tolist() == CATALOG_COLUMNS
    assert catalog.shape == (170653, len(CATALOG_COLUMNS))
    assert catalog["_model_index"].is_unique
    np.testing.assert_array_equal(
        catalog["_model_index"].to_numpy(), np.arange(len(catalog))
    )
    assert catalog["id"].is_unique
    assert not catalog[required].isna().any().any()
    assert np.isfinite(catalog[FEATURES].to_numpy(dtype=float)).all()
    assert features == FEATURES
    assert catalog.columns[-len(features):].tolist() == features
    assert len(catalog) == neighbors.n_samples_fit_
    transformed = scaler.transform(catalog[features])
    np.testing.assert_allclose(transformed, neighbors._fit_X, rtol=0, atol=1e-12)

    seeds = results[SEED_COLUMNS].drop_duplicates()
    assert len(seeds) == 5
    for seed in seeds.itertuples(index=False):
        catalog_row = catalog.iloc[int(seed.input_model_index)]
        assert str(catalog_row["name"]) == str(seed.input_song)
        assert str(catalog_row["artists"]) == str(seed.input_artists)


def test_temporary_output_manifest(temporary_recommender_run: RecommenderRun) -> None:
    required = [
        temporary_recommender_run.paths.tables / "recommendation_demo_results.csv",
        temporary_recommender_run.paths.tables / "recommendation_validation.csv",
        temporary_recommender_run.paths.model_artifacts / ARTIFACT_NAMES["scaler"],
        temporary_recommender_run.paths.model_artifacts / ARTIFACT_NAMES["neighbors"],
        temporary_recommender_run.paths.model_artifacts / ARTIFACT_NAMES["features"],
        temporary_recommender_run.paths.model_artifacts / ARTIFACT_NAMES["catalog"],
    ]
    for path in required:
        assert path.is_file(), f"Missing temporary recommender output: {path.name}"
        assert path.stat().st_size > 0, f"Empty temporary recommender output: {path.name}"
    assert temporary_recommender_run.features == FEATURES
    assert Path(temporary_recommender_run.summary["catalog_file"]) == (
        temporary_recommender_run.paths.model_artifacts / ARTIFACT_NAMES["catalog"]
    )
    assert temporary_recommender_run.summary["catalog_rows"] == len(
        temporary_recommender_run.catalog
    )


def test_temporary_seed_vector_alignment(
    synthetic_tracks: pd.DataFrame,
    temporary_recommender_run: RecommenderRun,
) -> None:
    expected_catalog = _catalog(synthetic_tracks, FEATURES)[CATALOG_COLUMNS]
    pd.testing.assert_frame_equal(
        temporary_recommender_run.catalog[
            ["_model_index", "id", "name", "artists"]
        ],
        expected_catalog[["_model_index", "id", "name", "artists"]],
        check_exact=True,
    )
    np.testing.assert_allclose(
        temporary_recommender_run.catalog[["year", "popularity", *FEATURES]],
        expected_catalog[["year", "popularity", *FEATURES]],
        rtol=0,
        atol=1e-12,
    )
    _assert_seed_alignment(
        temporary_recommender_run.results,
        temporary_recommender_run.catalog,
        temporary_recommender_run.scaler,
        temporary_recommender_run.neighbors,
        temporary_recommender_run.features,
    )


def test_temporary_exclusion_uniqueness_and_top_n(
    temporary_recommender_run: RecommenderRun,
) -> None:
    _assert_exclusion_and_uniqueness(
        temporary_recommender_run.results,
        temporary_recommender_run.catalog,
        TOP_N,
    )
    validation = temporary_recommender_run.validation
    assert len(validation) == 5
    assert validation["recommendations_returned"].eq(TOP_N).all()
    assert validation["exact_top_n"].all()
    assert validation["validation_passed"].all()
    for _, group in _seed_groups(temporary_recommender_run.results):
        assert group["rank"].tolist() == list(range(1, TOP_N + 1))
    assert temporary_recommender_run.summary["validation_passed"] is True


def test_temporary_similarity_and_validation_fields(
    temporary_recommender_run: RecommenderRun,
) -> None:
    results = temporary_recommender_run.results
    validation = temporary_recommender_run.validation
    assert np.isfinite(results[["cosine_distance", "similarity"]].to_numpy()).all()
    np.testing.assert_allclose(
        results["similarity"],
        1.0 - results["cosine_distance"],
        rtol=0,
        atol=1e-12,
    )
    assert validation[VALIDATION_BOOLEAN_COLUMNS].all().all()

    catalog = temporary_recommender_run.catalog
    for (seed_name, seed_artists, seed_index), group in _seed_groups(results):
        identities = {
            _identity(row.recommended_song, row.recommended_artists)
            for row in group.itertuples(index=False)
        }
        assert str(catalog.iloc[int(seed_index)]["name"]) == str(seed_name)
        assert str(catalog.iloc[int(seed_index)]["artists"]) == str(seed_artists)
        assert int(seed_index) not in set(group["recommended_model_index"].astype(int))
        assert _identity(seed_name, seed_artists) not in identities
        assert len(identities) == TOP_N
        assert len(group) == TOP_N


def test_artifact_reload_reconstructs_saved_queries(
    temporary_recommender_run: RecommenderRun,
) -> None:
    catalog = temporary_recommender_run.catalog
    assert catalog.columns.tolist() == CATALOG_COLUMNS
    np.testing.assert_array_equal(
        catalog["_model_index"].to_numpy(), np.arange(len(catalog))
    )
    assert len(catalog) == temporary_recommender_run.neighbors.n_samples_fit_
    scaled = temporary_recommender_run.scaler.transform(catalog[temporary_recommender_run.features])
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(
        scaled,
        temporary_recommender_run.neighbors._fit_X,
        rtol=0,
        atol=1e-12,
    )
    for (_, _, seed_index), group in _seed_groups(temporary_recommender_run.results):
        seed_index = int(seed_index)
        expected_indexes, expected_distances = _filtered_query(
            catalog,
            temporary_recommender_run.scaler,
            temporary_recommender_run.neighbors,
            temporary_recommender_run.features,
            seed_index,
            TOP_N,
        )
        np.testing.assert_array_equal(
            group["recommended_model_index"].astype(int).to_numpy(), expected_indexes
        )
        np.testing.assert_allclose(
            group["cosine_distance"].to_numpy(),
            expected_distances,
            rtol=0,
            atol=0.000051,
        )


def test_recommender_is_repeatable(
    project_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    temporary_recommender_run: RecommenderRun,
    tmp_path: Path,
) -> None:
    second_paths = project_module.make_paths(tmp_path, "repeat-output")
    second_summary = project_module.build_recommender_demo(
        synthetic_tracks.copy(deep=True),
        second_paths,
        n_examples=5,
        n_neighbors=TOP_N,
    )
    second = _load_run(second_paths, second_summary)
    pd.testing.assert_frame_equal(second.results, temporary_recommender_run.results)
    pd.testing.assert_frame_equal(second.validation, temporary_recommender_run.validation)
    pd.testing.assert_frame_equal(second.catalog, temporary_recommender_run.catalog)
    assert second.features == temporary_recommender_run.features
    np.testing.assert_allclose(second.scaler.mean_, temporary_recommender_run.scaler.mean_)
    np.testing.assert_allclose(second.scaler.scale_, temporary_recommender_run.scaler.scale_)
    assert second.neighbors._fit_X.shape == temporary_recommender_run.neighbors._fit_X.shape
    path_keys = {"validation_file", "catalog_file"}
    second_semantics = {
        key: value for key, value in second.summary.items() if key not in path_keys
    }
    first_semantics = {
        key: value
        for key, value in temporary_recommender_run.summary.items()
        if key not in path_keys
    }
    assert second_semantics == first_semantics
    assert Path(second.summary["validation_file"]).name == "recommendation_validation.csv"
    assert Path(second.summary["catalog_file"]).name == ARTIFACT_NAMES["catalog"]


def test_insufficient_catalog_is_not_validated_as_top_n(
    insufficient_recommender_run: RecommenderRun,
) -> None:
    results = insufficient_recommender_run.results
    validation = insufficient_recommender_run.validation
    assert not validation.empty
    assert validation["recommendations_returned"].lt(TOP_N).all()
    assert not validation["exact_top_n"].any()
    assert not validation["validation_passed"].any()
    assert insufficient_recommender_run.summary["validation_passed"] is False
    for seed, row in validation.set_index(SEED_COLUMNS).iterrows():
        group = results.loc[
            (results["input_song"] == seed[0])
            & (results["input_artists"] == seed[1])
            & (results["input_model_index"] == seed[2])
        ]
        assert len(group) == int(row["recommendations_returned"])
        assert group["rank"].tolist() == list(range(1, len(group) + 1))
