"""Characterization tests for the authoritative cleaned Spotify datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype


REQUIRED_CLEANED_FILES = {
    "data_clean.csv",
    "data_by_artist_clean.csv",
    "data_by_genres_clean.csv",
    "data_by_year_clean.csv",
    "data_w_genres_clean.csv",
    "cleaning_report.json",
    "feature_selection_report.json",
}

EXPECTED_SHAPES: dict[str, tuple[int, int]] = {
    "data_clean.csv": (170653, 20),
    "data_by_artist_clean.csv": (28680, 15),
    "data_by_genres_clean.csv": (2973, 15),
    "data_by_year_clean.csv": (100, 14),
    "data_w_genres_clean.csv": (28680, 17),
}

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "data_clean.csv": [
        "valence",
        "year",
        "acousticness",
        "artists",
        "danceability",
        "duration_ms",
        "energy",
        "explicit",
        "id",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "name",
        "popularity",
        "release_date",
        "speechiness",
        "tempo",
        "release_date_parsed",
    ],
    "data_by_artist_clean.csv": [
        "mode",
        "count",
        "acousticness",
        "artists",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "popularity",
        "key",
    ],
    "data_by_genres_clean.csv": [
        "mode",
        "genres",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "popularity",
        "key",
        "genres_clean",
    ],
    "data_by_year_clean.csv": [
        "mode",
        "year",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "popularity",
        "key",
    ],
    "data_w_genres_clean.csv": [
        "genres",
        "artists",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "popularity",
        "key",
        "mode",
        "count",
        "genres_clean",
    ],
}

EXPECTED_AUXILIARY_MISSING: dict[str, dict[str, int]] = {
    "data_by_artist_clean.csv": {},
    "data_by_genres_clean.csv": {"genres_clean": 1},
    "data_by_year_clean.csv": {},
    "data_w_genres_clean.csv": {"genres_clean": 9857},
}

UNIT_INTERVAL_FEATURES = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Locate the repository without depending on a machine-specific path."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cleaned_data_dir(project_root: Path) -> Path:
    return project_root / "cleaned_data"


@pytest.fixture(scope="session")
def loaded_datasets(cleaned_data_dir: Path) -> Mapping[str, pd.DataFrame]:
    """Load each CSV once so characterization remains fast and deterministic."""
    return {
        filename: pd.read_csv(cleaned_data_dir / filename, low_memory=False)
        for filename in EXPECTED_SHAPES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def cleaned_files_are_read_only(cleaned_data_dir: Path) -> Generator[None, None, None]:
    """Prove that the entire test session leaves every cleaned file unchanged."""
    before = {
        path.name: _sha256(path)
        for path in sorted(cleaned_data_dir.iterdir())
        if path.is_file()
    }
    yield
    after = {
        path.name: _sha256(path)
        for path in sorted(cleaned_data_dir.iterdir())
        if path.is_file()
    }
    assert after == before, "The test session changed one or more cleaned-data files"


def test_required_cleaned_files_exist(cleaned_data_dir: Path) -> None:
    missing = [
        filename
        for filename in sorted(REQUIRED_CLEANED_FILES)
        if not (cleaned_data_dir / filename).is_file()
    ]
    assert not missing, f"Missing required cleaned-data files: {missing}"


def test_cleaned_csv_manifest_is_exact(cleaned_data_dir: Path) -> None:
    actual = {path.name for path in cleaned_data_dir.glob("*.csv") if path.is_file()}
    expected = set(EXPECTED_SHAPES)
    assert actual == expected, f"CSV manifest differs: expected {expected}, got {actual}"


@pytest.mark.parametrize(("filename", "expected_shape"), EXPECTED_SHAPES.items())
def test_dataset_shape(
    filename: str,
    expected_shape: tuple[int, int],
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    actual_shape = loaded_datasets[filename].shape
    assert actual_shape == expected_shape, (
        f"{filename} shape changed: expected {expected_shape}, got {actual_shape}"
    )


@pytest.mark.parametrize(("filename", "expected_columns"), EXPECTED_COLUMNS.items())
def test_exact_column_order(
    filename: str,
    expected_columns: list[str],
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    actual_columns = loaded_datasets[filename].columns.tolist()
    assert actual_columns == expected_columns, (
        f"{filename} column order changed: expected {expected_columns}, "
        f"got {actual_columns}"
    )


@pytest.mark.parametrize(
    "report_name",
    ("cleaning_report.json", "feature_selection_report.json"),
)
def test_json_report_is_valid(cleaned_data_dir: Path, report_name: str) -> None:
    with (cleaned_data_dir / report_name).open(encoding="utf-8") as handle:
        report = json.load(handle)
    assert isinstance(report, dict), f"{report_name} must contain a JSON object"
    assert report, f"{report_name} must not be empty"


def test_main_missing_value_profile(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    missing = loaded_datasets["data_clean.csv"].isna().sum().to_dict()
    expected = {column: 0 for column in EXPECTED_COLUMNS["data_clean.csv"]}
    expected["release_date_parsed"] = 119798
    assert missing == expected, f"Main missing-value profile changed: {missing}"


@pytest.mark.parametrize(
    ("filename", "expected_nonzero"),
    EXPECTED_AUXILIARY_MISSING.items(),
)
def test_auxiliary_missing_value_profile(
    filename: str,
    expected_nonzero: dict[str, int],
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    actual = {
        column: int(count)
        for column, count in loaded_datasets[filename].isna().sum().items()
        if count
    }
    assert actual == expected_nonzero, (
        f"{filename} nonzero missing-value counts changed: {actual}"
    )


def test_full_row_duplicate_behavior(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    duplicate_counts = {
        filename: int(frame.duplicated().sum())
        for filename, frame in loaded_datasets.items()
    }
    assert duplicate_counts == {filename: 0 for filename in EXPECTED_SHAPES}, (
        f"Full-row duplicate counts changed: {duplicate_counts}"
    )


def test_main_id_behavior(loaded_datasets: Mapping[str, pd.DataFrame]) -> None:
    ids = loaded_datasets["data_clean.csv"]["id"]
    non_null_ids = ids.loc[ids.notna()]
    assert not ids.isna().any(), "Main id contains missing values"
    assert ids.is_unique, "Main id is no longer unique"
    assert int(non_null_ids.duplicated().sum()) == 0, "Non-null ids are duplicated"
    assert ids.astype(str).str.strip().str.len().gt(0).all(), "Main id contains blanks"


def test_main_name_and_artists_are_valid(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    main = loaded_datasets["data_clean.csv"]
    for column in ("name", "artists"):
        values = main[column]
        assert not values.isna().any(), f"Main {column} contains missing values"
        assert values.astype(str).str.strip().str.len().gt(0).all(), (
            f"Main {column} contains blank values"
        )


def test_name_artist_duplicate_characterization(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    main = loaded_datasets["data_clean.csv"]
    subset = ["name", "artists"]

    # keep="first" counts only extra occurrences after one group representative.
    extra_duplicate_rows = int(main.duplicated(subset=subset, keep="first").sum())
    # keep=False counts every row that belongs to a duplicated identity group.
    participating_mask = main.duplicated(subset=subset, keep=False)
    all_participating_rows = int(participating_mask.sum())
    duplicated_groups = int(
        main.loc[participating_mask, subset].drop_duplicates().shape[0]
    )

    assert extra_duplicate_rows == 12968
    assert all_participating_rows == 24001
    assert duplicated_groups == 11033
    assert all_participating_rows - duplicated_groups == extra_duplicate_rows


def test_popularity_domain(loaded_datasets: Mapping[str, pd.DataFrame]) -> None:
    popularity = loaded_datasets["data_clean.csv"]["popularity"]
    assert is_numeric_dtype(popularity), "Popularity must remain numeric"
    assert np.isfinite(popularity.to_numpy()).all(), "Popularity contains non-finite values"
    assert popularity.between(0, 100, inclusive="both").all(), (
        "Popularity must remain within [0, 100]"
    )


@pytest.mark.parametrize("feature", UNIT_INTERVAL_FEATURES)
def test_unit_interval_audio_feature(
    feature: str,
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    values = loaded_datasets["data_clean.csv"][feature]
    assert is_numeric_dtype(values), f"{feature} must remain numeric"
    assert np.isfinite(values.to_numpy()).all(), f"{feature} contains non-finite values"
    assert values.between(0, 1, inclusive="both").all(), (
        f"{feature} must remain within [0, 1]"
    )


def test_binary_and_categorical_domains(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    main = loaded_datasets["data_clean.csv"]
    assert set(main["explicit"].unique()).issubset({0, 1})
    assert set(main["mode"].unique()).issubset({0, 1})

    key = main["key"]
    assert is_numeric_dtype(key), "Key must remain numeric"
    assert np.isfinite(key.to_numpy()).all(), "Key contains non-finite values"
    assert key.eq(key.round()).all(), "Key values must be integer-equivalent"
    assert key.between(0, 11, inclusive="both").all(), "Key must remain within [0, 11]"


def test_year_domain(loaded_datasets: Mapping[str, pd.DataFrame]) -> None:
    year = loaded_datasets["data_clean.csv"]["year"]
    assert is_numeric_dtype(year), "Year must remain numeric"
    assert np.isfinite(year.to_numpy()).all(), "Year contains non-finite values"
    assert year.eq(year.round()).all(), "Year values must be integer-equivalent"
    assert int(year.min()) == 1921
    assert int(year.max()) == 2020


def test_continuous_numeric_finiteness(
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    main = loaded_datasets["data_clean.csv"]
    for feature in ("tempo", "loudness", "duration_ms"):
        values = main[feature]
        assert is_numeric_dtype(values), f"{feature} must remain numeric"
        assert np.isfinite(values.to_numpy()).all(), (
            f"{feature} contains non-finite values"
        )
    assert main["duration_ms"].gt(0).all(), "Duration must remain strictly positive"
    assert main["tempo"].ge(0).all(), "Tempo must remain non-negative"


def _explicit_schema_lists(value: Any, path: str = "root") -> dict[str, list[str]]:
    """Find only report lists whose keys clearly describe data columns/features."""
    supported_keys = {
        "candidate_numeric_features",
        "selected_numeric_features",
        "features",
        "feature_names",
        "columns",
        "column_names",
        "main_dataset_columns",
        "main_columns",
    }
    found: dict[str, list[str]] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if (
                key in supported_keys
                and isinstance(nested, list)
                and all(isinstance(item, str) for item in nested)
            ):
                found[nested_path] = nested
            elif isinstance(nested, (dict, list)):
                found.update(_explicit_schema_lists(nested, nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list)):
                found.update(_explicit_schema_lists(nested, f"{path}[{index}]"))
    return found


def test_feature_selection_report_matches_main_schema(
    cleaned_data_dir: Path,
    loaded_datasets: Mapping[str, pd.DataFrame],
) -> None:
    with (cleaned_data_dir / "feature_selection_report.json").open(
        encoding="utf-8"
    ) as handle:
        report = json.load(handle)

    main_columns = set(loaded_datasets["data_clean.csv"].columns)
    for report_path, reported_columns in _explicit_schema_lists(report).items():
        missing = sorted(set(reported_columns) - main_columns)
        assert not missing, (
            f"Feature report field {report_path} references unknown main columns: {missing}"
        )
