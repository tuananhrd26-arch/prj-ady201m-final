"""Characterization tests for the authoritative cleaned Spotify datasets."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from src.config import (
    RECOMMENDER_FEATURES,
    REGRESSION_AUDIO_FEATURES,
    REGRESSION_EXTENDED_FEATURES,
    REGRESSION_FEATURES,
    TARGET,
)
from src.data_loader import load_project_data, read_csv_if_exists
from src.preprocessing import clean_tracks_for_analysis


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

LOADER_PATHS = {
    "tracks": Path("cleaned_data") / "data_clean.csv",
    "artist_features": Path("cleaned_data") / "data_by_artist_clean.csv",
    "genre_features": Path("cleaned_data") / "data_by_genres_clean.csv",
    "year_features": Path("cleaned_data") / "data_by_year_clean.csv",
    "artist_genres": Path("cleaned_data") / "data_w_genres_clean.csv",
}

NUMERIC_CONVERSION_COLUMNS = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "explicit",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "popularity",
    "speechiness",
    "tempo",
    "valence",
    "year",
]

CANONICAL_PROCESSED_COLUMNS = EXPECTED_COLUMNS["data_clean.csv"] + ["decade"]
CANONICAL_PROCESSED_DTYPES = {
    "valence": "float64",
    "year": "int64",
    "acousticness": "float64",
    "artists": "str",
    "danceability": "float64",
    "duration_ms": "int64",
    "energy": "float64",
    "explicit": "int64",
    "id": "str",
    "instrumentalness": "float64",
    "key": "float64",
    "liveness": "float64",
    "loudness": "float64",
    "mode": "float64",
    "name": "str",
    "popularity": "int64",
    "release_date": "str",
    "speechiness": "float64",
    "tempo": "float64",
    "release_date_parsed": "str",
    "decade": "int64",
}
CANONICAL_PROCESSED_FINGERPRINT = (
    "3ccbd1356da53ff61ea16bacebd9058b98fc8c19712d30c27218932ab7e1e52f"
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


@pytest.fixture(scope="session")
def project_loader_result(
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load the authoritative project collection once through the public loader."""
    return load_project_data(project_root)


@pytest.fixture(scope="session")
def canonical_processed_tracks(
    project_loader_result: tuple[dict[str, pd.DataFrame], dict[str, Path]],
) -> pd.DataFrame:
    """Prepare the canonical tracks once for preprocessing characterization."""
    tracks = project_loader_result[0]["tracks"]
    return clean_tracks_for_analysis(tracks.copy(deep=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _dataframe_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values
    return hashlib.sha256(values.tobytes()).hexdigest()


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


def test_read_csv_if_exists_loads_existing_file_without_modification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "optional.csv"
    expected = pd.DataFrame({"name": ["Alpha", "Beta"], "value": [1, 2]})
    expected.to_csv(source, index=False)
    before = _sha256(source)

    actual = read_csv_if_exists(source)

    assert actual is not None
    pd.testing.assert_frame_equal(actual, expected)
    assert _sha256(source) == before


def test_read_csv_if_exists_returns_none_without_creating_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing" / "optional.csv"
    assert not source.parent.exists()

    assert read_csv_if_exists(source) is None

    assert not source.exists()
    assert not source.parent.exists()


def test_load_project_data_authoritative_contract(
    project_root: Path,
    project_loader_result: tuple[dict[str, pd.DataFrame], dict[str, Path]],
) -> None:
    data, input_files = project_loader_result
    expected_keys = list(LOADER_PATHS)
    loader_shapes = {
        key: EXPECTED_SHAPES[path.name]
        for key, path in LOADER_PATHS.items()
    }

    assert list(data) == expected_keys
    assert list(input_files) == expected_keys
    assert {key: frame.shape for key, frame in data.items()} == loader_shapes
    for key, relative_path in LOADER_PATHS.items():
        expected_path = project_root / relative_path
        assert input_files[key] == expected_path
        assert input_files[key].resolve() == expected_path.resolve()
    assert data["tracks"].shape == (170653, 20)
    assert data["tracks"].columns.tolist() == EXPECTED_COLUMNS["data_clean.csv"]
    assert "decade" not in data["tracks"].columns


def test_load_project_data_temporary_cleaned_layout(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    main = pd.DataFrame({"id": ["track-1"], "name": ["Track One"]})
    artists = pd.DataFrame({"artists": ["Artist One"], "count": [1]})
    years = pd.DataFrame({"year": [2001], "popularity": [50]})
    main.to_csv(cleaned / "data_clean.csv", index=False)
    artists.to_csv(cleaned / "data_by_artist_clean.csv", index=False)
    years.to_csv(cleaned / "data_by_year_clean.csv", index=False)

    data, input_files = load_project_data(tmp_path)

    assert list(data) == ["tracks", "artist_features", "year_features"]
    assert list(input_files) == ["tracks", "artist_features", "year_features"]
    pd.testing.assert_frame_equal(data["tracks"], main)
    pd.testing.assert_frame_equal(data["artist_features"], artists)
    pd.testing.assert_frame_equal(data["year_features"], years)
    assert input_files == {
        "tracks": cleaned / "data_clean.csv",
        "artist_features": cleaned / "data_by_artist_clean.csv",
        "year_features": cleaned / "data_by_year_clean.csv",
    }
    assert "genre_features" not in data
    assert "artist_genres" not in data


def test_load_project_data_uses_fallback_main_dataset(tmp_path: Path) -> None:
    fallback_dir = tmp_path / "data"
    fallback_dir.mkdir()
    fallback = fallback_dir / "data.csv"
    expected = pd.DataFrame({"id": ["fallback-1"], "name": ["Fallback Track"]})
    expected.to_csv(fallback, index=False)

    data, input_files = load_project_data(tmp_path)

    assert list(data) == ["tracks"]
    assert list(input_files) == ["tracks"]
    pd.testing.assert_frame_equal(data["tracks"], expected)
    assert input_files["tracks"] == fallback


def test_load_project_data_raises_when_main_dataset_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"cleaned_data/data_clean\.csv or data/data\.csv",
    ):
        load_project_data(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_load_project_data_has_no_write_side_effects(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned_data"
    cleaned.mkdir()
    pd.DataFrame({"id": ["track-1"], "name": ["Track One"]}).to_csv(
        cleaned / "data_clean.csv",
        index=False,
    )
    pd.DataFrame({"artists": ["Artist One"]}).to_csv(
        cleaned / "data_by_artist_clean.csv",
        index=False,
    )
    before_files = _tree_manifest(tmp_path)
    before_dirs = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_dir()
    }

    load_project_data(tmp_path)

    assert _tree_manifest(tmp_path) == before_files
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_dir()
    } == before_dirs
    assert not (tmp_path / "week7_outputs").exists()


def test_loader_functions_remain_public_through_monolith(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mpl_config = tmp_path / "matplotlib-config"
    mpl_config.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config))
    public = importlib.import_module("spotify_week7_analysis")

    assert public.load_project_data is load_project_data
    assert public.read_csv_if_exists is read_csv_if_exists


def test_preprocessing_function_remains_public_through_monolith() -> None:
    public = importlib.import_module("spotify_week7_analysis")

    assert public.clean_tracks_for_analysis is clean_tracks_for_analysis


def test_preprocessing_does_not_mutate_input_dataframe() -> None:
    source = pd.DataFrame(
        {
            "id": ["a", "b", None, "d"],
            "name": ["Alpha", "Beta", "Removed", "Delta"],
            "artists": ["Artist A", "Artist B", "Artist C", None],
            "year": ["1921", "1930", "1940", "2000"],
            "energy": ["0.1", None, "bad", "0.8"],
            "popularity": [10, 20, 30, None],
        },
        index=[10, 30, 50, 90],
    )
    before = source.copy(deep=True)

    result = clean_tracks_for_analysis(source)

    pd.testing.assert_frame_equal(source, before)
    assert result is not source


def test_preprocessing_numeric_conversion_contract() -> None:
    values: dict[str, list[object]] = {
        column: ["1", "bad", "3"] for column in NUMERIC_CONVERSION_COLUMNS
    }
    values["year"] = ["1921", "bad", "2020"]
    source = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["Alpha", "Beta", "Gamma"],
            **values,
            "unprocessed_text": ["1", "bad", "3"],
        }
    )
    expected_candidates = sorted(
        set(REGRESSION_FEATURES + RECOMMENDER_FEATURES + [TARGET])
    )

    result = clean_tracks_for_analysis(source)

    assert expected_candidates == NUMERIC_CONVERSION_COLUMNS
    for column in NUMERIC_CONVERSION_COLUMNS:
        assert is_numeric_dtype(result[column]), column
    for column in set(NUMERIC_CONVERSION_COLUMNS) - {"year"}:
        assert result.loc[1, column] == pytest.approx(2.0)
    assert result["year"].tolist() == [1921.0, 1970.5, 2020.0]
    assert result["decade"].tolist() == [1920, 1970, 2020]
    assert result["unprocessed_text"].tolist() == ["1", "bad", "3"]


def test_preprocessing_identity_filtering_contract() -> None:
    source = pd.DataFrame(
        {
            "id": ["valid", None, "missing-name", "no-artists", "", "blank-name"],
            "name": ["Valid", "Missing ID", None, "No Artists", "Blank ID", ""],
            "artists": ["Artist", "Artist", "Artist", None, "Artist", "Artist"],
            "year": [2000, 2001, 2002, 2003, 2004, 2005],
        }
    )

    result = clean_tracks_for_analysis(source)

    assert result["id"].tolist() == ["valid", "no-artists", "", "blank-name"]
    assert result["name"].tolist() == ["Valid", "No Artists", "Blank ID", ""]
    assert result["artists"].isna().sum() == 1


def test_preprocessing_median_fill_occurs_after_identity_filtering() -> None:
    source = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", None],
            "name": ["A", "B", "C", "D", "Removed"],
            "year": [2000, 2001, 2002, 2003, 2004],
            "energy": [1.0, 3.0, np.nan, np.nan, 100.0],
            "popularity": [10.0, 30.0, np.nan, np.nan, 100.0],
            "custom_numeric": [1.0, 3.0, np.nan, np.nan, 100.0],
            "odd_numeric": [1.0, 5.0, 9.0, np.nan, 100.0],
            "all_missing_numeric": [np.nan, np.nan, np.nan, np.nan, 100.0],
            "note": ["one", "two", None, "four", "removed"],
        }
    )

    result = clean_tracks_for_analysis(source)

    assert result["energy"].tolist() == [1.0, 3.0, 2.0, 2.0]
    assert result["popularity"].tolist() == [10.0, 30.0, 20.0, 20.0]
    assert result["custom_numeric"].tolist() == [1.0, 3.0, 2.0, 2.0]
    assert result["odd_numeric"].tolist() == [1.0, 5.0, 9.0, 5.0]
    assert result["all_missing_numeric"].isna().all()
    assert result["note"].isna().sum() == 1


def test_preprocessing_decade_construction_contract() -> None:
    years: list[object] = [1921, 1929, 1930, 1999, 2000, 2020, "bad", None]
    source = pd.DataFrame(
        {
            "id": [f"track-{index}" for index in range(len(years))],
            "name": [f"Track {index}" for index in range(len(years))],
            "year": years,
        }
    )

    result = clean_tracks_for_analysis(source)

    assert result["decade"].tolist() == [
        1920,
        1920,
        1930,
        1990,
        2000,
        2020,
        1960,
        1960,
    ]
    assert str(result["decade"].dtype) == "int64"
    assert not result["decade"].isna().any()
    with pytest.raises(pd.errors.IntCastingNaNError):
        clean_tracks_for_analysis(
            pd.DataFrame(
                {"id": ["a", "b"], "name": ["A", "B"], "year": [None, "bad"]}
            )
        )


def test_preprocessing_preserves_row_order_and_resets_index() -> None:
    source = pd.DataFrame(
        {
            "id": ["first", None, "third", "fourth"],
            "name": ["First", "Removed", "Third", "Fourth"],
            "year": [2003, 1990, 1921, 2020],
        },
        index=[50, 10, 90, 20],
    )

    result = clean_tracks_for_analysis(source)

    assert result["id"].tolist() == ["first", "third", "fourth"]
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.equals(pd.RangeIndex(3))
    assert "index" not in result.columns


def test_preprocessing_is_deterministic() -> None:
    source = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "year": [1921, 1955, 2020],
            "energy": [0.1, np.nan, 0.9],
        }
    )

    first = clean_tracks_for_analysis(source.copy(deep=True))
    second = clean_tracks_for_analysis(source.copy(deep=True))

    pd.testing.assert_frame_equal(first, second)


def test_preprocessing_matches_canonical_baseline(
    project_loader_result: tuple[dict[str, pd.DataFrame], dict[str, Path]],
    canonical_processed_tracks: pd.DataFrame,
) -> None:
    source = project_loader_result[0]["tracks"]
    processed = canonical_processed_tracks
    expected_missing = {column: 0 for column in CANONICAL_PROCESSED_COLUMNS}
    expected_missing["release_date_parsed"] = 119798

    assert source.shape == (170653, 20)
    assert processed.shape == (170653, 21)
    assert len(source) - len(processed) == 0
    assert processed.columns.tolist() == CANONICAL_PROCESSED_COLUMNS
    assert processed.index.equals(pd.RangeIndex(170653))
    assert {column: str(dtype) for column, dtype in processed.dtypes.items()} == (
        CANONICAL_PROCESSED_DTYPES
    )
    assert processed.isna().sum().to_dict() == expected_missing
    assert int(processed["decade"].min()) == 1920
    assert int(processed["decade"].max()) == 2020
    assert not processed["decade"].isna().any()
    assert int(processed["id"].duplicated().sum()) == 0
    assert processed["id"].head(5).tolist() == source["id"].head(5).tolist()
    assert _dataframe_fingerprint(processed) == CANONICAL_PROCESSED_FINGERPRINT


def test_preprocessing_preserves_regression_feature_compatibility(
    canonical_processed_tracks: pd.DataFrame,
) -> None:
    required = [TARGET, *REGRESSION_AUDIO_FEATURES, *REGRESSION_EXTENDED_FEATURES]

    assert set(required).issubset(canonical_processed_tracks.columns)
    for column in set(required):
        values = canonical_processed_tracks[column]
        assert is_numeric_dtype(values), column
        assert np.isfinite(values.to_numpy()).all(), column


def test_preprocessing_preserves_recommender_catalog_compatibility(
    canonical_processed_tracks: pd.DataFrame,
) -> None:
    required = ["id", "name", "artists", *RECOMMENDER_FEATURES]

    assert set(required).issubset(canonical_processed_tracks.columns)
    catalog = canonical_processed_tracks.dropna(
        subset=["name", "artists", *RECOMMENDER_FEATURES]
    ).reset_index(drop=True)
    assert len(catalog) == 170653
    assert catalog["id"].is_unique
    assert np.isfinite(catalog[RECOMMENDER_FEATURES].to_numpy()).all()


def test_preprocessing_has_no_file_system_side_effects(
    project_root: Path,
    cleaned_data_dir: Path,
    project_loader_result: tuple[dict[str, pd.DataFrame], dict[str, Path]],
) -> None:
    before_files = _tree_manifest(cleaned_data_dir)
    before_root_dirs = {
        path.name for path in project_root.iterdir() if path.is_dir()
    }

    clean_tracks_for_analysis(project_loader_result[0]["tracks"].copy(deep=True))

    assert _tree_manifest(cleaned_data_dir) == before_files
    assert {path.name for path in project_root.iterdir() if path.is_dir()} == before_root_dirs


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
