"""Characterization tests for pure EDA calculations and generated outputs."""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Generator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from src.data_loader import load_project_data
from src.eda import (
    compute_audio_feature_trends_by_year,
    compute_correlation_matrix,
    compute_dataset_overview,
    compute_decade_explicit_summary,
    compute_decade_feature_summary,
    compute_descriptive_statistics,
    compute_missing_values_summary,
    compute_popularity_by_decade,
    compute_top_genres_audio_profile,
    compute_track_counts_by_decade,
    compute_tracks_by_decade_summary,
    create_genre_decade_pivot,
    parse_artist_names,
    prepare_interactive_energy_loudness_data,
)
from src.preprocessing import clean_tracks_for_analysis


EXPECTED_RETURN_SHAPE = (170653, 21)
EXPECTED_RETURN_FINGERPRINT = (
    "3ccbd1356da53ff61ea16bacebd9058b98fc8c19712d30c27218932ab7e1e52f"
)
EXPECTED_RETURN_ATTRS = {
    "interactive_plot_note": (
        "Interactive Plotly HTML and static PNG preview created successfully."
    ),
    "genre_pivot_note": (
        "Genre pivot created for the 30 genres with the most track-artist matches."
    ),
}
EXPECTED_TABLES: dict[str, dict[str, Any]] = {
    "audio_feature_trends_by_year.csv": {
        "shape": (100, 5),
        "columns": ["year", "energy", "danceability", "acousticness", "valence"],
        "fingerprint": "9611dff1e51115a1d60609682e19db35d26b57ea20ca7ae596d59767cf8de59b",
    },
    "correlation_matrix.csv": {
        "shape": (10, 11),
        "columns": [
            "feature",
            "acousticness",
            "danceability",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
            "valence",
            "popularity",
        ],
        "fingerprint": "c43428b5abbb113d037fbbf596c15ea26a09b1145b3c541b78c66e8ae53fe6c8",
    },
    "dataset_overview.csv": {
        "shape": (5, 3),
        "columns": ["table", "rows", "columns"],
        "fingerprint": "0c05742bad16614cdbf5a274ec01c572e45a8cd3363a9e67100b88cc0aaa2f95",
    },
    "decade_explicit_multiindex_summary.csv": {
        "shape": (22, 7),
        "columns": [
            "decade",
            "explicit",
            "total_tracks",
            "average_popularity",
            "average_energy",
            "average_danceability",
            "average_acousticness",
        ],
        "fingerprint": "493b0ce61b96496d95f7c64cf9fa6a966df87f249bfc02f5a41547944d2d415e",
    },
    "decade_feature_summary.csv": {
        "shape": (11, 7),
        "columns": [
            "decade",
            "total_tracks",
            "avg_popularity",
            "avg_energy",
            "avg_danceability",
            "avg_acousticness",
            "avg_valence",
        ],
        "fingerprint": "7e82be3e2d3e0ef7f6dcb162e0ab2818b21d8e97f8b1aa6b8ef41bbe87bacc79",
    },
    "descriptive_statistics.csv": {
        "shape": (15, 9),
        "columns": [
            "feature",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ],
        "fingerprint": "2b55b095615845653c7eddd6db9b084ac5b7a6b713d118e6badb09190729567c",
    },
    "genre_decade_popularity_pivot.csv": {
        "shape": (30, 12),
        "columns": [
            "genre",
            "1920",
            "1930",
            "1940",
            "1950",
            "1960",
            "1970",
            "1980",
            "1990",
            "2000",
            "2010",
            "2020",
        ],
        "fingerprint": "e42db32720d0ebce23a3a48fb069428dc8e1366eab571d15401d4a3c4062828e",
    },
    "missing_values_after_cleaning.csv": {
        "shape": (21, 3),
        "columns": ["column", "missing_count", "missing_rate"],
        "fingerprint": "5c4682a44e759550b48a5e8b62d0ddd25ce131ea773d57672a6865c2eddfa7ac",
    },
    "top_genres_audio_profile.csv": {
        "shape": (20, 7),
        "columns": [
            "genres_clean",
            "popularity",
            "energy",
            "danceability",
            "acousticness",
            "valence",
            "tempo",
        ],
        "fingerprint": "17fd72d600a419242f8f10ad05e26b04e6988cb11e975e38dd1b5c909989e69d",
    },
    "tracks_by_decade.csv": {
        "shape": (11, 3),
        "columns": ["decade", "track_count", "avg_popularity"],
        "fingerprint": "001c9875a6fc5bdb4d3213125edc65b832dbd60d19527b198ad2a7a5908fdc16",
    },
}
EXPECTED_FIGURES = {
    "audio_feature_trends_by_year.png",
    "correlation_heatmap.png",
    "interactive_energy_loudness.html",
    "interactive_energy_loudness_preview.png",
    "popularity_by_decade_boxplot.png",
    "popularity_distribution.png",
    "tracks_by_decade.png",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _dataframe_fingerprint(frame: pd.DataFrame) -> str:
    hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def repository_preservation(project_root: Path) -> Generator[None, None, None]:
    protected = [project_root / "cleaned_data", project_root / "week7_outputs"]
    before = {path.name: _tree_manifest(path) for path in protected}

    yield

    after = {path.name: _tree_manifest(path) for path in protected}
    assert after == before


@pytest.fixture(scope="session")
def project_module(
    project_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ModuleType, None, None]:
    mpl_config = tmp_path_factory.mktemp("eda-matplotlib-config")
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
def canonical_project_data(
    project_root: Path,
) -> Mapping[str, pd.DataFrame]:
    data, _ = load_project_data(project_root)
    return data


@pytest.fixture(scope="session")
def canonical_processed_tracks(
    canonical_project_data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    return clean_tracks_for_analysis(canonical_project_data["tracks"].copy(deep=True))


@pytest.fixture(scope="session")
def temporary_eda_run(
    project_module: ModuleType,
    canonical_project_data: Mapping[str, pd.DataFrame],
    tmp_path_factory: pytest.TempPathFactory,
) -> Mapping[str, Any]:
    run_root = tmp_path_factory.mktemp("eda-run")
    paths = project_module.make_paths(run_root, "outputs")
    input_copies = {
        name: frame.copy(deep=True)
        for name, frame in canonical_project_data.items()
    }

    result = project_module.create_eda_outputs(input_copies, paths)
    table_paths = sorted(paths.tables.glob("*.csv"))
    figure_paths = sorted(path for path in paths.figures.iterdir() if path.is_file())
    tables = {path.name: pd.read_csv(path) for path in table_paths}

    return {
        "result": result,
        "paths": paths,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "tables": tables,
    }


def test_compute_dataset_overview_is_ordered_deterministic_and_immutable() -> None:
    data = {
        "tracks": pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]}),
        "genres": pd.DataFrame({"genre": ["rock"]}),
    }
    before = {name: frame.copy(deep=True) for name, frame in data.items()}
    expected = pd.DataFrame(
        {"table": ["tracks", "genres"], "rows": [2, 1], "columns": [2, 1]}
    )

    first = compute_dataset_overview(data)
    second = compute_dataset_overview(data)

    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(first, second)
    for name, frame in data.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_compute_descriptive_statistics_selects_available_features_in_order() -> None:
    tracks = pd.DataFrame(
        {
            "popularity": [10.0, 20.0, 30.0],
            "year": [2000.0, 2001.0, 2002.0],
            "energy": [0.2, 0.4, 0.6],
            "unrelated": [1.0, 2.0, 3.0],
        }
    )
    before = tracks.copy(deep=True)

    first = compute_descriptive_statistics(tracks)
    second = compute_descriptive_statistics(tracks)

    assert first["feature"].tolist() == ["energy", "year", "popularity"]
    assert first.columns.tolist() == [
        "feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]
    assert first.set_index("feature").loc["energy", "mean"] == pytest.approx(0.4)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_missing_values_summary_preserves_sort_and_rate_semantics() -> None:
    tracks = pd.DataFrame(
        {"none": [1.0, 2.0, 3.0], "one": [1.0, np.nan, 3.0], "two": [np.nan, np.nan, 3.0]}
    )
    before = tracks.copy(deep=True)

    result = compute_missing_values_summary(tracks)
    repeated = compute_missing_values_summary(tracks.copy(deep=True))

    assert result.columns.tolist() == ["column", "missing_count", "missing_rate"]
    assert result["column"].tolist() == ["two", "one", "none"]
    assert result["missing_count"].tolist() == [2, 1, 0]
    np.testing.assert_allclose(result["missing_rate"], [2 / 3, 1 / 3, 0])
    pd.testing.assert_frame_equal(result, repeated)
    pd.testing.assert_frame_equal(tracks, before)


def test_decade_track_summaries_preserve_count_and_size_distinction() -> None:
    tracks = pd.DataFrame(
        {
            "decade": [1990, 1990, 2000, 2000],
            "id": ["a", "b", "c", None],
            "popularity": [10.0, 30.0, 50.0, np.nan],
        }
    )
    before = tracks.copy(deep=True)

    summary = compute_tracks_by_decade_summary(tracks)
    counts = compute_track_counts_by_decade(tracks)
    repeated_summary = compute_tracks_by_decade_summary(tracks.copy(deep=True))
    repeated_counts = compute_track_counts_by_decade(tracks.copy(deep=True))

    expected_summary = pd.DataFrame(
        {"decade": [1990, 2000], "track_count": [2, 1], "avg_popularity": [20.0, 50.0]}
    )
    expected_counts = pd.DataFrame(
        {"decade": [1990, 2000], "track_count": [2, 2]}
    )
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(counts, expected_counts)
    pd.testing.assert_frame_equal(summary, repeated_summary)
    pd.testing.assert_frame_equal(counts, repeated_counts)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_popularity_by_decade_sorts_and_drops_missing_values() -> None:
    tracks = pd.DataFrame(
        {"decade": [2000, 1990, 2000, 1990], "popularity": [40.0, 10.0, np.nan, 30.0]}
    )
    before = tracks.copy(deep=True)

    decades, values = compute_popularity_by_decade(tracks)
    repeated_decades, repeated_values = compute_popularity_by_decade(
        tracks.copy(deep=True)
    )

    assert decades == [1990, 2000]
    np.testing.assert_array_equal(values[0], [10.0, 30.0])
    np.testing.assert_array_equal(values[1], [40.0])
    assert repeated_decades == decades
    for actual, repeated in zip(values, repeated_values):
        np.testing.assert_array_equal(actual, repeated)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_decade_feature_summary_uses_exact_aggregations() -> None:
    tracks = pd.DataFrame(
        {
            "decade": [1990, 1990, 2000],
            "popularity": [10.0, 30.0, 50.0],
            "energy": [0.2, 0.6, 1.0],
            "danceability": [0.1, 0.3, 0.5],
            "acousticness": [0.9, 0.5, 0.1],
            "valence": [0.2, 0.4, 0.8],
        }
    )
    before = tracks.copy(deep=True)

    result = compute_decade_feature_summary(tracks)
    repeated = compute_decade_feature_summary(tracks.copy(deep=True))

    assert result.columns.tolist() == [
        "decade", "total_tracks", "avg_popularity", "avg_energy",
        "avg_danceability", "avg_acousticness", "avg_valence"
    ]
    assert result["decade"].tolist() == [1990, 2000]
    assert result.loc[0].to_dict() == {
        "decade": 1990.0,
        "total_tracks": 2.0,
        "avg_popularity": 20.0,
        "avg_energy": 0.4,
        "avg_danceability": 0.2,
        "avg_acousticness": 0.7,
        "avg_valence": 0.3,
    }
    pd.testing.assert_frame_equal(result, repeated)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_audio_feature_trends_by_year_preserves_feature_order() -> None:
    tracks = pd.DataFrame(
        {
            "year": [2001, 2000, 2000],
            "energy": [0.9, 0.2, 0.6],
            "danceability": [0.8, 0.1, 0.3],
            "acousticness": [0.1, 0.9, 0.5],
            "valence": [0.7, 0.2, 0.4],
        }
    )
    before = tracks.copy(deep=True)

    first = compute_audio_feature_trends_by_year(tracks)
    second = compute_audio_feature_trends_by_year(tracks.copy(deep=True))

    assert first.columns.tolist() == [
        "year", "energy", "danceability", "acousticness", "valence"
    ]
    assert first["year"].tolist() == [2000, 2001]
    np.testing.assert_allclose(first.loc[0, first.columns[1:]], [0.4, 0.2, 0.7, 0.3])
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_correlation_matrix_is_ordered_symmetric_and_numeric_only() -> None:
    tracks = pd.DataFrame(
        {"energy": [1.0, 2.0, 3.0], "danceability": [3.0, 2.0, 1.0], "name": ["a", "b", "c"]}
    )
    before = tracks.copy(deep=True)

    result = compute_correlation_matrix(tracks, ["danceability", "energy"])
    repeated = compute_correlation_matrix(
        tracks.copy(deep=True), ["danceability", "energy"]
    )

    assert result.index.tolist() == ["danceability", "energy"]
    assert result.columns.tolist() == ["danceability", "energy"]
    np.testing.assert_allclose(result, [[1.0, -1.0], [-1.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(result, result.T, atol=0)
    pd.testing.assert_frame_equal(result, repeated)
    pd.testing.assert_frame_equal(tracks, before)


def test_compute_decade_explicit_summary_preserves_multi_group_order() -> None:
    tracks = pd.DataFrame(
        {
            "decade": [2000, 1990, 1990, 2000],
            "explicit": [1, 1, 0, 0],
            "popularity": [40.0, 30.0, 10.0, 20.0],
            "energy": [0.4, 0.3, 0.1, 0.2],
            "danceability": [0.8, 0.6, 0.2, 0.4],
            "acousticness": [0.2, 0.4, 0.8, 0.6],
        }
    )
    before = tracks.copy(deep=True)

    result = compute_decade_explicit_summary(tracks)
    repeated = compute_decade_explicit_summary(tracks.copy(deep=True))

    assert result[["decade", "explicit"]].values.tolist() == [
        [1990, 0], [1990, 1], [2000, 0], [2000, 1]
    ]
    assert result["total_tracks"].tolist() == [1, 1, 1, 1]
    pd.testing.assert_frame_equal(result, repeated)
    pd.testing.assert_frame_equal(tracks, before)


def _interactive_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "energy": [0.1, 0.2, 0.3, np.nan],
            "loudness": [-10.0, -9.0, -8.0, -7.0],
            "decade": [1990, 2000, 2010, 2020],
            "name": ["A", "B", "C", "D"],
            "artists": ["AA", "BB", "CC", "DD"],
            "popularity": [10, 20, 30, 40],
            "year": [1991, 2001, 2011, 2021],
            "danceability": [0.2, 0.4, 0.6, 0.8],
        }
    )


def test_prepare_interactive_data_is_deterministic_complete_and_immutable() -> None:
    tracks = _interactive_tracks()
    before = tracks.copy(deep=True)

    first, first_note = prepare_interactive_energy_loudness_data(tracks, sample_size=2)
    second, second_note = prepare_interactive_energy_loudness_data(tracks, sample_size=2)

    assert first is not None and second is not None
    assert first_note is None and second_note is None
    assert len(first) == 2
    assert str(first["decade"].dtype) == "str"
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(tracks, before)


def test_prepare_interactive_data_preserves_skip_reasons() -> None:
    missing_result, missing_note = prepare_interactive_energy_loudness_data(
        pd.DataFrame({"energy": [0.5]})
    )
    empty = _interactive_tracks().iloc[[3]].copy()
    empty_result, empty_note = prepare_interactive_energy_loudness_data(empty)

    assert missing_result is None
    assert missing_note == (
        "Interactive Plotly scatter skipped: missing columns loudness, decade, name, "
        "artists, popularity, year, danceability"
    )
    assert empty_result is None
    assert empty_note == "Interactive Plotly scatter skipped: no complete rows available."


def test_compute_top_genres_profile_filters_sorts_and_preserves_input() -> None:
    genres = pd.DataFrame(
        {
            "genres_clean": ["rock", None, "jazz", "pop"],
            "popularity": [40.0, 100.0, 80.0, 60.0],
            "energy": [0.8, 0.9, 0.5, 0.7],
            "danceability": [0.4, 0.5, 0.6, 0.7],
            "unused": [1, 2, 3, 4],
        }
    )
    before = genres.copy(deep=True)

    first = compute_top_genres_audio_profile(genres)
    second = compute_top_genres_audio_profile(genres)

    assert first.columns.tolist() == [
        "genres_clean", "popularity", "energy", "danceability"
    ]
    assert first["genres_clean"].tolist() == ["jazz", "pop", "rock"]
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(genres, before)

    fallback = compute_top_genres_audio_profile(
        pd.DataFrame({"genres": ["folk", "blues"], "popularity": [10.0, 20.0]})
    )
    assert fallback.columns.tolist() == ["genres", "popularity"]
    assert fallback["genres"].tolist() == ["blues", "folk"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([" A ", "", "B"], ["A", "B"]),
        ("['A', ' B ']", ["A", "B"]),
        ("('A', 'B')", ["A", "B"]),
        ("A", ["A"]),
        (None, []),
    ],
)
def test_parse_artist_names_preserves_existing_cases(
    value: Any,
    expected: list[str],
) -> None:
    assert parse_artist_names(value) == expected


def test_create_genre_decade_pivot_preserves_aggregation_and_missing_cells() -> None:
    tracks = pd.DataFrame(
        {
            "artists": ["['A']", "['A', 'B']", "B"],
            "decade": [1990, 1990, 2000],
            "popularity": [10.0, 30.0, 50.0],
        }
    )
    genres = pd.DataFrame(
        {"artists": ["A", "B"], "genres_clean": ["rock;pop", "rock;jazz"]}
    )
    tracks_before = tracks.copy(deep=True)
    genres_before = genres.copy(deep=True)

    first, first_note = create_genre_decade_pivot(tracks, genres)
    second, second_note = create_genre_decade_pivot(tracks, genres)

    assert first is not None and second is not None
    assert first.columns.tolist() == ["genre", "1990", "2000"]
    assert first["genre"].tolist() == ["jazz", "pop", "rock"]
    assert first.loc[first["genre"] == "jazz", "1990"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(first.loc[first["genre"] == "pop", "2000"].iloc[0])
    assert first.loc[first["genre"] == "rock", "1990"].iloc[0] == pytest.approx(70 / 3)
    assert first.loc[first["genre"] == "rock", "2000"].iloc[0] == pytest.approx(50.0)
    assert first_note == second_note == (
        "Genre pivot created for the 3 genres with the most track-artist matches."
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(tracks, tracks_before)
    pd.testing.assert_frame_equal(genres, genres_before)


@pytest.mark.parametrize(
    ("tracks", "genres", "message"),
    [
        (
            pd.DataFrame({"artists": ["A"]}),
            pd.DataFrame({"artists": ["A"], "genres_clean": ["rock"]}),
            "Genre pivot skipped: track-level artist/decade/popularity columns are incomplete.",
        ),
        (
            pd.DataFrame({"artists": ["A"], "decade": [2000], "popularity": [10]}),
            pd.DataFrame({"artists": ["A"]}),
            "Genre pivot skipped: artist-to-genre columns are incomplete.",
        ),
        (
            pd.DataFrame({"artists": ["A"], "decade": [2000], "popularity": [10]}),
            pd.DataFrame({"artists": ["B"], "genres_clean": ["rock"]}),
            "Genre pivot skipped: no track artists matched the artist-to-genre data.",
        ),
    ],
)
def test_create_genre_decade_pivot_preserves_optional_skip_behavior(
    tracks: pd.DataFrame,
    genres: pd.DataFrame,
    message: str,
) -> None:
    result, note = create_genre_decade_pivot(tracks, genres)
    assert result is None
    assert note == message


def test_eda_helpers_are_integrated_publicly(project_module: ModuleType) -> None:
    expected = {
        "compute_audio_feature_trends_by_year": compute_audio_feature_trends_by_year,
        "compute_correlation_matrix": compute_correlation_matrix,
        "compute_dataset_overview": compute_dataset_overview,
        "compute_decade_explicit_summary": compute_decade_explicit_summary,
        "compute_decade_feature_summary": compute_decade_feature_summary,
        "compute_descriptive_statistics": compute_descriptive_statistics,
        "compute_missing_values_summary": compute_missing_values_summary,
        "compute_popularity_by_decade": compute_popularity_by_decade,
        "compute_top_genres_audio_profile": compute_top_genres_audio_profile,
        "compute_track_counts_by_decade": compute_track_counts_by_decade,
        "compute_tracks_by_decade_summary": compute_tracks_by_decade_summary,
        "create_genre_decade_pivot": create_genre_decade_pivot,
        "parse_artist_names": parse_artist_names,
        "prepare_interactive_energy_loudness_data": prepare_interactive_energy_loudness_data,
    }
    for name, helper in expected.items():
        assert getattr(project_module, name) is helper


def test_create_eda_outputs_preserves_absent_auxiliary_dataset_behavior(
    project_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracks = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "artists": ["AA", "BB", "CC"],
            "year": [1991, 2001, 2011],
            "popularity": [10.0, 20.0, 30.0],
            "explicit": [0, 1, 0],
            "energy": [0.2, 0.4, 0.6],
            "danceability": [0.3, 0.5, 0.7],
            "acousticness": [0.8, 0.6, 0.4],
            "valence": [0.1, 0.3, 0.5],
        }
    )
    before = tracks.copy(deep=True)
    paths = project_module.make_paths(tmp_path, "outputs")
    monkeypatch.setattr(project_module, "plot_tracks_by_decade", lambda *_: None)
    monkeypatch.setattr(project_module, "plot_popularity_distribution", lambda *_: None)
    monkeypatch.setattr(
        project_module, "plot_popularity_by_decade_boxplot", lambda *_: None
    )
    monkeypatch.setattr(
        project_module,
        "plot_feature_trends",
        lambda frame, _: compute_audio_feature_trends_by_year(frame),
    )
    monkeypatch.setattr(
        project_module,
        "plot_correlation_heatmap",
        lambda frame, features, _: (
            compute_correlation_matrix(frame, features)
            .reset_index()
            .rename(columns={"index": "feature"})
        ),
    )
    monkeypatch.setattr(
        project_module,
        "plot_interactive_energy_loudness",
        lambda *_: "Synthetic interactive plot note.",
    )

    result = project_module.create_eda_outputs({"tracks": tracks}, paths)

    assert result.attrs["genre_pivot_note"] == (
        "Genre pivot skipped: artist-to-genre data was not loaded."
    )
    assert result.attrs["interactive_plot_note"] == "Synthetic interactive plot note."
    assert not (paths.tables / "top_genres_audio_profile.csv").exists()
    assert not (paths.tables / "genre_decade_popularity_pivot.csv").exists()
    pd.testing.assert_frame_equal(tracks, before)


def test_temporary_eda_return_matches_pre_extraction_baseline(
    temporary_eda_run: Mapping[str, Any],
    canonical_processed_tracks: pd.DataFrame,
) -> None:
    result = temporary_eda_run["result"]

    assert isinstance(result, pd.DataFrame)
    assert result.shape == EXPECTED_RETURN_SHAPE
    assert result.columns.tolist() == canonical_processed_tracks.columns.tolist()
    assert {column: str(dtype) for column, dtype in result.dtypes.items()} == {
        column: str(dtype) for column, dtype in canonical_processed_tracks.dtypes.items()
    }
    assert _dataframe_fingerprint(result) == EXPECTED_RETURN_FINGERPRINT
    assert result.attrs == EXPECTED_RETURN_ATTRS
    pd.testing.assert_frame_equal(result, canonical_processed_tracks)


def test_temporary_eda_output_manifests_are_exact(
    temporary_eda_run: Mapping[str, Any],
) -> None:
    assert {path.name for path in temporary_eda_run["table_paths"]} == set(EXPECTED_TABLES)
    assert {path.name for path in temporary_eda_run["figure_paths"]} == EXPECTED_FIGURES
    assert not list(temporary_eda_run["paths"].model_artifacts.iterdir())
    assert not list(temporary_eda_run["paths"].sql.iterdir())


@pytest.mark.parametrize("filename", sorted(EXPECTED_TABLES))
def test_temporary_eda_tables_match_baseline_and_canonical_output(
    filename: str,
    temporary_eda_run: Mapping[str, Any],
    project_root: Path,
) -> None:
    generated = temporary_eda_run["tables"][filename]
    canonical = pd.read_csv(project_root / "week7_outputs" / "tables" / filename)
    metadata = EXPECTED_TABLES[filename]

    assert generated.shape == metadata["shape"]
    assert generated.columns.tolist() == metadata["columns"]
    assert _dataframe_fingerprint(generated) == metadata["fingerprint"]
    assert {column: str(dtype) for column, dtype in generated.dtypes.items()} == {
        column: str(dtype) for column, dtype in canonical.dtypes.items()
    }
    assert generated.index.equals(canonical.index)
    for column in generated.columns:
        if is_numeric_dtype(generated[column]):
            np.testing.assert_allclose(
                generated[column], canonical[column], rtol=0, atol=1e-12, equal_nan=True
            )
        else:
            pd.testing.assert_series_equal(
                generated[column], canonical[column], check_exact=True
            )


def test_temporary_eda_figures_are_valid_and_nonempty(
    temporary_eda_run: Mapping[str, Any],
) -> None:
    figures = {path.name: path for path in temporary_eda_run["figure_paths"]}

    for name, path in figures.items():
        assert path.stat().st_size > 0, name
        if path.suffix.lower() == ".png":
            with path.open("rb") as handle:
                assert handle.read(8) == PNG_SIGNATURE
    html = figures["interactive_energy_loudness.html"]
    assert "<html" in html.read_text(encoding="utf-8").lower()
