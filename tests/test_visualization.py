"""Characterization tests for extracted EDA visualization functions."""

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

from src.config import RECOMMENDER_FEATURES, TARGET, TREND_FEATURES
from src.data_loader import load_project_data
from src.eda import (
    compute_audio_feature_trends_by_year,
    compute_correlation_matrix,
    compute_popularity_by_decade,
    prepare_interactive_energy_loudness_data,
)


EXPECTED_TABLES = {
    "audio_feature_trends_by_year.csv",
    "correlation_matrix.csv",
    "dataset_overview.csv",
    "decade_explicit_multiindex_summary.csv",
    "decade_feature_summary.csv",
    "descriptive_statistics.csv",
    "genre_decade_popularity_pivot.csv",
    "missing_values_after_cleaning.csv",
    "top_genres_audio_profile.csv",
    "tracks_by_decade.csv",
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


def _assert_png(path: Path, dimensions: tuple[int, int]) -> None:
    import matplotlib.image as mpimg

    assert path.is_file()
    assert path.stat().st_size > 0
    with path.open("rb") as handle:
        assert handle.read(8) == PNG_SIGNATURE
    pixels = mpimg.imread(path)
    assert (pixels.shape[1], pixels.shape[0]) == dimensions


def _assert_no_open_figures() -> None:
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


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
    mpl_config = tmp_path_factory.mktemp("visualization-matplotlib-config")
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
def visualization_module(project_module: ModuleType) -> ModuleType:
    module = importlib.import_module("src.visualization")
    assert project_module.plot_tracks_by_decade is module.plot_tracks_by_decade
    return module


@pytest.fixture
def synthetic_tracks() -> pd.DataFrame:
    rows = 12
    return pd.DataFrame(
        {
            "id": [f"id-{index}" for index in range(rows)],
            "name": [f"Track {index}" for index in range(rows)],
            "artists": [f"Artist {index % 4}" for index in range(rows)],
            "year": [2000 + index % 4 for index in range(rows)],
            "decade": [1990, 2000, 2010] * 4,
            "popularity": [10 + 5 * index for index in range(rows)],
            "explicit": [index % 2 for index in range(rows)],
            "energy": np.linspace(0.1, 0.9, rows),
            "loudness": np.linspace(-18.0, -4.0, rows),
            "danceability": np.linspace(0.2, 0.8, rows),
            "acousticness": np.linspace(0.9, 0.1, rows),
            "valence": np.linspace(0.15, 0.85, rows),
        }
    )


@pytest.fixture(scope="session")
def canonical_project_data(project_root: Path) -> Mapping[str, pd.DataFrame]:
    data, _ = load_project_data(project_root)
    return data


@pytest.fixture(scope="session")
def temporary_canonical_eda_run(
    project_module: ModuleType,
    canonical_project_data: Mapping[str, pd.DataFrame],
    tmp_path_factory: pytest.TempPathFactory,
) -> Mapping[str, Any]:
    run_root = tmp_path_factory.mktemp("visualization-canonical-eda")
    paths = project_module.make_paths(run_root, "outputs")
    input_copies = {
        name: frame.copy(deep=True)
        for name, frame in canonical_project_data.items()
    }
    result = project_module.create_eda_outputs(input_copies, paths)
    tables = {
        path.name: pd.read_csv(path)
        for path in sorted(paths.tables.glob("*.csv"))
    }
    figures = {
        path.name: path
        for path in sorted(paths.figures.iterdir())
        if path.is_file()
    }
    return {"result": result, "paths": paths, "tables": tables, "figures": figures}


def test_public_visualization_functions_are_identical(
    project_module: ModuleType,
    visualization_module: ModuleType,
) -> None:
    names = [
        "plot_tracks_by_decade",
        "plot_popularity_distribution",
        "plot_popularity_by_decade_boxplot",
        "plot_feature_trends",
        "plot_correlation_heatmap",
        "plot_interactive_energy_loudness",
    ]
    for name in names:
        assert getattr(project_module, name) is getattr(visualization_module, name)


def test_visualization_import_creates_no_figures(
    visualization_module: ModuleType,
) -> None:
    assert visualization_module.__name__ == "src.visualization"
    _assert_no_open_figures()


def test_plot_tracks_by_decade_png_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = synthetic_tracks.copy(deep=True)
    first = tmp_path / "tracks_by_decade.png"
    second = tmp_path / "repeat_tracks_by_decade.png"

    assert visualization_module.plot_tracks_by_decade(synthetic_tracks, first) is None
    assert visualization_module.plot_tracks_by_decade(synthetic_tracks, second) is None

    _assert_png(first, (1800, 900))
    _assert_png(second, (1800, 900))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_plot_popularity_distribution_png_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = synthetic_tracks.copy(deep=True)
    first = tmp_path / "popularity_distribution.png"
    second = tmp_path / "repeat_popularity_distribution.png"

    assert visualization_module.plot_popularity_distribution(synthetic_tracks, first) is None
    assert visualization_module.plot_popularity_distribution(synthetic_tracks, second) is None

    _assert_png(first, (1620, 900))
    _assert_png(second, (1620, 900))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_plot_popularity_by_decade_boxplot_png_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = synthetic_tracks.copy(deep=True)
    first = tmp_path / "popularity_by_decade_boxplot.png"
    second = tmp_path / "repeat_popularity_by_decade_boxplot.png"

    assert visualization_module.plot_popularity_by_decade_boxplot(synthetic_tracks, first) is None
    assert visualization_module.plot_popularity_by_decade_boxplot(synthetic_tracks, second) is None

    _assert_png(first, (1980, 1080))
    _assert_png(second, (1980, 1080))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_plot_feature_trends_png_and_return_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = synthetic_tracks.copy(deep=True)
    expected = compute_audio_feature_trends_by_year(synthetic_tracks)
    first_path = tmp_path / "audio_feature_trends_by_year.png"
    second_path = tmp_path / "repeat_audio_feature_trends_by_year.png"

    first = visualization_module.plot_feature_trends(synthetic_tracks, first_path)
    second = visualization_module.plot_feature_trends(synthetic_tracks, second_path)

    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(second, expected)
    assert first.columns.tolist() == ["year", *TREND_FEATURES]
    _assert_png(first_path, (1800, 900))
    _assert_png(second_path, (1800, 900))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_plot_correlation_heatmap_png_and_return_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    features = ["energy", "danceability", "acousticness", "valence", TARGET]
    before = synthetic_tracks.copy(deep=True)
    expected = (
        compute_correlation_matrix(synthetic_tracks, features)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    first_path = tmp_path / "correlation_heatmap.png"
    second_path = tmp_path / "repeat_correlation_heatmap.png"

    first = visualization_module.plot_correlation_heatmap(
        synthetic_tracks, features, first_path
    )
    second = visualization_module.plot_correlation_heatmap(
        synthetic_tracks, features, second_path
    )

    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(second, expected)
    assert first.columns.tolist() == ["feature", *features]
    assert first["feature"].tolist() == features
    _assert_png(first_path, (1620, 1260))
    _assert_png(second_path, (1620, 1260))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_tracks_by_decade_uses_group_size_not_id_count(
    visualization_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracks = pd.DataFrame(
        {"decade": [1990, 1990, 2000], "id": ["a", None, "c"]}
    )
    captured: dict[str, Any] = {}

    def capture_bar(x: Any, y: Any) -> list[Any]:
        captured["x"] = list(x)
        captured["y"] = list(y)
        return []

    monkeypatch.setattr(visualization_module.plt, "bar", capture_bar)
    visualization_module.plot_tracks_by_decade(
        tracks, tmp_path / "tracks_by_decade.png"
    )

    assert captured == {"x": ["1990", "2000"], "y": [2, 1]}
    _assert_no_open_figures()


def test_popularity_distribution_drops_missing_values_before_histogram(
    visualization_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracks = pd.DataFrame({TARGET: [10.0, np.nan, 30.0]})
    captured: dict[str, Any] = {}

    def capture_hist(values: Any, **kwargs: Any) -> tuple[list[Any], list[Any], list[Any]]:
        captured["values"] = list(values)
        captured.update(kwargs)
        return [], [], []

    monkeypatch.setattr(visualization_module.plt, "hist", capture_hist)
    visualization_module.plot_popularity_distribution(
        tracks, tmp_path / "popularity_distribution.png"
    )

    assert captured == {"values": [10.0, 30.0], "bins": 25, "edgecolor": "white"}
    _assert_no_open_figures()


def test_popularity_boxplot_uses_sorted_complete_helper_arrays(
    visualization_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracks = pd.DataFrame(
        {"decade": [2000, 1990, 2000, 1990], TARGET: [40.0, 10.0, np.nan, 30.0]}
    )
    expected_decades, expected_values = compute_popularity_by_decade(tracks)
    captured: dict[str, Any] = {}

    def capture_boxplot(values: Any, **kwargs: Any) -> dict[str, Any]:
        captured["values"] = [np.asarray(value) for value in values]
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(visualization_module.plt, "boxplot", capture_boxplot)
    visualization_module.plot_popularity_by_decade_boxplot(
        tracks, tmp_path / "popularity_by_decade_boxplot.png"
    )

    assert expected_decades == [1990, 2000]
    assert captured["tick_labels"] == ["1990", "2000"]
    assert captured["showfliers"] is False
    for actual, expected in zip(captured["values"], expected_values):
        np.testing.assert_array_equal(actual, expected)
    _assert_no_open_figures()


def test_feature_trend_lines_follow_exact_feature_order(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        visualization_module.plt,
        "plot",
        lambda _x, _y, label: labels.append(label),
    )
    monkeypatch.setattr(visualization_module.plt, "legend", lambda: None)

    visualization_module.plot_feature_trends(
        synthetic_tracks, tmp_path / "audio_feature_trends_by_year.png"
    )

    assert labels == TREND_FEATURES
    _assert_no_open_figures()


def test_plotly_success_output_contract(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path: Path,
) -> None:
    before = synthetic_tracks.copy(deep=True)
    html = tmp_path / "interactive_energy_loudness.html"
    preview = tmp_path / "interactive_energy_loudness_preview.png"

    note = visualization_module.plot_interactive_energy_loudness(
        synthetic_tracks, html, preview, sample_size=5
    )

    assert note == "Interactive Plotly HTML and static PNG preview created successfully."
    assert html.is_file() and html.stat().st_size > 0
    html_text = html.read_text(encoding="utf-8").lower()
    assert "<html" in html_text
    assert "plotly" in html_text
    _assert_png(preview, (2400, 1600))
    pd.testing.assert_frame_equal(synthetic_tracks, before)
    _assert_no_open_figures()


def test_plotly_uses_deterministic_sample_and_string_decades(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[pd.DataFrame] = []

    class FakeFigure:
        def write_html(self, path: Path, include_plotlyjs: str) -> None:
            assert include_plotlyjs == "cdn"
            path.write_text("<html>plotly</html>", encoding="utf-8")

        def write_image(
            self, path: Path, width: int, height: int, scale: int
        ) -> None:
            assert (width, height, scale) == (1200, 800, 2)

    def capture_scatter(frame: pd.DataFrame, **kwargs: Any) -> FakeFigure:
        captured.append(frame.copy(deep=True))
        assert kwargs == {
            "x": "energy",
            "y": "loudness",
            "color": "decade",
            "hover_data": ["name", "artists", TARGET, "year", "danceability"],
            "title": "Interactive Energy vs Loudness by Decade",
            "labels": {
                "energy": "Energy",
                "loudness": "Loudness",
                "decade": "Decade",
                TARGET: "Popularity",
            },
        }
        return FakeFigure()

    monkeypatch.setattr(visualization_module.px, "scatter", capture_scatter)
    visualization_module.plot_interactive_energy_loudness(
        synthetic_tracks,
        tmp_path / "first.html",
        tmp_path / "first.png",
        sample_size=5,
    )
    visualization_module.plot_interactive_energy_loudness(
        synthetic_tracks,
        tmp_path / "second.html",
        tmp_path / "second.png",
        sample_size=5,
    )

    expected, note = prepare_interactive_energy_loudness_data(
        synthetic_tracks, sample_size=5
    )
    assert expected is not None and note is None
    pd.testing.assert_frame_equal(captured[0], expected)
    pd.testing.assert_frame_equal(captured[1], expected)
    assert str(captured[0]["decade"].dtype) == "str"
    assert set(captured[0]["decade"]) <= {"1990", "2000", "2010"}


@pytest.mark.parametrize(
    ("tracks", "message"),
    [
        (
            pd.DataFrame({"energy": [0.5]}),
            "Interactive Plotly scatter skipped: missing columns loudness, decade, "
            "name, artists, popularity, year, danceability",
        ),
        (
            pd.DataFrame(
                {
                    "energy": [np.nan],
                    "loudness": [-8.0],
                    "decade": [2000],
                    "name": ["A"],
                    "artists": ["AA"],
                    TARGET: [20],
                    "year": [2001],
                    "danceability": [0.5],
                }
            ),
            "Interactive Plotly scatter skipped: no complete rows available.",
        ),
    ],
)
def test_plotly_skip_paths_create_no_files(
    visualization_module: ModuleType,
    tracks: pd.DataFrame,
    message: str,
    tmp_path: Path,
) -> None:
    html = tmp_path / "interactive_energy_loudness.html"
    preview = tmp_path / "interactive_energy_loudness_preview.png"

    assert (
        visualization_module.plot_interactive_energy_loudness(tracks, html, preview)
        == message
    )
    assert not html.exists()
    assert not preview.exists()


def test_plotly_preview_failure_preserves_html_and_exact_note(
    visualization_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingFigure:
        def write_html(self, path: Path, include_plotlyjs: str) -> None:
            path.write_text("<html>plotly</html>", encoding="utf-8")

        def write_image(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("preview failed")

    monkeypatch.setattr(
        visualization_module.px, "scatter", lambda *_args, **_kwargs: FailingFigure()
    )
    html = tmp_path / "interactive_energy_loudness.html"
    preview = tmp_path / "interactive_energy_loudness_preview.png"

    note = visualization_module.plot_interactive_energy_loudness(
        synthetic_tracks, html, preview
    )

    assert note == (
        "Interactive Plotly HTML created; static PNG preview unavailable. "
        "Plotly image export requires Kaleido and compatible browser support. "
        "Error: preview failed"
    )
    assert html.is_file() and html.stat().st_size > 0
    assert not preview.exists()


def test_canonical_eda_return_contract(
    temporary_canonical_eda_run: Mapping[str, Any],
) -> None:
    result = temporary_canonical_eda_run["result"]
    assert result.shape == (170653, 21)
    assert _dataframe_fingerprint(result) == EXPECTED_RETURN_FINGERPRINT
    assert result.attrs == EXPECTED_RETURN_ATTRS


def test_canonical_eda_table_manifest_and_values(
    temporary_canonical_eda_run: Mapping[str, Any],
    project_root: Path,
) -> None:
    tables = temporary_canonical_eda_run["tables"]
    assert set(tables) == EXPECTED_TABLES
    for name, generated in tables.items():
        canonical = pd.read_csv(project_root / "week7_outputs" / "tables" / name)
        pd.testing.assert_frame_equal(
            generated,
            canonical,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )


def test_canonical_eda_figure_manifest_and_files(
    temporary_canonical_eda_run: Mapping[str, Any],
) -> None:
    figures = temporary_canonical_eda_run["figures"]
    assert set(figures) == EXPECTED_FIGURES
    for name, path in figures.items():
        assert path.stat().st_size > 0, name
        if path.suffix.lower() == ".png":
            with path.open("rb") as handle:
                assert handle.read(8) == PNG_SIGNATURE
    html = figures["interactive_energy_loudness.html"]
    html_text = html.read_text(encoding="utf-8").lower()
    assert "<html" in html_text and "plotly" in html_text

    paths = temporary_canonical_eda_run["paths"]
    assert not list(paths.model_artifacts.iterdir())
    assert not list(paths.sql.iterdir())
