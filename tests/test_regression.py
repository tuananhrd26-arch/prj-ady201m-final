"""Deterministic characterization tests for popularity regression."""

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
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AUDIO_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "tempo",
    "loudness",
    "instrumentalness",
    "liveness",
    "speechiness",
]
EXTENDED_FEATURES = AUDIO_FEATURES + [
    "duration_ms",
    "year",
    "explicit",
    "key",
    "mode",
]
EXPECTED_EXPERIMENTS = {
    ("Audio Only", "Linear Regression"),
    ("Audio Only", "Ridge Regression"),
    ("Extended", "Linear Regression"),
    ("Extended", "Ridge Regression"),
}
CANONICAL_METRICS = {
    ("Audio Only", "Linear Regression"): (13.1118, 16.3080, 0.4442),
    ("Audio Only", "Ridge Regression"): (13.1119, 16.3080, 0.4442),
    ("Extended", "Linear Regression"): (7.9820, 10.7308, 0.7594),
    ("Extended", "Ridge Regression"): (7.9821, 10.7309, 0.7594),
}
ARTIFACT_NAME = "best_popularity_model.joblib"


@dataclass(frozen=True)
class RegressionRun:
    summary: dict[str, Any]
    paths: Any
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    artifact: Pipeline
    split_calls: tuple[dict[str, Any], ...]
    trained_pipelines: tuple[Pipeline, ...]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_module(
    project_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ModuleType, None, None]:
    """Import the script as a module without invoking its CLI entry point."""
    mpl_config = tmp_path_factory.mktemp("matplotlib-config")
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
def canonical_main_data(project_root: Path) -> pd.DataFrame:
    """Read the canonical main CSV once without mutating the source dataframe."""
    return pd.read_csv(project_root / "cleaned_data" / "data_clean.csv", low_memory=False)


@pytest.fixture(scope="session")
def canonical_regression_outputs(project_root: Path) -> Mapping[str, Any]:
    tables = project_root / "week7_outputs" / "tables"
    with (project_root / "week7_outputs" / "run_summary.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    return {
        "metrics": pd.read_csv(tables / "regression_metrics.csv"),
        "predictions": pd.read_csv(tables / "regression_actual_vs_predicted.csv"),
        "coefficients": pd.read_csv(tables / "regression_coefficients.csv"),
        "summary": summary,
        "artifact_path": (
            project_root
            / "week7_outputs"
            / "model_artifacts"
            / ARTIFACT_NAME
        ),
    }


@pytest.fixture(scope="session")
def synthetic_tracks() -> pd.DataFrame:
    """Build fast, varied data whose target uses audio and extended features."""
    rng = np.random.default_rng(20260716)
    rows = 360
    tracks = pd.DataFrame(
        {
            "danceability": rng.uniform(0.05, 0.95, rows),
            "energy": rng.uniform(0.05, 0.98, rows),
            "acousticness": rng.uniform(0.0, 1.0, rows),
            "valence": rng.uniform(0.02, 0.98, rows),
            "tempo": rng.uniform(55.0, 190.0, rows),
            "loudness": rng.uniform(-28.0, -1.0, rows),
            "instrumentalness": rng.uniform(0.0, 0.95, rows),
            "liveness": rng.uniform(0.01, 0.9, rows),
            "speechiness": rng.uniform(0.01, 0.7, rows),
            "duration_ms": rng.uniform(90000.0, 360000.0, rows),
            "year": rng.integers(1921, 2021, rows),
            "explicit": rng.integers(0, 2, rows),
            "key": rng.integers(0, 12, rows),
            "mode": rng.integers(0, 2, rows),
        }
    )
    noise = rng.normal(0.0, 0.35, rows)
    tracks["popularity"] = np.clip(
        11.0
        + 16.0 * tracks["danceability"]
        + 10.0 * tracks["energy"]
        - 7.0 * tracks["acousticness"]
        + 8.0 * tracks["valence"]
        + 0.025 * tracks["tempo"]
        + 0.20 * tracks["loudness"]
        - 2.0 * tracks["instrumentalness"]
        + 1.5 * tracks["liveness"]
        - 1.0 * tracks["speechiness"]
        + 0.000035 * tracks["duration_ms"]
        + 0.12 * (tracks["year"] - 1921)
        + 3.0 * tracks["explicit"]
        + 0.25 * tracks["key"]
        + 2.0 * tracks["mode"]
        + noise,
        0.0,
        100.0,
    )
    assert tracks.shape == (rows, len(EXTENDED_FEATURES) + 1)
    assert not tracks.isna().any().any()
    assert tracks[EXTENDED_FEATURES].nunique().gt(1).all()
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
    """Prove regression tests do not alter canonical data or outputs."""
    cleaned_root = project_root / "cleaned_data"
    outputs_root = project_root / "week7_outputs"
    cleaned_before = _sha256_manifest(cleaned_root)
    outputs_before = _sha256_manifest(outputs_root)
    yield
    assert _sha256_manifest(cleaned_root) == cleaned_before, "cleaned_data changed"
    assert _sha256_manifest(outputs_root) == outputs_before, "week7_outputs changed"


def _load_run(paths: Any, summary: dict[str, Any], **recording: Any) -> RegressionRun:
    artifact_path = paths.model_artifacts / ARTIFACT_NAME
    return RegressionRun(
        summary=summary,
        paths=paths,
        metrics=pd.read_csv(paths.tables / "regression_metrics.csv"),
        predictions=pd.read_csv(paths.tables / "regression_actual_vs_predicted.csv"),
        coefficients=pd.read_csv(paths.tables / "regression_coefficients.csv"),
        artifact=joblib.load(artifact_path),
        split_calls=tuple(recording.get("split_calls", ())),
        trained_pipelines=tuple(recording.get("trained_pipelines", ())),
    )


@pytest.fixture(scope="session")
def temporary_regression_run(
    project_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> RegressionRun:
    """Execute all four experiments once, entirely under pytest temporary storage."""
    run_root = tmp_path_factory.mktemp("regression-run")
    paths = project_module.make_paths(run_root, "outputs")
    split_calls: list[dict[str, Any]] = []
    trained_pipelines: list[Pipeline] = []

    def recording_split(*arrays: Any, **kwargs: Any) -> list[Any]:
        split_calls.append(
            {
                "test_size": kwargs.get("test_size"),
                "random_state": kwargs.get("random_state"),
                "shuffle": kwargs.get("shuffle", True),
                "stratify": kwargs.get("stratify"),
                "features": tuple(arrays[0].columns),
            }
        )
        return train_test_split(*arrays, **kwargs)

    def recording_pipeline(*args: Any, **kwargs: Any) -> Pipeline:
        pipeline = Pipeline(*args, **kwargs)
        trained_pipelines.append(pipeline)
        return pipeline

    patch = pytest.MonkeyPatch()
    patch.setattr(project_module, "train_test_split", recording_split)
    patch.setattr(project_module, "Pipeline", recording_pipeline)
    try:
        summary = project_module.regression_analysis(synthetic_tracks.copy(deep=True), paths)
    finally:
        patch.undo()
    return _load_run(
        paths,
        summary,
        split_calls=split_calls,
        trained_pipelines=trained_pipelines,
    )


def _experiment_pairs(metrics: pd.DataFrame) -> set[tuple[str, str]]:
    return set(metrics[["feature_set", "model"]].itertuples(index=False, name=None))


def _identity(summary_section: Mapping[str, Any]) -> tuple[str, str]:
    return str(summary_section["feature_set"]), str(summary_section["model"])


def _features_for_identity(identity: tuple[str, str]) -> list[str]:
    feature_set, _ = identity
    if feature_set == "Audio Only":
        return AUDIO_FEATURES
    if feature_set == "Extended":
        return EXTENDED_FEATURES
    raise AssertionError(f"Unknown feature set: {feature_set}")


def _model_name(estimator: Any) -> str:
    if isinstance(estimator, LinearRegression):
        return "Linear Regression"
    if isinstance(estimator, Ridge):
        return "Ridge Regression"
    raise AssertionError(f"Unexpected regression estimator: {type(estimator).__name__}")


def test_regression_constants(project_module: ModuleType) -> None:
    assert project_module.TARGET == "popularity"
    assert project_module.RANDOM_STATE == 42
    assert project_module.REGRESSION_AUDIO_FEATURES == AUDIO_FEATURES
    assert project_module.REGRESSION_EXTENDED_FEATURES == EXTENDED_FEATURES
    assert project_module.REGRESSION_FEATURES == EXTENDED_FEATURES


def test_canonical_metrics_experiment_manifest(
    canonical_regression_outputs: Mapping[str, Any],
) -> None:
    metrics = canonical_regression_outputs["metrics"]
    assert metrics.columns.tolist() == ["model", "feature_set", "MAE", "RMSE", "R2"]
    assert len(metrics) == 4
    assert not metrics.duplicated(subset=["feature_set", "model"]).any()
    assert _experiment_pairs(metrics) == EXPECTED_EXPERIMENTS
    for column in ("MAE", "RMSE", "R2"):
        assert is_numeric_dtype(metrics[column]), f"Canonical {column} must be numeric"
        assert np.isfinite(metrics[column].to_numpy()).all()
    assert metrics["MAE"].ge(0).all()
    assert metrics["RMSE"].ge(0).all()


def test_canonical_metric_values(
    canonical_regression_outputs: Mapping[str, Any],
) -> None:
    metrics = canonical_regression_outputs["metrics"].set_index(["feature_set", "model"])
    for identity, expected in CANONICAL_METRICS.items():
        actual = metrics.loc[identity, ["MAE", "RMSE", "R2"]].to_numpy(dtype=float)
        assert actual == pytest.approx(expected, abs=0.00005), (
            f"Canonical metrics changed for {identity}: {actual}"
        )


def test_canonical_best_model_identity(
    canonical_regression_outputs: Mapping[str, Any],
) -> None:
    metrics = canonical_regression_outputs["metrics"]
    summary = canonical_regression_outputs["summary"]
    linear_row = metrics.loc[
        (metrics["feature_set"] == "Extended")
        & (metrics["model"] == "Linear Regression")
    ].iloc[0]
    # The persisted table is rounded, so summary/artifact metadata resolve its R2 tie.
    assert linear_row["R2"] == metrics["R2"].max()
    assert _identity(summary["best_regression_model_by_R2"]) == (
        "Extended",
        "Linear Regression",
    )
    assert _identity(summary["regression_plot_model"]) == (
        "Extended",
        "Linear Regression",
    )
    assert ARTIFACT_NAME in summary["model_artifacts_created"]
    assert summary["regression_plot_model"]["model"] != "Ridge Regression"


def test_canonical_artifact_structure_and_prediction(
    canonical_regression_outputs: Mapping[str, Any],
    canonical_main_data: pd.DataFrame,
) -> None:
    artifact = joblib.load(canonical_regression_outputs["artifact_path"])
    assert isinstance(artifact, Pipeline)
    assert list(artifact.named_steps) == ["scaler", "model"]
    assert isinstance(artifact.named_steps["scaler"], StandardScaler)
    assert isinstance(artifact.named_steps["model"], LinearRegression)
    assert artifact.n_features_in_ == len(EXTENDED_FEATURES)
    assert artifact.feature_names_in_.tolist() == EXTENDED_FEATURES
    sample = canonical_main_data.loc[:, EXTENDED_FEATURES].head(8).copy()
    predictions = artifact.predict(sample)
    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()


def test_deterministic_split_configuration(
    temporary_regression_run: RegressionRun,
) -> None:
    calls = temporary_regression_run.split_calls
    assert len(calls) == 2, f"Expected two feature-set splits, got {len(calls)}"
    assert [call["features"] for call in calls] == [
        tuple(AUDIO_FEATURES),
        tuple(EXTENDED_FEATURES),
    ]
    for call in calls:
        assert call["test_size"] == pytest.approx(0.2)
        assert call["random_state"] == 42
        assert call["shuffle"] is True
        assert call["stratify"] is None


def test_four_models_are_trained(temporary_regression_run: RegressionRun) -> None:
    metrics = temporary_regression_run.metrics
    assert len(metrics) == 4
    assert _experiment_pairs(metrics) == EXPECTED_EXPERIMENTS
    assert not metrics.duplicated(subset=["feature_set", "model"]).any()
    assert np.isfinite(metrics[["MAE", "RMSE", "R2"]].to_numpy()).all()


def test_standard_scaler_in_every_pipeline(
    temporary_regression_run: RegressionRun,
) -> None:
    pipelines = temporary_regression_run.trained_pipelines
    assert len(pipelines) == 4
    for pipeline in pipelines:
        assert list(pipeline.named_steps) == ["scaler", "model"]
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    estimators = [pipeline.named_steps["model"] for pipeline in pipelines]
    assert sum(isinstance(estimator, LinearRegression) for estimator in estimators) == 2
    ridge_estimators = [estimator for estimator in estimators if isinstance(estimator, Ridge)]
    assert len(ridge_estimators) == 2
    assert all(estimator.alpha == pytest.approx(10.0) for estimator in ridge_estimators)
    assert all(estimator.random_state == 42 for estimator in ridge_estimators)


def test_selected_model_metric_formulas(
    temporary_regression_run: RegressionRun,
) -> None:
    selected = _identity(temporary_regression_run.summary["best_model"])
    selected_row = temporary_regression_run.metrics.set_index(
        ["feature_set", "model"]
    ).loc[selected]
    actual = temporary_regression_run.predictions["actual"]
    predicted = temporary_regression_run.predictions["predicted"]
    recalculated = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }
    for metric, value in recalculated.items():
        assert float(selected_row[metric]) == pytest.approx(value, abs=0.00015), (
            f"Stored {metric} does not match the selected-model prediction table"
        )


def test_best_plot_and_saved_model_identity_are_consistent(
    temporary_regression_run: RegressionRun,
) -> None:
    best_identity = _identity(temporary_regression_run.summary["best_model"])
    plot_identity = _identity(temporary_regression_run.summary["plot_model"])
    metrics = temporary_regression_run.metrics
    maximum = metrics["R2"].max()
    assert float(
        metrics.loc[
            (metrics["feature_set"] == best_identity[0])
            & (metrics["model"] == best_identity[1]),
            "R2",
        ].iloc[0]
    ) == maximum
    saved_identity = (
        "Extended"
        if temporary_regression_run.artifact.n_features_in_ == len(EXTENDED_FEATURES)
        else "Audio Only",
        _model_name(temporary_regression_run.artifact.named_steps["model"]),
    )
    assert best_identity == plot_identity == saved_identity
    assert (temporary_regression_run.paths.model_artifacts / ARTIFACT_NAME).is_file()


def test_artifact_reload_predictions_match_saved_table(
    temporary_regression_run: RegressionRun,
    synthetic_tracks: pd.DataFrame,
) -> None:
    selected = _identity(temporary_regression_run.summary["best_model"])
    features = _features_for_identity(selected)
    model_df = synthetic_tracks.dropna(subset=["popularity"] + features).copy()
    _, x_test, _, y_test = train_test_split(
        model_df[features],
        model_df["popularity"],
        test_size=0.2,
        random_state=42,
    )
    artifact = joblib.load(
        temporary_regression_run.paths.model_artifacts / ARTIFACT_NAME
    )
    reloaded_predictions = artifact.predict(x_test)
    np.testing.assert_allclose(
        temporary_regression_run.predictions["actual"].to_numpy(),
        y_test.to_numpy(),
        rtol=0,
        atol=0.000051,
    )
    np.testing.assert_allclose(
        temporary_regression_run.predictions["predicted"].to_numpy(),
        reloaded_predictions,
        rtol=0,
        atol=0.000051,
    )


def test_selected_coefficient_manifest(
    temporary_regression_run: RegressionRun,
) -> None:
    coefficients = temporary_regression_run.coefficients
    assert not coefficients.empty
    assert coefficients.columns.tolist() == [
        "model",
        "feature_set",
        "feature",
        "coefficient",
        "abs_coefficient",
    ]
    selected = _identity(temporary_regression_run.summary["best_model"])
    selected_coefficients = coefficients.loc[
        (coefficients["feature_set"] == selected[0])
        & (coefficients["model"] == selected[1])
    ]
    expected_features = _features_for_identity(selected)
    assert len(selected_coefficients) == len(expected_features)
    assert not selected_coefficients["feature"].duplicated().any()
    assert set(selected_coefficients["feature"]) == set(expected_features)
    assert is_numeric_dtype(selected_coefficients["coefficient"])
    assert np.isfinite(selected_coefficients["coefficient"].to_numpy()).all()


def test_temporary_output_manifest(temporary_regression_run: RegressionRun) -> None:
    required = [
        temporary_regression_run.paths.tables / "regression_metrics.csv",
        temporary_regression_run.paths.tables / "regression_actual_vs_predicted.csv",
        temporary_regression_run.paths.tables / "regression_coefficients.csv",
        temporary_regression_run.paths.figures / "regression_actual_vs_predicted.png",
        temporary_regression_run.paths.figures / "regression_residuals.png",
        temporary_regression_run.paths.figures / "regression_coefficients.png",
        temporary_regression_run.paths.model_artifacts / ARTIFACT_NAME,
    ]
    for path in required:
        assert path.is_file(), f"Temporary regression output missing: {path.name}"
        assert path.stat().st_size > 0, f"Temporary regression output is empty: {path.name}"
    png_signature = b"\x89PNG\r\n\x1a\n"
    for path in required:
        if path.suffix == ".png":
            with path.open("rb") as handle:
                assert handle.read(8) == png_signature, f"Invalid PNG signature: {path.name}"


def test_synthetic_regression_is_repeatable(
    project_module: ModuleType,
    synthetic_tracks: pd.DataFrame,
    temporary_regression_run: RegressionRun,
    tmp_path: Path,
) -> None:
    second_paths = project_module.make_paths(tmp_path, "repeat-output")
    second_summary = project_module.regression_analysis(
        synthetic_tracks.copy(deep=True),
        second_paths,
    )
    second = _load_run(second_paths, second_summary)
    assert _identity(second.summary["best_model"]) == _identity(
        temporary_regression_run.summary["best_model"]
    )
    assert _identity(second.summary["plot_model"]) == _identity(
        temporary_regression_run.summary["plot_model"]
    )
    pd.testing.assert_frame_equal(
        second.metrics,
        temporary_regression_run.metrics,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        second.predictions,
        temporary_regression_run.predictions,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        second.coefficients,
        temporary_regression_run.coefficients,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
