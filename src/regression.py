"""Popularity-regression experiments, diagnostics, and artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_POPULARITY_MODEL_FILENAME,
    ProjectPaths,
    RANDOM_STATE,
    REGRESSION_AUDIO_FEATURES,
    REGRESSION_EXTENDED_FEATURES,
    REGRESSION_FEATURES,
    TARGET,
)
from src.validation import require_columns


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def regression_analysis(
    tracks: pd.DataFrame,
    paths: ProjectPaths,
) -> Dict[str, Any]:
    feature_sets = {
        "Audio Only": REGRESSION_AUDIO_FEATURES,
        "Extended": REGRESSION_EXTENDED_FEATURES,
    }
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    }

    metrics_rows = []
    coefficient_frames = []
    fitted_results = []

    for feature_set_name, requested_features in feature_sets.items():
        require_columns(
            tracks,
            requested_features,
            context=f"Regression feature set '{feature_set_name}'",
            missing_label="columns",
        )

        model_df = tracks.dropna(subset=[TARGET] + requested_features).copy()
        X = model_df[requested_features]
        y = model_df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        for model_name, estimator in models.items():
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            metrics_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )

            coefs = pipeline.named_steps["model"].coef_
            coefficient_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "feature": requested_features,
                        "coefficient": coefs,
                        "abs_coefficient": np.abs(coefs),
                    }
                )
            )
            fitted_results.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "pipeline": pipeline,
                    "y_test": y_test,
                    "predictions": y_pred,
                    "r2": r2,
                }
            )

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "feature_set", "MAE", "RMSE", "R2"]
    ].round(4)
    coefficients = pd.concat(coefficient_frames, ignore_index=True).sort_values(
        ["feature_set", "model", "abs_coefficient"], ascending=[True, True, False]
    )
    coefficients = coefficients[
        ["model", "feature_set", "feature", "coefficient", "abs_coefficient"]
    ]

    _save_table(metrics, paths.tables / "regression_metrics.csv")
    _save_table(coefficients.round(5), paths.tables / "regression_coefficients.csv")

    if not fitted_results:
        raise RuntimeError("Regression model was not fitted correctly.")

    best_result = max(fitted_results, key=lambda result: result["r2"])
    selected_result = best_result
    selected_model_name = str(selected_result["model"])
    selected_feature_set = str(selected_result["feature_set"])
    selected_label = f"{selected_feature_set} {selected_model_name}"
    selected_pipeline = selected_result["pipeline"]
    y_test = selected_result["y_test"]
    predictions = selected_result["predictions"]

    joblib.dump(
        selected_pipeline,
        paths.model_artifacts / BEST_POPULARITY_MODEL_FILENAME,
    )

    # Actual vs predicted plot.
    prediction_frame = pd.DataFrame({"actual": y_test.values, "predicted": predictions})
    _save_table(prediction_frame.round(4), paths.tables / "regression_actual_vs_predicted.csv")

    sample = prediction_frame.sample(min(3000, len(prediction_frame)), random_state=RANDOM_STATE)
    plt.figure(figsize=(6, 6))
    plt.scatter(sample["actual"], sample["predicted"], alpha=0.25, s=10)
    min_value = min(sample["actual"].min(), sample["predicted"].min())
    max_value = max(sample["actual"].max(), sample["predicted"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.title(f"Actual vs Predicted Popularity\n{selected_label}")
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.tight_layout()
    plt.savefig(paths.figures / "regression_actual_vs_predicted.png", dpi=180)
    plt.close()

    # Residual plot.
    sample_residuals = sample.copy()
    sample_residuals["residual"] = sample_residuals["actual"] - sample_residuals["predicted"]
    plt.figure(figsize=(7, 5))
    plt.scatter(sample_residuals["predicted"], sample_residuals["residual"], alpha=0.25, s=10)
    plt.axhline(0, linestyle="--")
    plt.title(f"Regression Residual Plot\n{selected_label}")
    plt.xlabel("Predicted Popularity")
    plt.ylabel("Residual: Actual - Predicted")
    plt.tight_layout()
    plt.savefig(paths.figures / "regression_residuals.png", dpi=180)
    plt.close()

    # Coefficient plot for the model used in the prediction and residual plots.
    selected_coefs = coefficients[
        (coefficients["model"] == selected_model_name)
        & (coefficients["feature_set"] == selected_feature_set)
    ].copy()
    selected_coefs = selected_coefs.sort_values("coefficient")

    plt.figure(figsize=(8, 6))
    plt.barh(selected_coefs["feature"], selected_coefs["coefficient"])
    plt.title(f"Regression Coefficients for Popularity\n{selected_label}")
    plt.xlabel("Standardized Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(paths.figures / "regression_coefficients.png", dpi=180)
    plt.close()

    return {
        "models_trained": [
            f"{row['feature_set']} - {row['model']}"
            for row in metrics_rows
        ],
        "best_model": {
            "model": best_result["model"],
            "feature_set": best_result["feature_set"],
            "R2": round(float(best_result["r2"]), 4),
        },
        "metrics": metrics.to_dict(orient="records"),
        "plot_model": {
            "model": selected_model_name,
            "feature_set": selected_feature_set,
        },
    }
