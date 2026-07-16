"""
Spotify Week 7 Analysis
=======================
Clean, reproducible code for the ADY201M Spotify project.

What this script produces:
1. SQL analysis outputs from the cleaned CSV files.
2. Python EDA summary tables and figures.
3. Regression analysis for explaining/predicting track popularity.
4. A small content-based recommendation demo using audio-feature similarity.

How to run from the project folder:
    python spotify_week7_analysis.py --root .

Expected project structure:
    cleaned_data/data_clean.csv
    cleaned_data/data_by_artist_clean.csv
    cleaned_data/data_by_genres_clean.csv
    cleaned_data/data_by_year_clean.csv

Outputs are saved to:
    week7_outputs/tables
    week7_outputs/figures
    week7_outputs/sql
    week7_outputs/model_artifacts
"""

from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import (
    BEST_POPULARITY_MODEL_FILENAME,
    DEFAULT_OUTPUT_DIRNAME,
    FIGURES_DIRNAME,
    MODEL_ARTIFACTS_DIRNAME,
    ProjectPaths,
    RANDOM_STATE,
    RECOMMENDER_CATALOG_FILENAME,
    RECOMMENDER_FEATURES,
    RECOMMENDER_FEATURES_FILENAME,
    RECOMMENDER_NEIGHBORS_FILENAME,
    RECOMMENDER_SCALER_FILENAME,
    REGRESSION_AUDIO_FEATURES,
    REGRESSION_EXTENDED_FEATURES,
    REGRESSION_FEATURES,
    SQL_DIRNAME,
    TABLES_DIRNAME,
    TARGET,
    TREND_FEATURES,
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_paths(root: Path, output_dir: str = DEFAULT_OUTPUT_DIRNAME) -> ProjectPaths:
    output = root / output_dir
    paths = ProjectPaths(
        root=root,
        output=output,
        tables=output / TABLES_DIRNAME,
        figures=output / FIGURES_DIRNAME,
        sql=output / SQL_DIRNAME,
        model_artifacts=output / MODEL_ARTIFACTS_DIRNAME,
    )
    for path in [paths.output, paths.tables, paths.figures, paths.sql, paths.model_artifacts]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_project_data(
    root: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    """Load cleaned Spotify CSV files. Falls back to raw data/data.csv if needed."""
    data: Dict[str, pd.DataFrame] = {}
    input_files: Dict[str, Path] = {}

    tracks_path = root / "cleaned_data" / "data_clean.csv"
    if not tracks_path.exists():
        tracks_path = root / "data" / "data.csv"
    if not tracks_path.exists():
        raise FileNotFoundError("Could not find cleaned_data/data_clean.csv or data/data.csv")

    data["tracks"] = pd.read_csv(tracks_path)
    input_files["tracks"] = tracks_path

    optional_files = {
        "artist_features": root / "cleaned_data" / "data_by_artist_clean.csv",
        "genre_features": root / "cleaned_data" / "data_by_genres_clean.csv",
        "year_features": root / "cleaned_data" / "data_by_year_clean.csv",
        "artist_genres": root / "cleaned_data" / "data_w_genres_clean.csv",
    }
    for name, path in optional_files.items():
        frame = read_csv_if_exists(path)
        if frame is not None:
            data[name] = frame
            input_files[name] = path

    return data, input_files


def clean_tracks_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return a non-destructive cleaned copy suitable for EDA and modeling."""
    tracks = df.copy()

    # Standardize a few expected columns.
    if "release_year" in tracks.columns and "year" not in tracks.columns:
        tracks["year"] = tracks["release_year"]

    numeric_candidates = sorted(set(REGRESSION_FEATURES + RECOMMENDER_FEATURES + [TARGET]))
    for col in numeric_candidates:
        if col in tracks.columns:
            tracks[col] = pd.to_numeric(tracks[col], errors="coerce")

    # Keep only rows with essential identifiers.
    required_identity = [col for col in ["id", "name"] if col in tracks.columns]
    if required_identity:
        tracks = tracks.dropna(subset=required_identity)

    # Fill small numeric missingness with median to avoid breaking models.
    numeric_cols = tracks.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if tracks[col].isna().any():
            tracks[col] = tracks[col].fillna(tracks[col].median())

    # Derived column for decade-level analysis.
    if "year" in tracks.columns:
        tracks["decade"] = (tracks["year"].astype(int) // 10) * 10

    return tracks.reset_index(drop=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_tracks_by_decade(tracks: pd.DataFrame, fig_path: Path) -> None:
    decade_counts = tracks.groupby("decade", as_index=False).size()
    decade_counts = decade_counts.rename(columns={"size": "track_count"})

    plt.figure(figsize=(10, 5))
    plt.bar(decade_counts["decade"].astype(str), decade_counts["track_count"])
    plt.title("Number of Tracks by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Number of Tracks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_popularity_distribution(tracks: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(tracks[TARGET].dropna(), bins=25, edgecolor="white")
    plt.title("Distribution of Track Popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_popularity_by_decade_boxplot(
    tracks: pd.DataFrame,
    fig_path: Path,
) -> None:
    decades = sorted(tracks["decade"].dropna().unique())
    values = [
        tracks.loc[tracks["decade"] == decade, TARGET].dropna().values
        for decade in decades
    ]

    plt.figure(figsize=(11, 6))
    plt.boxplot(values, tick_labels=[str(int(decade)) for decade in decades], showfliers=False)
    plt.title("Track Popularity by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Popularity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_feature_trends(tracks: pd.DataFrame, fig_path: Path) -> pd.DataFrame:
    trend = tracks.groupby("year", as_index=False)[TREND_FEATURES].mean()

    plt.figure(figsize=(10, 5))
    for feature in TREND_FEATURES:
        plt.plot(trend["year"], trend[feature], label=feature)
    plt.title("Average Audio Feature Trends by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    return trend


def plot_correlation_heatmap(tracks: pd.DataFrame, features: List[str], fig_path: Path) -> pd.DataFrame:
    corr = tracks[features].corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix of Spotify Audio Features")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    return corr.reset_index().rename(columns={"index": "feature"})


def plot_interactive_energy_loudness(
    tracks: pd.DataFrame,
    html_path: Path,
    preview_path: Path,
    sample_size: int = 10000,
) -> str:
    required_cols = ["energy", "loudness", "decade", "name", "artists", TARGET, "year", "danceability"]
    missing_cols = [col for col in required_cols if col not in tracks.columns]
    if missing_cols:
        return "Interactive Plotly scatter skipped: missing columns " + ", ".join(missing_cols)

    plot_df = tracks.dropna(subset=required_cols).copy()
    if plot_df.empty:
        return "Interactive Plotly scatter skipped: no complete rows available."

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=RANDOM_STATE)

    plot_df["decade"] = plot_df["decade"].astype(int).astype(str)
    fig = px.scatter(
        plot_df,
        x="energy",
        y="loudness",
        color="decade",
        hover_data=["name", "artists", TARGET, "year", "danceability"],
        title="Interactive Energy vs Loudness by Decade",
        labels={
            "energy": "Energy",
            "loudness": "Loudness",
            "decade": "Decade",
            TARGET: "Popularity",
        },
    )
    fig.write_html(html_path, include_plotlyjs="cdn")

    try:
        fig.write_image(preview_path, width=1200, height=800, scale=2)
    except Exception as exc:
        return (
            "Interactive Plotly HTML created; static PNG preview unavailable. "
            f"Plotly image export requires Kaleido and compatible browser support. Error: {exc}"
        )

    return "Interactive Plotly HTML and static PNG preview created successfully."


def parse_artist_names(value: Any) -> List[str]:
    """Convert the serialized artist list in the track data to clean names."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value.strip()]

    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def create_genre_decade_pivot(
    tracks: pd.DataFrame,
    artist_genres: pd.DataFrame,
    max_genres: int = 30,
) -> Tuple[pd.DataFrame | None, str]:
    """Build a report-sized genre-by-decade popularity pivot via artist matching."""
    required_track_cols = {"artists", "decade", TARGET}
    required_genre_cols = {"artists", "genres_clean"}
    if not required_track_cols.issubset(tracks.columns):
        return None, "Genre pivot skipped: track-level artist/decade/popularity columns are incomplete."
    if not required_genre_cols.issubset(artist_genres.columns):
        return None, "Genre pivot skipped: artist-to-genre columns are incomplete."

    track_artist = tracks[["artists", "decade", TARGET]].copy()
    track_artist["artist"] = track_artist["artists"].map(parse_artist_names)
    track_artist = track_artist.explode("artist").dropna(subset=["artist"])
    track_artist = track_artist.drop(columns="artists")

    genre_lookup = artist_genres[["artists", "genres_clean"]].copy()
    genre_lookup = genre_lookup.dropna(subset=["artists", "genres_clean"])
    genre_lookup["genre"] = genre_lookup["genres_clean"].astype(str).str.split(";")
    genre_lookup = genre_lookup.explode("genre")
    genre_lookup["artist"] = genre_lookup["artists"].astype(str).str.strip()
    genre_lookup["genre"] = genre_lookup["genre"].astype(str).str.strip()
    genre_lookup = genre_lookup.loc[genre_lookup["genre"] != "", ["artist", "genre"]]
    genre_lookup = genre_lookup.drop_duplicates()

    matched = track_artist.merge(genre_lookup, on="artist", how="inner")
    if matched.empty:
        return None, "Genre pivot skipped: no track artists matched the artist-to-genre data."

    top_genres = (
        matched.groupby("genre")
        .size()
        .nlargest(max_genres)
        .index
    )
    matched = matched[matched["genre"].isin(top_genres)]
    pivot = matched.pivot_table(
        index="genre",
        columns="decade",
        values=TARGET,
        aggfunc="mean",
    )
    pivot = pivot.sort_index().rename(
        columns={decade: str(int(decade)) for decade in pivot.columns}
    )
    pivot = pivot.reset_index()
    note = (
        f"Genre pivot created for the {len(top_genres)} genres with the most "
        "track-artist matches."
    )
    return pivot, note


def create_eda_outputs(data: Dict[str, pd.DataFrame], paths: ProjectPaths) -> pd.DataFrame:
    tracks = clean_tracks_for_analysis(data["tracks"])

    overview = pd.DataFrame(
        [
            {"table": name, "rows": len(frame), "columns": frame.shape[1]}
            for name, frame in data.items()
        ]
    )
    save_table(overview, paths.tables / "dataset_overview.csv")

    descriptive_features = [f for f in REGRESSION_FEATURES + [TARGET] if f in tracks.columns]
    descriptive = tracks[descriptive_features].describe().T.reset_index().rename(columns={"index": "feature"})
    save_table(descriptive, paths.tables / "descriptive_statistics.csv")

    missing = (
        tracks.isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing["missing_rate"] = missing["missing_count"] / len(tracks)
    save_table(missing, paths.tables / "missing_values_after_cleaning.csv")

    decade_summary = (
        tracks.groupby("decade", as_index=False)
        .agg(track_count=("id", "count"), avg_popularity=(TARGET, "mean"))
        .round(3)
    )
    save_table(decade_summary, paths.tables / "tracks_by_decade.csv")
    plot_tracks_by_decade(tracks, paths.figures / "tracks_by_decade.png")
    plot_popularity_distribution(
        tracks,
        paths.figures / "popularity_distribution.png",
    )
    plot_popularity_by_decade_boxplot(
        tracks,
        paths.figures / "popularity_by_decade_boxplot.png",
    )

    decade_feature_summary = (
        tracks.groupby("decade", as_index=False)
        .agg(
            total_tracks=(TARGET, "size"),
            avg_popularity=(TARGET, "mean"),
            avg_energy=("energy", "mean"),
            avg_danceability=("danceability", "mean"),
            avg_acousticness=("acousticness", "mean"),
            avg_valence=("valence", "mean"),
        )
        .round(4)
    )
    save_table(
        decade_feature_summary,
        paths.tables / "decade_feature_summary.csv",
    )

    trend = plot_feature_trends(tracks, paths.figures / "audio_feature_trends_by_year.png")
    save_table(trend.round(4), paths.tables / "audio_feature_trends_by_year.csv")

    corr_features = [f for f in RECOMMENDER_FEATURES + [TARGET] if f in tracks.columns]
    corr = plot_correlation_heatmap(tracks, corr_features, paths.figures / "correlation_heatmap.png")
    save_table(corr.round(4), paths.tables / "correlation_matrix.csv")

    if {"decade", "explicit"}.issubset(tracks.columns):
        decade_explicit_summary = (
            tracks.groupby(["decade", "explicit"])
            .agg(
                total_tracks=(TARGET, "size"),
                average_popularity=(TARGET, "mean"),
                average_energy=("energy", "mean"),
                average_danceability=("danceability", "mean"),
                average_acousticness=("acousticness", "mean"),
            )
            .round(4)
        )
        save_table(
            decade_explicit_summary.reset_index(),
            paths.tables / "decade_explicit_multiindex_summary.csv",
        )

    tracks.attrs["interactive_plot_note"] = plot_interactive_energy_loudness(
        tracks,
        paths.figures / "interactive_energy_loudness.html",
        paths.figures / "interactive_energy_loudness_preview.png",
    )

    if "genre_features" in data:
        genre_df = data["genre_features"].copy()
        genre_name_col = "genres_clean" if "genres_clean" in genre_df.columns else "genres"
        profile_cols = [genre_name_col, "popularity", "energy", "danceability", "acousticness", "valence", "tempo"]
        profile_cols = [c for c in profile_cols if c in genre_df.columns]
        genre_profile = genre_df[profile_cols].dropna(subset=[genre_name_col]).sort_values("popularity", ascending=False).head(20)
        save_table(genre_profile.round(4), paths.tables / "top_genres_audio_profile.csv")

    genre_pivot_note = "Genre pivot skipped: artist-to-genre data was not loaded."
    if "artist_genres" in data:
        genre_pivot, genre_pivot_note = create_genre_decade_pivot(
            tracks,
            data["artist_genres"],
        )
        if genre_pivot is not None:
            save_table(
                genre_pivot.round(4),
                paths.tables / "genre_decade_popularity_pivot.csv",
            )
    tracks.attrs["genre_pivot_note"] = genre_pivot_note

    return tracks


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
        missing_features = [
            feature for feature in requested_features if feature not in tracks.columns
        ]
        if missing_features:
            raise ValueError(
                f"Regression feature set '{feature_set_name}' is missing columns: "
                + ", ".join(missing_features)
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

    save_table(metrics, paths.tables / "regression_metrics.csv")
    save_table(coefficients.round(5), paths.tables / "regression_coefficients.csv")

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
    save_table(prediction_frame.round(4), paths.tables / "regression_actual_vs_predicted.csv")

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


def build_recommender_demo(
    tracks: pd.DataFrame,
    paths: ProjectPaths,
    n_examples: int = 5,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    features = [feature for feature in RECOMMENDER_FEATURES if feature in tracks.columns]
    required_cols = ["name", "artists"] + features
    if "name" not in tracks.columns or "artists" not in tracks.columns:
        return {"status": "skipped", "reason": "Missing name or artists columns."}

    model_df = tracks.dropna(subset=required_cols).copy().reset_index(drop=True)
    if model_df.empty:
        return {"status": "skipped", "reason": "No complete recommendation rows."}
    model_df["_model_index"] = model_df.index

    catalog_columns = [
        "_model_index",
        "id",
        "name",
        "artists",
        "year",
        TARGET,
        *RECOMMENDER_FEATURES,
    ]
    missing_catalog_columns = [
        column for column in catalog_columns if column not in model_df.columns
    ]
    if missing_catalog_columns:
        raise ValueError(
            "Recommender catalog is missing required columns: "
            + ", ".join(missing_catalog_columns)
        )
    if features != RECOMMENDER_FEATURES:
        raise ValueError("Recommender feature order does not match RECOMMENDER_FEATURES.")

    catalog = model_df[catalog_columns].copy()
    expected_model_indexes = np.arange(len(model_df))
    if not np.array_equal(catalog["_model_index"].to_numpy(), expected_model_indexes):
        raise ValueError("Recommender catalog model indexes are not contiguous and row-aligned.")
    if not catalog["_model_index"].is_unique:
        raise ValueError("Recommender catalog model indexes are not unique.")
    if not catalog["id"].is_unique:
        raise ValueError("Recommender catalog track ids are not unique.")
    required_non_null = ["id", "name", "artists", *RECOMMENDER_FEATURES]
    if catalog[required_non_null].isna().any().any():
        raise ValueError("Recommender catalog contains missing identity or feature values.")
    if not np.isfinite(catalog[RECOMMENDER_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Recommender catalog contains non-finite feature values.")
    if catalog.columns[-len(RECOMMENDER_FEATURES):].tolist() != RECOMMENDER_FEATURES:
        raise ValueError("Recommender catalog feature columns are out of order.")

    scaler = StandardScaler()
    X = scaler.fit_transform(model_df[features])

    if len(catalog) != len(X):
        raise ValueError("Recommender catalog row count does not match the feature matrix.")

    candidate_count = len(model_df)
    nn = NearestNeighbors(n_neighbors=candidate_count, metric="cosine")
    nn.fit(X)

    if len(catalog) != nn.n_samples_fit_:
        raise ValueError("Recommender catalog row count does not match the fitted model.")

    joblib.dump(scaler, paths.model_artifacts / RECOMMENDER_SCALER_FILENAME)
    joblib.dump(nn, paths.model_artifacts / RECOMMENDER_NEIGHBORS_FILENAME)
    (paths.model_artifacts / RECOMMENDER_FEATURES_FILENAME).write_text(
        json.dumps(features, indent=2),
        encoding="utf-8",
    )
    catalog_path = paths.model_artifacts / RECOMMENDER_CATALOG_FILENAME
    save_table(catalog, catalog_path)

    demo_inputs = (
        model_df.sort_values(TARGET, ascending=False)
        .drop_duplicates(subset=["name", "artists"])
        .head(n_examples)
        .copy()
    )

    rows = []
    validation_rows = []
    for _, input_row in demo_inputs.iterrows():
        input_idx = int(input_row["_model_index"])
        query_row = model_df.iloc[input_idx]
        input_key = (str(input_row["name"]), str(input_row["artists"]))
        query_key = (str(query_row["name"]), str(query_row["artists"]))
        label_matches_query_vector = input_key == query_key

        distances, indices = nn.kneighbors(
            X[input_idx].reshape(1, -1),
            n_neighbors=candidate_count,
        )

        seen_recommendations = set()
        ranks = []
        formula_checks = []
        finite_checks = []

        for distance, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx == input_idx:
                continue

            neighbor = model_df.iloc[neighbor_idx]
            recommendation_key = (str(neighbor["name"]), str(neighbor["artists"]))
            if recommendation_key == input_key or recommendation_key in seen_recommendations:
                continue

            similarity = 1 - float(distance)
            formula_checks.append(np.isclose(similarity, 1 - float(distance)))
            finite_checks.append(np.isfinite(similarity))
            seen_recommendations.add(recommendation_key)
            rank = len(seen_recommendations)
            ranks.append(rank)

            rows.append(
                {
                    "input_song": input_row["name"],
                    "input_artists": input_row["artists"],
                    "input_model_index": input_idx,
                    "rank": rank,
                    "recommended_song": neighbor["name"],
                    "recommended_artists": neighbor["artists"],
                    "recommended_model_index": neighbor_idx,
                    "year": neighbor.get("year", np.nan),
                    "popularity": neighbor.get(TARGET, np.nan),
                    "cosine_distance": float(distance),
                    "similarity": similarity,
                }
            )

            if len(seen_recommendations) >= n_neighbors:
                break

        input_rows = [
            row for row in rows
            if row["input_song"] == input_row["name"]
            and row["input_artists"] == input_row["artists"]
        ]
        recommended_keys = [
            (str(row["recommended_song"]), str(row["recommended_artists"]))
            for row in input_rows
        ]
        expected_ranks = list(range(1, len(input_rows) + 1))
        exact_top_n = len(input_rows) == n_neighbors
        input_track_absent = input_key not in recommended_keys
        no_duplicate_recommendations = len(recommended_keys) == len(set(recommended_keys))
        rank_consecutive = ranks == expected_ranks
        similarity_formula_valid = bool(formula_checks) and all(formula_checks)
        finite_similarity = bool(finite_checks) and all(finite_checks)
        validation_passed = all(
            [
                label_matches_query_vector,
                input_track_absent,
                no_duplicate_recommendations,
                similarity_formula_valid,
                finite_similarity,
                rank_consecutive,
                exact_top_n,
            ]
        )
        validation_rows.append(
            {
                "input_song": input_row["name"],
                "input_artists": input_row["artists"],
                "input_model_index": input_idx,
                "recommendations_returned": len(input_rows),
                "label_matches_query_vector": label_matches_query_vector,
                "input_track_absent": input_track_absent,
                "no_duplicate_recommendations": no_duplicate_recommendations,
                "similarity_formula_valid": similarity_formula_valid,
                "finite_similarity": finite_similarity,
                "rank_consecutive": rank_consecutive,
                "exact_top_n": exact_top_n,
                "validation_passed": validation_passed,
            }
        )

    recommendation_output = pd.DataFrame(rows)
    validation_output = pd.DataFrame(validation_rows)
    save_table(recommendation_output.round(4), paths.tables / "recommendation_demo_results.csv")
    save_table(validation_output, paths.tables / "recommendation_validation.csv")

    return {
        "status": "created",
        "features": features,
        "scaler": "StandardScaler",
        "metric": "cosine",
        "requested_unique_recommendations_per_input": n_neighbors,
        "seed_index_preserved": True,
        "validation_passed": bool(validation_output["validation_passed"].all()) if not validation_output.empty else False,
        "validation_file": str(paths.tables / "recommendation_validation.csv"),
        "catalog_file": str(catalog_path),
        "catalog_rows": int(len(catalog)),
    }


def create_sqlite_database(data: Dict[str, pd.DataFrame], paths: ProjectPaths) -> Path:
    db_path = paths.sql / "spotify_week7.sqlite"
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        table_map = {
            "tracks": "tracks",
            "artist_features": "artist_features",
            "genre_features": "genre_features",
            "year_features": "year_features",
            "artist_genres": "artist_genres",
        }
        for source_name, table_name in table_map.items():
            if source_name in data:
                frame = data[source_name].copy()
                frame.to_sql(table_name, conn, index=False, if_exists="replace")

    return db_path


SQL_QUERIES: Dict[str, str] = {
    "01_tracks_by_decade": """
        SELECT
            CAST(year / 10 AS INTEGER) * 10 AS decade,
            COUNT(*) AS total_tracks,
            ROUND(AVG(popularity), 2) AS avg_popularity,
            ROUND(AVG(energy), 3) AS avg_energy,
            ROUND(AVG(danceability), 3) AS avg_danceability
        FROM tracks
        GROUP BY CAST(year / 10 AS INTEGER) * 10
        ORDER BY decade;
    """,
    "02_audio_features_by_year": """
        SELECT
            year,
            ROUND(AVG(energy), 3) AS avg_energy,
            ROUND(AVG(danceability), 3) AS avg_danceability,
            ROUND(AVG(acousticness), 3) AS avg_acousticness,
            ROUND(AVG(valence), 3) AS avg_valence,
            ROUND(AVG(popularity), 2) AS avg_popularity
        FROM tracks
        GROUP BY year
        ORDER BY year;
    """,
    "03_top_tracks_by_decade_window": """
        WITH ranked_tracks AS (
            SELECT
                name AS track_name,
                artists,
                popularity,
                year,
                CAST(year / 10 AS INTEGER) * 10 AS decade,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(year / 10 AS INTEGER) * 10
                    ORDER BY popularity DESC, name ASC
                ) AS rank_in_decade
            FROM tracks
        )
        SELECT *
        FROM ranked_tracks
        WHERE rank_in_decade <= 5
        ORDER BY decade, rank_in_decade;
    """,
    "04_above_average_popularity_subquery": """
        SELECT
            name AS track_name,
            artists,
            year,
            popularity,
            ROUND(popularity - (SELECT AVG(popularity) FROM tracks), 2) AS popularity_gap
        FROM tracks
        WHERE popularity > (SELECT AVG(popularity) FROM tracks)
        ORDER BY popularity_gap DESC, track_name ASC
        LIMIT 20;
    """,
    "05_top_artists_by_popularity": """
        SELECT
            artists AS artist_name,
            count AS total_tracks,
            ROUND(popularity, 2) AS avg_popularity,
            ROUND(energy, 3) AS avg_energy,
            ROUND(danceability, 3) AS avg_danceability
        FROM artist_features
        WHERE count >= 5
        ORDER BY popularity DESC
        LIMIT 20;
    """,
    "06_top_genre_audio_profiles": """
        SELECT
            genres_clean AS genre_name,
            ROUND(popularity, 2) AS avg_popularity,
            ROUND(energy, 3) AS avg_energy,
            ROUND(danceability, 3) AS avg_danceability,
            ROUND(acousticness, 3) AS avg_acousticness,
            ROUND(valence, 3) AS avg_valence,
            ROUND(tempo, 2) AS avg_tempo
        FROM genre_features
        WHERE genres_clean IS NOT NULL
        ORDER BY popularity DESC
        LIMIT 20;
    """,
}


def run_sql_outputs(db_path: Path, paths: ProjectPaths) -> None:
    with sqlite3.connect(db_path) as conn:
        for query_name, query in SQL_QUERIES.items():
            result = pd.read_sql_query(query, conn)
            save_table(result, paths.sql / f"{query_name}.csv")

    sql_script = "\n\n".join(
        f"-- {name}\n{query.strip()}" for name, query in SQL_QUERIES.items()
    )
    (paths.sql / "spotify_week7_queries.sql").write_text(sql_script + "\n", encoding="utf-8")


def write_run_summary(
    paths: ProjectPaths,
    input_files: Dict[str, Path],
    tracks: pd.DataFrame,
    regression_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    sql_executed: bool,
    run_started_at: datetime,
) -> None:
    run_completed_at = datetime.now().astimezone()
    summary = {
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "run_completed_at": run_completed_at.isoformat(timespec="seconds"),
        "run_duration_seconds": round(
            (run_completed_at - run_started_at).total_seconds(),
            2,
        ),
        "input_files": {
            name: str(path.resolve())
            for name, path in input_files.items()
        },
        "main_dataset_rows": int(tracks.shape[0]),
        "main_dataset_columns": int(tracks.shape[1]),
        "tables_folder": str(paths.tables),
        "figures_folder": str(paths.figures),
        "sql_folder": str(paths.sql),
        "model_artifacts_folder": str(paths.model_artifacts),
        "tables_created": [
            path.name for path in sorted(paths.tables.glob("*.csv"))
        ],
        "figures_created": [
            path.name
            for path in sorted(paths.figures.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".html"}
        ],
        "models_trained": regression_summary["models_trained"],
        "best_regression_model_by_R2": regression_summary["best_model"],
        "regression_plot_model": regression_summary["plot_model"],
        "recommendation_summary": recommendation_summary,
        "interactive_plot_note": tracks.attrs.get("interactive_plot_note"),
        "model_artifacts_created": [
            path.name for path in sorted(paths.model_artifacts.iterdir()) if path.is_file()
        ],
        "sql_reference_executed": sql_executed,
        "notes": [
            tracks.attrs.get("genre_pivot_note", "No genre pivot note was recorded."),
            (
                "SQLite/SQL outputs are optional reference outputs. "
                "The group's main SQL work is completed separately in "
                "SQL Server Management Studio."
            ),
        ],
    }
    (paths.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    run_started_at = datetime.now().astimezone()
    parser = argparse.ArgumentParser(description="Run clean Week 7 Spotify analysis outputs.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root folder")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIRNAME,
        help="Output folder name",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Skip optional SQLite database and SQL reference outputs",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    paths = make_paths(root, args.output)

    print(f"Loading project data from: {root}")
    data, input_files = load_project_data(root)

    print("Creating EDA outputs...")
    tracks = create_eda_outputs(data, paths)

    print("Running regression analysis...")
    regression_summary = regression_analysis(tracks, paths)

    print("Building recommendation demo outputs...")
    recommendation_summary = build_recommender_demo(tracks, paths)

    if args.skip_sql:
        print("Skipping optional SQLite/SQL reference outputs...")
    else:
        print("Creating SQLite database and SQL result tables...")
        db_path = create_sqlite_database(data, paths)
        run_sql_outputs(db_path, paths)

    write_run_summary(
        paths=paths,
        input_files=input_files,
        tracks=tracks,
        regression_summary=regression_summary,
        recommendation_summary=recommendation_summary,
        sql_executed=not args.skip_sql,
        run_started_at=run_started_at,
    )
    print(f"Done. Outputs saved to: {paths.output}")


if __name__ == "__main__":
    main()
