"""Reusable orchestration for the Spotify recommendation analytics pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import (
    DEFAULT_OUTPUT_DIRNAME,
    FIGURES_DIRNAME,
    MODEL_ARTIFACTS_DIRNAME,
    ProjectPaths,
    RECOMMENDER_FEATURES,
    SQL_DIRNAME,
    TABLES_DIRNAME,
    TARGET,
)
from src.data_loader import load_project_data
from src.eda import (
    compute_dataset_overview,
    compute_decade_explicit_summary,
    compute_decade_feature_summary,
    compute_descriptive_statistics,
    compute_missing_values_summary,
    compute_top_genres_audio_profile,
    compute_tracks_by_decade_summary,
    create_genre_decade_pivot,
)
from src.preprocessing import clean_tracks_for_analysis
from src.recommender import build_recommender_demo
from src.regression import regression_analysis
from src.sql_analysis import create_sqlite_database, run_sql_outputs
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_trends,
    plot_interactive_energy_loudness,
    plot_popularity_by_decade_boxplot,
    plot_popularity_distribution,
    plot_tracks_by_decade,
)


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
    for path in [
        paths.output,
        paths.tables,
        paths.figures,
        paths.sql,
        paths.model_artifacts,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def create_eda_outputs(
    data: Dict[str, pd.DataFrame],
    paths: ProjectPaths,
) -> pd.DataFrame:
    tracks = clean_tracks_for_analysis(data["tracks"])

    overview = compute_dataset_overview(data)
    save_table(overview, paths.tables / "dataset_overview.csv")

    descriptive = compute_descriptive_statistics(tracks)
    save_table(descriptive, paths.tables / "descriptive_statistics.csv")

    missing = compute_missing_values_summary(tracks)
    save_table(missing, paths.tables / "missing_values_after_cleaning.csv")

    decade_summary = compute_tracks_by_decade_summary(tracks)
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

    decade_feature_summary = compute_decade_feature_summary(tracks)
    save_table(
        decade_feature_summary,
        paths.tables / "decade_feature_summary.csv",
    )

    trend = plot_feature_trends(
        tracks,
        paths.figures / "audio_feature_trends_by_year.png",
    )
    save_table(trend.round(4), paths.tables / "audio_feature_trends_by_year.csv")

    corr_features = [f for f in RECOMMENDER_FEATURES + [TARGET] if f in tracks.columns]
    corr = plot_correlation_heatmap(
        tracks,
        corr_features,
        paths.figures / "correlation_heatmap.png",
    )
    save_table(corr.round(4), paths.tables / "correlation_matrix.csv")

    if {"decade", "explicit"}.issubset(tracks.columns):
        decade_explicit_summary = compute_decade_explicit_summary(tracks)
        save_table(
            decade_explicit_summary,
            paths.tables / "decade_explicit_multiindex_summary.csv",
        )

    tracks.attrs["interactive_plot_note"] = plot_interactive_energy_loudness(
        tracks,
        paths.figures / "interactive_energy_loudness.html",
        paths.figures / "interactive_energy_loudness_preview.png",
    )

    if "genre_features" in data:
        genre_profile = compute_top_genres_audio_profile(data["genre_features"])
        save_table(
            genre_profile.round(4),
            paths.tables / "top_genres_audio_profile.csv",
        )

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
            path.name
            for path in sorted(paths.model_artifacts.iterdir())
            if path.is_file()
        ],
        "sql_reference_executed": sql_executed,
        "notes": [
            tracks.attrs.get(
                "genre_pivot_note",
                "No genre pivot note was recorded.",
            ),
            (
                "SQLite/SQL outputs are optional reference outputs. "
                "The group's main SQL work is completed separately in "
                "SQL Server Management Studio."
            ),
        ],
    }
    (paths.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def run_pipeline(
    root: Path,
    output: str | Path = DEFAULT_OUTPUT_DIRNAME,
    *,
    skip_sql: bool = False,
) -> Dict[str, Any]:
    run_started_at = datetime.now().astimezone()
    root = Path(root).resolve()
    paths = make_paths(root, output)

    print(f"Loading project data from: {root}")
    data, input_files = load_project_data(root)

    print("Creating EDA outputs...")
    tracks = create_eda_outputs(data, paths)

    print("Running regression analysis...")
    regression_summary = regression_analysis(tracks, paths)

    print("Building recommendation demo outputs...")
    recommendation_summary = build_recommender_demo(tracks, paths)

    sql_reference_executed = False
    if skip_sql:
        print("Skipping optional SQLite/SQL reference outputs...")
    else:
        print("Creating SQLite database and SQL result tables...")
        db_path = create_sqlite_database(data, paths)
        run_sql_outputs(db_path, paths)
        sql_reference_executed = True

    write_run_summary(
        paths=paths,
        input_files=input_files,
        tracks=tracks,
        regression_summary=regression_summary,
        recommendation_summary=recommendation_summary,
        sql_executed=sql_reference_executed,
        run_started_at=run_started_at,
    )
    print(f"Done. Outputs saved to: {paths.output}")

    return {
        "paths": paths,
        "input_files": input_files,
        "tracks": tracks,
        "regression_summary": regression_summary,
        "recommender_summary": recommendation_summary,
        "sql_reference_executed": sql_reference_executed,
        "run_summary_path": paths.output / "run_summary.json",
    }
