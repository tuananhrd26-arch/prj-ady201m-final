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
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from src.data_loader import load_project_data, read_csv_if_exists
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
from src.regression import regression_analysis
from src.validation import (
    feature_order_matches,
    is_exact_top_n,
    require_columns,
    require_contiguous_model_index,
    require_finite_numeric,
    require_non_missing,
    require_row_count,
    require_unique,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_trends,
    plot_interactive_energy_loudness,
    plot_popularity_by_decade_boxplot,
    plot_popularity_distribution,
    plot_tracks_by_decade,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
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


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def create_eda_outputs(data: Dict[str, pd.DataFrame], paths: ProjectPaths) -> pd.DataFrame:
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

    trend = plot_feature_trends(tracks, paths.figures / "audio_feature_trends_by_year.png")
    save_table(trend.round(4), paths.tables / "audio_feature_trends_by_year.csv")

    corr_features = [f for f in RECOMMENDER_FEATURES + [TARGET] if f in tracks.columns]
    corr = plot_correlation_heatmap(tracks, corr_features, paths.figures / "correlation_heatmap.png")
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
    require_columns(model_df, catalog_columns, context="Recommender catalog")
    if not feature_order_matches(features, RECOMMENDER_FEATURES):
        raise ValueError("Recommender feature order does not match RECOMMENDER_FEATURES.")

    catalog = model_df[catalog_columns].copy()
    require_contiguous_model_index(catalog)
    require_unique(
        catalog,
        "id",
        context="Recommender catalog",
        value_label="track ids",
    )
    required_non_null = ["id", "name", "artists", *RECOMMENDER_FEATURES]
    require_non_missing(
        catalog,
        required_non_null,
        context="Recommender catalog",
        value_label="identity or feature values",
    )
    require_finite_numeric(
        catalog,
        RECOMMENDER_FEATURES,
        context="Recommender catalog",
        value_label="feature values",
    )
    if not feature_order_matches(
        catalog.columns[-len(RECOMMENDER_FEATURES):],
        RECOMMENDER_FEATURES,
    ):
        raise ValueError("Recommender catalog feature columns are out of order.")

    scaler = StandardScaler()
    X = scaler.fit_transform(model_df[features])

    require_row_count(
        len(catalog),
        len(X),
        context="Recommender catalog",
        expected_context="the feature matrix",
    )

    candidate_count = len(model_df)
    nn = NearestNeighbors(n_neighbors=candidate_count, metric="cosine")
    nn.fit(X)

    require_row_count(
        len(catalog),
        nn.n_samples_fit_,
        context="Recommender catalog",
        expected_context="the fitted model",
    )

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
        exact_top_n = is_exact_top_n(len(input_rows), n_neighbors)
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
