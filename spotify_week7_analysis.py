"""Public CLI and backward-compatible imports for Spotify analytics."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from src.pipeline import (
    create_eda_outputs,
    make_paths,
    run_pipeline,
    save_table,
    write_run_summary,
)
from src.preprocessing import clean_tracks_for_analysis
from src.recommender import build_recommender_demo
from src.regression import regression_analysis
from src.sql_analysis import SQL_QUERIES, create_sqlite_database, run_sql_outputs
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clean Week 7 Spotify analysis outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root folder",
    )
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

    run_pipeline(args.root, args.output, skip_sql=args.skip_sql)


if __name__ == "__main__":
    main()
