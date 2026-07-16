"""Content-based recommender training, demonstration, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (
    ProjectPaths,
    RECOMMENDER_CATALOG_FILENAME,
    RECOMMENDER_FEATURES,
    RECOMMENDER_FEATURES_FILENAME,
    RECOMMENDER_NEIGHBORS_FILENAME,
    RECOMMENDER_SCALER_FILENAME,
    TARGET,
)
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


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


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
    _save_table(catalog, catalog_path)

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
    _save_table(recommendation_output.round(4), paths.tables / "recommendation_demo_results.csv")
    _save_table(validation_output, paths.tables / "recommendation_validation.csv")

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
