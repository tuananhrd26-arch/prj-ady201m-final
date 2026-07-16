"""Read-only loading and querying of persisted recommender artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    RECOMMENDER_CATALOG_FILENAME,
    RECOMMENDER_FEATURES,
    RECOMMENDER_FEATURES_FILENAME,
    RECOMMENDER_NEIGHBORS_FILENAME,
    RECOMMENDER_SCALER_FILENAME,
    TARGET,
)
from src.validation import (
    feature_order_matches,
    require_contiguous_model_index,
    require_finite_numeric,
    require_non_missing,
    require_row_count,
    require_unique,
)


CATALOG_COLUMNS = [
    "_model_index",
    "id",
    "name",
    "artists",
    "year",
    TARGET,
    *RECOMMENDER_FEATURES,
]

RECOMMENDATION_COLUMNS = [
    "query_id",
    "query_name",
    "query_artists",
    "query_model_index",
    "rank",
    "recommended_id",
    "recommended_name",
    "recommended_artists",
    "recommended_model_index",
    "year",
    "popularity",
    "cosine_distance",
    "similarity",
]


@dataclass(frozen=True)
class RecommenderArtifacts:
    scaler: Any
    neighbors: Any
    features: tuple[str, ...]
    catalog: pd.DataFrame
    artifact_dir: Path


def load_recommender_artifacts(
    artifact_dir: Path,
    *,
    validate_alignment: bool = True,
) -> RecommenderArtifacts:
    """Load and validate the four persisted recommender artifacts."""
    artifact_dir = Path(artifact_dir)
    filenames = [
        RECOMMENDER_SCALER_FILENAME,
        RECOMMENDER_NEIGHBORS_FILENAME,
        RECOMMENDER_FEATURES_FILENAME,
        RECOMMENDER_CATALOG_FILENAME,
    ]
    paths = {filename: artifact_dir / filename for filename in filenames}
    missing = [filename for filename, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing recommender artifact files: " + ", ".join(missing)
        )

    scaler = joblib.load(paths[RECOMMENDER_SCALER_FILENAME])
    neighbors = joblib.load(paths[RECOMMENDER_NEIGHBORS_FILENAME])
    with paths[RECOMMENDER_FEATURES_FILENAME].open(encoding="utf-8") as handle:
        feature_data = json.load(handle)
    catalog = pd.read_csv(paths[RECOMMENDER_CATALOG_FILENAME])

    if not isinstance(feature_data, list):
        raise ValueError("Recommender feature JSON must contain a list.")
    features = tuple(feature_data)
    if not feature_order_matches(features, RECOMMENDER_FEATURES):
        raise ValueError("Recommender feature order does not match RECOMMENDER_FEATURES.")
    if catalog.columns.tolist() != CATALOG_COLUMNS:
        raise ValueError(
            "Recommender catalog columns do not match the approved schema."
        )
    if not feature_order_matches(
        catalog.columns[-len(RECOMMENDER_FEATURES):],
        RECOMMENDER_FEATURES,
    ):
        raise ValueError("Recommender catalog feature columns are out of order.")

    require_contiguous_model_index(catalog)
    require_unique(
        catalog,
        "id",
        context="Recommender catalog",
        value_label="track ids",
    )
    require_non_missing(
        catalog,
        ["id", "name", "artists", *RECOMMENDER_FEATURES],
        context="Recommender catalog",
        value_label="identity or feature values",
    )
    require_finite_numeric(
        catalog,
        RECOMMENDER_FEATURES,
        context="Recommender catalog",
        value_label="feature values",
    )

    expected_features = len(RECOMMENDER_FEATURES)
    if int(getattr(scaler, "n_features_in_", -1)) != expected_features:
        raise ValueError("Recommender scaler fitted feature count is not nine.")
    if int(getattr(neighbors, "n_features_in_", -1)) != expected_features:
        raise ValueError("Recommender neighbor fitted feature count is not nine.")
    require_row_count(
        len(catalog),
        int(getattr(neighbors, "n_samples_fit_", -1)),
        context="Recommender catalog",
        expected_context="the fitted model",
    )

    fitted_matrix = getattr(neighbors, "_fit_X", None)
    expected_shape = (len(catalog), expected_features)
    if fitted_matrix is None or tuple(fitted_matrix.shape) != expected_shape:
        raise ValueError("Recommender fitted matrix shape does not match the catalog.")

    if validate_alignment:
        transformed = scaler.transform(catalog[list(features)])
        if transformed.shape != fitted_matrix.shape:
            raise ValueError(
                "Transformed catalog shape does not match the fitted matrix."
            )
        if not np.allclose(transformed, fitted_matrix, rtol=0, atol=1e-12):
            raise ValueError(
                "Transformed catalog rows do not align with the fitted matrix."
            )

    return RecommenderArtifacts(
        scaler=scaler,
        neighbors=neighbors,
        features=features,
        catalog=catalog,
        artifact_dir=artifact_dir,
    )


def resolve_catalog_track(
    catalog: pd.DataFrame,
    *,
    track_id: str | None = None,
    model_index: int | None = None,
    name: str | None = None,
    artists: str | None = None,
) -> pd.Series:
    """Resolve exactly one persisted catalog row using one selector mode."""
    pair_requested = name is not None or artists is not None
    modes = int(track_id is not None) + int(model_index is not None) + int(pair_requested)
    if modes != 1:
        raise ValueError(
            "Provide exactly one selector: track_id, model_index, or name with artists."
        )

    if pair_requested:
        if name is None or artists is None:
            raise ValueError("The name selector requires both name and artists.")
        if not isinstance(name, str) or not isinstance(artists, str):
            raise ValueError("Name and artists selectors must be strings.")
        matches = catalog.loc[
            (catalog["name"] == name) & (catalog["artists"] == artists)
        ]
        description = f"name={name!r}, artists={artists!r}"
    elif track_id is not None:
        if not isinstance(track_id, str):
            raise ValueError("track_id must be a string.")
        matches = catalog.loc[catalog["id"] == track_id]
        description = f"track_id={track_id!r}"
    else:
        if isinstance(model_index, (bool, np.bool_)) or not isinstance(
            model_index, (int, np.integer)
        ):
            raise ValueError("model_index must be a non-negative integer.")
        if int(model_index) < 0:
            raise ValueError("model_index must be a non-negative integer.")
        matches = catalog.loc[catalog["_model_index"] == int(model_index)]
        description = f"model_index={int(model_index)}"

    if matches.empty:
        raise LookupError(f"No catalog track matches {description}.")
    if len(matches) != 1:
        raise LookupError(f"Catalog track selector is ambiguous for {description}.")
    return matches.iloc[0].copy()


def recommend_from_artifacts(
    artifacts: RecommenderArtifacts,
    *,
    track_id: str | None = None,
    model_index: int | None = None,
    name: str | None = None,
    artists: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return recommendations from persisted artifacts without fitting or writing."""
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer)):
        raise ValueError("top_n must be a positive integer.")
    if int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer.")

    query = resolve_catalog_track(
        artifacts.catalog,
        track_id=track_id,
        model_index=model_index,
        name=name,
        artists=artists,
    )
    query_index = int(query["_model_index"])
    query_frame = artifacts.catalog.iloc[[query_index]][list(artifacts.features)]
    transformed_query = artifacts.scaler.transform(query_frame)
    fitted_query = artifacts.neighbors._fit_X[query_index].reshape(1, -1)
    if transformed_query.shape != fitted_query.shape or not np.allclose(
        transformed_query,
        fitted_query,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Selected catalog row does not align with the fitted matrix.")

    candidate_count = int(artifacts.neighbors.n_samples_fit_)
    distances, indices = artifacts.neighbors.kneighbors(
        transformed_query,
        n_neighbors=candidate_count,
    )

    query_id = str(query["id"])
    query_pair = (str(query["name"]), str(query["artists"]))
    seen_pairs: set[tuple[str, str]] = set()
    seen_indexes: set[int] = set()
    rows: list[dict[str, Any]] = []

    for distance, neighbor_index_value in zip(distances[0], indices[0]):
        neighbor_index = int(neighbor_index_value)
        if neighbor_index == query_index or neighbor_index in seen_indexes:
            continue
        neighbor = artifacts.catalog.iloc[neighbor_index]
        neighbor_id = str(neighbor["id"])
        neighbor_pair = (str(neighbor["name"]), str(neighbor["artists"]))
        if neighbor_id == query_id or neighbor_pair == query_pair:
            continue
        if neighbor_pair in seen_pairs:
            continue

        cosine_distance = float(distance)
        similarity = 1 - cosine_distance
        if not np.isfinite(cosine_distance) or not np.isfinite(similarity):
            raise ValueError("Neighbor query returned a non-finite distance or similarity.")

        seen_indexes.add(neighbor_index)
        seen_pairs.add(neighbor_pair)
        rows.append(
            {
                "query_id": query_id,
                "query_name": query["name"],
                "query_artists": query["artists"],
                "query_model_index": query_index,
                "rank": len(rows) + 1,
                "recommended_id": neighbor_id,
                "recommended_name": neighbor["name"],
                "recommended_artists": neighbor["artists"],
                "recommended_model_index": neighbor_index,
                "year": neighbor.get("year", np.nan),
                "popularity": neighbor.get(TARGET, np.nan),
                "cosine_distance": cosine_distance,
                "similarity": similarity,
            }
        )
        if len(rows) >= int(top_n):
            break

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
