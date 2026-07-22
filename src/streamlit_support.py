"""Pure helpers used by the Streamlit dashboard.

The functions in this module do not import Streamlit, so they can be tested in
the normal project test environment.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from src.recommender_consumer import RecommenderArtifacts, resolve_catalog_track


def format_artists(value: Any) -> str:
    """Convert the dataset's serialized artist list into readable text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown artist"
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(parsed, (list, tuple)):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "Unknown artist"
    return str(parsed)


def search_catalog(
    catalog: pd.DataFrame,
    query: str,
    *,
    limit: int = 80,
) -> pd.DataFrame:
    """Return a presentation-friendly catalog subset for a search query."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"_model_index", "name", "artists", "year", "popularity"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError("catalog is missing columns: " + ", ".join(missing))

    normalized = query.strip().casefold()
    searchable = catalog.copy()
    if normalized:
        name_mask = searchable["name"].astype(str).str.casefold().str.contains(
            normalized,
            regex=False,
        )
        artist_mask = searchable["artists"].astype(str).str.casefold().str.contains(
            normalized,
            regex=False,
        )
        searchable = searchable.loc[name_mask | artist_mask]

    return (
        searchable.sort_values(
            ["popularity", "year", "name"],
            ascending=[False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def track_label(row: pd.Series) -> str:
    """Build a readable and stable select-box label."""
    year = "—" if pd.isna(row.get("year")) else str(int(row["year"]))
    return f"{row['name']} — {format_artists(row['artists'])} ({year})"


def normalize_profiles(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    features: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Robustly scale audio profiles to 0..1 for radar-chart display."""
    feature_list = list(features)
    lower = reference[feature_list].quantile(0.05)
    upper = reference[feature_list].quantile(0.95)
    span = (upper - lower).replace(0, 1)
    normalized = (frame[feature_list] - lower) / span
    return normalized.clip(0, 1)


def recommendation_display_frame(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Select and format columns used in the dashboard result table."""
    required = {
        "rank",
        "recommended_name",
        "recommended_artists",
        "year",
        "popularity",
        "similarity",
    }
    missing = sorted(required.difference(recommendations.columns))
    if missing:
        raise ValueError("recommendations are missing columns: " + ", ".join(missing))

    result = recommendations[
        [
            "rank",
            "recommended_name",
            "recommended_artists",
            "year",
            "popularity",
            "similarity",
        ]
    ].copy()
    result["recommended_artists"] = result["recommended_artists"].map(format_artists)
    result["similarity"] = (result["similarity"] * 100).round(2)
    result["popularity"] = result["popularity"].round(0).astype("Int64")
    result["year"] = result["year"].round(0).astype("Int64")
    return result.rename(
        columns={
            "rank": "Rank",
            "recommended_name": "Track",
            "recommended_artists": "Artist",
            "year": "Year",
            "popularity": "Popularity",
            "similarity": "Similarity (%)",
        }
    )


def fast_recommend(
    artifacts: RecommenderArtifacts,
    *,
    model_index: int,
    top_n: int,
) -> pd.DataFrame:
    """Query a growing neighbor pool until enough unique tracks are found."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    query = resolve_catalog_track(artifacts.catalog, model_index=model_index)
    query_index = int(query["_model_index"])
    query_features = artifacts.catalog.iloc[[query_index]][list(artifacts.features)]
    transformed = artifacts.scaler.transform(query_features)
    total = int(artifacts.neighbors.n_samples_fit_)
    pool = min(total, max(50, top_n * 8))

    while True:
        distances, indexes = artifacts.neighbors.kneighbors(
            transformed,
            n_neighbors=pool,
        )
        seen_pairs: set[tuple[str, str]] = set()
        rows: list[dict[str, object]] = []
        query_pair = (str(query["name"]), str(query["artists"]))

        for distance, index_value in zip(distances[0], indexes[0]):
            index = int(index_value)
            if index == query_index:
                continue
            neighbor = artifacts.catalog.iloc[index]
            pair = (str(neighbor["name"]), str(neighbor["artists"]))
            if pair == query_pair or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "recommended_model_index": index,
                    "recommended_name": neighbor["name"],
                    "recommended_artists": neighbor["artists"],
                    "year": neighbor.get("year", np.nan),
                    "popularity": neighbor.get("popularity", np.nan),
                    "cosine_distance": float(distance),
                    "similarity": 1 - float(distance),
                }
            )
            if len(rows) >= top_n:
                return pd.DataFrame(rows)

        if pool >= total:
            return pd.DataFrame(rows)
        pool = min(total, pool * 2)
