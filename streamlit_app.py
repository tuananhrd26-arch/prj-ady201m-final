"""Self-contained Streamlit application for the persisted Spotify recommender.

The application only loads accepted artifacts from ``models/``.  It never fits,
trains, or writes a model.
"""

from __future__ import annotations

import ast
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

BEST_MODEL_FILE = "best_popularity_model.joblib"
NEIGHBORS_FILE = "nearest_neighbors_recommender.joblib"
CATALOG_FILE = "recommender_catalog.csv"
FEATURES_FILE = "recommender_features.json"
SCALER_FILE = "recommender_scaler.joblib"

RECOMMENDER_FEATURES = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
)
CATALOG_COLUMNS = [
    "_model_index",
    "id",
    "name",
    "artists",
    "year",
    "popularity",
    *RECOMMENDER_FEATURES,
]
EXPECTED_CATALOG_ROWS = 170_653

ACCENT = "#2BD576"
ACCENT_BLUE = "#58A6FF"
GRID = "rgba(139, 148, 158, 0.16)"


@dataclass(frozen=True)
class RecommenderArtifacts:
    """Validated accepted artifacts used by the live application."""

    best_popularity_model: Any
    scaler: Any
    neighbors: Any
    features: tuple[str, ...]
    catalog: pd.DataFrame


def load_recommender_artifacts(
    model_dir: Path = MODEL_DIR,
    *,
    validate_alignment: bool = False,
) -> RecommenderArtifacts:
    """Load all five accepted artifacts and validate their shared contract."""
    model_dir = Path(model_dir)
    filenames = (
        BEST_MODEL_FILE,
        NEIGHBORS_FILE,
        CATALOG_FILE,
        FEATURES_FILE,
        SCALER_FILE,
    )
    missing = [name for name in filenames if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing)
        )

    best_model = joblib.load(model_dir / BEST_MODEL_FILE)
    neighbors = joblib.load(model_dir / NEIGHBORS_FILE)
    scaler = joblib.load(model_dir / SCALER_FILE)
    with (model_dir / FEATURES_FILE).open(encoding="utf-8") as handle:
        feature_data = json.load(handle)
    catalog = pd.read_csv(model_dir / CATALOG_FILE)

    if not isinstance(feature_data, list):
        raise ValueError("The recommender feature file must contain a JSON list.")
    features = tuple(feature_data)
    if features != RECOMMENDER_FEATURES:
        raise ValueError("The recommender feature order is not the accepted order.")
    if catalog.columns.tolist() != CATALOG_COLUMNS:
        raise ValueError("The recommender catalog columns are not in the accepted order.")
    if len(catalog) != EXPECTED_CATALOG_ROWS:
        raise ValueError(
            f"The recommender catalog has {len(catalog):,} rows; "
            f"expected {EXPECTED_CATALOG_ROWS:,}."
        )
    if not np.array_equal(
        catalog["_model_index"].to_numpy(), np.arange(len(catalog))
    ):
        raise ValueError("Catalog model indexes are not contiguous and row-aligned.")
    if not catalog["id"].is_unique:
        raise ValueError("Catalog track IDs are not unique.")
    if catalog[["id", "name", "artists", *features]].isna().any().any():
        raise ValueError("Catalog identity or audio-feature values are missing.")
    if not np.isfinite(catalog[list(features)].to_numpy(dtype=float)).all():
        raise ValueError("Catalog audio-feature values must be finite.")

    feature_count = len(features)
    if int(getattr(scaler, "n_features_in_", -1)) != feature_count:
        raise ValueError("The persisted scaler was not fitted with nine features.")
    if int(getattr(neighbors, "n_features_in_", -1)) != feature_count:
        raise ValueError("The neighbor model was not fitted with nine features.")
    if int(getattr(neighbors, "n_samples_fit_", -1)) != len(catalog):
        raise ValueError("The neighbor model row count does not match the catalog.")

    fitted_matrix = getattr(neighbors, "_fit_X", None)
    expected_shape = (len(catalog), feature_count)
    if fitted_matrix is None or tuple(fitted_matrix.shape) != expected_shape:
        raise ValueError("The fitted neighbor matrix does not match the catalog.")
    if validate_alignment:
        transformed = scaler.transform(catalog[list(features)])
        if not np.allclose(transformed, fitted_matrix, rtol=0, atol=1e-12):
            raise ValueError("Catalog rows do not align with the fitted model matrix.")

    return RecommenderArtifacts(
        best_popularity_model=best_model,
        scaler=scaler,
        neighbors=neighbors,
        features=features,
        catalog=catalog,
    )


@st.cache_resource(show_spinner="Loading the accepted recommendation model…")
def load_cached_artifacts() -> RecommenderArtifacts:
    """Cache artifact loading for interactive Streamlit reruns."""
    return load_recommender_artifacts(validate_alignment=False)


def resolve_catalog_track(
    catalog: pd.DataFrame,
    *,
    track_id: str | None = None,
    model_index: int | None = None,
) -> pd.Series:
    """Resolve exactly one catalog row by Spotify ID or fitted-row index."""
    if (track_id is None) == (model_index is None):
        raise ValueError("Provide exactly one of track_id or model_index.")
    if track_id is not None:
        matches = catalog.loc[catalog["id"] == str(track_id)]
        description = f"track ID {track_id!r}"
    else:
        if isinstance(model_index, bool) or not isinstance(model_index, (int, np.integer)):
            raise ValueError("model_index must be a non-negative integer.")
        if int(model_index) < 0:
            raise ValueError("model_index must be a non-negative integer.")
        matches = catalog.loc[catalog["_model_index"] == int(model_index)]
        description = f"model index {int(model_index)}"
    if matches.empty:
        raise LookupError(f"No catalog track matches {description}.")
    if len(matches) != 1:
        raise LookupError(f"The catalog selector is ambiguous for {description}.")
    return matches.iloc[0].copy()


def recommend_tracks(
    artifacts: RecommenderArtifacts,
    *,
    model_index: int,
    top_n: int,
) -> pd.DataFrame:
    """Return an exact Top N without fitting or changing persisted artifacts."""
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise ValueError("top_n must be a positive integer.")
    if int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer.")

    query = resolve_catalog_track(artifacts.catalog, model_index=model_index)
    query_index = int(query["_model_index"])
    query_frame = artifacts.catalog.iloc[[query_index]][list(artifacts.features)]
    transformed = artifacts.scaler.transform(query_frame)

    fitted_query = artifacts.neighbors._fit_X[query_index].reshape(1, -1)
    if not np.allclose(transformed, fitted_query, rtol=0, atol=1e-12):
        raise ValueError("The selected catalog row is not aligned with the model.")

    total = int(artifacts.neighbors.n_samples_fit_)
    pool = min(total, max(50, int(top_n) * 8))
    query_id = str(query["id"])
    query_pair = (str(query["name"]), str(query["artists"]))

    while True:
        distances, indexes = artifacts.neighbors.kneighbors(
            transformed,
            n_neighbors=pool,
        )
        seen_pairs: set[tuple[str, str]] = set()
        rows: list[dict[str, Any]] = []
        for distance, index_value in zip(distances[0], indexes[0]):
            index = int(index_value)
            neighbor = artifacts.catalog.iloc[index]
            pair = (str(neighbor["name"]), str(neighbor["artists"]))
            if (
                index == query_index
                or str(neighbor["id"]) == query_id
                or pair == query_pair
                or pair in seen_pairs
            ):
                continue
            similarity = 1.0 - float(distance)
            if not np.isfinite(similarity):
                raise ValueError("The model returned a non-finite similarity value.")
            seen_pairs.add(pair)
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "recommended_model_index": index,
                    "recommended_id": str(neighbor["id"]),
                    "recommended_name": neighbor["name"],
                    "recommended_artists": neighbor["artists"],
                    "year": neighbor["year"],
                    "popularity": neighbor["popularity"],
                    "cosine_distance": float(distance),
                    "similarity": similarity,
                }
            )
            if len(rows) == int(top_n):
                return pd.DataFrame(rows)
        if pool >= total:
            raise RuntimeError(f"Only {len(rows)} unique recommendations were available.")
        pool = min(total, pool * 2)


def format_artists(value: Any) -> str:
    """Turn the catalog's serialized artist list into readable text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown artist"
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, (list, tuple)):
        names = [str(item).strip() for item in parsed if str(item).strip()]
        return ", ".join(names) if names else "Unknown artist"
    return str(parsed)


def search_catalog(catalog: pd.DataFrame, query: str, limit: int = 80) -> pd.DataFrame:
    """Find popular catalog matches by track or artist text."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    normalized = query.strip().casefold()
    matches = catalog
    if normalized:
        name_match = catalog["name"].astype(str).str.casefold().str.contains(
            normalized, regex=False
        )
        artist_match = catalog["artists"].astype(str).str.casefold().str.contains(
            normalized, regex=False
        )
        matches = catalog.loc[name_match | artist_match]
    return (
        matches.sort_values(
            ["popularity", "year", "name"],
            ascending=[False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def track_label(row: pd.Series) -> str:
    """Create a stable select-box label."""
    year = "—" if pd.isna(row["year"]) else str(int(row["year"]))
    return f"{row['name']} — {format_artists(row['artists'])} ({year})"


def recommendation_display_frame(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Format recommendation rows for the exact-results table."""
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
    result["year"] = result["year"].round().astype("Int64")
    result["popularity"] = result["popularity"].round().astype("Int64")
    result["similarity"] = (result["similarity"] * 100).round(2)
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


def normalized_audio_profiles(
    rows: pd.DataFrame,
    catalog: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """Scale profiles robustly for display; this is not model preprocessing."""
    lower = catalog[list(features)].quantile(0.05)
    upper = catalog[list(features)].quantile(0.95)
    span = (upper - lower).replace(0, 1)
    return ((rows[list(features)] - lower) / span).clip(0, 1)


def recommendation_card(row: pd.Series) -> None:
    """Render one recommendation card."""
    similarity = float(row["similarity"]) * 100
    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-rank">RECOMMENDATION {int(row['rank']):02d}</div>
          <div class="rec-title">{html.escape(str(row['recommended_name']))}</div>
          <div class="rec-artist">{html.escape(format_artists(row['recommended_artists']))}</div>
          <div class="rec-meta">{int(row['year'])} · Popularity {int(round(float(row['popularity'])))}/100 · {similarity:.2f}% similar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit interface."""
    st.set_page_config(
        page_title="SoundScope · Track Recommender",
        page_icon="♫",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #080B10; }
        .block-container { max-width: 1200px; padding-top: 2rem; }
        .hero { padding: 2rem; border: 1px solid #26303d; border-radius: 24px;
                background: linear-gradient(125deg, #11161f, #0b2b20); margin-bottom: 1.4rem; }
        .hero h1 { color: #f5f7fa; font-size: 3.5rem; margin: 0; }
        .hero p, .rec-artist, .rec-meta { color: #aab4c0; }
        .eyebrow, .rec-rank { color: #2bd576; font-weight: 800; letter-spacing: .12em; }
        .rec-card { min-height: 145px; padding: 1rem 1.15rem; margin-bottom: .8rem;
                    background: #11161f; border: 1px solid #26303d; border-radius: 18px; }
        .rec-title { color: #f5f7fa; font-size: 1.2rem; font-weight: 800; margin-top: .55rem; }
        .rec-meta { margin-top: .65rem; font-size: .85rem; }
        </style>
        <div class="hero">
          <div class="eyebrow">GROUP 3 · PERSISTED MODEL DEMO</div>
          <h1>Track Recommender</h1>
          <p>Search for a catalog track and retrieve its closest standardized audio-profile neighbors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        artifacts = load_cached_artifacts()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"The accepted model artifacts could not be loaded: {error}")
        st.stop()

    catalog = artifacts.catalog
    search_column, count_column = st.columns([3, 1])
    query = search_column.text_input(
        "Search by track or artist",
        placeholder="Example: Dynamite or BTS",
    )
    top_n = count_column.slider("Recommendations", min_value=3, max_value=15, value=10)
    matches = search_catalog(catalog, query)
    if matches.empty:
        st.warning("No matching track was found. Try a shorter search phrase.")
        st.stop()

    row_by_index = {
        int(row["_model_index"]): row for _, row in matches.iterrows()
    }
    selected_index = st.selectbox(
        "Select the exact track, artist, and year",
        list(row_by_index),
        format_func=lambda index: track_label(row_by_index[index]),
    )
    selected = resolve_catalog_track(catalog, model_index=int(selected_index))

    summary = st.columns([1.5, 1.5, 0.7, 0.7])
    summary[0].metric("Track", str(selected["name"]))
    summary[1].metric("Artist", format_artists(selected["artists"]))
    summary[2].metric("Year", int(selected["year"]))
    summary[3].metric("Popularity", int(round(float(selected["popularity"]))))

    if st.button("Generate recommendations", type="primary", use_container_width=True):
        try:
            with st.spinner("Comparing standardized audio profiles…"):
                st.session_state["recommendations"] = recommend_tracks(
                    artifacts,
                    model_index=int(selected_index),
                    top_n=int(top_n),
                )
                st.session_state["recommendation_query"] = (int(selected_index), int(top_n))
        except (ValueError, LookupError, RuntimeError) as error:
            st.error(f"Recommendations could not be generated: {error}")

    recommendations = st.session_state.get("recommendations")
    same_query = st.session_state.get("recommendation_query") == (
        int(selected_index),
        int(top_n),
    )
    if recommendations is None or not same_query:
        st.info("Choose a track and generate recommendations. No model training occurs in this app.")
        return

    st.divider()
    st.subheader(f"Top {len(recommendations)} audio neighbors")
    st.caption("Similarity equals 1 − cosine distance; popularity is display context only.")
    columns = st.columns(2)
    for position, (_, row) in enumerate(recommendations.iterrows()):
        with columns[position % 2]:
            recommendation_card(row)

    with st.expander("View exact result table"):
        st.dataframe(
            recommendation_display_frame(recommendations),
            hide_index=True,
            use_container_width=True,
        )

    indexes = [int(selected_index)] + recommendations[
        "recommended_model_index"
    ].astype(int).tolist()
    profiles = normalized_audio_profiles(
        catalog.iloc[indexes],
        catalog,
        artifacts.features,
    )
    seed_profile = profiles.iloc[0]
    recommendation_average = profiles.iloc[1:].mean()
    labels = [feature.replace("_", " ").title() for feature in artifacts.features]
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=seed_profile.tolist() + [seed_profile.iloc[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Selected track",
            line=dict(color=ACCENT, width=3),
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=recommendation_average.tolist() + [recommendation_average.iloc[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Recommendation average",
            line=dict(color=ACCENT_BLUE, width=2.5),
        )
    )
    radar.update_layout(
        title="Audio Profile Comparison",
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D7DEE7"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID),
        ),
    )
    st.plotly_chart(radar, use_container_width=True)
    st.caption(
        "The radar chart uses percentile scaling for display only. "
        "The persisted recommender uses StandardScaler and cosine distance."
    )


if __name__ == "__main__":
    main()
