"""Pure dataframe calculations for Spotify exploratory analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    REGRESSION_FEATURES,
    RANDOM_STATE,
    TARGET,
    TREND_FEATURES,
)


def compute_dataset_overview(
    data: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return input table names and dimensions in mapping iteration order."""
    return pd.DataFrame(
        [
            {"table": name, "rows": len(frame), "columns": frame.shape[1]}
            for name, frame in data.items()
        ]
    )


def compute_descriptive_statistics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return transposed statistics for available regression and target features."""
    features = [
        feature
        for feature in REGRESSION_FEATURES + [TARGET]
        if feature in tracks.columns
    ]
    return (
        tracks[features]
        .describe()
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )


def compute_missing_values_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and rates ordered by descending missing count."""
    missing = (
        tracks.isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing["missing_rate"] = missing["missing_count"] / len(tracks)
    return missing


def compute_tracks_by_decade_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return track counts and mean popularity grouped by decade."""
    return (
        tracks.groupby("decade", as_index=False)
        .agg(track_count=("id", "count"), avg_popularity=(TARGET, "mean"))
        .round(3)
    )


def compute_track_counts_by_decade(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return group sizes by decade for the track-count figure."""
    counts = tracks.groupby("decade", as_index=False).size()
    return counts.rename(columns={"size": "track_count"})


def compute_popularity_by_decade(
    tracks: pd.DataFrame,
) -> tuple[list[Any], list[np.ndarray]]:
    """Return sorted decades and non-missing popularity arrays for box plotting."""
    decades = sorted(tracks["decade"].dropna().unique())
    values = [
        tracks.loc[tracks["decade"] == decade, TARGET].dropna().values
        for decade in decades
    ]
    return decades, values


def compute_decade_feature_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return rounded popularity and audio-feature aggregates by decade."""
    return (
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


def compute_audio_feature_trends_by_year(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return mean trend features by ascending year."""
    return tracks.groupby("year", as_index=False)[TREND_FEATURES].mean()


def compute_correlation_matrix(
    tracks: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Return the numeric-only correlation matrix for features in supplied order."""
    return tracks[list(features)].corr(numeric_only=True)


def compute_decade_explicit_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return rounded decade and explicit-status aggregates with a flat index."""
    return (
        tracks.groupby(["decade", "explicit"])
        .agg(
            total_tracks=(TARGET, "size"),
            average_popularity=(TARGET, "mean"),
            average_energy=("energy", "mean"),
            average_danceability=("danceability", "mean"),
            average_acousticness=("acousticness", "mean"),
        )
        .round(4)
        .reset_index()
    )


def prepare_interactive_energy_loudness_data(
    tracks: pd.DataFrame,
    sample_size: int = 10000,
) -> tuple[pd.DataFrame | None, str | None]:
    """Return complete deterministic scatter data or the existing skip reason."""
    required_columns = [
        "energy",
        "loudness",
        "decade",
        "name",
        "artists",
        TARGET,
        "year",
        "danceability",
    ]
    missing = [column for column in required_columns if column not in tracks.columns]
    if missing:
        return (
            None,
            "Interactive Plotly scatter skipped: missing columns " + ", ".join(missing),
        )

    plot_data = tracks.dropna(subset=required_columns).copy()
    if plot_data.empty:
        return None, "Interactive Plotly scatter skipped: no complete rows available."

    if len(plot_data) > sample_size:
        plot_data = plot_data.sample(sample_size, random_state=RANDOM_STATE)

    plot_data["decade"] = plot_data["decade"].astype(int).astype(str)
    return plot_data, None


def compute_top_genres_audio_profile(genre_features: pd.DataFrame) -> pd.DataFrame:
    """Return the 20 highest-popularity genre audio profiles."""
    genre_data = genre_features.copy()
    genre_name_column = (
        "genres_clean" if "genres_clean" in genre_data.columns else "genres"
    )
    profile_columns = [
        genre_name_column,
        "popularity",
        "energy",
        "danceability",
        "acousticness",
        "valence",
        "tempo",
    ]
    profile_columns = [
        column for column in profile_columns if column in genre_data.columns
    ]
    return (
        genre_data[profile_columns]
        .dropna(subset=[genre_name_column])
        .sort_values("popularity", ascending=False)
        .head(20)
    )


def parse_artist_names(value: Any) -> list[str]:
    """Convert a serialized artist list to stripped artist names."""
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
) -> tuple[pd.DataFrame | None, str]:
    """Build a genre-by-decade popularity pivot via artist matching."""
    required_track_columns = {"artists", "decade", TARGET}
    required_genre_columns = {"artists", "genres_clean"}
    if not required_track_columns.issubset(tracks.columns):
        return (
            None,
            "Genre pivot skipped: track-level artist/decade/popularity columns are incomplete.",
        )
    if not required_genre_columns.issubset(artist_genres.columns):
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
    genre_lookup = genre_lookup.loc[
        genre_lookup["genre"] != "", ["artist", "genre"]
    ]
    genre_lookup = genre_lookup.drop_duplicates()

    matched = track_artist.merge(genre_lookup, on="artist", how="inner")
    if matched.empty:
        return (
            None,
            "Genre pivot skipped: no track artists matched the artist-to-genre data.",
        )

    top_genres = matched.groupby("genre").size().nlargest(max_genres).index
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
