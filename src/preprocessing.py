"""In-memory track cleaning and analysis preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    RECOMMENDER_FEATURES,
    REGRESSION_FEATURES,
    TARGET,
)


def clean_tracks_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare tracks for analysis without modifying the caller's dataframe."""
    tracks = df.copy()

    # Standardize a few expected columns.
    if "release_year" in tracks.columns and "year" not in tracks.columns:
        tracks["year"] = tracks["release_year"]

    numeric_candidates = sorted(
        set(REGRESSION_FEATURES + RECOMMENDER_FEATURES + [TARGET])
    )
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
