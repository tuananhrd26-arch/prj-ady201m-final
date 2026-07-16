"""Stable configuration contracts for Spotify analytics and model artifacts.

Feature-list order is part of the fitted-model contract and must remain stable.
Artifact filenames are centralized so producers and future consumers share the
same persisted interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
TARGET = "popularity"

REGRESSION_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "tempo",
    "loudness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

REGRESSION_EXTENDED_FEATURES = REGRESSION_AUDIO_FEATURES + [
    "duration_ms",
    "year",
    "explicit",
    "key",
    "mode",
]

REGRESSION_FEATURES = REGRESSION_EXTENDED_FEATURES

RECOMMENDER_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

TREND_FEATURES = ["energy", "danceability", "acousticness", "valence"]

BEST_POPULARITY_MODEL_FILENAME = "best_popularity_model.joblib"
RECOMMENDER_SCALER_FILENAME = "recommender_scaler.joblib"
RECOMMENDER_NEIGHBORS_FILENAME = "nearest_neighbors_recommender.joblib"
RECOMMENDER_FEATURES_FILENAME = "recommender_features.json"
RECOMMENDER_CATALOG_FILENAME = "recommender_catalog.csv"

DEFAULT_OUTPUT_DIRNAME = "week7_outputs"
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"
SQL_DIRNAME = "sql"
MODEL_ARTIFACTS_DIRNAME = "model_artifacts"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem locations used by one analysis pipeline run."""

    root: Path
    output: Path
    tables: Path
    figures: Path
    sql: Path
    model_artifacts: Path
