"""CSV discovery and loading for the Spotify project datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


_CLEANED_DATA_DIRNAME = "cleaned_data"
_PRIMARY_TRACK_FILENAME = "data_clean.csv"
_FALLBACK_TRACK_PATH = Path("data") / "data.csv"
_OPTIONAL_DATASETS = {
    "artist_features": "data_by_artist_clean.csv",
    "genre_features": "data_by_genres_clean.csv",
    "year_features": "data_by_year_clean.csv",
    "artist_genres": "data_w_genres_clean.csv",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Return an optional CSV as a dataframe, or None when it is absent.

    The input file is read without modification.
    """
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_project_data(
    root: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    """Load the project's required and optional datasets without modifying them."""
    data: Dict[str, pd.DataFrame] = {}
    input_files: Dict[str, Path] = {}

    tracks_path = root / _CLEANED_DATA_DIRNAME / _PRIMARY_TRACK_FILENAME
    if not tracks_path.exists():
        tracks_path = root / _FALLBACK_TRACK_PATH
    if not tracks_path.exists():
        raise FileNotFoundError(
            "Could not find cleaned_data/data_clean.csv or data/data.csv"
        )

    data["tracks"] = pd.read_csv(tracks_path)
    input_files["tracks"] = tracks_path

    for name, filename in _OPTIONAL_DATASETS.items():
        path = root / _CLEANED_DATA_DIRNAME / filename
        frame = read_csv_if_exists(path)
        if frame is not None:
            data[name] = frame
            input_files[name] = path

    return data, input_files
