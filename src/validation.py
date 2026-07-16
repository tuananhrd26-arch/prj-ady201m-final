"""Pure validation helpers for dataframes and model-alignment contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
    missing_label: str = "required columns",
) -> None:
    """Raise ValueError when any named column is absent, preserving input order."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing {missing_label}: " + ", ".join(missing))


def require_non_missing(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
    value_label: str,
) -> None:
    """Raise ValueError when any selected value is missing."""
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{context} contains missing {value_label}.")


def require_finite_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
    value_label: str,
) -> None:
    """Raise ValueError when selected float-convertible values are non-finite."""
    if not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"{context} contains non-finite {value_label}.")


def require_unique(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
    value_label: str,
) -> None:
    """Raise ValueError when a selected column contains duplicate values."""
    if not frame[column].is_unique:
        raise ValueError(f"{context} {value_label} are not unique.")


def require_contiguous_model_index(
    frame: pd.DataFrame,
    *,
    column: str = "_model_index",
    context: str = "Recommender catalog",
) -> None:
    """Require a present, unique, zero-based, contiguous, row-aligned index."""
    require_columns(frame, [column], context=context)
    expected = np.arange(len(frame))
    if not np.array_equal(frame[column].to_numpy(), expected):
        raise ValueError(
            f"{context} model indexes are not contiguous and row-aligned."
        )
    if not frame[column].is_unique:
        raise ValueError(f"{context} model indexes are not unique.")


def require_row_count(
    actual: int,
    expected: int,
    *,
    context: str,
    expected_context: str,
) -> None:
    """Raise ValueError when two row counts differ."""
    if actual != expected:
        raise ValueError(f"{context} row count does not match {expected_context}.")


def feature_order_matches(
    actual: Sequence[str],
    expected: Sequence[str],
) -> bool:
    """Return whether two feature sequences contain the same items in order."""
    return list(actual) == list(expected)


def is_exact_top_n(returned_count: int, requested_count: int) -> bool:
    """Return whether the returned recommendation count exactly matches Top N."""
    return bool(returned_count == requested_count)
