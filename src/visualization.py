"""EDA visualization functions for Spotify analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from src.config import TARGET, TREND_FEATURES
from src.eda import (
    compute_audio_feature_trends_by_year,
    compute_correlation_matrix,
    compute_popularity_by_decade,
    compute_track_counts_by_decade,
    prepare_interactive_energy_loudness_data,
)


def plot_tracks_by_decade(tracks: pd.DataFrame, fig_path: Path) -> None:
    decade_counts = compute_track_counts_by_decade(tracks)

    plt.figure(figsize=(10, 5))
    plt.bar(decade_counts["decade"].astype(str), decade_counts["track_count"])
    plt.title("Number of Tracks by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Number of Tracks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_popularity_distribution(tracks: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(tracks[TARGET].dropna(), bins=25, edgecolor="white")
    plt.title("Distribution of Track Popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_popularity_by_decade_boxplot(
    tracks: pd.DataFrame,
    fig_path: Path,
) -> None:
    decades, values = compute_popularity_by_decade(tracks)

    plt.figure(figsize=(11, 6))
    plt.boxplot(
        values,
        tick_labels=[str(int(decade)) for decade in decades],
        showfliers=False,
    )
    plt.title("Track Popularity by Decade")
    plt.xlabel("Decade")
    plt.ylabel("Popularity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_feature_trends(tracks: pd.DataFrame, fig_path: Path) -> pd.DataFrame:
    trend = compute_audio_feature_trends_by_year(tracks)

    plt.figure(figsize=(10, 5))
    for feature in TREND_FEATURES:
        plt.plot(trend["year"], trend[feature], label=feature)
    plt.title("Average Audio Feature Trends by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    return trend


def plot_correlation_heatmap(
    tracks: pd.DataFrame,
    features: List[str],
    fig_path: Path,
) -> pd.DataFrame:
    corr = compute_correlation_matrix(tracks, features)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix of Spotify Audio Features")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    return corr.reset_index().rename(columns={"index": "feature"})


def plot_interactive_energy_loudness(
    tracks: pd.DataFrame,
    html_path: Path,
    preview_path: Path,
    sample_size: int = 10000,
) -> str:
    plot_df, skip_reason = prepare_interactive_energy_loudness_data(
        tracks,
        sample_size,
    )
    if plot_df is None:
        return str(skip_reason)

    fig = px.scatter(
        plot_df,
        x="energy",
        y="loudness",
        color="decade",
        hover_data=["name", "artists", TARGET, "year", "danceability"],
        title="Interactive Energy vs Loudness by Decade",
        labels={
            "energy": "Energy",
            "loudness": "Loudness",
            "decade": "Decade",
            TARGET: "Popularity",
        },
    )
    fig.write_html(html_path, include_plotlyjs="cdn")

    try:
        fig.write_image(preview_path, width=1200, height=800, scale=2)
    except Exception as exc:
        return (
            "Interactive Plotly HTML created; static PNG preview unavailable. "
            f"Plotly image export requires Kaleido and compatible browser support. Error: {exc}"
        )

    return "Interactive Plotly HTML and static PNG preview created successfully."
