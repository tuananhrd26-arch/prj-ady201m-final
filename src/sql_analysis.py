"""SQLite reference database creation and SQL-result persistence."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import ProjectPaths


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def create_sqlite_database(
    data: Dict[str, pd.DataFrame],
    paths: ProjectPaths,
) -> Path:
    db_path = paths.sql / "spotify_week7.sqlite"
    if db_path.exists():
        db_path.unlink()

    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            table_map = {
                "tracks": "tracks",
                "artist_features": "artist_features",
                "genre_features": "genre_features",
                "year_features": "year_features",
                "artist_genres": "artist_genres",
            }
            for source_name, table_name in table_map.items():
                if source_name in data:
                    frame = data[source_name].copy()
                    frame.to_sql(table_name, conn, index=False, if_exists="replace")

    return db_path


SQL_QUERIES: Dict[str, str] = {
    "01_tracks_by_decade": """
        SELECT
            CAST(year / 10 AS INTEGER) * 10 AS decade,
            COUNT(*) AS total_tracks,
            ROUND(AVG(popularity), 2) AS avg_popularity,
            ROUND(AVG(energy), 3) AS avg_energy,
            ROUND(AVG(danceability), 3) AS avg_danceability
        FROM tracks
        GROUP BY CAST(year / 10 AS INTEGER) * 10
        ORDER BY decade;
    """,
    "02_audio_features_by_year": """
        SELECT
            year,
            ROUND(AVG(energy), 3) AS avg_energy,
            ROUND(AVG(danceability), 3) AS avg_danceability,
            ROUND(AVG(acousticness), 3) AS avg_acousticness,
            ROUND(AVG(valence), 3) AS avg_valence,
            ROUND(AVG(popularity), 2) AS avg_popularity
        FROM tracks
        GROUP BY year
        ORDER BY year;
    """,
    "03_top_tracks_by_decade_window": """
        WITH ranked_tracks AS (
            SELECT
                name AS track_name,
                artists,
                popularity,
                year,
                CAST(year / 10 AS INTEGER) * 10 AS decade,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(year / 10 AS INTEGER) * 10
                    ORDER BY popularity DESC, name ASC
                ) AS rank_in_decade
            FROM tracks
        )
        SELECT *
        FROM ranked_tracks
        WHERE rank_in_decade <= 5
        ORDER BY decade, rank_in_decade;
    """,
    "04_above_average_popularity_subquery": """
        SELECT
            name AS track_name,
            artists,
            year,
            popularity,
            ROUND(popularity - (SELECT AVG(popularity) FROM tracks), 2) AS popularity_gap
        FROM tracks
        WHERE popularity > (SELECT AVG(popularity) FROM tracks)
        ORDER BY popularity_gap DESC, track_name ASC
        LIMIT 20;
    """,
    "05_top_artists_by_popularity": """
        SELECT
            artists AS artist_name,
            count AS total_tracks,
            ROUND(popularity, 2) AS avg_popularity,
            ROUND(energy, 3) AS avg_energy,
            ROUND(danceability, 3) AS avg_danceability
        FROM artist_features
        WHERE count >= 5
        ORDER BY popularity DESC
        LIMIT 20;
    """,
    "06_top_genre_audio_profiles": """
        SELECT
            genres_clean AS genre_name,
            ROUND(popularity, 2) AS avg_popularity,
            ROUND(energy, 3) AS avg_energy,
            ROUND(danceability, 3) AS avg_danceability,
            ROUND(acousticness, 3) AS avg_acousticness,
            ROUND(valence, 3) AS avg_valence,
            ROUND(tempo, 2) AS avg_tempo
        FROM genre_features
        WHERE genres_clean IS NOT NULL
        ORDER BY popularity DESC
        LIMIT 20;
    """,
}


def run_sql_outputs(db_path: Path, paths: ProjectPaths) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        for query_name, query in SQL_QUERIES.items():
            result = pd.read_sql_query(query, conn)
            _save_table(result, paths.sql / f"{query_name}.csv")

    sql_script = "\n\n".join(
        f"-- {name}\n{query.strip()}" for name, query in SQL_QUERIES.items()
    )
    (paths.sql / "spotify_week7_queries.sql").write_text(
        sql_script + "\n",
        encoding="utf-8",
    )
