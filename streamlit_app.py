"""Focused Streamlit demo for the persisted content-based recommender."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.recommender_consumer import (
    RecommenderArtifacts,
    load_recommender_artifacts,
    resolve_catalog_track,
)
from src.streamlit_support import (
    fast_recommend,
    format_artists,
    normalize_profiles,
    recommendation_display_frame,
    search_catalog,
    track_label,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "week7_outputs" / "model_artifacts"

ACCENT = "#2BD576"
ACCENT_BLUE = "#58A6FF"
GRID = "rgba(139, 148, 158, 0.16)"
PANEL = "#11161F"


st.set_page_config(
    page_title="SoundScope · Track Recommender",
    page_icon="♫",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 82% 3%, rgba(43, 213, 118, 0.12), transparent 25rem),
            radial-gradient(circle at 8% 55%, rgba(88, 166, 255, 0.08), transparent 28rem),
            #080B10;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { letter-spacing: -0.035em; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.35rem 2.45rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(139, 148, 158, 0.18);
        border-radius: 26px;
        background: linear-gradient(125deg, rgba(17, 22, 31, 0.98), rgba(11, 43, 32, 0.92));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
    }
    .hero::after {
        content: "♫";
        position: absolute;
        right: 2.2rem;
        top: -1.1rem;
        color: rgba(43, 213, 118, 0.11);
        font-size: 12rem;
        font-weight: 900;
        transform: rotate(-8deg);
    }
    .eyebrow {
        color: #2BD576;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.17em;
        margin-bottom: 0.65rem;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        max-width: 760px;
        color: #F5F7FA;
        font-size: clamp(2.8rem, 7vw, 5.6rem);
        line-height: 0.95;
    }
    .hero p {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 1.05rem 0 0;
        color: #AAB4C0;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .method-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.2rem;
    }
    .method-chip {
        padding: 0.42rem 0.75rem;
        color: #C9D1D9;
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(139,148,158,0.18);
        border-radius: 999px;
        font-size: 0.84rem;
    }
    .search-panel {
        padding: 1.25rem 1.35rem 0.25rem;
        border: 1px solid rgba(139, 148, 158, 0.17);
        border-radius: 20px;
        background: rgba(17, 22, 31, 0.88);
        margin-bottom: 1.4rem;
    }
    div[data-testid="stMetric"] {
        height: 100%;
        padding: 1rem 1.05rem;
        background: rgba(17, 22, 31, 0.90);
        border: 1px solid rgba(139, 148, 158, 0.16);
        border-radius: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #F5F7FA;
        font-size: 1.35rem;
    }
    .stButton > button {
        min-height: 3.15rem;
        border: 0;
        border-radius: 999px;
        background: #2BD576;
        color: #06110B;
        font-size: 1rem;
        font-weight: 850;
        box-shadow: 0 12px 32px rgba(43, 213, 118, 0.18);
    }
    .stButton > button:hover {
        border: 0;
        color: #06110B;
        background: #55E792;
    }
    .section-kicker {
        margin-top: 0.4rem;
        margin-bottom: -0.3rem;
        color: #2BD576;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .rec-card {
        min-height: 174px;
        padding: 1.15rem 1.2rem;
        margin-bottom: 0.85rem;
        background: linear-gradient(145deg, rgba(17, 22, 31, 0.98), rgba(18, 31, 29, 0.94));
        border: 1px solid rgba(139, 148, 158, 0.17);
        border-radius: 18px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.14);
    }
    .rec-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.65rem;
    }
    .rec-rank {
        color: #2BD576;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.12em;
    }
    .rec-score {
        color: #F5F7FA;
        font-size: 0.9rem;
        font-weight: 800;
    }
    .rec-title {
        overflow: hidden;
        color: #F5F7FA;
        font-size: 1.25rem;
        font-weight: 850;
        line-height: 1.2;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .rec-artist {
        overflow: hidden;
        margin-top: 0.32rem;
        color: #AAB4C0;
        font-size: 0.94rem;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .rec-meta {
        margin-top: 0.65rem;
        color: #7F8B99;
        font-size: 0.82rem;
    }
    .score-track {
        height: 5px;
        margin-top: 0.82rem;
        overflow: hidden;
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
    }
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #2BD576, #58A6FF);
        border-radius: 99px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(139, 148, 158, 0.16);
        border-radius: 16px;
        overflow: hidden;
    }
    .small-note { color: #8B949E; font-size: 0.87rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the recommendation model…")
def load_artifacts() -> RecommenderArtifacts:
    # Full catalog alignment was validated by the training pipeline. Skipping a
    # second whole-matrix comparison keeps the live demonstration responsive.
    return load_recommender_artifacts(
        ARTIFACT_DIR,
        validate_alignment=False,
    )


def style_radar(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=570,
        margin=dict(l=30, r=30, t=70, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D7DEE7", family="Inter, Arial, sans-serif"),
        title_font=dict(size=21, color="#F5F7FA"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.12),
        hoverlabel=dict(bgcolor=PANEL, font_color="#F5F7FA"),
    )
    return fig


def recommendation_card(row: pd.Series) -> None:
    similarity = float(row["similarity"]) * 100
    popularity = int(round(float(row["popularity"])))
    year = int(round(float(row["year"])))
    track = html.escape(str(row["recommended_name"]))
    artist = html.escape(format_artists(row["recommended_artists"]))
    rank = int(row["rank"])
    width = max(0.0, min(100.0, similarity))
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-topline">
                <div class="rec-rank">RECOMMENDATION {rank:02d}</div>
                <div class="rec-score">{similarity:.2f}% similar</div>
            </div>
            <div class="rec-title">{track}</div>
            <div class="rec-artist">{artist}</div>
            <div class="rec-meta">{year} · Popularity {popularity}/100</div>
            <div class="score-track"><div class="score-fill" style="width:{width:.2f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">GROUP 3 · LIVE MODEL DEMO</div>
        <h1>Track Recommender</h1>
        <p>
            Search one exact catalog track and retrieve its closest audio-feature
            neighbors from the persisted recommendation model.
        </p>
        <div class="method-strip">
            <span class="method-chip">9 audio features</span>
            <span class="method-chip">StandardScaler</span>
            <span class="method-chip">Cosine distance</span>
            <span class="method-chip">Top 5–15 unique tracks</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    artifacts = load_artifacts()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"The recommender artifacts could not be loaded: {error}")
    st.stop()

catalog = artifacts.catalog

st.markdown("<div class='section-kicker'>Choose the seed track</div>", unsafe_allow_html=True)
st.subheader("What should the model recommend from?")

with st.container(border=False):
    search_col, top_col = st.columns([3, 1])
    search_text = search_col.text_input(
        "Search by track or artist",
        placeholder="Example: Dynamite or BTS",
    )
    top_n = top_col.slider(
        "Recommendations",
        min_value=5,
        max_value=15,
        value=10,
    )

    matches = search_catalog(catalog, search_text, limit=80)
    if matches.empty:
        st.warning("No matching track was found. Try a shorter search phrase.")
        st.stop()

    options = matches["_model_index"].astype(int).tolist()
    row_by_index = {
        int(row["_model_index"]): row
        for _, row in matches.iterrows()
    }
    selected_index = st.selectbox(
        "Select the exact track, artist, and year",
        options,
        format_func=lambda index: track_label(row_by_index[index]),
    )

selected = resolve_catalog_track(catalog, model_index=int(selected_index))

st.markdown("<div class='section-kicker'>Selected catalog item</div>", unsafe_allow_html=True)
metrics = st.columns([1.4, 1.4, 0.7, 0.7])
metrics[0].metric("Track", str(selected["name"]))
metrics[1].metric("Artist", format_artists(selected["artists"]))
metrics[2].metric("Year", int(selected["year"]))
metrics[3].metric("Popularity", int(round(float(selected["popularity"]))))

if st.button(
    "Generate recommendations",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Comparing standardized audio profiles…"):
        st.session_state["recommendations"] = fast_recommend(
            artifacts,
            model_index=int(selected_index),
            top_n=int(top_n),
        )
        st.session_state["query_index"] = int(selected_index)
        st.session_state["query_top_n"] = int(top_n)

recommendations = st.session_state.get("recommendations")
same_query = st.session_state.get("query_index") == int(selected_index)
same_top_n = st.session_state.get("query_top_n") == int(top_n)

if recommendations is None or not same_query or not same_top_n:
    st.markdown(
        """
        <p class="small-note">
            Select a track and run the model. The app loads the persisted scaler,
            feature contract, catalog, and nearest-neighbor model without retraining.
        </p>
        """,
        unsafe_allow_html=True,
    )
else:
    st.divider()
    st.markdown("<div class='section-kicker'>Model output</div>", unsafe_allow_html=True)
    st.subheader(f"Top {len(recommendations)} audio neighbors")
    st.caption(
        "Similarity is calculated as 1 − cosine distance. Popularity is displayed for context and is not used by the recommender."
    )

    card_columns = st.columns(2)
    for position, (_, recommendation) in enumerate(recommendations.iterrows()):
        with card_columns[position % 2]:
            recommendation_card(recommendation)

    with st.expander("View exact result table"):
        st.dataframe(
            recommendation_display_frame(recommendations),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Similarity (%)": st.column_config.ProgressColumn(
                    "Similarity (%)",
                    min_value=0,
                    max_value=100,
                    format="%.2f%%",
                ),
                "Popularity": st.column_config.ProgressColumn(
                    "Popularity",
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
            },
        )

    indices = [int(selected_index)] + recommendations[
        "recommended_model_index"
    ].astype(int).tolist()
    profile_rows = catalog.iloc[indices].copy()
    normalized = normalize_profiles(
        profile_rows,
        catalog,
        list(artifacts.features),
    )
    query_profile = normalized.iloc[0]
    recommendation_mean = normalized.iloc[1:].mean()
    labels = [feature.replace("_", " ").title() for feature in artifacts.features]

    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=query_profile.tolist() + [query_profile.iloc[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Selected track",
            line=dict(color=ACCENT, width=3),
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=recommendation_mean.tolist() + [recommendation_mean.iloc[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Recommendation average",
            opacity=0.68,
            line=dict(color=ACCENT_BLUE, width=2.5),
        )
    )
    radar.update_layout(
        title="Audio Profile Comparison",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="#7F8B99"),
                gridcolor=GRID,
            ),
            angularaxis=dict(gridcolor=GRID),
        ),
    )
    st.plotly_chart(style_radar(radar), use_container_width=True)
    st.caption(
        "Radar values use robust 5th-to-95th-percentile scaling for display only. The recommendation model uses StandardScaler and cosine distance."
    )

with st.expander("How the recommender works"):
    st.markdown(
        """
        1. Each track is represented by nine audio features: acousticness,
           danceability, energy, instrumentalness, liveness, loudness,
           speechiness, tempo, and valence.
        2. The persisted StandardScaler transforms the selected track.
        3. NearestNeighbors compares the standardized vector using cosine distance.
        4. The seed track and repeated name–artist pairs are excluded.
        5. The application returns the requested number of unique tracks.

        **Important:** internal correctness checks do not prove listener satisfaction.
        """
    )
