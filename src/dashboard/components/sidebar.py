from __future__ import annotations

import pandas as pd
import streamlit as st


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Build sidebar controls and filter the dataframe."""
    if df.empty:
        st.sidebar.info("No data loaded")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    st.sidebar.header("Filters")
    start, end = st.sidebar.date_input(
        "Date range",
        [min_date.date(), max_date.date()],
    )

    score_min, score_max = st.sidebar.slider(
        "Sentiment score",
        -1.0,
        1.0,
        (-1.0, 1.0),
        step=0.1,
    )

    mask = (
        (df["timestamp"] >= pd.to_datetime(start))
        & (df["timestamp"] <= pd.to_datetime(end))
    )
    if "sentiment" in df.columns and isinstance(df.loc[0, "sentiment"], dict):
        scores = df["sentiment"].apply(lambda x: x.get("score", 0))
    else:
        scores = df.get("sentiment_score", pd.Series([0] * len(df)))
    mask &= scores.between(score_min, score_max)

    return df[mask]

