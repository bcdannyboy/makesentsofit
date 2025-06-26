from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from .data.loader import load_analysis, load_user_network
from .data.network_builder import build_triangular_network
from .components.sidebar import apply_filters
from .components.network_viz import draw_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MakeSenseOfIt Dashboard")
    parser.add_argument("--analysis", type=Path, required=True, help="Path to analysis JSON")
    parser.add_argument("--user-network", type=Path, default=None, help="Path to user network JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="MakeSenseOfIt Dashboard", layout="wide")
    st.title("MakeSenseOfIt Interactive Dashboard")

    df = load_analysis(args.analysis)
    user_network = load_user_network(args.user_network) if args.user_network else {}

    filtered_df = apply_filters(df)
    graph = build_triangular_network(filtered_df)

    st.subheader("Network Visualization")
    draw_network(graph)

    st.subheader("Data Preview")
    st.dataframe(filtered_df, use_container_width=True)

    if user_network:
        st.subheader("User Network Metrics")
        st.json(user_network)


if __name__ == "__main__":
    main()

