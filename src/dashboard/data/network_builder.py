from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


def _increment_edge(G: nx.Graph, n1: Any, n2: Any) -> None:
    if G.has_edge(n1, n2):
        G[n1][n2]["weight"] += 1
    else:
        G.add_edge(n1, n2, weight=1)


def build_triangular_network(df: pd.DataFrame) -> nx.Graph:
    """Create user-subreddit-topic network graph."""
    G = nx.Graph()
    if df.empty:
        return G

    for _, row in df.iterrows():
        user = row.get("author")
        subreddit = row.get("subreddit") or row.get("metadata", {}).get("subreddit")
        topic = row.get("query") or row.get("topic")

        if user:
            G.add_node(user, node_type="user")
        if subreddit:
            G.add_node(subreddit, node_type="subreddit")
        if topic:
            G.add_node(topic, node_type="topic")

        if user and subreddit:
            _increment_edge(G, user, subreddit)
        if user and topic:
            _increment_edge(G, user, topic)
        if subreddit and topic:
            _increment_edge(G, subreddit, topic)

    return G

