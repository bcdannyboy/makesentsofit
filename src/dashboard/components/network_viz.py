from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st


NODE_COLORS = {
    "user": "#1f77b4",
    "subreddit": "#ff7f0e",
    "topic": "#2ca02c",
}


def draw_network(graph: nx.Graph) -> None:
    """Render the network graph using Plotly and Streamlit."""
    if graph.number_of_nodes() == 0:
        st.info("No data available for this selection")
        return

    pos = nx.spring_layout(graph, k=0.6, seed=42)

    edge_x, edge_y = [], []
    for n1, n2 in graph.edges():
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#aaa"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y, node_text, node_color = [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(NODE_COLORS.get(data.get("node_type", "user"), "#1f77b4"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=12,
            color=node_color,
            line=dict(width=1, color="white"),
            opacity=0.9,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        plot_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

