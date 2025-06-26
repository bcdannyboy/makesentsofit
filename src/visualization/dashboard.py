import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from typing import List
from jinja2 import Environment, FileSystemLoader, select_autoescape


class DashboardGenerator:
    """Generate an interactive HTML dashboard for scraped posts."""

    def __init__(self):
        template_dir = Path(__file__).resolve().parent.parent / "export" / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _create_network_html(self, posts: List["Post"]) -> str:
        """Create HTML for subreddit-topic network graph."""
        G = nx.Graph()
        for post in posts:
            subreddit = post.metadata.get("subreddit", "unknown")
            topic = getattr(post, "query", "unknown")
            G.add_node(subreddit, node_type="subreddit")
            G.add_node(topic, node_type="topic")
            if G.has_edge(subreddit, topic):
                G[subreddit][topic]["weight"] += 1
            else:
                G.add_edge(subreddit, topic, weight=1)

        if len(G.nodes()) == 0:
            return ""

        pos = nx.spring_layout(G, k=1)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=text,
            textposition="bottom center",
            hoverinfo="text",
            marker=dict(size=12, color="lightblue", line=dict(width=1, color="darkblue")),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Subreddit / Topic Network",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def _create_sentiment_bar(self, df: pd.DataFrame) -> str:
        counts = df["sentiment"].value_counts()
        if counts.empty:
            return ""
        fig = go.Figure([
            go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color=["green", "red", "gray"])
        ])
        fig.update_layout(title="Sentiment Distribution", yaxis_title="Posts")
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def generate_dashboard(self, posts: List["Post"], output_path: str) -> None:
        """Generate dashboard HTML from posts."""
        if not posts:
            return
        df_rows = []
        for post in posts:
            row = {
                "author": post.author,
                "subreddit": post.metadata.get("subreddit"),
                "topic": getattr(post, "query", ""),
                "sentiment": getattr(post, "sentiment", {}).get("label") if hasattr(post, "sentiment") else None,
                "score": getattr(post, "sentiment", {}).get("score") if hasattr(post, "sentiment") else None,
                "title": post.title,
                "content": post.content,
                "url": post.url,
                "timestamp": post.timestamp,
            }
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        network_html = self._create_network_html(posts)
        sentiment_bar = self._create_sentiment_bar(df)
        table_html = df.to_html(classes="table table-striped", index=False, border=0)

        template = self.env.get_template("dashboard.html")
        html = template.render(
            network_graph=network_html,
            sentiment_graph=sentiment_bar,
            table_html=table_html,
            post_count=len(df),
        )
        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        return None
