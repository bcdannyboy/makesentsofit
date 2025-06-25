import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict


class NetworkGraphGenerator:
    """Generate network graphs of user interactions."""

    def create_user_network(self, posts: List["Post"], output_path: str, min_connections: int = 2) -> None:
        """Create network graph of user interactions."""
        if not posts:
            return

        G = nx.Graph()
        connections: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "subreddits": set()})

        for post in posts:
            author = post.author
            subreddit = post.metadata.get("subreddit")
            for mention in post.metadata.get("mentions", []):
                key = (author, mention)
                connections[key]["count"] += 1
                if post.platform == "reddit" and subreddit:
                    connections[key]["subreddits"].add(subreddit)

        for (author, mention), data in connections.items():
            if data["count"] >= min_connections:
                G.add_edge(author, mention, weight=data["count"], subreddits=list(data["subreddits"]))

        if len(G.nodes()) == 0:
            return

        degree_centrality = nx.degree_centrality(G)
        node_sizes = [300 + 1000 * degree_centrality.get(n, 0) for n in G.nodes()]

        node_colors = []
        author_sentiments: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
        for post in posts:
            if hasattr(post, "sentiment"):
                label = post.sentiment.get("label", "NEUTRAL").lower()
                author_sentiments[post.author][label] += 1

        for node in G.nodes():
            sentiments = author_sentiments[node]
            total = sum(sentiments.values())
            if total == 0:
                node_colors.append("gray")
            elif sentiments["negative"] / total > 0.6:
                node_colors.append("red")
            elif sentiments["positive"] / total > 0.6:
                node_colors.append("green")
            else:
                node_colors.append("yellow")

        fig, ax = plt.subplots(figsize=(12, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.3, ax=ax)

        labels = {node: node for node in G.nodes() if degree_centrality[node] > 0.1}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        ax.set_title("User Interaction Network", fontsize=16, fontweight="bold")
        ax.axis("off")

        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", label="Positive", markerfacecolor="g", markersize=10),
            plt.Line2D([0], [0], marker="o", color="w", label="Negative", markerfacecolor="r", markersize=10),
            plt.Line2D([0], [0], marker="o", color="w", label="Mixed/Neutral", markerfacecolor="y", markersize=10),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
