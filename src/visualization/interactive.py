import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List


class InteractiveChartGenerator:
    """Generate interactive Plotly charts."""

    def create_interactive_timeline(self, time_series_data: Dict, output_path: str) -> None:
        """Create interactive sentiment timeline."""
        if not time_series_data:
            return

        df_data = []
        for date, data in sorted(time_series_data.items()):
            df_data.append(
                {
                    "date": date,
                    "positive": data["positive"],
                    "negative": data["negative"],
                    "neutral": data["neutral"],
                    "total": data["total"],
                    "sentiment_ratio": data["sentiment_ratio"],
                }
            )
        df = pd.DataFrame(df_data)
        df["date"] = pd.to_datetime(df["date"])

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Sentiment Distribution", "Post Volume"),
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(
            go.Scatter(x=df["date"], y=df["positive"], name="Positive", fill="tonexty", line=dict(color="green", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["negative"], name="Negative", fill="tonexty", line=dict(color="red", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["neutral"], name="Neutral", fill="tonexty", line=dict(color="gray", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=df["date"], y=df["total"], name="Volume", marker_color="lightblue"),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Posts", row=2, col=1)
        fig.update_layout(title="Interactive Sentiment Timeline", hovermode="x unified", showlegend=True, height=600)
        fig.write_html(output_path)

    def create_3d_sentiment_scatter(self, posts: List["Post"], output_path: str) -> None:
        """Create 3D scatter plot of posts."""
        if not posts or len(posts) < 10:
            return

        data = []
        for post in posts[:1000]:
            if hasattr(post, "sentiment"):
                data.append(
                    {
                        "timestamp": post.timestamp,
                        "engagement": post.engagement.get("likes", 0) + post.engagement.get("retweets", 0),
                        "sentiment_score": post.sentiment.get("score", 0),
                        "sentiment_label": post.sentiment.get("label", "UNKNOWN"),
                        "author": post.author,
                        "content_preview": post.content[:100] + "...",
                    }
                )
        df = pd.DataFrame(data)
        fig = px.scatter_3d(
            df,
            x="timestamp",
            y="engagement",
            z="sentiment_score",
            color="sentiment_label",
            color_discrete_map={"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"},
            hover_data=["author", "content_preview"],
            title="3D Sentiment Analysis",
        )
        fig.update_layout(scene=dict(xaxis_title="Time", yaxis_title="Engagement", zaxis_title="Sentiment Score"), height=700)
        fig.write_html(output_path)
