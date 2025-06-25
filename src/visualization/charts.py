import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np


class ChartGenerator:
    """Generate static charts and graphs."""

    def __init__(self, style: str = "seaborn") -> None:
        try:
            if style in plt.style.available:
                plt.style.use(style)
            else:
                plt.style.use("default")
        except OSError:
            plt.style.use("default")
        sns.set_palette("husl")

    def sentiment_timeline(self, time_series_data: Dict, output_path: str) -> None:
        """Create sentiment timeline chart."""
        if not time_series_data:
            return

        dates: List[datetime] = []
        positive: List[float] = []
        negative: List[float] = []
        neutral: List[float] = []

        for date_str, data in sorted(time_series_data.items()):
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            total = data.get("total", 0)
            if total > 0:
                positive.append(data.get("positive", 0) / total * 100)
                negative.append(data.get("negative", 0) / total * 100)
                neutral.append(data.get("neutral", 0) / total * 100)
            else:
                positive.append(0)
                negative.append(0)
                neutral.append(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.fill_between(dates, 0, positive, label="Positive", alpha=0.7, color="green")
        ax1.fill_between(
            dates,
            positive,
            [p + n for p, n in zip(positive, negative)],
            label="Negative",
            alpha=0.7,
            color="red",
        )
        ax1.fill_between(
            dates,
            [p + n for p, n in zip(positive, negative)],
            [p + n + m for p, n, m in zip(positive, negative, neutral)],
            label="Neutral",
            alpha=0.7,
            color="gray",
        )
        ax1.set_ylabel("Sentiment Distribution (%)")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        volumes = [data.get("total", 0) for _, data in sorted(time_series_data.items())]
        ax2.bar(dates, volumes, alpha=0.5, width=0.8)
        ax2.set_ylabel("Post Volume")
        ax2.set_xlabel("Date")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def sentiment_distribution_pie(self, sentiment_dist: Dict, output_path: str) -> None:
        """Create sentiment distribution pie chart."""
        if not sentiment_dist:
            return

        labels = list(sentiment_dist.keys())
        sizes = list(sentiment_dist.values())
        colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=[colors.get(l, "blue") for l in labels],
            autopct="%1.1f%%",
            startangle=90,
        )
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

        ax.set_title("Sentiment Distribution", fontsize=16, fontweight="bold")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def engagement_heatmap(self, posts: List["Post"], output_path: str) -> None:
        """Create engagement heatmap by hour and day."""
        if not posts:
            return

        data = []
        for post in posts:
            engagement = post.engagement.get("likes", 0) + post.engagement.get("retweets", 0)
            data.append({"hour": post.timestamp.hour, "day": post.timestamp.strftime("%A"), "engagement": engagement})

        df = pd.DataFrame(data)
        pivot = df.pivot_table(values="engagement", index="day", columns="hour", aggfunc="mean")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", cbar_kws={"label": "Average Engagement"}, ax=ax)
        ax.set_title("Engagement Heatmap by Day and Hour", fontsize=16)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
