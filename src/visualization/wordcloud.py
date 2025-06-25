from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List
import re


class WordCloudGenerator:
    """Generate word clouds from posts."""

    def __init__(self) -> None:
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "rt",
            "via",
            "amp",
            "https",
            "http",
            "com",
            "it",
            "this",
            "that",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "i",
            "you",
            "he",
            "she",
            "they",
            "we",
            "my",
            "your",
            "their",
        }

    def create_wordcloud(self, posts: List["Post"], output_path: str, sentiment_filter: str | None = None) -> None:
        """Create word cloud from posts."""
        if not posts:
            return

        if sentiment_filter:
            posts = [p for p in posts if hasattr(p, "sentiment") and p.sentiment.get("label") == sentiment_filter]
        if not posts:
            return

        all_text = " ".join([f"{p.title} {p.content}" if p.title else p.content for p in posts])
        all_text = re.sub(r"https?://\S+", "", all_text)
        all_text = re.sub(r"@\w+", "", all_text)
        all_text = re.sub(r"#(\w+)", r"\1", all_text)
        all_text = re.sub(r"[^\w\s]", " ", all_text)
        all_text = all_text.lower()

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=self.stopwords,
            max_words=100,
            relative_scaling=0.5,
            colormap="viridis" if sentiment_filter == "POSITIVE" else "Reds",
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        title = "Word Cloud"
        if sentiment_filter:
            title += f" - {sentiment_filter} Sentiment"
        ax.set_title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
