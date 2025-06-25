import pytest
from pathlib import Path
from src.visualization import ChartGenerator, WordCloudGenerator
from src.scrapers.base import Post
from datetime import datetime


def test_chart_generation(tmp_path: Path):
    charts = ChartGenerator()
    time_series_data = {
        "2024-01-01": {"positive": 10, "negative": 5, "neutral": 5, "total": 20, "sentiment_ratio": 0.25},
        "2024-01-02": {"positive": 15, "negative": 3, "neutral": 7, "total": 25, "sentiment_ratio": 0.48},
    }
    output_path = tmp_path / "timeline.png"
    charts.sentiment_timeline(time_series_data, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_wordcloud_generation(tmp_path: Path):
    post = Post(
        id="1",
        platform="twitter",
        author="test",
        author_id="123",
        content="Python programming is amazing and fun",
        title=None,
        timestamp=datetime.now(),
        engagement={},
        url="",
        query="test",
        metadata={},
    )
    post.sentiment = {"label": "POSITIVE", "score": 0.9}
    posts = [post]
    wc = WordCloudGenerator()
    output_path = tmp_path / "wordcloud.png"
    wc.create_wordcloud(posts, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0
