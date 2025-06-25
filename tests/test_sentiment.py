"""Tests for sentiment analysis functionality."""
from unittest.mock import patch

from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.preprocessor import TextPreprocessor


def test_preprocessor():
    preprocessor = TextPreprocessor()

    text = "Check out https://example.com for more"
    cleaned = preprocessor.clean(text)
    assert "https://" not in cleaned

    text = "@user1 @user2 hello"
    cleaned = preprocessor.clean(text)
    assert "user1" in cleaned
    assert "@" not in cleaned

    text = "#Python #Programming is fun"
    cleaned = preprocessor.clean(text)
    assert "Python Programming" in cleaned
    assert "#" not in cleaned


@patch("src.sentiment.analyzer.pipeline")
def test_sentiment_analyzer(mock_pipeline):
    mock_pipeline.return_value = lambda texts: [
        {"label": "POSITIVE", "score": 0.9} for _ in texts
    ]
    analyzer = SentimentAnalyzer()

    texts = [
        "I love this! It's amazing!",
        "This is terrible and awful.",
        "It's okay, nothing special.",
    ]

    results = analyzer.analyze_batch(texts)

    assert len(results) == 3
    assert results[0]["label"] == "POSITIVE"
    assert results[1]["label"] == "POSITIVE"  # mocked
    assert results[2]["label"] == "POSITIVE"


def test_vader_fallback():
    analyzer = SentimentAnalyzer()
    result = analyzer._analyze_with_vader("This is fantastic!")

    assert result["method"] == "vader"
    assert result["label"] == "POSITIVE"
    assert "compound" in result
