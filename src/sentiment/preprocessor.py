"""Text preprocessing utilities for sentiment analysis."""
from typing import Dict
import re
import emoji

class TextPreprocessor:
    """Preprocess text for sentiment analysis."""

    def __init__(self) -> None:
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if not text:
            return ""

        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))

        # Remove URLs
        text = self.url_pattern.sub("", text)

        # Keep mentions but drop '@'
        text = self.mention_pattern.sub(lambda m: m.group()[1:], text)

        # Keep hashtags but drop '#'
        text = self.hashtag_pattern.sub(r"\1", text)

        # Normalize whitespace
        text = self.whitespace_pattern.sub(" ", text)
        text = text.strip()
        return text

    def extract_features(self, text: str) -> Dict:
        """Extract simple textual features."""
        return {
            "has_urls": bool(self.url_pattern.search(text)),
            "num_mentions": len(self.mention_pattern.findall(text)),
            "num_hashtags": len(self.hashtag_pattern.findall(text)),
            "num_emojis": len([c for c in text if c in emoji.EMOJI_DATA]),
            "text_length": len(text),
            "num_words": len(text.split()),
            "exclamation_marks": text.count("!"),
            "question_marks": text.count("?"),
            "all_caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }
