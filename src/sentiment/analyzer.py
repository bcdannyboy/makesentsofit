"""Sentiment analysis utilities using transformers with VADER fallback."""
from __future__ import annotations

import logging
from typing import List, Dict

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

# Don't import transformers at module level - will be imported lazily when needed
pipeline = None
transformers_available = None

from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment of text or posts."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()

        # Don't initialize transformer at creation time - wait until needed
        self.transformer_available = None  # Unknown until we try
        self.transformer = None
        self._transformer_init_attempted = False

        # Ensure VADER resources
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self.vader = SentimentIntensityAnalyzer()
        self.cache: Dict[str, Dict] = {}

    def _try_init_transformer(self):
        """Lazy initialization of transformer model."""
        if self._transformer_init_attempted:
            return
        
        self._transformer_init_attempted = True
        try:
            # Lazy import of transformers - only import when needed
            from transformers import pipeline as tf_pipeline
            
            device = 0 if torch and hasattr(torch, 'cuda') and torch.cuda.is_available() else -1
            self.transformer = tf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
                truncation=True,
                max_length=512,
            )
            self.transformer_available = True
            logger.info("Loaded transformer model: %s", self.model_name)
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("Could not load transformer model: %s", exc)
            self.transformer_available = False
            self.transformer = None

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts."""
        processed = [self.preprocessor.clean(t) for t in texts]

        # Try to initialize transformer on first use
        if self.transformer_available is None:
            self._try_init_transformer()

        if self.transformer_available:
            try:
                preds = self.transformer(processed)
                return [
                    {
                        "label": p["label"],
                        "score": p["score"],
                        "method": "transformer",
                        "original_text": t[:200],
                    }
                    for t, p in zip(texts, preds)
                ]
            except Exception as exc:  # pragma: no cover - network/model issues
                logger.error("Transformer batch processing failed: %s", exc)

        return [self._analyze_with_vader(t) for t in texts]

    def _analyze_with_vader(self, text: str) -> Dict:
        """Analyze a single text using VADER."""
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return {
            "label": label,
            "score": abs(compound),
            "method": "vader",
            "compound": compound,
            "original_text": text[:200],
            "details": scores,
        }

    def analyze_posts(self, posts: List["Post"]) -> List["Post"]:
        """Attach sentiment to Post objects."""
        texts = [f"{p.title} {p.content}" if getattr(p, "title", None) else p.content for p in posts]
        sentiments = self.analyze_batch(texts)
        for post, sent in zip(posts, sentiments):
            post.sentiment = sent
        return posts
