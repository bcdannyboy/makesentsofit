"""Sentiment analysis utilities using OpenAI or classic NLP."""
from __future__ import annotations

import logging
import os
import platform

# Comprehensive environment setup to prevent TensorFlow Metal backend issues on macOS
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORTS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Prevent TensorFlow from registering Metal backend multiple times
os.environ.setdefault("TF_DISABLE_METAL", "1")

from typing import List, Dict, Optional

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import opinion_lexicon
from textblob import TextBlob
import emoji
try:
    from openai import OpenAI  # type: ignore
    openai_available = True
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    openai_available = False

from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment of text or posts."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.preprocessor = TextPreprocessor()

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = bool(self.openai_api_key and openai_available)
        if self.use_openai and openai_available:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI API key provided - using ChatGPT for sentiment analysis")
            except TypeError:
                logger.warning("OpenAI client init failed, disabling OpenAI sentiment analysis")
                self.use_openai = False
                self.openai_client = None
        elif self.use_openai and not openai_available:
            logger.warning("openai package not installed, disabling OpenAI sentiment analysis")
            self.use_openai = False
        else:
            self.openai_client = None

        # Ensure VADER resources
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self.vader = SentimentIntensityAnalyzer()
        try:
            nltk.data.find("opinion_lexicon")
        except LookupError:  # pragma: no cover - first run
            nltk.download("opinion_lexicon", quiet=True)

        self.pos_words = set(opinion_lexicon.positive())
        self.neg_words = set(opinion_lexicon.negative())

        self.positive_emoji = {
            ":smile:", ":grinning_face_with_big_eyes:", ":heart_eyes:",
            ":thumbs_up:", ":grinning:", ":blush:", ":red_heart:"
        }
        self.negative_emoji = {
            ":frowning_face:", ":angry_face:", ":thumbs_down:",
            ":crying_face:", ":pensive_face:"
        }

        self.cache: Dict[str, Dict] = {}


    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts."""
        processed = [self.preprocessor.clean(t) for t in texts]

        if self.use_openai:
            results = []
            i = 0
            for raw, cleaned in zip(texts, processed):
                try:
                    resp = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Classify the sentiment of the following text as POSITIVE, NEGATIVE, NEUTRAL."},
                            {"role": "user", "content": cleaned},
                        ],
                        max_tokens=1,
                    )
                    
                    label = resp.choices[0].message.content.strip().upper()
                    
                    # Map abbreviated ChatGPT labels to full labels expected by the codebase
                    label_mapping = {
                        "POS": "POSITIVE",
                        "NEG": "NEGATIVE",
                        "NE": "NEUTRAL"
                    }
                    label = label_mapping.get(label, label)  # Use mapping if available, otherwise keep original

                    results.append({
                        "label": label,
                        "score": 1.0,
                        "method": "openai",
                        "original_text": raw[:200],
                    })
                except Exception as exc:  # pragma: no cover - network issues
                    logger.error("OpenAI sentiment failed: %s", exc)
                    results.append(self._analyze_with_combined(raw))
                i = i + 1
            return results

        return [self._analyze_with_combined(t) for t in texts]

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

    def _analyze_with_combined(self, text: str) -> Dict:
        """Analyze text using multiple classical NLP techniques."""
        cleaned = self.preprocessor.clean(text)
        vader = self.vader.polarity_scores(cleaned)["compound"]
        blob = TextBlob(cleaned).sentiment.polarity

        tokens = cleaned.lower().split()
        pos_count = sum(t in self.pos_words for t in tokens)
        neg_count = sum(t in self.neg_words for t in tokens)
        lex_score = 0.0
        if tokens:
            lex_score = (pos_count - neg_count) / len(tokens)

        emoji_text = emoji.demojize(text)
        pos_emoji = sum(e in emoji_text for e in self.positive_emoji)
        neg_emoji = sum(e in emoji_text for e in self.negative_emoji)
        emoji_score = 0.0
        if pos_emoji or neg_emoji:
            emoji_score = (pos_emoji - neg_emoji) / (pos_emoji + neg_emoji)

        features = self.preprocessor.extract_features(text)
        emphasis = features["exclamation_marks"] - features["question_marks"]

        combined = (vader + blob + lex_score) / 3
        combined += 0.1 * emoji_score + 0.02 * emphasis

        if combined >= 0.05:
            label = "POSITIVE"
        elif combined <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "score": abs(combined),
            "method": "comprehensive",
            "compound": combined,
            "original_text": text[:200],
            "details": {
                "vader": vader,
                "textblob": blob,
                "lexicon": lex_score,
                "emoji": emoji_score,
                "emphasis": emphasis,
            },
        }

    def analyze_posts(self, posts: List["Post"]) -> List["Post"]:
        """Attach sentiment to Post objects."""
        texts = [f"{p.title} {p.content}" if getattr(p, "title", None) else p.content for p in posts]
        sentiments = self.analyze_batch(texts)
        for post, sent in zip(posts, sentiments):
            post.sentiment = sent
        return posts
