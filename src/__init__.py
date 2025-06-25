"""
MakeSenseOfIt source package.
Social Media Sentiment Analysis tool.
"""

__version__ = '1.0.0'
__author__ = 'MakeSenseOfIt Team'
__email__ = 'contact@makesentsofit.com'

# Package metadata
__all__ = ['Config', 'setup_logging', 'get_logger', 'SentimentAnalyzer']

# Import main components for easier access
from .config import Config
from .logger import setup_logging, get_logger

# Lazy import for SentimentAnalyzer to avoid loading heavy dependencies
_sentiment_analyzer = None

def SentimentAnalyzer(*args, **kwargs):
    """Lazy loader for SentimentAnalyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from .sentiment import SentimentAnalyzer as _SA
        _sentiment_analyzer = _SA
    return _sentiment_analyzer(*args, **kwargs)