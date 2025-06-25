"""Helper for loading transformer sentiment models."""
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)


def load_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Load a sentiment analysis pipeline."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        logger.warning("Failed to load sentiment model %s: %s", model_name, exc)
        raise
