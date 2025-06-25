"""Helper for loading transformer sentiment models."""
import os
import logging

logger = logging.getLogger(__name__)


def load_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Load a sentiment analysis pipeline."""
    try:
        # Set comprehensive environment variables to prevent TensorFlow issues
        os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORTS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        
        # Lazy import to avoid module-level TensorFlow loading
        try:
            import torch
        except ImportError:
            torch = None
            
        from transformers import pipeline
        
        device = 0 if torch and torch.cuda.is_available() else -1
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512,
            framework="pt",  # Force PyTorch framework
        )
    except Exception as exc:
        logger.warning("Failed to load sentiment model %s: %s", model_name, exc)
        raise
