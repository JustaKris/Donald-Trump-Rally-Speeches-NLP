"""
ML model loading and inference utilities.

This module handles loading pre-trained models and running inference
for sentiment analysis and other NLP tasks.
"""

import logging
import os
import ssl
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForSequenceClassification

from .preprocessing import chunk_text_for_bert

# Suppress TensorFlow warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only show errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops messages

# Set TensorFlow logging level
tf.get_logger().setLevel("ERROR")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*tf.get_default_graph.*")

# Suppress transformers warnings about model initialization
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configure consistent logging format
logging.basicConfig(
    level=logging.INFO,
    format="INFO:     %(message)s",
)
logger = logging.getLogger(__name__)

# Handle SSL certificate issues when downloading models
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]


class SentimentAnalyzer:
    """
    Sentiment analysis using FinBERT model.

    This analyzer uses a pre-trained BERT model fine-tuned for sentiment
    classification to classify text into positive, negative, or neutral categories.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer: AutoTokenizer
        self.model: TFBertForSequenceClassification
        self._load_model()

    def _load_model(self):
        """Load the tokenizer and model from HuggingFace."""
        logger.info(f"Loading {self.model_name}...")

        # Suppress specific model loading warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not initialized from the model checkpoint.*"
            )
            warnings.filterwarnings("ignore", message=".*TRAIN this model.*")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFBertForSequenceClassification.from_pretrained(self.model_name)

        logger.info("Model loaded successfully!")

    def analyze_sentiment(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment of input text.

        For long texts, automatically chunks into smaller segments and
        averages the predictions.

        Args:
            text: Input text to analyze
            return_all_scores: If True, return all three sentiment scores

        Returns:
            Dictionary with sentiment scores and dominant sentiment
        """
        # Chunk text if needed
        chunks = chunk_text_for_bert(text, self.tokenizer, max_length=510)

        # Analyze each chunk
        all_predictions: List[np.ndarray] = []
        for chunk in chunks:
            outputs = self.model(chunk)
            probs = tf.nn.softmax(outputs.logits, axis=-1)
            all_predictions.append(probs.numpy())  # type: ignore[attr-defined]

        # Average predictions across chunks
        predictions_array = np.vstack(all_predictions)
        mean_sentiment = np.mean(predictions_array, axis=0)

        # Get label names
        labels = self.model.config.id2label

        # Prepare result
        dominant_idx = int(np.argmax(mean_sentiment))
        result = {
            "positive": float(mean_sentiment[0]),
            "negative": float(mean_sentiment[1]),
            "neutral": float(mean_sentiment[2]),
            "dominant": labels[dominant_idx],  # type: ignore[union-attr]
            "num_chunks": len(chunks),
        }

        if not return_all_scores:
            # Return only dominant sentiment
            return {
                "sentiment": labels[dominant_idx],  # type: ignore[union-attr]  # type: ignore[union-attr,index]
                "confidence": float(mean_sentiment[dominant_idx]),
                "num_chunks": len(chunks),
            }

        return result

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_sentiment(text, return_all_scores=True) for text in texts]


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """
    Factory function to get a sentiment analyzer instance.

    This allows for lazy loading and caching of the model.

    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer()
