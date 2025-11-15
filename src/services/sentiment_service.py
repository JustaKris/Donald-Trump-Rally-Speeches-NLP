"""
AI-powered sentiment analysis with emotion detection.

This module handles sentiment analysis using multiple models:
- FinBERT for financial/political sentiment (positive/negative/neutral)
- RoBERTa for emotion classification (anger, joy, fear, sadness, etc.)
- LLM for contextual sentiment interpretation
"""

import logging
import os
import ssl
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config.settings import get_settings
from ..utils.text_preprocessing import chunk_text_for_bert
from .llm.base import LLMProvider

# Force transformers to use PyTorch backend (not TensorFlow)
os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings if accidentally loaded

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress transformers warnings about model initialization
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Handle SSL certificate issues when downloading models
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]


class EnhancedSentimentAnalyzer:
    """
    AI-powered sentiment analysis with emotion detection and contextual interpretation.

    Combines multiple models:
    - FinBERT: Sentiment classification (positive/negative/neutral)
    - RoBERTa-Emotion: Multi-emotion detection (anger, joy, fear, sadness, surprise, disgust)
    - LLM: Contextual semantic sentiment dissection
    """

    def __init__(
        self,
        sentiment_model: str = "ProsusAI/finbert",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        llm_service: Optional[LLMProvider] = None,
    ):
        """
        Initialize the enhanced sentiment analyzer.

        Args:
            sentiment_model: HuggingFace model for sentiment classification
            emotion_model: HuggingFace model for emotion detection
            llm_service: LLM service instance for contextual analysis
        """
        self.settings = get_settings()
        self.sentiment_model_name = sentiment_model
        self.emotion_model_name = emotion_model
        self.llm_service = llm_service

        # Load models
        self._load_sentiment_model()
        self._load_emotion_model()

        logger.info("Enhanced sentiment analyzer initialized")

    def _load_sentiment_model(self):
        """Load the FinBERT sentiment model."""
        logger.debug(f"Loading sentiment model: {self.sentiment_model_name}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not initialized from the model checkpoint.*"
            )
            warnings.filterwarnings("ignore", message=".*TRAIN this model.*")

            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_name
            )
            self.sentiment_model.eval()

        logger.info(f"Sentiment model loaded: {self.sentiment_model_name}")

    def _load_emotion_model(self):
        """Load the RoBERTa emotion classification model."""
        logger.debug(f"Loading emotion model: {self.emotion_model_name}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not initialized from the model checkpoint.*"
            )
            warnings.filterwarnings("ignore", message=".*TRAIN this model.*")

            self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_name)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
                self.emotion_model_name
            )
            self.emotion_model.eval()

        logger.info(f"Emotion model loaded: {self.emotion_model_name}")

    def _analyze_sentiment_scores(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT model."""
        chunks = chunk_text_for_bert(text, self.sentiment_tokenizer, max_length=510)

        all_predictions: List[np.ndarray] = []
        for chunk in chunks:
            with torch.no_grad():
                outputs = self.sentiment_model(**chunk)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_predictions.append(probs.cpu().numpy())

        predictions_array = np.vstack(all_predictions)
        mean_sentiment = np.mean(predictions_array, axis=0)

        labels = self.sentiment_model.config.id2label
        dominant_idx = int(np.argmax(mean_sentiment))

        return {
            "positive": float(mean_sentiment[0]),
            "negative": float(mean_sentiment[1]),
            "neutral": float(mean_sentiment[2]),
            "dominant": labels[dominant_idx],
            "num_chunks": len(chunks),
        }

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions using RoBERTa emotion model."""
        # Truncate text if too long
        max_length = 512
        inputs = self.emotion_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            emotion_scores = probs.cpu().numpy()[0]

        # Get emotion labels from model config
        labels = self.emotion_model.config.id2label
        emotions = {labels[i]: float(emotion_scores[i]) for i in range(len(emotion_scores))}

        return emotions

    def _generate_contextual_interpretation(
        self, text: str, sentiment_scores: Dict[str, float], emotions: Dict[str, float]
    ) -> str:
        """Generate contextual sentiment interpretation using Gemini LLM."""
        if self.llm_service is None:
            # Fallback to simple interpretation without LLM
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            dominant_sentiment = max(
                [(k, v) for k, v in sentiment_scores.items() if k != "num_chunks"],
                key=lambda x: x[1],
            )
            return f"The text conveys a {dominant_sentiment[0]} sentiment with {top_emotion[0]} being the primary emotion detected."

        try:
            # Get top 3 emotions for context
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            emotion_summary = ", ".join([f"{e[0]} ({e[1]:.0%})" for e in top_emotions])

            # Identify dominant sentiment
            dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            dominant_emotion = top_emotions[0]

            # Create comprehensive prompt for LLM
            prompt = f"""You are analyzing the emotional and sentimental tone of a text excerpt. Provide a clear, insightful interpretation that explains WHY the models produced their results.

TEXT ANALYZED:
"{text[:600]}{'...' if len(text) > 600 else ''}"

SENTIMENT ANALYSIS RESULTS:
- Overall Sentiment: {dominant_sentiment[0].upper()} ({dominant_sentiment[1]:.0%} confidence)
- Positive: {sentiment_scores['positive']:.0%}
- Negative: {sentiment_scores['negative']:.0%}
- Neutral: {sentiment_scores['neutral']:.0%}

EMOTION DETECTION RESULTS:
- Primary Emotion: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]:.0%})
- Top 3 Emotions: {emotion_summary}

TASK:
Write a 2-3 sentence interpretation that:
1. Explains WHY the text received a {dominant_sentiment[0]} sentiment score
2. Explains WHY {dominant_emotion[0]} is the dominant emotion
3. Connects both findings to specific aspects of the text content

Focus on WHAT the speaker expresses emotion about, not just labeling emotions. Be specific and insightful.

Example style: "The text expresses strong positive sentiment about economic achievements, with joy emerging from pride in policy success. However, underlying anger surfaces when discussing immigration, creating emotional complexity that explains the mixed sentiment profile."

Your interpretation:"""

            # Generate using LLM provider interface
            response = self.llm_service.generate_content(prompt)

            # Check if response is valid (not blocked by safety filters)
            if not response or not hasattr(response, "text") or not response.text:
                raise ValueError("LLM response was blocked or empty")

            # Extract text from response
            interpretation = response.text.strip()
            if interpretation.startswith('"') and interpretation.endswith('"'):
                interpretation = interpretation[1:-1]

            return interpretation

        except Exception as e:
            logger.warning(f"Failed to generate contextual interpretation: {e}")
            # Fallback with more insightful interpretation
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            second_emotion = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[1]
            dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])

            # Create a more detailed fallback message
            if dominant_sentiment[0] in sentiment_scores:
                other_scores = {
                    k: v for k, v in sentiment_scores.items() if k != dominant_sentiment[0]
                }
                secondary = max(other_scores.items(), key=lambda x: x[1])
                return f"The text conveys {dominant_sentiment[0]} sentiment ({dominant_sentiment[1]:.0%}) with {secondary[0]} secondary notes ({secondary[1]:.0%}). The emotional tone is primarily {top_emotion[0]} ({top_emotion[1]:.0%}) with some {second_emotion[0]} ({second_emotion[1]:.0%}), suggesting a {dominant_sentiment[0].lower()} but complex emotional landscape."

            return f"The text exhibits {dominant_sentiment[0]} sentiment (score: {dominant_sentiment[1]:.0%}) with {top_emotion[0]} as the dominant emotion ({top_emotion[1]:.0%})."

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis with emotion detection and contextual interpretation.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment scores, emotions, and contextual interpretation
        """
        # Analyze sentiment
        sentiment_result = self._analyze_sentiment_scores(text)

        # Analyze emotions
        emotions = self._analyze_emotions(text)

        # Generate contextual interpretation
        contextual_sentiment = self._generate_contextual_interpretation(
            text,
            {
                "positive": sentiment_result["positive"],
                "negative": sentiment_result["negative"],
                "neutral": sentiment_result["neutral"],
            },
            emotions,
        )

        return {
            "sentiment": sentiment_result["dominant"],
            "confidence": sentiment_result[sentiment_result["dominant"]],
            "scores": {
                "positive": sentiment_result["positive"],
                "negative": sentiment_result["negative"],
                "neutral": sentiment_result["neutral"],
            },
            "emotions": emotions,
            "contextual_sentiment": contextual_sentiment,
            "num_chunks": sentiment_result["num_chunks"],
        }


def get_sentiment_analyzer(llm_service: Optional[Any] = None) -> EnhancedSentimentAnalyzer:
    """
    Factory function to get an enhanced sentiment analyzer instance.

    Args:
        llm_service: Optional LLM service for contextual analysis

    Returns:
        EnhancedSentimentAnalyzer instance
    """
    return EnhancedSentimentAnalyzer(llm_service=llm_service)
