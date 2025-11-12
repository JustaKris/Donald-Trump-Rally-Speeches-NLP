"""
NLP analysis endpoints.

Provides text analysis features including sentiment analysis, word frequency,
topic extraction, and n-gram analysis.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..models import (
    EnhancedTopicResponse,
    NGramRequest,
    SentimentResponse,
    StatsResponse,
    TextInput,
    TopicResponse,
    WordFrequencyResponse,
)
from ..services import NLPService, SentimentAnalyzer
from ..utils import clean_text
from .dependencies import get_nlp_service, get_sentiment_analyzer_dep, get_topic_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["nlp"])


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    input: TextInput,
    sentiment_analyzer: Optional[SentimentAnalyzer] = Depends(get_sentiment_analyzer_dep),
):
    """
    Analyze sentiment of any input text using FinBERT transformer model.

    This endpoint uses a pre-trained BERT-based model fine-tuned for sentiment analysis.
    Works with any type of text: social media posts, product reviews, news articles, etc.

    Returns the dominant sentiment (positive/negative/neutral) with confidence scores.
    For longer texts, automatically chunks and averages predictions for accuracy.
    """
    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment model not loaded. Please try again later.",
        )

    try:
        result = sentiment_analyzer.analyze_sentiment(input.text, return_all_scores=True)

        return SentimentResponse(
            sentiment=result["dominant"],
            confidence=result[result["dominant"]],
            scores={
                "positive": result["positive"],
                "negative": result["negative"],
                "neutral": result["neutral"],
            },
            num_chunks=result["num_chunks"],
        )
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@router.post("/words", response_model=WordFrequencyResponse)
async def analyze_word_frequency(
    input: TextInput,
    top_n: int = 50,
    nlp_service: NLPService = Depends(get_nlp_service),
):
    """
    Analyze word frequency in the input text.

    Returns the most common words with their frequencies, excluding stopwords.
    """
    try:
        stats = nlp_service.analyze_word_frequency(input.text, top_n=top_n)
        return WordFrequencyResponse(**stats)
    except Exception as e:
        logger.error(f"Word frequency error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@router.post("/topics", response_model=EnhancedTopicResponse)
async def analyze_topics(
    input: TextInput,
    top_n: int = 10,
    num_clusters: Optional[int] = None,
    snippets_per_topic: int = 3,
    topic_service=Depends(get_topic_service),
):
    """
    Extract topics with AI-powered semantic clustering and contextual analysis.

    This advanced topic extraction system provides:
    - **Semantic Clustering**: Groups related keywords using AI embeddings (e.g., "economy", "jobs" → "Economic Policy")
    - **Contextual Snippets**: Shows keywords in actual use with highlighting
    - **AI-Generated Summary**: Provides interpretive analysis of main themes via Gemini LLM
    - **Smart Filtering**: Excludes common verbs and low-relevance topics

    Returns structured topic clusters with labels, example snippets, and analytical summary.
    """
    if topic_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topic extraction not available. Service not initialized.",
        )

    try:
        result = topic_service.extract_topics_enhanced(
            text=input.text,
            top_n=top_n,
            num_clusters=num_clusters,
            snippets_per_topic=snippets_per_topic,
        )
        return EnhancedTopicResponse(**result)
    except Exception as e:
        logger.error(f"Topic extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@router.post("/ngrams")
async def extract_ngrams_endpoint(
    input: NGramRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
):
    """
    Extract n-grams from the input text.

    Returns the most common n-grams (bigrams, trigrams, etc.) found in the text.
    """
    try:
        return nlp_service.extract_ngrams(input.text, n=input.n, top_n=input.top_n)
    except Exception as e:
        logger.error(f"N-gram extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@router.post("/clean")
async def clean_text_endpoint(input: TextInput, remove_stopwords: bool = True):
    """
    Clean and normalize input text.

    Removes URLs, special characters, extra whitespace, and optionally stopwords.
    """
    try:
        cleaned = clean_text(input.text, remove_stopwords=remove_stopwords)
        return {
            "original_length": len(input.text),
            "cleaned_length": len(cleaned),
            "cleaned_text": cleaned,
        }
    except Exception as e:
        logger.error(f"Text cleaning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cleaning failed: {str(e)}"
        )


@router.get("/speeches/stats", response_model=StatsResponse)
async def get_speech_statistics(nlp_service: NLPService = Depends(get_nlp_service)):
    """
    Get statistics about the demo dataset (political rally speeches, 2019-2020).

    This endpoint demonstrates the API's analytical capabilities using a real-world
    dataset of 35+ political speeches. The dataset serves as a showcase example.

    Returns total speeches, word counts, date range, and locations.
    """
    try:
        stats = nlp_service.get_dataset_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load statistics: {str(e)}",
        )


@router.get("/speeches/list")
async def list_speeches(nlp_service: NLPService = Depends(get_nlp_service)):
    """
    List all speeches indexed in the RAG knowledge base.

    Returns all speeches in the demo dataset (35+ Trump rally speeches, 2019-2020)
    with metadata including filename, location, date, and word count.

    These speeches form the knowledge base for the RAG AI Q&A system.
    """
    try:
        return nlp_service.list_speeches()
    except Exception as e:
        logger.error(f"Speech listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load speeches: {str(e)}",
        )
