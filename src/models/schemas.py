"""
Pydantic schemas for API request and response models.

Defines the data structures for API endpoints using Pydantic for
automatic validation, serialization, and OpenAPI documentation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ============================================================================
# Request Models
# ============================================================================


class TextInput(BaseModel):
    """Input model for text analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze")

    class Config:
        json_schema_extra = {
            "example": {"text": "We're going to make America great again. Our economy is booming!"}
        }


class NGramRequest(BaseModel):
    """Request model for n-gram extraction."""

    text: str = Field(..., min_length=1)
    n: int = Field(2, ge=2, le=5, description="N-gram size (2-5)")
    top_n: int = Field(20, ge=1, le=100, description="Number of top n-grams to return")


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="Question to ask about the documents")
    top_k: int = Field(5, ge=1, le=15, description="Number of context chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main economic policies discussed?",
                "top_k": 5,
            }
        }


class RAGSearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


# ============================================================================
# Response Models
# ============================================================================


class SentimentResponse(BaseModel):
    """Response model for AI-powered sentiment analysis with emotion detection."""

    sentiment: str = Field(..., description="Dominant sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scores: Dict[str, float] = Field(..., description="All sentiment scores")
    emotions: Dict[str, float] = Field(..., description="Emotion classification scores")
    contextual_sentiment: str = Field(
        ..., description="AI-generated contextual interpretation of emotional tone"
    )
    num_chunks: int = Field(..., description="Number of text chunks analyzed")


class WordFrequencyResponse(BaseModel):
    """Response model for word frequency analysis."""

    total_tokens: int
    unique_tokens: int
    top_words: List[Dict[str, Any]]


class TopicResponse(BaseModel):
    """Response model for topic extraction."""

    topics: List[Dict[str, Any]]


class EnhancedTopicResponse(BaseModel):
    """Response model for enhanced topic extraction with clustering and snippets."""

    clustered_topics: List[Dict[str, Any]] = Field(
        ..., description="Semantically clustered topic groups with labels"
    )
    snippets: List[Dict[str, Any]] = Field(
        ..., description="Contextual text snippets for each topic cluster"
    )
    summary: Optional[str] = Field(None, description="AI-generated interpretation of main themes")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")


class StatsResponse(BaseModel):
    """Response model for dataset statistics."""

    total_speeches: int
    total_words: int
    avg_words_per_speech: float
    date_range: Dict[str, str]
    years: List[str]
    locations: List[str]


class RAGAnswerResponse(BaseModel):
    """Response model for RAG answers."""

    answer: str = Field(..., description="Generated answer")
    context: List[Dict[str, Any]] = Field(..., description="Context chunks used")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    confidence_score: float = Field(..., description="Numeric confidence score (0-1)")
    confidence_explanation: str = Field(..., description="Human-readable explanation of confidence")
    confidence_factors: Dict[str, Any] = Field(..., description="Breakdown of confidence factors")
    sources: List[str] = Field(..., description="Source documents")
    entity_statistics: Optional[Dict[str, Any]] = Field(
        None, description="Enhanced statistics about entities: mentions, sentiment, associations"
    )


class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""

    collection_name: str
    total_chunks: int
    unique_sources: int
    sources: List[str]
    embedding_model: int
    chunk_size: int
    chunk_overlap: int
