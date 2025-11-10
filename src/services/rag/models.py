"""
Domain models for RAG (Retrieval-Augmented Generation) service.

Provides Pydantic models for type-safe data structures used internally
in the RAG pipeline, separate from API request/response models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Single search result from RAG system."""

    document: str
    metadata: Dict[str, Any]
    distance: float
    id: str
    rerank_score: Optional[float] = None
    combined_score: Optional[float] = None
    sources: List[str] = Field(default_factory=list)

    @property
    def effective_score(self) -> float:
        """
        Get the most relevant score available.

        Priority: rerank_score > combined_score > (1.0 - distance)
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.combined_score is not None:
            return self.combined_score
        return 1.0 - self.distance

    @property
    def source_name(self) -> str:
        """Get source document name from metadata."""
        return self.metadata.get("source", "Unknown")

    @property
    def chunk_index(self) -> int:
        """Get chunk index from metadata."""
        return self.metadata.get("chunk_index", 0)


class ConfidenceFactors(BaseModel):
    """Factors contributing to confidence calculation."""

    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    chunk_coverage: float = Field(..., ge=0.0, le=1.0)  # Changed from int to normalized float
    entity_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)


class ConfidenceResult(BaseModel):
    """Confidence assessment result."""

    score: float = Field(..., ge=0.0, le=1.0)
    level: str = Field(..., pattern="^(high|medium|low)$")
    explanation: str
    factors: ConfidenceFactors


class EntitySentiment(BaseModel):
    """Sentiment analysis for an entity."""

    average_score: float = Field(..., ge=-1.0, le=1.0)
    classification: str = Field(..., pattern="^(Positive|Negative|Neutral|Unknown)$")
    sample_size: int = Field(..., ge=0)


class EntityStatistics(BaseModel):
    """Statistics about an entity's mentions across the corpus."""

    mention_count: int = Field(..., ge=0)
    speech_count: int = Field(..., ge=0)
    corpus_percentage: float = Field(..., ge=0.0, le=100.0)
    speeches: List[str]
    sentiment: Optional[EntitySentiment] = None
    associated_terms: Optional[List[str]] = None


class ContextChunk(BaseModel):
    """A chunk of context text with metadata."""

    text: str
    source: str
    chunk_index: int
    score: Optional[float] = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "ContextChunk":
        """Create ContextChunk from SearchResult."""
        return cls(
            text=result.document,
            source=result.source_name,
            chunk_index=result.chunk_index,
            score=result.effective_score,
        )
