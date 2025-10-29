"""
FastAPI Application for General-Purpose NLP Text Analysis.

This API provides endpoints for sentiment analysis, word frequency analysis,
topic extraction, and n-gram analysis using state-of-the-art NLP models.
Includes a demo dataset of political speeches for showcase purposes.

Run with: uvicorn src.api:app --reload
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import SentimentAnalyzer
from .preprocessing import clean_text, extract_ngrams, tokenize_text
from .rag_service import RAGService
from .utils import (
    extract_topics,
    get_dataset_statistics,
    get_word_frequency_stats,
    load_speeches_from_directory,
)

# Configure logging with clean format
logging.basicConfig(
    level=logging.INFO,
    format="INFO:     %(message)s",  # Match Uvicorn's format
)
logger = logging.getLogger(__name__)


# Custom filter to suppress ChromaDB telemetry warnings (harmless bugs in ChromaDB)
class SuppressChromaDBTelemetryFilter(logging.Filter):
    def filter(self, record):
        return "Failed to send telemetry event" not in record.getMessage()


# Apply filter to uvicorn logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.addFilter(SuppressChromaDBTelemetryFilter())


# Initialize FastAPI app
app = FastAPI(
    title="NLP Text Analysis API",
    description="Production-ready API for natural language processing: sentiment analysis, word frequency, topic extraction, and n-gram analysis. Powered by transformer models and statistical NLP methods.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Global model instance (loaded on startup)
sentiment_analyzer: Optional[SentimentAnalyzer] = None
rag_service: Optional[RAGService] = None


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    global sentiment_analyzer, rag_service
    logger.info("Loading sentiment analysis model...")
    try:
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("Sentiment analysis model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue without model - endpoints will return errors

    logger.info("Initializing RAG service...")
    try:
        rag_service = RAGService()
        # Check if collection is empty and load documents if needed
        if rag_service.collection.count() == 0:
            logger.info("Loading documents into RAG service...")
            docs_loaded = rag_service.load_documents()
            logger.info(f"Loaded {docs_loaded} documents into RAG service!")
        else:
            logger.info(f"RAG service initialized with {rag_service.collection.count()} chunks")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Continue without RAG - endpoints will return errors


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


# ============================================================================
# Request/Response Models
# ============================================================================


class TextInput(BaseModel):
    """Input model for text analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze")

    class Config:
        json_schema_extra = {
            "example": {"text": "We're going to make America great again. Our economy is booming!"}
        }


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    sentiment: str = Field(..., description="Dominant sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scores: Optional[Dict[str, float]] = Field(None, description="All sentiment scores")
    num_chunks: int = Field(..., description="Number of text chunks analyzed")


class WordFrequencyResponse(BaseModel):
    """Response model for word frequency analysis."""

    total_tokens: int
    unique_tokens: int
    top_words: List[Dict[str, Any]]


class TopicResponse(BaseModel):
    """Response model for topic extraction."""

    topics: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    """Response model for dataset statistics."""

    total_speeches: int
    total_words: int
    avg_words_per_speech: float
    date_range: Dict[str, str]
    years: List[str]
    locations: List[str]


class NGramRequest(BaseModel):
    """Request model for n-gram extraction."""

    text: str = Field(..., min_length=1)
    n: int = Field(2, ge=2, le=5, description="N-gram size (2-5)")
    top_n: int = Field(20, ge=1, le=100, description="Number of top n-grams to return")


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="Question to ask about the documents")
    top_k: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main economic policies discussed?",
                "top_k": 3,
            }
        }


class RAGSearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class RAGAnswerResponse(BaseModel):
    """Response model for RAG answers."""

    answer: str = Field(..., description="Generated answer")
    context: List[Dict[str, Any]] = Field(..., description="Context chunks used")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    sources: List[str] = Field(..., description="Source documents")


class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""

    collection_name: str
    total_chunks: int
    unique_sources: int
    sources: List[str]
    embedding_model: int
    chunk_size: int
    chunk_overlap: int


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page."""
    html_file = Path(__file__).parent.parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return """
    <html>
        <head><title>NLP Text Analysis API</title></head>
        <body>
            <h1>📊 NLP Text Analysis API</h1>
            <p>Professional natural language processing API for text analysis</p>
            <ul>
                <li><a href="/docs">📚 Interactive API Documentation</a></li>
                <li><a href="/redoc">📖 ReDoc Documentation</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": sentiment_analyzer is not None, "version": "0.1.0"}


@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input: TextInput):
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


@app.post("/analyze/words", response_model=WordFrequencyResponse)
async def analyze_word_frequency(input: TextInput, top_n: int = 50):
    """
    Analyze word frequency in the input text.

    Returns the most common words with their frequencies, excluding stopwords.
    """
    try:
        stats = get_word_frequency_stats(input.text, top_n=top_n)
        return WordFrequencyResponse(**stats)
    except Exception as e:
        logger.error(f"Word frequency error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/topics", response_model=TopicResponse)
async def analyze_topics(input: TextInput, top_n: int = 10):
    """
    Extract key topics/themes from the input text.

    Returns topics ranked by relevance with mention counts.
    """
    try:
        topics = extract_topics(input.text, top_n=top_n)
        return TopicResponse(topics=topics)
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/ngrams")
async def extract_ngrams_endpoint(input: NGramRequest):
    """
    Extract n-grams from the input text.

    Returns the most common n-grams (bigrams, trigrams, etc.) found in the text.
    """
    try:
        tokens = tokenize_text(input.text)
        # Remove stopwords for better n-grams
        from .preprocessing import get_stopwords

        stop_words = get_stopwords()
        tokens = [t for t in tokens if t not in stop_words and t.isalpha()]

        ngrams = extract_ngrams(tokens, n=input.n)

        # Count and rank n-grams
        from collections import Counter

        ngram_counts = Counter(ngrams)
        top_ngrams = ngram_counts.most_common(input.top_n)

        return {
            "n": input.n,
            "total_ngrams": len(ngrams),
            "unique_ngrams": len(set(ngrams)),
            "top_ngrams": [{"ngram": ngram, "count": count} for ngram, count in top_ngrams],
        }
    except Exception as e:
        logger.error(f"N-gram extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {str(e)}"
        )


@app.get("/speeches/stats", response_model=StatsResponse)
async def get_speech_statistics():
    """
    Get statistics about the demo dataset (political rally speeches, 2019-2020).

    This endpoint demonstrates the API's analytical capabilities using a real-world
    dataset of 35+ political speeches. The dataset serves as a showcase example.

    Returns total speeches, word counts, date range, and locations.
    """
    try:
        stats = get_dataset_statistics()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load statistics: {str(e)}",
        )


@app.get("/speeches/list")
async def list_speeches():
    """
    List all speeches in the demo dataset with metadata.

    The demo dataset contains 35+ political rally speeches from 2019-2020,
    showcasing the API's ability to process and analyze real-world text data.

    Returns a list of all speeches with location, date, and word count.
    """
    try:
        df = load_speeches_from_directory()
        speeches = df[["filename", "location", "month", "year", "word_count"]].to_dict("records")
        return {"total": len(speeches), "speeches": speeches}
    except Exception as e:
        logger.error(f"Speech listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load speeches: {str(e)}",
        )


@app.post("/text/clean")
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


# ============================================================================
# RAG (Retrieval-Augmented Generation) Endpoints
# ============================================================================


@app.post("/rag/ask", response_model=RAGAnswerResponse)
async def rag_ask_question(query: RAGQueryRequest):
    """
    Ask a question about the indexed documents using RAG.

    Uses semantic search to retrieve relevant context and generates an answer
    based on the retrieved information. Works with any text corpus that has been indexed.

    Returns an answer with supporting context, confidence level, and source documents.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        result = rag_service.ask(query.question, top_k=query.top_k)
        return RAGAnswerResponse(**result)
    except Exception as e:
        logger.error(f"RAG question error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}",
        )


@app.post("/rag/search")
async def rag_semantic_search(query: RAGSearchRequest):
    """
    Perform semantic search over indexed documents.

    Searches for text chunks similar to the query using semantic embeddings.
    Returns the most relevant chunks with metadata and similarity scores.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        results = rag_service.search(query.query, top_k=query.top_k)
        return {"query": query.query, "results": results}
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/rag/stats", response_model=RAGStatsResponse)
async def get_rag_statistics():
    """
    Get statistics about the RAG collection.

    Returns information about indexed documents, chunk counts, sources,
    and embedding configuration.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        stats = rag_service.get_stats()
        return RAGStatsResponse(**stats)
    except Exception as e:
        logger.error(f"RAG stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


@app.post("/rag/index")
async def index_documents(data_dir: str = "data/Donald Trump Rally Speeches"):
    """
    Index or re-index documents from a directory.

    Loads text files, chunks them, generates embeddings, and stores them
    in the vector database for semantic search and retrieval.

    Use this to update the RAG knowledge base with new documents.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized. Please try again later.",
        )

    try:
        # Clear existing collection
        rag_service.clear_collection()
        # Load new documents
        docs_loaded = rag_service.load_documents(data_dir)
        stats = rag_service.get_stats()

        return {
            "status": "success",
            "documents_loaded": docs_loaded,
            "total_chunks": stats["total_chunks"],
            "sources": stats["sources"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"RAG indexing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
