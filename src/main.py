"""
Trump Speeches NLP Chatbot API - Production-Ready AI Platform

Comprehensive NLP and RAG platform for analyzing Trump rally speeches (2019-2020).
Features AI-powered Q&A with Gemini, sentiment analysis with FinBERT, semantic search,
and advanced text analytics. Built with FastAPI, ChromaDB, and LangChain.

Run with: uvicorn src.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import chatbot_router, health_router, nlp_router
from .api.dependencies import set_llm_service, set_nlp_service, set_rag_service, set_sentiment_analyzer
from .core import get_settings
from .services import GeminiLLM, NLPService, RAGService, SentimentAnalyzer

# Get configuration
settings = get_settings()

# Configure logging based on settings
settings.setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles model loading on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("=" * 70)
    settings.log_startup_info(logger)
    logger.info("=" * 70)

    # Load sentiment analysis model
    logger.info(f"Loading sentiment analysis model: {settings.sentiment_model_name}")
    try:
        sentiment_analyzer = SentimentAnalyzer(model_name=settings.sentiment_model_name)
        set_sentiment_analyzer(sentiment_analyzer)
        logger.info("✓ Sentiment analysis model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load sentiment model: {e}")
        # Continue without model - endpoints will return errors

    # Initialize NLP service
    logger.info("Initializing NLP service...")
    try:
        nlp_service = NLPService()
        set_nlp_service(nlp_service)
        logger.info("✓ NLP service initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize NLP service: {e}")

    # Initialize LLM service if configured
    llm_service = None
    if settings.is_llm_configured():
        try:
            logger.info(f"Initializing {settings.llm_provider.upper()} LLM service...")
            llm_service = GeminiLLM(
                api_key=settings.get_llm_api_key(),  # type: ignore[arg-type]
                model_name=settings.get_llm_model_name(),
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
            )

            # Test connection
            if llm_service.test_connection():
                logger.info("✓ LLM service initialized and tested successfully")
                set_llm_service(llm_service)
            else:
                logger.warning("⚠️  LLM connection test failed")
                llm_service = None
        except Exception as e:
            logger.error(f"✗ Failed to initialize LLM: {e}")
            llm_service = None
    else:
        logger.warning("⚠️  LLM not configured - RAG will use extraction-based answers")
        logger.warning("   Set GEMINI_API_KEY in .env file for AI-powered answers")

    # Initialize RAG service
    logger.info("Initializing RAG service...")
    try:
        rag_service = RAGService(
            collection_name=settings.chromadb_collection_name,
            persist_directory=settings.chromadb_persist_directory,
            embedding_model=settings.embedding_model_name,
            reranker_model=settings.reranker_model_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            llm_service=llm_service,
            use_reranking=settings.use_reranking,
            use_hybrid_search=settings.use_hybrid_search,
        )

        # Check if collection is empty and load documents if needed
        if rag_service.collection.count() == 0:
            logger.info("Loading documents into RAG service...")
            docs_loaded = rag_service.load_documents(settings.speeches_directory)
            logger.info(f"✓ Loaded {docs_loaded} documents into RAG service")
        else:
            chunk_count = rag_service.collection.count()
            logger.info(f"✓ RAG service initialized with {chunk_count} existing chunks")

        set_rag_service(rag_service)
    except Exception as e:
        logger.error(f"✗ Failed to initialize RAG service: {e}", exc_info=True)
        # Continue without RAG - endpoints will return errors

    logger.info("=" * 70)
    logger.info("Application startup complete")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.app_name,
    description="Production-ready NLP and RAG API for sentiment analysis, topic modeling, and conversational Q&A over Trump rally speeches. Built with FastAPI, ChromaDB, and Gemini.",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Include routers
app.include_router(health_router)
app.include_router(nlp_router)
app.include_router(chatbot_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
