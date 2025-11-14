"""
Configuration management for the NLP Chatbot API.

Uses Pydantic Settings for type-safe configuration with environment variable
support and .env file loading. This is the industry-standard approach for
modern Python applications, especially those deployed to cloud platforms.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables or a .env file.
    The .env file takes precedence, making it ideal for Azure deployments.

    Example .env file:
        GEMINI_API_KEY=your_api_key_here
        LLM_PROVIDER=gemini
        LLM_MODEL_NAME=gemini-2.5-flash
        LOG_LEVEL=INFO
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )

    # ============================================================================
    # Application Settings
    # ============================================================================

    app_name: str = Field(
        default="Trump Speeches NLP Chatbot API",
        description="Application name displayed in logs and API docs",
    )

    app_version: str = Field(default="0.1.0", description="Application version")

    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # ============================================================================
    # LLM Provider Configuration
    # ============================================================================

    llm_provider: Literal["gemini", "openai", "anthropic", "none"] = Field(
        default="gemini",
        description="LLM provider to use for answer generation (set 'none' to disable)",
    )

    llm_enabled: bool = Field(
        default=True,
        description="Enable/disable LLM-powered answer generation",
    )

    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider (works with any provider)",
    )

    llm_model_name: str = Field(
        default="gemini-2.5-flash",
        description="Model name for LLM provider (e.g., gemini-2.5-flash, gpt-4o-mini, claude-3-5-sonnet-20241022)",
    )

    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Generation temperature (0.0-1.0, lower = more focused)",
    )

    llm_max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum tokens in generated responses",
    )

    # ============================================================================
    # ML Models Configuration
    # ============================================================================

    # Sentiment Analysis Model
    sentiment_model_name: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model for sentiment analysis",
    )

    # Embedding Model for RAG
    embedding_model_name: str = Field(
        default="all-mpnet-base-v2",
        description="SentenceTransformer model for embeddings",
    )

    # Re-ranking Model
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking search results",
    )

    # ============================================================================
    # RAG Configuration
    # ============================================================================

    # ChromaDB Settings
    chromadb_persist_directory: str = Field(
        default="./data/chromadb",
        description="Directory for ChromaDB persistence",
    )

    chromadb_collection_name: str = Field(
        default="speeches",
        description="ChromaDB collection name",
    )

    # Text Chunking
    chunk_size: int = Field(
        default=2048,
        ge=256,
        le=4096,
        description="Maximum size of text chunks in characters",
    )

    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=512,
        description="Overlap between chunks in characters",
    )

    # Search Settings
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of chunks to retrieve",
    )

    use_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder re-ranking for better results",
    )

    use_hybrid_search: bool = Field(
        default=True,
        description="Combine semantic and keyword (BM25) search",
    )

    # ============================================================================
    # NLP Analysis Configuration
    # ============================================================================

    # Topic Extraction Settings
    topic_relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for topic clusters (0.0-1.0)",
    )

    topic_min_clusters: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum number of topic clusters to keep",
    )

    topic_excluded_verbs: str = Field(
        default="want,think,know,make,get,go,see,come,take,give,say,tell,ask,use,find,work,call,try,feel,leave,put,mean,keep,let,begin,seem,help,talk,turn,start,show,hear,play,run,move,like,live,believe,bring,happen,write,sit,stand,lose,pay,meet,include,continue,learn,change,lead,understand,watch,follow,stop,create,speak,read,allow,add,spend,grow,open,walk,win,offer,remember,love,consider",
        description="Comma-separated list of common verbs to exclude from topics",
    )

    # Sentiment Analysis Settings
    sentiment_model_name: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model for sentiment classification",
    )

    emotion_model_name: str = Field(
        default="j-hartmann/emotion-english-distilroberta-base",
        description="HuggingFace model for emotion detection",
    )

    sentiment_interpretation_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="LLM temperature for sentiment interpretation (0.0-1.0)",
    )

    sentiment_interpretation_max_tokens: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Max tokens for sentiment interpretation",
    )

    # ============================================================================
    # Data Directories
    # ============================================================================

    data_root_directory: str = Field(
        default="./data",
        description="Root directory for all data files",
    )

    speeches_directory: str = Field(
        default="./data/Donald Trump Rally Speeches",
        description="Directory containing text documents to index",
    )

    # ============================================================================
    # API Configuration
    # ============================================================================

    api_host: str = Field(default="0.0.0.0", description="API host address")

    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")

    api_reload: bool = Field(
        default=False,
        description="Enable auto-reload in development (disable in production)",
    )

    cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated, or * for all)",
    )

    # ============================================================================
    # Validators
    # ============================================================================

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v_upper

    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API keys (don't log them)."""
        if v and len(v) < 10:
            raise ValueError("API key appears to be too short")
        return v

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def get_llm_api_key(self) -> Optional[str]:
        """
        Get the API key for the selected LLM provider.

        Returns:
            API key string, or None if not configured
        """
        return self.llm_api_key

    def get_llm_model_name(self) -> str:
        """
        Get the model name for the selected LLM provider.

        Returns:
            Model name string
        """
        return self.llm_model_name

    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured."""
        return self.llm_enabled and self.get_llm_api_key() is not None

    def get_speeches_path(self) -> Path:
        """Get Path object for speeches directory."""
        return Path(self.speeches_directory)

    def get_chromadb_path(self) -> Path:
        """Get Path object for ChromaDB directory."""
        return Path(self.chromadb_persist_directory)

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins string into list.

        Supports:
        - "*" -> ["*"]
        - "https://example.com" -> ["https://example.com"]
        - "https://example.com,https://other.com" -> ["https://example.com", "https://other.com"]
        """
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_excluded_verbs(self) -> set[str]:
        """Parse excluded verbs string into a set."""
        return {
            verb.strip().lower() for verb in self.topic_excluded_verbs.split(",") if verb.strip()
        }

    def setup_logging(self) -> None:
        """Configure application-wide logging based on settings."""
        from .logging_config import configure_logging

        # Use JSON logging in production, colored in development
        use_json = self.environment == "production"

        configure_logging(
            level=self.log_level,
            use_json=use_json,
            include_uvicorn=True,
        )

    def log_startup_info(self, logger: logging.Logger) -> None:
        """Log configuration info at startup."""
        logger.info(f"Application: {self.app_name} v{self.app_version}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info(f"LLM Provider: {self.llm_provider}")
        logger.info(f"LLM Enabled: {self.llm_enabled}")

        if self.is_llm_configured():
            logger.info(f"LLM Model: {self.get_llm_model_name()}")
            logger.info("LLM API Key: ✓ Configured")
        else:
            logger.warning("LLM API Key: ✗ Not configured - using extraction-based answers")

        logger.info(f"Sentiment Model: {self.sentiment_model_name}")
        logger.info(f"Embedding Model: {self.embedding_model_name}")
        logger.info(f"ChromaDB Path: {self.chromadb_persist_directory}")
        logger.info(f"Speeches Path: {self.speeches_directory}")
        logger.info(f"Hybrid Search: {'Enabled' if self.use_hybrid_search else 'Disabled'}")
        logger.info(f"Re-ranking: {'Enabled' if self.use_reranking else 'Disabled'}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global settings instance.

    This ensures we only load settings once and reuse the same instance
    throughout the application lifecycle.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment/file.

    Useful for testing or dynamic configuration changes.
    """
    global _settings
    _settings = Settings()
    return _settings
