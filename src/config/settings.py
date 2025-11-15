"""Application configuration using Pydantic v2 Settings.

This module centralizes configuration loading with the following precedence:

1. Code defaults (Field defaults in Settings models)
2. YAML config file (`configs/<environment>.yaml`)
3. .env file
4. Environment variables (highest precedence)

Sensitive values (API keys, secrets) should be in .env or environment variables.
General configuration should be in YAML files.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path("configs")
DEFAULT_ENVIRONMENT = "development"


class APISettings(BaseSettings):
    """API configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", 8000)), ge=1, le=65535)
    reload: bool = False
    cors_origins: list[str] = ["*"]


class RAGSettings(BaseSettings):
    """RAG (Retrieval-Augmented Generation) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chromadb_persist_directory: str = "./data/chromadb"
    chromadb_collection_name: str = "speeches"
    chunk_size: int = Field(default=2048, ge=256, le=4096)
    chunk_overlap: int = Field(default=150, ge=0, le=512)
    default_top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    use_hybrid_search: bool = True


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sentiment_model_name: str = "ProsusAI/finbert"
    embedding_model_name: str = "all-mpnet-base-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"


class PathSettings(BaseSettings):
    """File system path configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PATH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root_directory: str = "./data"
    speeches_directory: str = "./data/Donald Trump Rally Speeches"


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["gemini", "openai", "anthropic", "none"] = "gemini"
    enabled: bool = True
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)


class Settings(BaseSettings):
    """Top-level application settings.

    Configuration is loaded with the following precedence (highest to lowest):
    1. Environment variables
    2. .env file
    3. YAML config file (configs/<environment>.yaml)
    4. Field defaults in code

    Use YAML for general configuration and .env for secrets and local overrides.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core metadata
    app_name: str = "Trump Speeches NLP Chatbot API"
    app_version: str = "0.1.0"
    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = "INFO"

    # Nested settings (loaded from YAML and env vars)
    api: APISettings = Field(default_factory=APISettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Additional configuration
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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return v_upper

    # ------------------------------------------------------------------
    # Convenience helpers (API stability for callers)
    # ------------------------------------------------------------------

    def get_llm_api_key(self) -> Optional[str]:
        """Get the configured LLM API key."""
        return self.llm.api_key

    def get_llm_model_name(self) -> str:
        """Get the configured LLM model name."""
        return self.llm.model_name

    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured with an API key."""
        return self.llm.enabled and self.llm.api_key is not None

    def get_speeches_path(self) -> Path:
        """Get the path to the speeches directory."""
        return Path(self.paths.speeches_directory)

    def get_chromadb_path(self) -> Path:
        """Get the path to the ChromaDB persistence directory."""
        return Path(self.rag.chromadb_persist_directory)

    def get_cors_origins(self) -> list[str]:
        """Get the list of allowed CORS origins."""
        return self.api.cors_origins

    def get_excluded_verbs(self) -> set[str]:
        """Get the set of excluded verbs for topic extraction."""
        default_verbs = (
            "want,think,know,make,get,go,see,come,take,give,say,tell,ask,use,find,work,call,"
            "try,feel,leave,put,mean,keep,let,begin,seem,help,talk,turn,start,show,hear,play,"
            "run,move,like,live,believe,bring,happen,write,sit,stand,lose,pay,meet,include,"
            "continue,learn,change,lead,understand,watch,follow,stop,create,speak,read,allow,"
            "add,spend,grow,open,walk,win,offer,remember,love,consider"
        )
        return {verb.strip().lower() for verb in default_verbs.split(",") if verb.strip()}

    def setup_logging(self) -> None:
        """Configure application logging based on settings."""
        from src.core.logging_config import configure_logging

        use_json = self.environment.lower() == "production"
        configure_logging(
            level=self.log_level,
            use_json=use_json,
            include_uvicorn=True,
        )

    def log_startup_info(self, logger: logging.Logger) -> None:
        """Log application startup information."""
        logger.info(f"Application: {self.app_name} v{self.app_version}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info(f"LLM Provider: {self.llm.provider}")
        logger.info(f"LLM Enabled: {self.llm.enabled}")

        if self.is_llm_configured():
            logger.info(f"LLM Model: {self.get_llm_model_name()}")
            logger.info("LLM API Key: ✓ Configured")
        else:
            logger.warning("LLM API Key: ✗ Not configured - using extraction-based answers")

        logger.info(f"Sentiment Model: {self.models.sentiment_model_name}")
        logger.info(f"Embedding Model: {self.models.embedding_model_name}")
        logger.info(f"ChromaDB Path: {self.rag.chromadb_persist_directory}")
        logger.info(f"Speeches Path: {self.paths.speeches_directory}")
        logger.info(f"Hybrid Search: {'Enabled' if self.rag.use_hybrid_search else 'Disabled'}")
        logger.info(f"Re-ranking: {'Enabled' if self.rag.use_reranking else 'Disabled'}")


# ------------------------------------------------------------------
# YAML loading helpers
# ------------------------------------------------------------------


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load YAML configuration file and return nested dict structure.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Dictionary with nested configuration structure

    Raises:
        ValueError: If YAML file doesn't contain a mapping at top level
    """
    if not yaml_path.is_file():
        return {}

    with yaml_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Configuration file {yaml_path} must contain a YAML mapping at the top level."
        )

    return loaded


# ------------------------------------------------------------------
# Singleton pattern with lru_cache
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached Settings instance (singleton pattern).

    Precedence order (lowest to highest):
    1. Field defaults (code)
    2. YAML config for selected environment
    3. .env file
    4. Real environment variables
    """

    # 1. Start from code defaults by creating a bare Settings to read environment name
    base = Settings()

    # 2. Load YAML for the selected environment (may override code defaults)
    yaml_path = CONFIG_DIR / f"{base.environment}.yaml"
    yaml_data = load_yaml(yaml_path)

    # 3/4. Let pydantic handle .env and real env vars overriding YAML + defaults
    # We pass YAML as **kwargs so it sits between defaults and env-based sources
    return Settings(**yaml_data)


def reload_settings() -> Settings:
    """Force reload settings from YAML and environment.

    Clears the cache and creates a new Settings instance.
    Useful for testing or dynamic configuration changes.

    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()
