"""
Factory for creating LLM provider instances.

Provides a centralized way to instantiate the correct LLM provider
based on configuration settings.
"""

import logging
from typing import Optional

from ...config.settings import Settings
from .base import LLMProvider
from .gemini import GeminiLLM

# Import optional providers (may not be installed)
try:
    from .openai import OpenAILLM

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from .anthropic import AnthropicLLM

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_llm_provider(settings: Settings) -> Optional[LLMProvider]:
    """
    Create an LLM provider instance based on configuration.

    This factory method instantiates the appropriate LLM provider
    (Gemini, OpenAI, Anthropic) based on the settings.

    Args:
        settings: Application settings object

    Returns:
        LLM provider instance, or None if LLM is disabled or not configured

    Raises:
        ValueError: If provider is specified but required configuration is missing
        ImportError: If required library for provider is not installed
    """
    if not settings.llm.enabled:
        logger.info("LLM disabled via configuration")
        return None

    provider = settings.llm.provider.lower()

    # Get API key and model name using model-agnostic config
    api_key = settings.get_llm_api_key()
    model_name = settings.get_llm_model_name()
    temperature = settings.llm.temperature
    max_tokens = settings.llm.max_output_tokens

    if not api_key:
        logger.warning(
            f"LLM provider '{provider}' selected but no API key configured. "
            f"Set LLM_API_KEY in .env file."
        )
        return None

    try:
        if provider == "gemini":
            logger.info(f"Creating Gemini LLM provider with model: {model_name}")
            return GeminiLLM(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI provider requires 'openai' package. Install with: pip install openai"
                )

            logger.info(f"Creating OpenAI LLM provider with model: {model_name}")
            return OpenAILLM(  # type: ignore[possibly-unbound]
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "Anthropic provider requires 'anthropic' package. Install with: pip install anthropic"
                )

            logger.info(f"Creating Anthropic LLM provider with model: {model_name}")
            return AnthropicLLM(  # type: ignore[possibly-unbound]
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        elif provider == "none":
            logger.info("LLM provider set to 'none' - RAG will use extraction-based answers")
            return None

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported providers: gemini, openai, anthropic, none"
            )

    except Exception as e:
        logger.error(f"Failed to create LLM provider '{provider}': {e}", exc_info=True)
        raise
