"""
LLM Provider Services.

This module contains all LLM (Large Language Model) provider implementations
and related utilities for the NLP Chatbot API.

Providers:
- GeminiLLM: Google Gemini integration
- OpenAILLM: OpenAI GPT integration (optional)
- AnthropicLLM: Anthropic Claude integration (optional)
"""

from .base import LLMProvider
from .factory import create_llm_provider
from .gemini import GeminiLLM

# Optional providers (may not be installed)
try:
    from .openai import OpenAILLM
except ImportError:
    OpenAILLM = None  # type: ignore[assignment, misc]

try:
    from .anthropic import AnthropicLLM
except ImportError:
    AnthropicLLM = None  # type: ignore[assignment, misc]

__all__ = [
    "LLMProvider",
    "create_llm_provider",
    "GeminiLLM",
    "OpenAILLM",
    "AnthropicLLM",
]
