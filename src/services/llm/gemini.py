"""
Google Gemini LLM provider implementation.

Provides integration with Google Gemini for high-quality answer synthesis
from retrieved context chunks.
"""

import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from .base import LLMProvider

logger = logging.getLogger(__name__)

# Gemini Safety Settings - Disabled for political speech analysis
# Allows analysis of controversial political content without blocking
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


class GeminiLLM(LLMProvider):
    """
    Google Gemini LLM provider.

    Uses Gemini to synthesize high-quality answers from retrieved context,
    with proper citations and confidence assessment.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        """
        Initialize Gemini LLM service.

        Args:
            api_key: Google API key (required)
            model_name: Gemini model to use (flash for speed, pro for quality)
            temperature: Generation temperature (0.0-1.0, lower = more focused)
            max_output_tokens: Maximum tokens in generated responses
        """
        if not api_key:
            raise ValueError("Gemini API key is required")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        logger.debug(f"Initializing Gemini LLM with model: {model_name}")

        # Configure Gemini
        genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]

        # Use centralized safety settings from constants
        self.model = genai.GenerativeModel(  # type: ignore[arg-type]
            model_name, safety_settings=GEMINI_SAFETY_SETTINGS
        )

        # Generation config
        self.generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_output_tokens,
        )

        logger.info(f"Gemini LLM initialized: model={model_name}, temp={temperature}")

    def generate_content(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Generate content from a prompt (implements LLMProvider interface).

        Args:
            prompt: Text prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional Gemini-specific arguments

        Returns:
            Gemini response object with .text attribute
        """
        # Build generation config with overrides
        gen_config = genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            top_p=0.95,
            top_k=40,
        )

        # Pass safety settings directly to generate_content for reliability
        return self.model.generate_content(
            prompt, generation_config=gen_config, safety_settings=GEMINI_SAFETY_SETTINGS
        )

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000,
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer to a question using provided context.

        Args:
            question: User's question
            context_chunks: List of context dicts with 'text', 'source', 'chunk_index'
            max_context_length: Maximum characters to include in context
            entities: Optional list of detected entities for entity-focused prompting

        Returns:
            Dict with 'answer', 'reasoning', and 'sources_used'
        """
        if not context_chunks:
            return {
                "answer": "I don't have enough information to answer this question based on the available documents.",
                "reasoning": "No relevant context was found.",
                "sources_used": [],
            }

        # Prepare context with source attribution
        context_parts = []
        total_length = 0
        sources_used = set()

        for i, chunk in enumerate(context_chunks):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            chunk_idx = chunk.get("chunk_index", 0)

            # Check length limit
            if total_length + len(text) > max_context_length:
                break

            context_parts.append(f"[Source {i + 1}: {source}, Part {chunk_idx + 1}]\n{text}")
            sources_used.add(source)
            total_length += len(text)

        context_text = "\n\n".join(context_parts)

        # Use inherited _build_rag_prompt from base class
        prompt = self._build_rag_prompt(question, context_text, list(sources_used), entities)

        try:
            # Generate answer
            logger.debug(f"Sending prompt to Gemini (length: {len(prompt)} chars)")
            response = self.model.generate_content(prompt, generation_config=self.generation_config)

            # Check if response was blocked or empty
            if not response or not hasattr(response, "text"):
                raise ValueError("Gemini response was empty or blocked by safety filters")

            answer_text = response.text.strip()

            if not answer_text:
                raise ValueError("Gemini returned empty answer")

            logger.info(f"Gemini generated answer successfully (length: {len(answer_text)} chars)")

            return {
                "answer": answer_text,
                "reasoning": "Generated using Gemini based on retrieved context",
                "sources_used": list(sources_used),
            }

        except Exception as e:
            # Fallback to extraction-based answer on error
            logger.error(f"Gemini generation failed: {str(e)}", exc_info=True)
            fallback_answer = self._extraction_fallback(question, context_chunks)
            return {
                "answer": fallback_answer,
                "reasoning": f"Gemini error (fallback to extraction): {str(e)}",
                "sources_used": list(sources_used),
            }

    def test_connection(self) -> bool:
        """
        Test if Gemini API is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.debug("Testing Gemini API connection...")
            response = self.model.generate_content(
                "Say 'OK' to confirm you are working.", generation_config=self.generation_config
            )
            # Try to access the text - this will fail if blocked/no content
            result = response.text
            logger.info(f"Gemini API connection test successful: {result[:50]}")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.

        Returns:
            Dict with model details
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key),
            "provider": "Google Gemini",
        }
