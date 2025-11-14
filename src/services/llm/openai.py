"""
OpenAI LLM provider implementation.

Provides integration with OpenAI's GPT models for answer generation.
"""

import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI  # type: ignore[import-untyped]

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    """
    OpenAI LLM provider.

    Uses OpenAI's GPT models (GPT-4, GPT-3.5, etc.) for text generation
    and question answering.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        """
        Initialize OpenAI LLM service.

        Args:
            api_key: OpenAI API key (required)
            model_name: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
            temperature: Generation temperature (0.0-1.0, lower = more focused)
            max_output_tokens: Maximum tokens in generated responses
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        logger.debug(f"Initializing OpenAI LLM with model: {model_name}")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"OpenAI LLM initialized: model={model_name}, temp={temperature}")

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
            **kwargs: Additional OpenAI-specific arguments

        Returns:
            Response object with .text attribute for compatibility
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            top_p=0.95,
            **kwargs,
        )

        # Create a response object compatible with the interface
        class OpenAIResponse:
            def __init__(self, content: str):
                self.text = content

        return OpenAIResponse(response.choices[0].message.content or "")

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

            if total_length + len(text) > max_context_length:
                break

            context_parts.append(f"[Source {i + 1}: {source}, Part {chunk_idx + 1}]\n{text}")
            sources_used.add(source)
            total_length += len(text)

        context_text = "\n\n".join(context_parts)

        # Use inherited _build_rag_prompt from base class
        prompt = self._build_rag_prompt(question, context_text, list(sources_used), entities)

        try:
            logger.debug(f"Sending prompt to OpenAI (length: {len(prompt)} chars)")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=0.95,
            )

            answer_text = response.choices[0].message.content

            if not answer_text:
                raise ValueError("OpenAI returned empty answer")

            logger.info(f"OpenAI generated answer successfully (length: {len(answer_text)} chars)")

            return {
                "answer": answer_text.strip(),
                "reasoning": "Generated using OpenAI based on retrieved context",
                "sources_used": list(sources_used),
            }

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}", exc_info=True)
            fallback_answer = self._extraction_fallback(question, context_chunks)
            return {
                "answer": fallback_answer,
                "reasoning": f"OpenAI error (fallback to extraction): {str(e)}",
                "sources_used": list(sources_used),
            }

    def test_connection(self) -> bool:
        """
        Test if OpenAI API is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.debug("Testing OpenAI API connection...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Say 'OK' to confirm you are working."}],
                temperature=self.temperature,
                max_tokens=10,
            )
            result = response.choices[0].message.content or ""
            logger.info(f"OpenAI API connection test successful: {result[:50]}")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
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
            "provider": "OpenAI",
        }
