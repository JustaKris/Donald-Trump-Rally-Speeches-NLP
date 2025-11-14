"""
Abstract base class for LLM providers.

Defines the interface that all LLM providers (Gemini, OpenAI, Anthropic, etc.)
must implement for consistent integration across the application.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for Large Language Model providers.

    All LLM implementations must inherit from this class and implement
    the required methods. This ensures consistent behavior across different
    providers (Gemini, OpenAI, Anthropic, etc.).
    """

    @abstractmethod
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider
            model_name: Model identifier (e.g., "gemini-2.5-flash", "gpt-4o-mini")
            temperature: Generation temperature (0.0-1.0, lower = more focused)
            max_output_tokens: Maximum tokens in generated responses
        """
        pass

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000,
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer to a question using provided context.

        This is the primary method for RAG (Retrieval-Augmented Generation).

        Args:
            question: User's question
            context_chunks: List of context dicts with 'text', 'source', 'chunk_index'
            max_context_length: Maximum characters to include in context
            entities: Optional list of detected entities for entity-focused prompting

        Returns:
            Dict with 'answer', 'reasoning', and 'sources_used'
        """
        pass

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Generate content from a prompt (general-purpose generation).

        This method is used for tasks like sentiment interpretation,
        topic labeling, and other non-RAG text generation.

        Args:
            prompt: Text prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Provider-specific arguments

        Returns:
            Provider-specific response object (must have .text attribute)
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the LLM API is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.

        Returns:
            Dict with model details (name, provider, temperature, etc.)
        """
        pass

    def _build_rag_prompt(
        self,
        question: str,
        context: str,
        sources: List[str],
        entities: Optional[List[str]] = None,
    ) -> str:
        """
        Build a standardized prompt for RAG question answering.

        This method provides a consistent prompt structure across all providers.
        Can be overridden by subclasses if needed.

        Args:
            question: User's question
            context: Retrieved context with source attribution
            sources: List of source document names
            entities: Optional list of entities to focus on

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert research assistant analyzing political speech documents.

CONTEXT from {len(sources)} document(s): {', '.join(sources)}

{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a direct, concise answer (2-4 sentences maximum)
2. Base your answer ONLY on the context provided above
3. If the context doesn't contain the information, clearly state: "The available documents don't contain information about this topic"
4. Cite sources naturally (e.g., "In the rally speech from [location/date]...")
5. Don't repeat the same information multiple times
6. Focus on answering the specific question asked"""

        if entities:
            entity_instruction = f"""
7. IMPORTANT: The question is about {', '.join(entities)}. Focus specifically on direct mentions, quotes, and references to these entities. Prioritize exact quotes and specific statements."""
            prompt += entity_instruction

        prompt += "\n\nYour answer:"
        return prompt

    def _extraction_fallback(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Fallback to extraction-based answer if LLM fails.

        Args:
            question: User's question
            context_chunks: Context chunks

        Returns:
            Simple extracted answer
        """
        if not context_chunks:
            return "Unable to generate answer due to technical issues."

        first_chunk = context_chunks[0]
        text = first_chunk.get("text", "")
        source = first_chunk.get("source", "unknown")

        if len(text) > 300:
            text = text[:300] + "..."

        return f"Based on {source}: {text}"
