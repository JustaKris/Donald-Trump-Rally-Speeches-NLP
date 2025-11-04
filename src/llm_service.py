"""
LLM Service for answer generation in RAG system.

Provides integration with Google Gemini for high-quality answer synthesis
from retrieved context chunks.
"""

import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiLLM:
    """
    Service for generating answers using Google Gemini.

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

        # Configure safety settings to allow political content
        # Type ignore needed as the library accepts dict but type hints don't reflect this
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        self.model = genai.GenerativeModel(model_name, safety_settings=safety_settings)  # type: ignore[arg-type]

        # Generation config
        self.generation_config = genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_output_tokens,
        )

        logger.info(f"Gemini LLM initialized: model={model_name}, temp={temperature}")

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

        # Build prompt with entity awareness
        prompt = self._build_prompt(question, context_text, list(sources_used), entities)

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

    def _build_prompt(
        self, question: str, context: str, sources: List[str], entities: Optional[List[str]] = None
    ) -> str:
        """
        Build the prompt for Gemini with optional entity-focused instructions.

        Args:
            question: User's question
            context: Retrieved context with source attribution
            sources: List of source document names
            entities: Optional list of entities to focus on

        Returns:
            Formatted prompt string
        """
        # Base prompt
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

        # Add entity-specific instructions if entities detected
        if entities:
            entity_instruction = f"""
7. IMPORTANT: The question is about {', '.join(entities)}. Focus specifically on direct mentions, quotes, and references to these entities. Prioritize exact quotes and specific statements."""
            prompt += entity_instruction

        prompt += "\n\nYour answer:"

        return prompt

    def _extraction_fallback(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Fallback to extraction-based answer if Gemini fails.

        Args:
            question: User's question
            context_chunks: Context chunks

        Returns:
            Simple extracted answer
        """
        if not context_chunks:
            return "Unable to generate answer due to technical issues."

        # Return the most relevant chunk with source
        first_chunk = context_chunks[0]
        text = first_chunk.get("text", "")
        source = first_chunk.get("source", "unknown")

        # Truncate if too long
        if len(text) > 300:
            text = text[:300] + "..."

        return f"Based on {source}: {text}"

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
