"""
LLM Service for answer generation in RAG system.

Provides integration with Google Gemini for high-quality answer synthesis
from retrieved context chunks.
"""

import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiLLM:
    """
    Service for generating answers using Google Gemini.

    Uses Gemini to synthesize high-quality answers from retrieved context,
    with proper citations and confidence assessment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        """
        Initialize Gemini LLM service.

        Args:
            api_key: Google API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model to use (flash for speed, pro for quality)
            temperature: Generation temperature (0.0-1.0, lower = more focused)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model_name = model_name
        self.temperature = temperature

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
            max_output_tokens=1024,
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

        # Build prompt with entity awareness
        prompt = self._build_prompt(question, context_text, list(sources_used), entities)

        try:
            # Generate answer
            response = self.model.generate_content(prompt, generation_config=self.generation_config)

            answer_text = response.text.strip()

            return {
                "answer": answer_text,
                "reasoning": "Generated using Gemini based on retrieved context",
                "sources_used": list(sources_used),
            }

        except Exception as e:
            # Fallback to extraction-based answer on error
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
            response = self.model.generate_content(
                "Hello", generation_config=self.generation_config
            )
            # Try to access the text - this will fail if blocked/no content
            _ = response.text
            return True
        except Exception:
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
