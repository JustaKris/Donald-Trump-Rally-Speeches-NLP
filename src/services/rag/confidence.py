"""
Confidence calculator component for RAG service.

Calculates confidence scores and generates human-readable explanations
for RAG answer quality based on multiple factors.
"""

import logging
from typing import List, Optional

from .models import ConfidenceFactors, ConfidenceResult, ContextChunk, SearchResult

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Calculates confidence scores for RAG answers.

    Uses multiple factors including retrieval quality, score consistency,
    chunk coverage, and entity mention frequency to assess answer confidence.
    """

    # Weight factors for confidence calculation
    RETRIEVAL_WEIGHT = 0.4  # 40% - Average retrieval score
    CONSISTENCY_WEIGHT = 0.25  # 25% - Score consistency (low variance)
    CHUNK_COVERAGE_WEIGHT = 0.2  # 20% - Number of supporting chunks
    ENTITY_COVERAGE_WEIGHT = 0.15  # 15% - Entity mention frequency

    # Thresholds for confidence levels
    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.4

    def __init__(self):
        """Initialize confidence calculator."""
        logger.debug("ConfidenceCalculator initialized")

    def calculate(
        self,
        question: str,
        context_chunks: List[ContextChunk],
        search_results: List[SearchResult],
    ) -> ConfidenceResult:
        """
        Calculate confidence using multiple factors for accurate assessment.

        Factors considered:
        1. Average retrieval score (semantic similarity) - 40%
        2. Score consistency (low variance = more confident) - 25%
        3. Number of supporting chunks - 20%
        4. Entity mention frequency (if applicable) - 15%

        Args:
            question: Original question
            context_chunks: Retrieved context with scores
            search_results: Raw search results

        Returns:
            ConfidenceResult with score, level, and explanation
        """
        if not context_chunks:
            return ConfidenceResult(
                score=0.0,
                level="low",
                explanation="No relevant context found for the question.",
                factors=ConfidenceFactors(
                    retrieval_score=0.0,
                    consistency=0.0,
                    chunk_coverage=0,
                    entity_coverage=None,
                ),
            )

        # Factor 1: Average retrieval score (40% weight)
        scores = [c.score for c in context_chunks if c.score is not None]
        if not scores:
            scores = [0.0]

        # Normalize scores to [0, 1] range (BM25 can produce negative scores)
        # For negative scores, we clamp to 0. For very high scores, we normalize.
        max_score = max(scores)
        if max_score > 1.0:
            # Normalize all scores by dividing by max
            normalized_scores = [max(0.0, min(1.0, s / max_score)) for s in scores]
        else:
            # Clamp negative scores to 0, keep positive scores as-is
            normalized_scores = [max(0.0, min(1.0, s)) for s in scores]

        avg_score = sum(normalized_scores) / len(normalized_scores)
        score_factor = avg_score * self.RETRIEVAL_WEIGHT

        # Factor 2: Score consistency - low variance is good (25% weight)
        if len(normalized_scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in normalized_scores) / len(normalized_scores)
            consistency = max(0, 1 - variance)  # Convert variance to consistency
        else:
            consistency = 1.0
        consistency_factor = consistency * self.CONSISTENCY_WEIGHT

        # Factor 3: Number of supporting chunks (20% weight)
        chunk_count_score = min(len(context_chunks) / 10.0, 1.0)  # Normalize to max 10
        chunk_factor = chunk_count_score * self.CHUNK_COVERAGE_WEIGHT

        # Factor 4: Entity mention frequency (15% weight)
        entities = self._extract_entities(question)
        if entities:
            entity_coverage = self._calculate_entity_coverage(entities, context_chunks)
        else:
            entity_coverage = None  # No entities detected

        # Use neutral score if no entities
        entity_factor = (
            entity_coverage * self.ENTITY_COVERAGE_WEIGHT
            if entity_coverage is not None
            else 0.5 * self.ENTITY_COVERAGE_WEIGHT
        )

        # Combine all factors
        final_score = score_factor + consistency_factor + chunk_factor + entity_factor

        # Determine confidence level
        if final_score >= self.HIGH_THRESHOLD:
            level = "high"
        elif final_score >= self.MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        # Generate human-readable explanation
        explanation = self._generate_explanation(
            level=level,
            score=final_score,
            retrieval_score=avg_score,
            consistency=consistency,
            chunk_count=len(context_chunks),
            entity_coverage=entity_coverage,
            entities=entities,
        )

        return ConfidenceResult(
            score=round(final_score, 3),
            level=level,
            explanation=explanation,
            factors=ConfidenceFactors(
                retrieval_score=round(avg_score, 3),
                consistency=round(consistency, 3),
                chunk_coverage=round(chunk_count_score, 3),  # Use normalized score
                entity_coverage=round(entity_coverage, 3) if entity_coverage else None,
            ),
        )

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential entities from question.

        Simple heuristic: capitalized words longer than 2 chars.

        Args:
            text: Question text

        Returns:
            List of potential entity names
        """
        words = text.split()
        entities = [w.strip(".,!?;:\"'") for w in words if len(w) > 2 and w[0].isupper()]
        return entities

    def _calculate_entity_coverage(
        self, entities: List[str], context_chunks: List[ContextChunk]
    ) -> float:
        """
        Calculate what percentage of chunks mention the entities.

        Args:
            entities: Entity names to look for
            context_chunks: Context chunks to search in

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        entity_mentions = 0

        for chunk in context_chunks:
            text_lower = chunk.text.lower()
            if any(entity.lower() in text_lower for entity in entities):
                entity_mentions += 1

        return entity_mentions / len(context_chunks) if context_chunks else 0.0

    def _generate_explanation(
        self,
        level: str,
        score: float,
        retrieval_score: float,
        consistency: float,
        chunk_count: int,
        entity_coverage: Optional[float],
        entities: Optional[List[str]],
    ) -> str:
        """
        Generate human-readable explanation for confidence level.

        Args:
            level: Confidence level (high/medium/low)
            score: Overall confidence score
            retrieval_score: Average semantic similarity
            consistency: Score consistency
            chunk_count: Number of chunks
            entity_coverage: Entity coverage ratio
            entities: Detected entities

        Returns:
            Human-readable explanation string
        """
        explanations = []

        # Main score explanation
        explanations.append(f"Overall confidence is {level.upper()} (score: {score:.2f})")

        # Retrieval quality
        if retrieval_score > 0.8:
            explanations.append(f"excellent semantic match (similarity: {retrieval_score:.2f})")
        elif retrieval_score > 0.6:
            explanations.append(f"good semantic match (similarity: {retrieval_score:.2f})")
        elif retrieval_score > 0.4:
            explanations.append(f"moderate semantic match (similarity: {retrieval_score:.2f})")
        else:
            explanations.append(f"weak semantic match (similarity: {retrieval_score:.2f})")

        # Consistency
        if consistency > 0.9:
            explanations.append(f"very consistent results (consistency: {consistency:.2f})")
        elif consistency > 0.7:
            explanations.append(f"consistent results (consistency: {consistency:.2f})")
        else:
            explanations.append(f"varied results (consistency: {consistency:.2f})")

        # Chunk coverage
        explanations.append(f"{chunk_count} supporting context chunks")

        # Entity coverage
        if entities and entity_coverage is not None:
            entity_names = ", ".join(entities[:2])  # First 2 entities
            if entity_coverage > 0.8:
                explanations.append(f"'{entity_names}' mentioned in all retrieved chunks")
            elif entity_coverage > 0.5:
                explanations.append(
                    f"'{entity_names}' mentioned in most chunks ({entity_coverage:.0%})"
                )
            else:
                explanations.append(
                    f"'{entity_names}' mentioned in some chunks ({entity_coverage:.0%})"
                )

        # Join with appropriate separators
        base = explanations[0]
        if len(explanations) > 1:
            details = ", ".join(explanations[1:])
            return f"{base} based on {details}."
        return f"{base}."
