"""
Tests for ConfidenceCalculator component.

Tests confidence score calculation and explanation generation.
"""

import pytest

from src.services.rag.confidence import ConfidenceCalculator
from src.services.rag.models import ContextChunk, SearchResult


class TestConfidenceCalculator:
    """Test suite for ConfidenceCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create ConfidenceCalculator instance."""
        return ConfidenceCalculator()

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                document="Test document 1",
                metadata={"source": "test1.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.3,
                id="test1_chunk_0",
                rerank_score=0.9,
            ),
            SearchResult(
                document="Test document 2",
                metadata={"source": "test2.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.4,
                id="test2_chunk_0",
                rerank_score=0.8,
            ),
        ]

    @pytest.fixture
    def sample_context_chunks(self, sample_search_results):
        """Create sample context chunks from search results."""
        return [ContextChunk.from_search_result(r) for r in sample_search_results]

    def test_initialization(self, calculator):
        """Test ConfidenceCalculator initialization."""
        assert calculator.RETRIEVAL_WEIGHT == 0.4
        assert calculator.CONSISTENCY_WEIGHT == 0.25
        assert calculator.CHUNK_COVERAGE_WEIGHT == 0.2
        assert calculator.ENTITY_COVERAGE_WEIGHT == 0.15

    def test_calculate_with_good_results(
        self, calculator, sample_search_results, sample_context_chunks
    ):
        """Test confidence calculation with high-quality results."""
        result = calculator.calculate(
            question="What is the topic?",
            search_results=sample_search_results,
            context_chunks=sample_context_chunks,
        )

        assert 0.0 <= result.score <= 1.0
        assert result.level in ["high", "medium", "low"]
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

        # Check factors
        assert 0.0 <= result.factors.retrieval_score <= 1.0
        assert 0.0 <= result.factors.consistency <= 1.0
        assert result.factors.chunk_coverage >= 0

    def test_calculate_with_no_results(self, calculator):
        """Test confidence calculation with no results."""
        result = calculator.calculate(
            question="What is nothing?", search_results=[], context_chunks=[]
        )

        assert result.score == 0.0
        assert result.level == "low"
        assert "No relevant context" in result.explanation
        assert result.factors.retrieval_score == 0.0
        assert result.factors.chunk_coverage == 0

    def test_calculate_with_single_result(self, calculator):
        """Test confidence with single result."""
        search_result = SearchResult(
            document="Single doc",
            metadata={"source": "single.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.2,
            id="single_chunk_0",
            rerank_score=0.95,
        )
        context_chunk = ContextChunk.from_search_result(search_result)

        result = calculator.calculate(
            question="Test question", search_results=[search_result], context_chunks=[context_chunk]
        )

        # With single result, consistency should be 1.0 (no variance)
        assert result.factors.consistency == 1.0
        # chunk_coverage is normalized: 1 chunk / 10 = 0.1
        assert result.factors.chunk_coverage == 0.1

    def test_score_normalization(self, calculator):
        """Test that negative scores are normalized to [0, 1]."""
        # Create results with negative scores (like BM25 can produce)
        search_results = [
            SearchResult(
                document="Doc with negative score",
                metadata={"source": "neg.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.5,
                id="neg_chunk_0",
                combined_score=-5.0,  # Negative score
            )
        ]
        context_chunks = [
            ContextChunk(
                text="Test text", source="neg.txt", chunk_index=0, score=-5.0  # Negative score
            )
        ]

        result = calculator.calculate(
            question="Test", search_results=search_results, context_chunks=context_chunks
        )

        # Should normalize to valid range
        assert 0.0 <= result.factors.retrieval_score <= 1.0
        assert 0.0 <= result.score <= 1.0

    def test_high_confidence_level(self, calculator):
        """Test that high-quality results produce high confidence."""
        # Create excellent results
        search_results = [
            SearchResult(
                document=f"High quality doc {i}",
                metadata={"source": f"doc{i}.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.1 + i * 0.01,
                id=f"doc{i}_chunk_0",
                rerank_score=0.95 - i * 0.01,
            )
            for i in range(5)
        ]
        context_chunks = [ContextChunk.from_search_result(r) for r in search_results]

        result = calculator.calculate(
            question="What are Trump's views?",
            search_results=search_results,
            context_chunks=context_chunks,
        )

        # Should be high confidence with good scores and multiple chunks
        assert result.score >= calculator.MEDIUM_THRESHOLD

    def test_low_confidence_level(self, calculator):
        """Test that poor results produce low confidence."""
        # Create poor quality results
        search_results = [
            SearchResult(
                document="Poor match",
                metadata={"source": "poor.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.9,  # Very distant
                id="poor_chunk_0",
                rerank_score=0.2,  # Low rerank score
            )
        ]
        context_chunks = [ContextChunk.from_search_result(r) for r in search_results]

        result = calculator.calculate(
            question="Unrelated question",
            search_results=search_results,
            context_chunks=context_chunks,
        )

        # Should be low or medium confidence
        assert result.score < calculator.HIGH_THRESHOLD

    def test_entity_extraction(self, calculator):
        """Test entity extraction from questions."""
        entities = calculator._extract_entities("What did Trump say about Biden?")

        assert "Trump" in entities
        assert "Biden" in entities
        # Small words should be filtered
        assert "did" not in entities
        assert "say" not in entities

    def test_entity_extraction_no_entities(self, calculator):
        """Test entity extraction with no capitalized words."""
        entities = calculator._extract_entities("what is the topic?")
        assert len(entities) == 0

    def test_consistency_calculation(self, calculator):
        """Test consistency factor with varying scores."""
        # Low variance (high consistency)
        high_consistency_chunks = [
            ContextChunk(text="Text", source="doc.txt", chunk_index=i, score=0.9) for i in range(3)
        ]
        search_results = [
            SearchResult(
                document="Doc",
                metadata={"source": "doc.txt", "chunk_index": i, "total_chunks": 3},
                distance=0.1,
                id=f"doc_chunk_{i}",
                rerank_score=0.9,
            )
            for i in range(3)
        ]

        result = calculator.calculate(
            question="Test", search_results=search_results, context_chunks=high_consistency_chunks
        )

        # High consistency (low variance) should give high consistency factor
        assert result.factors.consistency > 0.9

    def test_explanation_generation(self, calculator, sample_search_results, sample_context_chunks):
        """Test that explanations are generated properly."""
        result = calculator.calculate(
            question="Test question",
            search_results=sample_search_results,
            context_chunks=sample_context_chunks,
        )

        explanation = result.explanation
        assert isinstance(explanation, str)
        assert len(explanation) > 20  # Should be a meaningful explanation
        # Should mention confidence level
        assert result.level in explanation.lower()
