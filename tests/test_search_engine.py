"""
Tests for SearchEngine component.

Tests semantic search, hybrid search, and re-ranking functionality.
Note: Some tests require actual embedding models which are mocked.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.services.rag.models import SearchResult
from src.services.rag.search_engine import SearchEngine


class TestSearchEngine:
    """Test suite for SearchEngine."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        model = Mock()
        # Return numpy array (as real model does)
        model.encode = MagicMock(return_value=np.array([[0.1] * 768]))  # 768-dim embedding
        return model

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        collection = Mock()
        collection.query = MagicMock(
            return_value={
                "documents": [["Test document 1", "Test document 2"]],
                "metadatas": [
                    [
                        {"source": "test1.txt", "chunk_index": 0, "total_chunks": 1},
                        {"source": "test2.txt", "chunk_index": 0, "total_chunks": 1},
                    ]
                ],
                "distances": [[0.3, 0.5]],
                "ids": [["test1_chunk_0", "test2_chunk_0"]],
            }
        )
        collection.get = MagicMock(
            return_value={
                "documents": ["Doc 1", "Doc 2", "Doc 3"],
                "metadatas": [{"source": "1.txt"}, {"source": "2.txt"}, {"source": "3.txt"}],
                "ids": ["1", "2", "3"],
            }
        )
        return collection

    @pytest.fixture
    def search_engine_semantic(self, mock_embedding_model, mock_collection):
        """Create SearchEngine with semantic search only."""
        return SearchEngine(
            embedding_model=mock_embedding_model,
            collection=mock_collection,
            use_reranking=False,
            use_hybrid_search=False,
        )

    @pytest.fixture
    def search_engine_hybrid(self, mock_embedding_model, mock_collection):
        """Create SearchEngine with hybrid search."""
        engine = SearchEngine(
            embedding_model=mock_embedding_model,
            collection=mock_collection,
            use_reranking=False,
            use_hybrid_search=True,
        )
        # Initialize BM25
        engine.initialize_bm25(["document one", "document two", "document three"])
        return engine

    def test_initialization_semantic(self, mock_embedding_model, mock_collection):
        """Test SearchEngine initialization with semantic search."""
        engine = SearchEngine(
            embedding_model=mock_embedding_model,
            collection=mock_collection,
            use_reranking=False,
            use_hybrid_search=False,
        )

        assert engine.embedding_model == mock_embedding_model
        assert engine.collection == mock_collection
        assert engine.use_reranking is False
        assert engine.use_hybrid_search is False
        assert engine.bm25 is None

    def test_initialization_hybrid(self, mock_embedding_model, mock_collection):
        """Test SearchEngine initialization with hybrid search."""
        engine = SearchEngine(
            embedding_model=mock_embedding_model,
            collection=mock_collection,
            use_reranking=False,
            use_hybrid_search=True,
        )

        assert engine.use_hybrid_search is True
        assert engine.bm25 is None  # Not initialized until documents loaded

    def test_initialization_with_reranking(self, mock_embedding_model, mock_collection):
        """Test SearchEngine initialization with re-ranking."""
        with patch("src.services.rag.search_engine.CrossEncoder") as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            engine = SearchEngine(
                embedding_model=mock_embedding_model,
                collection=mock_collection,
                use_reranking=True,
                use_hybrid_search=False,
            )

            assert engine.use_reranking is True
            assert engine.reranker is not None

    def test_semantic_search_basic(self, search_engine_semantic):
        """Test basic semantic search."""
        results = search_engine_semantic.search("test query", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_semantic_search_returns_correct_fields(self, search_engine_semantic):
        """Test that semantic search returns all required fields."""
        results = search_engine_semantic.search("test query", top_k=1)

        if results:
            result = results[0]
            assert hasattr(result, "document")
            assert hasattr(result, "metadata")
            assert hasattr(result, "distance")
            assert hasattr(result, "id")
            assert hasattr(result, "sources")

    def test_bm25_initialization(self, search_engine_hybrid):
        """Test BM25 initialization."""
        documents = ["test document one", "test document two"]
        search_engine_hybrid.initialize_bm25(documents)

        assert search_engine_hybrid.bm25 is not None
        assert len(search_engine_hybrid.bm25_corpus) == len(documents)

    def test_bm25_initialization_empty_documents(self, search_engine_hybrid):
        """Test BM25 initialization with empty documents."""
        search_engine_hybrid.initialize_bm25([])

        # Should not create BM25 with empty corpus
        assert search_engine_hybrid.bm25 is None

    def test_hybrid_search_combines_scores(self, search_engine_hybrid):
        """Test that hybrid search combines semantic and BM25 scores."""
        results = search_engine_hybrid.search("document query", top_k=2)

        assert isinstance(results, list)
        # Results should have combined scores
        for result in results:
            # Hybrid search should set combined_score
            assert result.combined_score is not None or result.rerank_score is not None

    def test_search_with_zero_top_k(self, search_engine_semantic):
        """Test search with top_k=0."""
        results = search_engine_semantic.search("test", top_k=0)
        assert results == []

    def test_search_with_negative_top_k(self, search_engine_semantic):
        """Test search with negative top_k."""
        results = search_engine_semantic.search("test", top_k=-1)
        assert results == []

    def test_search_empty_query(self, search_engine_semantic):
        """Test search with empty query."""
        results = search_engine_semantic.search("", top_k=5)
        # Should still work, just might return no results
        assert isinstance(results, list)

    def test_search_result_ordering(self, search_engine_semantic):
        """Test that search results are ordered by score (best first)."""
        results = search_engine_semantic.search("test query", top_k=5)

        if len(results) > 1:
            scores = [r.effective_score for r in results]
            # Scores should be in descending order (best first)
            assert scores == sorted(scores, reverse=True)

    def test_search_result_sources(self, search_engine_semantic):
        """Test that search results include source information."""
        results = search_engine_semantic.search("test", top_k=2)

        for result in results:
            assert isinstance(result.sources, list)
            assert result.source_name in result.sources

    def test_effective_score_property(self):
        """Test SearchResult effective_score property."""
        # Test with rerank_score
        result1 = SearchResult(
            document="Test",
            metadata={"source": "test.txt"},
            distance=0.5,
            id="test_1",
            rerank_score=0.95,
        )
        assert result1.effective_score == 0.95

        # Test with combined_score (no rerank)
        result2 = SearchResult(
            document="Test",
            metadata={"source": "test.txt"},
            distance=0.5,
            id="test_2",
            combined_score=0.8,
        )
        assert result2.effective_score == 0.8

        # Test with only distance
        result3 = SearchResult(
            document="Test", metadata={"source": "test.txt"}, distance=0.3, id="test_3"
        )
        assert result3.effective_score == 0.7  # 1.0 - 0.3

    def test_search_deduplication(self, search_engine_semantic):
        """Test that search removes duplicate results."""
        # Mock collection to return duplicates
        search_engine_semantic.collection.query = MagicMock(
            return_value={
                "documents": [["Same doc", "Same doc", "Different doc"]],
                "metadatas": [
                    [
                        {"source": "test.txt", "chunk_index": 0},
                        {"source": "test.txt", "chunk_index": 0},  # Duplicate
                        {"source": "test2.txt", "chunk_index": 0},
                    ]
                ],
                "distances": [[0.3, 0.3, 0.5]],
                "ids": [["test_0", "test_0", "test2_0"]],  # Duplicate ID
            }
        )

        results = search_engine_semantic.search("test", top_k=5)

        # Should have deduplicated by ID
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids))  # No duplicate IDs

    def test_hybrid_search_weight_distribution(self, search_engine_hybrid):
        """Test that hybrid search uses proper weight distribution."""
        # This tests that the SEMANTIC_WEIGHT and BM25_WEIGHT are applied
        results = search_engine_hybrid.search("test document", top_k=3)

        # Should return results (testing that weights don't break the search)
        assert isinstance(results, list)

    @patch("src.services.rag.search_engine.CrossEncoder")
    def test_reranking_applied(
        self, mock_cross_encoder_class, mock_embedding_model, mock_collection
    ):
        """Test that re-ranking is applied when enabled."""
        # Mock the cross-encoder
        mock_reranker = Mock()
        mock_reranker.predict = MagicMock(return_value=[0.9, 0.7])
        mock_cross_encoder_class.return_value = mock_reranker

        engine = SearchEngine(
            embedding_model=mock_embedding_model,
            collection=mock_collection,
            use_reranking=True,
            use_hybrid_search=False,
        )

        results = engine.search("test query", top_k=2)

        # Reranker should have been called
        if results:
            # Check that results have rerank scores
            assert any(r.rerank_score is not None for r in results)
