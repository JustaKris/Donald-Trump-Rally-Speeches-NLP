"""
Integration tests for RAG (Retrieval-Augmented Generation) Service.

Tests the complete RAG pipeline including document loading, chunking,
embedding, semantic search, and question answering with the refactored
modular architecture.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from src.services.rag_service import RAGService


def cleanup_chromadb(persist_dir: str, service=None):
    """Helper function to properly cleanup ChromaDB resources on Windows."""
    try:
        if service is not None:
            try:
                service.chroma_client.clear_system_cache()
            except Exception:
                pass
            del service
        time.sleep(0.2)  # Give Windows time to release file handles
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample text files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample documents
        doc1 = Path(tmpdir) / "doc1.txt"
        doc1.write_text(
            "The economy is growing rapidly. Jobs are being created every month. "
            "Economic policies are focusing on growth and development. "
            "The stock market has reached new highs this year."
        )

        doc2 = Path(tmpdir) / "doc2.txt"
        doc2.write_text(
            "Healthcare reform is a priority. Access to medical care needs improvement. "
            "The healthcare system requires significant changes. "
            "Many people struggle with healthcare costs."
        )

        doc3 = Path(tmpdir) / "doc3.txt"
        doc3.write_text(
            "Education funding is increasing. Schools need more resources. "
            "Teachers deserve better compensation. "
            "Student performance is improving with new programs."
        )

        yield tmpdir


@pytest.fixture
def rag_service(temp_data_dir):
    """Create a RAG service instance with test data."""
    persist_dir = tempfile.mkdtemp()
    service = None
    try:
        service = RAGService(
            collection_name="test_collection",
            persist_directory=persist_dir,
            chunk_size=100,
            chunk_overlap=20,
        )
        service.load_documents(data_dir=temp_data_dir)
        yield service
    finally:
        cleanup_chromadb(persist_dir, service)


class TestRAGServiceInitialization:
    """Test RAG service initialization and configuration."""

    def test_service_creation_default_params(self):
        """Test that RAG service can be created with default parameters."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            assert service is not None
            assert service.collection_name == "speeches"
            assert service.chunk_size == 2048
            assert service.chunk_overlap == 150
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_service_creation_custom_params(self):
        """Test RAG service with custom parameters."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(
                collection_name="custom",
                persist_directory=persist_dir,
                chunk_size=200,
                chunk_overlap=30,
            )
            assert service.collection_name == "custom"
            assert service.chunk_size == 200
            assert service.chunk_overlap == 30
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_persistence_directory_creation(self):
        """Test that persistence directory is created automatically."""
        tmpdir = tempfile.mkdtemp()
        persist_dir = os.path.join(tmpdir, "chromadb")
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            assert os.path.exists(persist_dir)
        finally:
            cleanup_chromadb(persist_dir, service)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_embedding_model_configuration(self):
        """Test that embedding model is properly configured."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            # Should be using mpnet-base-v2 by default
            assert "mpnet" in str(service.embedding_model).lower()
        finally:
            cleanup_chromadb(persist_dir, service)


class TestDocumentLoading:
    """Test document loading and indexing functionality."""

    def test_load_documents_from_directory(self, temp_data_dir, rag_service):
        """Test loading documents from directory."""
        stats = rag_service.get_stats()
        assert stats["total_chunks"] > 0
        assert stats["unique_sources"] == 3
        assert "doc1.txt" in stats["sources"]
        assert "doc2.txt" in stats["sources"]
        assert "doc3.txt" in stats["sources"]

    def test_load_nonexistent_directory(self):
        """Test that loading from non-existent directory raises error."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            with pytest.raises(FileNotFoundError):
                service.load_documents(data_dir="/nonexistent/path")
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_document_chunking_small_chunks(self, temp_data_dir):
        """Test that documents are properly chunked with small chunk size."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(
                persist_directory=persist_dir,
                chunk_size=50,  # Small chunks for testing
                chunk_overlap=10,
            )
            service.load_documents(data_dir=temp_data_dir)
            stats = service.get_stats()
            # With small chunk size, should create multiple chunks
            assert stats["total_chunks"] >= 3
        finally:
            cleanup_chromadb(persist_dir, service)


class TestSemanticSearch:
    """Test semantic search functionality."""

    def test_search_returns_results(self, rag_service):
        """Test that search returns relevant results."""
        results = rag_service.search("economy and jobs", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_search_relevance(self, rag_service):
        """Test that search returns relevant documents."""
        results = rag_service.search("healthcare and medical", top_k=1)
        assert len(results) > 0
        # Should find doc2.txt which is about healthcare
        assert (
            "healthcare" in results[0]["document"].lower()
            or "medical" in results[0]["document"].lower()
        )

    def test_search_with_different_top_k(self, rag_service):
        """Test search with different top_k values."""
        results_1 = rag_service.search("education", top_k=1)
        results_5 = rag_service.search("education", top_k=5)

        assert len(results_1) == 1
        assert len(results_5) > len(results_1)

    def test_empty_query_handling(self, rag_service):
        """Test search with empty query."""
        results = rag_service.search("", top_k=3)
        # Should still return results (chromadb handles empty queries)
        assert isinstance(results, list)


class TestQuestionAnswering:
    """Test RAG question answering functionality."""

    def test_ask_returns_complete_response(self, rag_service):
        """Test asking a question returns all expected fields."""
        result = rag_service.ask("What is happening with the economy?")

        # Check all required fields are present
        assert "answer" in result
        assert "context" in result
        assert "confidence" in result
        assert "confidence_score" in result
        assert "confidence_explanation" in result
        assert "confidence_factors" in result
        assert "sources" in result
        assert "entities" in result

        # Check field types
        assert isinstance(result["answer"], str)
        assert isinstance(result["context"], list)
        assert result["confidence"] in ["high", "medium", "low"]
        assert isinstance(result["sources"], list)
        assert isinstance(result["entities"], list)

    def test_confidence_levels(self, rag_service):
        """Test that confidence levels are assigned correctly."""
        result = rag_service.ask("economy", top_k=3)
        assert result["confidence"] in ["high", "medium", "low"]
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_context_structure(self, rag_service):
        """Test that returned context has proper structure."""
        result = rag_service.ask("healthcare reform")

        for ctx in result["context"]:
            assert "text" in ctx
            assert "source" in ctx
            assert "chunk_index" in ctx
            assert isinstance(ctx["text"], str)
            assert isinstance(ctx["source"], str)
            assert isinstance(ctx["chunk_index"], int)

    def test_sources_are_unique(self, rag_service):
        """Test that sources list contains unique values."""
        result = rag_service.ask("policies", top_k=3)

        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        # Sources should be unique
        assert len(result["sources"]) == len(set(result["sources"]))

    def test_confidence_factors_structure(self, rag_service):
        """Test that confidence factors are properly structured."""
        result = rag_service.ask("education funding", top_k=3)

        factors = result["confidence_factors"]
        assert "retrieval_score" in factors
        assert "consistency" in factors
        assert "chunk_coverage" in factors
        assert "entity_coverage" in factors

        # All factors should be numeric (except entity_coverage which can be None)
        for factor_name, factor_value in factors.items():
            if factor_name == "entity_coverage" and factor_value is None:
                continue  # entity_coverage can be None if no entities detected
            assert isinstance(factor_value, (int, float))
            assert 0.0 <= factor_value <= 1.0


class TestCollectionManagement:
    """Test collection statistics and management."""

    def test_get_stats_structure(self, rag_service):
        """Test getting collection statistics."""
        stats = rag_service.get_stats()

        assert "collection_name" in stats
        assert "total_chunks" in stats
        assert "unique_sources" in stats
        assert "sources" in stats
        assert "embedding_model" in stats
        assert "chunk_size" in stats
        assert "chunk_overlap" in stats

        assert isinstance(stats["total_chunks"], int)
        assert isinstance(stats["unique_sources"], int)
        assert isinstance(stats["sources"], list)

    def test_clear_collection(self, rag_service):
        """Test clearing the collection."""
        # Get initial count
        initial_stats = rag_service.get_stats()
        assert initial_stats["total_chunks"] > 0

        # Clear collection
        success = rag_service.clear_collection()
        assert success is True

        # Verify collection is empty
        final_stats = rag_service.get_stats()
        assert final_stats["total_chunks"] == 0

    def test_collection_persistence(self, temp_data_dir):
        """Test that collection persists across instances."""
        persist_dir = tempfile.mkdtemp()
        service1 = None
        service2 = None
        try:
            # Create first service instance and load documents
            service1 = RAGService(persist_directory=persist_dir)
            service1.load_documents(data_dir=temp_data_dir)
            count1 = service1.get_stats()["total_chunks"]

            # Clean up first service
            del service1
            time.sleep(0.2)

            # Create second service instance with same persist directory
            service2 = RAGService(persist_directory=persist_dir)
            count2 = service2.get_stats()["total_chunks"]

            # Should have the same number of chunks
            assert count1 == count2
            assert count2 > 0
        finally:
            cleanup_chromadb(persist_dir, service2)

    def test_count_method(self, rag_service):
        """Test the count() method."""
        count = rag_service.count()
        assert isinstance(count, int)
        assert count > 0
        assert count == rag_service.get_stats()["total_chunks"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_collection_search(self):
        """Test searching on empty collection."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            results = service.search("test query", top_k=3)
            assert isinstance(results, list)
            assert len(results) == 0
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_empty_collection_ask(self):
        """Test asking question on empty collection."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            service = RAGService(persist_directory=persist_dir)
            result = service.ask("test question")

            assert "answer" in result
            assert "context" in result
            assert result["confidence"] == "low"
            assert len(result["context"]) == 0
            assert result["confidence_score"] == 0.0
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_very_long_query(self, rag_service):
        """Test handling of very long queries."""
        long_query = " ".join(["economy healthcare education"] * 100)
        results = rag_service.search(long_query, top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0


@pytest.mark.integration
class TestRAGPipeline:
    """End-to-end integration tests for complete RAG pipeline."""

    def test_end_to_end_workflow(self, temp_data_dir):
        """Test complete workflow from indexing to querying."""
        persist_dir = tempfile.mkdtemp()
        service = None
        try:
            # 1. Initialize service
            service = RAGService(persist_directory=persist_dir)

            # 2. Load documents
            docs_loaded = service.load_documents(data_dir=temp_data_dir)
            assert docs_loaded == 3

            # 3. Get stats
            stats = service.get_stats()
            assert stats["total_chunks"] > 0
            assert stats["unique_sources"] == 3

            # 4. Perform search
            search_results = service.search("economy", top_k=3)
            assert len(search_results) > 0

            # 5. Ask question
            answer = service.ask("What are the main topics discussed?")
            assert answer["answer"] is not None
            assert len(answer["context"]) > 0
            assert "confidence_factors" in answer

            # 6. Clear and verify
            service.clear_collection()
            final_stats = service.get_stats()
            assert final_stats["total_chunks"] == 0
        finally:
            cleanup_chromadb(persist_dir, service)

    def test_ask_with_entity_extraction(self, rag_service):
        """Test that entities are extracted from questions."""
        result = rag_service.ask("What are the economy policies?", top_k=5)

        # Check response structure
        assert "answer" in result
        assert "confidence" in result
        assert "confidence_score" in result
        assert "confidence_factors" in result
        assert "entities" in result

        # Entities should be a list
        assert isinstance(result["entities"], list)

    def test_confidence_calculation_integration(self, rag_service):
        """Test that confidence metrics are properly calculated."""
        result = rag_service.ask("What topics are discussed?", top_k=5)

        # Should have confidence fields
        assert "confidence" in result
        assert "confidence_score" in result
        assert "confidence_factors" in result
        assert "confidence_explanation" in result

        # Validate confidence_factors structure
        factors = result["confidence_factors"]
        assert "retrieval_score" in factors
        assert "consistency" in factors
        assert "chunk_coverage" in factors
        assert "entity_coverage" in factors

        # All scores should be in valid range (except entity_coverage which can be None)
        for factor_name, score in factors.items():
            if factor_name == "entity_coverage" and score is None:
                continue  # entity_coverage can be None
            assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
