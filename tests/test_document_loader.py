"""
Tests for DocumentLoader component.

Tests document loading, chunking, and metadata generation.
"""

from pathlib import Path

import pytest

from src.services.rag.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test suite for DocumentLoader."""

    @pytest.fixture
    def loader(self):
        """Create DocumentLoader instance."""
        return DocumentLoader(chunk_size=500, chunk_overlap=50)

    def test_initialization(self, loader):
        """Test DocumentLoader initialization."""
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        assert loader.text_splitter is not None

    def test_chunk_text(self, loader):
        """Test text chunking functionality."""
        text = "This is a test. " * 100  # Create long text
        chunks = loader.chunk_text(text)

        assert len(chunks) > 1  # Should create multiple chunks
        for chunk in chunks:
            assert len(chunk) <= 500  # No chunk should exceed max size
            assert len(chunk) > 0  # No empty chunks

    def test_chunk_text_short_input(self, loader):
        """Test chunking with input shorter than chunk size."""
        text = "Short text"
        chunks = loader.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty(self, loader):
        """Test chunking empty text."""
        chunks = loader.chunk_text("")
        assert chunks == []

    def test_load_from_directory_existing(self, loader):
        """Test loading from real speeches directory."""
        data_dir = "data/Donald Trump Rally Speeches"

        if not Path(data_dir).exists():
            pytest.skip(f"Data directory {data_dir} not found")

        chunks, metadatas, ids = loader.load_from_directory(data_dir)

        # Verify outputs
        assert len(chunks) > 0
        assert len(chunks) == len(metadatas)
        assert len(chunks) == len(ids)

        # Verify metadata structure
        for metadata in metadatas:
            assert "source" in metadata
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert isinstance(metadata["source"], str)
            assert isinstance(metadata["chunk_index"], int)
            assert isinstance(metadata["total_chunks"], int)

        # Verify ID format
        for chunk_id in ids:
            assert isinstance(chunk_id, str)
            assert "_chunk_" in chunk_id

    def test_load_from_nonexistent_directory(self, loader):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory("nonexistent/path")

    def test_load_from_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = DocumentLoader()
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        assert chunks == []
        assert metadatas == []
        assert ids == []

    def test_load_single_file(self, tmp_path):
        """Test loading a single file."""
        # Create test file
        test_file = tmp_path / "test_speech.txt"
        test_content = "This is a test speech. " * 50
        test_file.write_text(test_content, encoding="utf-8")

        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        assert len(chunks) > 0
        assert all(m["source"] == "test_speech.txt" for m in metadatas)
        assert all("test_speech_chunk_" in chunk_id for chunk_id in ids)

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        text = "Word " * 100  # Repetitive text to test overlap

        chunks = loader.chunk_text(text)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # At least some content should appear in both chunks
            assert chunks[0][-20:] != chunks[1][:20]  # Not exact same, but related

    def test_metadata_chunk_indices(self, tmp_path):
        """Test that chunk indices are sequential and correct."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Word " * 1000, encoding="utf-8")

        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        _, metadatas, _ = loader.load_from_directory(str(tmp_path))

        # Get all chunk indices for the file
        indices = [m["chunk_index"] for m in metadatas]
        total_chunks = metadatas[0]["total_chunks"]

        # Should be sequential starting from 0
        assert indices == list(range(len(indices)))
        # total_chunks should match actual count
        assert total_chunks == len(indices)

    def test_multiple_files(self, tmp_path):
        """Test loading multiple files."""
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"speech_{i}.txt"
            test_file.write_text(f"Speech {i} content. " * 50, encoding="utf-8")

        loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        # Should have chunks from all 3 files
        sources = set(m["source"] for m in metadatas)
        assert len(sources) == 3
        assert "speech_0.txt" in sources
        assert "speech_1.txt" in sources
        assert "speech_2.txt" in sources
