"""
Document loader component for RAG service.

Handles loading text documents from disk, chunking them into appropriate sizes,
and preparing them for embedding and vector storage.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads and chunks documents for RAG indexing.

    Uses LangChain's RecursiveCharacterTextSplitter for intelligent
    text chunking that preserves semantic boundaries.
    """

    def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 150):
        """
        Initialize document loader.

        Args:
            chunk_size: Maximum size of text chunks in characters (~512-768 tokens)
            chunk_overlap: Overlap between chunks in characters (~100-150 tokens)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(f"DocumentLoader initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def load_from_directory(self, data_dir: str) -> Tuple[List[str], List[dict], List[str]]:
        """
        Load and chunk all text documents from a directory.

        Args:
            data_dir: Directory containing text files

        Returns:
            Tuple of (chunks, metadatas, chunk_ids)

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        all_chunks = []
        all_metadatas = []
        all_ids = []
        files_processed = 0

        for file_path in data_path.glob("*.txt"):
            try:
                chunks, metadatas, ids = self._load_file(file_path)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)
                files_processed += 1
                logger.debug(f"Loaded {len(chunks)} chunks from {file_path.name}")

            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue

        logger.info(f"Loaded {len(all_chunks)} chunks from {files_processed} documents")
        return all_chunks, all_metadatas, all_ids

    def _load_file(self, file_path: Path) -> Tuple[List[str], List[dict], List[str]]:
        """
        Load and chunk a single text file.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (chunks, metadatas, chunk_ids)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split document into chunks
        chunks = self.text_splitter.split_text(content)

        # Create metadata for each chunk
        metadatas = []
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.stem}_chunk_{i}"
            chunk_ids.append(chunk_id)

            metadatas.append(
                {
                    "source": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        return chunks, metadatas, chunk_ids

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk a single text string.

        Useful for processing ad-hoc text without file I/O.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
