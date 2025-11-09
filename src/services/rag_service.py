"""
RAG (Retrieval-Augmented Generation) Service - Refactored Version.

This is the streamlined version that delegates to modular components.
Provides semantic search and question-answering capabilities over text documents
using ChromaDB for vector storage, sentence-transformers for embeddings,
Gemini for answer generation, and hybrid search for improved retrieval.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..services.llm_service import GeminiLLM
from .rag.confidence import ConfidenceCalculator
from .rag.document_loader import DocumentLoader
from .rag.entity_analyzer import EntityAnalyzer
from .rag.models import SearchResult
from .rag.search_engine import SearchEngine

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)


class RAGService:
    """
    Refactored RAG service that delegates to modular components.

    This version maintains backward compatibility while using the new
    modular architecture under the hood.
    """

    def __init__(
        self,
        collection_name: str = "speeches",
        persist_directory: str = "./data/chromadb",
        embedding_model: str = "all-mpnet-base-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 2048,
        chunk_overlap: int = 150,
        llm_service: Optional[GeminiLLM] = None,
        use_reranking: bool = True,
        use_hybrid_search: bool = True,
    ):
        """
        Initialize RAG service with modular components.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for ChromaDB persistence
            embedding_model: HuggingFace model for embeddings
            reranker_model: Cross-encoder model for re-ranking
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            llm_service: Optional LLM service instance
            use_reranking: Use cross-encoder for re-ranking results
            use_hybrid_search: Combine semantic and keyword search
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        # Store chunk parameters as properties for backward compatibility
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(f"Initializing RAG service: collection={collection_name}")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Set LLM service
        self.llm = llm_service
        if llm_service:
            logger.info("RAG service using LLM for answer generation")
        else:
            logger.info("RAG service using extraction-based answers (no LLM)")

        # Initialize ChromaDB client with persistence
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Text documents for semantic search"},
        )

        # Initialize modular components
        self.document_loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.search_engine = SearchEngine(
            embedding_model=self.embedding_model,
            collection=self.collection,
            reranker_model=reranker_model,
            use_reranking=use_reranking,
            use_hybrid_search=use_hybrid_search,
        )

        self.entity_analyzer = EntityAnalyzer(
            collection=self.collection, sentiment_analyzer=None  # Can be set later via property
        )

        self.confidence_calculator = ConfidenceCalculator()

        # Check if collection has existing data
        count = self.collection.count()
        if count > 0:
            logger.info(f"✓ RAG service initialized with {count} existing chunks")
            # Initialize BM25 with existing documents if using hybrid search
            if use_hybrid_search:
                self._initialize_bm25_from_collection()

    def _initialize_bm25_from_collection(self):
        """Initialize BM25 index from existing ChromaDB collection."""
        try:
            # Get all documents from collection (ChromaDB returns dict with 'documents' key)
            result = self.collection.get()
            if result and result.get("documents"):
                documents = result["documents"]
                # Ensure documents is a list of strings
                if isinstance(documents, list) and all(isinstance(d, str) for d in documents):
                    self.search_engine.initialize_bm25(documents)
                    logger.info(f"✓ BM25 index initialized with {len(documents)} documents")
        except Exception as e:
            logger.warning(f"Could not initialize BM25 from existing collection: {e}")

    @property
    def sentiment_analyzer(self):
        """Get sentiment analyzer."""
        return self.entity_analyzer.sentiment_analyzer

    @sentiment_analyzer.setter
    def sentiment_analyzer(self, analyzer):
        """Set sentiment analyzer for entity analysis."""
        self.entity_analyzer.sentiment_analyzer = analyzer

    def load_documents(self, data_dir: str = "data/Donald Trump Rally Speeches") -> int:
        """
        Load and index all text documents from a directory.

        Args:
            data_dir: Directory containing text files

        Returns:
            Number of documents indexed
        """
        # Use DocumentLoader to load and chunk documents
        chunks, metadatas, ids = self.document_loader.load_from_directory(data_dir)

        if not chunks:
            logger.warning(f"No documents found in {data_dir}")
            return 0

        logger.info(f"Loaded {len(chunks)} chunks from {data_dir}")

        # Generate embeddings for all chunks
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            chunks, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        # Add to ChromaDB
        logger.info("Adding documents to vector store...")
        # Type ignore needed due to ChromaDB's loose typing
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )

        # Initialize BM25 for hybrid search
        if self.use_hybrid_search:
            self.search_engine.initialize_bm25(chunks)
            logger.info("✓ BM25 index initialized")

        count = self.collection.count()
        logger.info(f"✓ Total documents in collection: {count}")

        # Count unique source files
        unique_sources = len(set(m["source"] for m in metadatas))
        return unique_sources

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using the configured search strategy.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Delegate to SearchEngine
            results = self.search_engine.search(query, top_k)

            # Convert SearchResult objects to dictionaries for backward compatibility
            return [
                {
                    "document": r.document,
                    "metadata": r.metadata,
                    "distance": r.distance,
                    "id": r.id,
                    "rerank_score": r.rerank_score,
                    "combined_score": r.combined_score,
                    "sources": r.sources,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG (retrieval + generation).

        Args:
            question: Question to answer
            top_k: Number of context chunks to retrieve

        Returns:
            Dictionary with answer, confidence, sources, and metadata
        """
        logger.info(f"Question: {question}")

        # Extract entities from question
        entities = self.entity_analyzer.extract_entities(question)
        logger.debug(f"Extracted entities: {entities}")

        # Search for relevant context
        results = self.search_engine.search(question, top_k)

        if not results:
            return {
                "answer": "I don't have enough information to answer that question.",
                "context": [],
                "confidence": "low",
                "confidence_score": 0.0,
                "confidence_explanation": "No relevant documents found",
                "confidence_factors": {
                    "retrieval_score": 0.0,
                    "consistency": 0.0,
                    "chunk_coverage": 0.0,
                    "entity_coverage": 0.0,
                },
                "sources": [],
                "entities": entities,
            }

        # Convert to ContextChunk objects for confidence calculation
        from .rag.models import ContextChunk

        context_chunks = [ContextChunk.from_search_result(r) for r in results]

        # Generate answer using LLM or extraction
        if self.llm:
            answer = self._generate_llm_answer(question, context_chunks)
        else:
            # Fallback: use the most relevant chunk
            answer = context_chunks[0].text

        # Calculate confidence (note: ConfidenceCalculator doesn't use answer text)
        confidence_result = self.confidence_calculator.calculate(
            question=question, search_results=results, context_chunks=context_chunks
        )

        # Get entity statistics if entities were found
        entity_stats = None
        if entities and self.entity_analyzer.sentiment_analyzer:
            try:
                entity_stats = self.entity_analyzer.get_statistics(
                    entities, include_sentiment=True, include_associations=True
                )
            except Exception as e:
                logger.warning(f"Could not get entity statistics: {e}")

        # Build context for response (convert ContextChunk to dict)
        context_list = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
            }
            for chunk in context_chunks
        ]

        # Build confidence factors dict
        confidence_factors_dict = {
            "retrieval_score": confidence_result.factors.retrieval_score,
            "consistency": confidence_result.factors.consistency,
            "chunk_coverage": confidence_result.factors.chunk_coverage,
            "entity_coverage": confidence_result.factors.entity_coverage,
        }

        # Build response
        response = {
            "answer": answer,
            "context": context_list,
            "confidence": confidence_result.level,
            "confidence_score": confidence_result.score,
            "confidence_explanation": confidence_result.explanation,
            "confidence_factors": confidence_factors_dict,
            "sources": sorted(list(set(c.source for c in context_chunks))),  # Deduplicate and sort
            "entities": entities,
        }

        if entity_stats:
            response["entity_statistics"] = entity_stats

        return response

    def _generate_llm_answer(self, question: str, context_chunks: List) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context_chunks: Retrieved context chunks (ContextChunk objects)

        Returns:
            Generated answer
        """
        # Convert ContextChunk objects to dicts for LLM service
        context_dicts = [
            {"text": chunk.text, "source": chunk.source, "chunk_index": chunk.chunk_index}
            for chunk in context_chunks
        ]

        if self.llm is None:
            # No LLM available, use most relevant chunk
            return context_chunks[0].text if context_chunks else "Unable to generate answer."

        try:
            # Type narrowed: self.llm is not None here
            result = self.llm.generate_answer(question=question, context_chunks=context_dicts)
            answer = result.get("answer", "Unable to generate answer.")
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Fallback to most relevant chunk
            return context_chunks[0].text if context_chunks else "Unable to generate answer."

    def get_entity_statistics(
        self,
        entities: List[str],
        include_sentiment: bool = False,
        include_associations: bool = True,
    ) -> Dict[str, Any]:
        """
        Get statistics about entities across the corpus.

        Args:
            entities: List of entity names
            include_sentiment: Include sentiment analysis
            include_associations: Include term associations

        Returns:
            Dictionary with entity statistics
        """
        return self.entity_analyzer.get_statistics(
            entities, include_sentiment=include_sentiment, include_associations=include_associations
        )

    def clear(self):
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from collection")
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Text documents for semantic search"},
        )
        # Reset BM25
        self.search_engine.bm25 = None
        self.search_engine.bm25_corpus = []
        logger.info("✓ Collection cleared")

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        self.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection statistics including:
            - collection_name: Name of the collection
            - total_chunks: Number of chunks in collection
            - unique_sources: Number of unique source files
            - sources: List of source filenames
            - embedding_model: Name of embedding model
            - chunk_size: Chunk size in characters
            - chunk_overlap: Chunk overlap in characters
        """
        count = self.collection.count()

        # Get unique sources
        sources = []
        unique_sources = 0
        if count > 0:
            try:
                result = self.collection.get()
                if result and result.get("metadatas"):
                    metadatas = result["metadatas"]
                    if metadatas:
                        source_set = {
                            m.get("source", "") for m in metadatas if m and isinstance(m, dict)
                        }
                        sources = sorted(list(source_set))
                        unique_sources = len(sources)
            except Exception as e:
                logger.warning(f"Could not get source statistics: {e}")

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "unique_sources": unique_sources,
            "sources": sources,
            "embedding_model": str(self.embedding_model),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def count(self) -> int:
        """Get total number of chunks in collection."""
        return self.collection.count()
