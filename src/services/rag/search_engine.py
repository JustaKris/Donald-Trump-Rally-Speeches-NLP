"""
Search engine component for RAG service.

Handles semantic search, hybrid search (semantic + BM25),
and cross-encoder re-ranking for improved retrieval accuracy.
"""

import logging
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .models import SearchResult

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Manages search operations for RAG.

    Supports pure semantic search, hybrid search (semantic + BM25),
    and optional cross-encoder re-ranking for improved accuracy.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        collection,
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranking: bool = True,
        use_hybrid_search: bool = True,
    ):
        """
        Initialize search engine.

        Args:
            embedding_model: SentenceTransformer model for embeddings
            collection: ChromaDB collection
            reranker_model: Cross-encoder model name for re-ranking
            use_reranking: Enable cross-encoder re-ranking
            use_hybrid_search: Enable hybrid search (semantic + BM25)
        """
        self.embedding_model = embedding_model
        self.collection = collection
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search

        # Initialize re-ranker if enabled
        self.reranker: Optional[CrossEncoder] = None
        if use_reranking and reranker_model:
            try:
                logger.info(f"Loading re-ranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                logger.warning(f"Could not load re-ranker: {e}")
                self.use_reranking = False

        # BM25 will be initialized when documents are loaded
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []

        logger.info(
            f"SearchEngine initialized: hybrid={use_hybrid_search}, reranking={use_reranking}"
        )

    def initialize_bm25(self, documents: List[str]):
        """
        Initialize BM25 index for keyword search.

        Args:
            documents: List of document chunks
        """
        if not self.use_hybrid_search:
            logger.debug("BM25 disabled, skipping initialization")
            return

        if not documents:
            logger.warning("Cannot initialize BM25 with empty document list")
            self.bm25 = None
            self.bm25_corpus = []
            return

        # Tokenize documents (simple split by whitespace and lowercase)
        self.bm25_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info(f"BM25 index initialized with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform search over indexed documents.

        Uses hybrid search if enabled, otherwise pure semantic search.
        Optionally re-ranks results using cross-encoder.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Return empty list for invalid top_k
        if top_k <= 0:
            return []

        if self.use_hybrid_search and self.bm25 is not None:
            return self._hybrid_search(query, top_k)
        else:
            return self._semantic_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Perform pure semantic search using embeddings.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Query ChromaDB
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # Format results
        formatted = self._format_chromadb_results(results)

        # Apply reranking if enabled
        if self.use_reranking:
            formatted = self._rerank_results(query, formatted, top_k)

        return formatted

    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Final number of results to return

        Returns:
            List of merged and optionally re-ranked results
        """
        # Retrieve more candidates for re-ranking
        candidate_count = top_k * 2 if self.use_reranking else top_k

        # 1. Semantic search (70% weight)
        semantic_results = self._semantic_search(query, candidate_count)

        # 2. BM25 keyword search (30% weight)
        if self.bm25 is None:
            logger.warning("BM25 not initialized, falling back to semantic search")
            return semantic_results[:top_k]

        bm25_scores = self.bm25.get_scores(query.lower().split())

        # Get top BM25 results
        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_count]

        # Fetch documents for BM25 results
        all_docs = self.collection.get()
        bm25_results = []

        # Build BM25 results
        if all_docs["documents"] is not None:
            max_score = max(bm25_scores) if bm25_scores.size > 0 else 1.0

            for idx in bm25_top_indices:
                if idx < len(all_docs["documents"]):
                    bm25_results.append(
                        SearchResult(
                            document=all_docs["documents"][idx],
                            metadata=(all_docs["metadatas"][idx] if all_docs["metadatas"] else {}),
                            distance=1.0 - (bm25_scores[idx] / max_score),  # Normalize
                            id=all_docs["ids"][idx],
                            sources=["bm25"],
                        )
                    )

        # 3. Merge results with weighted scores
        merged = self._merge_results(semantic_results, bm25_results)

        # 4. Re-rank if enabled
        if self.use_reranking and self.reranker is not None:
            merged = self._rerank_results(query, merged, top_k)
        else:
            merged = merged[:top_k]

        return merged

    def _merge_results(
        self, semantic_results: List[SearchResult], bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Merge semantic and BM25 results with weighted scoring.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search

        Returns:
            Merged and sorted results
        """
        # Create a dict to combine scores for same documents
        combined: Dict[str, SearchResult] = {}

        # Add semantic results (70% weight)
        for result in semantic_results:
            score = (1.0 - result.distance) * 0.7  # Convert distance to score
            result.combined_score = score
            result.sources = ["semantic"]
            combined[result.id] = result

        # Add/merge BM25 results (30% weight)
        for result in bm25_results:
            bm25_score = (1.0 - result.distance) * 0.3

            if result.id in combined:
                existing = combined[result.id]
                existing.combined_score = (existing.combined_score or 0.0) + bm25_score
                existing.sources.append("bm25")
            else:
                result.combined_score = bm25_score
                combined[result.id] = result

        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.combined_score or 0.0,
            reverse=True,
        )

        return sorted_results

    def _rerank_results(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """
        Re-rank results using cross-encoder for improved accuracy.

        Args:
            query: Original query
            results: Results to re-rank
            top_k: Number of top results to return

        Returns:
            Re-ranked results
        """
        if not results or self.reranker is None:
            return results[:top_k]

        # Prepare query-document pairs
        pairs = [[query, result.document] for result in results]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Add scores to results
        for result, score in zip(results, rerank_scores):
            result.rerank_score = float(score)

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.rerank_score or 0.0, reverse=True)

        logger.debug(f"Re-ranked {len(results)} results, returning top {top_k}")
        return reranked[:top_k]

    def _format_chromadb_results(self, results: Any) -> List[SearchResult]:
        """
        Format ChromaDB query results into SearchResult objects.

        Args:
            results: Raw results from ChromaDB

        Returns:
            List of SearchResult objects (deduplicated by ID)
        """
        formatted_results = []
        seen_ids = set()

        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc_id = results["ids"][0][i]
                # Skip duplicates
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                result = SearchResult(
                    document=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                    id=doc_id,
                )
                # Populate sources list with source file name
                if "source" in result.metadata:
                    result.sources = [result.metadata["source"]]
                formatted_results.append(result)

        return formatted_results
