"""
RAG (Retrieval-Augmented Generation) Service for NLP Text Analysis API.

Provides semantic search and question-answering capabilities over text documents
using ChromaDB for vector storage, sentence-transformers for embeddings,
Gemini for answer generation, and hybrid search for improved retrieval.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..services.llm_service import GeminiLLM

# Disable ChromaDB telemetry to avoid annoying warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for Retrieval-Augmented Generation over text documents.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings
    to enable semantic search and context retrieval.
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
        Initialize RAG service with ChromaDB, sentence-transformers, and optional LLM.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for ChromaDB persistence
            embedding_model: HuggingFace model for embeddings (default: all-mpnet-base-v2)
            reranker_model: Cross-encoder model for re-ranking
            chunk_size: Maximum size of text chunks in characters (~512-768 tokens)
            chunk_overlap: Overlap between chunks in characters (~100-150 tokens)
            llm_service: Optional LLM service instance (if None, uses extraction-based answers)
            use_reranking: Use cross-encoder for re-ranking results
            use_hybrid_search: Combine semantic and keyword search
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search

        logger.debug(f"Initializing RAG service: collection={collection_name}")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize re-ranker if enabled
        self.reranker = None
        if use_reranking:
            try:
                logger.info(f"Loading re-ranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                logger.warning(f"Could not load re-ranker: {e}")
                self.use_reranking = False

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

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # BM25 will be initialized when documents are loaded
        self.bm25 = None
        self.bm25_corpus: List[List[str]] = []  # Store tokenized documents for BM25

    def load_documents(self, data_dir: str = "data/Donald Trump Rally Speeches") -> int:
        """
        Load and index all text documents from a directory.

        Args:
            data_dir: Directory containing text files

        Returns:
            Number of documents indexed
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        documents_added = 0
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for file_path in data_path.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split document into chunks
                chunks = self.text_splitter.split_text(content)

                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_path.stem}_chunk_{i}"
                    all_chunks.append(chunk)
                    all_metadatas.append(
                        {
                            "source": file_path.name,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )
                    all_ids.append(chunk_id)

                documents_added += 1

            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue

        # Add all chunks to ChromaDB in batch (skip duplicates)
        if all_chunks:
            # Check for existing IDs to avoid duplicates
            try:
                existing_data = self.collection.get()
                existing_ids = set(existing_data.get("ids", []))
                logger.debug(f"Found {len(existing_ids)} existing chunks in collection")
            except Exception as e:
                logger.warning(f"Could not fetch existing IDs: {e}")
                existing_ids = set()

            # Filter out chunks that already exist
            new_chunks = []
            new_metadatas = []
            new_ids = []

            for chunk, metadata, chunk_id in zip(all_chunks, all_metadatas, all_ids):
                if chunk_id not in existing_ids:
                    new_chunks.append(chunk)
                    new_metadatas.append(metadata)
                    new_ids.append(chunk_id)

            if new_chunks:
                logger.info(
                    f"Adding {len(new_chunks)} new chunks to collection (skipped {len(all_chunks) - len(new_chunks)} duplicates)"
                )

                # Generate embeddings only for new chunks
                embeddings = self.embedding_model.encode(
                    new_chunks, show_progress_bar=True, convert_to_numpy=True
                ).tolist()

                # Add to collection
                self.collection.add(
                    documents=new_chunks,
                    embeddings=embeddings,
                    metadatas=new_metadatas,  # type: ignore[arg-type]
                    ids=new_ids,
                )
            else:
                logger.info("All chunks already exist in collection, skipping add operation")

            # Initialize BM25 for hybrid search (use all chunks including existing)
            if self.use_hybrid_search:
                # Fetch all documents from collection for BM25
                all_collection_data = self.collection.get()
                all_collection_docs = all_collection_data.get("documents", [])
                if all_collection_docs:
                    self._initialize_bm25(all_collection_docs)

        return documents_added

    def _initialize_bm25(self, documents: List[str]):
        """
        Initialize BM25 index for keyword search.

        Args:
            documents: List of document chunks
        """
        # Tokenize documents (simple split by whitespace and lowercase)
        self.bm25_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search over indexed documents with optional re-ranking.

        Combines semantic search (ChromaDB) with keyword search (BM25) and
        optionally re-ranks results using a cross-encoder for improved accuracy.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with documents, metadata, and distances
        """
        if self.use_hybrid_search and self.bm25 is not None:
            return self._hybrid_search(query, top_k)
        else:
            return self._semantic_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform pure semantic search using embeddings.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Query ChromaDB
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # Format results
        return self._format_search_results(results)

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Final number of results to return

        Returns:
            List of merged and re-ranked results
        """
        # Retrieve more candidates for re-ranking
        candidate_count = top_k * 2 if self.use_reranking else top_k

        # 1. Semantic search (70% weight)
        semantic_results = self._semantic_search(query, candidate_count)

        # 2. BM25 keyword search (30% weight)
        if self.bm25 is None:
            # If BM25 not initialized, return semantic results only
            return semantic_results[:top_k]

        bm25_scores = self.bm25.get_scores(query.lower().split())

        # Get top BM25 results
        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_count]

        # Fetch documents for BM25 results
        all_docs = self.collection.get()
        bm25_results = []

        # Type guard: check if documents exist
        if all_docs["documents"] is not None:
            for idx in bm25_top_indices:
                if idx < len(all_docs["documents"]):
                    bm25_results.append(
                        {
                            "document": all_docs["documents"][idx],
                            "metadata": all_docs["metadatas"][idx] if all_docs["metadatas"] else {},
                            "distance": 1.0
                            - (bm25_scores[idx] / max(bm25_scores + [1])),  # Normalize
                            "id": all_docs["ids"][idx],
                            "source": "bm25",
                        }
                    )

        # 3. Merge results with weighted scores
        merged = self._merge_results(semantic_results, bm25_results)

        # 4. Re-rank if enabled
        if self.use_reranking and self.reranker is not None:
            merged = self._rerank_results(query, merged, top_k)
        else:
            merged = merged[:top_k]

        return merged

    def _merge_results(self, semantic_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        Merge semantic and BM25 results with weighted scoring.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search

        Returns:
            Merged and sorted results
        """
        # Create a dict to combine scores for same documents
        combined = {}

        # Add semantic results (70% weight)
        for result in semantic_results:
            doc_id = result["id"]
            score = (1.0 - result["distance"]) * 0.7  # Convert distance to score
            combined[doc_id] = {**result, "combined_score": score, "sources": ["semantic"]}

        # Add/merge BM25 results (30% weight)
        for result in bm25_results:
            doc_id = result["id"]
            bm25_score = (1.0 - result["distance"]) * 0.3

            if doc_id in combined:
                combined[doc_id]["combined_score"] += bm25_score
                combined[doc_id]["sources"].append("bm25")
            else:
                combined[doc_id] = {**result, "combined_score": bm25_score, "sources": ["bm25"]}

        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

        return sorted_results

    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-rank results using cross-encoder for improved accuracy.

        Args:
            query: Original query
            results: Results to re-rank
            top_k: Number of top results to return

        Returns:
            Re-ranked results
        """
        if not results:
            return results

        # Type guard: reranker should be initialized if this method is called
        if self.reranker is None:
            return results[:top_k]

        # Prepare query-document pairs
        pairs = [[query, result["document"]] for result in results]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Add scores to results
        for result, score in zip(results, rerank_scores):
            result["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked[:top_k]

    def _format_search_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        Format ChromaDB query results into standard format.

        Args:
            results: Raw results from ChromaDB (QueryResult object)

        Returns:
            Formatted results list
        """
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "id": results["ids"][0][i],
                    }
                )

        return formatted_results

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG with optional LLM-powered generation.

        If use_llm is enabled and Gemini is available, generates synthesized
        answers with citations. Otherwise, falls back to extraction-based answering.

        Args:
            question: Question to answer
            top_k: Number of context chunks to retrieve (default: 5 for better evidence)

        Returns:
            Answer with sources and metadata matching API format
        """
        # Detect entities in question for entity-aware retrieval
        entities = self._extract_entities(question)

        # Retrieve relevant context
        search_results = self.search(question, top_k=top_k)

        if not search_results:
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "context": [],
                "confidence": "low",
                "confidence_score": 0.0,
                "sources": [],
            }

        # Extract context for return format
        context_items = [
            {
                "text": result["document"],
                "source": result["metadata"].get("source", "Unknown"),
                "chunk_index": result["metadata"].get("chunk_index", 0),
            }
            for result in search_results
        ]

        # Get unique sources
        sources = list(set(item["source"] for item in context_items))

        # Use LLM-powered generation if available
        if self.llm is not None:
            try:
                # Prepare context for LLM with flattened metadata
                context_chunks = [
                    {
                        "text": result["document"],
                        "source": result["metadata"].get("source", "Unknown"),
                        "chunk_index": result["metadata"].get("chunk_index", 0),
                        "score": result.get(
                            "rerank_score",
                            result.get("combined_score", 1.0 - result.get("distance", 0.0)),
                        ),
                    }
                    for result in search_results
                ]

                # Generate answer using LLM with entity awareness
                logger.debug(
                    f"Generating answer with {len(context_chunks)} context chunks using LLM"
                )
                llm_response = self.llm.generate_answer(question, context_chunks, entities=entities)
                answer_text = llm_response.get("answer", "Unable to generate answer.")
                logger.debug(f"LLM response received (length: {len(answer_text)} chars)")

                # Calculate enhanced confidence with multiple factors
                confidence_data = self._calculate_confidence(
                    question, context_chunks, search_results
                )

                # Get entity statistics if entities were detected
                entity_stats = None
                if entities:
                    entity_stats = self._get_entity_statistics(entities)

                response = {
                    "answer": answer_text,
                    "context": context_items,
                    "confidence": confidence_data["level"],
                    "confidence_score": confidence_data["score"],
                    "confidence_explanation": confidence_data["explanation"],
                    "confidence_factors": confidence_data["factors"],
                    "sources": sources,
                }

                # Add entity statistics if available
                if entity_stats:
                    response["entity_statistics"] = entity_stats

                return response

            except Exception as e:
                # Fall back to extraction if LLM fails
                logger.error(
                    f"LLM generation failed, falling back to extraction: {e}", exc_info=True
                )
                return self._extraction_based_answer_formatted(
                    question, search_results, context_items, sources
                )
        else:
            # Use extraction-based answering
            logger.debug("Using extraction-based answering (no LLM)")
            return self._extraction_based_answer_formatted(
                question, search_results, context_items, sources
            )

    def _analyze_entity_sentiment(self, entity: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of text chunks mentioning an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity

        Returns:
            Dict with average sentiment and classification
        """
        # Import sentiment analyzer only if needed
        from .sentiment_service import SentimentAnalyzer

        try:
            # Initialize analyzer if not already cached
            if not hasattr(self, "_sentiment_analyzer"):
                self._sentiment_analyzer = SentimentAnalyzer()

            # Analyze sentiment for each context
            sentiments = []
            for context in contexts[:50]:  # Limit to 50 contexts for performance
                try:
                    result = self._sentiment_analyzer.analyze_sentiment(context)
                    # Convert to numeric score: positive=1, neutral=0, negative=-1
                    if result.get("dominant") == "positive":
                        sentiments.append(result.get("positive", 0))
                    elif result.get("dominant") == "negative":
                        sentiments.append(-result.get("negative", 0))
                    else:
                        sentiments.append(0)
                except Exception:
                    # Skip failed analyses
                    continue

            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)

                # Classify overall sentiment
                if avg_sentiment > 0.2:
                    classification = "Positive"
                elif avg_sentiment < -0.2:
                    classification = "Negative"
                else:
                    classification = "Neutral"

                return {
                    "average_score": round(avg_sentiment, 2),
                    "classification": classification,
                    "sample_size": len(sentiments),
                }

        except Exception as e:
            # Fallback if sentiment analysis fails
            logger.warning(f"Sentiment analysis failed: {e}")

        return {
            "average_score": 0.0,
            "classification": "Unknown",
            "sample_size": 0,
        }

    def _find_entity_associations(
        self, entity: str, contexts: List[str], top_n: int = 5
    ) -> List[str]:
        """
        Find most common terms associated with an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity
            top_n: Number of top associations to return

        Returns:
            List of most common associated terms
        """
        import re
        from collections import Counter

        # Common stopwords to exclude
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "just",
            "now",
            "said",
            "says",
            "going",
            "go",
            "get",
            "got",
            "know",
            "think",
            "see",
            "make",
            "made",
            "like",
            "ve",
            "re",
            "ll",
            "m",
            "don",
            "didn",
            "wasn",
        }

        # Extract words from contexts
        words = []
        entity_lower = entity.lower()

        for context in contexts[:50]:  # Limit for performance
            # Find sentences or windows containing the entity
            context_lower = context.lower()
            if entity_lower in context_lower:
                # Extract words near the entity (window-based approach)
                words_in_context = re.findall(r"\b[a-z]{3,}\b", context_lower)
                words.extend(
                    [w for w in words_in_context if w not in stopwords and w != entity_lower]
                )

        # Count and return most common
        if words:
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(top_n)]

        return []

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential named entities from text using simple heuristics.

        Uses capitalized words as entity candidates. For production,
        consider using a proper NER model.

        Args:
            text: Text to extract entities from

        Returns:
            List of potential entity names
        """
        words = text.split()
        # Find capitalized words that aren't at sentence start
        entities = []
        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = word.strip(".,!?;:\"'")
            # Check if capitalized and not first word, and longer than 2 chars
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                # Skip common question words
                if clean_word.lower() not in [
                    "what",
                    "when",
                    "where",
                    "who",
                    "why",
                    "how",
                    "which",
                ]:
                    entities.append(clean_word)

        return list(set(entities))  # Remove duplicates

    def _get_entity_statistics(
        self,
        entities: List[str],
        include_sentiment: bool = False,
        include_associations: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about entity mentions across the corpus.

        Args:
            entities: List of entity names to analyze
            include_sentiment: Calculate sentiment for entity mentions
            include_associations: Find commonly associated terms

        Returns:
            Dict with entity statistics including frequency, sentiment, and associations
        """
        if not entities:
            return {}

        stats = {}
        all_docs = self.collection.get(include=[IncludeEnum.documents, IncludeEnum.metadatas])

        if not all_docs["documents"]:
            return {}

        # Import for text analysis
        import re
        from collections import Counter

        for entity in entities:
            entity_lower = entity.lower()
            mentions = 0
            speeches_with_entity = set()
            total_chars = 0
            entity_contexts: List[str] = []  # Store contexts for sentiment and association analysis

            for i, doc in enumerate(all_docs["documents"]):
                doc_lower = doc.lower()
                count = doc_lower.count(entity_lower)
                if count > 0:
                    mentions += count
                    total_chars += len(doc)
                    if all_docs["metadatas"] and i < len(all_docs["metadatas"]):
                        source = all_docs["metadatas"][i].get("source", "unknown")
                        speeches_with_entity.add(source)

                    # Store context for analysis (limit to avoid memory issues)
                    if len(entity_contexts) < 100:
                        entity_contexts.append(doc)

            if mentions > 0:
                # Calculate percentage of corpus
                total_corpus_chars = sum(len(doc) for doc in all_docs["documents"])
                percentage = (
                    (total_chars / total_corpus_chars * 100) if total_corpus_chars > 0 else 0
                )

                entity_stats = {
                    "mention_count": mentions,
                    "speech_count": len(speeches_with_entity),
                    "corpus_percentage": round(percentage, 2),
                    "speeches": sorted(list(speeches_with_entity))[:10],  # Limit to first 10
                }

                # Add sentiment analysis
                if include_sentiment and entity_contexts:
                    sentiment_data = self._analyze_entity_sentiment(entity, entity_contexts)
                    entity_stats["sentiment"] = sentiment_data

                # Add associated terms
                if include_associations and entity_contexts:
                    associations = self._find_entity_associations(entity, entity_contexts)
                    entity_stats["associated_terms"] = associations

                stats[entity] = entity_stats

        return stats

    def _generate_confidence_explanation(
        self,
        level: str,
        score: float,
        retrieval_score: float,
        consistency: float,
        chunk_count: int,
        entity_coverage: float,
        entities: Optional[List[str]] = None,
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

    def _calculate_confidence(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate confidence using multiple factors for more accurate assessment.

        Factors considered:
        1. Average retrieval score (semantic similarity)
        2. Score consistency (low variance = more confident)
        3. Number of supporting chunks
        4. Entity mention frequency (if applicable)

        Args:
            question: Original question
            context_chunks: Retrieved context with scores
            search_results: Raw search results

        Returns:
            Dict with 'score' (0-1), 'level' (high/medium/low), and 'factors'
        """
        if not context_chunks:
            return {
                "score": 0.0,
                "level": "low",
                "factors": {"reason": "No relevant context found"},
            }

        # Factor 1: Average retrieval score (40% weight)
        scores = [c.get("score", 0.0) for c in context_chunks]
        avg_score = sum(scores) / len(scores)
        score_factor = avg_score * 0.4

        # Factor 2: Score consistency - low variance is good (25% weight)
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0, 1 - variance)  # Convert variance to consistency score
        else:
            consistency = 1.0
        consistency_factor = consistency * 0.25

        # Factor 3: Number of supporting chunks (20% weight)
        # More chunks with good scores = higher confidence
        chunk_count_score = min(len(context_chunks) / 10.0, 1.0)  # Normalize to max 10 chunks
        chunk_factor = chunk_count_score * 0.2

        # Factor 4: Entity mention frequency (15% weight)
        # Extract potential entities from question (simple heuristic: capitalized words)
        question_words = question.split()
        entities = [w for w in question_words if w[0].isupper() and len(w) > 2]

        if entities:
            # Count how many chunks mention the entity
            entity_mentions = 0
            for chunk in context_chunks:
                text = chunk.get("text", "").lower()
                if any(entity.lower() in text for entity in entities):
                    entity_mentions += 1
            entity_coverage = entity_mentions / len(context_chunks) if context_chunks else 0
        else:
            entity_coverage = 0.5  # Neutral if no entities detected
        entity_factor = entity_coverage * 0.15

        # Combine all factors
        final_score = score_factor + consistency_factor + chunk_factor + entity_factor

        # Determine confidence level
        if final_score >= 0.7:
            level = "high"
        elif final_score >= 0.4:
            level = "medium"
        else:
            level = "low"

        # Generate human-readable explanation
        explanation = self._generate_confidence_explanation(
            level,
            final_score,
            avg_score,
            consistency,
            len(context_chunks),
            entity_coverage,
            entities,
        )

        return {
            "score": round(final_score, 3),
            "level": level,
            "explanation": explanation,
            "factors": {
                "retrieval_score": round(avg_score, 3),
                "consistency": round(consistency, 3),
                "chunk_coverage": len(context_chunks),
                "entity_coverage": round(entity_coverage, 3) if entities else None,
            },
        }

    def _extraction_based_answer_formatted(
        self,
        question: str,
        search_results: List[Dict],
        context_items: List[Dict],
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Generate answer using extraction in API-compatible format.

        Args:
            question: Question to answer
            search_results: Retrieved context documents
            context_items: Formatted context for API response
            sources: List of unique source documents

        Returns:
            Answer dictionary in API format
        """
        # Extract relevant sentences
        answer_sentences = []
        for result in search_results:
            # Split into sentences and find most relevant
            sentences = result["document"].split(".")
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Avoid short fragments
                    answer_sentences.append(sentence.strip())
                if len(answer_sentences) >= 3:
                    break
            if len(answer_sentences) >= 3:
                break

        # Combine into answer
        answer = (
            ". ".join(answer_sentences[:3]) + "."
            if answer_sentences
            else "Based on the documents: " + search_results[0]["document"][:300] + "..."
        )

        # Calculate enhanced confidence
        context_chunks = [
            {
                "text": result["document"],
                "score": result.get(
                    "rerank_score",
                    result.get("combined_score", 1.0 - result.get("distance", 0.0)),
                ),
            }
            for result in search_results
        ]
        confidence_data = self._calculate_confidence(question, context_chunks, search_results)

        return {
            "answer": answer,
            "context": context_items,
            "confidence": confidence_data["level"],
            "confidence_score": confidence_data["score"],
            "confidence_explanation": confidence_data["explanation"],
            "confidence_factors": confidence_data["factors"],
            "sources": sources,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed collection.

        Returns:
            Dictionary with collection statistics
        """
        collection_count = self.collection.count()

        # Get unique sources
        all_metadatas = self.collection.get(include=[IncludeEnum.metadatas])
        unique_sources = set()
        if all_metadatas["metadatas"]:
            unique_sources = set(
                meta.get("source", "unknown") for meta in all_metadatas["metadatas"]
            )

        return {
            "collection_name": self.collection_name,
            "total_chunks": collection_count,
            "unique_sources": len(unique_sources),
            "sources": sorted(list(unique_sources)),
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Text documents for semantic search"},
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
