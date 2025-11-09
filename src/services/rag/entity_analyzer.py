"""
Entity analyzer component for RAG service.

Handles entity extraction, statistics gathering, sentiment analysis,
and finding associated terms for named entities in the corpus.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from chromadb.api.types import IncludeEnum

from .models import EntitySentiment, EntityStatistics

logger = logging.getLogger(__name__)


class EntityAnalyzer:
    """
    Analyzes named entities in text and corpus.

    Provides entity extraction, frequency statistics, sentiment analysis,
    and term association discovery.
    """

    def __init__(self, collection=None, sentiment_analyzer=None):
        """
        Initialize entity analyzer.

        Args:
            collection: ChromaDB collection for corpus-wide analysis
            sentiment_analyzer: Optional sentiment analyzer for entity sentiment
        """
        self.collection = collection
        self.sentiment_analyzer = sentiment_analyzer

        # Common stopwords to exclude from associations
        self.stopwords = {
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

        logger.debug("EntityAnalyzer initialized")

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract potential named entities from text using simple heuristics.

        Uses capitalized words as entity candidates. For production,
        consider using a proper NER model (spaCy, Hugging Face).

        Args:
            text: Text to extract entities from

        Returns:
            List of potential entity names
        """
        words = text.split()
        entities = []

        for word in words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:\"'")

            # Check if capitalized and longer than 2 chars
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

        # Remove duplicates and return
        unique_entities = list(set(entities))
        logger.debug(f"Extracted {len(unique_entities)} entities from text")
        return unique_entities

    def get_statistics(
        self,
        entities: List[str],
        include_sentiment: bool = False,
        include_associations: bool = True,
    ) -> Dict[str, EntityStatistics]:
        """
        Get comprehensive statistics about entity mentions across the corpus.

        Args:
            entities: List of entity names to analyze
            include_sentiment: Calculate sentiment for entity mentions
            include_associations: Find commonly associated terms

        Returns:
            Dict mapping entity names to their statistics
        """
        if not entities or self.collection is None:
            return {}

        stats = {}
        all_docs = self.collection.get(include=[IncludeEnum.documents, IncludeEnum.metadatas])

        if not all_docs["documents"]:
            return {}

        for entity in entities:
            entity_stats = self._analyze_single_entity(
                entity, all_docs, include_sentiment, include_associations
            )
            if entity_stats:
                stats[entity] = entity_stats

        logger.info(f"Generated statistics for {len(stats)} entities")
        return stats

    def _analyze_single_entity(
        self,
        entity: str,
        all_docs: Dict[str, Any],
        include_sentiment: bool,
        include_associations: bool,
    ) -> Optional[EntityStatistics]:
        """
        Analyze a single entity across the corpus.

        Args:
            entity: Entity name
            all_docs: All documents from collection
            include_sentiment: Include sentiment analysis
            include_associations: Include term associations

        Returns:
            EntityStatistics if entity found, None otherwise
        """
        entity_lower = entity.lower()
        mentions = 0
        speeches_with_entity = set()
        total_chars = 0
        entity_contexts: List[str] = []

        # Count mentions across all documents
        for i, doc in enumerate(all_docs["documents"]):
            doc_lower = doc.lower()
            count = doc_lower.count(entity_lower)

            if count > 0:
                mentions += count
                total_chars += len(doc)

                # Track which speeches mention this entity
                if all_docs["metadatas"] and i < len(all_docs["metadatas"]):
                    source = all_docs["metadatas"][i].get("source", "unknown")
                    speeches_with_entity.add(source)

                # Store context for analysis (limit to avoid memory issues)
                if len(entity_contexts) < 100:
                    entity_contexts.append(doc)

        # Return None if entity not found
        if mentions == 0:
            return None

        # Calculate percentage of corpus (based on number of speeches)
        total_doc_count = self.collection.count() if self.collection else len(all_docs["documents"])
        speech_count = len(speeches_with_entity)
        percentage = (speech_count / total_doc_count * 100) if total_doc_count > 0 else 0

        # Build base statistics
        entity_data = {
            "mention_count": mentions,
            "speech_count": len(speeches_with_entity),
            "corpus_percentage": round(percentage, 2),
            "speeches": sorted(list(speeches_with_entity))[:10],  # Limit to first 10
        }

        # Add sentiment analysis if requested
        if include_sentiment and entity_contexts:
            sentiment = self.analyze_sentiment(entity, entity_contexts)
            entity_data["sentiment"] = sentiment

        # Add associated terms if requested
        if include_associations and entity_contexts:
            associations = self.find_associations(entity, entity_contexts)
            entity_data["associated_terms"] = associations

        return EntityStatistics(**entity_data)

    def analyze_sentiment(self, entity: str, contexts: List[str]) -> EntitySentiment:
        """
        Analyze sentiment of text chunks mentioning an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity

        Returns:
            EntitySentiment with average score and classification
        """
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return EntitySentiment(
                average_score=0.0,
                classification="Unknown",
                sample_size=0,
            )

        try:
            # Analyze sentiment for each context
            sentiments = []

            for context in contexts[:50]:  # Limit to 50 contexts for performance
                try:
                    result = self.sentiment_analyzer.analyze_sentiment(context)

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

                return EntitySentiment(
                    average_score=round(avg_sentiment, 2),
                    classification=classification,
                    sample_size=len(sentiments),
                )

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {entity}: {e}")

        return EntitySentiment(
            average_score=0.0,
            classification="Unknown",
            sample_size=0,
        )

    def find_associations(self, entity: str, contexts: List[str], top_n: int = 5) -> List[str]:
        """
        Find most common terms associated with an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity
            top_n: Number of top associations to return

        Returns:
            List of most common associated terms
        """
        entity_lower = entity.lower()
        words = []

        for context in contexts[:50]:  # Limit for performance
            context_lower = context.lower()

            if entity_lower in context_lower:
                # Extract words near the entity (window-based approach)
                words_in_context = re.findall(r"\b[a-z]{3,}\b", context_lower)
                words.extend(
                    [w for w in words_in_context if w not in self.stopwords and w != entity_lower]
                )

        # Count and return most common
        if words:
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(top_n)]

        return []
