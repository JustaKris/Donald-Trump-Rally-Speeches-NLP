"""
Enhanced Topic Extraction Service with Semantic Clustering.

Provides advanced topic analysis using:
- Semantic clustering of keywords using embeddings
- Contextual snippet extraction
- AI-generated topic summaries via Gemini
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import get_settings
from ..services.llm.base import LLMProvider
from ..utils.text_preprocessing import clean_text, tokenize_text

logger = logging.getLogger(__name__)


class TopicExtractionService:
    """
    Enhanced topic extraction service with semantic clustering and AI analysis.

    Features:
    - Semantic grouping of related keywords using embeddings
    - Contextual snippet extraction showing topics in use
    - AI-generated interpretive summaries of main themes
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        llm_service: Optional[LLMProvider] = None,
    ):
        """
        Initialize topic extraction service.

        Args:
            embedding_model: Pre-initialized SentenceTransformer for embeddings
            llm_service: Optional Gemini LLM service for summaries
        """
        self.embedding_model = embedding_model or SentenceTransformer("all-mpnet-base-v2")
        self.llm_service = llm_service
        self.settings = get_settings()
        self.excluded_verbs = self.settings.get_excluded_verbs()

        logger.info("TopicExtractionService initialized")
        logger.debug(f"Excluded {len(self.excluded_verbs)} common verbs from topic extraction")

    def extract_topics_enhanced(
        self,
        text: str,
        top_n: int = 10,
        num_clusters: Optional[int] = None,
        snippets_per_topic: int = 3,
        snippet_context: int = 100,
    ) -> Dict[str, Any]:
        """
        Extract topics with clustering, snippets, and AI summary.

        Args:
            text: Input text to analyze
            top_n: Number of top keywords to extract before clustering
            num_clusters: Number of topic clusters (auto if None)
            snippets_per_topic: Number of example snippets per topic
            snippet_context: Characters of context around keyword in snippets

        Returns:
            Dict with:
                - clustered_topics: List of topic clusters with keywords
                - snippets: Example snippets for each cluster
                - summary: AI-generated interpretation
                - metadata: Analysis metadata
        """
        if not text.strip():
            return self._empty_response()

        logger.debug(f"Extracting topics from text (length: {len(text)} chars)")

        # Step 1: Extract top keywords with TF-IDF style scoring
        keywords_data = self._extract_keywords(text, top_n=top_n * 2)  # Get more for clustering

        if not keywords_data:
            return self._empty_response()

        # Step 2: Cluster keywords semantically
        clusters = self._cluster_keywords(keywords_data, num_clusters=num_clusters)

        # Step 3: Generate cluster labels using LLM
        labeled_clusters = self._label_clusters(clusters)

        # Step 4: Extract contextual snippets
        snippets = self._extract_snippets(
            text,
            labeled_clusters,
            snippets_per_topic=snippets_per_topic,
            context_chars=snippet_context,
        )

        # Step 5: Generate AI summary
        summary = self._generate_topic_summary(text, labeled_clusters)

        # Filter clusters by relevance threshold from config
        filtered_clusters = [
            c
            for c in labeled_clusters
            if c["avg_relevance"] >= self.settings.topic_relevance_threshold
        ]

        # If filtering removed too many, keep at least minimum from config
        if (
            len(filtered_clusters) < self.settings.topic_min_clusters
            and len(labeled_clusters) >= self.settings.topic_min_clusters
        ):
            filtered_clusters = labeled_clusters[: self.settings.topic_min_clusters]
        elif len(filtered_clusters) == 0:
            filtered_clusters = labeled_clusters

        # Limit to top_n clusters
        final_clusters = filtered_clusters[:top_n]

        # Update snippets to match filtered clusters
        final_snippets = snippets[: len(final_clusters)]

        return {
            "clustered_topics": final_clusters,
            "snippets": final_snippets,
            "summary": summary,
            "metadata": {
                "total_keywords": len(keywords_data),
                "num_clusters": len(final_clusters),
                "text_length": len(text),
                "has_ai_summary": summary is not None,
            },
        }

    def _extract_keywords(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract top keywords with frequency and positions.

        Args:
            text: Input text
            top_n: Number of keywords to extract

        Returns:
            List of keyword dicts with word, count, positions, and relevance
        """
        # Clean and tokenize
        cleaned = clean_text(text, remove_stopwords=True)
        tokens = tokenize_text(cleaned)

        # Filter alphabetic tokens and exclude common verbs
        tokens = [
            t for t in tokens if t.isalpha() and len(t) > 2 and t.lower() not in self.excluded_verbs
        ]

        if not tokens:
            return []

        # Count frequencies
        freq = Counter(tokens)
        top_words = freq.most_common(top_n)

        if not top_words:
            return []

        max_count = top_words[0][1]

        # Find positions of each keyword in original text (case-insensitive)
        keywords_with_positions = []
        text_lower = text.lower()

        for word, count in top_words:
            # Find all positions
            positions = [
                m.start() for m in re.finditer(r"\b" + re.escape(word) + r"\b", text_lower)
            ]

            keywords_with_positions.append(
                {
                    "word": word,
                    "count": count,
                    "relevance": round(count / max_count, 3),
                    "positions": positions[:10],  # Limit positions stored
                }
            )

        return keywords_with_positions

    def _cluster_keywords(
        self, keywords_data: List[Dict[str, Any]], num_clusters: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Cluster keywords using semantic embeddings.

        Args:
            keywords_data: List of keyword dicts
            num_clusters: Number of clusters (auto-determined if None)

        Returns:
            List of cluster dicts with keywords and avg relevance
        """
        if len(keywords_data) < 2:
            # Single cluster for insufficient data
            return [
                {
                    "keywords": keywords_data,
                    "avg_relevance": keywords_data[0]["relevance"] if keywords_data else 0,
                    "total_mentions": sum(kw["count"] for kw in keywords_data),
                }
            ]

        # Generate embeddings for keywords
        words = [kw["word"] for kw in keywords_data]
        logger.debug(f"Generating embeddings for {len(words)} keywords")
        embeddings = self.embedding_model.encode(words, show_progress_bar=False)

        # Auto-determine number of clusters (between 3 and 6)
        if num_clusters is None:
            num_clusters = min(max(3, len(keywords_data) // 3), 6)

        num_clusters = min(num_clusters, len(keywords_data))

        # Cluster using KMeans
        logger.debug(f"Clustering into {num_clusters} groups")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group keywords by cluster
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters_dict[label].append(keywords_data[idx])

        # Convert to sorted list
        clusters = []
        for cluster_id, keywords in clusters_dict.items():
            avg_relevance = np.mean([kw["relevance"] for kw in keywords])
            total_mentions = sum(kw["count"] for kw in keywords)

            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "keywords": sorted(keywords, key=lambda x: x["relevance"], reverse=True),
                    "avg_relevance": float(avg_relevance),
                    "total_mentions": int(total_mentions),
                }
            )

        # Sort clusters by total mentions (most important first)
        clusters.sort(key=lambda x: x["total_mentions"], reverse=True)

        return clusters

    def _label_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate semantic labels for clusters using LLM.

        Args:
            clusters: List of cluster dicts

        Returns:
            Clusters with added 'label' field
        """
        if not self.llm_service:
            # Fallback: use top keyword as label
            for cluster in clusters:
                top_keyword = cluster["keywords"][0]["word"]
                cluster["label"] = top_keyword.title()
            return clusters

        # Use LLM to generate meaningful labels
        for cluster in clusters:
            keywords = [kw["word"] for kw in cluster["keywords"][:5]]  # Top 5 keywords
            label = self._generate_cluster_label(keywords)
            cluster["label"] = label

        return clusters

    def _generate_cluster_label(self, keywords: List[str]) -> str:
        """
        Generate a short, descriptive label for a keyword cluster.

        Args:
            keywords: List of keywords in the cluster

        Returns:
            Short label string (2-4 words)
        """
        if not self.llm_service:
            return keywords[0].title()

        prompt = f"""Given these related keywords: {', '.join(keywords)}

Generate a concise 2-4 word label that captures the main theme these keywords represent.

Examples:
- Keywords: economy, jobs, employment, market → "Economic Policy"
- Keywords: border, wall, immigration, security → "Border Security"
- Keywords: media, news, fake, press → "Media Criticism"

Respond with ONLY the label, nothing else."""

        try:
            response = self.llm_service.generate_content(prompt)
            # Check if response is valid (not blocked by safety filters)
            if not response or not hasattr(response, "text") or not response.text:
                logger.debug("LLM response blocked or empty for cluster label, using fallback")
                return keywords[0].title()

            label = response.text.strip().strip('"').strip("'")
            # Limit length
            if len(label) > 50:
                label = keywords[0].title()
            return label
        except Exception as e:
            logger.warning(f"Failed to generate cluster label: {e}")
            return keywords[0].title()

    def _extract_snippets(
        self,
        text: str,
        clusters: List[Dict[str, Any]],
        snippets_per_topic: int = 3,
        context_chars: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Extract contextual snippets showing keywords in use.

        Args:
            text: Original text
            clusters: Labeled clusters
            snippets_per_topic: Number of snippets per cluster
            context_chars: Characters of context around keyword

        Returns:
            List of snippet dicts for each cluster
        """
        cluster_snippets = []

        for cluster in clusters:
            # Get all positions from all keywords in cluster
            all_positions = []
            for kw in cluster["keywords"]:
                for pos in kw["positions"]:
                    all_positions.append((pos, kw["word"]))

            # Sort by position and deduplicate nearby ones
            all_positions.sort()
            selected_positions = self._deduplicate_positions(
                all_positions, min_distance=context_chars * 2
            )

            # Extract snippets
            snippets = []
            for pos, keyword in selected_positions[:snippets_per_topic]:
                snippet = self._extract_snippet(text, pos, keyword, context_chars)
                if snippet:
                    snippets.append(snippet)

            cluster_snippets.append(
                {
                    "label": cluster["label"],
                    "snippets": snippets,
                    "keyword_count": len(cluster["keywords"]),
                }
            )

        return cluster_snippets

    def _deduplicate_positions(
        self, positions: List[Tuple[int, str]], min_distance: int = 200
    ) -> List[Tuple[int, str]]:
        """
        Remove positions that are too close to each other.

        Args:
            positions: List of (position, keyword) tuples
            min_distance: Minimum distance between selected positions

        Returns:
            Filtered list of positions
        """
        if not positions:
            return []

        selected = [positions[0]]
        for pos, keyword in positions[1:]:
            if pos - selected[-1][0] >= min_distance:
                selected.append((pos, keyword))

        return selected

    def _extract_snippet(
        self, text: str, position: int, keyword: str, context_chars: int = 100
    ) -> Optional[str]:
        """
        Extract a text snippet around a keyword occurrence.

        Args:
            text: Full text
            position: Position of keyword
            keyword: The keyword to highlight
            context_chars: Characters of context on each side

        Returns:
            Snippet string with keyword highlighted or None
        """
        # Calculate snippet boundaries
        start = max(0, position - context_chars)
        end = min(len(text), position + len(keyword) + context_chars)

        # Extract snippet
        snippet = text[start:end]

        # Clean up - try to start/end at sentence boundaries
        if start > 0:
            # Find first sentence start or space
            first_period = snippet.find(". ")
            if first_period != -1 and first_period < context_chars // 2:
                snippet = snippet[first_period + 2 :]

        if end < len(text):
            # Find last sentence end
            last_period = snippet.rfind(". ")
            if last_period != -1 and last_period > len(snippet) - context_chars // 2:
                snippet = snippet[: last_period + 1]

        # Highlight keyword (case-insensitive)
        snippet = re.sub(
            r"\b(" + re.escape(keyword) + r")\b", r"**\1**", snippet, flags=re.IGNORECASE, count=1
        )

        return snippet.strip()

    def _generate_topic_summary(self, text: str, clusters: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate AI interpretation of main topics.

        Args:
            text: Original text
            clusters: Labeled clusters

        Returns:
            Summary string or None if LLM unavailable
        """
        if not self.llm_service or not clusters:
            return None

        # Build topic overview
        topic_list = []
        for cluster in clusters[:5]:  # Top 5 clusters
            label = cluster["label"]
            keywords = [kw["word"] for kw in cluster["keywords"][:3]]
            mentions = cluster["total_mentions"]
            topic_list.append(f"- {label}: {', '.join(keywords)} ({mentions} mentions)")

        topics_text = "\n".join(topic_list)

        # Get a sample of the text
        text_sample = text[:2000] if len(text) > 2000 else text

        prompt = f"""Analyze these extracted topics and provide a brief 2-3 sentence summary of the main themes in the text.

EXTRACTED TOPICS:
{topics_text}

TEXT SAMPLE:
{text_sample}

Provide a concise analytical summary that:
1. Identifies the 2-3 dominant themes
2. Notes any interesting patterns or emphasis
3. Stays objective and factual

Summary:"""

        try:
            logger.debug("Generating topic summary with LLM")
            response = self.llm_service.generate_content(prompt)
            # Check if response is valid (not blocked by safety filters)
            if not response or not hasattr(response, "text") or not response.text:
                logger.debug("LLM response blocked or empty for topic summary")
                return None

            summary = response.text.strip()
            logger.info(f"Generated topic summary (length: {len(summary)} chars)")
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate topic summary: {e}")
            return None

    def _empty_response(self) -> Dict[str, Any]:
        """Return empty response structure."""
        return {
            "clustered_topics": [],
            "snippets": [],
            "summary": None,
            "metadata": {
                "total_keywords": 0,
                "num_clusters": 0,
                "text_length": 0,
                "has_ai_summary": False,
            },
        }
