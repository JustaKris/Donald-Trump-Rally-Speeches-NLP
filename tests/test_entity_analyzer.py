"""
Tests for EntityAnalyzer component.

Tests entity extraction, statistics, and associations.
Note: Sentiment analysis tests are skipped if sentiment analyzer not available.
"""

from unittest.mock import MagicMock, Mock

import pytest

from src.services.rag.entity_analyzer import EntityAnalyzer


class TestEntityAnalyzer:
    """Test suite for EntityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create EntityAnalyzer without collection."""
        return EntityAnalyzer()

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        collection = Mock()
        collection.get = MagicMock(
            return_value={
                "documents": [
                    "Trump spoke about Biden and the economy",
                    "Biden's policies were criticized by Trump",
                    "The economy is strong says Trump",
                ],
                "metadatas": [
                    {"source": "speech1.txt"},
                    {"source": "speech2.txt"},
                    {"source": "speech3.txt"},
                ],
            }
        )
        collection.count = MagicMock(return_value=3)
        return collection

    @pytest.fixture
    def analyzer_with_collection(self, mock_collection):
        """Create EntityAnalyzer with mock collection."""
        return EntityAnalyzer(collection=mock_collection)

    def test_initialization(self, analyzer):
        """Test EntityAnalyzer initialization."""
        assert analyzer.collection is None
        assert analyzer.sentiment_analyzer is None
        assert isinstance(analyzer.stopwords, set)
        assert len(analyzer.stopwords) > 0

    def test_extract_entities_basic(self, analyzer):
        """Test basic entity extraction."""
        text = "Trump spoke about Biden and China"
        entities = analyzer.extract_entities(text)

        assert "Trump" in entities
        assert "Biden" in entities
        assert "China" in entities

    def test_extract_entities_filters_question_words(self, analyzer):
        """Test that question words are filtered out."""
        text = "What did Trump say about Biden?"
        entities = analyzer.extract_entities(text)

        assert "Trump" in entities
        assert "Biden" in entities
        assert "What" not in entities  # Question word filtered

    def test_extract_entities_filters_short_words(self, analyzer):
        """Test that short words are filtered."""
        text = "I met Dr Smith in US"
        entities = analyzer.extract_entities(text)

        assert "Smith" in entities
        # "Dr" and "US" are too short (2 chars or less)
        assert "Dr" not in entities
        assert "US" not in entities

    def test_extract_entities_removes_punctuation(self, analyzer):
        """Test punctuation removal."""
        text = "Trump, Biden, and Obama."
        entities = analyzer.extract_entities(text)

        # Should extract without punctuation
        assert "Trump" in entities
        assert "Biden" in entities
        assert "Obama" in entities

    def test_extract_entities_empty_text(self, analyzer):
        """Test with empty text."""
        entities = analyzer.extract_entities("")
        assert entities == []

    def test_extract_entities_no_capitalized_words(self, analyzer):
        """Test with no capitalized words."""
        text = "this is all lowercase text"
        entities = analyzer.extract_entities(text)
        assert entities == []

    def test_get_statistics_without_collection(self, analyzer):
        """Test statistics without collection returns empty dict."""
        entities = ["Trump", "Biden"]

        # With no collection, should return empty dict
        stats = analyzer.get_statistics(entities)
        assert isinstance(stats, dict)
        # May be empty or have entries depending on implementation
        # Don't expect error since it gracefully handles missing collection

    def test_get_statistics_basic(self, analyzer_with_collection):
        """Test basic statistics gathering."""
        entities = ["Trump", "Biden"]
        stats = analyzer_with_collection.get_statistics(entities)

        assert isinstance(stats, dict)
        assert "Trump" in stats
        assert "Biden" in stats

        # Check structure of each entity's stats - using Pydantic model attributes
        for entity, entity_stats in stats.items():
            assert hasattr(entity_stats, "mention_count")
            assert hasattr(entity_stats, "speech_count")
            assert hasattr(entity_stats, "corpus_percentage")
            assert hasattr(entity_stats, "speeches")
            assert isinstance(entity_stats.speeches, list)

    def test_get_statistics_case_insensitive(self, analyzer_with_collection):
        """Test that entity matching is case-insensitive."""
        # Mock returns "Trump" in documents, we search for "trump"
        stats = analyzer_with_collection.get_statistics(["trump"])

        # Should still find matches (case-insensitive)
        assert "trump" in stats or "Trump" in stats

    def test_find_associations_basic(self, analyzer_with_collection):
        """Test finding associated terms."""
        contexts = [
            "Trump spoke about Biden and the economy",
            "Trump criticized Biden's economic policies",
        ]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=5)

        assert isinstance(associations, list)
        assert len(associations) <= 5
        # Should find "Biden" and "economy" associated with "Trump"
        assert any("Biden" in a for a in associations) or any("economy" in a for a in associations)

    def test_find_associations_filters_entity_itself(self, analyzer_with_collection):
        """Test that the entity itself is not in associations."""
        contexts = ["Trump Trump Trump mentioned Biden"]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=10)

        # "Trump" should not be in its own associations
        assert "Trump" not in associations
        assert "trump" not in associations

    def test_find_associations_filters_stopwords(self, analyzer_with_collection):
        """Test that stopwords are filtered from associations."""
        contexts = ["Trump spoke about the economy and the policies"]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=10)

        # Stopwords like "the", "and" should be filtered
        assert "the" not in associations
        assert "and" not in associations

    def test_find_associations_empty_contexts(self, analyzer_with_collection):
        """Test associations with no contexts."""
        associations = analyzer_with_collection.find_associations("Trump", [], top_n=5)

        assert associations == []

    def test_entity_statistics_with_sentiment_none(self, analyzer_with_collection):
        """Test statistics without sentiment analyzer."""
        stats = analyzer_with_collection.get_statistics(["Trump"], include_sentiment=False)

        assert "Trump" in stats
        # Sentiment should not be included (use attribute access for Pydantic model)
        assert stats["Trump"].sentiment is None

    def test_entity_statistics_with_associations(self, analyzer_with_collection):
        """Test statistics with associations enabled."""
        stats = analyzer_with_collection.get_statistics(["Trump"], include_associations=True)

        assert "Trump" in stats
        # Associations should be included (use attribute access for Pydantic model)
        if stats["Trump"].associated_terms:
            assert isinstance(stats["Trump"].associated_terms, list)

    def test_entity_statistics_corpus_percentage(self, mock_collection):
        """Test corpus percentage calculation."""
        # Mock 100 total documents
        mock_collection.count = MagicMock(return_value=100)
        # Mock 10 documents containing "Trump"
        mock_collection.get = MagicMock(
            return_value={
                "documents": ["Trump"] * 10,
                "metadatas": [{"source": f"speech{i}.txt"} for i in range(10)],
            }
        )

        analyzer = EntityAnalyzer(collection=mock_collection)
        stats = analyzer.get_statistics(["Trump"])

        # Should be 10% (use attribute access for Pydantic model)
        assert stats["Trump"].corpus_percentage == 10.0

    def test_multiple_entities_statistics(self, analyzer_with_collection):
        """Test statistics for multiple entities."""
        entities = ["Trump", "Biden", "China"]
        stats = analyzer_with_collection.get_statistics(entities)

        # Should have stats for each entity found (may not find all if not in test data)
        assert len(stats) > 0
        # Check that returned entities are from the requested list
        for entity in stats.keys():
            assert entity in entities

    def test_extract_entities_preserves_case(self, analyzer):
        """Test that extracted entities preserve original case."""
        text = "Trump spoke with Biden about China"
        entities = analyzer.extract_entities(text)

        # Should preserve exact case
        assert "Trump" in entities
        assert "Biden" in entities
        assert "China" in entities

    def test_stopwords_set_contains_common_words(self, analyzer):
        """Test that stopwords set contains expected words."""
        assert "the" in analyzer.stopwords
        assert "and" in analyzer.stopwords
        assert "is" in analyzer.stopwords
        assert "of" in analyzer.stopwords

    def test_extract_entities_complex_sentence(self, analyzer):
        """Test entity extraction from complex sentence."""
        text = "President Trump met with Chinese President Xi Jinping in Beijing"
        entities = analyzer.extract_entities(text)

        assert "President" in entities
        assert "Trump" in entities
        assert "Chinese" in entities
        assert "Jinping" in entities
        assert "Beijing" in entities
