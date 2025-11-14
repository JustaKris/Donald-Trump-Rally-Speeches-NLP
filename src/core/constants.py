"""
Application-wide constants for the Trump Rally Speeches NLP Chatbot.

This module centralizes all constant values used across the application,
following Python conventions for better maintainability and configuration
management.
"""

from typing import Final

# ==================== Entity Analysis Stopwords ====================
# Words to filter out when extracting entities from text
# These are common words that are often capitalized but not entities

ENTITY_STOPWORDS: Final[set[str]] = {
    # Determiners and pronouns
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
    "up",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    # Auxiliary verbs
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    # Pronouns
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
    # Question words
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    # Quantifiers
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
    # Negations and intensifiers
    "no",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    # Common contractions (fragments)
    "s",
    "t",
    "ve",
    "re",
    "ll",
    "m",
    "don",
    "didn",
    "wasn",
    # Common verbs
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
}

# ==================== Entity Question Words ====================
# Question words to skip when extracting entities from questions

ENTITY_QUESTION_WORDS: Final[set[str]] = {
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
}

# ==================== Default Topic Excluded Verbs ====================
# Common verbs to exclude from topic extraction for better semantic clarity
# This is the default set; can be overridden via config.topic_excluded_verbs

DEFAULT_EXCLUDED_VERBS: Final[str] = (
    "want,think,know,make,get,go,see,come,take,give,say,tell,ask,use,find,"
    "work,call,try,feel,leave,put,mean,keep,let,begin,seem,help,talk,turn,"
    "start,show,hear,play,run,move,like,live,believe,bring,happen,write,sit,"
    "stand,lose,pay,meet,include,continue,learn,change,lead,understand,watch,"
    "follow,stop,create,speak,read,allow,add,spend,grow,open,walk,win,offer,"
    "remember,love,consider"
)

# ==================== Model Default Names ====================
# Default HuggingFace model identifiers

DEFAULT_SENTIMENT_MODEL: Final[str] = "ProsusAI/finbert"
DEFAULT_EMOTION_MODEL: Final[str] = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_EMBEDDING_MODEL: Final[str] = "all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ==================== LLM Provider Names ====================
# Supported LLM providers

LLM_PROVIDER_GEMINI: Final[str] = "gemini"
LLM_PROVIDER_OPENAI: Final[str] = "openai"
LLM_PROVIDER_ANTHROPIC: Final[str] = "anthropic"
LLM_PROVIDER_NONE: Final[str] = "none"

# ==================== API Response Limits ====================
# Maximum values for API responses

MAX_CONTEXT_LENGTH: Final[int] = 4000  # Max characters for LLM context
MAX_TOPIC_LABEL_LENGTH: Final[int] = 50  # Max characters for topic labels
MAX_KEYWORDS_FOR_LABEL: Final[int] = 5  # Max keywords to use for label generation
MAX_CLUSTERS_FOR_SUMMARY: Final[int] = 5  # Max clusters to include in summary
MAX_TEXT_SAMPLE_LENGTH: Final[int] = 2000  # Max characters for text samples

# ==================== Default Thresholds ====================
# Default threshold values for various NLP tasks

DEFAULT_TOPIC_RELEVANCE_THRESHOLD: Final[float] = 0.5
DEFAULT_TOPIC_MIN_CLUSTERS: Final[int] = 3
DEFAULT_SENTIMENT_TEMPERATURE: Final[float] = 0.4
DEFAULT_SENTIMENT_MAX_TOKENS: Final[int] = 200

# ==================== File Extensions ====================
# Supported file types

SUPPORTED_TEXT_EXTENSIONS: Final[tuple[str, ...]] = (".txt", ".md", ".text")

# ==================== Logging Formats ====================
# Log format templates

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# ==================== Environment Variable Names ====================
# Standard environment variable names used in the application

ENV_GEMINI_API_KEY: Final[str] = "GEMINI_API_KEY"
ENV_SPEECHES_PATH: Final[str] = "SPEECHES_PATH"
ENV_CHROMADB_PATH: Final[str] = "CHROMADB_PERSIST_DIRECTORY"
ENV_LOG_LEVEL: Final[str] = "LOG_LEVEL"
