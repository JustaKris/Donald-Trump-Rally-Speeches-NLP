"""
Text preprocessing utilities for NLP analysis.

This module contains functions for cleaning, tokenizing, and preparing
text data for various NLP tasks.
"""

import re
from typing import List, Set

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def get_stopwords() -> Set[str]:
    """
    Get English stopwords set.

    Returns:
        Set of stopword strings
    """
    words: List[str] = stopwords.words("english")
    return set(words)


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean and normalize text for analysis.

    Args:
        text: Input text to clean
        remove_stopwords: Whether to remove common English stopwords

    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s\.\,\!\?]", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    if remove_stopwords:
        stop_words = get_stopwords()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        text = " ".join(tokens)

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text to tokenize

    Returns:
        List of token strings
    """
    return word_tokenize(text.lower())


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens: List of token strings
        n: N-gram size (2 for bigrams, 3 for trigrams, etc.)

    Returns:
        List of n-gram strings
    """
    ngrams: List[str] = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        ngrams.append(ngram)
    return ngrams


def chunk_text_for_bert(text: str, tokenizer, max_length: int = 510) -> List[dict]:
    """
    Split text into chunks that fit within BERT's token limits.

    Args:
        text: Input text to chunk
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum tokens per chunk (510 to leave room for [CLS] and [SEP])

    Returns:
        List of encoded chunks ready for model input
    """
    # Tokenize the full text
    tokens = tokenizer.tokenize(text)

    # Split into chunks
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i : i + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        # Encode with special tokens
        encoding = tokenizer.encode_plus(
            chunk_text,
            add_special_tokens=True,
            max_length=max_length + 2,  # +2 for [CLS] and [SEP]
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        chunks.append(encoding)

    return chunks


def calculate_word_frequency(text: str, top_n: int = 20, remove_stopwords: bool = True) -> dict:
    """
    Calculate word frequency distribution.

    Args:
        text: Input text to analyze
        top_n: Number of top words to return
        remove_stopwords: Whether to filter out stopwords

    Returns:
        Dictionary mapping words to their frequencies
    """
    # Clean and tokenize
    cleaned = clean_text(text, remove_stopwords=remove_stopwords)
    tokens = tokenize_text(cleaned)

    # Filter out punctuation
    tokens = [t for t in tokens if t.isalpha()]

    # Count frequencies
    from collections import Counter

    freq = Counter(tokens)

    # Return top N
    return dict(freq.most_common(top_n))
