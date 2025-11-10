"""
Utility functions for data loading and statistics.

This module provides helper functions for loading speech data and
computing aggregate statistics.
"""

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_speeches_from_directory(data_dir: Optional[Any] = None) -> pd.DataFrame:
    """
    Load all speeches from the data directory.

    Args:
        data_dir: Path to directory containing speech text files.
                 If None, uses default data/Donald Trump Rally Speeches/

    Returns:
        DataFrame with columns: filename, location, month, year, content, word_count
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "Donald Trump Rally Speeches"
    else:
        data_dir = Path(data_dir)

    speeches = []

    for file_path in sorted(data_dir.glob("*.txt")):
        # Parse filename (e.g., "GreenvilleJul17_2019.txt")
        filename = file_path.name
        name_parts = filename.replace(".txt", "").split("_")

        if len(name_parts) == 2:
            location_date = name_parts[0]
            year = name_parts[1]

            # Extract location and month from the first part
            # Find where the month starts (first uppercase letter after beginning)
            import re

            match = re.search(r"([A-Z][a-z]+)(\d+)", location_date)
            if match:
                location_month = location_date[: match.start(2)]
                # Extract location (everything before last 3 chars which is month abbreviation)
                location = location_month[:-3] if len(location_month) > 3 else location_month
                month = location_month[-3:] if len(location_month) > 3 else ""
            else:
                location = location_date
                month = ""

            # Read content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            speeches.append(
                {
                    "filename": filename,
                    "location": location,
                    "month": month,
                    "year": year,
                    "content": content,
                    "word_count": len(content.split()),
                }
            )

    return pd.DataFrame(speeches)


def get_dataset_statistics() -> Dict[str, Any]:
    """
    Get aggregate statistics about the speech dataset.

    Returns:
        Dictionary with dataset statistics
    """
    df = load_speeches_from_directory()

    return {
        "total_speeches": len(df),
        "total_words": int(df["word_count"].sum()),
        "avg_words_per_speech": float(df["word_count"].mean()),
        "date_range": {
            "start": f"{df['month'].iloc[0]} {df['year'].iloc[0]}",
            "end": f"{df['month'].iloc[-1]} {df['year'].iloc[-1]}",
        },
        "years": df["year"].unique().tolist(),
        "locations": df["location"].unique().tolist(),
    }


def get_word_frequency_stats(text: str, top_n: int = 50) -> Dict[str, Any]:
    """
    Get word frequency statistics from text.

    Args:
        text: Input text to analyze
        top_n: Number of top words to include

    Returns:
        Dictionary with word frequency data
    """
    from .text_preprocessing import clean_text, tokenize_text

    # Clean and tokenize
    cleaned = clean_text(text, remove_stopwords=True)
    tokens = tokenize_text(cleaned)

    # Filter alphabetic tokens only
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]

    # Count frequencies
    freq = Counter(tokens)
    top_words = freq.most_common(top_n)

    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "top_words": [{"word": word, "count": count} for word, count in top_words],
    }


def extract_topics(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Extract key topics/themes from text based on word frequency.

    Args:
        text: Input text to analyze
        top_n: Number of topics to extract

    Returns:
        List of topic dictionaries with word and relevance score
    """
    stats = get_word_frequency_stats(text, top_n=top_n)

    # Normalize counts to 0-1 range for relevance score
    top_words = stats["top_words"]
    if not top_words:
        return []

    max_count = top_words[0]["count"]

    return [
        {
            "topic": word_data["word"],
            "relevance": round(word_data["count"] / max_count, 3),
            "mentions": word_data["count"],
        }
        for word_data in top_words
    ]
