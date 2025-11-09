"""
Output formatting utilities.

Provides functions for formatting API responses and display data.
"""

from typing import Any, Dict, List


def format_sentiment_badge(sentiment: str, confidence: float) -> Dict[str, Any]:
    """
    Format sentiment data for display.

    Args:
        sentiment: Sentiment label (positive/negative/neutral)
        confidence: Confidence score (0-1)

    Returns:
        Formatted sentiment badge data
    """
    colors = {
        "positive": "#28a745",
        "negative": "#dc3545",
        "neutral": "#17a2b8",
    }

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "color": colors.get(sentiment.lower(), "#6c757d"),
        "display": f"{sentiment.upper()} ({confidence * 100:.1f}%)",
    }


def format_confidence_badge(confidence_level: str, confidence_score: float) -> Dict[str, Any]:
    """
    Format confidence data for display.

    Args:
        confidence_level: Confidence level (high/medium/low)
        confidence_score: Numeric confidence score (0-1)

    Returns:
        Formatted confidence badge data
    """
    colors = {
        "high": "#28a745",
        "medium": "#ffc107",
        "low": "#dc3545",
    }

    icons = {
        "high": "✅",
        "medium": "⚠️",
        "low": "❌",
    }

    return {
        "level": confidence_level,
        "score": round(confidence_score, 3),
        "color": colors.get(confidence_level.lower(), "#6c757d"),
        "icon": icons.get(confidence_level.lower(), "❓"),
        "display": f"{icons.get(confidence_level.lower(), '❓')} {confidence_level.upper()} ({confidence_score * 100:.1f}%)",
    }


def format_sources(sources: List[str]) -> List[Dict[str, str]]:
    """
    Format source document list for display.

    Args:
        sources: List of source document names

    Returns:
        List of formatted source objects
    """
    return [{"name": source, "icon": "📄"} for source in sources]


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a float as a percentage string.

    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
