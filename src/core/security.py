"""
Security utilities for the NLP Chatbot API.

Provides authentication, authorization, and security-related utilities.
Currently implements basic API key validation. Can be extended for more
sophisticated security features (JWT, OAuth, rate limiting, etc.).
"""

import hashlib
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status


def generate_api_key() -> str:
    """
    Generate a secure API key.

    Returns:
        Randomly generated API key
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.

    Args:
        api_key: API key to hash

    Returns:
        SHA-256 hash of the API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for authentication")
) -> str:
    """
    Verify API key from request header.

    This is a dependency that can be added to routes requiring authentication.
    Currently disabled for demo purposes, but can be easily enabled.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # For demo purposes, API key validation is disabled
    # To enable, set valid API keys in environment and uncomment below:

    # if not x_api_key:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="API key required",
    #         headers={"WWW-Authenticate": "ApiKey"},
    #     )
    #
    # # Validate against configured keys
    # settings = get_settings()
    # valid_keys = settings.api_keys.split(",")  # Would need to add to config
    #
    # if x_api_key not in valid_keys:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API key",
    #         headers={"WWW-Authenticate": "ApiKey"},
    #     )

    return x_api_key or "demo"


def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input text.

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        HTTPException: If input exceeds maximum length
    """
    # Remove null bytes
    text = text.replace("\0", "")

    # Check length
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Input text exceeds maximum length of {max_length} characters",
        )

    return text.strip()
