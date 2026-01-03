"""
Authentication Utilities
Helper functions for API key management and authentication.
"""

import streamlit as st
import os
import logging

logger = logging.getLogger("rag_app.utils.auth")


def get_api_key() -> str:
    """
    Get Anthropic API key from Streamlit secrets or environment variable.

    Priority:
    1. Streamlit secrets (st.secrets["ANTHROPIC_API_KEY"])
    2. Environment variable (ANTHROPIC_API_KEY)

    Returns:
        API key string or empty string if not found
    """
    # Try Streamlit secrets first
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            logger.info("API key found in Streamlit secrets")
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception as e:
        logger.debug(f"Could not read Streamlit secrets: {e}")

    # Fall back to environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        logger.info("API key found in environment variable")
    else:
        logger.warning("No API key found in secrets or environment")
    return env_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate that an API key is present and has the correct format.

    Args:
        api_key: The API key to validate

    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False

    # Basic validation - Anthropic keys start with 'sk-ant-'
    if api_key.startswith('sk-ant-'):
        return True

    # For backward compatibility, accept any non-empty key
    return len(api_key) > 10


def setup_authentication():
    """
    Setup authentication for the application.
    Can be extended to support multiple auth methods.
    """
    # For now, just verify API key is available
    api_key = get_api_key()

    if not api_key:
        st.warning(
            "No API key found. Please add your Anthropic API key to "
            "`.streamlit/secrets.toml` or set the `ANTHROPIC_API_KEY` environment variable."
        )
        return None

    return api_key
