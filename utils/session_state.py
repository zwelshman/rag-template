"""
Session State Management
Utilities for managing Streamlit session state.
"""

import streamlit as st
import logging

logger = logging.getLogger("rag_app.utils.session_state")


def init_session_state():
    """Initialize session state variables."""
    logger.info("Initializing session state")

    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
        logger.debug("Session state: rag_pipeline initialized to None")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.debug("Session state: chat_history initialized to empty list")

    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
        logger.debug("Session state: documents_loaded initialized to False")

    if 'llm_configured' not in st.session_state:
        st.session_state.llm_configured = False
        logger.debug("Session state: llm_configured initialized to False")


def clear_chat_history():
    """Clear the chat history."""
    logger.info("Clearing chat history")
    st.session_state.chat_history = []


def clear_documents():
    """Clear all loaded documents."""
    logger.info("Clearing documents")
    if st.session_state.get('rag_pipeline'):
        st.session_state.rag_pipeline.clear()
    st.session_state.documents_loaded = False


def reset_session():
    """Reset the entire session state."""
    logger.info("Resetting session state")
    st.session_state.rag_pipeline = None
    st.session_state.chat_history = []
    st.session_state.documents_loaded = False
    st.session_state.llm_configured = False
