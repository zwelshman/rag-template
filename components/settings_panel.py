"""
Settings Panel Component
Sidebar configuration panel for the RAG application.
"""

import streamlit as st
import logging
from typing import Tuple, Optional
from core.retrieval_engine import create_pipeline, SearchMode
from utils.auth import get_api_key

logger = logging.getLogger("rag_app.components.settings_panel")


def render_sidebar() -> Tuple[float, int]:
    """
    Render the sidebar with configuration options.

    Returns:
        Tuple of (temperature, max_tokens)
    """
    with st.sidebar:
        st.header("Configuration")

        # LLM Configuration
        st.subheader("LLM Settings")

        # Demo version: OpenAI only
        st.info("**Model:** GPT-4o (OpenAI)")
        st.caption("Demo version uses OpenAI's GPT-4o model")

        # Get API key from secrets or environment
        api_key = get_api_key("OPENAI_API_KEY")
        if api_key:
            st.success("OpenAI API key loaded", icon="✅")
        else:
            st.warning("⚠️ Please add your OpenAI API key to `.streamlit/secrets.toml`", icon="⚠️")

        # Search Configuration
        st.subheader("Search Settings")

        search_mode = st.selectbox(
            "Search Mode",
            options=["Hybrid (Combined)", "Vector (Semantic)", "BM25 (Keyword)"],
            index=0,
            help="Select the search strategy",
        )

        n_results = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of document chunks to retrieve",
        )

        # Chunking Configuration
        st.subheader("Chunking Settings")

        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum size of each text chunk",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks",
        )

        # Generation Settings
        st.subheader("Generation Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum tokens in the response",
        )

        # Initialize/Update Pipeline button
        if st.button("Initialize Pipeline", type="primary", use_container_width=True):
            logger.info("Initialize Pipeline button clicked")
            if not api_key:
                logger.warning("Pipeline initialization failed: No API key provided")
                st.error("Please enter your OpenAI API key")
            else:
                try:
                    with st.spinner("Initializing pipeline..."):
                        logger.info("Starting pipeline initialization...")
                        pipeline = create_pipeline(
                            api_key=api_key,
                            search_mode=search_mode,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            n_results=n_results,
                        )
                        st.session_state.rag_pipeline = pipeline
                        st.session_state.llm_configured = True
                        st.session_state.temperature = temperature
                        st.session_state.max_tokens = max_tokens
                    st.success("Pipeline initialized with GPT-4o!")
                    logger.info("Pipeline initialization complete - ready for documents")
                except Exception as e:
                    logger.error(f"Pipeline initialization failed: {e}")
                    st.error(f"Error initializing pipeline: {e}")

        # Show pipeline stats if initialized
        if st.session_state.get('rag_pipeline'):
            st.divider()
            st.subheader("Pipeline Status")
            stats = st.session_state.rag_pipeline.get_stats()
            st.metric("Documents Indexed", stats['vector_store_count'])
            st.caption(f"Search Mode: {stats['search_mode']}")

        return temperature, max_tokens
