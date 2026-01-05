"""
Chat Interface Component
Handles the chat UI and interaction with the RAG pipeline.
"""

import streamlit as st
import logging
from core.retrieval_engine import SearchMode

logger = logging.getLogger("rag_app.components.chat_interface")


def render_chat_interface(temperature: float = 0.7, max_tokens: int = 1000):
    """
    Render the chat interface.

    Args:
        temperature: LLM temperature setting
        max_tokens: Maximum tokens for response
    """
    st.header("Ask Questions")

    # Demo version: limit to 10 queries
    MAX_QUERIES = 10
    query_count = st.session_state.get('query_count', 0)
    remaining_queries = MAX_QUERIES - query_count

    if remaining_queries <= 0:
        st.warning(f"⚠️ Demo limit reached: You have used all {MAX_QUERIES} queries for this session. Please refresh the page to start a new session.")
        return

    st.info(f"💬 Demo Version: {remaining_queries} of {MAX_QUERIES} queries remaining")

    if not st.session_state.get('rag_pipeline'):
        st.info("Please initialize the pipeline in the sidebar to start asking questions.")
        return

    pipeline_stats = st.session_state.rag_pipeline.get_stats()
    if not st.session_state.get('documents_loaded') and pipeline_stats['vector_store_count'] == 0:
        st.info("Please upload and process some documents first.")
        return

    # Display chat history
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                from .citation_viewer import render_sources_expander
                render_sources_expander(message["sources"])

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        logger.info("=" * 60)
        logger.info("USER QUERY RECEIVED")
        logger.info("=" * 60)
        logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max tokens: {max_tokens}")

        # Add user message to history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use streaming if supported
                    response_placeholder = st.empty()
                    full_response = ""

                    # Get sources first
                    logger.info("Searching for relevant documents...")
                    sources = st.session_state.rag_pipeline.search(question)
                    logger.info(f"Found {len(sources)} relevant document chunks")

                    # Log source information
                    for i, src in enumerate(sources, 1):
                        source_name = src.get('metadata', {}).get('source', 'Unknown')
                        score_key = 'score' if 'score' in src else 'rrf_score'
                        score = src.get(score_key, 'N/A')
                        logger.info(f"  Source {i}: {source_name} (score: {score})")

                    # Stream the response
                    logger.info("Generating response with Claude Sonnet 4.5...")
                    for token in st.session_state.rag_pipeline.query_stream(
                        question=question,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)
                    logger.info(f"Response generated: {len(full_response)} characters")
                    logger.info("QUERY COMPLETE")
                    logger.info("=" * 60)

                    # Show sources
                    if sources:
                        from .citation_viewer import render_sources_expander
                        render_sources_expander(sources)

                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    })

                    # Increment query count
                    st.session_state.query_count += 1

                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    st.error(f"Error generating response: {e}")
