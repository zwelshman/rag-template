"""
Citation Viewer Component
Displays source citations and document references.
"""

import streamlit as st
from typing import List, Dict, Any


def render_citation_viewer(sources: List[Dict[str, Any]]):
    """
    Render citations for a set of sources.

    Args:
        sources: List of source documents with metadata
    """
    if not sources:
        st.info("No sources available")
        return

    st.subheader("Sources")

    for i, source in enumerate(sources, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                source_name = source.get('metadata', {}).get('source', 'Unknown')
                st.markdown(f"**{i}. {source_name}**")

            with col2:
                # Show relevance score
                if 'score' in source:
                    st.caption(f"Score: {source['score']:.4f}")
                elif 'rrf_score' in source:
                    st.caption(f"RRF: {source['rrf_score']:.4f}")

            # Show content preview
            content = source.get('content', '')
            st.text(content[:500] + ("..." if len(content) > 500 else ""))

            # Show full metadata in expander
            with st.expander("View metadata"):
                st.json(source.get('metadata', {}))

            if i < len(sources):
                st.divider()


def render_sources_expander(sources: List[Dict[str, Any]]):
    """
    Render sources in an expander (for chat interface).

    Args:
        sources: List of source documents
    """
    with st.expander("View Sources"):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}:** {source.get('metadata', {}).get('source', 'Unknown')}")
            score_key = 'score' if 'score' in source else 'rrf_score'
            if score_key in source:
                st.caption(f"Relevance: {source[score_key]:.4f}")
            st.caption(source.get('content', '')[:500] + "...")
            st.divider()
