"""
RAG Template - Streamlit Application
A demonstration app for Retrieval-Augmented Generation using ChromaDB and BM25.

Supports: Text files, PDFs, Word documents, Excel files, and CSV files.
"""

import streamlit as st
from typing import List, Optional
import os

from utils import (
    DocumentLoader,
    RAGPipeline,
    LLMClient,
)
from utils.rag_pipeline import SearchMode


# Page configuration
st.set_page_config(
    page_title="RAG Template",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False

    if 'llm_configured' not in st.session_state:
        st.session_state.llm_configured = False


def create_pipeline(
    provider: str,
    api_key: str,
    model: str,
    search_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    n_results: int,
) -> RAGPipeline:
    """Create a new RAG pipeline with the given configuration."""
    mode_map = {
        "Vector (Semantic)": SearchMode.VECTOR,
        "BM25 (Keyword)": SearchMode.BM25,
        "Hybrid (Combined)": SearchMode.HYBRID,
    }

    pipeline = RAGPipeline(
        llm_provider=provider.lower(),
        llm_api_key=api_key,
        llm_model=model,
        search_mode=mode_map.get(search_mode, SearchMode.HYBRID),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        n_results=n_results,
    )

    return pipeline


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")

        # LLM Configuration
        st.subheader("LLM Settings")

        provider = st.selectbox(
            "Provider",
            options=["OpenAI", "Anthropic"],
            index=0,
            help="Select your LLM provider",
        )

        # Get available models for the selected provider
        available_models = LLMClient.get_available_models(provider.lower())

        model = st.selectbox(
            "Model",
            options=available_models,
            index=0,
            help="Select the model to use",
        )

        api_key = st.text_input(
            "API Key",
            type="password",
            help=f"Enter your {provider} API key",
            value=os.environ.get(f"{provider.upper()}_API_KEY", ""),
        )

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
            if not api_key:
                st.error("Please enter your API key")
            else:
                try:
                    with st.spinner("Initializing pipeline..."):
                        pipeline = create_pipeline(
                            provider=provider,
                            api_key=api_key,
                            model=model,
                            search_mode=search_mode,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            n_results=n_results,
                        )
                        st.session_state.rag_pipeline = pipeline
                        st.session_state.llm_configured = True
                        st.session_state.temperature = temperature
                        st.session_state.max_tokens = max_tokens
                    st.success("Pipeline initialized!")
                except Exception as e:
                    st.error(f"Error initializing pipeline: {e}")

        # Show pipeline stats if initialized
        if st.session_state.rag_pipeline:
            st.divider()
            st.subheader("Pipeline Status")
            stats = st.session_state.rag_pipeline.get_stats()
            st.metric("Documents Indexed", stats['vector_store_count'])
            st.caption(f"Search Mode: {stats['search_mode']}")

        return temperature, max_tokens


def render_file_upload():
    """Render the file upload section."""
    st.header("Document Upload")

    # Supported file types
    supported_types = DocumentLoader.get_supported_extensions()
    st.caption(f"Supported formats: {', '.join(supported_types)}")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=[ext.lstrip('.') for ext in supported_types],
        accept_multiple_files=True,
        help="Upload documents to add to the knowledge base",
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            if not st.session_state.rag_pipeline:
                st.error("Please initialize the pipeline first (see sidebar)")
                return

            with st.spinner("Processing documents..."):
                # Prepare file bytes list
                file_bytes_list = [
                    (file.getvalue(), file.name)
                    for file in uploaded_files
                ]

                try:
                    result = st.session_state.rag_pipeline.add_documents(
                        file_bytes_list=file_bytes_list
                    )

                    st.success(
                        f"Processed {result['documents']} documents into "
                        f"{result['chunks']} chunks"
                    )
                    st.session_state.documents_loaded = True

                except Exception as e:
                    st.error(f"Error processing documents: {e}")


def render_chat_interface(temperature: float, max_tokens: int):
    """Render the chat interface."""
    st.header("Ask Questions")

    if not st.session_state.rag_pipeline:
        st.info("Please initialize the pipeline in the sidebar to start asking questions.")
        return

    if not st.session_state.documents_loaded and st.session_state.rag_pipeline.get_stats()['vector_store_count'] == 0:
        st.info("Please upload and process some documents first.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source.get('metadata', {}).get('source', 'Unknown')}")
                        st.caption(source.get('content', '')[:500] + "...")
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
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
                    sources = st.session_state.rag_pipeline.search(question)

                    # Stream the response
                    for token in st.session_state.rag_pipeline.query_stream(
                        question=question,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Show sources
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source.get('metadata', {}).get('source', 'Unknown')}")
                                score_key = 'score' if 'score' in source else 'rrf_score'
                                if score_key in source:
                                    st.caption(f"Relevance: {source[score_key]:.4f}")
                                st.caption(source.get('content', '')[:500] + "...")
                                st.divider()

                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Error generating response: {e}")


def render_search_demo():
    """Render a search demonstration section."""
    st.header("Search Demo")

    if not st.session_state.rag_pipeline:
        st.info("Initialize the pipeline to use search demo.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Enter search query", key="search_demo_query")

    with col2:
        search_mode = st.selectbox(
            "Mode",
            options=["Hybrid", "Vector", "BM25"],
            key="search_demo_mode",
        )

    if st.button("Search", key="search_demo_btn") and search_query:
        mode_map = {
            "Hybrid": SearchMode.HYBRID,
            "Vector": SearchMode.VECTOR,
            "BM25": SearchMode.BM25,
        }

        with st.spinner("Searching..."):
            results = st.session_state.rag_pipeline.search(
                query=search_query,
                mode=mode_map[search_mode],
            )

        if results:
            st.success(f"Found {len(results)} results")

            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result.get('metadata', {}).get('source', 'Unknown')}"):
                    # Show scores
                    if 'score' in result:
                        st.metric("Similarity Score", f"{result['score']:.4f}")
                    if 'rrf_score' in result:
                        st.metric("RRF Score", f"{result['rrf_score']:.4f}")
                    if 'vector_rank' in result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Vector Rank", result.get('vector_rank', 'N/A'))
                        with col2:
                            st.metric("BM25 Rank", result.get('bm25_rank', 'N/A'))

                    st.divider()
                    st.markdown("**Content:**")
                    st.text(result.get('content', '')[:1000])

                    st.divider()
                    st.markdown("**Metadata:**")
                    st.json(result.get('metadata', {}))
        else:
            st.warning("No results found")


def render_clear_data():
    """Render clear data section."""
    st.header("Manage Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared")
            st.rerun()

    with col2:
        if st.button("Clear All Documents", type="secondary", use_container_width=True):
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear()
                st.session_state.documents_loaded = False
                st.success("All documents cleared")
                st.rerun()


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # App title
    st.title("RAG Template Demo")
    st.caption("Retrieval-Augmented Generation with ChromaDB and BM25")

    # Render sidebar and get generation settings
    temperature, max_tokens = render_sidebar()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload Documents",
        "Ask Questions",
        "Search Demo",
        "Manage Data",
    ])

    with tab1:
        render_file_upload()

    with tab2:
        render_chat_interface(temperature, max_tokens)

    with tab3:
        render_search_demo()

    with tab4:
        render_clear_data()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, ChromaDB, and BM25 | "
        "Supports: TXT, PDF, DOCX, XLSX, CSV"
    )


if __name__ == "__main__":
    main()
