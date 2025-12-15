"""
RAG Template - Streamlit Application
A demonstration app for Retrieval-Augmented Generation using ChromaDB and BM25.

Supports: Text files, PDFs, Word documents, Excel files, and CSV files.
Uses Anthropic Claude Sonnet 4.5 for LLM generation.
"""

import streamlit as st
from typing import List, Optional
import os
import sys
import logging

from utils import (
    DocumentLoader,
    RAGPipeline,
    LLMClient,
)
from utils.rag_pipeline import SearchMode


# Configure logging to stdout with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)


# Page configuration
st.set_page_config(
    page_title="RAG Template",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def create_pipeline(
    api_key: str,
    search_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    n_results: int,
) -> RAGPipeline:
    """Create a new RAG pipeline with Anthropic Claude Sonnet 4.5."""
    logger.info("=" * 60)
    logger.info("CREATING NEW RAG PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Provider: Anthropic")
    logger.info(f"Model: claude-sonnet-4-5")
    logger.info(f"Search Mode: {search_mode}")
    logger.info(f"Chunk Size: {chunk_size}")
    logger.info(f"Chunk Overlap: {chunk_overlap}")
    logger.info(f"Number of Results: {n_results}")

    mode_map = {
        "Vector (Semantic)": SearchMode.VECTOR,
        "BM25 (Keyword)": SearchMode.BM25,
        "Hybrid (Combined)": SearchMode.HYBRID,
    }

    pipeline = RAGPipeline(
        llm_provider="anthropic",
        llm_api_key=api_key,
        llm_model="claude-sonnet-4-5",
        search_mode=mode_map.get(search_mode, SearchMode.HYBRID),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        n_results=n_results,
    )

    logger.info("RAG Pipeline created successfully")
    logger.info("=" * 60)

    return pipeline


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("Configuration")

        # LLM Configuration
        st.subheader("LLM Settings")

        # Fixed to Anthropic with Sonnet 4.5
        st.info("**Model:** Claude Sonnet 4.5")
        st.caption("Using Anthropic's latest model for best results")

        # Get API key from secrets or environment
        api_key = get_api_key()
        if api_key:
            st.success("API key auto-loaded from secrets/environment", icon="✅")

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
                st.error("Please enter your Anthropic API key")
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
                    st.success("Pipeline initialized with Claude Sonnet 4.5!")
                    logger.info("Pipeline initialization complete - ready for documents")
                except Exception as e:
                    logger.error(f"Pipeline initialization failed: {e}")
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
        logger.info(f"Files selected for upload: {[f.name for f in uploaded_files]}")
        if st.button("Process Documents", type="primary"):
            if not st.session_state.rag_pipeline:
                logger.warning("Process Documents clicked but pipeline not initialized")
                st.error("Please initialize the pipeline first (see sidebar)")
                return

            with st.spinner("Processing documents..."):
                logger.info("=" * 60)
                logger.info("DOCUMENT PROCESSING STARTED")
                logger.info("=" * 60)

                # Prepare file bytes list
                file_bytes_list = [
                    (file.getvalue(), file.name)
                    for file in uploaded_files
                ]

                for filename in [f.name for f in uploaded_files]:
                    logger.info(f"Processing file: {filename}")

                try:
                    result = st.session_state.rag_pipeline.add_documents(
                        file_bytes_list=file_bytes_list
                    )

                    logger.info(f"Documents processed: {result['documents']}")
                    logger.info(f"Chunks created: {result['chunks']}")
                    logger.info(f"Vector store count: {result.get('vector_store_count', 'N/A')}")
                    logger.info(f"BM25 index count: {result.get('bm25_count', 'N/A')}")
                    logger.info("DOCUMENT PROCESSING COMPLETE")
                    logger.info("=" * 60)

                    st.success(
                        f"Processed {result['documents']} documents into "
                        f"{result['chunks']} chunks"
                    )
                    st.session_state.documents_loaded = True

                except Exception as e:
                    logger.error(f"Document processing failed: {e}")
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
        logger.info("=" * 60)
        logger.info("USER QUERY RECEIVED")
        logger.info("=" * 60)
        logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Max tokens: {max_tokens}")

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
                    logger.error(f"Query failed: {e}")
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
        logger.info("=" * 60)
        logger.info("SEARCH DEMO QUERY")
        logger.info("=" * 60)
        logger.info(f"Query: {search_query}")
        logger.info(f"Mode: {search_mode}")

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

        logger.info(f"Search returned {len(results)} results")
        logger.info("=" * 60)

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
            logger.info("No results found for query")
            st.warning("No results found")


def render_clear_data():
    """Render clear data section."""
    st.header("Manage Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            logger.info("Clearing chat history")
            st.session_state.chat_history = []
            st.success("Chat history cleared")
            st.rerun()

    with col2:
        if st.button("Clear All Documents", type="secondary", use_container_width=True):
            if st.session_state.rag_pipeline:
                logger.info("Clearing all documents from pipeline")
                st.session_state.rag_pipeline.clear()
                st.session_state.documents_loaded = False
                logger.info("All documents cleared successfully")
                st.success("All documents cleared")
                st.rerun()


def render_instructions():
    """Render step-by-step instructions for using the app."""
    with st.expander("How to Use This App (Step-by-Step Guide)", expanded=False):
        st.markdown("""
### Getting Started with RAG Template

Follow these steps to use the Retrieval-Augmented Generation (RAG) system:

---

#### Step 1: Configure Your API Key
1. Look at the **sidebar on the left**
2. Your Anthropic API key should be auto-loaded from Streamlit secrets
3. If not, enter your Anthropic API key manually in the "Anthropic API Key" field
4. Click **"Initialize Pipeline"** to start the system

> **Tip:** To set up Streamlit secrets, create a file at `.streamlit/secrets.toml` with:
> ```toml
> ANTHROPIC_API_KEY = "your-api-key-here"
> ```

---

#### Step 2: Upload Your Documents
1. Go to the **"Upload Documents"** tab
2. Click **"Browse files"** to select your documents
3. Supported formats: TXT, PDF, DOCX, XLSX, CSV
4. Select one or multiple files
5. Click **"Process Documents"** to index them

> **What happens:** Your documents are split into chunks, embedded using AI, and stored in a vector database for fast retrieval.

---

#### Step 3: Ask Questions
1. Go to the **"Ask Questions"** tab
2. Type your question in the chat input at the bottom
3. The system will:
   - Search your documents for relevant information
   - Use Claude Sonnet 4.5 to generate an answer
   - Show you the source documents used

> **Tip:** Click "View Sources" under any answer to see exactly which document chunks were used.

---

#### Step 4: Explore Search (Optional)
1. Go to the **"Search Demo"** tab
2. Test different search modes:
   - **Hybrid**: Combines semantic + keyword search (recommended)
   - **Vector**: Semantic similarity search
   - **BM25**: Traditional keyword matching
3. See how each mode ranks your documents differently

---

#### Step 5: Manage Your Data
1. Go to the **"Manage Data"** tab
2. **Clear Chat History**: Start a fresh conversation
3. **Clear All Documents**: Remove all indexed documents

---

### Configuration Options (Sidebar)

| Setting | Description | Default |
|---------|-------------|---------|
| **Search Mode** | How documents are retrieved | Hybrid |
| **Number of Results** | How many chunks to retrieve | 5 |
| **Chunk Size** | Size of document chunks | 1000 |
| **Chunk Overlap** | Overlap between chunks | 200 |
| **Temperature** | Response creativity (0-1) | 0.7 |
| **Max Tokens** | Maximum response length | 1000 |

---

### Monitoring & Logs

Check your terminal/console for detailed logs showing:
- Pipeline initialization status
- Document processing progress
- Search queries and results
- LLM generation activity
        """)
    return


def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("RAG TEMPLATE APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info("Model: Claude Sonnet 4.5 (claude-sonnet-4-5-20241022)")
    logger.info("Provider: Anthropic")

    # Initialize session state
    init_session_state()

    # App title
    st.title("RAG Template Demo")
    st.caption("Retrieval-Augmented Generation powered by Claude Sonnet 4.5")

    # Render step-by-step instructions
    render_instructions()

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
        "Built with Streamlit, ChromaDB, BM25, and Claude Sonnet 4.5 | "
        "Supports: TXT, PDF, DOCX, XLSX, CSV"
    )

    logger.debug("Main render complete")


if __name__ == "__main__":
    main()
