"""
Streamlit RAG Kit - Main Application
A modular, production-ready RAG application powered by Claude Sonnet 4.5.
"""

import streamlit as st
import sys
import logging

# Import components
from components import (
    render_file_upload,
    render_chat_interface,
    render_sidebar,
)

# Import utilities
from utils.session_state import init_session_state


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)


# Page configuration
st.set_page_config(
    page_title="Streamlit RAG Kit",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_instructions():
    """Render step-by-step instructions for using the app."""
    with st.expander("How to Use This App (Step-by-Step Guide)", expanded=False):
        st.markdown("""
### Getting Started with Streamlit RAG Kit

Follow these steps to use the Retrieval-Augmented Generation (RAG) system:

---

#### Step 1: Configure Your API Key
1. Look at the **sidebar on the left**
2. Your Anthropic API key should be auto-loaded from Streamlit secrets
3. If not, add it to `.streamlit/secrets.toml`
4. Click **"Initialize Pipeline"** to start the system

> **Tip:** To set up Streamlit secrets, create a file at `.streamlit/secrets.toml` with:
> ```toml
> ANTHROPIC_API_KEY = "your-api-key-here"
> ```

---

#### Step 2: Upload Your Documents
1. Go to the **"Upload Documents"** tab
2. Click **"Browse files"** to select your documents
3. Supported formats: TXT, PDF, DOCX, XLSX, CSV, JSON
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

#### Step 4: Manage Your Data
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


def render_clear_data():
    """Render data management section."""
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
            if st.session_state.get('rag_pipeline'):
                logger.info("Clearing all documents from pipeline")
                st.session_state.rag_pipeline.clear()
                st.session_state.documents_loaded = False
                logger.info("All documents cleared successfully")
                st.success("All documents cleared")
                st.rerun()


def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("STREAMLIT RAG KIT - APPLICATION STARTING")
    logger.info("=" * 60)
    logger.info("Model: Claude Sonnet 4.5")
    logger.info("Provider: Anthropic")
    logger.info("Architecture: Modular Components")

    # Initialize session state
    init_session_state()

    # App title
    st.title("📚 Streamlit RAG Kit")
    st.caption("Production-Ready RAG powered by Claude Sonnet 4.5")

    # Render step-by-step instructions
    render_instructions()

    # Render sidebar and get generation settings
    temperature, max_tokens = render_sidebar()

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload Documents",
        "💬 Ask Questions",
        "🗑️ Manage Data",
    ])

    with tab1:
        render_file_upload()

    with tab2:
        render_chat_interface(temperature, max_tokens)

    with tab3:
        render_clear_data()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, ChromaDB, BM25, and Claude Sonnet 4.5 | "
        "Supports: TXT, PDF, DOCX, XLSX, CSV, JSON | "
        "Modular Architecture"
    )

    logger.debug("Main render complete")


if __name__ == "__main__":
    main()
