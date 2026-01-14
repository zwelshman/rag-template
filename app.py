"""
Streamlit RAG Kit - Main Application
A modular, production-ready RAG application powered by Meta Llama 3.1 8B Instruct.
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
    page_title="RAG Starter Kit - Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_instructions():
    """Render step-by-step instructions for using the app."""
    with st.expander("How to Use This Demo (Step-by-Step Guide)", expanded=False):
        st.markdown("""
### Getting Started with RAG Starter Kit Demo

Follow these steps to try out the RAG (Retrieval-Augmented Generation) system:

---

#### 🎯 Demo Limitations
- **3 documents** maximum per session
- **10 queries** maximum per session
- **ChromaDB** vector store (local, no cloud services)
- **Meta Llama 3.1 8B Instruct** (Hugging Face) - with model selection dropdown
- Basic chat interface

---

#### Step 1: Configure Your API Key
1. Look at the **sidebar on the left**
2. Add your Anthropic API key to `.streamlit/secrets.toml`
3. Click **"Initialize Pipeline"** to start the system

> **Tip:** Create a file at `.streamlit/secrets.toml` with:
> ```toml
> ANTHROPIC_API_KEY = "sk-ant-your-anthropic-api-key-here"
> ```

---

#### Step 2: Upload Your Documents (Max 3)
1. Go to the **"Upload Documents"** tab
2. Click **"Browse files"** to select your documents
3. Supported formats: TXT, PDF, DOCX, XLSX, CSV, JSON
4. Upload up to 3 documents total
5. Click **"Process Documents"** to index them

---

#### Step 3: Ask Questions (Max 10)
1. Go to the **"Ask Questions"** tab
2. Type your question in the chat input
3. The system will search your documents and generate an answer
4. Click "View Sources" to see which document chunks were used

---

#### Step 4: Manage Your Data
1. Go to the **"Manage Data"** tab
2. **Clear Chat History**: Start a fresh conversation
3. **Clear All Documents**: Remove all documents and reset limits

---

### Want More?

This is a limited demo version. The full version includes:
- ✅ Unlimited documents and queries
- ✅ Multiple LLM providers (OpenAI, Anthropic)
- ✅ Advanced prompt templates
- ✅ Cost tracking and analytics
- ✅ Pinecone cloud vector store
- ✅ Enhanced customization options
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
                st.session_state.document_count = 0
                logger.info("All documents cleared successfully")
                st.success("All documents cleared and limits reset")
                st.rerun()


def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("RAG STARTER KIT - DEMO VERSION")
    logger.info("=" * 60)
    logger.info("Model: Meta Llama 3.1 8B Instruct (Hugging Face)")
    logger.info("Provider: Hugging Face")
    logger.info("Version: Demo (3 docs, 10 queries)")

    # Initialize session state
    init_session_state()

    # App title and watermark
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 RAG Starter Kit - Demo Version")
        st.caption("Try out RAG with Meta Llama 3.1 8B Instruct (Hugging Face) • 3 docs • 10 queries")
    with col2:
        st.markdown(
            """
            <div style='text-align: right; padding-top: 20px;'>
                <p style='color: #888; font-size: 0.9em; margin: 0;'>
                    🚀 Powered by<br/>
                    <strong>RAG Starter Kit</strong>
                </p>
                <p style='color: #FF6B6B; font-size: 0.8em; margin-top: 5px;'>
                    <a href='#' style='color: #FF6B6B; text-decoration: none;'>
                        Get Full Version →
                    </a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

    # Footer with watermark
    st.divider()
    footer_col1, footer_col2 = st.columns([2, 1])
    with footer_col1:
        st.caption(
            "Built with Streamlit, ChromaDB, BM25, and Meta Llama 3.1 (Hugging Face) | "
            "Supports: TXT, PDF, DOCX, XLSX, CSV, JSON"
        )
    with footer_col2:
        st.markdown(
    """
    <div style='text-align: right; padding-top: 20px;'>
        <p style='color: #888; font-size: 0.9em; margin: 0;'>
            🚀 Powered by<br/>
            <strong>RAG Starter Kit</strong>
        </p>
        <p style='color: #FF6B6B; font-size: 0.8em; margin-top: 5px;'>
            <a href='https://zmswelshman.gumroad.com/l/rag-knowledge-base' target='_blank' style='color: #FF6B6B; text-decoration: none;'>
                Get Full Version →
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    logger.debug("Main render complete")


if __name__ == "__main__":
    main()
