# 📚 Streamlit RAG Kit

A production-ready, modular Retrieval-Augmented Generation (RAG) application powered by **Meta Llama 3.1** via Hugging Face, built with Streamlit.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://document-search-template.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🚀 **Production-Ready**: Modular architecture with separation of concerns
- 🧠 **Powered by Meta Llama 3.1**: Open source LLM via Hugging Face Inference API
- 🔄 **Automatic Fallback**: Gracefully handles deprecated models with fallback support
- 📁 **Multi-Format Support**: PDF, DOCX, TXT, CSV, XLSX, JSON
- 🔍 **Hybrid Search**: Combines vector similarity and BM25 keyword search
- 💬 **Streaming Responses**: Real-time response generation
- 📊 **Source Citations**: View which documents informed each answer
- 🎨 **Clean UI**: Intuitive Streamlit interface
- 🔧 **Highly Configurable**: Adjust chunk size, search mode, and LLM parameters
- 📦 **Easy Deployment**: Deploy to Streamlit Cloud, Docker, or any cloud platform

## 🏗️ Architecture

```
streamlit-rag-kit/
├── 🎯 app.py                          # Main Streamlit app
├── 📋 requirements.txt                # Dependencies
├── 📁 components/                     # UI Components
│   ├── file_uploader.py              # Document upload UI
│   ├── chat_interface.py             # Chat UI
│   ├── settings_panel.py             # Configuration panel
│   └── citation_viewer.py            # Source display
├── 📁 core/                           # Core RAG Engine
│   ├── vector_stores/                # Vector DB implementations
│   │   ├── base.py                   # Base interface
│   │   └── chroma.py                 # ChromaDB (default)
│   ├── llm_providers/                # LLM integrations
│   │   ├── base.py                   # Base interface
│   │   ├── huggingface_provider.py   # Hugging Face (default)
│   │   ├── openai_provider.py        # OpenAI support
│   │   ├── anthropic_provider.py     # Anthropic support
│   │   └── factory.py                # Provider factory
│   ├── document_processor.py         # Document loading & chunking
│   └── retrieval_engine.py           # Main RAG pipeline
├── 📁 prompts/                        # Prompt templates
│   ├── default_system.txt
│   ├── detailed_analysis.txt
│   └── concise_summary.txt
├── 📁 utils/                          # Utilities
│   ├── auth.py                       # API key management
│   ├── session_state.py              # State management
│   ├── cost_tracking.py              # Usage tracking
│   └── bm25_search.py                # BM25 implementation
├── 📁 deployment/                     # Deployment configs
│   ├── streamlit_cloud_guide.md
│   ├── docker-compose.yml
│   └── Dockerfile
├── 📁 docs/                           # Documentation
│   ├── quick_start.md
│   └── api_key_setup.md
└── 📁 .streamlit/
    ├── config.toml                   # App configuration
    └── secrets.toml.example          # API key template
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd streamlit-rag-kit

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up API Key

Create `.streamlit/secrets.toml`:

```toml
HF_API_KEY = "hf_your-huggingface-api-key-here"
```

Get your API key from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 4. Use the App

1. **Initialize Pipeline**: Click "Initialize Pipeline" in the sidebar
2. **Upload Documents**: Upload your files (PDF, DOCX, TXT, etc.)
3. **Ask Questions**: Start chatting with your documents!

## 📖 Documentation

- [📚 Quick Start Guide](docs/quick_start.md) - Get started in 5 minutes
- [🔑 API Key Setup](docs/api_key_setup.md) - Configure API keys
- [🚀 Deployment Guide](deployment/streamlit_cloud_guide.md) - Deploy to production

## Project Structure

```
rag-template/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/
│   ├── __init__.py       # Package exports
│   ├── document_loader.py # Multi-format document loading
│   ├── text_splitter.py   # Text chunking utilities
│   ├── vector_store.py    # ChromaDB integration
│   ├── bm25_search.py     # BM25 search implementation
│   ├── llm.py            # LLM client (OpenAI/Anthropic)
│   └── rag_pipeline.py    # Main RAG pipeline
└── data/                 # Data directory (created on init)
```

## Programmatic Usage

You can also use the RAG pipeline programmatically:

```python
from utils import RAGPipeline
from utils.rag_pipeline import SearchMode

# Initialize the pipeline
pipeline = RAGPipeline(
    llm_provider="huggingface",
    llm_api_key="your-hf-api-key",
    llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    search_mode=SearchMode.HYBRID,
    chunk_size=1000,
    chunk_overlap=200,
)

# Add documents
result = pipeline.add_documents(file_paths=["document.pdf", "data.csv"])
print(f"Added {result['chunks']} chunks from {result['documents']} documents")

# Query the pipeline
response = pipeline.query("What is the main topic of the documents?")
print(response.answer)

# Access sources
for source in response.sources:
    print(f"Source: {source['metadata']['source']}")
```

## Search Modes

### Vector Search (Semantic)
Uses sentence transformers to create embeddings and ChromaDB to find semantically similar content. Best for:
- Finding conceptually related content
- Handling paraphrased queries
- Understanding context and meaning

### BM25 Search (Keyword)
Traditional keyword-based search using term frequency. Best for:
- Exact term matching
- Technical terminology
- Specific names or codes

### Hybrid Search (Recommended)
Combines both methods using Reciprocal Rank Fusion. Benefits:
- Leverages strengths of both approaches
- More robust retrieval
- Better coverage of different query types

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Overlap between consecutive chunks |
| `n_results` | 5 | Number of chunks to retrieve |
| `temperature` | 0.7 | LLM response randomness (0-1) |
| `max_tokens` | 1000 | Maximum response length |
| `embedding_model` | all-MiniLM-L6-v2 | Sentence transformer model |

## Dependencies

- **streamlit**: Web application framework
- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Text embeddings
- **rank-bm25**: BM25 search implementation
- **pypdf**: PDF parsing
- **python-docx**: Word document parsing
- **openpyxl**: Excel file parsing
- **pandas**: Data manipulation
- **huggingface_hub**: Hugging Face Inference API (default)
- **openai**: OpenAI API client (optional)
- **anthropic**: Anthropic API client (optional)

## 🎯 Use Cases

- 📄 **Document Q&A**: Ask questions about your documents
- 🔍 **Research Assistant**: Search through research papers
- 📚 **Knowledge Base**: Build a searchable knowledge base
- 💼 **Business Intelligence**: Query business documents
- 📖 **Study Aid**: Interact with textbooks and notes

## 🚀 Deployment

### Streamlit Cloud (Easiest)

1. Push to GitHub
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Add API key in Secrets
4. Deploy!

See [Deployment Guide](deployment/streamlit_cloud_guide.md) for details.

### Docker

```bash
docker-compose up
```

## 📝 License

MIT License

---

Made with ❤️ using Meta Llama 3.1 via Hugging Face
