# RAG Template

A comprehensive Streamlit application template for demonstrating Retrieval-Augmented Generation (RAG) using ChromaDB for vector search and BM25 for keyword-based search.

## Features

- **Multiple File Format Support**
  - Text files (.txt)
  - PDF documents (.pdf)
  - Word documents (.docx)
  - Excel spreadsheets (.xlsx, .xls)
  - CSV files (.csv)

- **Hybrid Search**
  - **Vector Search**: Semantic similarity using sentence transformers and ChromaDB
  - **BM25 Search**: Traditional keyword-based retrieval
  - **Hybrid Search**: Combines both methods using Reciprocal Rank Fusion (RRF)

- **LLM Integration**
  - OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
  - Anthropic Claude (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
  - Streaming responses

- **Configurable Pipeline**
  - Adjustable chunk size and overlap
  - Configurable number of retrieved results
  - Temperature and max tokens control

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-template
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the App

1. **Configure the Pipeline** (Sidebar)
   - Select your LLM provider (OpenAI or Anthropic)
   - Choose a model
   - Enter your API key
   - Adjust search and chunking settings
   - Click "Initialize Pipeline"

2. **Upload Documents** (Tab 1)
   - Upload one or more supported documents
   - Click "Process Documents" to index them

3. **Ask Questions** (Tab 2)
   - Type your question in the chat input
   - View the AI-generated answer with source citations
   - Expand "View Sources" to see the retrieved context

4. **Search Demo** (Tab 3)
   - Test different search modes directly
   - Compare vector, BM25, and hybrid search results

### Environment Variables

You can set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

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
    llm_provider="openai",
    llm_api_key="your-api-key",
    llm_model="gpt-4o-mini",
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
- **openai**: OpenAI API client
- **anthropic**: Anthropic API client

## License

MIT License
