"""
Retrieval Engine
Main RAG pipeline that combines document processing, retrieval, and generation.
"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import logging

from .document_processor import DocumentProcessor, Document
from .vector_stores.chroma import ChromaVectorStore
from .llm_providers.factory import LLMClient
from utils.bm25_search import BM25Search, HybridSearch

logger = logging.getLogger("rag_app.retrieval_engine")


class SearchMode(Enum):
    """Search mode options."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    search_mode: str


class RAGPipeline:
    """
    End-to-end RAG pipeline that combines:
    - Document loading and processing
    - Text chunking
    - Vector storage (ChromaDB)
    - BM25 keyword search
    - Hybrid search
    - LLM-based answer generation
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the user's question. If you cannot find the answer in the context,
say so clearly. Always cite your sources when possible.

Be concise but thorough in your responses. If the context contains relevant information,
use it to provide a comprehensive answer."""

    DEFAULT_RAG_PROMPT_TEMPLATE = """Context:
{context}

---

Question: {question}

Please provide a detailed answer based on the context above. If the information to answer
the question is not present in the context, clearly state that you cannot find the answer
in the provided documents."""

    def __init__(
        self,
        llm_provider: str = "groq",
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        search_mode: SearchMode = SearchMode.HYBRID,
        n_results: int = 5,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            llm_provider: LLM provider ('groq', 'openai', or 'anthropic')
            llm_api_key: API key for the LLM provider
            llm_model: Model name for the LLM
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            search_mode: Search mode (vector, bm25, or hybrid)
            n_results: Number of results to retrieve
        """
        logger.info("Initializing RAG Pipeline...")
        logger.info(f"  LLM Provider: {llm_provider}")
        logger.info(f"  LLM Model: {llm_model or 'default'}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Embedding Model: {embedding_model}")
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info(f"  Chunk Overlap: {chunk_overlap}")
        logger.info(f"  Search Mode: {search_mode.value}")
        logger.info(f"  N Results: {n_results}")

        # Store configuration
        self.search_mode = search_mode
        self.n_results = n_results
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        logger.info("Initializing document processor...")
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info("Initializing vector store (ChromaDB)...")
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )

        logger.info("Initializing BM25 search...")
        self.bm25_search = BM25Search()

        logger.info("Initializing hybrid search...")
        self.hybrid_search = HybridSearch(
            vector_store=self.vector_store,
            bm25_search=self.bm25_search,
        )

        # Initialize LLM (can be set later)
        self._llm = None
        self._llm_config = {
            'provider': llm_provider,
            'api_key': llm_api_key,
            'model': llm_model,
        }

        # Prompt configuration
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.rag_prompt_template = self.DEFAULT_RAG_PROMPT_TEMPLATE

        logger.info("RAG Pipeline initialization complete")

    def _get_llm(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm is None:
            logger.info("Creating LLM client on first use...")
            self._llm = LLMClient(
                provider=self._llm_config['provider'],
                api_key=self._llm_config['api_key'],
                model=self._llm_config['model'],
            )
            logger.info("LLM client created successfully")
        return self._llm

    def add_documents(
        self,
        file_paths: Optional[List[str]] = None,
        file_bytes_list: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents to the RAG pipeline.

        Args:
            file_paths: List of file paths to load
            file_bytes_list: List of (bytes, filename) tuples

        Returns:
            Summary of added documents
        """
        logger.info("Adding documents to RAG pipeline...")
        all_documents = []

        # Load from bytes
        if file_bytes_list:
            logger.info(f"Loading {len(file_bytes_list)} files from bytes...")
            for file_bytes, filename in file_bytes_list:
                try:
                    logger.info(f"  Loading: {filename}")
                    docs = self.document_processor.load_from_bytes(file_bytes, filename)
                    all_documents.extend(docs)
                    logger.info(f"    Loaded {len(docs)} document(s)")
                except Exception as e:
                    logger.error(f"  Error loading {filename}: {e}")

        if not all_documents:
            logger.warning("No documents were loaded successfully")
            return {'documents': 0, 'chunks': 0}

        logger.info(f"Total documents loaded: {len(all_documents)}")

        # Split into chunks
        logger.info("Splitting documents into chunks...")
        chunks = self.document_processor.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Add to both vector store and BM25 index
        logger.info("Adding chunks to vector store...")
        num_vector = self.vector_store.add_documents(chunks)
        logger.info(f"Vector store now contains {num_vector} documents")

        logger.info("Adding chunks to BM25 index...")
        num_bm25 = self.bm25_search.add_documents(chunks)
        logger.info(f"BM25 index now contains {num_bm25} documents")

        logger.info("Document indexing complete")

        return {
            'documents': len(all_documents),
            'chunks': len(chunks),
            'vector_store_count': num_vector,
            'bm25_count': num_bm25,
        }

    def search(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            mode: Search mode (defaults to pipeline setting)
            n_results: Number of results (defaults to pipeline setting)

        Returns:
            List of search results
        """
        mode = mode or self.search_mode
        n_results = n_results or self.n_results

        logger.info(f"Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        logger.info(f"  Mode: {mode.value}")
        logger.info(f"  Requested results: {n_results}")

        if mode == SearchMode.VECTOR:
            logger.info("  Using vector (semantic) search...")
            results = self.vector_store.search(query, n_results=n_results)
        elif mode == SearchMode.BM25:
            logger.info("  Using BM25 (keyword) search...")
            results = self.bm25_search.search(query, n_results=n_results)
        else:  # HYBRID
            logger.info("  Using hybrid (vector + BM25) search...")
            results = self.hybrid_search.search(query, n_results=n_results)

        logger.info(f"  Found {len(results)} results")
        return results

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM."""
        context_parts = []

        for i, result in enumerate(results, 1):
            source = result.get('metadata', {}).get('source', 'Unknown')
            content = result.get('content', '')
            score_info = ""

            if 'score' in result:
                score_info = f" (relevance: {result['score']:.2f})"
            elif 'rrf_score' in result:
                score_info = f" (relevance: {result['rrf_score']:.4f})"

            context_parts.append(
                f"[Source {i}: {source}{score_info}]\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def query_stream(
        self,
        question: str,
        mode: Optional[SearchMode] = None,
        n_results: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """
        Query the RAG pipeline with streaming response.

        Args:
            question: User's question
            mode: Search mode
            n_results: Number of results to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Yields:
            Tokens from the LLM response
        """
        logger.info("Processing RAG query (streaming)...")
        logger.info(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")

        # Search for relevant documents
        results = self.search(question, mode=mode, n_results=n_results)

        if not results:
            logger.warning("No relevant documents found for query")
            yield "I couldn't find any relevant information in the documents to answer your question."
            return

        # Format context
        logger.info("Formatting context from search results...")
        context = self._format_context(results)
        logger.info(f"  Context length: {len(context)} chars")

        # Create prompt
        prompt = self.rag_prompt_template.format(
            context=context,
            question=question,
        )

        # Generate streaming answer
        logger.info("Starting streaming response from LLM...")
        llm = self._get_llm()
        for token in llm.generate_stream(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token
        logger.info("Streaming response complete")

    def clear(self) -> None:
        """Clear all documents from the pipeline."""
        logger.info("Clearing all documents from pipeline...")
        self.vector_store.clear()
        logger.info("  Vector store cleared")
        self.bm25_search.clear()
        logger.info("  BM25 index cleared")
        logger.info("Pipeline cleared successfully")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'vector_store_count': self.vector_store.count(),
            'bm25_count': self.bm25_search.count(),
            'search_mode': self.search_mode.value,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'n_results': self.n_results,
        }


def create_pipeline(
    api_key: str,
    search_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    n_results: int,
) -> RAGPipeline:
    """
    Create a new RAG pipeline with Groq Llama 3.1 (optimized for latency).

    Args:
        api_key: Groq API key
        search_mode: Search mode string
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks
        n_results: Number of results to retrieve

    Returns:
        Initialized RAGPipeline
    """
    logger.info("=" * 60)
    logger.info("CREATING NEW RAG PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Provider: Groq")
    logger.info(f"Model: llama-3.1-8b-instant")
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
        llm_provider="groq",
        llm_api_key=api_key,
        llm_model="llama-3.1-8b-instant",
        search_mode=mode_map.get(search_mode, SearchMode.HYBRID),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        n_results=n_results,
    )

    logger.info("RAG Pipeline created successfully")
    logger.info("=" * 60)

    return pipeline
