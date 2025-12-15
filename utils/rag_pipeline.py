"""
RAG Pipeline Module
Combines document loading, chunking, indexing, retrieval, and generation
into a unified pipeline for Retrieval-Augmented Generation.
"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum

from .document_loader import DocumentLoader, Document
from .text_splitter import TextSplitter, TextChunk
from .vector_store import VectorStore
from .bm25_search import BM25Search, HybridSearch
from .llm import LLMClient


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
        llm_provider: str = "openai",
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
            llm_provider: LLM provider ('openai' or 'anthropic')
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
        # Store configuration
        self.search_mode = search_mode
        self.n_results = n_results
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
        self.bm25_search = BM25Search()
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

    def _get_llm(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm is None:
            self._llm = LLMClient(
                provider=self._llm_config['provider'],
                api_key=self._llm_config['api_key'],
                model=self._llm_config['model'],
            )
        return self._llm

    def set_llm(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
    ) -> None:
        """
        Set or update the LLM configuration.

        Args:
            provider: LLM provider
            api_key: API key
            model: Model name
        """
        self._llm_config = {
            'provider': provider,
            'api_key': api_key,
            'model': model,
        }
        self._llm = None  # Reset to create new client on next use

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
        all_documents = []

        # Load from file paths
        if file_paths:
            for path in file_paths:
                try:
                    docs = self.document_loader.load(path)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        # Load from bytes
        if file_bytes_list:
            for file_bytes, filename in file_bytes_list:
                try:
                    docs = self.document_loader.load_from_bytes(file_bytes, filename)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        if not all_documents:
            return {'documents': 0, 'chunks': 0}

        # Split into chunks
        chunks = self.text_splitter.split_documents(all_documents)

        # Add to both vector store and BM25 index
        num_vector = self.vector_store.add_documents(chunks)
        num_bm25 = self.bm25_search.add_documents(chunks)

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

        if mode == SearchMode.VECTOR:
            return self.vector_store.search(query, n_results=n_results)
        elif mode == SearchMode.BM25:
            return self.bm25_search.search(query, n_results=n_results)
        else:  # HYBRID
            return self.hybrid_search.search(query, n_results=n_results)

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

    def query(
        self,
        question: str,
        mode: Optional[SearchMode] = None,
        n_results: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.

        Args:
            question: User's question
            mode: Search mode
            n_results: Number of results to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens in response

        Returns:
            RAGResponse with answer and sources
        """
        # Search for relevant documents
        results = self.search(question, mode=mode, n_results=n_results)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                query=question,
                search_mode=(mode or self.search_mode).value,
            )

        # Format context
        context = self._format_context(results)

        # Create prompt
        prompt = self.rag_prompt_template.format(
            context=context,
            question=question,
        )

        # Generate answer
        llm = self._get_llm()
        answer = llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return RAGResponse(
            answer=answer,
            sources=results,
            query=question,
            search_mode=(mode or self.search_mode).value,
        )

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
        # Search for relevant documents
        results = self.search(question, mode=mode, n_results=n_results)

        if not results:
            yield "I couldn't find any relevant information in the documents to answer your question."
            return

        # Format context
        context = self._format_context(results)

        # Create prompt
        prompt = self.rag_prompt_template.format(
            context=context,
            question=question,
        )

        # Generate streaming answer
        llm = self._get_llm()
        for token in llm.generate_stream(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    def clear(self) -> None:
        """Clear all documents from the pipeline."""
        self.vector_store.clear()
        self.bm25_search.clear()

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

    def set_prompts(
        self,
        system_prompt: Optional[str] = None,
        rag_prompt_template: Optional[str] = None,
    ) -> None:
        """
        Update prompt templates.

        Args:
            system_prompt: New system prompt
            rag_prompt_template: New RAG prompt template (must include {context} and {question})
        """
        if system_prompt:
            self.system_prompt = system_prompt

        if rag_prompt_template:
            # Validate template has required placeholders
            if '{context}' not in rag_prompt_template or '{question}' not in rag_prompt_template:
                raise ValueError("RAG prompt template must contain {context} and {question} placeholders")
            self.rag_prompt_template = rag_prompt_template
