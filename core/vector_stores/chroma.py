"""
ChromaDB Vector Store Implementation
"""

from typing import List, Dict, Any, Optional
import os
from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store for document embeddings.
    Supports both persistent and in-memory storage.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (None for in-memory)
            embedding_model: Sentence transformer model for embeddings
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Initialize ChromaDB client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Set up embedding function
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List, batch_size: int = 100) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of TextChunk objects
            batch_size: Number of documents to add per batch

        Returns:
            Number of documents added
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.id
            ids.append(chunk_id)
            documents.append(chunk.content)
            # Ensure metadata values are serializable
            metadata = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }
            metadatas.append(metadata)

        # Add in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]

            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )

        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Filter conditions for metadata

        Returns:
            List of search results with content, metadata, and score
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'score': 1 - results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None,
                })

        return formatted_results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    @property
    def client(self):
        """Get the underlying ChromaDB client."""
        return self._client

    @property
    def collection(self):
        """Get the underlying ChromaDB collection."""
        return self._collection
