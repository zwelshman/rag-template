"""
Base Vector Store Interface
Abstract base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    Defines the interface that all vector stores must implement.
    """

    @abstractmethod
    def add_documents(self, chunks: List, batch_size: int = 100) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of TextChunk objects
            batch_size: Number of documents to add per batch

        Returns:
            Number of documents added
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the vector store."""
        pass
