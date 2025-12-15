"""
BM25 Search Module
Implements BM25 (Best Matching 25) algorithm for keyword-based document retrieval.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import string


class BM25Search:
    """
    BM25 search implementation for keyword-based document retrieval.
    Complements vector search by providing lexical matching capabilities.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 search.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: Floor for IDF values (default: 0.25)
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._documents = []
        self._tokenized_docs = []
        self._doc_metadata = []
        self._bm25 = None
        self._is_indexed = False

    def add_documents(self, chunks: List) -> int:
        """
        Add document chunks to the BM25 index.

        Args:
            chunks: List of TextChunk objects

        Returns:
            Number of documents added
        """
        for chunk in chunks:
            self._documents.append(chunk.content)
            self._doc_metadata.append(chunk.metadata)
            self._tokenized_docs.append(self._tokenize(chunk.content))

        # Rebuild index
        self._build_index()

        return len(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Split on whitespace
        tokens = text.split()

        # Remove stopwords (basic English stopwords)
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'also',
        }

        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

        return tokens

    def _build_index(self) -> None:
        """Build or rebuild the BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            )

        if self._tokenized_docs:
            self._bm25 = BM25Okapi(
                self._tokenized_docs,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )
            self._is_indexed = True
        else:
            self._bm25 = None
            self._is_indexed = False

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of search results with content, metadata, and score
        """
        if not self._is_indexed or self._bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Get top N results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    'content': self._documents[idx],
                    'metadata': self._doc_metadata[idx],
                    'score': float(scores[idx]),
                    'index': idx,
                })

        return results

    def search_with_scores(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search and return results with scores.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of tuples (content, score, metadata)
        """
        results = self.search(query, n_results)
        return [
            (r['content'], r['score'], r['metadata'])
            for r in results
        ]

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []
        self._tokenized_docs = []
        self._doc_metadata = []
        self._bm25 = None
        self._is_indexed = False

    def count(self) -> int:
        """Get the number of documents in the index."""
        return len(self._documents)

    def get_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents in the index.

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of all documents with metadata
        """
        return [
            {
                'content': self._documents[i],
                'metadata': self._doc_metadata[i],
                'index': i,
            }
            for i in range(min(limit, len(self._documents)))
        ]


class HybridSearch:
    """
    Combines BM25 and vector search for hybrid retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(
        self,
        vector_store,
        bm25_search: BM25Search,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search.

        Args:
            vector_store: VectorStore instance
            bm25_search: BM25Search instance
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results
            rrf_k: RRF constant (default: 60)
        """
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        n_results: int = 5,
        n_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            n_results: Number of final results to return
            n_candidates: Number of candidates to fetch from each search method

        Returns:
            List of search results with combined scores
        """
        # Get results from both search methods
        vector_results = self.vector_store.search(query, n_results=n_candidates)
        bm25_results = self.bm25_search.search(query, n_results=n_candidates)

        # Create content-to-result mapping
        results_map = {}

        # Process vector search results
        for rank, result in enumerate(vector_results):
            content = result['content']
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)

            if content not in results_map:
                results_map[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'vector_score': result.get('score', 0),
                    'bm25_score': 0,
                    'rrf_score': 0,
                    'vector_rank': rank + 1,
                    'bm25_rank': None,
                }

            results_map[content]['rrf_score'] += rrf_score

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            content = result['content']
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)

            if content not in results_map:
                results_map[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'vector_score': 0,
                    'bm25_score': result.get('score', 0),
                    'rrf_score': 0,
                    'vector_rank': None,
                    'bm25_rank': rank + 1,
                }
            else:
                results_map[content]['bm25_score'] = result.get('score', 0)
                results_map[content]['bm25_rank'] = rank + 1

            results_map[content]['rrf_score'] += rrf_score

        # Sort by RRF score and return top N
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        return sorted_results[:n_results]
