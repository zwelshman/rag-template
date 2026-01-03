"""
Core RAG Engine
Contains the main logic for document processing, retrieval, and generation.
"""

from .document_processor import DocumentProcessor, Document, SUPPORTED_EXTENSIONS
from .retrieval_engine import RAGPipeline, SearchMode, create_pipeline

__all__ = [
    'DocumentProcessor',
    'Document',
    'SUPPORTED_EXTENSIONS',
    'RAGPipeline',
    'SearchMode',
    'create_pipeline',
]
