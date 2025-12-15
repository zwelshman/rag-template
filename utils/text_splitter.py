"""
Text Splitter Module
Handles splitting documents into chunks for embedding and retrieval.
"""

from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: dict
    chunk_index: int

    @property
    def id(self) -> str:
        """Generate a unique ID for the chunk."""
        source = self.metadata.get('source', 'unknown')
        return f"{source}_chunk_{self.chunk_index}"


class TextSplitter:
    """
    Text splitter that chunks documents while preserving semantic coherence.
    Supports multiple splitting strategies.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        length_function: Optional[callable] = None,
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk (in characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            separator: Primary separator to use for splitting
            length_function: Function to calculate length (default: len)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.length_function = length_function or len

        # Fallback separators in order of preference
        self._separators = [
            "\n\n",  # Double newline (paragraphs)
            "\n",    # Single newline
            ". ",    # Sentence ending
            "! ",    # Exclamation
            "? ",    # Question
            "; ",    # Semicolon
            ", ",    # Comma
            " ",     # Space
            "",      # Character level (last resort)
        ]

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # If text is smaller than chunk size, return as is
        if self.length_function(text) <= self.chunk_size:
            return [text.strip()]

        return self._recursive_split(text, self._separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using available separators."""
        final_chunks = []

        # Find the best separator to use
        separator = separators[-1]  # Default to last separator
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split by the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks of appropriate size
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            # If a single split is too large, recursively split it
            if split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if chunk_text.strip():
                        final_chunks.append(chunk_text.strip())
                    current_chunk = []
                    current_length = 0

                # Use next separator level
                if separator in separators:
                    idx = separators.index(separator)
                    if idx < len(separators) - 1:
                        sub_chunks = self._recursive_split(split, separators[idx + 1:])
                        final_chunks.extend(sub_chunks)
                    else:
                        # Force split at chunk_size if no more separators
                        for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                            final_chunks.append(split[i:i + self.chunk_size].strip())
                continue

            # Check if adding this split exceeds chunk size
            potential_length = current_length + split_length
            if current_chunk:
                potential_length += len(separator)

            if potential_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                if chunk_text.strip():
                    final_chunks.append(chunk_text.strip())

                # Handle overlap
                overlap_chunks = []
                overlap_length = 0
                for chunk in reversed(current_chunk):
                    chunk_len = self.length_function(chunk)
                    if overlap_length + chunk_len <= self.chunk_overlap:
                        overlap_chunks.insert(0, chunk)
                        overlap_length += chunk_len + len(separator)
                    else:
                        break

                current_chunk = overlap_chunks
                current_length = overlap_length

            current_chunk.append(split)
            current_length += split_length + (len(separator) if current_chunk else 0)

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                final_chunks.append(chunk_text.strip())

        return final_chunks

    def split_documents(self, documents: List) -> List[TextChunk]:
        """
        Split a list of documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        chunk_index = 0

        for doc in documents:
            text_chunks = self.split_text(doc.content)

            for i, chunk_text in enumerate(text_chunks):
                chunk = TextChunk(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_index_in_doc': i,
                        'total_chunks_in_doc': len(text_chunks),
                    },
                    chunk_index=chunk_index,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks


class SentenceSplitter(TextSplitter):
    """
    Text splitter that tries to keep sentences intact.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Sentence-ending pattern
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def split_text(self, text: str) -> List[str]:
        """Split text by sentences while respecting chunk size."""
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._sentence_pattern.split(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = self.length_function(sentence)

            # If single sentence is too long, use parent's recursive split
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence
                sub_chunks = super().split_text(sentence)
                chunks.extend(sub_chunks)
                continue

            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length + 1 > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Handle overlap - keep last few sentences that fit in overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_len = self.length_function(s)
                    if overlap_length + s_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_len + 1
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
