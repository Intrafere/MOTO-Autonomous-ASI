"""
Common utility functions for the ASI Aggregator System.
"""
import tiktoken
import re
from typing import List
import logging

logger = logging.getLogger(__name__)

# The hosted Dockerfile bakes this encoding into TIKTOKEN_CACHE_DIR at build time.
TIKTOKEN_ENCODING_NAME = "cl100k_base"


def count_tokens(text: str, encoding_name: str = TIKTOKEN_ENCODING_NAME) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to count tokens: {e}. Using approximation.")
        # Fallback: approximate 1 token ≈ 4 characters
        return len(text) // 4


def compress_text(text: str, preserve_entities: bool = True) -> str:
    """
    Compress text while preserving important entities, numbers, and dates.
    """
    if not text:
        return text
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove redundant phrases (if not preserving all content)
    if not preserve_entities:
        redundant_patterns = [
            r'\b(very|really|quite|rather|actually|basically|literally)\b',
            r'\b(in order to)\b',
            r'\b(due to the fact that)\b',
        ]
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up spacing again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for semantic chunking."""
    # Simple sentence splitter (can be enhanced with NLTK/spaCy)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_citations(text: str) -> List[int]:
    """Extract [E#] citation markers from text."""
    pattern = r'\[E(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def truncate_with_ellipsis(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def generate_chunk_id(source_file: str, position: int, chunk_size: int) -> str:
    """Generate a unique chunk ID."""
    return f"{source_file}::{position}::{chunk_size}"

