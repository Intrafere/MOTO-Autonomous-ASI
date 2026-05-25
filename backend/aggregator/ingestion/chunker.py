"""
Multi-configuration chunking for RAG system.
Generates chunks at different sizes (256/512/768/1024 chars) with 20% overlap.
"""
from typing import List, Dict
from backend.shared.config import rag_config
from backend.shared.models import DocumentChunk
from backend.shared.utils import generate_chunk_id, split_into_sentences
from backend.aggregator.ingestion.metadata_extractor import extract_chunk_metadata


class Chunker:
    """Handles multi-configuration document chunking."""
    
    def __init__(self):
        self.chunk_sizes = rag_config.submitter_chunk_intervals
        self.overlap_percentage = rag_config.chunk_overlap_percentage
    
    def chunk_text(
        self,
        text: str,
        source_file: str,
        chunk_sizes: List[int] = None,
        is_user_file: bool = False
    ) -> Dict[int, List[DocumentChunk]]:
        """
        Chunk text at multiple sizes.
        
        Args:
            text: Text to chunk
            source_file: Source file name
            chunk_sizes: Sizes to chunk at (if None, uses all configs)
            is_user_file: Whether this is a user-uploaded file (never evicted)
        
        Returns:
            Dict mapping chunk_size -> list of DocumentChunks
        """
        if chunk_sizes is None:
            chunk_sizes = self.chunk_sizes
        
        result = {}
        for chunk_size in chunk_sizes:
            chunks = self._chunk_at_size(text, source_file, chunk_size, is_user_file)
            result[chunk_size] = chunks
        
        return result
    
    def _chunk_at_size(
        self,
        text: str,
        source_file: str,
        chunk_size: int,
        is_user_file: bool
    ) -> List[DocumentChunk]:
        """Chunk text at a specific size with semantic boundaries."""
        overlap = rag_config.get_chunk_overlap(chunk_size)
        chunks = []
        
        # Split into sentences for semantic chunking
        sentences = split_into_sentences(text)
        
        current_chunk = ""
        position_counter = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                # Create chunk
                chunk_id = generate_chunk_id(source_file, position_counter, chunk_size)
                metadata = extract_chunk_metadata(
                    current_chunk,
                    source_file,
                    position_counter,
                    chunk_size
                )
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk.strip(),
                    source_file=source_file,
                    position=position_counter,
                    chunk_size=chunk_size,
                    chunk_type=metadata['chunk_type'],
                    metadata=metadata,
                    is_user_file=is_user_file,
                    is_permanent=is_user_file
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                # Keep last 'overlap' characters
                if overlap > 0 and len(current_chunk) > overlap:
                    # Find sentence boundary for overlap
                    overlap_text = current_chunk[-overlap:]
                    # Try to start at sentence boundary
                    sentences_in_overlap = split_into_sentences(overlap_text)
                    if sentences_in_overlap:
                        current_chunk = ' '.join(sentences_in_overlap)
                    else:
                        current_chunk = overlap_text
                else:
                    current_chunk = ""
                
                position_counter += 1
            
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_id = generate_chunk_id(source_file, position_counter, chunk_size)
            metadata = extract_chunk_metadata(
                current_chunk,
                source_file,
                position_counter,
                chunk_size
            )
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                source_file=source_file,
                position=position_counter,
                chunk_size=chunk_size,
                chunk_type=metadata['chunk_type'],
                metadata=metadata,
                is_user_file=is_user_file,
                is_permanent=is_user_file
            )
            chunks.append(chunk)
        
        return chunks


# Global chunker instance
chunker = Chunker()

