"""
RAG Manager - 4-stage retrieval pipeline orchestrator.
Stages: Query Rewriting -> Hybrid Recall -> Reranking+MMR -> Packing+Compression
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from collections import OrderedDict
import asyncio
import logging
import hashlib
import time
from pathlib import Path

from backend.shared.config import rag_config, system_config
from backend.shared.models import DocumentChunk, ContextPack
from backend.shared.api_client_manager import api_client_manager
from backend.shared.rag_lock import rag_operation_lock
from backend.shared.utils import count_tokens, compress_text
from backend.aggregator.ingestion.pipeline import ingestion_pipeline

logger = logging.getLogger(__name__)


class RAGManager:
    """
    RAG Manager with 4-stage retrieval pipeline.
    """
    
    def __init__(self):
        # ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=system_config.chroma_db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections for different chunk sizes
        self.collections = {}
        for size in rag_config.submitter_chunk_intervals:
            collection_name = f"chunks_{size}"
            self.collections[size] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"chunk_size": size}
            )
        
        # In-memory chunk storage for BM25
        self.chunks_by_size: Dict[int, List[DocumentChunk]] = {
            size: [] for size in rag_config.submitter_chunk_intervals
        }
        
        # BM25 index (rebuilt when chunks change)
        self.bm25_index: Dict[int, Optional[BM25Okapi]] = {
            size: None for size in rag_config.submitter_chunk_intervals
        }
        
        # Caches
        self.rewrite_cache: OrderedDict = OrderedDict()
        self.bm25_cache: OrderedDict = OrderedDict()
        self.context_pack_cache: OrderedDict = OrderedDict()
        
        # Document tracking
        self.document_count = 0
        self.permanent_documents = set()  # User files never evicted
        self.document_access_order: OrderedDict = OrderedDict()  # LRU tracking: source_name -> last_access_time
    
    async def add_document(
        self,
        file_path: str,
        chunk_sizes: List[int] = None,
        is_user_file: bool = False
    ) -> None:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to document
            chunk_sizes: Sizes to chunk at (None = all configs for user files)
            is_user_file: Whether this is a user file (never evicted)
        """
        try:
            # Ingest document
            chunks_by_size = await ingestion_pipeline.ingest_file(
                file_path,
                chunk_sizes,
                is_user_file
            )
            
            # Add to ChromaDB and memory
            for chunk_size, chunks in chunks_by_size.items():
                await self._add_chunks(chunks, chunk_size)
            
            # Track document (only increment count for genuinely new sources)
            source_name = Path(file_path).name
            if source_name not in self.document_access_order:
                self.document_count += 1
            self.document_access_order[source_name] = time.time()
            if is_user_file:
                self.permanent_documents.add(source_name)
            
            # Check if need to evict
            if self.document_count > rag_config.max_documents:
                await self._evict_lru_document()
            
            # Enforce per-size chunk cap
            await self._enforce_chunk_cap()
            
            logger.info(f"Added document: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            raise
    
    async def add_text(
        self,
        text: str,
        source_name: str,
        chunk_sizes: List[int] = None,
        is_permanent: bool = False
    ) -> None:
        """
        Add raw text to the RAG system.
        
        Args:
            text: Text content
            source_name: Name for this content
            chunk_sizes: Sizes to chunk at
            is_permanent: Whether to protect from eviction
        """
        try:
            # Ingest text
            chunks_by_size = await ingestion_pipeline.ingest_text(
                text,
                source_name,
                chunk_sizes,
                is_permanent
            )
            
            # Add to ChromaDB and memory
            for chunk_size, chunks in chunks_by_size.items():
                await self._add_chunks(chunks, chunk_size)
            
            # Track document (only increment count for genuinely new sources)
            if source_name not in self.document_access_order:
                self.document_count += 1
            self.document_access_order[source_name] = time.time()
            if is_permanent:
                self.permanent_documents.add(source_name)
            
            # Check if need to evict
            if self.document_count > rag_config.max_documents:
                await self._evict_lru_document()
            
            # Enforce per-size chunk cap
            await self._enforce_chunk_cap()
            
            logger.info(f"Added text: {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to add text {source_name}: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        chunk_size: int = 512,
        max_tokens: int = None,
        exclude_sources: Optional[List[str]] = None
    ) -> ContextPack:
        """
        4-stage retrieval pipeline.
        
        Args:
            query: Search query
            chunk_size: Chunk size to retrieve from
            max_tokens: Maximum tokens in result
            exclude_sources: Source names to skip during packing (already direct-injected)
        
        Returns:
            ContextPack with retrieved context
        """
        max_tokens = max_tokens or rag_config.get_available_input_tokens(rag_config.submitter_context_window, rag_config.submitter_max_output_tokens)
        
        # Stage A: Query Rewriting
        logger.debug(f"RAG Stage 1/4: Query rewriting for '{query[:50]}...'")
        queries = await self._rewrite_query(query)
        logger.debug(f"RAG Stage 1/4 complete: Generated {len(queries)} query variants")
        
        # Stage B: Hybrid Recall (BM25 + Vector)
        logger.debug(f"RAG Stage 2/4: Hybrid recall (BM25 + Vector) with chunk_size={chunk_size}")
        candidates = await self._hybrid_recall(queries, chunk_size)
        logger.debug(f"RAG Stage 2/4 complete: Retrieved {len(candidates)} candidate chunks")
        
        # Stage C: Reranking + MMR
        logger.debug(f"RAG Stage 3/4: Reranking and MMR diversification")
        ranked_chunks = self._rerank_and_diversify(candidates, query, chunk_size)
        logger.debug(f"RAG Stage 3/4 complete: Ranked to {len(ranked_chunks)} chunks")
        
        # Stage D: Packing + Compression
        logger.debug(f"RAG Stage 4/4: Packing and compression (max_tokens={max_tokens})")
        if exclude_sources:
            logger.info(f"RAG Stage 4/4: Excluding sources already direct-injected: {exclude_sources}")
        context_pack = await self._pack_and_compress(ranked_chunks, query, max_tokens, exclude_sources)
        logger.debug(f"RAG Stage 4/4 complete: Packed {len(context_pack.evidence)} evidence items, coverage={context_pack.coverage:.2f}")
        
        return context_pack
    
    async def _add_chunks(self, chunks: List[DocumentChunk], chunk_size: int) -> None:
        """Add chunks to ChromaDB and memory with global lock."""
        if not chunks:
            return
        
        texts = [chunk.text for chunk in chunks]

        if system_config.generic_mode:
            embeddings = await api_client_manager.get_embeddings(texts)
            await rag_operation_lock.acquire(f"RAGManager add_chunks write (size={chunk_size})")
        else:
            await rag_operation_lock.acquire(f"RAGManager add_chunks (size={chunk_size})")
            embeddings = await api_client_manager.get_embeddings(texts)

        try:
            # Update chunks with embeddings and tokens
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                chunk.tokens = chunk.text.lower().split()

            # ChromaDB writes stay under the global RAG lock in both modes.
            collection = self.collections[chunk_size]
            try:
                collection.add(
                    ids=[chunk.chunk_id for chunk in chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[chunk.metadata for chunk in chunks]
                )
                logger.debug(f"Added {len(chunks)} chunks to ChromaDB collection (size={chunk_size})")
            except Exception as e:
                logger.error(f"CRITICAL: ChromaDB add failed for chunk_size={chunk_size}: {type(e).__name__}: {e}")
                logger.error(f"Attempting to add {len(chunks)} chunks with IDs: {[c.chunk_id for c in chunks][:5]}...")
                raise

            # Add to memory
            self.chunks_by_size[chunk_size].extend(chunks)

            # Invalidate BM25 index for this size
            self.bm25_index[chunk_size] = None
        finally:
            rag_operation_lock.release()
    
    async def _rewrite_query(self, query: str) -> List[str]:
        """Stage A: Expand query into semantic variants."""
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.rewrite_cache:
            return self.rewrite_cache[cache_key]
        
        # Filter short queries
        if len(query.split()) < 3:
            return [query]
        
        # Generate variants (simple approach - can be enhanced with LLM)
        queries = [query]
        
        # Add variations
        words = query.split()
        if len(words) > 3:
            # Add phrase without first/last word
            queries.append(' '.join(words[1:]))
            queries.append(' '.join(words[:-1]))
        
        # Limit to configured number
        queries = queries[:rag_config.query_rewrite_variants]
        
        # Cache
        self.rewrite_cache[cache_key] = queries
        if len(self.rewrite_cache) > rag_config.rewrite_cache_size:
            self.rewrite_cache.popitem(last=False)
        
        return queries
    
    async def _hybrid_recall(
        self,
        queries: List[str],
        chunk_size: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """Stage B: Hybrid BM25 + Vector search."""
        chunks = self.chunks_by_size[chunk_size]
        if not chunks:
            return []
        
        # Vector search
        vector_results = await self._vector_search(queries, chunk_size)
        
        # BM25 search
        bm25_results = self._bm25_search(queries, chunk_size)
        
        # Combine and deduplicate
        combined = {}
        for chunk, score in vector_results:
            combined[chunk.chunk_id] = (chunk, score * rag_config.vector_weight)
        
        for chunk, score in bm25_results:
            if chunk.chunk_id in combined:
                chunk_obj, vec_score = combined[chunk.chunk_id]
                combined[chunk.chunk_id] = (chunk_obj, vec_score + score * rag_config.bm25_weight)
            else:
                combined[chunk.chunk_id] = (chunk, score * rag_config.bm25_weight)
        
        # Return top K
        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:rag_config.hybrid_recall_top_k * 2]
    
    async def _vector_search(
        self,
        queries: List[str],
        chunk_size: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """Vector similarity search with retry logic for HNSW index race conditions."""
        collection = self.collections[chunk_size]
        chunks = self.chunks_by_size[chunk_size]
        
        if not chunks:
            return []
        
        query_embeddings = await api_client_manager.get_embeddings(queries)
        all_results = []
        for query_embedding in query_embeddings:
            # Search with retry logic for transient HNSW errors during concurrent writes
            max_retries = 3
            retry_delay = 0.5  # Start with 500ms delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(rag_config.hybrid_recall_top_k, len(chunks))
                    )
                    break  # Success - exit retry loop
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    # Check if this is the specific HNSW index race condition error
                    if "hnsw" in error_str or "nothing found on disk" in error_str or "segment reader" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"ChromaDB HNSW index temporarily unavailable (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    # Re-raise non-HNSW errors or if max retries exceeded
                    raise
            else:
                # All retries failed
                if last_error:
                    logger.error(f"ChromaDB query failed after {max_retries} retries: {last_error}")
                    raise last_error
            
            # Map back to chunks
            for chunk_id, distance in zip(results['ids'][0], results['distances'][0]):
                chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
                if chunk:
                    # Convert distance to similarity (cosine distance -> similarity)
                    similarity = 1.0 - distance
                    all_results.append((chunk, similarity))
        
        # Deduplicate and return top
        seen = set()
        unique_results = []
        for chunk, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_results.append((chunk, score))
        
        return unique_results[:rag_config.hybrid_recall_top_k]
    
    def _bm25_search(
        self,
        queries: List[str],
        chunk_size: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """BM25 lexical search."""
        chunks = self.chunks_by_size[chunk_size]
        if not chunks:
            return []
        
        # Build or get BM25 index
        if self.bm25_index[chunk_size] is None:
            corpus = [chunk.tokens for chunk in chunks]
            self.bm25_index[chunk_size] = BM25Okapi(corpus)
        
        bm25 = self.bm25_index[chunk_size]
        
        all_scores = np.zeros(len(chunks))
        for query in queries:
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            all_scores += scores
        
        # Normalize scores
        if all_scores.max() > 0:
            all_scores = all_scores / all_scores.max()
        
        # Get top results
        top_indices = np.argsort(all_scores)[::-1][:rag_config.hybrid_recall_top_k]
        results = [(chunks[i], float(all_scores[i])) for i in top_indices if all_scores[i] > 0]
        
        return results
    
    def _rerank_and_diversify(
        self,
        candidates: List[Tuple[DocumentChunk, float]],
        query: str,
        chunk_size: int
    ) -> List[DocumentChunk]:
        """Stage C: Reranking with MMR diversification."""
        if not candidates:
            return []
        
        # Apply MMR (Maximal Marginal Relevance)
        selected = []
        remaining = candidates.copy()
        
        while remaining and len(selected) < rag_config.hybrid_recall_top_k:
            if not selected:
                # Select most relevant
                best_idx = 0
                selected.append(remaining[best_idx][0])
                remaining.pop(best_idx)
            else:
                # Balance relevance and diversity
                best_score = -float('inf')
                best_idx = 0
                
                for idx, (chunk, relevance) in enumerate(remaining):
                    # Calculate diversity (min similarity to selected)
                    diversities = []
                    for sel_chunk in selected:
                        similarity = self._cosine_similarity(
                            chunk.embedding,
                            sel_chunk.embedding
                        )
                        diversities.append(similarity)
                    
                    diversity = 1.0 - min(diversities) if diversities else 1.0
                    
                    # MMR score
                    mmr_score = (
                        rag_config.mmr_lambda * relevance +
                        (1 - rag_config.mmr_lambda) * diversity
                    )
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                selected.append(remaining[best_idx][0])
                remaining.pop(best_idx)
        
        # Remove near-duplicates
        final = []
        for chunk in selected:
            is_duplicate = False
            for existing in final:
                similarity = self._cosine_similarity(chunk.embedding, existing.embedding)
                if similarity > rag_config.similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final.append(chunk)
        
        return final
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    async def _pack_and_compress(
        self,
        chunks: List[DocumentChunk],
        query: str,
        max_tokens: int,
        exclude_sources: Optional[List[str]] = None
    ) -> ContextPack:
        """
        Stage D: Pack chunks into ContextPack with strict token limit enforcement.
        
        CRITICAL: This function MUST NOT exceed max_tokens. We pack chunks incrementally
        until we hit the limit, then stop. Compression is NOT used because it's unreliable.
        
        Chunks from exclude_sources are skipped (already direct-injected in the prompt).
        """
        if not chunks:
            return ContextPack(
                text="",
                evidence=[],
                source_map={},
                coverage=0.0,
                answerability=0.0,
                needs_more_context=True
            )
        
        exclude_set = set(exclude_sources) if exclude_sources else set()
        skipped_count = 0
        
        # Assemble evidence INCREMENTALLY until we hit max_tokens
        evidence = []
        source_map = {}
        assembled_text = []
        current_tokens = 0
        evidence_idx = 0
        
        for chunk in chunks:
            # Skip chunks from excluded sources (already direct-injected)
            if chunk.source_file in exclude_set:
                skipped_count += 1
                continue
            
            evidence_idx += 1
            
            # Format this chunk's evidence entry
            chunk_entry = f"[Evidence {evidence_idx} from {chunk.source_file}]\n{chunk.text}\n"
            chunk_tokens = count_tokens(chunk_entry)
            
            # Check if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > max_tokens:
                # Stop here - we've hit the limit
                logger.debug(f"RAG packing stopped at {evidence_idx-1} packed chunks ({current_tokens} tokens, limit={max_tokens})")
                break
            
            # Add this chunk
            evidence_entry = {
                "id": evidence_idx,
                "source": chunk.source_file,
                "text": chunk.text,
                "position": chunk.position
            }
            evidence.append(evidence_entry)
            source_map[f"E{evidence_idx}"] = chunk.source_file
            assembled_text.append(chunk_entry)
            current_tokens += chunk_tokens
            
            # Update LRU access time for this document
            if chunk.source_file in self.document_access_order:
                self.document_access_order[chunk.source_file] = time.time()
        
        if skipped_count > 0:
            logger.info(f"RAG packing: Skipped {skipped_count} chunks from excluded sources (already direct-injected)")
        
        full_text = "\n".join(assembled_text)
        token_count = current_tokens  # We already counted during packing
        
        # Calculate coverage and answerability (simplified)
        query_terms = set(query.lower().split())
        text_terms = set(full_text.lower().split())
        coverage = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
        
        # Answerability - heuristic based on chunk count and coverage
        answerability = min(1.0, len(chunks) / 10.0 * coverage)
        
        return ContextPack(
            text=full_text,
            evidence=evidence,
            source_map=source_map,
            coverage=coverage,
            answerability=answerability,
            metadata={
                "chunk_count": len(chunks),
                "token_count": token_count,
                "compressed": token_count > max_tokens
            },
            needs_more_context=coverage < rag_config.coverage_threshold
        )
    
    async def _enforce_chunk_cap(self) -> None:
        """Trim oldest non-permanent chunks when any size bucket exceeds max_chunks_per_size."""
        cap = rag_config.max_chunks_per_size
        for chunk_size in rag_config.submitter_chunk_intervals:
            chunks = self.chunks_by_size[chunk_size]
            if len(chunks) <= cap:
                continue

            overflow = len(chunks) - cap
            evict_ids = []
            keep = []
            removed = 0

            for chunk in chunks:
                if removed < overflow and not chunk.is_permanent:
                    evict_ids.append(chunk.chunk_id)
                    chunk.embedding = None
                    removed += 1
                else:
                    keep.append(chunk)

            if evict_ids:
                collection = self.collections[chunk_size]
                try:
                    collection.delete(ids=evict_ids)
                except Exception as e:
                    logger.error(f"ChromaDB delete during chunk cap enforcement (size={chunk_size}): {e}")

                self.chunks_by_size[chunk_size] = keep
                self.bm25_index[chunk_size] = None
                logger.info(f"Chunk cap enforced for size={chunk_size}: removed {len(evict_ids)} oldest non-permanent chunks ({len(keep)} remaining)")

    async def _evict_lru_document(self) -> None:
        """Evict least recently used document (except permanent ones)."""
        # Find oldest non-permanent document
        oldest_doc = None
        oldest_time = float('inf')
        
        for source_name, access_time in self.document_access_order.items():
            if source_name not in self.permanent_documents and access_time < oldest_time:
                oldest_time = access_time
                oldest_doc = source_name
        
        if oldest_doc is None:
            logger.warning("Document limit reached but no evictable documents found (all are permanent).")
            return
        
        # Evict the oldest document
        logger.info(f"LRU eviction: Removing oldest document '{oldest_doc}' (last accessed: {oldest_time})")
        
        try:
            await self.remove_document(oldest_doc)
            # Remove from access tracking
            if oldest_doc in self.document_access_order:
                del self.document_access_order[oldest_doc]
            logger.info(f"LRU eviction complete: '{oldest_doc}' removed successfully")
        except Exception as e:
            logger.error(f"LRU eviction failed for '{oldest_doc}': {e}")
    
    async def remove_document(self, source_name: str) -> None:
        """Remove a document from all collections."""
        was_tracked = source_name in self.document_access_order
        
        for chunk_size in rag_config.submitter_chunk_intervals:
            # Remove from memory
            self.chunks_by_size[chunk_size] = [
                c for c in self.chunks_by_size[chunk_size]
                if c.source_file != source_name
            ]
            
            # Remove from ChromaDB
            collection = self.collections[chunk_size]
            # Get IDs for this source
            results = collection.get(where={"source_file": source_name})
            if results['ids']:
                collection.delete(ids=results['ids'])
            
            # Invalidate BM25
            self.bm25_index[chunk_size] = None
        
        if was_tracked:
            self.document_count = max(0, self.document_count - 1)
        
        # Clean up LRU tracking
        if source_name in self.document_access_order:
            del self.document_access_order[source_name]
        if source_name in self.permanent_documents:
            self.permanent_documents.discard(source_name)
        
        logger.info(f"Removed document: {source_name}")
    
    def clear_all_documents(self) -> None:
        """Clear all documents from RAG database (synchronous for cleanup).
        
        Uses graceful degradation: clears what it can even if some operations fail.
        Only raises if critical operations (collection creation) fail.
        """
        logger.info("Clearing all documents from RAG database...")
        
        collection_errors = []
        
        try:
            # Delete all collections (non-critical if individual deletions fail)
            for chunk_size in list(self.collections.keys()):
                try:
                    self.chroma_client.delete_collection(f"chunks_{chunk_size}")
                    logger.info(f"Deleted collection chunks_{chunk_size}")
                except Exception as e:
                    collection_errors.append(f"chunks_{chunk_size}: {e}")
                    logger.warning(f"Failed to delete collection chunks_{chunk_size}: {e}")
            
            # Recreate fresh collections (CRITICAL - must succeed)
            self.collections = {}
            for size in rag_config.submitter_chunk_intervals:
                collection_name = f"chunks_{size}"
                try:
                    self.collections[size] = self.chroma_client.get_or_create_collection(
                        name=collection_name,
                        metadata={"chunk_size": size}
                    )
                    logger.info(f"Recreated collection {collection_name}")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to recreate collection {collection_name}: {e}")
                    raise  # Critical failure - cannot continue without collections
            
            # Clear in-memory storage (safe operations)
            self.chunks_by_size = {
                size: [] for size in rag_config.submitter_chunk_intervals
            }
            
            # Clear BM25 indices
            self.bm25_index = {
                size: None for size in rag_config.submitter_chunk_intervals
            }
            
            # Clear caches
            self.rewrite_cache.clear()
            self.bm25_cache.clear()
            self.context_pack_cache.clear()
            
            # Reset counters
            self.document_count = 0
            self.permanent_documents.clear()
            self.document_access_order.clear()
            
            if collection_errors:
                logger.warning(f"RAG cleared with {len(collection_errors)} non-critical warnings: {'; '.join(collection_errors)}")
            else:
                logger.info("Successfully cleared all RAG documents")
            
        except Exception as e:
            logger.error(f"CRITICAL error clearing RAG database: {e}")
            raise


# Global RAG manager instance
rag_manager = RAGManager()

