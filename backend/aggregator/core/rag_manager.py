"""
RAG Manager - 4-stage retrieval pipeline orchestrator.
Stages: Query Rewriting -> Hybrid Recall -> Reranking+MMR -> Packing+Compression
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from collections import OrderedDict
import asyncio
import logging
import hashlib
import time
import gc
import platform
from pathlib import Path

from backend.shared.config import rag_config, system_config
from backend.shared.models import DocumentChunk, ContextPack
from backend.shared.api_client_manager import api_client_manager
from backend.shared.rag_lock import rag_operation_lock
from backend.shared.utils import count_tokens
from backend.shared.log_redaction import redact_log_text
from backend.aggregator.ingestion.pipeline import ingestion_pipeline
from backend.aggregator.core.chroma_cache import (
    abort_chroma_cache_rebuild,
    complete_chroma_cache_rebuild,
    maintain_chroma_cache_directory,
    quarantine_chroma_cache,
)

logger = logging.getLogger(__name__)
CHROMA_DELETE_BATCH_SIZE = 1000


class RAGManager:
    """
    RAG Manager with 4-stage retrieval pipeline.
    """
    
    def __init__(self):
        # Open Chroma lazily so imports are filesystem side-effect free and the
        # runtime root can be bound before first use.
        self.chroma_client = None
        self.collections = {}
        self._root_identity = None
        self._prepared_root_identities = set()
        # Rust collection/client wrappers can retain Windows directory handles
        # until their finalizers run even after the shared system is stopped.
        gc.collect()
        
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

    @property
    def is_initialized(self) -> bool:
        return self.chroma_client is not None

    def _prepare_process_generation_cache_locked(self) -> Path | None:
        """Replace prior-process Windows Chroma state before Rust can open it."""
        identity = system_config.runtime_root_identity()
        if identity in self._prepared_root_identities:
            return None
        if platform.system() != "Windows" or system_config.generic_mode:
            self._prepared_root_identities.add(identity)
            return None

        cache_dir = Path(system_config.chroma_db_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._prepared_root_identities.add(identity)
            return None

        _, quarantine = quarantine_chroma_cache(
            system_config.chroma_db_dir,
            system_config.data_dir,
        )
        if quarantine is not None:
            logger.warning(
                "Replaced prior-process Windows Chroma cache before native initialization; "
                "RAG sources will be rebuilt lazily from durable files."
            )
        return quarantine

    def _ensure_initialized_locked(self) -> None:
        """Worker-only open; callers must hold the async lifecycle boundary."""
        identity = system_config.runtime_root_identity()
        if self.chroma_client is not None and self._root_identity == identity:
            return
        if self.chroma_client is not None:
            self._close_locked()
            self._reset_memory_state()
        quarantine = self._prepare_process_generation_cache_locked()
        if quarantine is None:
            self._maintain_persistent_cache()
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=system_config.chroma_db_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collections = {
                size: self.chroma_client.get_or_create_collection(
                    name=f"chunks_{size}",
                    metadata={"chunk_size": size},
                )
                for size in rag_config.submitter_chunk_intervals
            }
            self._root_identity = identity
        except BaseException:
            self._close_locked()
            try:
                abort_chroma_cache_rebuild(
                    system_config.chroma_db_dir,
                    system_config.data_dir,
                    quarantine,
                )
            finally:
                self._prepared_root_identities.discard(identity)
            raise
        else:
            self._prepared_root_identities.add(identity)
            if quarantine is not None:
                complete_chroma_cache_rebuild(
                    system_config.chroma_db_dir,
                    system_config.data_dir,
                    quarantine,
                )

    async def ensure_initialized(self) -> None:
        async with rag_operation_lock.operation("Chroma initialize"):
            await self._await_native_worker(
                "Chroma initialize",
                self._ensure_initialized_locked,
            )

    async def prepare_process_generation_cache(self) -> None:
        """Prepare the active cache before any workflow can enter Chroma."""
        async with rag_operation_lock.operation("Chroma process-generation prepare"):
            await self._await_native_worker(
                "Chroma process-generation prepare",
                self._ensure_initialized_locked,
            )

    async def _await_native_worker(self, operation_name: str, func, *args, **kwargs):
        """Run one native call and drain it before propagating cancellation."""
        worker = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            uncancel = getattr(current, "uncancel", None)
            pending_cancellations = 0
            if callable(uncancel):
                while current is not None and current.cancelling():
                    uncancel()
                    pending_cancellations += 1
            try:
                try:
                    await asyncio.shield(worker)
                except Exception:
                    logger.debug(
                        "Chroma worker failed while completing cancelled operation %s",
                        operation_name,
                        exc_info=True,
                    )
            finally:
                for _ in range(pending_cancellations):
                    if current is not None:
                        current.cancel()
            raise

    async def _run_chroma_call(
        self,
        operation_name: str,
        chunk_size: int,
        method_name: str,
        *args,
        **kwargs,
    ):
        """Run one Chroma call serially and never abandon its native worker.

        Cancelling ``asyncio.to_thread`` only cancels the awaiter; the native
        Chroma/Rust call keeps running. Waiting for that worker before releasing
        the global lock prevents a subsequent reset/query from entering Chroma
        concurrently with the abandoned call.
        """
        async with rag_operation_lock.operation(operation_name):
            await self._await_native_worker(
                f"{operation_name} initialization",
                self._ensure_initialized_locked,
            )
            collection = self.collections[chunk_size]
            return await self._await_native_worker(
                operation_name,
                getattr(collection, method_name),
                *args,
                **kwargs,
            )

    def _close_locked(self) -> None:
        """Release the current Chroma client without touching durable data."""
        client = self.chroma_client
        if client is not None:
            close_client = getattr(client, "close", None)
            stop = getattr(getattr(client, "_system", None), "stop", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception as exc:
                    logger.debug("Chroma client close reported: %s", exc)
            elif callable(stop):  # Compatibility with older supported Chroma.
                try:
                    stop()
                except Exception as exc:
                    logger.debug("Chroma client stop reported: %s", exc)
            clear_system_cache = getattr(client, "clear_system_cache", None)
            if callable(clear_system_cache):
                try:
                    clear_system_cache()
                except Exception as exc:
                    logger.debug("Chroma shared-system cache cleanup reported: %s", exc)
        self.chroma_client = None
        self.collections = {}
        self._root_identity = None

    def _reset_memory_state(self) -> None:
        self.chunks_by_size = {size: [] for size in rag_config.submitter_chunk_intervals}
        self.bm25_index = {size: None for size in rag_config.submitter_chunk_intervals}
        self.rewrite_cache.clear()
        self.bm25_cache.clear()
        self.context_pack_cache.clear()
        self.document_count = 0
        self.permanent_documents.clear()
        self.document_access_order.clear()

    async def close(self) -> None:
        """Drain native work, then close the Chroma client exactly once."""
        async with rag_operation_lock.operation("Chroma close"):
            await self._await_native_worker("Chroma close", self._close_locked)

    async def reset(self) -> None:
        """Close Chroma and clear all process-local retrieval state."""
        async with rag_operation_lock.operation("Chroma reset"):
            await self._await_native_worker("Chroma reset", self._close_locked)
            self._reset_memory_state()

    def _maintain_persistent_cache(self) -> None:
        """Clean orphaned Chroma cache artifacts before opening the client."""
        try:
            result = maintain_chroma_cache_directory(
                system_config.chroma_db_dir,
                system_config.data_dir,
            )
        except Exception as exc:
            logger.warning("Chroma cache maintenance skipped: %s", exc)
            return

        if result.reset_performed:
            logger.warning(
                "Chroma cache was reset to remove %d orphaned UUID directories; "
                "RAG sources will be re-indexed from durable files as workflows start.",
                result.unreferenced_uuid_dir_count,
            )
        else:
            logger.debug(
                "Chroma cache maintenance completed without reset: %s (%d UUID dirs, %d unreferenced).",
                result.reason,
                result.uuid_dir_count,
                result.unreferenced_uuid_dir_count,
            )
    
    async def add_document(
        self,
        file_path: str,
        chunk_sizes: List[int] = None,
        is_user_file: bool = False,
        trusted_roots: List[str | Path] | None = None,
    ) -> None:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to document
            chunk_sizes: Sizes to chunk at (None = all configs for user files)
            is_user_file: Whether this is a user file (never evicted)
        """
        try:
            if trusted_roots is None:
                trusted_roots = [
                    system_config.data_dir,
                    system_config.user_uploads_dir,
                ]

            # Ingest document
            chunks_by_size = await ingestion_pipeline.ingest_file(
                file_path,
                chunk_sizes,
                is_user_file,
                trusted_roots=trusted_roots,
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
            
            logger.info("Added document: %s", redact_log_text(Path(file_path).name, 120))
            
        except Exception as e:
            logger.error(
                "Failed to add document %s: %s",
                redact_log_text(Path(file_path).name, 120),
                redact_log_text(e, 240),
            )
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
            
            logger.info("Added text source with %d characters", len(text or ""))
            
        except Exception as e:
            logger.error(
                "Failed to add text source: %s",
                redact_log_text(e, 240),
            )
            raise
    
    async def retrieve(
        self,
        query: str,
        chunk_size: int = 512,
        max_tokens: int = None,
        exclude_sources: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        include_source_prefixes: Optional[List[str]] = None
    ) -> ContextPack:
        """
        4-stage retrieval pipeline.
        
        Args:
            query: Search query
            chunk_size: Chunk size to retrieve from
            max_tokens: Maximum tokens in result
            exclude_sources: Source names to skip during packing (already direct-injected)
            include_sources: Optional source allowlist for scoped retrieval
            include_source_prefixes: Optional source-name prefixes for scoped retrieval
        
        Returns:
            ContextPack with retrieved context
        """
        if max_tokens is None:
            max_tokens = rag_config.get_available_input_tokens(
                rag_config.submitter_context_window,
                rag_config.submitter_max_output_tokens,
            )
        elif int(max_tokens or 0) <= 0:
            raise ValueError("RAG retrieval max_tokens must be a positive integer.")
        
        # Stage A: Query Rewriting
        logger.debug(f"RAG Stage 1/4: Query rewriting for '{query[:50]}...'")
        queries = await self._rewrite_query(query)
        logger.debug(f"RAG Stage 1/4 complete: Generated {len(queries)} query variants")
        
        # Stage B: Hybrid Recall (BM25 + Vector)
        logger.debug(f"RAG Stage 2/4: Hybrid recall (BM25 + Vector) with chunk_size={chunk_size}")
        if include_sources or include_source_prefixes:
            logger.info(
                "RAG Stage 2/4: Restricting retrieval scope to sources=%s prefixes=%s",
                include_sources or [],
                include_source_prefixes or [],
            )
        candidates = await self._hybrid_recall(
            queries,
            chunk_size,
            exclude_sources=exclude_sources,
            include_sources=include_sources,
            include_source_prefixes=include_source_prefixes,
        )
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
        await self.ensure_initialized()
        embeddings = await api_client_manager.get_embeddings(texts)

        # Update chunks with embeddings and tokens
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.tokens = chunk.text.lower().split()

        # Commit the native write and its process-local mirror under one owner.
        operation_name = f"Chroma upsert (size={chunk_size})"
        async with rag_operation_lock.operation(operation_name):
            await self._await_native_worker(
                f"{operation_name} initialization",
                self._ensure_initialized_locked,
            )
            collection = self.collections[chunk_size]
            try:
                await self._await_native_worker(
                    operation_name,
                    collection.upsert,
                    ids=[chunk.chunk_id for chunk in chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[chunk.metadata for chunk in chunks],
                )
                logger.debug(f"Upserted {len(chunks)} chunks in ChromaDB collection (size={chunk_size})")
            except Exception as e:
                logger.error(f"CRITICAL: ChromaDB upsert failed for chunk_size={chunk_size}: {type(e).__name__}: {e}")
                logger.error(f"Attempting to upsert {len(chunks)} chunks with IDs: {[c.chunk_id for c in chunks][:5]}...")
                raise

            incoming_ids = {chunk.chunk_id for chunk in chunks}
            self.chunks_by_size[chunk_size] = [
                existing
                for existing in self.chunks_by_size[chunk_size]
                if existing.chunk_id not in incoming_ids
            ] + chunks
            self.bm25_index[chunk_size] = None
    
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
        chunk_size: int,
        exclude_sources: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        include_source_prefixes: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Stage B: Hybrid BM25 + Vector search."""
        await self.ensure_initialized()
        # Work from a stable snapshot so threaded scoring does not race with
        # concurrent RAG add/remove operations mutating the live chunk lists.
        chunks = list(self._filter_chunks_by_source_scope(
            self.chunks_by_size[chunk_size],
            exclude_sources=exclude_sources,
            include_sources=include_sources,
            include_source_prefixes=include_source_prefixes,
        ))
        if not chunks:
            return []
        
        # Vector search
        vector_results = await self._vector_search(queries, chunk_size, candidate_chunks=chunks)
        
        # BM25 search
        bm25_results = await asyncio.to_thread(
            self._bm25_search,
            queries,
            chunk_size,
            chunks,
        )
        
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
        chunk_size: int,
        candidate_chunks: Optional[List[DocumentChunk]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Vector similarity search with retry logic for HNSW index race conditions."""
        chunks = candidate_chunks if candidate_chunks is not None else self.chunks_by_size[chunk_size]
        
        if not chunks:
            return []
        
        query_embeddings = await api_client_manager.get_embeddings(queries)
        if candidate_chunks is not None and len(candidate_chunks) != len(self.chunks_by_size[chunk_size]):
            return await asyncio.to_thread(
                self._score_vector_candidates,
                query_embeddings,
                chunks,
            )

        all_results = []
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for query_embedding in query_embeddings:
            # Search with retry logic for transient HNSW errors during concurrent writes
            max_retries = 3
            retry_delay = 0.5  # Start with 500ms delay
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    results = await self._run_chroma_call(
                        f"Chroma query (size={chunk_size})",
                        chunk_size,
                        "query",
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
                chunk = chunk_by_id.get(chunk_id)
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

    def _score_vector_candidates(
        self,
        query_embeddings: List[List[float]],
        chunks: List[DocumentChunk],
    ) -> List[Tuple[DocumentChunk, float]]:
        """Score a scoped chunk snapshot in memory without blocking the event loop."""
        scored: List[Tuple[DocumentChunk, float]] = []
        for query_embedding in query_embeddings:
            for chunk in chunks:
                if not chunk.embedding:
                    continue
                scored.append((chunk, self._cosine_similarity(query_embedding, chunk.embedding)))

        seen = set()
        unique_results = []
        for chunk, score in sorted(scored, key=lambda x: x[1], reverse=True):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique_results.append((chunk, score))
        return unique_results[:rag_config.hybrid_recall_top_k]
    
    def _bm25_search(
        self,
        queries: List[str],
        chunk_size: int,
        candidate_chunks: Optional[List[DocumentChunk]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """BM25 lexical search."""
        chunks = list(candidate_chunks) if candidate_chunks is not None else list(self.chunks_by_size[chunk_size])
        if not chunks:
            return []
        
        # Build a local index for the snapshot. This runs in a worker thread
        # and intentionally does not mutate self.bm25_index across threads.
        bm25 = BM25Okapi([chunk.tokens for chunk in chunks])
        
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

    @staticmethod
    def _filter_chunks_by_source_scope(
        chunks: List[DocumentChunk],
        *,
        exclude_sources: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        include_source_prefixes: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """Limit chunks to an explicit source allowlist/prefixes and exclusions."""
        exclude_set = {source for source in (exclude_sources or []) if source}
        include_set = {source for source in (include_sources or []) if source}
        prefixes = tuple(prefix for prefix in (include_source_prefixes or []) if prefix)
        if not exclude_set and not include_set and not prefixes:
            return chunks

        scoped = []
        for chunk in chunks:
            if chunk.source_file in exclude_set:
                continue
            if not include_set and not prefixes:
                scoped.append(chunk)
            elif chunk.source_file in include_set or (prefixes and chunk.source_file.startswith(prefixes)):
                scoped.append(chunk)
        return scoped
    
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
                    removed += 1
                else:
                    keep.append(chunk)

            if evict_ids:
                try:
                    await self._run_chroma_call(
                        f"Chroma chunk-cap delete (size={chunk_size})",
                        chunk_size,
                        "delete",
                        ids=evict_ids,
                    )
                except Exception as e:
                    logger.error(f"ChromaDB delete during chunk cap enforcement (size={chunk_size}): {e}")
                    raise

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
        logger.info(
            "LRU eviction: Removing oldest document '%s' (last accessed: %s)",
            redact_log_text(oldest_doc, 120),
            oldest_time,
        )
        
        try:
            await self.remove_document(oldest_doc)
            # Remove from access tracking
            if oldest_doc in self.document_access_order:
                del self.document_access_order[oldest_doc]
            logger.info(
                "LRU eviction complete: '%s' removed successfully",
                redact_log_text(oldest_doc, 120),
            )
        except Exception as e:
            logger.error(
                "LRU eviction failed for '%s': %s",
                redact_log_text(oldest_doc, 120),
                redact_log_text(e, 240),
            )
    
    async def remove_document(self, source_name: str) -> None:
        """Remove a source from every collection before changing memory."""
        async with rag_operation_lock.operation(f"Chroma source removal: {source_name}"):
            await self._await_native_worker(
                f"Chroma source removal initialization: {source_name}",
                self._ensure_initialized_locked,
            )
            was_tracked = source_name in self.document_access_order
            failures = []
            for chunk_size in rag_config.submitter_chunk_intervals:
                try:
                    await self._delete_matching_ids(
                        chunk_size,
                        f"Chroma document delete (size={chunk_size})",
                        where={"source_file": source_name},
                    )
                except Exception as exc:
                    failures.append(f"chunks_{chunk_size}: {exc}")
            if failures:
                raise RuntimeError(
                    f"Failed to remove RAG source {source_name!r}: {'; '.join(failures)}"
                )

            for chunk_size in rag_config.submitter_chunk_intervals:
                self.chunks_by_size[chunk_size] = [
                    c for c in self.chunks_by_size[chunk_size]
                    if c.source_file != source_name
                ]
                self.bm25_index[chunk_size] = None

            if was_tracked:
                self.document_count = max(0, self.document_count - 1)
            self.document_access_order.pop(source_name, None)
            self.permanent_documents.discard(source_name)
        
        logger.info("Removed document: %s", redact_log_text(source_name, 120))

    async def _delete_matching_ids(self, chunk_size: int, operation_name: str, *, where=None) -> int:
        async with rag_operation_lock.operation(operation_name):
            await self._await_native_worker(
                f"{operation_name} initialization",
                self._ensure_initialized_locked,
            )
            collection = self.collections[chunk_size]
            deleted = 0
            previous_batch = None
            while True:
                results = await self._await_native_worker(
                    f"{operation_name} lookup",
                    collection.get,
                    limit=CHROMA_DELETE_BATCH_SIZE,
                    include=[],
                    **({"where": where} if where else {}),
                )
                ids = tuple(results.get("ids") or ())
                if not ids:
                    return deleted
                if ids == previous_batch:
                    raise RuntimeError(f"{operation_name} made no progress")
                await self._await_native_worker(
                    operation_name,
                    collection.delete,
                    ids=list(ids),
                )
                deleted += len(ids)
                previous_batch = ids

    async def clear_all_documents_async(self) -> None:
        """Atomically replace the rebuildable Chroma cache with an empty cache."""
        async with rag_operation_lock.operation("Chroma cache rebuild"):
            await asyncio.to_thread(self._close_locked)
            self._reset_memory_state()
            cache_dir, quarantine = await asyncio.to_thread(
                quarantine_chroma_cache,
                system_config.chroma_db_dir,
                system_config.data_dir,
            )
            try:
                await asyncio.to_thread(self._ensure_initialized_locked)
            except BaseException:
                # Leave the marker/quarantine for deterministic startup recovery.
                raise
            await asyncio.to_thread(
                complete_chroma_cache_rebuild,
                cache_dir,
                system_config.data_dir,
                quarantine,
            )


# Global RAG manager instance
rag_manager = RAGManager()

