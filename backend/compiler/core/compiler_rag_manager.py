"""
Compiler RAG Manager - wrapper around aggregator RAG with configurable token budget.
Handles compiler-specific context routing and document management.
Default context window: 4096 tokens (user-configurable via settings).
"""
import logging
from typing import Optional, List
from pathlib import Path

from backend.shared.config import system_config, rag_config
from backend.shared.models import ContextPack
from backend.shared.rag_lock import rag_operation_lock
from backend.aggregator.core.rag_manager import rag_manager
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.memory.paper_memory import paper_memory

logger = logging.getLogger(__name__)


class CompilerRAGManager:
    """
    Compiler-specific RAG manager.
    - Wraps aggregator RAG manager
    - Configurable token budget (default 4096, vs 8192 for aggregator)
    - Manages compiler-specific documents (outline, paper, aggregator DB)
    - Handles context priority and direct injection decisions
    """
    
    def __init__(self):
        # Use the largest of the 3 context windows for RAG budget allocation (conservative approach)
        self.context_window = max(
            system_config.compiler_validator_context_window,
            system_config.compiler_high_context_context_window,
            system_config.compiler_high_param_context_window
        )
        # Use the largest output tokens for conservative budget calculation
        self.max_output_tokens = max(
            system_config.compiler_validator_max_output_tokens,
            system_config.compiler_high_context_max_output_tokens,
            system_config.compiler_high_param_max_output_tokens
        )
        self.available_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
        
        self._aggregator_db_loaded = False
        self._initialized = False
        self.aggregator_submissions_ragged = 0  # Track which aggregator submissions have been RAG'd
    
    def update_context_window(self, new_context_window: int, new_max_output_tokens: int = None) -> None:
        """
        Update the context window size.
        Called when compiler is started with a user-specified context size.
        
        Args:
            new_context_window: New context window size
            new_max_output_tokens: Optional new max output tokens (defaults to existing)
        """
        logger.info(f"Updating compiler context window from {self.context_window} to {new_context_window}")
        self.context_window = new_context_window
        if new_max_output_tokens is not None:
            self.max_output_tokens = new_max_output_tokens
        self.available_tokens = rag_config.get_available_input_tokens(self.context_window, self.max_output_tokens)
    
    def reset(self) -> None:
        """
        Reset compiler RAG manager state for a fresh session.
        Call this before initialize() to ensure clean state between compiler runs.
        
        CRITICAL: This resets tracking flags so that:
        - load_aggregator_database() will actually load (not skip due to _aggregator_db_loaded)
        - initialize() will run setup (not skip due to _initialized)
        """
        logger.info("Resetting compiler RAG manager state...")
        self._aggregator_db_loaded = False
        self._initialized = False
        self.aggregator_submissions_ragged = 0
        logger.info("Compiler RAG manager state reset")
    
    async def initialize(self) -> None:
        """Initialize compiler RAG manager."""
        if self._initialized:
            return
        
        logger.info("Initializing compiler RAG manager...")
        
        # Update context window from system config (in case it was changed)
        # Use the largest of the 3 context windows
        max_context_window = max(
            system_config.compiler_validator_context_window,
            system_config.compiler_high_context_context_window,
            system_config.compiler_high_param_context_window
        )
        self.update_context_window(max_context_window)
        
        # Set up re-chunking callbacks for outline and paper
        outline_memory.set_rechunk_callback(self._rechunk_outline)
        paper_memory.set_rechunk_callback(self._rechunk_paper)
        
        self._initialized = True
        logger.info("Compiler RAG manager initialized")
    
    async def load_aggregator_database(self) -> None:
        """
        Load aggregator shared training database into RAG.
        Pre-chunked at all 4 configs (256/512/768/1024) as a user file.
        CRITICAL: Holds RAG lock for entire operation to prevent aggregator interruption.
        """
        if self._aggregator_db_loaded:
            logger.info("Aggregator database already loaded (initial load)")
            return
        
        try:
            # Import here to avoid circular dependency
            from backend.aggregator.memory.shared_training import shared_training_memory
            
            # CRITICAL: Ensure shared_training_memory is initialized
            # This handles the case where compiler starts without aggregator running
            # but a saved aggregator database file exists from a previous session
            aggregator_file_path = Path(shared_training_memory.file_path)
            
            # Check if file exists and needs to be loaded
            if aggregator_file_path.exists() and aggregator_file_path.stat().st_size > 0:
                # File exists with content - ensure shared_training_memory has loaded it
                if not shared_training_memory.insights:
                    logger.info("Initializing shared_training_memory (aggregator database file exists but not loaded)")
                    await shared_training_memory.initialize()
            
            # Get current submission count
            submission_count = await shared_training_memory.get_insights_count()
            
            if submission_count == 0:
                logger.info("No aggregator submissions to load yet")
                self._aggregator_db_loaded = True
                return
            
            logger.info(f"Initial load: Processing aggregator database with {submission_count} submissions")
            
            # ACQUIRE GLOBAL RAG LOCK FOR ENTIRE INITIAL LOAD
            # This prevents the aggregator's re-chunking from interrupting between chunk sizes
            await rag_operation_lock.acquire("Compiler initial load (all 4 chunk sizes)")
            
            try:
                # Load the aggregator database file directly as a user file
                # This will chunk it at all 4 configs (256/512/768/1024)
                aggregator_file_path = str(shared_training_memory.file_path)
                
                if Path(aggregator_file_path).exists():
                    # Import ingestion pipeline to manually control chunking
                    from backend.aggregator.ingestion.pipeline import ingestion_pipeline
                    
                    # Ingest at all 4 chunk sizes
                    chunks_by_size = await ingestion_pipeline.ingest_file(
                        aggregator_file_path,
                        rag_config.submitter_chunk_intervals,  # All 4 configs
                        is_user_file=True
                    )
                    
                    # Add all chunks while holding the lock
                    for chunk_size, chunks in chunks_by_size.items():
                        await rag_manager._add_chunks(chunks, chunk_size)
                    
                    # Track document
                    rag_manager.document_count += 1
                    rag_manager.permanent_documents.add(Path(aggregator_file_path).name)
                    
                    logger.info(f"Aggregator database loaded as user file: {submission_count} submissions, 4 chunk configs")
                else:
                    logger.warning(f"Aggregator database file does not exist: {aggregator_file_path}")
                
                # Update tracking
                self.aggregator_submissions_ragged = submission_count
                self._aggregator_db_loaded = True
                
            finally:
                # ALWAYS RELEASE LOCK
                rag_operation_lock.release()
            
        except Exception as e:
            logger.error(f"Failed to load aggregator database: {e}")
            raise
    
    async def incremental_rerag_aggregator_database(self) -> None:
        """
        Incrementally re-RAG aggregator database with new submissions.
        Called every 10 aggregator acceptances by the monitoring task.
        Removes old chunks and re-adds the entire updated file with global lock.
        """
        try:
            # ACQUIRE GLOBAL RAG LOCK
            await rag_operation_lock.acquire("Compiler re-RAG aggregator DB")
            
            # Import here to avoid circular dependency
            from backend.aggregator.memory.shared_training import shared_training_memory
            
            current_count = await shared_training_memory.get_insights_count()
            
            if current_count <= self.aggregator_submissions_ragged:
                logger.info("Incremental re-RAG: No new aggregator submissions to process")
                return
            
            new_submissions_count = current_count - self.aggregator_submissions_ragged
            logger.info(f"Incremental re-RAG: Processing {new_submissions_count} new aggregator submissions ({current_count} total)")
            
            # Remove old aggregator database chunks from RAG
            aggregator_file_path = str(shared_training_memory.file_path)
            await rag_manager.remove_document("rag_shared_training.txt")
            
            # Re-add the entire updated file with all 4 chunk configs
            if Path(aggregator_file_path).exists():
                await rag_manager.add_document(
                    aggregator_file_path,
                    chunk_sizes=rag_config.submitter_chunk_intervals,  # All 4 configs
                    is_user_file=True  # Treated as permanent, never evicted
                )
                logger.info(f"Incremental re-RAG complete: {new_submissions_count} new submissions added ({current_count} total)")
            else:
                logger.error(f"Aggregator database file not found during re-RAG: {aggregator_file_path}")
            
            # Update tracking
            self.aggregator_submissions_ragged = current_count
            
        except Exception as e:
            logger.error(f"Failed to incrementally re-RAG aggregator database: {e}")
            raise
        finally:
            # ALWAYS RELEASE LOCK
            rag_operation_lock.release()
    
    async def retrieve_for_mode(
        self,
        query: str,
        mode: str,
        max_tokens: Optional[int] = None,
        exclude_sources: Optional[List[str]] = None
    ) -> ContextPack:
        """
        Retrieve context optimized for specific compiler mode.
        
        Args:
            query: Search query
            mode: Compiler mode (construction, outline, review, rigor)
            max_tokens: Override max tokens (defaults to available_tokens)
            exclude_sources: Source names to skip (already direct-injected in prompt)
        
        Returns:
            ContextPack with retrieved context
        """
        import time
        
        logger.info(f"Starting RAG retrieval for mode={mode}, query_length={len(query)}")
        if exclude_sources:
            logger.info(f"Excluding direct-injected sources: {exclude_sources}")
        start_time = time.time()
        
        try:
            max_tokens = max_tokens or self.available_tokens
            
            # Use 512 chunks (constant for compiler)
            chunk_size = rag_config.validator_chunk_size
            
            # Retrieve from RAG
            context_pack = await rag_manager.retrieve(
                query=query,
                chunk_size=chunk_size,
                max_tokens=max_tokens,
                exclude_sources=exclude_sources
            )
            
            elapsed = time.time() - start_time
            logger.info(f"RAG retrieval complete for mode={mode} in {elapsed:.2f}s "
                       f"(coverage={context_pack.coverage:.2f}, tokens={len(context_pack.text.split())})")
            
            return context_pack
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"RAG retrieval failed for mode={mode} after {elapsed:.2f}s: {e}")
            raise
    
    async def _rechunk_outline(self, outline_content: str) -> None:
        """
        Re-chunk outline when updated.
        
        Args:
            outline_content: New outline content (passed from update_outline to avoid deadlock)
        """
        try:
            if not outline_content.strip():
                logger.info("Outline is empty, skipping re-chunking")
                return
            
            # Remove old outline chunks
            await rag_manager.remove_document("compiler_outline.txt")
            
            # Add new outline chunks (512 chars, constant)
            await rag_manager.add_text(
                outline_content,
                "compiler_outline.txt",
                chunk_sizes=[rag_config.validator_chunk_size],
                is_permanent=False
            )
            
            logger.info("Outline re-chunked successfully")
            
        except Exception as e:
            logger.error(f"Failed to re-chunk outline: {e}")
    
    async def _rechunk_paper(self, paper_content: str) -> None:
        """
        Re-chunk paper when updated.
        
        Args:
            paper_content: New paper content (passed from update_paper to avoid deadlock)
        """
        try:
            if not paper_content.strip():
                logger.info("Paper is empty, skipping re-chunking")
                return
            
            # Remove old paper chunks
            await rag_manager.remove_document("compiler_paper.txt")
            
            # Add new paper chunks (512 chars, constant)
            await rag_manager.add_text(
                paper_content,
                "compiler_paper.txt",
                chunk_sizes=[rag_config.validator_chunk_size],
                is_permanent=False
            )
            
            logger.info("Paper re-chunked successfully")
            
        except Exception as e:
            logger.error(f"Failed to re-chunk paper: {e}")
    
    def get_available_tokens(self) -> int:
        """Get available tokens for content."""
        return self.available_tokens
    
    def get_context_window(self) -> int:
        """Get total context window."""
        return self.context_window


# Global compiler RAG manager instance
compiler_rag_manager = CompilerRAGManager()

