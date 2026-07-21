"""
Global RAG operation lock to prevent collisions between Aggregator and Compiler.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RAGOperationLock:
    """
    Async reentrant lock for RAG operations that modify ChromaDB or call embedding API.
    Ensures only one mode (Aggregator or Compiler) performs heavy RAG operations at a time.
    Supports nested acquisition by the same task (reentrant).
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_holder = None
        self._current_task = None
        self._acquisition_count = 0
    
    async def acquire(self, operation_name: str):
        """
        Acquire lock for RAG operation.
        Supports reentrant acquisition - same task can acquire multiple times.
        """
        current_task = asyncio.current_task()
        
        # Check if current task already holds the lock (reentrant)
        if self._current_task == current_task and self._acquisition_count > 0:
            self._acquisition_count += 1
            logger.debug(f"RAG lock reentrant acquisition by: {operation_name} (count={self._acquisition_count})")
            return
        
        # Otherwise, acquire lock normally
        logger.debug(f"RAG lock requested by: {operation_name}")
        await self._lock.acquire()
        self._current_holder = operation_name
        self._current_task = current_task
        self._acquisition_count = 1
        logger.debug(f"RAG lock acquired by: {operation_name}")
    
    def release(self):
        """
        Release lock.
        Only fully releases when acquisition count reaches 0 (handles reentrant acquisitions).
        """
        current_task = asyncio.current_task()
        if self._acquisition_count <= 0:
            logger.warning("Attempted to release RAG lock when not held")
            return False
        if self._current_task is not current_task:
            logger.error(
                "Refusing RAG lock release by non-owner task (holder=%s)",
                self._current_holder,
            )
            return False
        
        self._acquisition_count -= 1
        
        if self._acquisition_count == 0:
            logger.debug(f"RAG lock released by: {self._current_holder}")
            self._current_holder = None
            self._current_task = None
            self._lock.release()
        else:
            logger.debug(f"RAG lock reentrant release (count={self._acquisition_count})")
        return True

    def owned_by_current_task(self) -> bool:
        return (
            self._acquisition_count > 0
            and self._current_task is asyncio.current_task()
        )

    @asynccontextmanager
    async def operation(self, operation_name: str):
        """Acquire and owner-safely release one reentrant RAG operation."""
        await self.acquire(operation_name)
        try:
            yield self
        finally:
            self.release()
    
    async def __aenter__(self):
        await self.acquire("context_manager")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Global instance
rag_operation_lock = RAGOperationLock()

