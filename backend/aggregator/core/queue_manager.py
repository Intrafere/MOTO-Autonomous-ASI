"""
Submission queue manager.
The coordinator handles overflow by pausing submitters; queued submissions stay FIFO.
"""
import asyncio
from typing import List, Optional
from collections import deque
import logging

from backend.shared.config import system_config
from backend.shared.models import Submission

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Thread-safe FIFO submission queue.
    """
    
    def __init__(self):
        self.queue: deque[Submission] = deque()
        self._lock = asyncio.Lock()
        self.overflow_threshold = system_config.queue_overflow_threshold
    
    async def enqueue(self, submission: Submission) -> None:
        """Add submission to queue."""
        async with self._lock:
            self.queue.append(submission)
            logger.debug(f"Enqueued submission {submission.submission_id}. Queue size: {len(self.queue)}")
    
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self.queue)

    async def count_for_submitter(self, submitter_id: int) -> int:
        """Count how many pending submissions were produced by a specific submitter."""
        async with self._lock:
            return sum(1 for s in self.queue if s.submitter_id == submitter_id)
    
    async def clear(self) -> None:
        """Clear the queue."""
        async with self._lock:
            self.queue.clear()
            logger.info("Queue cleared")
    
    async def peek(self) -> Optional[Submission]:
        """Peek at next submission without removing."""
        async with self._lock:
            return self.queue[0] if self.queue else None
    
    async def dequeue_batch(self, max_count: int = 3) -> List[Submission]:
        """
        Dequeue up to max_count submissions at once (batch validation).
        
        This method does NOT wait for submissions - returns whatever is available.
        
        Queue overflow monitoring: If queue >= overflow_threshold, logs a warning.
        Submitters will be paused by the coordinator when this threshold is reached.
        
        Args:
            max_count: Maximum number of submissions to dequeue (default 3)
            
        Returns:
            List of 0 to max_count submissions (empty list if queue is empty)
        """
        async with self._lock:
            if not self.queue:
                return []
            
            # Check for overflow (for monitoring/logging only - coordinator handles pausing)
            if len(self.queue) >= self.overflow_threshold:
                logger.debug(
                    f"Queue size ({len(self.queue)}) >= overflow threshold ({self.overflow_threshold}). "
                    f"Submitters should be paused by coordinator."
                )
            
            # Normal batch dequeue - take up to max_count from the front (FIFO)
            take_count = min(max_count, len(self.queue))
            submissions = []
            
            for _ in range(take_count):
                submissions.append(self.queue.popleft())
            
            logger.debug(f"Batch dequeued {take_count} submissions. Queue size: {len(self.queue)}")
            
            return submissions


# Global queue manager instance
queue_manager = QueueManager()

