from unittest import IsolatedAsyncioTestCase

from backend.aggregator.core.queue_manager import QueueManager
from backend.shared.models import Submission


def _submission(submission_id: str) -> Submission:
    return Submission(
        submission_id=submission_id,
        submitter_id=1,
        content=submission_id,
        reasoning="test",
        chunk_size_used=256,
    )


class QueueManagerTests(IsolatedAsyncioTestCase):
    async def test_prepend_batch_restores_original_fifo_order(self) -> None:
        manager = QueueManager()
        first = _submission("first")
        second = _submission("second")
        later = _submission("later")

        await manager.enqueue(first)
        await manager.enqueue(second)
        in_flight = await manager.dequeue_batch(max_count=2)
        await manager.enqueue(later)
        await manager.prepend_batch(in_flight)

        restored = await manager.dequeue_batch(max_count=3)
        self.assertEqual(
            [submission.submission_id for submission in restored],
            ["first", "second", "later"],
        )
