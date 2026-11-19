"""A failed search-index reset at Start must not leave an empty session behind."""
import importlib
from unittest import IsolatedAsyncioTestCase, mock

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.shared.models import SubmitterConfig

coordinator_module = importlib.import_module("backend.autonomous.core.autonomous_coordinator")


class AutonomousStartCacheResetTests(IsolatedAsyncioTestCase):
    async def test_failed_cache_reset_creates_no_session_folder(self) -> None:
        coordinator = AutonomousCoordinator()
        submitter = SubmitterConfig(
            submitter_id=1,
            model_id="submitter-model",
            context_window=1000,
            max_output_tokens=100,
        )

        with (
            mock.patch.object(
                coordinator_module.session_manager,
                "find_interrupted_session",
                new=mock.AsyncMock(return_value=None),
            ),
            mock.patch.object(
                coordinator_module.session_manager,
                "initialize",
                new=mock.AsyncMock(),
            ) as create_session,
            mock.patch.object(
                coordinator_module.autonomous_rag_manager,
                "reset",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                coordinator_module.rag_manager,
                "clear_all_documents_async",
                new=mock.AsyncMock(side_effect=PermissionError(5, "Access is denied")),
            ),
        ):
            with self.assertRaises(PermissionError):
                await coordinator.initialize(
                    user_research_prompt="Research prompt",
                    submitter_configs=[submitter],
                    validator_model="validator-model",
                    validator_context_window=1000,
                    validator_max_tokens=100,
                    writer_context_window=1000,
                    writer_max_tokens=100,
                    high_param_context_window=1000,
                    high_param_max_tokens=100,
                    assistant_context_window=1000,
                    assistant_max_tokens=100,
                )

        create_session.assert_not_awaited()
