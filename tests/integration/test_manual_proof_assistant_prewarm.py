import asyncio
import unittest
from unittest import mock

from backend.api.routes import proofs as proofs_route
from backend.shared.config import system_config
from backend.shared.models import ProofCheckRequest, ProofRoleConfigSnapshot, ProofRuntimeConfigSnapshot


class ManualProofAssistantPrewarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_proof_check_owns_sleep_until_terminal_cleanup(self) -> None:
        lifecycle = []

        class FakeAssistantCoordinator:
            def submit_target(self, snapshot):
                return snapshot.stable_hash()

            async def stop_all(self, **_kwargs):
                lifecycle.append("assistant_stopped")

        class FakeStage:
            async def run_manual(self, **_kwargs):
                lifecycle.append("stage_started")
                raise RuntimeError("stage failed")

        runtime_snapshot = ProofRuntimeConfigSnapshot(
            brainstorm=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            paper=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            validator=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="validator-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )

        async def fake_resolve_manual_source(request, scoped_proof_database=None):
            return "SOURCE CONTENT", "Manual source title", "User prompt"

        async def fake_runtime_snapshot(request=None):
            return runtime_snapshot

        async def fake_broadcast(_event, _payload):
            return None

        owner = ("manual_proof_check", "lifecycle")
        acquire = mock.Mock(side_effect=lambda actual: lifecycle.append(("acquire", actual)))
        release = mock.Mock(side_effect=lambda actual: lifecycle.append(("release", actual)))
        release_source = mock.AsyncMock(
            side_effect=lambda *_args: lifecycle.append("source_released")
        )
        with (
            mock.patch.object(proofs_route, "assistant_proof_search_coordinator", FakeAssistantCoordinator()),
            mock.patch.object(proofs_route, "_resolve_manual_source", new=fake_resolve_manual_source),
            mock.patch.object(proofs_route, "_get_runtime_snapshot", new=fake_runtime_snapshot),
            mock.patch.object(proofs_route, "websocket") as websocket_mock,
            mock.patch.object(proofs_route.autonomous_coordinator, "_proof_verification_stage", FakeStage()),
            mock.patch.object(proofs_route.ProofVerificationStage, "release_source", new=release_source),
            mock.patch.object(proofs_route.sleep_inhibitor, "acquire", new=acquire),
            mock.patch.object(proofs_route.sleep_inhibitor, "release", new=release),
        ):
            websocket_mock.broadcast_event = fake_broadcast
            await proofs_route._run_manual_proof_check(
                ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator"),
                owner,
            )

        self.assertEqual(lifecycle[0], ("acquire", owner))
        self.assertIn("stage_started", lifecycle)
        self.assertIn("source_released", lifecycle)
        self.assertEqual(lifecycle[-1], ("release", owner))
        release_source.assert_awaited_once_with("brainstorm", "manual_aggregator")

    async def test_manual_proof_check_cancellation_while_queued_releases_owners(self) -> None:
        request = ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator")
        owner = ("manual_proof_check", "cancelled")
        acquire = mock.Mock()
        release = mock.Mock()
        release_source = mock.AsyncMock()
        await proofs_route._manual_proof_run_lock.acquire()
        task = None
        try:
            with (
                mock.patch.object(proofs_route.ProofVerificationStage, "release_source", new=release_source),
                mock.patch.object(proofs_route.sleep_inhibitor, "acquire", new=acquire),
                mock.patch.object(proofs_route.sleep_inhibitor, "release", new=release),
                mock.patch.object(proofs_route.assistant_proof_search_coordinator, "stop_all", new=mock.AsyncMock()),
            ):
                task = asyncio.create_task(
                    proofs_route._run_manual_proof_check(request, owner)
                )
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        finally:
            proofs_route._manual_proof_run_lock.release()
            if task is not None and not task.done():
                task.cancel()

        acquire.assert_called_once_with(owner)
        release.assert_called_once_with(owner)
        release_source.assert_awaited_once_with("brainstorm", "manual_aggregator")

    async def test_try_to_prove_refreshes_assistant_before_stage_prompt_preflight_error(self) -> None:
        old_memory_enabled = system_config.agent_conversation_memory_enabled
        system_config.agent_conversation_memory_enabled = True
        snapshots = []

        class FakeAssistantCoordinator:
            def submit_target(self, snapshot):
                snapshots.append(snapshot)
                return snapshot.stable_hash()

            async def stop_all(self, **kwargs):
                self.stop_kwargs = kwargs

        class FakeStage:
            async def run_manual(self, **kwargs):
                raise RuntimeError("Proof identification prompt exceeds the configured context window")

        runtime_snapshot = ProofRuntimeConfigSnapshot(
            brainstorm=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            paper=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            validator=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="validator-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            assistant=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="assistant-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )

        async def fake_resolve_manual_source(request, scoped_proof_database=None):
            return "SOURCE CONTENT " * 100, "Manual source title", "User prompt"

        async def fake_runtime_snapshot(request=None):
            return runtime_snapshot

        async def fake_broadcast(event, payload):
            return None

        try:
            with (
                mock.patch.object(proofs_route, "assistant_proof_search_coordinator", FakeAssistantCoordinator()),
                mock.patch.object(proofs_route, "_resolve_manual_source", new=fake_resolve_manual_source),
                mock.patch.object(proofs_route, "_get_runtime_snapshot", new=fake_runtime_snapshot),
                mock.patch.object(proofs_route, "websocket") as websocket_mock,
                mock.patch.object(proofs_route.autonomous_coordinator, "_proof_verification_stage", FakeStage()),
                mock.patch.object(proofs_route.ProofVerificationStage, "release_source", new=mock.AsyncMock()),
            ):
                websocket_mock.broadcast_event = fake_broadcast
                await proofs_route._run_manual_proof_check(
                    ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator"),
                    ("manual_proof_check", "preflight-error"),
                )
        finally:
            system_config.agent_conversation_memory_enabled = old_memory_enabled

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].workflow_mode, "manual_proof_check")
        self.assertEqual(snapshots[0].target_kind, "proof_candidate")
        self.assertEqual(snapshots[0].workflow_phase, "manual_try_to_prove")
        self.assertEqual(snapshots[0].user_prompt, "User prompt")
        self.assertEqual(snapshots[0].target_statement, "User prompt")
        self.assertEqual(snapshots[0].imports, ["Mathlib"])
        self.assertEqual(snapshots[0].source_type, "manual_brainstorm")
        self.assertEqual(snapshots[0].source_id, "manual_aggregator")

    async def test_try_to_prove_does_not_wait_for_assistant_refresh(self) -> None:
        old_memory_enabled = system_config.agent_conversation_memory_enabled
        system_config.agent_conversation_memory_enabled = True
        stage_started = False

        class FakeAssistantCoordinator:
            def submit_target(self, snapshot):
                return snapshot.stable_hash()

            async def stop_all(self, **kwargs):
                return None

        class FakeStage:
            async def run_manual(self, **kwargs):
                nonlocal stage_started
                stage_started = True

        runtime_snapshot = ProofRuntimeConfigSnapshot(
            brainstorm=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            paper=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="proof-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            validator=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="validator-model",
                context_window=4096,
                max_output_tokens=512,
            ),
            assistant=ProofRoleConfigSnapshot(
                provider="openrouter",
                model_id="assistant-model",
                context_window=4096,
                max_output_tokens=512,
            ),
        )

        async def fake_resolve_manual_source(request, scoped_proof_database=None):
            return "SOURCE CONTENT", "Manual source title", "User prompt"

        async def fake_runtime_snapshot(request=None):
            return runtime_snapshot

        async def fake_broadcast(event, payload):
            return None

        try:
            with (
                mock.patch.object(proofs_route, "assistant_proof_search_coordinator", FakeAssistantCoordinator()),
                mock.patch.object(proofs_route, "_resolve_manual_source", new=fake_resolve_manual_source),
                mock.patch.object(proofs_route, "_get_runtime_snapshot", new=fake_runtime_snapshot),
                mock.patch.object(proofs_route, "websocket") as websocket_mock,
                mock.patch.object(proofs_route.autonomous_coordinator, "_proof_verification_stage", FakeStage()),
                mock.patch.object(proofs_route.ProofVerificationStage, "release_source", new=mock.AsyncMock()),
            ):
                websocket_mock.broadcast_event = fake_broadcast
                await proofs_route._run_manual_proof_check(
                    ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator"),
                    ("manual_proof_check", "nonblocking-refresh"),
                )
        finally:
            system_config.agent_conversation_memory_enabled = old_memory_enabled

        self.assertTrue(stage_started)


if __name__ == "__main__":
    unittest.main()

