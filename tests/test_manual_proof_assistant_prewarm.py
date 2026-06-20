import unittest
from unittest import mock

from backend.api.routes import proofs as proofs_route
from backend.shared.config import system_config
from backend.shared.models import ProofCheckRequest, ProofRoleConfigSnapshot, ProofRuntimeConfigSnapshot


class ManualProofAssistantPrewarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_try_to_prove_refreshes_assistant_before_stage_prompt_preflight_error(self) -> None:
        old_memory_enabled = system_config.agent_conversation_memory_enabled
        system_config.agent_conversation_memory_enabled = True
        snapshots = []

        class FakeAssistantCoordinator:
            async def refresh_now(self, snapshot):
                snapshots.append(snapshot)
                return None

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
                    ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator")
                )
        finally:
            system_config.agent_conversation_memory_enabled = old_memory_enabled

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].workflow_mode, "manual_proof_check")
        self.assertEqual(snapshots[0].workflow_phase, "manual_try_to_prove")
        self.assertEqual(snapshots[0].source_type, "manual_brainstorm")
        self.assertEqual(snapshots[0].source_id, "manual_aggregator")


if __name__ == "__main__":
    unittest.main()

