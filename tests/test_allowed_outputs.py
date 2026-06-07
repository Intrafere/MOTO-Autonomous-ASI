from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException

from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import proofs as proofs_route
from backend.shared.config import system_config
from backend.shared.models import (
    AutonomousResearchStartRequest,
    CompilerStartRequest,
    ProofCheckRequest,
    SubmitterConfig,
)


class AllowedOutputRouteTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._generic_mode = system_config.generic_mode
        self._lean_enabled = system_config.lean4_enabled

    def tearDown(self) -> None:
        system_config.generic_mode = self._generic_mode
        system_config.lean4_enabled = self._lean_enabled

    def _compiler_request(self, **overrides) -> CompilerStartRequest:
        data = {
            "compiler_prompt": "Write a paper",
            "allow_mathematical_proofs": True,
            "allow_research_papers": True,
            "validator_model": "validator-model",
            "validator_context_size": 1000,
            "validator_max_output_tokens": 100,
            "high_context_model": "high-context-model",
            "high_context_context_size": 1000,
            "high_context_max_output_tokens": 100,
            "high_param_model": "high-param-model",
            "high_param_context_size": 1000,
            "high_param_max_output_tokens": 100,
            "critique_submitter_model": "critique-model",
            "critique_submitter_context_window": 1000,
            "critique_submitter_max_tokens": 100,
        }
        data.update(overrides)
        return CompilerStartRequest(**data)

    def _autonomous_request(self, **overrides) -> AutonomousResearchStartRequest:
        data = {
            "user_research_prompt": "Research prompt",
            "submitter_configs": [
                SubmitterConfig(
                    submitter_id=1,
                    model_id="submitter-model",
                    context_window=1000,
                    max_output_tokens=100,
                )
            ],
            "allow_mathematical_proofs": True,
            "allow_research_papers": True,
            "validator_model": "validator-model",
            "validator_context_window": 1000,
            "validator_max_tokens": 100,
            "high_context_model": "high-context-model",
            "high_context_context_window": 1000,
            "high_context_max_tokens": 100,
            "high_param_model": "high-param-model",
            "high_param_context_window": 1000,
            "high_param_max_tokens": 100,
            "critique_submitter_model": "critique-model",
            "critique_submitter_context_window": 1000,
            "critique_submitter_max_tokens": 100,
        }
        data.update(overrides)
        return AutonomousResearchStartRequest(**data)

    async def test_compiler_rejects_proof_requested_when_lean_disabled_in_desktop_mode(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_enabled = False

        with self.assertRaises(HTTPException) as exc:
            await compiler_route.start_compiler(self._compiler_request())

        self.assertEqual(exc.exception.status_code, 501)
        self.assertIn("Lean 4", str(exc.exception.detail))

    async def test_compiler_generic_paper_run_downgrades_proof_output(self) -> None:
        system_config.generic_mode = True
        system_config.lean4_enabled = False
        initialize = mock.AsyncMock()

        with mock.patch.object(compiler_route, "_get_start_conflict", return_value=None):
            with mock.patch.object(compiler_route.compiler_coordinator, "initialize", initialize):
                with mock.patch.object(compiler_route.compiler_coordinator, "start", mock.AsyncMock()):
                    response = await compiler_route.start_compiler(self._compiler_request())

        self.assertEqual(response["status"], "started")
        self.assertFalse(initialize.await_args.kwargs["allow_mathematical_proofs"])

    async def test_compiler_model_diagnostics_unavailable_in_generic_mode(self) -> None:
        system_config.generic_mode = True

        with self.assertRaises(HTTPException) as exc:
            await compiler_route.test_models(self._compiler_request())

        self.assertEqual(exc.exception.status_code, 501)
        self.assertTrue(exc.exception.detail["generic_mode"])

    async def test_autonomous_rejects_proof_requested_when_lean_disabled_in_desktop_mode(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_enabled = False

        with self.assertRaises(HTTPException) as exc:
            await autonomous_route.start_autonomous_research(self._autonomous_request())

        self.assertEqual(exc.exception.status_code, 501)
        self.assertIn("Lean 4", str(exc.exception.detail))

    async def test_autonomous_generic_paper_run_downgrades_proof_output(self) -> None:
        system_config.generic_mode = True
        system_config.lean4_enabled = False
        initialize = mock.AsyncMock()

        with mock.patch.object(autonomous_route, "_get_start_conflict", return_value=None):
            with mock.patch.object(autonomous_route.autonomous_coordinator, "initialize", initialize):
                with mock.patch.object(
                    autonomous_route.autonomous_coordinator,
                    "start_in_background",
                    return_value=True,
                ):
                    response = await autonomous_route.start_autonomous_research(self._autonomous_request())

        self.assertTrue(response["success"])
        self.assertFalse(initialize.await_args.kwargs["allow_mathematical_proofs"])


class ProofStatusReadinessTests(IsolatedAsyncioTestCase):
    async def test_status_marks_manual_check_unready_when_workspace_is_not_ready(self) -> None:
        previous_lean_enabled = proofs_route.system_config.lean4_enabled
        previous_smt_enabled = proofs_route.system_config.smt_enabled
        proofs_route.system_config.lean4_enabled = True
        proofs_route.system_config.smt_enabled = False

        class FakeLeanClient:
            async def get_version(self):
                return "Lean (version 4.0.0)"

            async def ensure_workspace(self):
                return False

            def get_mathlib_commit(self):
                return ""

            def is_server_active(self):
                return False

        try:
            with mock.patch.object(
                proofs_route,
                "_get_manual_check_status",
                mock.AsyncMock(return_value=(True, "")),
            ):
                with mock.patch.object(proofs_route, "get_lean4_client", return_value=FakeLeanClient()):
                    payload = await proofs_route.get_proofs_status()
        finally:
            proofs_route.system_config.lean4_enabled = previous_lean_enabled
            proofs_route.system_config.smt_enabled = previous_smt_enabled

        self.assertFalse(payload["manual_check_ready"])
        self.assertEqual(payload["manual_check_message"], "Lean 4 is still starting up.")

    async def test_manual_aggregator_snapshot_does_not_fall_back_to_autonomous_config(self) -> None:
        autonomous_snapshot = {
            "brainstorm": {
                "provider": "openrouter",
                "model_id": "google/gemini-flash-latest",
                "context_window": 1000,
                "max_output_tokens": 100,
            },
            "paper": {
                "provider": "openrouter",
                "model_id": "google/gemini-flash-latest",
                "context_window": 1000,
                "max_output_tokens": 100,
            },
            "validator": {
                "provider": "openrouter",
                "model_id": "google/gemini-flash-latest",
                "context_window": 1000,
                "max_output_tokens": 100,
            },
        }
        request = ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator")

        with mock.patch.object(proofs_route.coordinator, "submitter_configs", []):
            with mock.patch.object(proofs_route.coordinator, "validator_model", ""):
                with mock.patch.object(
                    proofs_route.autonomous_coordinator,
                    "get_proof_runtime_config",
                    return_value=autonomous_snapshot,
                ):
                    snapshot = await proofs_route._get_runtime_snapshot(request)

        self.assertIsNone(snapshot)

    async def test_manual_aggregator_snapshot_uses_active_manual_validator(self) -> None:
        request = ProofCheckRequest(source_type="brainstorm", source_id="manual_aggregator")
        submitter = SubmitterConfig(
            submitter_id=1,
            provider="openrouter",
            model_id="anthropic/claude-opus-4.7",
            context_window=1000000,
            max_output_tokens=128000,
        )

        patches = [
            mock.patch.object(proofs_route.coordinator, "submitter_configs", [submitter]),
            mock.patch.object(proofs_route.coordinator, "validator_model", "anthropic/claude-opus-4.7"),
            mock.patch.object(proofs_route.coordinator, "validator_provider", "openrouter"),
            mock.patch.object(proofs_route.coordinator, "validator_openrouter_provider", "Anthropic"),
            mock.patch.object(proofs_route.coordinator, "validator_openrouter_reasoning_effort", "xhigh"),
            mock.patch.object(proofs_route.coordinator, "validator_lm_studio_fallback", None),
            mock.patch.object(proofs_route.coordinator, "validator_context_window", 1000000),
            mock.patch.object(proofs_route.coordinator, "validator_max_tokens", 128000),
            mock.patch.object(proofs_route.coordinator, "validator_supercharge_enabled", False),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            snapshot = await proofs_route._get_runtime_snapshot(request)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.brainstorm.model_id, "anthropic/claude-opus-4.7")
        self.assertEqual(snapshot.validator.model_id, "anthropic/claude-opus-4.7")
        self.assertEqual(snapshot.validator.openrouter_provider, "Anthropic")
