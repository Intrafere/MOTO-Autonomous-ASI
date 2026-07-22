import importlib
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException

from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import proofs as proofs_route
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.shared.config import system_config
from backend.shared.models import (
    AutonomousResearchStartRequest,
    CompilerStartRequest,
    ProofCheckRequest,
    SubmitterConfig,
)

autonomous_coordinator_module = importlib.import_module(
    "backend.autonomous.core.autonomous_coordinator"
)
api_client_manager_module = importlib.import_module("backend.shared.api_client_manager")


class AutonomousProofFramingAllowedOutputTests(IsolatedAsyncioTestCase):
    async def test_proof_framing_gate_is_not_called_when_proof_output_is_disabled(self) -> None:
        coordinator = AutonomousCoordinator()
        coordinator._allow_mathematical_proofs = False
        coordinator._proof_framing_active = True
        coordinator._proof_framing_context = "stale proof emphasis"

        with mock.patch.object(
            autonomous_coordinator_module.api_client_manager,
            "generate_completion",
            new=mock.AsyncMock(),
        ) as generate_completion:
            await coordinator._run_proof_framing_gate()

        generate_completion.assert_not_awaited()
        self.assertFalse(coordinator._proof_framing_active)
        self.assertEqual(coordinator._proof_framing_context, "")

    def test_disabled_proof_output_blocks_persisted_framing_injection(self) -> None:
        coordinator = AutonomousCoordinator()
        coordinator._allow_mathematical_proofs = False
        coordinator._proof_framing_active = True
        coordinator._proof_framing_context = "stale proof emphasis"

        self.assertEqual(coordinator._append_proof_framing("Research prompt"), "Research prompt")

    def test_lightweight_legacy_coordinator_defaults_to_proof_enabled_boundary(self) -> None:
        coordinator = AutonomousCoordinator.__new__(AutonomousCoordinator)
        coordinator._proof_framing_active = True
        coordinator._proof_framing_context = "Proof emphasis"

        framed = coordinator._append_proof_framing("Research prompt")

        self.assertIn("Research prompt", framed)
        self.assertIn("Proof emphasis", framed)

    async def test_disabled_proof_output_blocks_all_proof_context_injection(self) -> None:
        coordinator = AutonomousCoordinator()
        coordinator._allow_mathematical_proofs = False
        coordinator._current_topic_id = "topic-1"

        with (
            mock.patch.object(
                autonomous_coordinator_module.proof_database,
                "inject_into_prompt",
            ) as inject_verified,
            mock.patch.object(
                autonomous_coordinator_module.proof_database,
                "inject_failure_hints_into_prompt",
                new=mock.AsyncMock(),
            ) as inject_failures,
            mock.patch.object(
                autonomous_coordinator_module.proof_database,
                "count_proofs",
            ) as count_proofs,
        ):
            effective_user = coordinator._get_effective_user_research_prompt()
            effective_brainstorm = await coordinator._get_effective_brainstorm_prompt(
                "Brainstorm prompt"
            )

        self.assertEqual(effective_user, "")
        self.assertEqual(effective_brainstorm, "Brainstorm prompt")
        inject_verified.assert_not_called()
        inject_failures.assert_not_awaited()
        count_proofs.assert_not_called()

    async def test_resume_does_not_restore_persisted_framing_when_proofs_are_disabled(self) -> None:
        coordinator = AutonomousCoordinator()
        coordinator._allow_mathematical_proofs = False
        coordinator._user_research_prompt = "Research prompt"
        persisted_state = {
            "current_tier": "tier1_aggregation",
            "proof_framing_active": True,
            "proof_framing_context": "stale proof emphasis",
            "proof_framing_reasoning": "Previously enabled",
        }
        metadata = mock.Mock()
        metadata.has_interrupted_workflow.return_value = True
        metadata.get_workflow_state = mock.AsyncMock(return_value=persisted_state)
        metadata.get_base_user_prompt = mock.AsyncMock(return_value="Research prompt")
        metadata.set_proof_framing_state = mock.AsyncMock()

        with (
            mock.patch.object(
                coordinator,
                "_normalize_resume_state",
                new=mock.AsyncMock(return_value=persisted_state),
            ),
            mock.patch.object(
                autonomous_coordinator_module,
                "research_metadata",
                metadata,
            ),
        ):
            await coordinator._check_resume_state()

        self.assertFalse(coordinator._proof_framing_active)
        self.assertEqual(coordinator._proof_framing_context, "")
        self.assertEqual(coordinator._append_proof_framing("Research prompt"), "Research prompt")
        metadata.set_proof_framing_state.assert_awaited_once()
        self.assertFalse(metadata.set_proof_framing_state.await_args.kwargs["active"])

    async def test_papers_only_run_suppresses_assistant_proof_memory(self) -> None:
        manager = autonomous_coordinator_module.api_client_manager
        manager.set_assistant_memory_suppressed("autonomous_allowed_outputs", True)
        try:
            with mock.patch.object(
                api_client_manager_module.assistant_proof_search_coordinator,
                "submit_target",
            ) as submit_target:
                messages, target_hash = await manager._maybe_add_assistant_memory_context(
                    task_id="agg_sub1_001",
                    role_id="autonomous_topic_selector",
                    role_config=mock.Mock(),
                    messages=[{"role": "user", "content": "Research prompt"}],
                    max_tokens=100,
                    tools=None,
                    tool_choice=None,
                    workflow_mode_override="autonomous",
                )
        finally:
            manager.set_assistant_memory_suppressed("autonomous_allowed_outputs", False)

        self.assertEqual(messages, [{"role": "user", "content": "Research prompt"}])
        self.assertEqual(target_hash, "")
        submit_target.assert_not_called()


class AllowedOutputRouteTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._generic_mode = system_config.generic_mode
        self._lean_enabled = system_config.lean4_enabled

    def tearDown(self) -> None:
        system_config.generic_mode = self._generic_mode
        system_config.lean4_enabled = self._lean_enabled
        compiler_route.compiler_coordinator.is_running = False
        compiler_route._release_compiler_workflow_lease()

    def _compiler_request(self, **overrides) -> CompilerStartRequest:
        data = {
            "compiler_prompt": "Write a paper",
            "allow_mathematical_proofs": True,
            "allow_research_papers": True,
            "validator_model": "validator-model",
            "validator_context_size": 1000,
            "validator_max_output_tokens": 100,
            "writer_model": "writer-model",
            "writer_context_size": 1000,
            "writer_max_output_tokens": 100,
            "high_param_model": "rigor-model",
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
            "writer_model": "writer-model",
            "writer_context_window": 1000,
            "writer_max_tokens": 100,
            "high_param_model": "rigor-model",
            "high_param_context_window": 1000,
            "high_param_max_tokens": 100,
            "critique_submitter_model": "critique-model",
            "critique_submitter_context_window": 1000,
            "critique_submitter_max_tokens": 100,
        }
        data.update(overrides)
        return AutonomousResearchStartRequest(**data)

    def test_compiler_request_accepts_previous_writer_field_names(self) -> None:
        legacy_prefix = "_".join(("high", "context"))
        request = self._compiler_request(
            writer_model="new-writer-model",
            **{
                f"{legacy_prefix}_model": "previous-writer-model",
                f"{legacy_prefix}_context_size": 2000,
                f"{legacy_prefix}_max_output_tokens": 200,
            },
        )

        self.assertEqual(request.writer_model, "new-writer-model")

        legacy_only = CompilerStartRequest(
            compiler_prompt="Write a paper",
            validator_model="validator-model",
            validator_context_size=1000,
            validator_max_output_tokens=100,
            **{
                f"{legacy_prefix}_model": "previous-writer-model",
                f"{legacy_prefix}_context_size": 2000,
                f"{legacy_prefix}_max_output_tokens": 200,
            },
            high_param_model="rigor-model",
            high_param_context_size=1000,
            high_param_max_output_tokens=100,
        )
        self.assertEqual(legacy_only.writer_model, "previous-writer-model")
        self.assertEqual(legacy_only.writer_context_size, 2000)
        self.assertEqual(legacy_only.writer_max_output_tokens, 200)

    def test_autonomous_request_accepts_previous_writer_field_names(self) -> None:
        legacy_prefix = "_".join(("high", "context"))
        data = {
            "user_research_prompt": "Research prompt",
            "submitter_configs": [],
            "validator_model": "validator-model",
            f"{legacy_prefix}_model": "previous-writer-model",
            f"{legacy_prefix}_context_window": 2000,
            f"{legacy_prefix}_max_tokens": 200,
            "high_param_model": "rigor-model",
            "high_param_context_window": 3000,
            "high_param_max_tokens": 300,
        }

        request = AutonomousResearchStartRequest(**data)

        self.assertEqual(request.writer_model, "previous-writer-model")
        self.assertEqual(request.writer_context_window, 2000)
        self.assertEqual(request.writer_max_tokens, 200)

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
        async def mark_started():
            compiler_route.compiler_coordinator.is_running = True

        with mock.patch.object(compiler_route, "_get_start_conflict", return_value=None):
            with mock.patch.object(compiler_route.compiler_coordinator, "initialize", initialize):
                with mock.patch.object(
                    compiler_route.compiler_coordinator,
                    "start",
                    mock.AsyncMock(side_effect=mark_started),
                ):
                    response = await compiler_route.start_compiler(self._compiler_request())

        self.assertEqual(response["status"], "started")
        self.assertFalse(initialize.await_args.kwargs["allow_mathematical_proofs"])

    async def test_compiler_model_diagnostics_unavailable_in_generic_mode(self) -> None:
        system_config.generic_mode = True

        with self.assertRaises(HTTPException) as exc:
            await compiler_route.test_models(self._compiler_request())

        self.assertEqual(exc.exception.status_code, 501)
        self.assertTrue(exc.exception.detail["generic_mode"])

    async def test_compiler_proof_only_uses_rigor_and_proofs_submitter_settings(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_enabled = True
        request = self._compiler_request(
            allow_research_papers=False,
            writer_model="writer-model",
            writer_context_size=2000,
            writer_max_output_tokens=200,
            high_param_provider="openrouter",
            high_param_model="rigor-model",
            high_param_openrouter_provider="RigorHost",
            high_param_openrouter_reasoning_effort="high",
            high_param_lm_studio_fallback="rigor-fallback",
            high_param_context_size=3000,
            high_param_max_output_tokens=300,
            high_param_supercharge_enabled=True,
        )
        configured_roles = {}
        stage_calls = {}

        class FakeStage:
            async def run(self, **kwargs):
                stage_calls.update(kwargs)

        def capture_role(role_id, config):
            configured_roles[role_id] = config

        with (
            mock.patch.object(compiler_route, "_read_manual_aggregator_context", new=mock.AsyncMock(return_value="accepted brainstorm")),
            mock.patch.object(compiler_route, "_release_pre_reserved_source", new=mock.AsyncMock()),
            mock.patch.object(compiler_route.websocket, "broadcast_event", new=mock.AsyncMock()),
            mock.patch.object(compiler_route.api_client_manager, "configure_role", side_effect=capture_role),
            mock.patch.object(compiler_route, "ProofVerificationStage", return_value=FakeStage()),
            mock.patch.object(compiler_route.assistant_proof_search_coordinator, "stop_all", new=mock.AsyncMock()),
            mock.patch.object(compiler_route.token_tracker, "reset"),
            mock.patch.object(compiler_route.token_tracker, "start_timer"),
            mock.patch.object(compiler_route.token_tracker, "stop_timer"),
        ):
            await compiler_route._run_compiler_aggregator_proof_check(request, source_reserved=True)

        proof_role = configured_roles["autonomous_proof_formalization_compiler_aggregator"]
        self.assertEqual(proof_role.provider, "openrouter")
        self.assertEqual(proof_role.model_id, "rigor-model")
        self.assertEqual(proof_role.openrouter_provider, "RigorHost")
        self.assertEqual(proof_role.openrouter_reasoning_effort, "high")
        self.assertEqual(proof_role.lm_studio_fallback_id, "rigor-fallback")
        self.assertEqual(proof_role.context_window, 3000)
        self.assertEqual(proof_role.max_output_tokens, 300)
        self.assertTrue(proof_role.supercharge_enabled)
        self.assertEqual(stage_calls["submitter_model"], "rigor-model")
        self.assertEqual(stage_calls["submitter_context"], 3000)
        self.assertEqual(stage_calls["submitter_max_tokens"], 300)

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

    async def test_autonomous_desktop_papers_only_run_preserves_disabled_proof_output(self) -> None:
        system_config.generic_mode = False
        system_config.lean4_enabled = True
        initialize = mock.AsyncMock()

        with mock.patch.object(autonomous_route, "_get_start_conflict", return_value=None):
            with mock.patch.object(autonomous_route, "require_embedding_provider_ready", mock.AsyncMock()):
                with mock.patch.object(autonomous_route.autonomous_coordinator, "initialize", initialize):
                    with mock.patch.object(
                        autonomous_route.autonomous_coordinator,
                        "start_in_background",
                        return_value=True,
                    ):
                        response = await autonomous_route.start_autonomous_research(
                            self._autonomous_request(
                                allow_mathematical_proofs=False,
                                allow_research_papers=True,
                            )
                        )

        self.assertTrue(response["success"])
        self.assertFalse(initialize.await_args.kwargs["allow_mathematical_proofs"])
        self.assertTrue(initialize.await_args.kwargs["allow_research_papers"])


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
