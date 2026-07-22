import unittest
from types import SimpleNamespace
from unittest import mock

from backend.api.routes import aggregator as aggregator_route
from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import proofs as proofs_route
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.shared.models import (
    AggregatorStartRequest,
    AutonomousResearchStartRequest,
    CompilerStartRequest,
    LeanOJRoleConfig,
    LeanOJStartRequest,
    ModelConfig,
    ProofCheckRequest,
    SubmitterConfig,
)


def _submitter(model_id: str = "submitter-model") -> SubmitterConfig:
    return SubmitterConfig(
        submitter_id=1,
        provider="openrouter",
        model_id=model_id,
        openrouter_provider="SubmitterHost",
        openrouter_reasoning_effort="medium",
        context_window=4096,
        max_output_tokens=512,
    )


def _leanoj_role(model_id: str, provider: str = "openrouter") -> LeanOJRoleConfig:
    return LeanOJRoleConfig(
        provider=provider,
        model_id=model_id,
        openrouter_provider=f"{model_id}-host" if provider == "openrouter" else None,
        openrouter_reasoning_effort="high",
        context_window=4096,
        max_output_tokens=512,
        supercharge_enabled=True,
    )


class AssistantRoleSchemaTests(unittest.TestCase):
    def test_start_request_schemas_accept_explicit_assistant_fields(self) -> None:
        aggregator = AggregatorStartRequest(
            user_prompt="Aggregate.",
            submitter_configs=[_submitter()],
            validator_provider="openrouter",
            validator_model="validator-model",
            assistant_provider="openrouter",
            assistant_model="assistant-model",
            assistant_openrouter_provider="AssistantHost",
            assistant_openrouter_reasoning_effort="low",
            assistant_context_size=8192,
            assistant_max_output_tokens=1024,
            assistant_supercharge_enabled=True,
        )
        compiler = CompilerStartRequest(
            compiler_prompt="Compile.",
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            writer_provider="openrouter",
            writer_model="writer-model",
            high_param_provider="openrouter",
            high_param_model="hp-model",
            critique_submitter_provider="openrouter",
            critique_submitter_model="critique-model",
            assistant_provider="openrouter",
            assistant_model="assistant-model",
            assistant_openrouter_provider="AssistantHost",
            assistant_context_size=8192,
            assistant_max_output_tokens=1024,
            assistant_supercharge_enabled=True,
        )
        autonomous = AutonomousResearchStartRequest(
            user_research_prompt="Research.",
            submitter_configs=[_submitter()],
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            writer_provider="openrouter",
            writer_model="writer-model",
            high_param_provider="openrouter",
            high_param_model="hp-model",
            critique_submitter_provider="openrouter",
            critique_submitter_model="critique-model",
            assistant_provider="openrouter",
            assistant_model="assistant-model",
            assistant_openrouter_provider="AssistantHost",
            assistant_context_window=8192,
            assistant_max_tokens=1024,
            assistant_supercharge_enabled=True,
        )
        leanoj = LeanOJStartRequest(
            user_prompt="Solve.",
            lean_template="theorem target : True := by\n  trivial",
            topic_generator=_leanoj_role("topic-generator"),
            topic_validator=_leanoj_role("topic-validator"),
            brainstorm_submitters=[_leanoj_role("brainstorm-submitter")],
            brainstorm_validator=_leanoj_role("brainstorm-validator"),
            final_solver=_leanoj_role("final-solver"),
            assistant=_leanoj_role("assistant-model"),
        )

        self.assertEqual(aggregator.assistant_model, "assistant-model")
        self.assertEqual(compiler.assistant_model, "assistant-model")
        self.assertEqual(autonomous.assistant_model, "assistant-model")
        self.assertEqual(leanoj.assistant.model_id, "assistant-model")

    def test_start_request_schemas_allow_omitted_assistant_for_validator_fallback(self) -> None:
        aggregator = AggregatorStartRequest(
            user_prompt="Aggregate.",
            submitter_configs=[_submitter()],
            validator_provider="openrouter",
            validator_model="validator-model",
        )
        compiler = CompilerStartRequest(
            compiler_prompt="Compile.",
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            writer_provider="openrouter",
            writer_model="writer-model",
            writer_context_size=4096,
            writer_max_output_tokens=512,
            high_param_provider="openrouter",
            high_param_model="hp-model",
            high_param_context_size=4096,
            high_param_max_output_tokens=512,
            critique_submitter_provider="openrouter",
            critique_submitter_model="critique-model",
            critique_submitter_context_window=4096,
            critique_submitter_max_tokens=512,
        )
        autonomous = AutonomousResearchStartRequest(
            user_research_prompt="Research.",
            submitter_configs=[_submitter()],
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
        )
        leanoj = LeanOJStartRequest(
            user_prompt="Solve.",
            lean_template="theorem target : True := by\n  trivial",
            topic_generator=_leanoj_role("topic-generator"),
            topic_validator=_leanoj_role("topic-validator"),
            brainstorm_submitters=[_leanoj_role("brainstorm-submitter")],
            brainstorm_validator=_leanoj_role("brainstorm-validator"),
            final_solver=_leanoj_role("final-solver"),
        )

        self.assertEqual(aggregator.assistant_model, "")
        self.assertEqual(compiler.assistant_model, "")
        self.assertEqual(autonomous.assistant_model, "")
        self.assertEqual(leanoj.assistant.model_id, "")

    def test_compiler_start_request_allows_omitted_deprecated_critique_role(self) -> None:
        compiler = CompilerStartRequest(
            compiler_prompt="Compile.",
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            validator_context_size=4096,
            validator_max_output_tokens=512,
            writer_provider="openrouter",
            writer_model="writer-model",
            writer_context_size=4096,
            writer_max_output_tokens=512,
            high_param_provider="openrouter",
            high_param_model="rigor-model",
            high_param_context_size=8192,
            high_param_max_output_tokens=1024,
        )

        self.assertEqual(compiler.critique_submitter_model, "")
        self.assertEqual(compiler.high_param_model, "rigor-model")


class AssistantRoleDefaultingTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        aggregator_route.coordinator.is_running = False
        compiler_route.compiler_coordinator.is_running = False
        aggregator_route._release_aggregator_workflow_lease()
        compiler_route._release_compiler_workflow_lease()

    async def test_aggregator_route_defaults_omitted_assistant_to_validator_config(self) -> None:
        request = AggregatorStartRequest(
            user_prompt="Aggregate.",
            submitter_configs=[_submitter()],
            validator_provider="openrouter",
            validator_model="validator-model",
            validator_openrouter_provider="ValidatorHost",
            validator_openrouter_reasoning_effort="xhigh",
            validator_lm_studio_fallback="validator-fallback",
            validator_context_size=7777,
            validator_max_output_tokens=777,
            validator_supercharge_enabled=True,
        )

        with (
            mock.patch.object(aggregator_route, "_get_start_conflict", return_value=None),
            mock.patch.object(aggregator_route, "save_manual_aggregator_prompt", new=mock.AsyncMock()),
            mock.patch.object(aggregator_route, "require_embedding_provider_ready", new=mock.AsyncMock()),
            mock.patch.object(aggregator_route.api_client_manager, "configure_role") as configure_role,
            mock.patch.object(aggregator_route.coordinator, "initialize", new=mock.AsyncMock()),
            mock.patch.object(
                aggregator_route.coordinator,
                "start",
                new=mock.AsyncMock(
                    side_effect=lambda: setattr(aggregator_route.coordinator, "is_running", True)
                ),
            ),
            mock.patch.object(aggregator_route.token_tracker, "reset"),
            mock.patch.object(aggregator_route.token_tracker, "start_timer"),
        ):
            await aggregator_route.start_aggregator(request)

        role_id, config = configure_role.call_args.args
        self.assertEqual(role_id, "aggregator_assistant")
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.model_id, "validator-model")
        self.assertEqual(config.openrouter_provider, "ValidatorHost")
        self.assertEqual(config.openrouter_reasoning_effort, "xhigh")
        self.assertEqual(config.lm_studio_fallback_id, "validator-fallback")
        self.assertEqual(config.context_window, 7777)
        self.assertEqual(config.max_output_tokens, 777)
        self.assertTrue(config.supercharge_enabled)

    async def test_compiler_route_defaults_omitted_assistant_to_validator_config(self) -> None:
        request = CompilerStartRequest(
            compiler_prompt="Compile.",
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            validator_openrouter_provider="ValidatorHost",
            validator_openrouter_reasoning_effort="xhigh",
            validator_lm_studio_fallback="validator-fallback",
            validator_context_size=7777,
            validator_max_output_tokens=777,
            validator_supercharge_enabled=True,
            writer_provider="openrouter",
            writer_model="writer-model",
            writer_context_size=4096,
            writer_max_output_tokens=512,
            high_param_provider="openrouter",
            high_param_model="hp-model",
            high_param_context_size=4096,
            high_param_max_output_tokens=512,
            critique_submitter_provider="openrouter",
            critique_submitter_model="critique-model",
            critique_submitter_context_window=4096,
            critique_submitter_max_tokens=512,
        )

        with (
            mock.patch.object(compiler_route, "_get_start_conflict", return_value=None),
            mock.patch.object(compiler_route, "save_manual_compiler_prompt", new=mock.AsyncMock()),
            mock.patch.object(compiler_route, "require_embedding_provider_ready", new=mock.AsyncMock()),
            mock.patch.object(compiler_route.api_client_manager, "configure_role") as configure_role,
            mock.patch.object(compiler_route.compiler_coordinator, "initialize", new=mock.AsyncMock()),
            mock.patch.object(
                compiler_route.compiler_coordinator,
                "start",
                new=mock.AsyncMock(
                    side_effect=lambda: setattr(
                        compiler_route.compiler_coordinator, "is_running", True
                    )
                ),
            ),
            mock.patch.object(compiler_route.token_tracker, "reset"),
            mock.patch.object(compiler_route.token_tracker, "start_timer"),
        ):
            await compiler_route.start_compiler(request)

        role_id, config = configure_role.call_args.args
        self.assertEqual(role_id, "compiler_assistant")
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.model_id, "validator-model")
        self.assertEqual(config.openrouter_provider, "ValidatorHost")
        self.assertEqual(config.openrouter_reasoning_effort, "xhigh")
        self.assertEqual(config.lm_studio_fallback_id, "validator-fallback")
        self.assertEqual(config.context_window, 7777)
        self.assertEqual(config.max_output_tokens, 777)
        self.assertTrue(config.supercharge_enabled)

    async def test_compiler_route_mirrors_deprecated_critique_fields_from_rigor_config(self) -> None:
        request = CompilerStartRequest(
            compiler_prompt="Compile.",
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            validator_context_size=4096,
            validator_max_output_tokens=512,
            writer_provider="openrouter",
            writer_model="writer-model",
            writer_context_size=4096,
            writer_max_output_tokens=512,
            high_param_provider="openrouter",
            high_param_model="rigor-model",
            high_param_openrouter_provider="RigorHost",
            high_param_openrouter_reasoning_effort="high",
            high_param_lm_studio_fallback="rigor-fallback",
            high_param_context_size=8192,
            high_param_max_output_tokens=1024,
            high_param_supercharge_enabled=True,
        )

        with (
            mock.patch.object(compiler_route, "_get_start_conflict", return_value=None),
            mock.patch.object(compiler_route, "save_manual_compiler_prompt", new=mock.AsyncMock()),
            mock.patch.object(compiler_route, "require_embedding_provider_ready", new=mock.AsyncMock()),
            mock.patch.object(compiler_route.api_client_manager, "configure_role"),
            mock.patch.object(compiler_route.compiler_coordinator, "initialize", new=mock.AsyncMock()) as initialize,
            mock.patch.object(
                compiler_route.compiler_coordinator,
                "start",
                new=mock.AsyncMock(
                    side_effect=lambda: setattr(
                        compiler_route.compiler_coordinator, "is_running", True
                    )
                ),
            ),
            mock.patch.object(compiler_route.token_tracker, "reset"),
            mock.patch.object(compiler_route.token_tracker, "start_timer"),
        ):
            await compiler_route.start_compiler(request)

        kwargs = initialize.await_args.kwargs
        self.assertEqual(kwargs["critique_submitter_model"], "rigor-model")
        self.assertEqual(kwargs["critique_submitter_provider"], "openrouter")
        self.assertEqual(kwargs["critique_submitter_openrouter_provider"], "RigorHost")
        self.assertEqual(kwargs["critique_submitter_openrouter_reasoning_effort"], "high")
        self.assertEqual(kwargs["critique_submitter_lm_studio_fallback"], "rigor-fallback")
        self.assertTrue(kwargs["critique_submitter_supercharge_enabled"])
        self.assertEqual(compiler_route.system_config.compiler_critique_submitter_model, "rigor-model")
        self.assertEqual(compiler_route.system_config.compiler_critique_submitter_context_window, 8192)
        self.assertEqual(compiler_route.system_config.compiler_critique_submitter_max_tokens, 1024)

    async def test_autonomous_route_defaults_omitted_assistant_to_validator_config(self) -> None:
        request = AutonomousResearchStartRequest(
            user_research_prompt="Research.",
            submitter_configs=[_submitter()],
            allow_mathematical_proofs=False,
            validator_provider="openrouter",
            validator_model="validator-model",
            validator_openrouter_provider="ValidatorHost",
            validator_openrouter_reasoning_effort="xhigh",
            validator_lm_studio_fallback="validator-fallback",
            validator_context_window=7777,
            validator_max_tokens=777,
            validator_supercharge_enabled=True,
        )

        with (
            mock.patch.object(autonomous_route, "_get_start_conflict", return_value=None),
            mock.patch.object(autonomous_route, "require_embedding_provider_ready", new=mock.AsyncMock()),
            mock.patch.object(autonomous_route.autonomous_coordinator, "initialize", new=mock.AsyncMock()) as initialize,
            mock.patch.object(autonomous_route.autonomous_coordinator, "start_in_background", return_value=True),
        ):
            await autonomous_route.start_autonomous_research(request)

        kwargs = initialize.await_args.kwargs
        self.assertEqual(kwargs["assistant_provider"], "openrouter")
        self.assertEqual(kwargs["assistant_model"], "validator-model")
        self.assertEqual(kwargs["assistant_openrouter_provider"], "ValidatorHost")
        self.assertEqual(kwargs["assistant_openrouter_reasoning_effort"], "xhigh")
        self.assertEqual(kwargs["assistant_lm_studio_fallback"], "validator-fallback")
        self.assertEqual(kwargs["assistant_context_window"], 7777)
        self.assertEqual(kwargs["assistant_max_tokens"], 777)
        self.assertTrue(kwargs["assistant_supercharge_enabled"])

    def test_leanoj_role_config_defaults_omitted_assistant_to_topic_validator(self) -> None:
        request = LeanOJStartRequest(
            user_prompt="Solve.",
            lean_template="theorem target : True := by\n  trivial",
            topic_generator=_leanoj_role("topic-generator"),
            topic_validator=_leanoj_role("topic-validator"),
            brainstorm_submitters=[_leanoj_role("brainstorm-submitter")],
            brainstorm_validator=_leanoj_role("brainstorm-validator"),
            final_solver=_leanoj_role("final-solver"),
        )
        coordinator = LeanOJCoordinator()

        with mock.patch(
            "backend.leanoj.core.leanoj_coordinator.api_client_manager.configure_role"
        ) as configure_role:
            coordinator._configure_roles(request)

        assistant_calls = [
            call for call in configure_role.call_args_list if call.args[0] == "leanoj_assistant"
        ]
        self.assertEqual(len(assistant_calls), 1)
        assistant_config = assistant_calls[0].args[1]
        self.assertEqual(assistant_config.provider, "openrouter")
        self.assertEqual(assistant_config.model_id, "topic-validator")
        self.assertEqual(assistant_config.openrouter_provider, "topic-validator-host")
        self.assertEqual(assistant_config.openrouter_reasoning_effort, "high")
        self.assertTrue(assistant_config.supercharge_enabled)


class ManualProofAssistantSnapshotTests(unittest.TestCase):
    def _preserve_role_config(self, role_id: str):
        configs = proofs_route.api_client_manager._role_model_configs
        old_present = role_id in configs
        old_config = configs.get(role_id)

        def restore():
            if old_present:
                configs[role_id] = old_config
            else:
                configs.pop(role_id, None)

        return restore

    def test_active_manual_aggregator_snapshot_preserves_distinct_assistant_role(self) -> None:
        restore = self._preserve_role_config("aggregator_assistant")
        assistant_config = ModelConfig(
            provider="openrouter",
            model_id="assistant-model",
            openrouter_provider="AssistantHost",
            openrouter_reasoning_effort="medium",
            context_window=9999,
            max_output_tokens=999,
            supercharge_enabled=True,
        )
        try:
            proofs_route.api_client_manager.configure_role("aggregator_assistant", assistant_config)
            with (
                mock.patch.object(proofs_route.coordinator, "submitter_configs", [_submitter("submitter-model")]),
                mock.patch.object(proofs_route.coordinator, "validator_provider", "openrouter"),
                mock.patch.object(proofs_route.coordinator, "validator_model", "validator-model"),
                mock.patch.object(proofs_route.coordinator, "validator_openrouter_provider", "ValidatorHost"),
                mock.patch.object(proofs_route.coordinator, "validator_openrouter_reasoning_effort", "xhigh"),
                mock.patch.object(proofs_route.coordinator, "validator_lm_studio_fallback", None),
                mock.patch.object(proofs_route.coordinator, "validator_context_window", 7777),
                mock.patch.object(proofs_route.coordinator, "validator_max_tokens", 777),
                mock.patch.object(proofs_route.coordinator, "validator_supercharge_enabled", False),
            ):
                snapshot = proofs_route._get_active_manual_runtime_snapshot(
                    ProofCheckRequest(source_type="brainstorm", source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID)
                )
        finally:
            restore()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.assistant.model_id, "assistant-model")
        self.assertEqual(snapshot.assistant.openrouter_provider, "AssistantHost")
        self.assertEqual(snapshot.assistant.context_window, 9999)
        self.assertTrue(snapshot.assistant.supercharge_enabled)

    def test_active_manual_compiler_snapshot_preserves_distinct_assistant_role(self) -> None:
        restore = self._preserve_role_config("compiler_assistant")
        assistant_config = ModelConfig(
            provider="openrouter",
            model_id="compiler-assistant",
            openrouter_provider="CompilerAssistantHost",
            openrouter_reasoning_effort="high",
            context_window=8888,
            max_output_tokens=888,
        )
        try:
            proofs_route.api_client_manager.configure_role("compiler_assistant", assistant_config)
            with (
                mock.patch.object(
                    proofs_route.compiler_coordinator,
                    "high_param_submitter",
                    SimpleNamespace(model_name="rigor-model"),
                ),
                mock.patch.object(proofs_route.compiler_coordinator, "high_param_provider", "openrouter", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "high_param_openrouter_provider", "RigorHost", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "high_param_openrouter_reasoning_effort", "xhigh", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "high_param_lm_studio_fallback", None, create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "high_param_supercharge_enabled", False, create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_provider", "openrouter", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_model", "validator-model", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_openrouter_provider", "ValidatorHost", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_openrouter_reasoning_effort", "xhigh", create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_lm_studio_fallback", None, create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_context_window", 7777, create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_max_tokens", 777, create=True),
                mock.patch.object(proofs_route.compiler_coordinator, "validator_supercharge_enabled", False, create=True),
                mock.patch.object(proofs_route.system_config, "compiler_high_param_context_window", 8192),
                mock.patch.object(proofs_route.system_config, "compiler_high_param_max_output_tokens", 1024),
            ):
                snapshot = proofs_route._get_active_manual_runtime_snapshot(
                    ProofCheckRequest(source_type="paper", source_id=proofs_route.MANUAL_COMPILER_CURRENT_SOURCE_ID)
                )
        finally:
            restore()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.assistant.model_id, "compiler-assistant")
        self.assertEqual(snapshot.assistant.openrouter_provider, "CompilerAssistantHost")
        self.assertEqual(snapshot.assistant.context_window, 8888)
