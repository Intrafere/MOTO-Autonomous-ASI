"""Manual proof-run provider errors carry route identity in Live Activity."""
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

import httpx

from backend.api.routes import proofs as proofs_route
from backend.autonomous.core.proof_run_manager import ProofRunControl
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import system_config
from backend.shared.models import (
    ProofCheckRequest,
    ProofRoleConfigSnapshot,
    ProofRunSnapshot,
    ProofRuntimeConfigSnapshot,
)
from backend.shared.openrouter_client import CreditExhaustionError
from backend.shared.provider_errors import (
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)

PROOF_MODEL = "vendor/test-free:free"
AUTO_IDENTITY = (
    f"[Manual proof check · Proof Brainstorm · {PROOF_MODEL} via OpenRouter · "
    "host auto (upstream not reported)]"
)
PROOF_RUN_ID = "proof-run-identity"
# The run-scoped role a manual check registers for its source.
CODEX_ROLE = proofs_route._manual_proof_role_ids("brainstorm", PROOF_RUN_ID)[0]
CODEX_IDENTITY = (
    "[Manual proof check · Proof Identification (manual brainstorm) · "
    "gpt-test-codex via OpenAI Codex]"
)


def _role(model_id: str) -> ProofRoleConfigSnapshot:
    return ProofRoleConfigSnapshot(
        provider="openrouter",
        model_id=model_id,
        context_window=4096,
        max_output_tokens=512,
    )


def _runtime_snapshot() -> ProofRuntimeConfigSnapshot:
    return ProofRuntimeConfigSnapshot(
        brainstorm=_role(PROOF_MODEL),
        paper=_role(PROOF_MODEL),
        validator=_role("vendor/test-validator:free"),
    )


class _FakeProofDatabase:
    async def get_live_context_run_id(self) -> str:
        return "manual-run"

    async def get_proof_set_revision(self) -> int:
        return 0


class _FakeEventLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    async def add_event(self, event_type: str, message: str, data: dict) -> None:
        self.entries.append((event_type, message))


class _FakeAssistantCoordinator:
    def submit_target(self, snapshot):
        return snapshot.stable_hash()

    async def stop_all(self, **_kwargs) -> None:
        return None


def _source() -> proofs_route.ProofSourceAdapter:
    return proofs_route.ProofSourceAdapter(
        source_type="brainstorm",
        source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
        source_title="Manual source",
        source_content="SOURCE CONTENT",
        user_prompt="User prompt",
        canonical_user_prompt="User prompt",
        proof_database=_FakeProofDatabase(),
        proof_store_id="manual:active",
        scope=proofs_route.PROOF_SCOPE_MANUAL,
        writable=True,
        append_to_source=False,
        append_proof_callback=None,
    )


def _codex_retry_error(summary: str) -> RetryableProviderError:
    return RetryableProviderError(
        provider="openai_codex_oauth",
        provider_label="OpenAI Codex",
        role_id=CODEX_ROLE,
        model="gpt-test-codex",
        reason="transient_provider_error",
        message=summary,
        error_summary=summary,
        retry_after_seconds=7,
    )


class ManualProofRetryIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        stage_errors: list[BaseException],
        *,
        active_mode: str = "",
        adapter_retry_payload: dict | None = None,
    ):
        events: list[tuple[str, dict]] = []
        event_log = _FakeEventLog()
        provider_broadcast = mock.AsyncMock()
        calls = 0

        async def capture(event_type: str, payload: dict) -> None:
            events.append((event_type, dict(payload)))

        class FakeStage:
            async def run_manual(self, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1 and adapter_retry_payload is not None:
                    # A provider adapter retry inside the check's model call.
                    await api_client_manager._provider_retry_callback(CODEX_ROLE)(
                        dict(adapter_retry_payload)
                    )
                if calls <= len(stage_errors):
                    raise stage_errors[calls - 1]
                return None

        control = ProofRunControl(
            snapshot=ProofRunSnapshot(
                proof_run_id=PROOF_RUN_ID,
                run_mode="one_round",
                scope="manual",
                source_type="brainstorm",
                source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                proof_store_id="manual:active",
                run_id="manual-run",
                lifecycle_generation=1,
                status="running",
            ),
            # Production wiring for manual Aggregator sources: broadcast plus
            # the persisted manual event log.
            event_callback=proofs_route._broadcast_manual_aggregator_proof_event,
        )
        with tempfile.TemporaryDirectory() as data_dir, ExitStack() as stack:
            stack.enter_context(mock.patch.object(system_config, "data_dir", data_dir))
            stack.enter_context(mock.patch.dict(api_client_manager._role_model_configs))
            stack.enter_context(
                mock.patch.object(api_client_manager, "_retryable_provider_backoff_state", {})
            )
            stack.enter_context(
                mock.patch.object(
                    api_client_manager,
                    "_sleep_with_optional_stop",
                    new=mock.AsyncMock(return_value=False),
                )
            )
            stack.enter_context(
                mock.patch.object(api_client_manager, "_broadcast", new=provider_broadcast)
            )
            stack.enter_context(
                mock.patch(
                    "backend.shared.api_client_manager._active_notification_workflow_mode",
                    return_value=active_mode,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    proofs_route,
                    "_resolve_proof_source_adapter",
                    new=mock.AsyncMock(return_value=_source()),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    proofs_route,
                    "_refresh_manual_assistant_memory",
                    new=mock.AsyncMock(),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    proofs_route,
                    "assistant_proof_search_coordinator",
                    _FakeAssistantCoordinator(),
                )
            )
            websocket_mock = stack.enter_context(mock.patch.object(proofs_route, "websocket"))
            websocket_mock.broadcast_event = capture
            stack.enter_context(mock.patch.object(proofs_route, "event_log", event_log))
            stack.enter_context(
                mock.patch.object(
                    proofs_route.autonomous_coordinator,
                    "_proof_verification_stage",
                    FakeStage(),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    proofs_route.proof_run_manager,
                    "wait_for_provider_or_control_wake",
                    new=mock.AsyncMock(),
                )
            )
            await proofs_route._run_manual_proof_check(
                ProofCheckRequest(
                    source_type="brainstorm",
                    source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                ),
                control,
                _runtime_snapshot(),
            )
        provider_events = [
            (call.args[0], call.args[1]) for call in provider_broadcast.await_args_list
        ]
        return events, dict(event_log.entries), provider_events, control

    async def test_retryable_provider_error_events_keep_identity_and_detail_parsing(self):
        summary = (
            "OpenAI Codex SSE event response.failed ended with status=failed. "
            "Detail: upstream overloaded Waiting for capacity"
        )
        events, persisted, provider_events, _control = await self._run(
            [_codex_retry_error(summary)]
        )
        by_event = dict(events)

        scheduled = by_event["proof_retry_scheduled"]
        self.assertEqual(scheduled["workflow_mode"], "manual_proof_check")
        self.assertEqual(scheduled["role_id"], CODEX_ROLE)
        self.assertEqual(scheduled["model"], "gpt-test-codex")
        self.assertEqual(scheduled["provider"], "openai_codex_oauth")
        self.assertEqual(scheduled["host_routing"], "direct")
        self.assertEqual(scheduled["route_identity"], CODEX_IDENTITY)
        self.assertTrue(scheduled["message"].startswith(f"{CODEX_IDENTITY} OpenAI Codex API error:"))
        self.assertNotIn(CODEX_IDENTITY, scheduled["error_summary"])

        started = by_event["proof_retry_started"]
        self.assertEqual(
            started["message"],
            f"{CODEX_IDENTITY} Retry 1 is starting now from the preserved "
            "proof-round checkpoint. Last error: upstream overloaded",
        )
        self.assertEqual(started["message"].count(CODEX_IDENTITY), 1)

        resumed = by_event["proof_run_provider_resumed"]
        self.assertEqual(
            resumed["message"],
            f"{CODEX_IDENTITY} OpenAI Codex retry 1 succeeded; the preserved proof round resumed.",
        )

        self.assertTrue(persisted["proof_retry_scheduled"].startswith(CODEX_IDENTITY))
        self.assertEqual(persisted["proof_retry_started"], started["message"])
        self.assertEqual(
            persisted["proof_run_provider_resumed"],
            f"{CODEX_IDENTITY} Round 1 resumed after the provider pause",
        )

        cooldown = dict(provider_events)["provider_retry_cooldown"]
        self.assertEqual(cooldown["workflow_mode"], "manual_proof_check")
        self.assertEqual(cooldown["route_identity"], CODEX_IDENTITY)
        self.assertTrue(cooldown["message"].startswith(CODEX_IDENTITY))

    async def test_retry_started_last_error_comes_from_the_error_summary(self):
        cases = (
            (
                "OpenAI Codex SSE event response.failed ended with status=failed. "
                "Detail: upstream overloaded",
                " Last error: upstream overloaded",
            ),
            (
                "OpenAI Codex SSE event response.failed ended with status=failed. "
                'Detail: {"status": "failed"}',
                ' Last error: {"status": "failed"}',
            ),
            ("OpenAI Codex stream disconnected before completion.", ""),
        )
        for summary, last_error in cases:
            with self.subTest(summary=summary):
                events, persisted, _provider_events, _control = await self._run(
                    [_codex_retry_error(summary)]
                )
                started = dict(events)["proof_retry_started"]
                self.assertEqual(
                    started["message"],
                    f"{CODEX_IDENTITY} Retry 1 is starting now from the preserved "
                    f"proof-round checkpoint.{last_error}",
                )
                self.assertNotIn("will retry after", started["message"])
                self.assertEqual(persisted["proof_retry_started"], started["message"])
                # The waiting row keeps the composed provider message.
                self.assertIn(
                    "will retry after a 7-second cooldown",
                    dict(events)["proof_retry_scheduled"]["message"],
                )

    async def test_one_check_labels_every_retry_manual_under_any_active_owner(self):
        summary = (
            "OpenAI Codex SSE event response.failed ended with status=failed. "
            "Detail: upstream overloaded"
        )
        adapter_retry = {
            "provider": "openai_codex_oauth",
            "provider_label": "OpenAI Codex",
            "model": "gpt-test-codex",
            "retry_attempt": 1,
            "max_attempts": 3,
            "retry_after_seconds": 2,
            "reason": "stream_error",
            "error_summary": "synthetic stream disconnect",
        }
        for owner_mode in ("autonomous", "compiler", "aggregator"):
            with self.subTest(owner=owner_mode):
                _events, _persisted, provider_events, _control = await self._run(
                    [
                        _codex_retry_error(summary),
                        RuntimeError("OpenRouter connection timeout"),
                    ],
                    active_mode=owner_mode,
                    adapter_retry_payload=adapter_retry,
                )
                cooldowns = [
                    payload
                    for event, payload in provider_events
                    if event == "provider_retry_cooldown"
                ]
                # Adapter-level, workflow-level retryable, and transient retries.
                self.assertEqual(
                    [payload["reason"] for payload in cooldowns],
                    ["stream_error", "transient_provider_error", "transient_model_error"],
                )
                for payload in cooldowns:
                    self.assertEqual(payload["workflow_mode"], "manual_proof_check")
                    self.assertTrue(
                        payload["route_identity"].startswith("[Manual proof check · "),
                        payload["route_identity"],
                    )
                    # Every retry names the check's source so the UI routes it to
                    # the feed that owns the check.
                    self.assertEqual(
                        payload["source_id"],
                        proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                    )
                    self.assertEqual(payload["source_type"], "brainstorm")
                # Only the workflow backoff, which also renders as the source's
                # proof_retry_* line, is marked as a workflow-layer retry.
                self.assertEqual(
                    [payload["retry_layer"] for payload in cooldowns],
                    ["adapter", "workflow", "adapter"],
                )
                # The source registration ends with the check.
                self.assertNotIn(CODEX_ROLE, api_client_manager._role_activity_sources)

    async def test_transient_error_without_route_uses_proof_role_settings(self):
        error = RuntimeError("OpenRouter connection timeout")
        _events, _persisted, provider_events, _control = await self._run([error])

        cooldown = dict(provider_events)["provider_retry_cooldown"]
        self.assertEqual(cooldown["workflow_mode"], "manual_proof_check")
        self.assertEqual(cooldown["role_id"], "manual_proof_brainstorm")
        self.assertEqual(cooldown["model"], PROOF_MODEL)
        self.assertEqual(cooldown["provider"], "openrouter")
        self.assertEqual(cooldown["provider_label"], "OpenRouter")
        self.assertEqual(cooldown["configured_host_provider"], "")
        self.assertEqual(cooldown["host_routing"], "auto")
        self.assertEqual(cooldown["route_identity"], AUTO_IDENTITY)
        self.assertEqual(cooldown["error_summary"], "OpenRouter connection timeout")
        self.assertEqual(
            cooldown["message"],
            f"{AUTO_IDENTITY} OpenRouter API error: OpenRouter connection timeout. "
            "OpenRouter API call will retry after a 5-second cooldown (retry 1).",
        )

    async def test_transient_route_error_uses_the_failing_route(self):
        error = ProviderRouteError(
            "OpenRouter upstream reset",
            route=ProviderRouteIdentity(
                provider="openrouter",
                model="vendor/route-model",
                role_id="autonomous_proof_formalization_manual_brainstorm_testrun",
                host_provider="TestHost",
            ),
            cause=httpx.ConnectError("connection reset"),
        )
        _events, _persisted, provider_events, _control = await self._run([error])

        cooldown = dict(provider_events)["provider_retry_cooldown"]
        identity = (
            "[Manual proof check · Proof Formalization (manual brainstorm) · "
            "vendor/route-model via OpenRouter · host TestHost (pinned)]"
        )
        self.assertEqual(
            cooldown["role_id"],
            "autonomous_proof_formalization_manual_brainstorm_testrun",
        )
        self.assertEqual(cooldown["model"], "vendor/route-model")
        self.assertEqual(cooldown["configured_host_provider"], "TestHost")
        self.assertEqual(cooldown["host_routing"], "pinned")
        self.assertEqual(cooldown["route_identity"], identity)
        self.assertTrue(cooldown["message"].startswith(f"{identity} OpenRouter API error:"))

    async def test_repair_required_terminal_summary_names_the_route(self):
        error = ProviderRepairRequiredError(
            provider="xai_grok_oauth",
            provider_label="xAI Grok",
            role_id=CODEX_ROLE,
            model="grok-test",
            reason="provider_spending_limit",
            message="xAI Grok spending limit reached.",
            error_detail="xAI Grok spending limit reached; raise the limit.",
        )
        events, persisted, _provider_events, control = await self._run([error])

        identity = (
            "[Manual proof check · Proof Identification (manual brainstorm) · "
            "grok-test via xAI Grok]"
        )
        terminal = dict(events)["proof_run_terminal"]
        self.assertEqual(terminal["terminal_reason"], "repair_required")
        self.assertEqual(terminal["route_identity"], identity)
        self.assertEqual(terminal["model"], "grok-test")
        self.assertEqual(terminal["provider"], "xai_grok_oauth")
        self.assertTrue(terminal["last_error_summary"].startswith(f"{identity} xAI Grok spending limit"))
        self.assertTrue(control.snapshot.last_error_summary.startswith(identity))
        self.assertTrue(persisted["proof_run_terminal"].startswith(f"Proof run ended: {identity}"))
        self.assertNotIn(identity, str(error))

    async def test_credit_pause_names_the_proof_route(self):
        events, persisted, _provider_events, _control = await self._run(
            [CreditExhaustionError("OpenRouter credits exhausted")]
        )

        paused = dict(events)["proof_run_provider_paused"]
        self.assertEqual(paused["workflow_mode"], "manual_proof_check")
        self.assertEqual(paused["model"], PROOF_MODEL)
        self.assertEqual(paused["route_identity"], AUTO_IDENTITY)
        self.assertEqual(
            paused["message"],
            f"{AUTO_IDENTITY} Proof run paused for provider credits.",
        )
        self.assertEqual(
            persisted["proof_run_provider_paused"],
            f"{AUTO_IDENTITY} Round 1 paused for provider credits",
        )


class ManualAggregatorProofEventMessageTests(unittest.TestCase):
    def test_retry_events_keep_payload_message_with_identity(self):
        message = f"{CODEX_IDENTITY} OpenAI Codex API error: overloaded."
        for event_type in ("proof_retry_scheduled", "proof_retry_started"):
            self.assertEqual(
                proofs_route._manual_aggregator_proof_event_message(
                    event_type,
                    {"message": message, "route_identity": CODEX_IDENTITY},
                ),
                message,
            )

    def test_retry_events_prefix_identity_once_when_message_lacks_it(self):
        self.assertEqual(
            proofs_route._manual_aggregator_proof_event_message(
                "proof_retry_started",
                {"retry_attempt": 2, "route_identity": CODEX_IDENTITY},
            ),
            f"{CODEX_IDENTITY} Proof retry 2 started",
        )

    def test_legacy_events_without_identity_are_unchanged(self):
        self.assertEqual(
            proofs_route._manual_aggregator_proof_event_message(
                "proof_run_provider_paused",
                {"round_index": 3},
            ),
            "Round 3 paused for provider credits",
        )
        self.assertEqual(
            proofs_route._manual_aggregator_proof_event_message(
                "proof_run_provider_resumed",
                {"round_index": 3},
            ),
            "Round 3 resumed after the provider pause",
        )


if __name__ == "__main__":
    unittest.main()
