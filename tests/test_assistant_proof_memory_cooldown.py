import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.shared.config import system_config
from backend.shared.proof_search.assistant_cache import (
    AssistantCooldownState,
    AssistantRankCache,
)
from backend.shared.proof_search.assistant_coordinator import (
    AssistantProofSearchCoordinator,
    _advance_success_state,
    _advance_zero_useful_state,
    _consume_cooldown_turn,
    _support_signature,
)
from backend.shared.proof_search.assistant_models import (
    AssistantProofPack,
    AssistantProofSupport,
    AssistantTargetSnapshot,
)
from backend.shared.proof_search.models import ProofSearchRequest, UnifiedProofSearchRecord


def _snapshot() -> AssistantTargetSnapshot:
    return AssistantTargetSnapshot(
        workflow_mode="autonomous",
        target_kind="proof_candidate",
        workflow_phase="brainstorm_proof_verification",
        user_prompt="Prove a useful theorem.",
        target_statement="theorem target : True",
        source_type="brainstorm",
        source_id="topic_001",
        target_hash="target_hash",
    )


def _support(search_id: str) -> AssistantProofSupport:
    return AssistantProofSupport(
        search_id=search_id,
        corpus="manual",
        corpus_scope="history",
        proof_id=search_id.replace(":", "_"),
        theorem_name=f"Memory.{search_id.replace(':', '_')}",
        theorem_statement=f"theorem {search_id.replace(':', '_')} : True",
        imports=["Mathlib"],
    )


def _pack(*search_ids: str, selection_mode: str = "assistant_llm") -> AssistantProofPack:
    return AssistantProofPack(
        workflow_mode="autonomous",
        target_kind="proof_candidate",
        target_hash="target_hash",
        results=[_support(search_id) for search_id in search_ids],
        selection_mode=selection_mode,
    )


def _record(index: int) -> UnifiedProofSearchRecord:
    return UnifiedProofSearchRecord(
        search_id=f"manual:proof_{index}",
        corpus="manual",
        corpus_scope="history",
        source_kind="verified_proof",
        proof_id=f"proof_{index}",
        source_title=f"Proof Source {index}",
        display_title=f"Helper {index}",
        theorem_name=f"Helper{index}",
        theorem_statement=f"theorem helper_{index} : True",
        lean_code="import Mathlib\n\ntheorem helper : True := by\n  trivial\n",
        lean_code_hash=f"code_hash_{index}",
        theorem_statement_hash=f"stmt_hash_{index}",
        imports=["Mathlib"],
        verified=True,
        canonical_uri=f"moto://proofs/proof_{index}",
    )


class _EmptyProofSearchService:
    async def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: list[str] | None = None,
        exclude_session_ids: list[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        return []


class _FailingProofSearchService:
    async def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: list[str] | None = None,
        exclude_session_ids: list[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        raise RuntimeError("search index unavailable")


class _NoCandidateProofSearchService:
    def __init__(self) -> None:
        self.requests: list[ProofSearchRequest] = []

    async def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: list[str] | None = None,
        exclude_session_ids: list[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        self.requests.append(request)
        return [_record(1)]


class AssistantCooldownStateMachineTests(unittest.TestCase):
    def test_support_signature_is_stable_and_order_sensitive(self) -> None:
        first = _support_signature(_pack("manual:proof_1", "manual:proof_2"))
        second = _support_signature(_pack("manual:proof_1", "manual:proof_2"))
        reversed_order = _support_signature(_pack("manual:proof_2", "manual:proof_1"))

        self.assertEqual(first, second)
        self.assertNotEqual(first, reversed_order)
        self.assertEqual(_support_signature(_pack()), "")

    def test_oauth_cooldown_selection_modes_are_valid_pack_modes(self) -> None:
        cached_pack = _pack("manual:proof_cached", selection_mode="cached_oauth_cooldown")
        deterministic_pack = _pack("manual:proof_deterministic", selection_mode="deterministic_oauth_cooldown")

        self.assertEqual(cached_pack.selection_mode, "cached_oauth_cooldown")
        self.assertEqual(deterministic_pack.selection_mode, "deterministic_oauth_cooldown")

    def test_zero_useful_escalates_to_shutdown_after_steady_81_batches(self) -> None:
        snapshot = _snapshot()
        state = AssistantCooldownState.empty("autonomous:brainstorm:topic_001")

        expected_windows = [(1, 3), (2, 9), (3, 81), (3, 81), (3, 81)]
        for expected_stage, expected_skips in expected_windows:
            for _ in range(4):
                state, event_name, payload = _advance_zero_useful_state(snapshot, state)
            self.assertEqual(event_name, "assistant_proof_memory_cooldown")
            self.assertIsNotNone(payload)
            self.assertEqual(state.zero_cooldown_stage, expected_stage)
            self.assertEqual(state.zero_cooldown_skips_remaining, expected_skips)
            self.assertFalse(state.zero_shutdown_active)
            while state.zero_cooldown_skips_remaining:
                state, skip_payload = _consume_cooldown_turn(snapshot, state)
                self.assertIsNotNone(skip_payload)

        for _ in range(4):
            state, event_name, payload = _advance_zero_useful_state(snapshot, state)

        self.assertEqual(event_name, "assistant_proof_memory_shutdown")
        self.assertIsNotNone(payload)
        self.assertTrue(state.zero_shutdown_active)
        self.assertEqual(state.zero_steady_81_batches, 3)
        self.assertEqual(state.zero_cooldown_skips_remaining, 0)

    def test_useful_pack_resets_zero_useful_state(self) -> None:
        snapshot = _snapshot()
        state = AssistantCooldownState(
            run_key="autonomous:brainstorm:topic_001",
            zero_attempts_in_batch=3,
            zero_cooldown_stage=2,
            zero_cooldown_skips_remaining=9,
            zero_steady_81_batches=1,
            zero_shutdown_active=True,
        )

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))

        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.zero_attempts_in_batch, 0)
        self.assertEqual(state.zero_cooldown_stage, 0)
        self.assertEqual(state.zero_cooldown_skips_remaining, 0)
        self.assertEqual(state.zero_steady_81_batches, 0)
        self.assertFalse(state.zero_shutdown_active)

    def test_stagnant_pack_enters_cooldown_and_changed_signature_resets(self) -> None:
        snapshot = _snapshot()
        state = AssistantCooldownState.empty("autonomous:brainstorm:topic_001")

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.stagnant_same_count, 1)

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.stagnant_same_count, 2)

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        self.assertEqual(event_name, "assistant_proof_memory_cooldown")
        self.assertIsNotNone(payload)
        self.assertEqual(state.stagnant_cooldown_stage, 1)
        self.assertEqual(state.stagnant_cooldown_skips_remaining, 3)
        self.assertFalse(state.zero_shutdown_active)

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_2"))
        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.stagnant_same_count, 1)
        self.assertEqual(state.stagnant_attempts_in_batch, 1)
        self.assertEqual(state.stagnant_cooldown_stage, 0)
        self.assertEqual(state.stagnant_cooldown_skips_remaining, 0)

    def test_zero_useful_breaks_stagnant_consecutive_chain(self) -> None:
        snapshot = _snapshot()
        state = AssistantCooldownState.empty("autonomous:brainstorm:topic_001")

        state, _, _ = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        state, _, _ = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        self.assertEqual(state.stagnant_same_count, 2)

        state, event_name, payload = _advance_zero_useful_state(snapshot, state)
        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.stagnant_same_count, 0)
        self.assertEqual(state.stagnant_attempts_in_batch, 0)
        self.assertEqual(state.last_signature, "")

        state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
        self.assertEqual(event_name, "")
        self.assertIsNone(payload)
        self.assertEqual(state.stagnant_same_count, 1)

    def test_stagnant_cooldown_reaches_max_stage_without_shutdown(self) -> None:
        snapshot = _snapshot()
        state = AssistantCooldownState.empty("autonomous:brainstorm:topic_001")

        for expected_stage in [1, 2, 3, 3]:
            for _ in range(3):
                state, event_name, payload = _advance_success_state(snapshot, state, _pack("manual:proof_1"))
            self.assertEqual(event_name, "assistant_proof_memory_cooldown")
            self.assertIsNotNone(payload)
            self.assertEqual(state.stagnant_cooldown_stage, expected_stage)
            self.assertEqual(state.stagnant_cooldown_skips_remaining, [3, 9, 81][expected_stage - 1])
            self.assertFalse(state.zero_shutdown_active)
            while state.stagnant_cooldown_skips_remaining:
                state, skip_payload = _consume_cooldown_turn(snapshot, state)
                self.assertIsNotNone(skip_payload)


class AssistantCooldownCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_memory_enabled = system_config.agent_conversation_memory_enabled
        self._old_syntheticlib4_enabled = system_config.syntheticlib4_enabled
        system_config.agent_conversation_memory_enabled = True
        system_config.syntheticlib4_enabled = False

    async def asyncTearDown(self) -> None:
        system_config.agent_conversation_memory_enabled = self._old_memory_enabled
        system_config.syntheticlib4_enabled = self._old_syntheticlib4_enabled

    async def test_selection_mode_unavailable_does_not_advance_zero_useful(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite")
            coordinator = AssistantProofSearchCoordinator(
                service=_NoCandidateProofSearchService(),
                cache=cache,
            )
            snapshot = _snapshot()
            run_key = coordinator._run_key_for_snapshot(snapshot)

            await coordinator._record_cooldown_outcome(
                snapshot,
                _pack(selection_mode="unavailable"),
            )

            state = cache.load_cooldown_state(run_key)
            self.assertEqual(state.zero_attempts_in_batch, 0)
            self.assertEqual(state.zero_cooldown_stage, 0)
            self.assertFalse(state.zero_shutdown_active)

    async def test_run_key_groups_transient_compiler_task_ids_across_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = AssistantProofSearchCoordinator(
                service=_NoCandidateProofSearchService(),
                cache=AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite"),
            )
            first = AssistantTargetSnapshot(
                workflow_mode="compiler",
                target_kind="outline_context",
                workflow_phase="outline",
                source_type="comp_writer",
                source_id="comp_writer_001",
                target_hash="first",
            )
            later = first.model_copy(update={"source_id": "comp_writer_009", "target_hash": "later"})
            rigor = first.model_copy(
                update={
                    "source_type": "comp_hp",
                    "source_id": "comp_hp_002",
                    "target_hash": "rigor",
                }
            )

            self.assertEqual(
                coordinator._run_key_for_snapshot(first),
                coordinator._run_key_for_snapshot(later),
            )
            self.assertEqual(
                coordinator._run_key_for_snapshot(first),
                coordinator._run_key_for_snapshot(rigor),
            )

    async def test_run_key_keeps_real_source_ids_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = AssistantProofSearchCoordinator(
                service=_NoCandidateProofSearchService(),
                cache=AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite"),
            )
            first = _snapshot()
            second = first.model_copy(update={"source_id": "topic_002"})

            self.assertNotEqual(
                coordinator._run_key_for_snapshot(first),
                coordinator._run_key_for_snapshot(second),
            )

    async def test_cooldown_state_round_trips_and_clear_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite")
            coordinator = AssistantProofSearchCoordinator(
                service=_EmptyProofSearchService(),
                cache=cache,
            )
            state = AssistantCooldownState(
                run_key="autonomous:brainstorm:topic_001",
                zero_attempts_in_batch=4,
                zero_cooldown_stage=2,
                zero_cooldown_skips_remaining=9,
                last_reason="fixture",
            )
            cache.save_cooldown_state(state)

            await coordinator.stop_all(clear_packs=True)
            preserved = cache.load_cooldown_state(state.run_key)
            self.assertEqual(preserved.zero_cooldown_stage, 2)
            self.assertEqual(preserved.zero_cooldown_skips_remaining, 9)

            await coordinator.clear_cooldown_state(state.run_key)
            cleared = cache.load_cooldown_state(state.run_key)
            self.assertEqual(cleared.zero_cooldown_stage, 0)
            self.assertEqual(cleared.zero_cooldown_skips_remaining, 0)

    async def test_true_empty_external_history_emits_unavailable_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite")
            coordinator = AssistantProofSearchCoordinator(
                service=_EmptyProofSearchService(),
                cache=cache,
            )
            events: list[tuple[str, dict]] = []

            async def _capture(event_type: str, payload: dict) -> None:
                events.append((event_type, payload))

            with mock.patch("backend.api.routes.websocket.broadcast_event", new=_capture):
                await coordinator.refresh_now(_snapshot())

            self.assertEqual([event for event, _ in events], ["assistant_proof_memory_unavailable"])
            self.assertIn("only performs proof-memory retrieval for now", events[0][1]["reason"])

    async def test_search_failure_does_not_claim_no_external_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite")
            coordinator = AssistantProofSearchCoordinator(
                service=_FailingProofSearchService(),
                cache=cache,
            )
            events: list[tuple[str, dict]] = []

            async def _capture(event_type: str, payload: dict) -> None:
                events.append((event_type, payload))

            with mock.patch("backend.api.routes.websocket.broadcast_event", new=_capture):
                await coordinator.refresh_now(_snapshot())

            self.assertEqual([event for event, _ in events], ["assistant_proof_pack_warning"])
            self.assertIn("search failed", events[0][1]["reason"].lower())

    async def test_compiler_zero_useful_batches_accumulate_across_transient_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AssistantRankCache(Path(temp_dir) / "assistant_ranker.sqlite")

            async def _empty_selector(*args) -> tuple[list[str], str]:
                return [], "none useful"

            coordinator = AssistantProofSearchCoordinator(
                service=_NoCandidateProofSearchService(),
                cache=cache,
                assistant_selector=_empty_selector,
            )
            events: list[tuple[str, dict]] = []

            async def _capture(event_type: str, payload: dict) -> None:
                events.append((event_type, payload))

            with mock.patch("backend.api.routes.websocket.broadcast_event", new=_capture):
                for index in range(1, 5):
                    snapshot = AssistantTargetSnapshot(
                        workflow_mode="compiler",
                        target_kind="outline_context",
                        workflow_phase="outline",
                        source_type="compiler_writer",
                        source_id=f"comp_writer_{index:03d}",
                        user_prompt=f"Compiler prompt {index}",
                        target_statement=f"theorem target_{index} : True",
                    )
                    await coordinator.refresh_now(snapshot)

                state = cache.load_cooldown_state("compiler:workflow:compiler")
                self.assertEqual(state.zero_cooldown_stage, 1)
                self.assertEqual(state.zero_cooldown_skips_remaining, 3)
                self.assertIn(
                    "assistant_proof_memory_cooldown",
                    [event for event, _ in events],
                )

                skipped_snapshot = AssistantTargetSnapshot(
                    workflow_mode="compiler",
                    target_kind="outline_context",
                    workflow_phase="outline",
                    source_type="compiler_writer",
                    source_id="comp_writer_005",
                    user_prompt="Compiler prompt skipped",
                    target_statement="theorem target_skipped : True",
                )
                before_requests = len(coordinator._service.requests)
                await coordinator.refresh_now(skipped_snapshot)

            self.assertEqual(len(coordinator._service.requests), before_requests)
            skipped_state = cache.load_cooldown_state("compiler:workflow:compiler")
            self.assertEqual(skipped_state.zero_cooldown_skips_remaining, 2)


if __name__ == "__main__":
    unittest.main()
