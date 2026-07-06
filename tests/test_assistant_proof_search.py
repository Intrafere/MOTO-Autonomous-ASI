import asyncio
from contextlib import contextmanager
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.proof_search.assistant_cache import AssistantRankCache
from backend.shared.proof_search.assistant_coordinator import AssistantProofSearchCoordinator
from backend.shared.proof_search.assistant_models import (
    AssistantProofPack,
    AssistantProofSupport,
    AssistantTargetSnapshot,
)
from backend.shared.proof_search.assistant_ranker import rank_assistant_proof_candidates
from backend.shared.proof_search.indexer import ProofSearchIndexer
from backend.shared.proof_search.models import ProofSearchRequest, ProofSearchResponse, UnifiedProofSearchRecord
from backend.shared.models import ModelConfig
from backend.autonomous.memory.session_manager import session_manager


def _record(
    index: int,
    *,
    search_id: str | None = None,
    fingerprint: str | None = None,
    corpus: str = "moto",
    corpus_scope: str = "history",
    session_id: str = "",
    source_kind: str = "verified_proof",
    verified: bool = True,
    lean_code: str | None = None,
) -> UnifiedProofSearchRecord:
    statement = f"theorem helper_{index} : True"
    code = lean_code if lean_code is not None else f"import Mathlib\n\ntheorem helper_{index} : True := by\n  trivial\n"
    return UnifiedProofSearchRecord(
        search_id=search_id or f"{corpus}:helper_{index}",
        corpus=corpus,
        corpus_scope=corpus_scope,
        source_kind=source_kind,
        proof_id=f"proof_{index}",
        external_fingerprint=fingerprint or f"fp_{index}",
        session_id=session_id,
        source_title=f"Proof Source {index}",
        display_title=f"Helper {index}",
        theorem_name=f"Assistant.Helper{index}",
        theorem_statement=statement,
        informal_statement="A true helper theorem.",
        proof_description="Uses trivial to prove True.",
        formal_sketch="Apply trivial.",
        lean_code=code,
        lean_code_hash=f"code_hash_{index}",
        theorem_statement_hash=f"stmt_hash_{index}",
        imports=["Mathlib"],
        dependency_names=["True.intro", f"Dep.{index}"],
        topic_tags=["logic"],
        domain_tags=["test"],
        module="Assistant.Test",
        source_path=f"Assistant/Test{index}.lean",
        novelty_tier="novel_formulation",
        novelty_reasoning="Fixture proof.",
        verified=verified,
        created_at=f"2026-06-12T00:00:{index:02d}+00:00",
        canonical_uri=f"moto://proofs/proof_{index}",
    )


class _FakeProofSearchService:
    def __init__(self, records: list[UnifiedProofSearchRecord]) -> None:
        self.records = records
        self.requests: list[ProofSearchRequest] = []
        self.candidate_pool_kwargs: list[dict[str, object]] = []

    async def search(self, request: ProofSearchRequest) -> ProofSearchResponse:
        self.requests.append(request)
        return ProofSearchResponse(
            results=self.records,
            result_count=len(self.records),
            searched_corpora=request.corpora,
        )

    async def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: list[str] | None = None,
        exclude_session_ids: list[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        self.requests.append(request)
        self.candidate_pool_kwargs.append(
            {
                "exclude_corpus_scopes": list(exclude_corpus_scopes or []),
                "exclude_session_ids": list(exclude_session_ids or []),
            }
        )
        excluded_scopes = set(exclude_corpus_scopes or [])
        excluded_sessions = set(exclude_session_ids or [])
        return [
            record
            for record in self.records
            if record.corpus_scope not in excluded_scopes
            and record.session_id not in excluded_sessions
        ][:pool_limit]


async def _fake_assistant_selector(
    snapshot: AssistantTargetSnapshot,
    shortlist: list[AssistantProofSupport],
    assistant_role_id: str,
    assistant_model_id: str,
    task_id: str,
) -> tuple[list[str], str]:
    return (
        [support.search_id for support in shortlist[:7]],
        f"Fixture Assistant selection for {snapshot.workflow_mode}:{snapshot.target_kind}.",
    )


def _response_json(payload: str) -> dict:
    return {"choices": [{"message": {"content": payload}}]}


def _preserve_role_config(role_id: str):
    configs = api_client_manager._role_model_configs
    old_present = role_id in configs
    old_config = configs.get(role_id)

    def restore() -> None:
        if old_present:
            configs[role_id] = old_config
        else:
            configs.pop(role_id, None)

    return restore


@contextmanager
def _assistant_test_environment(
    *,
    memory_enabled: bool = True,
    syntheticlib4_enabled: bool = False,
    session_id: str | None = None,
):
    old_data_dir = system_config.data_dir
    old_memory_enabled = system_config.agent_conversation_memory_enabled
    old_syntheticlib4_enabled = system_config.syntheticlib4_enabled
    old_session_path = session_manager._session_path
    old_session_id = session_manager._session_id
    old_user_prompt = session_manager._user_prompt
    old_base_dir = session_manager._base_dir

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            system_config.data_dir = str(temp_path / "data")
            system_config.agent_conversation_memory_enabled = memory_enabled
            system_config.syntheticlib4_enabled = syntheticlib4_enabled

            if session_id is not None:
                session_manager._base_dir = temp_path / "auto_sessions"
                session_manager._session_id = session_id
                session_manager._session_path = session_manager._base_dir / session_id
                session_manager._user_prompt = "Active prompt"
                session_manager._session_path.mkdir(parents=True)

            yield temp_path
    finally:
        system_config.data_dir = old_data_dir
        system_config.agent_conversation_memory_enabled = old_memory_enabled
        system_config.syntheticlib4_enabled = old_syntheticlib4_enabled
        session_manager._base_dir = old_base_dir
        session_manager._session_path = old_session_path
        session_manager._session_id = old_session_id
        session_manager._user_prompt = old_user_prompt


class AssistantProofRankerTests(unittest.TestCase):
    def test_ranker_caps_filters_and_dedupes_verified_supports(self) -> None:
        target = AssistantTargetSnapshot(
            workflow_mode="autonomous",
            target_kind="proof_candidate",
            user_prompt="Prove a true helper theorem.",
            target_statement="theorem target : True",
            dependency_names=["True.intro"],
            imports=["Mathlib"],
        )
        records = [_record(index) for index in range(10)]
        records.append(_record(100, search_id="moto:duplicate", fingerprint="fp_0"))
        records.append(_record(101, source_kind="partial_proof"))
        records.append(_record(102, verified=False))

        supports = rank_assistant_proof_candidates(records, target, limit=7)

        self.assertLessEqual(len(supports), 7)
        self.assertEqual(sum(support.fingerprint == "fp_0" for support in supports), 1)
        self.assertTrue(all(support.source_kind == "verified_proof" for support in supports))
        self.assertTrue(all(support.theorem_statement for support in supports))

    def test_internal_candidate_pool_exceeds_public_search_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild([_record(index, corpus="manual") for index in range(12)])
            request = ProofSearchRequest(
                query="helper",
                corpora=["manual"],
                verified_only=True,
                limit=12,
                hydrate_lean_code=False,
            )

            public_results = indexer.search(request).results
            assistant_pool = indexer.search_candidate_pool(request, pool_limit=12)

            self.assertEqual(len(public_results), 7)
            self.assertGreater(len(assistant_pool), len(public_results))
            self.assertEqual(len(assistant_pool), 12)

    def test_internal_candidate_pool_excludes_scopes_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            active_records = [
                _record(index, corpus="moto", corpus_scope="current")
                for index in range(80)
            ]
            history_records = [
                _record(100, search_id="moto:old_session:proof_100", session_id="old_session"),
                _record(101, search_id="syntheticlib4:proof_101", corpus="syntheticlib4"),
            ]
            indexer.rebuild([*active_records, *history_records])

            assistant_pool = indexer.search_candidate_pool(
                ProofSearchRequest(
                    query="helper",
                    corpora=["moto", "syntheticlib4"],
                    verified_only=True,
                    limit=64,
                    hydrate_lean_code=False,
                ),
                pool_limit=64,
                exclude_corpus_scopes=["active", "current"],
                exclude_session_ids=[],
            )

            self.assertEqual(
                {record.search_id for record in assistant_pool},
                {"moto:old_session:proof_100", "syntheticlib4:proof_101"},
            )

    def test_internal_candidate_pool_keeps_syntheticlib4_current_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild(
                [
                    _record(1, corpus="moto", corpus_scope="current"),
                    _record(
                        2,
                        search_id="syntheticlib4:proof_2",
                        corpus="syntheticlib4",
                        corpus_scope="current",
                    ),
                ]
            )

            assistant_pool = indexer.search_candidate_pool(
                ProofSearchRequest(
                    query="helper",
                    corpora=["moto", "syntheticlib4"],
                    verified_only=True,
                    limit=64,
                    hydrate_lean_code=False,
                ),
                pool_limit=64,
                exclude_corpus_scopes=["active", "current"],
                exclude_session_ids=[],
            )

            self.assertEqual(
                [record.search_id for record in assistant_pool],
                ["syntheticlib4:proof_2"],
            )


class AssistantProofCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_assistant_selector_retries_missing_selected_search_ids(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="openrouter",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [
                AssistantProofSupport.from_record(_record(1, corpus="manual")),
                AssistantProofSupport.from_record(_record(2, corpus="manual")),
            ]
            responses = [
                _response_json('{"reasoning":"missing selected IDs"}'),
                _response_json(
                    '{"selected_search_ids":["manual:helper_2"],"reasoning":"use helper 2"}'
                ),
            ]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(side_effect=responses),
            ) as generate_completion:
                selected_ids, reasoning = await coordinator._select_with_assistant(
                    snapshot,
                    shortlist,
                    assistant_role_id="manual_proof_assistant",
                    assistant_model_id="assistant-model",
                    task_id="assistant_pack_manual_001",
                )

            self.assertEqual(selected_ids, ["manual:helper_2"])
            self.assertEqual(reasoning, "use helper 2")
            self.assertEqual(generate_completion.await_count, 2)
            first_call = generate_completion.await_args_list[0].kwargs
            retry_call = generate_completion.await_args_list[1].kwargs
            self.assertEqual(first_call["max_tokens"], 2048)
            self.assertEqual(first_call["response_format"], {"type": "json_object"})
            self.assertTrue(first_call["_moto_disable_supercharge"])
            self.assertEqual(first_call["_moto_reasoning_effort_override"], "none")
            self.assertEqual(retry_call["task_id"], "assistant_pack_manual_001_retry")
            self.assertEqual(retry_call["response_format"], {"type": "json_object"})
            self.assertTrue(retry_call["_moto_disable_supercharge"])
            self.assertEqual(retry_call["_moto_reasoning_effort_override"], "none")
            self.assertIn("invalid", retry_call["messages"][0]["content"])
        finally:
            restore()

    async def test_lm_studio_assistant_selector_uses_text_response_format(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="lm_studio",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [AssistantProofSupport.from_record(_record(1, corpus="manual"))]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(
                    return_value=_response_json(
                        '{"selected_search_ids":["manual:helper_1"],"reasoning":"use helper 1"}'
                    )
                ),
            ) as generate_completion:
                selected_ids, reasoning = await coordinator._select_with_assistant(
                    snapshot,
                    shortlist,
                    assistant_role_id="manual_proof_assistant",
                    assistant_model_id="assistant-model",
                    task_id="assistant_pack_manual_001",
                )

            self.assertEqual(selected_ids, ["manual:helper_1"])
            self.assertEqual(reasoning, "use helper 1")
            self.assertEqual(generate_completion.await_args.kwargs["response_format"], {"type": "text"})
        finally:
            restore()

    async def test_assistant_selector_retries_ids_outside_shortlist(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="openrouter",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [
                AssistantProofSupport.from_record(_record(1, corpus="manual")),
                AssistantProofSupport.from_record(_record(2, corpus="manual")),
            ]
            responses = [
                _response_json(
                    '{"selected_search_ids":["manual:not_in_shortlist"],"reasoning":"use a nearby theorem"}'
                ),
                _response_json(
                    '{"selected_search_ids":["manual:helper_1"],"reasoning":"use helper 1"}'
                ),
            ]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(side_effect=responses),
            ) as generate_completion:
                selected_ids, reasoning = await coordinator._select_with_assistant(
                    snapshot,
                    shortlist,
                    assistant_role_id="manual_proof_assistant",
                    assistant_model_id="assistant-model",
                    task_id="assistant_pack_manual_001",
                )

            self.assertEqual(selected_ids, ["manual:helper_1"])
            self.assertEqual(reasoning, "use helper 1")
            self.assertEqual(generate_completion.await_count, 2)
            retry_call = generate_completion.await_args_list[1].kwargs
            self.assertEqual(retry_call["task_id"], "assistant_pack_manual_001_retry")
            self.assertIn("outside the candidate shortlist", retry_call["messages"][0]["content"])
        finally:
            restore()

    async def test_assistant_selector_canonicalizes_unique_bare_proof_ids(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="openrouter",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [
                AssistantProofSupport.from_record(
                    _record(44, search_id="manual:archived_session:proof_44", corpus="manual")
                ),
                AssistantProofSupport.from_record(
                    _record(11, search_id="manual:archived_session:proof_11", corpus="manual")
                ),
            ]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(
                    return_value=_response_json(
                        '{"selected_search_ids":["proof_44","proof_11"],"reasoning":"use residue certificates"}'
                    )
                ),
            ) as generate_completion:
                selected_ids, reasoning = await coordinator._select_with_assistant(
                    snapshot,
                    shortlist,
                    assistant_role_id="manual_proof_assistant",
                    assistant_model_id="assistant-model",
                    task_id="assistant_pack_manual_001",
                )

            self.assertEqual(
                selected_ids,
                ["manual:archived_session:proof_44", "manual:archived_session:proof_11"],
            )
            self.assertEqual(reasoning, "use residue certificates")
            self.assertEqual(generate_completion.await_count, 1)
            prompt = generate_completion.await_args.kwargs["messages"][0]["content"]
            self.assertIn("SELECT_ID: manual:archived_session:proof_44", prompt)
            self.assertIn("proof_id: proof_44", prompt)
            self.assertIn("do not return proof_id/display IDs", prompt)
        finally:
            restore()

    async def test_assistant_selector_valid_empty_selection_is_not_retried(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="openrouter",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [AssistantProofSupport.from_record(_record(1, corpus="manual"))]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(
                    return_value=_response_json(
                        '{"selected_search_ids":[],"reasoning":"no listed proof is useful"}'
                    )
                ),
            ) as generate_completion:
                selected_ids, reasoning = await coordinator._select_with_assistant(
                    snapshot,
                    shortlist,
                    assistant_role_id="manual_proof_assistant",
                    assistant_model_id="assistant-model",
                    task_id="assistant_pack_manual_001",
                )

            self.assertEqual(selected_ids, [])
            self.assertEqual(reasoning, "no listed proof is useful")
            self.assertEqual(generate_completion.await_count, 1)
        finally:
            restore()

    async def test_assistant_selector_failure_publishes_unavailable_pack(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            with _assistant_test_environment():
                api_client_manager.configure_role(
                    "manual_proof_assistant",
                    ModelConfig(
                        provider="openrouter",
                        model_id="assistant-model",
                        context_window=4096,
                        max_output_tokens=2048,
                    ),
                )
                coordinator = AssistantProofSearchCoordinator(
                    service=_FakeProofSearchService(
                        [_record(index, corpus="manual") for index in range(1, 4)]
                    )
                )
                snapshot = AssistantTargetSnapshot(
                    workflow_mode="manual_proof_check",
                    target_kind="proof_candidate",
                    user_prompt="Prove the manual target.",
                    target_statement="theorem target : True",
                )

                with mock.patch.object(
                    api_client_manager,
                    "generate_completion",
                    new=mock.AsyncMock(
                        return_value=_response_json('{"reasoning":"still missing selected IDs"}')
                    ),
                ) as generate_completion:
                    pack = await coordinator.refresh_now(snapshot)

                self.assertIsNotNone(pack)
                self.assertEqual(pack.selection_mode, "unavailable")
                self.assertEqual(pack.results, [])
                self.assertEqual(pack.assistant_role_id, "manual_proof_assistant")
                self.assertEqual(pack.assistant_model_id, "assistant-model")
                self.assertEqual(generate_completion.await_count, 2)
                self.assertTrue(
                    any("Assistant LLM selection failed" in warning for warning in pack.warnings)
                )
        finally:
            restore()

    async def test_assistant_selector_does_not_repair_retry_provider_errors(self) -> None:
        restore = _preserve_role_config("manual_proof_assistant")
        try:
            api_client_manager.configure_role(
                "manual_proof_assistant",
                ModelConfig(
                    provider="openrouter",
                    model_id="assistant-model",
                    context_window=4096,
                    max_output_tokens=2048,
                ),
            )
            coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
            )
            shortlist = [AssistantProofSupport.from_record(_record(1, corpus="manual"))]

            with mock.patch.object(
                api_client_manager,
                "generate_completion",
                new=mock.AsyncMock(side_effect=RuntimeError("provider unavailable")),
            ) as generate_completion:
                with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                    await coordinator._select_with_assistant(
                        snapshot,
                        shortlist,
                        assistant_role_id="manual_proof_assistant",
                        assistant_model_id="assistant-model",
                        task_id="assistant_pack_manual_001",
                    )

            self.assertEqual(generate_completion.await_count, 1)
        finally:
            restore()

    async def test_refresh_persists_metadata_only_status_and_stop_clears_pack(self) -> None:
        with _assistant_test_environment():
            service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(9)])
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the manual target.",
                target_statement="theorem target : True",
                dependency_names=["True.intro"],
            )

            pack = await coordinator.refresh_now(snapshot)
            status = coordinator.get_status()
            persisted_path = Path(system_config.data_dir) / "proof_search" / "assistant_latest_pack.json"
            persisted = json.loads(persisted_path.read_text(encoding="utf-8"))

            self.assertIsNotNone(pack)
            self.assertLessEqual(len(pack.results), 7)
            self.assertEqual(pack.selection_mode, "assistant_llm")
            self.assertEqual(pack.assistant_role_id, "manual_proof_assistant")
            self.assertEqual(pack.assistant_model_id, "injected-assistant")
            self.assertEqual(status["latest_target_hash"], pack.target_hash)
            self.assertEqual(status["latest_result_count"], len(pack.results))
            self.assertTrue(service.requests)
            self.assertTrue(persisted["results"][0]["has_hydrated_code"])
            self.assertEqual(persisted["results"][0]["lean_code"], "")
            self.assertEqual(persisted["selection_mode"], "assistant_llm")

            await coordinator.stop_all(clear_packs=True)

            self.assertIsNone(coordinator.get_latest_pack())
            self.assertFalse(persisted_path.exists())
            self.assertEqual(coordinator.get_status()["cached_pack_count"], 0)

    async def test_refresh_excludes_active_run_candidates_before_assistant_selection(self) -> None:
        with _assistant_test_environment(syntheticlib4_enabled=True, session_id="active_session"):
            service = _FakeProofSearchService(
                [
                    *[
                        _record(index, corpus="moto", corpus_scope="current")
                        for index in range(80)
                    ],
                    _record(200, corpus="manual", corpus_scope="active"),
                    _record(
                        201,
                        search_id="moto:active_session:proof_201",
                        corpus="moto",
                        corpus_scope="history",
                        session_id="active_session",
                    ),
                    _record(
                        202,
                        search_id="moto:previous_session:proof_202",
                        corpus="moto",
                        corpus_scope="history",
                        session_id="previous_session",
                    ),
                    _record(203, search_id="syntheticlib4:proof_203", corpus="syntheticlib4"),
                ]
            )
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the target.",
                target_statement="theorem target : True",
                dependency_names=["True.intro"],
            )

            pack = await coordinator.refresh_now(snapshot)

            selected_ids = {support.search_id for support in pack.results}
            self.assertEqual(
                selected_ids,
                {"moto:previous_session:proof_202", "syntheticlib4:proof_203"},
            )
            self.assertTrue(service.candidate_pool_kwargs)
            self.assertIn(
                "current",
                service.candidate_pool_kwargs[0]["exclude_corpus_scopes"],
            )
            self.assertIn(
                "active_session",
                service.candidate_pool_kwargs[0]["exclude_session_ids"],
            )

    async def test_cached_pack_drops_current_run_supports_before_injection(self) -> None:
        coordinator = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
        current_pack = AssistantProofPack(
            workflow_mode="manual_proof_check",
            target_kind="proof_candidate",
            target_hash="target",
            results=[
                AssistantProofSupport.from_record(
                    _record(1, corpus="manual", corpus_scope="active")
                ),
                AssistantProofSupport.from_record(
                    _record(2, corpus="manual", corpus_scope="archived")
                ),
            ],
        )
        coordinator._packs["target"] = current_pack

        filtered_pack = coordinator.get_latest_pack("target")

        self.assertIsNotNone(filtered_pack)
        self.assertEqual(
            [support.search_id for support in filtered_pack.results],
            ["manual:helper_2"],
        )

    async def test_persistent_visit_counts_expand_repeated_pack_selection(self) -> None:
        with _assistant_test_environment():
            service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(8)])
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the target.",
                target_statement="theorem target : True",
                dependency_names=["True.intro"],
            )

            first_pack = await coordinator.refresh_now(snapshot)
            second_pack = await coordinator.refresh_now(snapshot)
            first_ids = {support.search_id for support in first_pack.results}
            second_ids = {support.search_id for support in second_pack.results}
            cache = AssistantRankCache()
            stats = cache.load_candidate_stats(first_pack.target_hash)

            self.assertEqual(len(first_ids), 7)
            self.assertEqual(len(second_ids), 7)
            self.assertTrue(second_ids.difference(first_ids))
            self.assertGreaterEqual(
                sum(item.visits for item in stats.values()),
                len(first_ids) + len(second_ids),
            )

    async def test_submit_target_reuses_cached_pack_without_background_refresh(self) -> None:
        with _assistant_test_environment():
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the cached target.",
                target_statement="theorem cached_target : True",
            )
            first = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(3)]),
                assistant_selector=_fake_assistant_selector,
            )
            initial_pack = await first.refresh_now(snapshot)

            second = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            target_hash = second.submit_target(snapshot)
            cached_pack = second.get_latest_pack(target_hash)

            self.assertIsNotNone(initial_pack)
            self.assertIsNotNone(cached_pack)
            self.assertEqual(cached_pack.freshness, "cached")
            self.assertEqual(
                [support.search_id for support in cached_pack.results],
                [support.search_id for support in initial_pack.results],
            )
            self.assertFalse(any(not task.done() for task in second._tasks.values()))
            await second.stop_all(clear_packs=True)

    async def test_useful_pack_refreshes_only_after_two_receiver_reads(self) -> None:
        with _assistant_test_environment():
            first_snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the first target.",
                target_statement="theorem first_target : True",
            )
            second_snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the second target.",
                target_statement="theorem second_target : True",
            )
            coordinator = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(3)]),
                assistant_selector=_fake_assistant_selector,
            )

            first_pack = await coordinator.refresh_now(first_snapshot)
            self.assertIsNotNone(first_pack)
            self.assertEqual(len(first_pack.results), 3)

            first_hash = first_snapshot.stable_hash()
            coordinator.mark_pack_consumed_by_solver(
                first_hash,
                role_id="manual_proof_submitter",
                task_id="proof_form_001",
            )
            deferred_hash = coordinator.submit_target(second_snapshot)
            deferred_pack = coordinator.get_latest_pack(deferred_hash)

            self.assertEqual(deferred_pack.selection_mode, "stale-but-best-known")
            self.assertFalse(any(not task.done() for task in coordinator._tasks.values()))

            coordinator.mark_pack_consumed_by_solver(
                deferred_hash,
                role_id="manual_proof_submitter",
                task_id="proof_form_002",
            )
            refreshed_hash = coordinator.submit_target(second_snapshot)
            running_task = coordinator._tasks.get(refreshed_hash)
            self.assertIsNotNone(running_task)
            await asyncio.gather(running_task)
            refreshed_pack = coordinator.get_latest_pack(refreshed_hash)

            self.assertEqual(refreshed_pack.selection_mode, "assistant_llm")
            self.assertEqual(refreshed_pack.target_hash, second_snapshot.stable_hash())
            await coordinator.stop_all(clear_packs=True)

    async def test_exact_cached_pack_keeps_receiver_read_count_across_same_target_reuse(self) -> None:
        with _assistant_test_environment():
            snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="proof_candidate",
                user_prompt="Prove the cached target.",
                target_statement="theorem cached_target : True",
            )
            coordinator = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(3)]),
                assistant_selector=_fake_assistant_selector,
            )

            pack = await coordinator.refresh_now(snapshot)
            target_hash = pack.target_hash
            request_count = len(coordinator._service.requests)

            coordinator.mark_pack_consumed_by_solver(
                target_hash,
                role_id="manual_proof_submitter",
                task_id="proof_form_001",
            )
            same_target_hash = coordinator.submit_target(snapshot)
            self.assertEqual(same_target_hash, target_hash)
            self.assertFalse(coordinator._latest_pack_has_enough_receiver_reads())
            self.assertEqual(len(coordinator._service.requests), request_count)

            coordinator.mark_pack_consumed_by_solver(
                same_target_hash,
                role_id="manual_proof_submitter",
                task_id="proof_form_002",
            )
            self.assertTrue(coordinator._latest_pack_has_enough_receiver_reads())
            await coordinator.stop_all(clear_packs=True)

    async def test_submit_target_reruns_after_cached_unavailable_empty_pack(self) -> None:
        with _assistant_test_environment():
            snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="completion_review_context",
                workflow_phase="outline",
                user_prompt="Review whether the brainstorm is complete.",
                target_statement="Find proof supports for completion review.",
            )

            async def failing_selector(
                snapshot: AssistantTargetSnapshot,
                shortlist: list[AssistantProofSupport],
                assistant_role_id: str,
                assistant_model_id: str,
                task_id: str,
            ) -> tuple[list[str], str]:
                raise RuntimeError("LM Studio rejected response_format")

            first = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(1, 4)]),
                assistant_selector=failing_selector,
            )
            failed_pack = await first.refresh_now(snapshot)
            self.assertEqual(failed_pack.selection_mode, "unavailable")
            self.assertEqual(failed_pack.results, [])
            self.assertGreater(failed_pack.candidate_count, 0)

            second_service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(1, 4)])
            second = AssistantProofSearchCoordinator(
                service=second_service,
                assistant_selector=_fake_assistant_selector,
            )
            target_hash = second.submit_target(snapshot)

            self.assertIn(target_hash, second._tasks)
            await second._tasks[target_hash]
            recovered_pack = second.get_latest_pack(target_hash)

            self.assertIsNotNone(recovered_pack)
            self.assertEqual(recovered_pack.selection_mode, "assistant_llm")
            self.assertGreater(len(recovered_pack.results), 0)
            self.assertTrue(second_service.requests)
            await second.stop_all(clear_packs=True)

    async def test_submit_target_does_not_rerun_assistant_for_same_cached_target_after_consumption(self) -> None:
        with _assistant_test_environment():
            service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(8)])
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="brainstorm_context",
                workflow_phase="brainstorm",
                user_prompt="Explore a stable brainstorm target.",
                current_prompt_or_topic="Find reusable proof patterns.",
                source_type="autonomous_brainstorm_submitters",
                source_id="shared_brainstorm_pack",
            )

            target_hash = coordinator.submit_target(snapshot)
            await coordinator._tasks[target_hash]
            await asyncio.sleep(0)
            request_count = len(service.requests)

            coordinator.mark_pack_consumed_by_solver(
                target_hash,
                role_id="aggregator_submitter_1",
                task_id="agg_sub1_000",
            )
            same_target_hash = coordinator.submit_target(snapshot)

            self.assertEqual(same_target_hash, target_hash)
            self.assertEqual(len(service.requests), request_count)
            self.assertFalse(any(not task.done() for task in coordinator._tasks.values()))
            await coordinator.stop_all(clear_packs=True)

    async def test_cached_pack_refreshes_after_consumption_when_broad_target_state_changes(self) -> None:
        with _assistant_test_environment():
            service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(8)])
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            first_snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="brainstorm_context",
                workflow_phase="brainstorm",
                user_prompt="Explore a stable brainstorm target.",
                current_prompt_or_topic="Find reusable proof patterns.",
                accepted_memory_summary="Submission 1: initial brainstorm state.",
                source_type="autonomous_brainstorm_submitters",
                source_id="shared_brainstorm_pack",
            )

            first_target_hash = coordinator.submit_target(first_snapshot)
            await coordinator._tasks[first_target_hash]
            await asyncio.sleep(0)
            request_count = len(service.requests)

            coordinator.mark_pack_consumed_by_solver(
                first_target_hash,
                role_id="aggregator_submitter_1",
                task_id="agg_sub1_000",
            )

            changed_snapshot = first_snapshot.model_copy(
                update={
                    "accepted_memory_summary": "Submission 1: initial brainstorm state. Submission 2: newly accepted progress.",
                }
            )

            second_target_hash = coordinator.submit_target(changed_snapshot)
            deferred_pack = coordinator.get_latest_pack(second_target_hash)

            self.assertIsNotNone(deferred_pack)
            self.assertEqual(deferred_pack.selection_mode, "stale-but-best-known")
            self.assertFalse(any(not task.done() for task in coordinator._tasks.values()))
            self.assertEqual(len(service.requests), request_count)

            coordinator.mark_pack_consumed_by_solver(
                second_target_hash,
                role_id="aggregator_submitter_2",
                task_id="agg_sub2_000",
            )
            second_target_hash = coordinator.submit_target(changed_snapshot)
            self.assertIn(second_target_hash, coordinator._tasks)
            await coordinator._tasks[second_target_hash]
            self.assertGreater(len(service.requests), request_count)
            self.assertNotEqual(first_target_hash, second_target_hash)
            await coordinator.stop_all(clear_packs=True)

    async def test_submit_target_defers_refresh_until_solver_consumes_pack(self) -> None:
        with _assistant_test_environment():
            selector_started = asyncio.Event()
            release_selector = asyncio.Event()

            async def blocking_selector(
                snapshot: AssistantTargetSnapshot,
                shortlist: list[AssistantProofSupport],
                assistant_role_id: str,
                assistant_model_id: str,
                task_id: str,
            ) -> tuple[list[str], str]:
                selector_started.set()
                await release_selector.wait()
                return [support.search_id for support in shortlist[:7]], "selected"

            coordinator = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(8)]),
                assistant_selector=blocking_selector,
            )
            first_snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the target.",
                target_statement="theorem target : True",
                source_id="proof_form_001",
            )
            second_snapshot = first_snapshot.model_copy(
                update={
                    "rejection_feedback": "New solver feedback after the same theorem target.",
                    "source_id": "proof_form_002",
                }
            )

            first_target = coordinator.submit_target(first_snapshot)
            await selector_started.wait()
            second_target = coordinator.submit_target(second_snapshot)

            self.assertIn(first_target, coordinator._tasks)
            self.assertNotIn(second_target, coordinator._tasks)

            release_selector.set()
            await coordinator._tasks[first_target]

            deferred_target = coordinator.submit_target(second_snapshot)
            deferred_pack = coordinator.get_latest_pack(deferred_target)

            self.assertIsNotNone(deferred_pack)
            self.assertFalse(any(not task.done() for task in coordinator._tasks.values()))

            coordinator.mark_pack_consumed_by_solver(
                deferred_target,
                role_id="autonomous_proof_formalization",
                task_id="proof_form_002",
            )
            next_target = coordinator.submit_target(second_snapshot)
            self.assertNotIn(next_target, coordinator._tasks)

            coordinator.mark_pack_consumed_by_solver(
                deferred_target,
                role_id="autonomous_proof_formalization",
                task_id="proof_form_003",
            )
            next_target = coordinator.submit_target(second_snapshot)

            self.assertIn(next_target, coordinator._tasks)
            await coordinator.stop_all(clear_packs=True)

    async def test_stale_pack_consumption_does_not_unblock_latest_refresh(self) -> None:
        with _assistant_test_environment():
            service = _FakeProofSearchService([_record(index, corpus="manual") for index in range(8)])
            coordinator = AssistantProofSearchCoordinator(
                service=service,
                assistant_selector=_fake_assistant_selector,
            )
            first_snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the first target.",
                target_statement="theorem first_target : True",
                source_id="proof_form_001",
            )
            second_snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the second target.",
                target_statement="theorem second_target : True",
                source_id="proof_form_002",
            )

            first_pack = await coordinator.refresh_now(first_snapshot)
            first_hash = first_pack.target_hash
            coordinator.mark_pack_consumed_by_solver(
                first_hash,
                role_id="autonomous_proof_formalization",
                task_id="proof_form_001",
            )
            coordinator.mark_pack_consumed_by_solver(
                first_hash,
                role_id="autonomous_proof_formalization",
                task_id="proof_form_001_retry",
            )
            service.records = [_record(index + 100, corpus="manual") for index in range(8)]
            second_hash = coordinator.submit_target(second_snapshot)
            await coordinator._tasks[second_hash]

            self.assertEqual(coordinator._latest_pack_target_hash, second_hash)
            self.assertFalse(coordinator._latest_pack_has_enough_receiver_reads())

            coordinator.mark_pack_consumed_by_solver(
                first_hash,
                role_id="autonomous_proof_formalization",
                task_id="proof_form_001",
            )

            self.assertFalse(coordinator._latest_pack_has_enough_receiver_reads())
            await coordinator.stop_all(clear_packs=True)

    async def test_goal_cache_reuses_pack_across_prompt_feedback_and_position_changes(self) -> None:
        with _assistant_test_environment():
            original_snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="lean_error",
                user_prompt="Original prompt.",
                target_statement="theorem cached_target : True",
                lean_error="stdin:1:2: line 1, column 2: unsolved goals",
            )
            first = AssistantProofSearchCoordinator(
                service=_FakeProofSearchService([_record(index, corpus="manual") for index in range(3)]),
                assistant_selector=_fake_assistant_selector,
            )
            initial_pack = await first.refresh_now(original_snapshot)

            changed_snapshot = AssistantTargetSnapshot(
                workflow_mode="manual_proof_check",
                target_kind="lean_error",
                user_prompt="Different surrounding prompt.",
                target_statement="theorem cached_target : True",
                lean_error="stdin:9:20: line 9, column 20: unsolved goals",
                rejection_feedback="New feedback should not defeat exact goal-cache reuse.",
            )
            in_memory_target_hash = first.submit_target(changed_snapshot)
            in_memory_pack = first.get_latest_pack(in_memory_target_hash)

            self.assertIsNotNone(in_memory_pack)
            self.assertEqual(in_memory_pack.freshness, "stale-but-best-known")
            self.assertTrue(in_memory_pack.results[0].lean_code.strip())
            await first.stop_all(clear_packs=False)

            second = AssistantProofSearchCoordinator(service=_FakeProofSearchService([]))
            target_hash = second.submit_target(changed_snapshot)
            cached_pack = second.get_latest_pack(target_hash)

            self.assertIsNotNone(initial_pack)
            self.assertIsNotNone(cached_pack)
            self.assertNotEqual(initial_pack.target_hash, target_hash)
            self.assertEqual(cached_pack.target_hash, target_hash)
            self.assertEqual(cached_pack.freshness, "stale-but-best-known")
            self.assertEqual(
                [support.search_id for support in cached_pack.results],
                [support.search_id for support in initial_pack.results],
            )
            await second.stop_all(clear_packs=True)

    async def test_assistant_cache_retention_prunes_old_targets(self) -> None:
        old_data_dir = system_config.data_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                system_config.data_dir = str(Path(temp_dir) / "data")
                cache = AssistantRankCache()
                for index in range(130):
                    snapshot = AssistantTargetSnapshot(
                        workflow_mode="autonomous",
                        target_kind="proof_candidate",
                        target_statement=f"theorem target_{index} : True",
                    )
                    target_hash = snapshot.stable_hash()
                    cache.upsert_candidates(
                        target_hash=target_hash,
                        candidates=[
                            {
                                "search_id": f"manual:proof_{index}",
                                "proof_source": "manual",
                                "proof_id": f"proof_{index}",
                                "theorem_statement_hash": f"stmt_{index}",
                                "lean_code_hash": f"code_{index}",
                                "query_variant": "",
                                "retrieval_score": 0.5,
                                "exact_match_score": 0.5,
                                "semantic_score": 0.5,
                                "dependency_overlap_score": 0.0,
                                "corpus_trust_score": 0.95,
                                "recency_score": 0.5,
                                "duplicate_group": f"proof_{index}",
                            }
                        ],
                    )
                    cache.record_pack(
                        snapshot=snapshot.model_copy(update={"target_hash": target_hash}),
                        pack=AssistantProofPack(
                            workflow_mode="autonomous",
                            target_kind="proof_candidate",
                            target_hash=target_hash,
                            query_summary=f"target {index}",
                            results=[],
                        ),
                        selected_search_ids=[f"manual:proof_{index}"],
                    )

                db_path = Path(system_config.data_dir) / "proof_search" / "assistant_ranker.sqlite"
                conn = sqlite3.connect(str(db_path))
                try:
                    pack_count = conn.execute(
                        "SELECT COUNT(DISTINCT target_hash) FROM assistant_proof_packs"
                    ).fetchone()[0]
                    candidate_count = conn.execute(
                        "SELECT COUNT(DISTINCT target_hash) FROM assistant_proof_candidates"
                    ).fetchone()[0]
                    goal_count = conn.execute(
                        "SELECT COUNT(*) FROM assistant_goal_cache"
                    ).fetchone()[0]
                finally:
                    conn.close()

                self.assertLessEqual(pack_count, 128)
                self.assertLessEqual(candidate_count, 128)
                self.assertLessEqual(goal_count, 128)
        finally:
            system_config.data_dir = old_data_dir

    async def test_disabled_agent_conversation_memory_does_not_start_assistant(self) -> None:
        old_memory_enabled = system_config.agent_conversation_memory_enabled
        try:
            system_config.agent_conversation_memory_enabled = False
            service = _FakeProofSearchService([_record(1)])
            coordinator = AssistantProofSearchCoordinator(service=service)
            snapshot = AssistantTargetSnapshot(
                workflow_mode="autonomous",
                target_kind="proof_candidate",
                user_prompt="Prove the target.",
                target_statement="theorem target : True",
            )

            target_hash = coordinator.submit_target(snapshot)
            pack = await coordinator.refresh_now(snapshot)
            status = coordinator.get_status()

            self.assertIsNone(pack)
            self.assertIsNone(coordinator.get_latest_pack(target_hash))
            self.assertFalse(service.requests)
            self.assertFalse(status["enabled"])
            self.assertIn("Session History Memory", status["disabled_reason"])
        finally:
            system_config.agent_conversation_memory_enabled = old_memory_enabled


