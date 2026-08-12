from datetime import datetime
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import proofs as proofs_route
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import (
    ProofCandidate,
    ProofCheckRequest,
    ProofRecord,
    ProofRunCollectionResponse,
    ProofRunSnapshot,
    ProofRunSourceLookupResponse,
)


class ProofBuild01ModelTests(TestCase):
    def test_legacy_proof_check_defaults_to_one_round(self) -> None:
        request = ProofCheckRequest(source_type="paper", source_id="paper_001")
        self.assertEqual(request.run_mode, "one_round")

    def test_continuous_mode_is_accepted_and_status_exposes_build05_fields(self) -> None:
        request = ProofCheckRequest(
            source_type="paper",
            source_id="paper_001",
            run_mode="loop_with_pruning",
        )
        snapshot = ProofRunSnapshot(
            proof_run_id="proof-run-loop",
            run_mode=request.run_mode,
            scope="manual",
            source_type="paper",
            source_id="paper_001",
            proof_store_id="manual:active",
            run_id="manual-run",
            lifecycle_generation=2,
            status="provider_paused",
            round_limit=None,
            unbounded=True,
            current_round=3,
            idle_reason="provider_credit_pause",
            idle_policy="provider_reset",
            provider_state={"provider": "openrouter"},
            pruning_status="idle",
            terminal_reason="",
        )
        payload = snapshot.model_dump(mode="json")
        for field in (
            "current_round",
            "idle_reason",
            "provider_state",
            "pruning_status",
            "terminal_reason",
            "round_limit",
            "unbounded",
        ):
            self.assertIn(field, payload)

    def test_legacy_proof_record_defaults_to_active_live_context(self) -> None:
        record = ProofRecord(
            proof_id="proof_001",
            theorem_statement="True",
            source_type="paper",
            source_id="paper_001",
            lean_code="theorem build01_true : True := by trivial",
        )
        self.assertEqual(record.live_context_status, "active")
        self.assertEqual(record.live_context_owner_run_id, "")


class ProofBuild01PersistenceTests(IsolatedAsyncioTestCase):
    async def test_prune_preserves_lean_and_syntheticlib_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            stored = await database.add_proof_occurrence(
                ProofRecord(
                    proof_id="proof_001",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper_001",
                    run_id="owning-run",
                    lean_code="theorem build01_prune_true : True := by trivial",
                    novel=True,
                    novelty_tier="mathematical_discovery",
                    canonical_theorem_statement_hash="theorem-hash",
                    canonical_lean_code_hash="lean-hash",
                )
            )
            lean_before = await database.get_lean_code(stored.proof_id)
            counts_before = database.count_proofs()
            revision = await database.get_proof_set_revision()
            updated, next_revision = await database.set_live_context_status(
                proof_id=stored.proof_id,
                status="pruned",
                expected_run_id="owning-run",
                expected_proof_set_revision=revision,
                actor="user",
                reason="Redundant in this live run only.",
                expected_theorem_hash="theorem-hash",
                expected_lean_hash="lean-hash",
            )

            self.assertEqual(updated.live_context_status, "pruned")
            self.assertGreater(next_revision, revision)
            self.assertEqual(await database.get_lean_code(stored.proof_id), lean_before)
            counts_after = database.count_proofs()
            self.assertEqual(
                counts_after["syntheticlib_novel"],
                counts_before["syntheticlib_novel"],
            )
            self.assertEqual(counts_after["live_context_pruned"], 1)

    async def test_live_context_mutation_rolls_back_record_and_revision_on_index_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database = ProofDatabase()
            database.set_base_dir(Path(tmpdir))
            await database.initialize()
            stored = await database.add_proof_occurrence(
                ProofRecord(
                    proof_id="proof_rollback",
                    theorem_statement="True",
                    source_type="paper",
                    source_id="paper_rollback",
                    run_id="owning-run",
                    lean_code="theorem rollback_true : True := by trivial",
                    novel=True,
                    novelty_tier="mathematical_discovery",
                )
            )
            revision = await database.get_proof_set_revision()

            with mock.patch.object(
                database,
                "_save_index",
                mock.AsyncMock(side_effect=OSError("index publication failed")),
            ):
                with self.assertRaises(OSError):
                    await database.set_live_context_status(
                        proof_id=stored.proof_id,
                        status="pruned",
                        expected_run_id="owning-run",
                        expected_proof_set_revision=revision,
                        actor="user",
                        reason="Attempted mutation.",
                    )

            reloaded = ProofDatabase()
            reloaded.set_base_dir(Path(tmpdir))
            await reloaded.initialize()
            restored = await reloaded.get_proof(stored.proof_id)
            self.assertEqual(restored.live_context_status, "active")
            self.assertEqual(await reloaded.get_proof_set_revision(), revision)


class ManualProofScopeRouteTests(TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(proofs_route.router)
        self.client = TestClient(app)
        self._lean_enabled = proofs_route.system_config.lean4_enabled
        proofs_route.system_config.lean4_enabled = False

    def tearDown(self) -> None:
        proofs_route.system_config.lean4_enabled = self._lean_enabled

    def test_current_manual_proof_listing_uses_manual_database(self) -> None:
        manual_db = mock.Mock()
        manual_db.get_all_proofs = mock.AsyncMock(return_value=[])
        manual_db.get_proof_set_revision = mock.AsyncMock(return_value=7)
        manual_db.count_proofs.return_value = {"total": 0, "novel": 0, "known": 0}

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            response = self.client.get("/api/proofs?scope=manual")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scope"], "manual")
        self.assertEqual(response.json()["proof_set_revision"], 7)
        self.assertEqual(response.headers["cache-control"], "no-store")
        manual_db.get_all_proofs.assert_awaited_once_with()

    def test_openapi_exposes_process_local_run_discovery_without_resume_controls(self) -> None:
        schema = self.client.get("/openapi.json").json()
        paths = schema["paths"]
        self.assertNotIn("/api/proofs/runs/{proof_run_id}/resume", paths)
        self.assertNotIn("/api/proofs/runs/{proof_run_id}/next-round", paths)
        self.assertEqual(
            paths["/api/proofs"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "#/components/schemas/CurrentProofListResponse",
        )
        self.assertEqual(
            paths["/api/proofs/runs"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "#/components/schemas/ProofRunCollectionResponse",
        )
        self.assertEqual(
            paths["/api/proofs/runs/by-source"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "#/components/schemas/ProofRunSourceLookupResponse",
        )
        mutation = paths["/api/proofs/{proof_id}/live-context"]["patch"]
        self.assertEqual(
            mutation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/ProofLiveContextMutationRequest",
        )
        self.assertEqual(
            mutation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/ProofLiveContextMutationResponse",
        )
        scope_parameter = next(
            parameter
            for parameter in mutation["parameters"]
            if parameter["name"] == "scope"
        )
        self.assertEqual(
            set(scope_parameter["schema"]["enum"]),
            {"autonomous", "manual"},
        )

    def test_proof_run_collection_and_source_lookup_are_no_store(self) -> None:
        collection = ProofRunCollectionResponse(runs=[], count=0, limit=20)
        lookup = ProofRunSourceLookupResponse(
            runs=[],
            count=0,
            limit=20,
            scope="manual",
            source_type="paper",
            source_id="paper_001",
        )
        with (
            mock.patch.object(
                proofs_route.proof_run_manager,
                "list_runs",
                mock.AsyncMock(return_value=collection),
            ),
            mock.patch.object(
                proofs_route.proof_run_manager,
                "find_by_source",
                mock.AsyncMock(return_value=lookup),
            ),
        ):
            collection_response = self.client.get("/api/proofs/runs?scope=manual")
            lookup_response = self.client.get(
                "/api/proofs/runs/by-source",
                params={
                    "scope": "manual",
                    "source_type": "paper",
                    "source_id": "paper_001",
                },
            )

        self.assertEqual(collection_response.status_code, 200)
        self.assertEqual(collection_response.headers["cache-control"], "no-store")
        self.assertEqual(lookup_response.status_code, 200)
        self.assertEqual(lookup_response.headers["cache-control"], "no-store")

    def test_proof_run_source_lookup_rejects_path_components(self) -> None:
        response = self.client.get(
            "/api/proofs/runs/by-source",
            params={
                "scope": "manual",
                "source_type": "paper",
                "source_id": "../paper_001",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_proof_run_id_routes_reject_path_components(self) -> None:
        get_response = self.client.get("/api/proofs/runs/bad%5Cid")
        stop_response = self.client.post(
            "/api/proofs/runs/bad%5Cid/stop",
            json={"expected_lifecycle_generation": 1},
        )

        self.assertEqual(get_response.status_code, 400)
        self.assertEqual(stop_response.status_code, 400)

    def test_proof_run_source_lookup_accepts_history_paper_composite_ids(self) -> None:
        lookup = ProofRunSourceLookupResponse(
            runs=[],
            count=0,
            limit=20,
            scope="autonomous",
            source_type="paper",
            source_id="session_2026-08-06_10-00:paper_001",
        )
        with mock.patch.object(
            proofs_route.proof_run_manager,
            "find_by_source",
            mock.AsyncMock(return_value=lookup),
        ) as find_by_source:
            response = self.client.get(
                "/api/proofs/runs/by-source",
                params={
                    "scope": "autonomous",
                    "source_type": "paper",
                    "source_id": "session_2026-08-06_10-00:paper_001",
                },
            )

        self.assertEqual(response.status_code, 200)
        find_by_source.assert_awaited_once()
        self.assertEqual(
            find_by_source.await_args.kwargs["source_id"],
            "session_2026-08-06_10-00:paper_001",
        )

    def test_manual_proof_library_uses_archived_history_only(self) -> None:
        manual_db = mock.Mock()
        manual_db.list_proof_library_from_history = mock.AsyncMock(
            side_effect=[
                [
                    {
                        "proof_id": "proof_history",
                        "session_id": "manual_proofs_2026-01-01_00-00-00",
                        "novel": True,
                    }
                ],
                [
                    {
                        "proof_id": "proof_history",
                        "session_id": "manual_proofs_2026-01-01_00-00-00",
                        "novel": True,
                    }
                ],
            ]
        )

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            response = self.client.get("/api/proofs/library?scope=manual")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["scope"], "manual")
        self.assertEqual(payload["counts"]["listed"], 1)
        self.assertEqual(manual_db.list_proof_library_from_history.await_count, 2)

    def test_proof_library_category_filter_routes_to_database(self) -> None:
        proof_db = mock.Mock()
        proof_db.list_proof_library = mock.AsyncMock(
            side_effect=[
                [
                    {
                        "proof_id": "proof_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "mathematical_discovery",
                    },
                    {
                        "proof_id": "proof_duplicate_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "duplicate_novel",
                    },
                    {
                        "proof_id": "proof_known",
                        "session_id": "session_a",
                        "novel": False,
                        "novelty_tier": "not_novel",
                    },
                ],
                [
                    {
                        "proof_id": "proof_duplicate_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "duplicate_novel",
                    }
                ],
            ]
        )

        with mock.patch.object(proofs_route, "proof_database", proof_db):
            response = self.client.get("/api/proofs/library?category=duplicate_novel")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["category"], "duplicate_novel")
        self.assertEqual(payload["counts"]["total"], 3)
        self.assertEqual(payload["counts"]["listed"], 1)
        self.assertEqual(payload["counts"]["novel"], 1)
        self.assertEqual(payload["counts"]["duplicate_novel"], 1)
        self.assertEqual(payload["counts"]["not_novel"], 1)
        self.assertEqual(
            proof_db.list_proof_library.await_args_list,
            [
                mock.call(novel_only=None, category="all"),
                mock.call(novel_only=None, category="duplicate_novel"),
            ],
        )

    def test_proof_library_category_counts_are_global_for_filtered_tabs(self) -> None:
        proof_db = mock.Mock()
        proof_db.list_proof_library = mock.AsyncMock(
            side_effect=[
                [
                    {
                        "proof_id": "proof_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "mathematical_discovery",
                    },
                    {
                        "proof_id": "proof_duplicate_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "duplicate_novel",
                    },
                ],
                [
                    {
                        "proof_id": "proof_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "mathematical_discovery",
                    }
                ],
            ]
        )

        with mock.patch.object(proofs_route, "proof_database", proof_db):
            response = self.client.get("/api/proofs/library?category=novel")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["counts"]["listed"], 1)
        self.assertEqual(payload["counts"]["novel"], 1)
        self.assertEqual(payload["counts"]["duplicate_novel"], 1)
        self.assertEqual(payload["counts"]["total"], 2)

    def test_proof_library_novel_count_uses_strict_novelty_tiers(self) -> None:
        proof_db = mock.Mock()
        proof_db.list_proof_library = mock.AsyncMock(
            side_effect=[
                [
                    {
                        "proof_id": "proof_legacy_true_not_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "not_novel",
                    },
                    {
                        "proof_id": "proof_unknown_true",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "novel",
                    },
                    {
                        "proof_id": "proof_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "novel_variant",
                    },
                ],
                [
                    {
                        "proof_id": "proof_novel",
                        "session_id": "session_a",
                        "novel": True,
                        "novelty_tier": "novel_variant",
                    },
                ],
            ]
        )

        with mock.patch.object(proofs_route, "proof_database", proof_db):
            response = self.client.get("/api/proofs/library?category=novel")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["counts"]["total"], 3)
        self.assertEqual(payload["counts"]["listed"], 1)
        self.assertEqual(payload["counts"]["novel"], 1)

    def test_proof_library_category_filter_routes_to_manual_database(self) -> None:
        manual_db = mock.Mock()
        manual_db.list_proof_library_from_history = mock.AsyncMock(
            side_effect=[
                [
                    {
                        "proof_id": "proof_novel",
                        "session_id": "manual_session",
                        "novel": True,
                        "novelty_tier": "mathematical_discovery",
                    },
                    {
                        "proof_id": "proof_duplicate_novel",
                        "session_id": "manual_session",
                        "novel": True,
                        "novelty_tier": "duplicate_novel",
                    },
                ],
                [
                    {
                        "proof_id": "proof_duplicate_novel",
                        "session_id": "manual_session",
                        "novel": True,
                        "novelty_tier": "duplicate_novel",
                    }
                ],
            ]
        )

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            response = self.client.get("/api/proofs/library?scope=manual&category=duplicate_novel")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["scope"], "manual")
        self.assertEqual(payload["counts"]["total"], 2)
        self.assertEqual(payload["counts"]["listed"], 1)
        self.assertEqual(payload["counts"]["duplicate_novel"], 1)
        self.assertEqual(
            manual_db.list_proof_library_from_history.await_args_list,
            [
                mock.call(
                    proofs_route._manual_proof_history_root(),
                    novel_only=None,
                    category="all",
                ),
                mock.call(
                    proofs_route._manual_proof_history_root(),
                    novel_only=None,
                    category="duplicate_novel",
                ),
            ],
        )

    def test_manual_certificate_routes_use_manual_database(self) -> None:
        proof = ProofRecord(
            proof_id="manual_proof_1",
            theorem_statement="Manual theorem statement.",
            theorem_name="manual_theorem",
            source_type="brainstorm",
            source_id="manual_aggregator",
            source_title="Manual Aggregator",
            lean_code="theorem manual_theorem : True := by trivial",
            solver="Lean 4",
            created_at=datetime(2026, 1, 1),
            novel=True,
            novelty_reasoning="Manual proof route test.",
            attempt_count=1,
        )
        manual_db = mock.Mock()
        manual_db.get_proof = mock.AsyncMock(return_value=proof)
        manual_db.get_lean_code = mock.AsyncMock(return_value=proof.lean_code)

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            json_response = self.client.get("/api/proofs/manual_proof_1/certificate?scope=manual")
            lean_response = self.client.get("/api/proofs/manual_proof_1/certificate.lean?scope=manual")

        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(json_response.json()["proof_id"], "manual_proof_1")
        self.assertEqual(lean_response.status_code, 200)
        self.assertIn("theorem manual_theorem", lean_response.text)
        self.assertEqual(manual_db.get_proof.await_count, 2)
        self.assertEqual(manual_db.get_lean_code.await_count, 2)

    def test_archived_certificate_normalizes_legacy_mapping_at_response_boundary(self) -> None:
        legacy_payload = {
            "proof_id": "legacy_archived_proof",
            "theorem_statement": "Legacy archived theorem.",
            "source_type": "paper",
            "source_id": "legacy_paper",
            "source_title": "Legacy Paper",
            "lean_code": "theorem legacy_archived_theorem : True := by trivial",
            "novel": True,
            "novelty_reasoning": "Legacy novelty reasoning.",
            "created_at": "2025-01-02T03:04:05",
            "solver_hints": None,
            "dependencies": None,
        }
        proof_db = mock.Mock()
        proof_db.get_library_proof = mock.AsyncMock(return_value=legacy_payload)

        with mock.patch.object(proofs_route, "proof_database", proof_db):
            response = self.client.get(
                "/api/proofs/library/legacy_session/legacy_archived_proof/certificate"
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["proof_id"], "legacy_archived_proof")
        self.assertEqual(payload["run_id"], "legacy:paper:legacy_paper")
        self.assertEqual(payload["user_prompt"], "Legacy Paper")
        self.assertEqual(payload["novelty_tier"], "novel_formulation")
        self.assertEqual(payload["independent_novelty_tier"], "novel_formulation")
        self.assertEqual(
            payload["independent_novelty_reasoning"],
            "Legacy novelty reasoning.",
        )
        self.assertEqual(payload["solver_hints"], [])
        self.assertEqual(payload["dependencies"], [])


class ManualAggregatorProofEventLogTests(IsolatedAsyncioTestCase):
    def test_empty_discovery_messages_explain_model_search_without_changing_counts(self) -> None:
        no_candidates = proofs_route._manual_aggregator_proof_event_message(
            "proof_check_no_candidates",
            {},
        )
        round_complete = proofs_route._manual_aggregator_proof_event_message(
            "proof_run_round_complete",
            {
                "round_index": 3,
                "candidate_count": 0,
                "run_mode": "loop_with_pruning",
                "next_round_automatic": True,
            },
        )
        completion = proofs_route._manual_aggregator_proof_event_message(
            "proof_check_complete",
            {"verified_count": 0, "novel_count": 0},
        )

        self.assertIn("model searched for useful novel proof candidates", no_candidates)
        self.assertIn("no Lean proof attempts were needed", no_candidates)
        self.assertIn("model searched for useful novel proof candidates", round_complete)
        self.assertIn("next round will start automatically", round_complete.lower())
        self.assertEqual(completion, "Proof check complete: 0 verified, 0 novel")

    async def test_manual_aggregator_proof_event_is_broadcast_and_persisted_with_same_id(self) -> None:
        with (
            mock.patch.object(proofs_route.websocket, "broadcast_event", new=mock.AsyncMock()) as broadcast,
            mock.patch.object(proofs_route.event_log, "add_event", new=mock.AsyncMock()) as add_event,
        ):
            await proofs_route._broadcast_manual_aggregator_proof_event(
                "proof_check_complete",
                {
                    "source_type": "brainstorm",
                    "source_id": "manual_aggregator",
                    "verified_count": 1,
                    "novel_count": 0,
                },
            )

        broadcast.assert_awaited_once()
        add_event.assert_awaited_once()
        _, broadcast_payload = broadcast.await_args.args
        _, message, persisted_payload = add_event.await_args.args

        self.assertEqual(message, "Proof check complete: 1 verified, 0 novel")
        self.assertEqual(
            broadcast_payload["manual_event_id"],
            persisted_payload["manual_event_id"],
        )
        self.assertEqual(persisted_payload["source_id"], "manual_aggregator")


class ProofSourceAdapterTests(IsolatedAsyncioTestCase):
    async def test_history_paper_reads_writes_and_stores_in_original_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sessions_root = root / "auto_sessions"
            history_session = sessions_root / "older_session"
            history_papers = history_session / "papers"
            history_papers.mkdir(parents=True)
            history_paper_path = history_papers / "paper_paper_001.txt"
            history_paper_path.write_text("Historical paper body.", encoding="utf-8")
            active_proofs = root / "active_proofs"
            active_proofs.mkdir()

            history_payload = {
                "content": "Historical paper body.",
                "title": "Historical Paper",
                "user_prompt": "Historical canonical prompt",
                "source_brainstorm_ids": [],
            }
            request = ProofCheckRequest(
                source_type="paper",
                source_id="older_session:paper_001",
            )

            with (
                mock.patch.object(
                    proofs_route.system_config,
                    "auto_sessions_base_dir",
                    str(sessions_root),
                ),
                mock.patch.object(
                    proofs_route.paper_library,
                    "get_history_paper",
                    new=mock.AsyncMock(return_value=history_payload),
                ),
                mock.patch.object(
                    proofs_route.paper_library,
                    "get_history_papers_dir",
                    return_value=history_papers,
                ),
                mock.patch.object(
                    proofs_route,
                    "proof_database",
                    mock.Mock(_base_dir=active_proofs),
                ),
            ):
                adapter = await proofs_route._resolve_proof_source_adapter(request)

            self.assertEqual(adapter.canonical_user_prompt, "Historical canonical prompt")
            self.assertEqual(adapter.proof_database._base_dir, history_session / "proofs")
            self.assertIn("Historical paper body.", adapter.source_content)
            self.assertTrue(adapter.proof_store_id.startswith("autonomous:session:older_session:"))
            self.assertNotEqual(adapter.proof_store_id, "autonomous:active")
            self.assertTrue(adapter.writable)
            self.assertFalse(adapter.append_to_source)
            self.assertIsNotNone(adapter.append_proof_callback)

            proof = ProofRecord(
                proof_id="proof_history_write",
                theorem_statement="True",
                theorem_name="history_write",
                source_type="paper",
                source_id=request.source_id,
                source_title="Historical Paper",
                lean_code="theorem history_write : True := by trivial",
                novel=True,
                novelty_tier="novel_variant",
            )
            await adapter.proof_database.add_proof_occurrence(proof)
            appended = await adapter.append_proof_callback(proof)

            self.assertTrue(appended)
            self.assertTrue((history_session / "proofs" / "proof_proof_history_write.json").exists())
            self.assertIn("history_write", history_paper_path.read_text(encoding="utf-8"))
            self.assertFalse(any(active_proofs.iterdir()))

    async def test_manual_and_current_autonomous_sources_keep_existing_stores(self) -> None:
        cases = (
            (
                ProofCheckRequest(
                    source_type="brainstorm",
                    source_id=proofs_route.MANUAL_AGGREGATOR_SOURCE_ID,
                ),
                proofs_route.manual_proof_database,
                proofs_route.PROOF_SCOPE_MANUAL,
                "manual:active",
                False,
            ),
            (
                ProofCheckRequest(source_type="paper", source_id="paper_current"),
                proofs_route.proof_database,
                proofs_route.PROOF_SCOPE_AUTONOMOUS,
                "autonomous:active",
                True,
            ),
        )
        for request, database, scope, store_id, append_to_source in cases:
            with (
                self.subTest(source_id=request.source_id),
                mock.patch.object(
                    proofs_route,
                    "_resolve_manual_source",
                    new=mock.AsyncMock(
                        return_value=("source content", "source title", "prompt with proofs")
                    ),
                ),
                mock.patch.object(
                    proofs_route.research_metadata,
                    "get_user_prompt",
                    new=mock.AsyncMock(return_value="canonical prompt"),
                ),
                mock.patch.object(
                    proofs_route,
                    "_manual_aggregator_prompt",
                    new=mock.AsyncMock(return_value="manual canonical prompt"),
                ),
            ):
                adapter = await proofs_route._resolve_proof_source_adapter(request)

            self.assertIs(adapter.proof_database, database)
            self.assertEqual(adapter.scope, scope)
            self.assertEqual(adapter.proof_store_id, store_id)
            self.assertEqual(adapter.append_to_source, append_to_source)


class ProofDatabaseCleanupTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tempdir.name) / "proofs"
        self.history_root = Path(self._tempdir.name) / "manual_proof_runs"
        self.db = ProofDatabase()
        self.db.set_base_dir(self.base_dir)
        await self.db.initialize()

    async def asyncTearDown(self) -> None:
        self._tempdir.cleanup()

    def _proof_record(self) -> ProofRecord:
        return ProofRecord(
            proof_id="proof_001",
            theorem_statement="Cleanup theorem statement.",
            theorem_name="cleanup_theorem",
            source_type="brainstorm",
            source_id="topic_cleanup",
            source_title="Cleanup Topic",
            lean_code="theorem cleanup_theorem : True := by trivial",
            solver="Lean 4",
            created_at=datetime(2026, 1, 1),
            novel=True,
            novelty_reasoning="Cleanup regression proof.",
            attempt_count=1,
        )

    def _candidate(self) -> ProofCandidate:
        return ProofCandidate(
            theorem_id="failed_cleanup_candidate",
            statement="Failed cleanup candidate statement.",
            formal_sketch="Try proving by contradiction.",
            expected_novelty_tier="mathematical_discovery",
            prompt_relevance_rationale="Directly relevant to cleanup regression.",
            novelty_rationale="Not a standard known result in this test.",
            why_not_standard_known_result="Synthetic test target.",
            source_excerpt="Failed source excerpt.",
        )

    async def test_clear_failed_candidates_preserves_verified_proof_files(self) -> None:
        await self.db.add_proof(self._proof_record())
        await self.db.record_failed_candidate(
            "topic_cleanup",
            self._candidate(),
            "Lean failed before cleanup.",
        )

        self.assertTrue((self.base_dir / "proof_proof_001.json").exists())
        self.assertTrue((self.base_dir / "proof_proof_001_lean.lean").exists())
        self.assertTrue((self.base_dir / "failed" / "topic_cleanup.json").exists())

        await self.db.clear_failed_candidates()

        self.assertTrue((self.base_dir / "proof_proof_001.json").exists())
        self.assertTrue((self.base_dir / "proof_proof_001_lean.lean").exists())
        self.assertFalse((self.base_dir / "failed" / "topic_cleanup.json").exists())
        self.assertTrue((self.base_dir / "failed").exists())

    async def test_archive_current_run_does_not_archive_failed_retry_hints(self) -> None:
        await self.db.add_proof(self._proof_record())
        await self.db.record_failed_candidate(
            "topic_cleanup",
            self._candidate(),
            "Lean failed before archive.",
        )

        metadata = await self.db.archive_current_run(
            self.history_root,
            user_prompt="Manual prompt.",
            reason="cleanup_test",
        )

        self.assertIsNotNone(metadata)
        archived_proofs = self.history_root / metadata["session_id"] / "proofs"
        self.assertTrue((archived_proofs / "proof_proof_001.json").exists())
        self.assertTrue((archived_proofs / "proof_proof_001_lean.lean").exists())
        self.assertFalse((archived_proofs / "failed").exists())
        self.assertTrue((self.base_dir / "failed").exists())
        self.assertFalse((self.base_dir / "failed" / "topic_cleanup.json").exists())
