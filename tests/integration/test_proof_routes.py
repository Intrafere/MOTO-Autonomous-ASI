from datetime import datetime
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import proofs as proofs_route
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.models import ProofCandidate, ProofRecord


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
        manual_db.count_proofs.return_value = {"total": 0, "novel": 0, "known": 0}

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            response = self.client.get("/api/proofs?scope=manual")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scope"], "manual")
        manual_db.get_all_proofs.assert_awaited_once_with()

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
