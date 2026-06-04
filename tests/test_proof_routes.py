from datetime import datetime
from unittest import TestCase, mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import proofs as proofs_route
from backend.shared.models import ProofRecord


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
            return_value=[
                {
                    "proof_id": "proof_history",
                    "session_id": "manual_proofs_2026-01-01_00-00-00",
                    "novel": True,
                }
            ]
        )

        with mock.patch.object(proofs_route, "manual_proof_database", manual_db):
            response = self.client.get("/api/proofs/library?scope=manual")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["scope"], "manual")
        self.assertEqual(payload["counts"]["listed"], 1)
        manual_db.list_proof_library_from_history.assert_awaited_once()

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
