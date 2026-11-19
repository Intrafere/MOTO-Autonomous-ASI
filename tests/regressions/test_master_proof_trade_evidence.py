"""Verified master proofs must carry the same proof-trade evidence as individual proofs.

A master proof is one Lean file holding its scaffold steps plus the master theorem
last. It is registered as an ordinary canonical occurrence, so the SyntheticLib
evidence auditor must see the same complete receipt/identity/dependency evidence
and the same source selection as for an individual proof. Its scaffold working
state lives beside the store and must never look like a canonical artifact, and
linking an older record to its scaffold state on load must not change trade
evidence (selection snapshots bind that evidence).
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.build_info import BuildInfo
from backend.shared.config import system_config
from backend.shared.installation_identity import InstallationBinding
from backend.shared.lean4_client import Lean4Client
from backend.shared.lean_proof_integrity import extract_primary_lean_theorem
from backend.shared.models import (
    LeanCompileReceipt,
    LocalNoveltyAssessmentProvenance,
    MasterProofStep,
    MasterProofSummary,
    ProofRecord,
)
from backend.shared.proof_identity import canonical_proof_identity
from backend.shared.syntheticlib_trade_eligibility import project_canonical_proof_store

PERMANENT_BLOCKERS = (
    "frozen_syntheticlib_contract_unpublished",
    "novelty_trade_mapping_unavailable",
)

MASTER_STATEMENT = "For every natural number n, (n + 0) * 1 = n."
MASTER_CODE = """import Mathlib

lemma step_add_zero (n : Nat) : n + 0 = n := by
  simp

lemma step_mul_one (n : Nat) : n * 1 = n := by
  simp

theorem master_identity (n : Nat) : (n + 0) * 1 = n := by
  rw [step_add_zero, step_mul_one]
"""
INDIVIDUAL_CODE = "import Mathlib\n\ntheorem individual_fixture : 1 + 1 = 2 := by norm_num\n"

SUMMARY = MasterProofSummary(
    master_key="feedbeef",
    root_theorem_name="master_identity",
    scaffold_revision=1,
    total_attempts=4,
    steps=[
        MasterProofStep(name="step_add_zero", signature="lemma step_add_zero (n : Nat) : n + 0 = n"),
        MasterProofStep(name="step_mul_one", signature="lemma step_mul_one (n : Nat) : n * 1 = n"),
    ],
)


async def _full_proof_receipt(tmpdir: Path, source: str) -> LeanCompileReceipt:
    """Receipt the source exactly as the checker does for a placeholder-free pass."""
    client = Lean4Client(lean_path="lean", workspace_dir=str(tmpdir))
    with mock.patch.object(
        client, "get_version", mock.AsyncMock(return_value="Lean (version 4.0.0)")
    ), mock.patch.object(
        client, "get_mathlib_commit", return_value="a" * 40
    ), mock.patch(
        "backend.shared.lean4_client.get_build_info",
        return_value=BuildInfo(
            version="1.1.10",
            build_commit="fixture_commit",
            update_channel="main",
            api_contract_version="fixture_contract",
        ),
    ):
        return await client._build_compile_receipt(
            original_source=source,
            checked_source=source,
            checker_mode="subprocess",
            timeout=60,
            # The master proof's final whole-file check runs with placeholders off.
            verification_profile="full_proof",
        )


class MasterProofTradeEvidenceTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # A manual archived-run store resolves to the "manual" locator store kind.
        self.store = (
            Path(system_config.data_dir) / "manual_proof_runs" / f"run_{uuid.uuid4().hex}"
        )
        self.addCleanup(shutil.rmtree, self.store, True)
        self.scratch = self.store.parent / f"scratch_{uuid.uuid4().hex}"
        self.scratch.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.scratch, True)
        binding_patch = mock.patch(
            "backend.autonomous.memory.proof_database.ensure_installation_binding",
            return_value=InstallationBinding(
                installation_id="moto_installation_test",
                root_binding_id="moto_root_test",
            ),
        )
        binding_patch.start()
        self.addCleanup(binding_patch.stop)

    async def _database(self) -> ProofDatabase:
        database = ProofDatabase()
        database.set_base_dir(self.store)
        await database.initialize()
        return database

    async def _store(
        self,
        database: ProofDatabase,
        *,
        proof_id: str,
        statement: str,
        name: str,
        code: str,
        master: bool,
    ) -> ProofRecord:
        record = ProofRecord(
            proof_id=proof_id,
            theorem_statement=statement,
            theorem_name=name,
            source_type="paper",
            source_id="manual_paper",
            run_id="run-master-trade",
            lean_code=code,
            novel=True,
            novelty_tier="novel_variant",
            independent_novelty_tier="novel_variant",
            local_novelty_assessment_provenance=LocalNoveltyAssessmentProvenance(
                assessment_origin="validator_assessed",
                source_local_rank="novel_variant",
            ),
            compile_receipt=await _full_proof_receipt(self.scratch, code),
            verification_evidence_status="complete",
            proof_kind="master_proof" if master else "individual",
            master_proof=SUMMARY if master else None,
        )
        stored = await database.add_proof_occurrence(record)
        # The proof stage completes dependency extraction for every registered proof.
        updated = await database.update_proof_dependencies(
            stored.proof_id, [], portable_dependencies=[], extraction_status="complete"
        )
        return updated or stored

    def _diagnostics(self) -> dict:
        projection = project_canonical_proof_store(self.store, store_role="archived_manual")
        return {item.client_proof_id: item for item in projection.diagnostics}

    async def test_master_proof_record_hashes_its_master_theorem(self) -> None:
        name, header = extract_primary_lean_theorem(MASTER_CODE)
        self.assertEqual(name, "master_identity")
        self.assertIn("(n + 0) * 1 = n", header)

    async def test_master_proof_has_the_same_trade_evidence_as_an_individual_proof(self) -> None:
        database = await self._database()
        master = await self._store(
            database,
            proof_id="proof_001",
            statement=MASTER_STATEMENT,
            name="master_identity",
            code=MASTER_CODE,
            master=True,
        )
        individual = await self._store(
            database,
            proof_id="proof_002",
            statement="1 + 1 = 2",
            name="individual_fixture",
            code=INDIVIDUAL_CODE,
            master=False,
        )
        # Scaffold working state sits beside the store but is never a canonical artifact.
        masters = database.get_master_proofs_dir()
        masters.mkdir(parents=True, exist_ok=True)
        (masters / "master_proof_feedbeef.json").write_text(
            json.dumps({"master_key": "feedbeef", "status": "verified", "scaffold_code": MASTER_CODE}),
            encoding="utf-8",
        )
        (masters / "master_proof_feedbeef.lean").write_text(MASTER_CODE, encoding="utf-8")

        self.assertEqual(master.proof_kind, "master_proof")
        self.assertEqual(
            master.canonical_theorem_statement_hash,
            canonical_proof_identity(MASTER_STATEMENT, MASTER_CODE).theorem_statement_hash,
        )
        projection = project_canonical_proof_store(self.store, store_role="archived_manual")
        self.assertTrue(projection.evidence_complete, projection.reason_codes)
        self.assertEqual(len(projection.diagnostics), 2)
        diagnostics = self._diagnostics()
        for record in (master, individual):
            diagnostic = diagnostics[record.client_proof_id]
            self.assertEqual(diagnostic.reason_codes, PERMANENT_BLOCKERS)
            self.assertTrue(diagnostic.evidence_complete)
            self.assertTrue(diagnostic.source_selected)
            self.assertEqual(diagnostic.source_category, "archived_manual")
            self.assertEqual(diagnostic.evidence_class, "full")
            self.assertFalse(diagnostic.eligible_to_offer)

    async def test_linking_a_record_to_its_master_state_leaves_trade_evidence_unchanged(self) -> None:
        database = await self._database()
        stored = await self._store(
            database,
            proof_id="proof_001",
            statement=MASTER_STATEMENT,
            name="master_identity",
            code=MASTER_CODE,
            master=False,
        )
        before = project_canonical_proof_store(self.store, store_role="archived_manual")

        masters = database.get_master_proofs_dir()
        masters.mkdir(parents=True, exist_ok=True)
        (masters / "master_proof_feedbeef.json").write_text(
            json.dumps(
                {
                    "master_key": "feedbeef",
                    "status": "verified",
                    "root_name": "master_identity",
                    "scaffold_code": MASTER_CODE,
                    "scaffold_revision": 1,
                    "total_attempts": 4,
                    "nodes": {},
                }
            ),
            encoding="utf-8",
        )
        reloaded = await self._database()
        linked = await reloaded.get_proof(stored.proof_id)
        self.assertEqual(linked.proof_kind, "master_proof")
        self.assertEqual(linked.compile_receipt, stored.compile_receipt)
        self.assertEqual(
            linked.local_novelty_assessment_receipt, stored.local_novelty_assessment_receipt
        )

        after = project_canonical_proof_store(self.store, store_role="archived_manual")
        self.assertEqual(after.diagnostics, before.diagnostics)
        self.assertEqual(after.projection_fingerprint, before.projection_fingerprint)
