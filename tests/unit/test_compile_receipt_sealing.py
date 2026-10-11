import json
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.build_info import BuildInfo
from backend.shared.installation_identity import InstallationBinding
from backend.shared.lean4_client import Lean4Client
from backend.shared.lean_source_evidence import receipt_digest
from backend.shared.models import (
    LeanCompileReceipt,
    LocalNoveltyAssessmentProvenance,
    ProofRecord,
)

LEAN_CODE = "import Mathlib\n\ntheorem sealed_fixture : 1 + 1 = 2 := by norm_num\n"


async def _checker_receipt(tmpdir: str) -> LeanCompileReceipt:
    """Build a receipt through the real Lean checker evidence path."""
    client = Lean4Client(lean_path="lean", workspace_dir=tmpdir)
    with mock.patch.object(
        client, "get_version", mock.AsyncMock(return_value="Lean (version 4.0.0)")
    ), mock.patch.object(
        client, "get_mathlib_commit", return_value="a" * 40
    ), mock.patch(
        "backend.shared.lean4_client.get_build_info",
        return_value=BuildInfo(
            version="1.1.08",
            build_commit="fixture_commit",
            update_channel="main",
            api_contract_version="fixture_contract",
        ),
    ):
        return await client._build_compile_receipt(
            original_source=LEAN_CODE,
            checked_source=LEAN_CODE,
            checker_mode="subprocess",
            timeout=60,
            verification_profile="full_proof",
        )


def _v1_1_08_sealed(receipt: LeanCompileReceipt) -> LeanCompileReceipt:
    """Reproduce the v1.1.08 digest, computed over raw ``str(datetime)`` values."""
    payload = receipt.model_dump(mode="python")
    payload["receipt_digest"] = receipt_digest(payload)
    return LeanCompileReceipt.model_validate(payload)


def _record(receipt: LeanCompileReceipt) -> ProofRecord:
    return ProofRecord(
        proof_id="proof_001",
        theorem_statement="1 + 1 = 2",
        theorem_name="sealed_fixture",
        source_type="brainstorm",
        source_id="brainstorm_sealed",
        run_id="run-sealed",
        lean_code=LEAN_CODE,
        novelty_tier="not_novel",
        independent_novelty_tier="not_novel",
        local_novelty_assessment_provenance=LocalNoveltyAssessmentProvenance(
            assessment_origin="validator_assessed",
            source_local_rank="not_novel",
        ),
        compile_receipt=receipt,
        verification_evidence_status="complete",
    )


class CompileReceiptSealingTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
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
        database.set_base_dir(Path(self._tmp.name) / "proofs")
        await database.initialize()
        return database

    async def test_checker_receipt_digest_covers_its_json_serialization(self) -> None:
        receipt = await _checker_receipt(self._tmp.name)
        payload = receipt.model_dump(mode="json")

        assert receipt.receipt_digest == receipt_digest(payload)
        round_tripped = LeanCompileReceipt.model_validate(json.loads(json.dumps(payload)))
        assert round_tripped.receipt_digest == receipt_digest(
            round_tripped.model_dump(mode="json")
        )

    async def test_checker_receipt_persists_with_lean_verified_proof(self) -> None:
        database = await self._database()
        receipt = await _checker_receipt(self._tmp.name)

        stored = await database.add_proof_occurrence(_record(receipt))

        sealed = stored.compile_receipt
        assert sealed.checked_source_artifact
        assert sealed.receipt_digest == receipt_digest(sealed.model_dump(mode="json"))

    async def test_v1_1_08_checkpointed_receipt_is_resealed_on_persist(self) -> None:
        database = await self._database()
        legacy = _v1_1_08_sealed(await _checker_receipt(self._tmp.name))
        checkpointed = LeanCompileReceipt.model_validate(
            json.loads(json.dumps(legacy.model_dump(mode="json")))
        )

        stored = await database.add_proof_occurrence(_record(checkpointed))

        sealed = stored.compile_receipt
        assert sealed.receipt_digest != legacy.receipt_digest
        assert sealed.receipt_digest == receipt_digest(sealed.model_dump(mode="json"))

    async def test_tampered_receipt_is_rejected(self) -> None:
        database = await self._database()
        receipt = await _checker_receipt(self._tmp.name)
        tampered = receipt.model_copy(update={"lean_version": "Lean (forged)"})

        with self.assertRaisesRegex(ValueError, "Compile receipt digest is invalid"):
            await database.add_proof_occurrence(_record(tampered))
