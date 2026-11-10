"""Accepted brainstorm Lean proofs must register with their exact checked source.

The Lean checker always checks (and its compile receipt hashes) source that ends
with a newline. The Aggregator and LeanOJ brainstorm paths used to strip that
source before registration, so the receipt never matched and every accepted
brainstorm proof was dropped from the proofs database with "Compile receipt does
not bind the Lean source being registered".
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from backend.aggregator.core.coordinator import Coordinator
from backend.autonomous.core import proof_registration
from backend.autonomous.core.proof_registration import ensure_compile_receipt_binds_source
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.shared.build_info import BuildInfo
from backend.shared.lean4_client import Lean4Client
from backend.shared.models import (
    LeanCompileReceipt,
    LeanOJRoleConfig,
    LeanOJStartRequest,
    ProofAttemptFeedback,
)


async def _checked_source_and_receipt(tmp_path) -> tuple[str, LeanCompileReceipt]:
    """Prepare and receipt a source exactly as the Lean checker does on success."""
    client = Lean4Client(lean_path="lean", workspace_dir=str(tmp_path))
    original = "theorem brainstorm_true : True := by trivial"
    checked = client._prepare_lean_code(original)
    with mock.patch.object(
        client, "get_version", mock.AsyncMock(return_value="Lean (version 4.0.0)")
    ), mock.patch.object(
        client, "get_mathlib_commit", return_value="a" * 40
    ), mock.patch(
        "backend.shared.lean4_client.get_build_info",
        return_value=BuildInfo(
            version="1.1.09",
            build_commit="fixture_commit",
            update_channel="main",
            api_contract_version="fixture_contract",
        ),
    ):
        receipt = await client._build_compile_receipt(
            original_source=original,
            checked_source=checked,
            checker_mode="subprocess",
            timeout=60,
            verification_profile="brainstorm_proof_gate",
        )
    return checked, receipt


def _proof_payload(checked: str, receipt: LeanCompileReceipt) -> dict:
    attempt = ProofAttemptFeedback(
        attempt=1,
        theorem_id="brainstorm_inline_proof",
        lean_code=checked,
        success=True,
        compile_receipt=receipt,
    )
    return {
        "theorem_statement": "True is true.",
        "theorem_name": "brainstorm_true",
        "formal_sketch": "trivial",
        "lean_code": checked,
        "attempts": [attempt.model_dump(mode="json")],
        "attempt_count": 1,
        "compile_receipt": receipt.model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_checked_source_keeps_trailing_newline_and_stripping_breaks_the_receipt(tmp_path):
    checked, receipt = await _checked_source_and_receipt(tmp_path)

    assert checked.endswith("\n")
    ensure_compile_receipt_binds_source(checked, receipt)
    with pytest.raises(ValueError, match="does not bind"):
        ensure_compile_receipt_binds_source(checked.strip(), receipt)


@pytest.mark.asyncio
async def test_aggregator_registers_accepted_brainstorm_proof_with_exact_checked_source(
    tmp_path, monkeypatch
):
    checked, receipt = await _checked_source_and_receipt(tmp_path)
    registered: dict = {}

    async def fake_register(**kwargs):
        ensure_compile_receipt_binds_source(kwargs["lean_code"], kwargs["compile_receipt"])
        registered.update(kwargs)
        return SimpleNamespace(record=SimpleNamespace(proof_id="proof_brainstorm_1"))

    monkeypatch.setattr(proof_registration, "register_verified_lean_proof", fake_register)
    coordinator = Coordinator()
    coordinator.total_acceptances = 1
    submission = SimpleNamespace(
        submission_id="submission-1",
        submitter_id=1,
        metadata={"brainstorm_lean_proof": _proof_payload(checked, receipt)},
    )

    await coordinator._register_accepted_brainstorm_proof(submission)

    assert registered["lean_code"] == checked
    assert registered["compile_receipt"] == receipt
    assert submission.metadata["proof_id"] == "proof_brainstorm_1"


def _leanoj_role() -> LeanOJRoleConfig:
    return LeanOJRoleConfig(model_id="test-model", context_window=8192, max_output_tokens=1024)


@pytest.mark.asyncio
async def test_leanoj_registers_accepted_brainstorm_proof_with_exact_checked_source(tmp_path):
    checked, receipt = await _checked_source_and_receipt(tmp_path)
    registered: dict = {}
    coordinator = LeanOJCoordinator()

    async def fake_register(_request, **kwargs):
        ensure_compile_receipt_binds_source(kwargs["lean_code"], kwargs["compile_receipt"])
        registered.update(kwargs)
        return None

    async def noop(*_args, **_kwargs):
        return None

    coordinator._register_verified_leanoj_proof = fake_register  # type: ignore[method-assign]
    coordinator._persist_and_broadcast = noop  # type: ignore[method-assign]
    request = LeanOJStartRequest(
        user_prompt="Prove True.",
        lean_template="import Mathlib\n\nexample : True := by\n  sorry",
        topic_generator=_leanoj_role(),
        topic_validator=_leanoj_role(),
        brainstorm_submitters=[_leanoj_role()],
        brainstorm_validator=_leanoj_role(),
        path_decider=_leanoj_role(),
        final_solver=_leanoj_role(),
    )

    await coordinator._record_accepted_brainstorm_proof(
        request,
        1,
        {"brainstorm_lean_proof": _proof_payload(checked, receipt)},
    )

    assert registered["lean_code"] == checked
    assert registered["compile_receipt"] == receipt
    assert coordinator.get_state().verified_subproofs[-1].lean_code == checked
