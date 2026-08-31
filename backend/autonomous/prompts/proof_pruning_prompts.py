"""Dedicated prompt contracts for proof live-context pruning.

These builders are intentionally isolated from proof identification,
formalization, novelty, integrity, and dependency prompts.
"""
import json

from backend.shared.models import ProofPruneProposal, ProofPruneSnapshot


_PROPOSER_CONTRACT = """{
  "action": "no_prune | propose_prune",
  "proof_id": "stable proof ID or null",
  "expected_theorem_hash": "canonical theorem hash or null",
  "expected_lean_hash": "canonical Lean hash or null",
  "prune_category": "outdated | contextually_misaligned | redundant | superseded | low_unique_utility | null",
  "supporting_proof_ids": ["one to three retained proof IDs"],
  "coverage_claims": [{"target_contribution": "material contribution", "preserved_by_proof_ids": ["retained proof IDs"], "explanation": "how it remains represented"}],
  "reasoning": "bounded explanation"
}"""

_VALIDATOR_CONTRACT = """{
  "decision": "accept | reject",
  "proof_id": "the exact proposed stable proof ID",
  "supporting_proof_ids": ["the exact ordered supporting proof IDs"],
  "coverage_confirmed": true or false,
  "reasoning": "bounded independent assessment"
}"""


def _snapshot_payload(snapshot: ProofPruneSnapshot, *, include_lean: bool) -> str:
    payload = snapshot.model_dump(mode="json")
    if not include_lean:
        for key in ("candidate_descriptors",):
            for descriptor in payload.get(key, []):
                descriptor["lean_code"] = ""
                descriptor["lean_code_included"] = False
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def build_proof_pruning_proposer_prompt(
    snapshot: ProofPruneSnapshot,
    *,
    include_lean: bool = True,
) -> str:
    return f"""You are the Rigor & Proofs proof live-context pruning proposer.

Your authority is narrow: review the user's whole objective and the immutable
active-proof snapshot below, then either nominate exactly one eligible weak
occurrence or decline pruning. This decision changes only future model context
for the owning run; it never deletes proof history, Lean code, certificates,
graphs, future-run memory, or SyntheticLib-facing availability.

Choose propose_prune only when the target is contextually obsolete or misaligned,
fully redundant, superseded by stronger active evidence, or has no remaining
unique utility toward the user's objective. Cite retained active proofs that
preserve every material contribution. Lean acceptance remains authoritative for
the target's exact formal statement: contextual misalignment is not mathematical
invalidity. Do not nominate a proof merely because it is old, long, costly,
known, duplicate-novel, difficult, or unused by the latest attempt. Missing
dependency or semantic evidence is not evidence of safe removal.
Use only proof IDs marked eligible_candidate in the deterministic whole-set
accounting, and cite supporting proofs only from hydrated descriptors. You may
choose no_prune whenever evidence is insufficient.

Return exactly this JSON object and no other fields:
{_PROPOSER_CONTRACT}

Dependent-field rules:
- no_prune requires all target/category fields to be null and both arrays empty.
- propose_prune requires one non-empty proof_id, category, both matching hashes,
  one to three supporting proof IDs, and at least one coverage claim.
- reasoning must be non-empty.

IMMUTABLE SNAPSHOT:
{_snapshot_payload(snapshot, include_lean=include_lean)}

Return only one valid JSON object matching the proposer contract."""


def build_proof_pruning_validator_prompt(
    snapshot: ProofPruneSnapshot,
    proposal: ProofPruneProposal,
    *,
    guard_summary: str,
    include_lean: bool = True,
) -> str:
    proposal_json = json.dumps(
        proposal.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return f"""You are the independent Validator for one proposed proof
live-context prune. Judge only whether this exact target can be removed from
future model context in the owning run without losing unique value toward the
user's whole objective. Do not recheck Lean correctness or novelty, nominate a
different proof, or treat shorter context alone as sufficient benefit.

Accept only when stronger active proof evidence preserves the full material
contribution, dependency implications are clear, no distinct route is lost, and
subsequent solving context is improved or preserved. Uncertainty means reject.
Age, length, context pressure, low novelty, or topical disagreement alone are
never sufficient. Lean validity is not under review. Echo the proposal's exact
ordered supporting proof IDs; do not substitute a different semantic basis.

Return exactly this JSON object and no other fields:
{_VALIDATOR_CONTRACT}

IMMUTABLE SNAPSHOT:
{_snapshot_payload(snapshot, include_lean=include_lean)}

DETERMINISTIC PROPOSAL:
{proposal_json}

DETERMINISTIC GUARD RESULTS:
{guard_summary}

Return only one valid JSON object matching the Validator contract. The proof_id
must exactly equal {json.dumps(proposal.proof_id)} and supporting_proof_ids must
exactly equal {json.dumps(proposal.supporting_proof_ids)}."""


def build_proof_pruning_repair_prompt(*, contract: str, error_summary: str) -> str:
    return f"""Your previous proof-pruning response could not be used.
Return the same decision as one valid JSON object only.

STRUCTURAL ERROR:
{str(error_summary or "invalid response")[:500]}

REQUIRED CONTRACT:
{contract}

Do not add markdown, commentary, or extra fields."""


def proposer_contract_text() -> str:
    return _PROPOSER_CONTRACT


def validator_contract_text() -> str:
    return _VALIDATOR_CONTRACT
