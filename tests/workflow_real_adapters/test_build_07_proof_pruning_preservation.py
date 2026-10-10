from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from backend.api.routes import proofs as proofs_route
from backend.autonomous.memory.proof_database import ProofDatabase
from backend.shared.config import system_config
from backend.shared.models import ProofDependency, ProofRecord
from backend.shared.proof_search.indexer import ProofSearchIndexer
from backend.shared.proof_search.models import ProofSearchRequest
from backend.shared.proof_search.moto_sources import normalize_proof_record


def _record(
    proof_id: str,
    theorem_name: str,
    *,
    dependencies: list[ProofDependency] | None = None,
) -> ProofRecord:
    return ProofRecord(
        proof_id=proof_id,
        theorem_statement=f"{theorem_name} statement",
        theorem_name=theorem_name,
        source_type="paper",
        source_id="paper-pruning",
        source_title="Pruning preservation paper",
        run_id="creation-run",
        user_prompt="Preserve every canonical proof surface.",
        lean_code=f"theorem {theorem_name} : True := by trivial",
        novel=True,
        novelty_tier="mathematical_discovery",
        independent_novelty_tier="mathematical_discovery",
        canonical_identity_version="moto-proof-identity-v1",
        canonical_theorem_statement_hash=f"{proof_id}-theorem",
        canonical_lean_code_hash=f"{proof_id}-lean",
        dependencies=list(dependencies or []),
        dependency_extraction_status="complete",
    )


@pytest.mark.asyncio
async def test_pruned_occurrence_remains_exportable_graph_visible_searchable_and_eligible(
    tmp_path,
):
    database = ProofDatabase()
    database.set_base_dir(tmp_path / "proofs")
    await database.initialize()
    target = await database.add_proof_occurrence(
        _record("proof-target", "targetProof")
    )
    dependent = await database.add_proof_occurrence(
        _record(
            "proof-dependent",
            "dependentProof",
            dependencies=[
                ProofDependency(
                    kind="moto",
                    name=target.theorem_name,
                    source_ref=target.proof_id,
                )
            ],
        )
    )
    counts_before = database.count_proofs()
    lean_before = await database.get_lean_code(target.proof_id)
    revision = await database.get_proof_set_revision()

    updated, _ = await database.set_live_context_status(
        proof_id=target.proof_id,
        status="pruned",
        expected_run_id=target.run_id,
        expected_proof_set_revision=revision,
        actor="user",
        reason="Redundant only in the owner run.",
        owning_run_id="owner-run",
        expected_theorem_hash=target.canonical_theorem_statement_hash,
        expected_lean_hash=target.canonical_lean_code_hash,
    )
    assert updated.live_context_status == "pruned"

    with (
        patch.object(proofs_route, "proof_database", database),
        patch.object(system_config, "lean4_enabled", True),
    ):
        certificate = await proofs_route.get_proof_certificate(
            target.proof_id, scope="autonomous"
        )
        lean_export = await proofs_route.get_proof_certificate_lean(
            target.proof_id, scope="autonomous"
        )
        dependencies = await proofs_route.get_proof_dependencies(
            target.proof_id, scope="autonomous"
        )
        graph = await proofs_route.get_proof_graph(scope="autonomous")

    certificate_payload = json.loads(certificate.body)
    assert certificate_payload["proof_id"] == target.proof_id
    assert lean_export.body.decode() == lean_before
    assert dependencies["depended_on_by"][0]["proof_id"] == dependent.proof_id
    assert {node["proof_id"] for node in graph["nodes"]} == {
        target.proof_id,
        dependent.proof_id,
    }
    assert any(
        edge["from"] == dependent.proof_id
        and edge["to"] == target.proof_id
        for edge in graph["edges_moto"]
    )

    records = [
        normalize_proof_record(record)
        for record in await database.get_all_proofs()
    ]
    indexer = ProofSearchIndexer(tmp_path / "proof-search.sqlite")
    indexer.rebuild(records)
    request = ProofSearchRequest(
        query=target.theorem_name,
        corpora=["moto"],
        search_mode="text",
    )
    human = indexer.search(request)
    owner = indexer.search(
        request,
        requesting_run_id="owner-run",
        filter_live_context=True,
    )
    future = indexer.search(
        request,
        requesting_run_id="future-run",
        filter_live_context=True,
    )
    assert target.proof_id in {record.proof_id for record in human.results}
    assert target.proof_id not in {record.proof_id for record in owner.results}
    assert target.proof_id in {record.proof_id for record in future.results}

    reloaded = await database.get_proof(target.proof_id)
    assert reloaded is not None
    assert reloaded.canonical_theorem_statement_hash == (
        target.canonical_theorem_statement_hash
    )
    assert reloaded.canonical_lean_code_hash == target.canonical_lean_code_hash
    assert database.count_proofs()["syntheticlib_novel"] == (
        counts_before["syntheticlib_novel"]
    )
