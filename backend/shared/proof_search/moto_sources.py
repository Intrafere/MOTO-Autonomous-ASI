"""Canonical MOTO proof database normalization for unified proof search."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from backend.autonomous.memory.proof_database import manual_proof_database, proof_database
from backend.shared.models import ProofDependency, ProofRecord
from backend.shared.proof_search.models import UnifiedProofSearchRecord


async def load_moto_proof_records() -> list[UnifiedProofSearchRecord]:
    """Collect current canonical MOTO proof records without reading display appendices."""
    records: list[UnifiedProofSearchRecord] = []
    records.extend(await _records_from_database(proof_database, default_corpus="moto"))
    records.extend(await _records_from_autonomous_history())
    records.extend(await _records_from_database(manual_proof_database, default_corpus="manual"))
    records.extend(await _records_from_manual_history())
    return records


async def _records_from_database(database, *, default_corpus: str) -> list[UnifiedProofSearchRecord]:
    try:
        proofs = await database.get_all_proofs(novel_only=None)
    except Exception:
        return []
    return [
        normalize_proof_record(proof, default_corpus=default_corpus, session_id="")
        for proof in proofs
    ]


async def _records_from_autonomous_history() -> list[UnifiedProofSearchRecord]:
    try:
        entries = await proof_database.list_proof_library(novel_only=False)
    except Exception:
        return []
    records: list[UnifiedProofSearchRecord] = []
    for entry in entries:
        full_entry = entry
        session_id = str(entry.get("session_id", "") or "")
        proof_id = str(entry.get("proof_id", "") or "")
        if session_id and proof_id:
            try:
                hydrated = await proof_database.get_library_proof(session_id, proof_id)
            except Exception:
                hydrated = None
            if hydrated:
                full_entry = {**entry, **hydrated}
        records.append(_record_from_library_entry(full_entry, default_corpus="moto"))
    return records


async def _records_from_manual_history() -> list[UnifiedProofSearchRecord]:
    try:
        from backend.shared.config import system_config

        entries = await manual_proof_database.list_proof_library_from_history(
            Path(system_config.data_dir) / "manual_proof_runs",
            novel_only=False,
        )
    except Exception:
        return []

    return [_record_from_library_entry(entry, default_corpus="manual") for entry in entries]


def _record_from_library_entry(
    entry: dict[str, Any],
    *,
    default_corpus: str,
) -> UnifiedProofSearchRecord:
    theorem_statement = str(entry.get("theorem_statement", "") or "")
    proof_id = str(entry.get("proof_id", "") or "")
    session_id = str(entry.get("session_id", "") or "")
    source_type = str(entry.get("source_type", "") or "")
    corpus = "leanoj" if source_type.startswith("leanoj_") else default_corpus
    lean_code = str(entry.get("lean_code", "") or "")
    return UnifiedProofSearchRecord(
        search_id=f"{corpus}:{session_id}:{proof_id}",
        corpus=corpus,
        corpus_scope="archived" if default_corpus == "manual" else "history",
        source_kind="verified_proof",
        proof_id=proof_id,
        session_id=session_id,
        source_type=source_type,
        source_id=str(entry.get("source_id", "") or ""),
        source_title=str(entry.get("source_title", "") or ""),
        display_title=str(entry.get("theorem_name", "") or proof_id),
        theorem_name=str(entry.get("theorem_name", "") or ""),
        theorem_statement=theorem_statement,
        formal_sketch=str(entry.get("formal_sketch", "") or ""),
        lean_code=lean_code,
        lean_code_hash=_sha256_text(lean_code) if lean_code else "",
        theorem_statement_hash=_sha256_text(theorem_statement),
        dependency_names=_dependency_names(entry.get("dependencies", [])),
        novelty_tier=str(entry.get("novelty_tier", "") or ""),
        novelty_reasoning=str(entry.get("novelty_reasoning", "") or ""),
        verified=True,
        created_at=str(entry.get("created_at", "") or ""),
        canonical_uri=f"moto-proof://{corpus}/{session_id}/{proof_id}",
        metadata={
            "novel": bool(entry.get("novel", False)),
            "solver": str(entry.get("solver", "") or "Lean 4"),
            "attempt_count": entry.get("attempt_count", 0),
            "verification_notes": str(entry.get("verification_notes", "") or ""),
            "user_prompt": str(entry.get("user_prompt", "") or ""),
        },
    )


def normalize_proof_record(
    proof: ProofRecord,
    *,
    default_corpus: str = "moto",
    session_id: str = "",
) -> UnifiedProofSearchRecord:
    """Convert a stored ProofRecord into the shared search model."""
    corpus = "leanoj" if proof.source_type.startswith("leanoj_") else default_corpus
    lean_code_hash = _sha256_text(proof.lean_code) if proof.lean_code else ""
    statement_hash = _sha256_text(proof.theorem_statement)
    scope = "active" if default_corpus == "manual" else "current"

    return UnifiedProofSearchRecord(
        search_id=f"{corpus}:{session_id}:{proof.proof_id}",
        corpus=corpus,
        corpus_scope=scope,
        source_kind="verified_proof",
        proof_id=proof.proof_id,
        session_id=session_id,
        source_type=proof.source_type,
        source_id=proof.source_id,
        source_title=proof.source_title,
        display_title=proof.theorem_name or proof.theorem_id or proof.proof_id,
        theorem_name=proof.theorem_name,
        theorem_statement=proof.theorem_statement,
        formal_sketch=proof.formal_sketch,
        lean_code=proof.lean_code,
        lean_code_hash=lean_code_hash,
        theorem_statement_hash=statement_hash,
        imports=_import_names(proof.dependencies),
        dependency_names=_dependency_names(proof.dependencies),
        novelty_tier=proof.novelty_tier,
        novelty_reasoning=proof.novelty_reasoning,
        verified=True,
        created_at=proof.created_at.isoformat() if proof.created_at else "",
        canonical_uri=f"moto-proof://{corpus}/{proof.proof_id}",
        metadata={
            "theorem_id": proof.theorem_id,
            "solver": proof.solver,
            "novel": proof.novel,
            "verification_notes": proof.verification_notes,
            "attempt_count": proof.attempt_count,
        },
    )


def _dependency_names(dependencies: list[Any]) -> list[str]:
    names: list[str] = []
    for dependency in dependencies or []:
        if isinstance(dependency, ProofDependency):
            name = dependency.name
        elif isinstance(dependency, dict):
            name = str(dependency.get("name", "") or "")
        else:
            name = ""
        if name:
            names.append(name)
    return names


def _import_names(dependencies: list[Any]) -> list[str]:
    imports = []
    for dependency in dependencies or []:
        kind = dependency.kind if isinstance(dependency, ProofDependency) else dependency.get("kind", "")
        name = dependency.name if isinstance(dependency, ProofDependency) else dependency.get("name", "")
        if kind == "mathlib" and name:
            imports.append(str(name).split(".")[0])
    return sorted(set(imports))


def _sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()

