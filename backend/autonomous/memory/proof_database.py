"""
Proof database for Lean 4 verified results.

Stores both novel and non-novel verified proofs centrally for UI/API access.
Novel proofs are also formatted for highest-priority direct prompt injection.
"""
import asyncio
import json
import logging
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiofiles

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.models import FailedProofCandidate, ProofCandidate, ProofRecord
from backend.shared.path_safety import resolve_path_within_root, validate_single_path_component
from backend.shared.proof_identity import canonical_proof_identity
from backend.autonomous.prompts.proof_prompts import format_failure_hints_for_injection

logger = logging.getLogger(__name__)

DUPLICATE_NOVEL_TIER = "duplicate_novel"
NOT_NOVEL_TIER = "not_novel"
PROOF_LIBRARY_CATEGORIES = frozenset({"novel", "duplicate_novel", "not_novel", "all"})
PROMPT_INJECTION_NOVEL_TIERS = frozenset(
    {
        "novel_formulation",
        "novel_variant",
        "mathematical_discovery",
        "major_mathematical_discovery",
    }
)


def is_duplicate_novel_tier(novelty_tier: str) -> bool:
    return str(novelty_tier or "").strip().lower() == DUPLICATE_NOVEL_TIER


def is_not_novel_tier(novelty_tier: str) -> bool:
    return str(novelty_tier or "").strip().lower() == NOT_NOVEL_TIER


def is_syntheticlib_novel_tier(novelty_tier: str) -> bool:
    return not is_not_novel_tier(novelty_tier)


def is_prompt_injection_novel_tier(novelty_tier: str) -> bool:
    return str(novelty_tier or "").strip().lower() in PROMPT_INJECTION_NOVEL_TIERS


def normalize_proof_library_category(category: Optional[str] = None, novel_only: Optional[bool] = None) -> str:
    normalized = str(category or "").strip().lower()
    if normalized in PROOF_LIBRARY_CATEGORIES:
        return normalized
    if novel_only is None:
        return "novel"
    return "novel" if novel_only else "all"


def proof_matches_library_category(proof_data: Dict[str, Any], category: str) -> bool:
    normalized_category = normalize_proof_library_category(category, None)
    if normalized_category == "all":
        return True
    novelty_tier = str(proof_data.get("novelty_tier") or "").strip().lower()
    if normalized_category == "duplicate_novel":
        return novelty_tier == DUPLICATE_NOVEL_TIER
    if normalized_category == "not_novel":
        return novelty_tier == NOT_NOVEL_TIER or (not novelty_tier and not bool(proof_data.get("novel")))
    return (
        bool(proof_data.get("novel"))
        and (is_prompt_injection_novel_tier(novelty_tier) or not novelty_tier)
    )


class ProofDatabase:
    """
    Session-aware storage for Lean 4 verified proofs.

    Storage layout:
      - proofs_index.json
      - proof_<proof_id>.json
      - proof_<proof_id>_lean.lean
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._base_dir = Path(system_config.data_dir) / "proofs"
        self._root_relative_default: Optional[str] = "proofs"
        self._root_generation = system_config.runtime_root_generation
        self._session_manager = None
        self._index_data: Optional[Dict[str, Any]] = None
        self._mathlib_reverse_index: Dict[str, List[str]] = {}
        self._mathlib_reverse_short_index: Dict[str, List[str]] = {}

    def set_session_manager(self, session_manager) -> None:
        """Switch storage to the active session directory when available."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            self._base_dir = session_manager.get_proofs_dir()
            self._root_relative_default = None
        else:
            self._base_dir = Path(system_config.data_dir) / "proofs"
            self._root_relative_default = "proofs"
            self._root_generation = system_config.runtime_root_generation
        self._index_data = None
        logger.info("Proof database using path: %s", self._base_dir)

    def set_base_dir(self, base_dir: Path) -> None:
        """Use a fixed proof-storage directory independent of autonomous sessions."""
        self._session_manager = None
        self._base_dir = Path(base_dir)
        data_root = Path(system_config.data_dir).resolve(strict=False)
        resolved = self._base_dir.resolve(strict=False)
        try:
            relative = resolved.relative_to(data_root)
        except ValueError:
            self._root_relative_default = None
        else:
            self._root_relative_default = str(relative)
            self._root_generation = system_config.runtime_root_generation
        self._index_data = None
        logger.info("Proof database using fixed path: %s", self._base_dir)

    def _safe_proof_id(self, proof_id: str) -> str:
        return validate_single_path_component(proof_id, "proof ID")

    def _refresh_runtime_root(self) -> None:
        if (
            self._session_manager is None
            and self._root_relative_default is not None
            and self._root_generation != system_config.runtime_root_generation
        ):
            self._base_dir = Path(system_config.data_dir) / self._root_relative_default
            self._root_generation = system_config.runtime_root_generation
            self._index_data = None

    def _get_index_path(self) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / "proofs_index.json"

    def _get_record_path(self, proof_id: str) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / f"proof_{self._safe_proof_id(proof_id)}.json"

    def _get_lean_path(self, proof_id: str) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / f"proof_{self._safe_proof_id(proof_id)}_lean.lean"

    def _get_failed_dir(self) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / "failed"

    def _get_failed_candidates_path(self, source_brainstorm_id: str) -> Path:
        safe_id = validate_single_path_component(source_brainstorm_id, "brainstorm ID")
        return self._get_failed_dir() / f"{safe_id}.json"

    def _default_index(self) -> Dict[str, Any]:
        return {
            "next_proof_id": 1,
            "proofs": [],
        }

    async def get_or_create_active_run_id(self) -> str:
        """Return the durable explicit run ID owned by this proof database."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()
            run_id = str(self._index_data.get("active_run_id") or "").strip()
            if not run_id:
                run_id = f"manual-{uuid.uuid4().hex}"
                self._index_data["active_run_id"] = run_id
                await self._save_index()
            return run_id

    def _rebuild_reverse_indexes(self) -> None:
        self._mathlib_reverse_index = {}
        self._mathlib_reverse_short_index = {}

        proofs = self._index_data.get("proofs", []) if self._index_data else []
        for proof in proofs:
            proof_id = str(proof.get("proof_id", "")).strip()
            if not proof_id:
                continue
            for dependency in proof.get("dependencies", []) or []:
                if not isinstance(dependency, dict):
                    continue
                if dependency.get("kind") != "mathlib":
                    continue
                name = str(dependency.get("name", "")).strip()
                if not name:
                    continue
                short_name = name.split(".")[-1]
                self._mathlib_reverse_index.setdefault(name, [])
                if proof_id not in self._mathlib_reverse_index[name]:
                    self._mathlib_reverse_index[name].append(proof_id)
                self._mathlib_reverse_short_index.setdefault(short_name, [])
                if proof_id not in self._mathlib_reverse_short_index[short_name]:
                    self._mathlib_reverse_short_index[short_name].append(proof_id)

    def _rebuild_index_from_record_files_sync(self) -> Dict[str, Any]:
        self._refresh_runtime_root()
        proofs: List[Dict[str, Any]] = []
        for record_path in self._base_dir.glob("proof_*.json"):
            if record_path.name.endswith("_metadata.json"):
                continue
            try:
                data = json.loads(record_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not data.get("proof_id"):
                    continue
                proofs.append(data)
            except Exception as exc:
                logger.warning("Skipping unreadable proof record during index rebuild: %s (%s)", record_path, exc)

        proofs.sort(key=lambda proof: proof.get("created_at", ""), reverse=True)
        max_numeric_id = 0
        for proof in proofs:
            proof_id = str(proof.get("proof_id", ""))
            match = re.search(r"(\d+)$", proof_id)
            if match:
                max_numeric_id = max(max_numeric_id, int(match.group(1)))
        return {
            "next_proof_id": max(max_numeric_id + 1, len(proofs) + 1, 1),
            "proofs": proofs,
        }

    async def initialize(self) -> None:
        """Ensure storage exists and load the index."""
        if self._session_manager and self._session_manager.is_session_active:
            self._base_dir = self._session_manager.get_proofs_dir()
        else:
            self._refresh_runtime_root()

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._get_failed_dir().mkdir(parents=True, exist_ok=True)
        await self._load_index()

    async def _load_index(self) -> None:
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                async with aiofiles.open(index_path, "r", encoding="utf-8") as handle:
                    self._index_data = json.loads(await handle.read())
            except Exception as exc:
                logger.error("Failed to load proofs index: %s", exc)
                self._index_data = await asyncio.to_thread(self._rebuild_index_from_record_files_sync)
                logger.warning(
                    "Rebuilt proofs index from %s record file(s) after index load failure",
                    len(self._index_data.get("proofs", [])),
                )
        else:
            self._index_data = self._default_index()
            await self._save_index()

        if "next_proof_id" not in self._index_data:
            self._index_data["next_proof_id"] = len(self._index_data.get("proofs", [])) + 1
        if "proofs" not in self._index_data:
            self._index_data["proofs"] = []
        self._rebuild_reverse_indexes()

    def _ensure_index_loaded_sync(self) -> None:
        if self._index_data is not None:
            return

        index_path = self._get_index_path()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            try:
                self._index_data = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Failed to synchronously load proofs index: %s", exc)
                self._index_data = self._rebuild_index_from_record_files_sync()
        else:
            self._index_data = self._default_index()

        self._index_data.setdefault("next_proof_id", len(self._index_data.get("proofs", [])) + 1)
        self._index_data.setdefault("proofs", [])
        self._rebuild_reverse_indexes()

    async def _save_index(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._get_index_path(), "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(self._index_data, indent=2))

    @staticmethod
    def _serialize_record(record: ProofRecord) -> Dict[str, Any]:
        return record.model_dump(mode="json")

    @staticmethod
    def _deserialize_record(data: Dict[str, Any]) -> ProofRecord:
        return ProofRecord(**data)

    @staticmethod
    def _serialize_failed_candidate(candidate: FailedProofCandidate) -> Dict[str, Any]:
        return candidate.model_dump(mode="json")

    @staticmethod
    def _deserialize_failed_candidate(data: Dict[str, Any]) -> FailedProofCandidate:
        return FailedProofCandidate(**data)

    async def _load_failed_candidates(self, source_brainstorm_id: str) -> List[FailedProofCandidate]:
        failed_path = self._get_failed_candidates_path(source_brainstorm_id)
        if not failed_path.exists():
            return []

        try:
            async with aiofiles.open(failed_path, "r", encoding="utf-8") as handle:
                payload = json.loads(await handle.read())
            items = payload.get("items", []) if isinstance(payload, dict) else payload
            return [
                self._deserialize_failed_candidate(item)
                for item in items
                if isinstance(item, dict)
            ]
        except Exception as exc:
            logger.error("Failed to load failed proof candidates for %s: %s", source_brainstorm_id, exc)
            return []

    async def _save_failed_candidates(
        self,
        source_brainstorm_id: str,
        failed_candidates: List[FailedProofCandidate],
    ) -> None:
        self._get_failed_dir().mkdir(parents=True, exist_ok=True)
        failed_path = self._get_failed_candidates_path(source_brainstorm_id)
        payload = {
            "source_brainstorm_id": source_brainstorm_id,
            "items": [
                self._serialize_failed_candidate(candidate)
                for candidate in failed_candidates
            ],
        }
        async with aiofiles.open(failed_path, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(payload, indent=2))

    async def clear_failed_candidates(self) -> None:
        """Remove active failed proof retry hints without touching verified proofs."""
        async with self._lock:
            failed_dir = self._get_failed_dir()
            if failed_dir.exists():
                await asyncio.to_thread(shutil.rmtree, failed_dir, True)
            failed_dir.mkdir(parents=True, exist_ok=True)

    async def add_proof(self, record: ProofRecord) -> ProofRecord:
        """Persist a proof record and return the stored copy."""
        stored_record, _duplicate = await self.add_proof_if_absent(record)
        return stored_record

    async def add_proof_occurrence(self, record: ProofRecord) -> ProofRecord:
        """Persist a full record for each newly verified current-run occurrence."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proof_id = record.proof_id or f"proof_{self._index_data['next_proof_id']:03d}"
            stored_record = record.model_copy(update={"proof_id": proof_id})
            serialized = self._serialize_record(stored_record)
            async with aiofiles.open(self._get_record_path(proof_id), "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(serialized, indent=2))
            async with aiofiles.open(self._get_lean_path(proof_id), "w", encoding="utf-8") as handle:
                await handle.write(stored_record.lean_code)

            proofs = [
                proof
                for proof in self._index_data.get("proofs", [])
                if proof.get("proof_id") != proof_id
            ]
            proofs.append(serialized)
            proofs.sort(key=lambda proof: proof.get("created_at", ""), reverse=True)
            self._index_data["proofs"] = proofs
            current_number = self._index_data.get("next_proof_id", 1)
            self._index_data["next_proof_id"] = max(current_number + 1, len(proofs) + 1)
            self._rebuild_reverse_indexes()
            await self._save_index()
            return stored_record

    async def add_proof_if_absent(self, record: ProofRecord) -> tuple[ProofRecord, bool]:
        """Persist a proof record unless an identical source/theorem/code exists."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            identity = canonical_proof_identity(record.theorem_statement, record.lean_code)
            for existing in self._index_data.get("proofs", []):
                if existing.get("source_type") != record.source_type or existing.get("source_id") != record.source_id:
                    continue
                existing_identity = canonical_proof_identity(
                    str(existing.get("theorem_statement") or ""),
                    str(existing.get("lean_code") or ""),
                )
                if existing_identity.key != identity.key:
                    continue
                return self._deserialize_record(existing), True

            proof_id = record.proof_id or f"proof_{self._index_data['next_proof_id']:03d}"
            stored_record = record.model_copy(update={"proof_id": proof_id})
            serialized = self._serialize_record(stored_record)

            async with aiofiles.open(self._get_record_path(proof_id), "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(serialized, indent=2))
            async with aiofiles.open(self._get_lean_path(proof_id), "w", encoding="utf-8") as handle:
                await handle.write(stored_record.lean_code)

            proofs = [
                proof
                for proof in self._index_data.get("proofs", [])
                if proof.get("proof_id") != proof_id
            ]
            proofs.append(serialized)
            proofs.sort(key=lambda proof: proof.get("created_at", ""), reverse=True)

            self._index_data["proofs"] = proofs
            current_number = self._index_data.get("next_proof_id", 1)
            self._index_data["next_proof_id"] = max(current_number, len(proofs) + 1)
            self._rebuild_reverse_indexes()
            await self._save_index()

            logger.info(
                "Stored proof %s (%s, novel=%s) from %s %s",
                proof_id,
                stored_record.theorem_statement[:80],
                stored_record.novel,
                stored_record.source_type,
                stored_record.source_id,
            )
            return stored_record, False

    async def record_failed_candidate(
        self,
        source_brainstorm_id: str,
        theorem_candidate: ProofCandidate,
        error_summary: str,
        suggested_lemma_targets: Optional[List[str]] = None,
    ) -> FailedProofCandidate:
        """Persist a failed brainstorm theorem so later papers can retry it."""
        async with self._lock:
            failed_candidates = await self._load_failed_candidates(source_brainstorm_id)
            existing = None
            for candidate in failed_candidates:
                if candidate.theorem_id == theorem_candidate.theorem_id:
                    existing = candidate
                    break

            now = datetime.now()
            cleaned_targets = []
            for target in suggested_lemma_targets or []:
                normalized = str(target or "").strip()
                if normalized and normalized not in cleaned_targets:
                    cleaned_targets.append(normalized)
            if existing:
                existing.theorem_statement = theorem_candidate.statement
                existing.formal_sketch = theorem_candidate.formal_sketch
                existing.expected_novelty_tier = theorem_candidate.expected_novelty_tier
                existing.prompt_relevance_rationale = theorem_candidate.prompt_relevance_rationale
                existing.novelty_rationale = theorem_candidate.novelty_rationale
                existing.why_not_standard_known_result = theorem_candidate.why_not_standard_known_result
                existing.source_excerpt = theorem_candidate.source_excerpt
                existing.error_summary = error_summary
                if cleaned_targets:
                    existing.suggested_lemma_targets = cleaned_targets
                existing.updated_at = now
                stored_candidate = existing
            else:
                stored_candidate = FailedProofCandidate(
                    source_brainstorm_id=source_brainstorm_id,
                    theorem_id=theorem_candidate.theorem_id,
                    theorem_statement=theorem_candidate.statement,
                    formal_sketch=theorem_candidate.formal_sketch,
                    expected_novelty_tier=theorem_candidate.expected_novelty_tier,
                    prompt_relevance_rationale=theorem_candidate.prompt_relevance_rationale,
                    novelty_rationale=theorem_candidate.novelty_rationale,
                    why_not_standard_known_result=theorem_candidate.why_not_standard_known_result,
                    source_excerpt=theorem_candidate.source_excerpt,
                    error_summary=error_summary,
                    suggested_lemma_targets=cleaned_targets,
                    created_at=now,
                    updated_at=now,
                )
                failed_candidates.append(stored_candidate)

            await self._save_failed_candidates(source_brainstorm_id, failed_candidates)
            return stored_candidate

    async def get_pending_retries(
        self,
        source_brainstorm_id: str,
        retry_source_id: str = "",
    ) -> List[FailedProofCandidate]:
        """Return unresolved failed candidates eligible for retry."""
        async with self._lock:
            failed_candidates = await self._load_failed_candidates(source_brainstorm_id)
            pending = [
                candidate
                for candidate in failed_candidates
                if not candidate.resolved_proof_id
                and (not retry_source_id or candidate.last_retry_source_id != retry_source_id)
            ]
            pending.sort(key=lambda candidate: candidate.updated_at, reverse=True)
            return pending

    async def mark_retried(
        self,
        source_brainstorm_id: str,
        theorem_id: str,
        retry_source_id: str,
    ) -> None:
        """Mark a failed candidate as having been retried for a specific paper/source."""
        async with self._lock:
            failed_candidates = await self._load_failed_candidates(source_brainstorm_id)
            updated = False
            for candidate in failed_candidates:
                if candidate.theorem_id != theorem_id:
                    continue
                candidate.retry_count += 1
                candidate.last_retry_source_id = retry_source_id
                candidate.updated_at = datetime.now()
                updated = True
                break

            if updated:
                await self._save_failed_candidates(source_brainstorm_id, failed_candidates)

    async def mark_resolved_retry(
        self,
        source_brainstorm_id: str,
        theorem_id: str,
        proof_id: str,
    ) -> None:
        """Mark a failed candidate as resolved by a later verified proof."""
        async with self._lock:
            failed_candidates = await self._load_failed_candidates(source_brainstorm_id)
            updated = False
            for candidate in failed_candidates:
                if candidate.theorem_id != theorem_id:
                    continue
                candidate.resolved_proof_id = proof_id
                candidate.updated_at = datetime.now()
                updated = True
                break

            if updated:
                await self._save_failed_candidates(source_brainstorm_id, failed_candidates)

    async def get_recent_failure_hints(
        self,
        source_brainstorm_id: str,
        *,
        limit: int = 5,
    ) -> List[FailedProofCandidate]:
        """Return recent unresolved failed proof hints for brainstorm prompt injection."""
        async with self._lock:
            failed_candidates = await self._load_failed_candidates(source_brainstorm_id)
            hints = [candidate for candidate in failed_candidates if not candidate.resolved_proof_id]
            hints.sort(key=lambda candidate: candidate.updated_at, reverse=True)
            return hints[:limit]

    async def get_lean_code(self, proof_id: str) -> str:
        """Return the raw saved Lean file for a proof when available."""
        async with self._lock:
            lean_path = self._get_lean_path(proof_id)
            if lean_path.exists():
                try:
                    async with aiofiles.open(lean_path, "r", encoding="utf-8") as handle:
                        return await handle.read()
                except Exception as exc:
                    logger.error(
                        "Failed to read Lean file for %s: %s",
                        redact_log_text(proof_id, 120),
                        redact_log_text(exc, 240),
                    )

            if self._index_data is None:
                await self._load_index()
            for proof in self._index_data.get("proofs", []) if self._index_data else []:
                if proof.get("proof_id") == proof_id:
                    return str(proof.get("lean_code", "") or "")
            return ""

    async def get_all_proofs(self, novel_only: Optional[bool] = None) -> List[ProofRecord]:
        """Return all stored proofs, optionally filtered by novelty."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proofs = [
                self._deserialize_record(proof)
                for proof in self._index_data.get("proofs", [])
            ]
            if novel_only is None:
                return proofs
            if novel_only:
                return [
                    proof for proof in proofs
                    if proof.novel and (
                        is_prompt_injection_novel_tier(proof.novelty_tier)
                        or not str(proof.novelty_tier or "").strip()
                    )
                ]
            return [
                proof for proof in proofs
                if not proof.novel or (
                    bool(str(proof.novelty_tier or "").strip())
                    and not is_prompt_injection_novel_tier(proof.novelty_tier)
                )
            ]

    async def update_proof_dependencies(self, proof_id: str, dependencies) -> Optional[ProofRecord]:
        """Persist a new dependency list for an existing proof record."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            updated_record: Optional[ProofRecord] = None
            updated_proofs: List[Dict[str, Any]] = []

            for proof_data in self._index_data.get("proofs", []):
                if proof_data.get("proof_id") != proof_id:
                    updated_proofs.append(proof_data)
                    continue
                record = self._deserialize_record(proof_data)
                updated_record = record.model_copy(update={"dependencies": list(dependencies or [])})
                updated_proofs.append(self._serialize_record(updated_record))

            if updated_record is None:
                return None

            self._index_data["proofs"] = updated_proofs
            self._rebuild_reverse_indexes()

            async with aiofiles.open(self._get_record_path(proof_id), "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(self._serialize_record(updated_record), indent=2))
            await self._save_index()
            return updated_record

    async def get_dependencies(self, proof_id: str):
        """Return dependency edges for one proof."""
        proof = await self.get_proof(proof_id)
        if proof is None:
            return []
        return list(proof.dependencies or [])

    async def get_proofs_using_mathlib(self, name: str) -> List[ProofRecord]:
        """Return proofs that reference a specific Mathlib lemma name."""
        requested_name = str(name or "").strip()
        if not requested_name:
            return []

        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proof_ids = []
            for candidate_id in self._mathlib_reverse_index.get(requested_name, []):
                if candidate_id not in proof_ids:
                    proof_ids.append(candidate_id)

            short_name = requested_name.split(".")[-1]
            if not proof_ids:
                for candidate_id in self._mathlib_reverse_short_index.get(short_name, []):
                    if candidate_id not in proof_ids:
                        proof_ids.append(candidate_id)

            proofs: List[ProofRecord] = []
            for proof_data in self._index_data.get("proofs", []):
                proof_id = str(proof_data.get("proof_id", "")).strip()
                if proof_id and proof_id in proof_ids:
                    proofs.append(self._deserialize_record(proof_data))
            return proofs

    async def get_proofs_depending_on(self, proof_id: str) -> List[ProofRecord]:
        """Return proofs whose MOTO ancestry depends on the given proof."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proofs = [
                self._deserialize_record(proof)
                for proof in self._index_data.get("proofs", [])
            ]
            return [
                proof
                for proof in proofs
                if any(
                    dependency.kind == "moto" and dependency.source_ref == proof_id
                    for dependency in (proof.dependencies or [])
                )
            ]

    async def get_graph(self) -> Dict[str, Any]:
        """Return the proof graph in one pass for graph-oriented UIs."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proofs = [
                self._deserialize_record(proof)
                for proof in self._index_data.get("proofs", [])
            ]

        nodes = [
            {
                "proof_id": proof.proof_id,
                "theorem_name": proof.theorem_name,
                "theorem_statement": proof.theorem_statement,
                "source_type": proof.source_type,
                "source_id": proof.source_id,
                "source_title": proof.source_title,
                "solver": proof.solver,
                "is_novel": proof.novel,
                "novelty_tier": proof.novelty_tier,
                "created_at": proof.created_at.isoformat() if proof.created_at else None,
            }
            for proof in proofs
        ]

        edges_moto: List[Dict[str, str]] = []
        edges_mathlib: List[Dict[str, str]] = []
        for proof in proofs:
            for dependency in proof.dependencies or []:
                if dependency.kind == "moto" and dependency.source_ref:
                    edges_moto.append(
                        {
                            "from": proof.proof_id,
                            "to": dependency.source_ref,
                            "name": dependency.name,
                        }
                    )
                elif dependency.kind == "mathlib":
                    edges_mathlib.append(
                        {
                            "from": proof.proof_id,
                            "name": dependency.name,
                            "source_ref": dependency.source_ref,
                        }
                    )

        return {
            "nodes": nodes,
            "edges_moto": edges_moto,
            "edges_mathlib": edges_mathlib,
        }

    async def get_proof(self, proof_id: str) -> Optional[ProofRecord]:
        """Return one stored proof."""
        async with self._lock:
            record_path = self._get_record_path(proof_id)
            if record_path.exists():
                try:
                    async with aiofiles.open(record_path, "r", encoding="utf-8") as handle:
                        return self._deserialize_record(json.loads(await handle.read()))
                except Exception as exc:
                    logger.error(
                        "Failed to read proof %s: %s",
                        redact_log_text(proof_id, 120),
                        redact_log_text(exc, 240),
                    )

            if self._index_data is None:
                await self._load_index()
            for proof in self._index_data.get("proofs", []):
                if proof.get("proof_id") == proof_id:
                    return self._deserialize_record(proof)
        return None

    def count_proofs(self) -> Dict[str, int]:
        """Return proof counts for display and prompt gating."""
        self._ensure_index_loaded_sync()
        proofs = self._index_data.get("proofs", []) if self._index_data else []
        duplicate_novel_count = sum(
            1 for proof in proofs if is_duplicate_novel_tier(proof.get("novelty_tier", ""))
        )
        prompt_novel_count = sum(
            1 for proof in proofs if proof.get("novel") and not is_duplicate_novel_tier(proof.get("novelty_tier", ""))
        )
        syntheticlib_novel_count = prompt_novel_count + duplicate_novel_count
        not_novel_count = sum(
            1 for proof in proofs if is_not_novel_tier(proof.get("novelty_tier", NOT_NOVEL_TIER))
        )
        return {
            "total": len(proofs),
            "novel": prompt_novel_count,
            "syntheticlib_novel": syntheticlib_novel_count,
            "duplicate_novel": duplicate_novel_count,
            "not_novel": not_novel_count,
            "known": len(proofs) - syntheticlib_novel_count,
        }

    def get_known_proofs_summary_for_browsing(
        self,
        source_id: Optional[str] = None,
        limit: int = 15,
    ) -> str:
        """Return a compact summary of known (non-novel) proofs for optional prompt injection.

        Unlike novel proof injection this is NOT automatically prepended to prompts.
        It is called on-demand so the system can review what standard results have
        already been Lean 4-verified before brainstorming, avoiding redundant work.

        Args:
            source_id: When provided, only proofs whose source_id matches are
                included (e.g. a brainstorm topic ID or paper ID).  Pass None to
                include all known proofs across the session.
            limit: Maximum number of proof entries to include.  The most recent
                entries are selected.  Lean 4 code is intentionally omitted to
                keep the block compact.

        Returns:
            A formatted string block, or an empty string when no known proofs exist.
        """
        self._ensure_index_loaded_sync()
        proofs = self._index_data.get("proofs", []) if self._index_data else []
        known_proofs = [
            p for p in proofs
            if not p.get("novel") or is_duplicate_novel_tier(p.get("novelty_tier", ""))
        ]

        if source_id:
            known_proofs = [p for p in known_proofs if p.get("source_id") == source_id]

        if not known_proofs:
            return ""

        total = len(known_proofs)
        # Most-recent first (index is already sorted newest-first by add_proof)
        shown = known_proofs[:limit]

        lines = [
            f"=== KNOWN VERIFIED PROOFS ({len(shown)} of {total} shown, Lean 4 Verified) ===",
            "[Standard/known results already formally verified. For reference to avoid re-proving.]",
            "",
        ]
        for index, proof in enumerate(shown, start=1):
            statement = proof.get("theorem_statement", "").strip()
            src_type = proof.get("source_type", "")
            src_id = proof.get("source_id", "")
            proof_id = proof.get("proof_id", "")
            lines.append(
                f"KNOWN {index}: {statement}"
                f"  (source: {src_type} {src_id}, id: {proof_id})".rstrip()
            )
        lines.append("")
        lines.append("=== END KNOWN PROOFS ===")
        return "\n".join(lines)

    def get_novel_proofs_for_injection(self) -> str:
        """Format the novel proofs block for highest-priority prompt injection."""
        self._ensure_index_loaded_sync()
        proofs = self._index_data.get("proofs", []) if self._index_data else []
        novel_proofs = [
            proof for proof in proofs
            if proof.get("novel") and (
                is_prompt_injection_novel_tier(proof.get("novelty_tier", ""))
                or not str(proof.get("novelty_tier") or "").strip()
            )
        ]

        if not novel_proofs:
            return ""

        lines = [
            "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ===",
            "[These proofs have been formally verified. They represent proven mathematical truths.",
            "Novelty tiers: Major Mathematical Discovery (highest — possible prize-level discovery), Mathematical Discovery (new result), Novel Reformulation (novel reformulation of known proof), Novel Formalization (citable formulation/formalization absent from standard references or Mathlib).]",
            "",
        ]
        for index, proof in enumerate(novel_proofs, start=1):
            tier = proof.get("novelty_tier", "")
            tier_label = {
                "major_mathematical_discovery": "Major Mathematical Discovery",
                "mathematical_discovery": "Mathematical Discovery",
                "novel_variant": "Novel Reformulation",
                "novel_formulation": "Novel Formalization",
            }.get(tier, "Novel")
            lines.extend(
                [
                    f"PROOF {index} [{tier_label}]: {proof.get('theorem_statement', '').strip()}",
                    f"Source: {proof.get('source_type', '')} {proof.get('source_id', '')}".strip(),
                    "Lean 4 Code:",
                    proof.get("lean_code", "").strip(),
                    "---",
                ]
            )
        lines.append("=== END VERIFIED PROOFS ===")
        return "\n".join(lines)

    def inject_into_prompt(self, prompt: str) -> str:
        """Prepend the verified novel proofs block when available."""
        proofs_block = self.get_novel_proofs_for_injection()
        if not proofs_block:
            return prompt
        if "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ===" in prompt:
            return prompt
        if not prompt:
            return proofs_block
        return f"{proofs_block}\n\n{prompt}"

    async def inject_failure_hints_into_prompt(
        self,
        prompt: str,
        source_brainstorm_id: str,
        *,
        limit: int = 5,
    ) -> str:
        """Prepend recent failed proof targets for the active brainstorm when available."""
        if not source_brainstorm_id:
            return prompt

        hints = await self.get_recent_failure_hints(source_brainstorm_id, limit=limit)
        hints_block = format_failure_hints_for_injection(hints)
        if not hints_block:
            return prompt
        if "=== OPEN PROOF TARGETS LEAN 4 COULD NOT YET CLOSE ===" in prompt:
            return prompt
        if not prompt:
            return hints_block
        return f"{hints_block}\n\n{prompt}"

    async def list_proof_library(
        self,
        novel_only: Optional[bool] = True,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all proofs across all sessions (legacy + session-based) for the proof library.

        Mirrors the cross-session listing pattern used by PaperLibrary.list_history_papers().
        """
        normalized_category = normalize_proof_library_category(category, novel_only)
        all_proofs: List[Dict[str, Any]] = []

        legacy_proofs_dir = Path(system_config.data_dir) / "proofs"
        if legacy_proofs_dir.exists():
            all_proofs.extend(
                await self._list_proofs_from_directory(legacy_proofs_dir, "legacy", normalized_category)
            )

        sessions_dir = Path(system_config.auto_sessions_base_dir)
        if sessions_dir.exists():
            for session_dir in sorted(
                (p for p in sessions_dir.iterdir() if p.is_dir()), reverse=True
            ):
                proofs_dir = session_dir / "proofs"
                if not proofs_dir.exists():
                    continue
                all_proofs.extend(
                    await self._list_proofs_from_directory(proofs_dir, session_dir.name, normalized_category)
                )

        all_proofs.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        return all_proofs

    async def archive_current_run(
        self,
        history_root: Path,
        *,
        user_prompt: str = "",
        reason: str = "manual_run_cleared",
    ) -> Optional[Dict[str, Any]]:
        """Archive the active fixed proof directory, then reset it to an empty run.

        This is used by manual mode: archived proofs remain browsable/downloadable,
        but the active proof database becomes empty so future manual prompts cannot
        inherit proofs from a cleared run.
        """
        async with self._lock:
            self._ensure_index_loaded_sync()
            proof_count = len(self._index_data.get("proofs", []) if self._index_data else [])
            has_files = self._base_dir.exists() and any(self._base_dir.iterdir())
            failed_dir = self._get_failed_dir()
            has_failed_state = failed_dir.exists() and any(failed_dir.iterdir())
            if not has_files or (proof_count == 0 and not has_failed_state):
                if self._base_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, self._base_dir, True)
                self._index_data = self._default_index()
                self._base_dir.mkdir(parents=True, exist_ok=True)
                self._get_failed_dir().mkdir(parents=True, exist_ok=True)
                self._rebuild_reverse_indexes()
                await self._save_index()
                return None

            timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
            history_root = Path(history_root)
            history_root.mkdir(parents=True, exist_ok=True)

            active_run_id = str(self._index_data.get("active_run_id") or "").strip()
            base_run_id = active_run_id or f"manual-{uuid.uuid4().hex}"
            run_id = base_run_id
            suffix = 2
            while (history_root / run_id).exists():
                run_id = f"{base_run_id}_{suffix}"
                suffix += 1

            run_dir = history_root / run_id
            target_proofs_dir = run_dir / "proofs"

            def _copy_active_run() -> None:
                run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(self._base_dir, target_proofs_dir)
                if proof_count:
                    shutil.rmtree(target_proofs_dir / "failed", ignore_errors=True)

            await asyncio.to_thread(_copy_active_run)

            metadata = {
                "session_id": run_id,
                "run_type": "manual",
                "status": "cleared",
                "reason": reason,
                "user_prompt": (user_prompt or "").strip(),
                "created_at": timestamp,
                "archived_at": datetime.utcnow().isoformat(),
                "proof_count": proof_count,
                "has_failed_state": has_failed_state,
            }
            metadata_path = run_dir / "session_metadata.json"
            await asyncio.to_thread(
                metadata_path.write_text,
                json.dumps(metadata, indent=2),
                "utf-8",
            )

            await asyncio.to_thread(shutil.rmtree, self._base_dir, True)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._get_failed_dir().mkdir(parents=True, exist_ok=True)
            self._index_data = self._default_index()
            self._rebuild_reverse_indexes()
            await self._save_index()
            return metadata

    async def list_proof_library_from_history(
        self,
        history_root: Path,
        novel_only: Optional[bool] = True,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List archived manual proof runs without including the active DB."""
        normalized_category = normalize_proof_library_category(category, novel_only)
        history_root = Path(history_root)
        all_proofs: List[Dict[str, Any]] = []
        if history_root.exists():
            for run_dir in sorted((p for p in history_root.iterdir() if p.is_dir()), reverse=True):
                proofs_dir = run_dir / "proofs"
                if not proofs_dir.exists():
                    continue
                all_proofs.extend(
                    await self._list_proofs_from_directory(proofs_dir, run_dir.name, normalized_category)
                )
        all_proofs.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        return all_proofs

    async def _list_proofs_from_directory(
        self, proofs_dir: Path, session_id: str, category: str
    ) -> List[Dict[str, Any]]:
        """Read the proofs index from a specific directory and return library entries."""
        index_path = proofs_dir / "proofs_index.json"
        if not index_path.exists():
            return []

        try:
            async with aiofiles.open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.loads(await handle.read())
        except Exception as exc:
            logger.warning("Failed to read proofs index at %s: %s", index_path, exc)
            return []

        session_metadata_path = proofs_dir.parent / "session_metadata.json"
        session_user_prompt = ""
        session_run_id = session_id
        if session_metadata_path.exists():
            try:
                async with aiofiles.open(session_metadata_path, "r", encoding="utf-8") as handle:
                    meta = json.loads(await handle.read())
                    session_user_prompt = str(meta.get("user_prompt", "") or "").strip()
                    session_run_id = str(
                        meta.get("run_id") or meta.get("session_id") or session_id
                    ).strip()
            except Exception as exc:
                logger.debug("Failed to read proof library session metadata at %s: %s", session_metadata_path, exc)

        results: List[Dict[str, Any]] = []
        for proof_data in index_data.get("proofs", []):
            is_novel = proof_data.get("novel", False)
            if not proof_matches_library_category(proof_data, category):
                continue
            run_id = str(proof_data.get("run_id") or session_run_id or session_id).strip()
            user_prompt = str(
                proof_data.get("user_prompt") or session_user_prompt or proof_data.get("source_title") or ""
            ).strip()

            results.append({
                "library_id": f"{session_id}:{proof_data.get('proof_id', '')}",
                "session_id": session_id,
                "proof_id": proof_data.get("proof_id", ""),
                "theorem_name": proof_data.get("theorem_name", ""),
                "theorem_statement": proof_data.get("theorem_statement", ""),
                "formal_sketch": proof_data.get("formal_sketch", ""),
                "source_type": proof_data.get("source_type", ""),
                "source_id": proof_data.get("source_id", ""),
                "source_title": proof_data.get("source_title", ""),
                "run_id": run_id,
                "solver": proof_data.get("solver", "Lean 4"),
                "novel": is_novel,
                "novelty_tier": proof_data.get("novelty_tier", "not_novel"),
                "novelty_reasoning": proof_data.get("novelty_reasoning", ""),
                "artifact_purpose": proof_data.get(
                    "artifact_purpose", "verified_occurrence"
                ),
                "verification_notes": proof_data.get("verification_notes", ""),
                "attempt_count": proof_data.get("attempt_count", 0),
                "created_at": proof_data.get("created_at", ""),
                "user_prompt": user_prompt,
                "dependencies": proof_data.get("dependencies", []),
            })

        return results

    async def get_library_proof(self, session_id: str, proof_id: str) -> Optional[Dict[str, Any]]:
        """Get a single proof from a specific session for the proof library viewer."""
        if session_id == "legacy":
            proofs_dir = Path(system_config.data_dir) / "proofs"
        else:
            safe_session = validate_single_path_component(session_id, "session ID")
            proofs_dir = resolve_path_within_root(
                Path(system_config.auto_sessions_base_dir), safe_session, "proofs"
            )

        if not proofs_dir.exists():
            return None

        return await self.get_library_proof_from_directory(proofs_dir, session_id, proof_id)

    async def get_library_proof_from_history(
        self,
        history_root: Path,
        session_id: str,
        proof_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get one archived manual proof by run id and proof id."""
        safe_session = validate_single_path_component(session_id, "manual proof run ID")
        proofs_dir = resolve_path_within_root(Path(history_root), safe_session, "proofs")
        if not proofs_dir.exists():
            return None
        return await self.get_library_proof_from_directory(proofs_dir, safe_session, proof_id)

    async def get_library_proof_from_directory(
        self,
        proofs_dir: Path,
        session_id: str,
        proof_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read a proof record from an explicit proofs directory."""
        safe_id = validate_single_path_component(proof_id, "proof ID")
        record_path = resolve_path_within_root(proofs_dir, f"proof_{safe_id}.json")
        lean_path = resolve_path_within_root(proofs_dir, f"proof_{safe_id}_lean.lean")

        if not record_path.exists():
            return None

        try:
            async with aiofiles.open(str(record_path), "r", encoding="utf-8") as handle:
                proof_data = json.loads(await handle.read())
        except Exception as exc:
            logger.error(
                "Failed to read proof %s from session %s: %s",
                redact_log_text(proof_id, 120),
                redact_log_text(session_id, 160),
                redact_log_text(exc, 240),
            )
            return None

        lean_code = ""
        if lean_path.exists():
            try:
                async with aiofiles.open(str(lean_path), "r", encoding="utf-8") as handle:
                    lean_code = await handle.read()
            except Exception as exc:
                logger.debug("Failed to read Lean source %s; using embedded proof record code: %s", lean_path, exc)
                lean_code = str(proof_data.get("lean_code", "") or "")
        else:
            lean_code = str(proof_data.get("lean_code", "") or "")

        session_user_prompt = ""
        session_run_id = session_id
        metadata_path = proofs_dir.parent / "session_metadata.json"
        if metadata_path.exists():
            try:
                async with aiofiles.open(str(metadata_path), "r", encoding="utf-8") as handle:
                    metadata = json.loads(await handle.read())
                session_user_prompt = str(metadata.get("user_prompt", "") or "").strip()
                session_run_id = str(
                    metadata.get("run_id") or metadata.get("session_id") or session_id
                ).strip()
            except Exception as exc:
                logger.debug("Failed to read proof detail session metadata at %s: %s", metadata_path, exc)

        return {
            "library_id": f"{session_id}:{proof_id}",
            "session_id": session_id,
            **proof_data,
            "run_id": str(proof_data.get("run_id") or session_run_id or session_id).strip(),
            "user_prompt": str(
                proof_data.get("user_prompt")
                or session_user_prompt
                or proof_data.get("source_title")
                or ""
            ).strip(),
            "lean_code": lean_code,
        }

    async def clear_all(self) -> None:
        """Remove all proof files and reset the index."""
        async with self._lock:
            if self._base_dir.exists():
                shutil.rmtree(self._base_dir, ignore_errors=True)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._index_data = self._default_index()
            self._rebuild_reverse_indexes()
            await self._save_index()


proof_database = ProofDatabase()
manual_proof_database = ProofDatabase()
manual_proof_database.set_base_dir(Path(system_config.data_dir) / "manual_proofs")
