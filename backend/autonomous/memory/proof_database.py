"""
Proof database for Lean 4 verified results.

Stores both novel and non-novel verified proofs centrally for UI/API access.
Novel proofs are also formatted for highest-priority direct prompt injection.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Mapping, Optional, Union

import aiofiles

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.models import (
    FailedProofCandidate,
    ProofCandidate,
    ProofPruneAggregateEntry,
    ProofPruneCommitIntent,
    ProofPruneContextPressure,
    ProofPruneProofDescriptor,
    ProofPruneSnapshot,
    ProofRecord,
)
from backend.shared.path_safety import resolve_filename_within_root, validate_single_path_component
from backend.shared.proof_identity import canonical_proof_identity
from backend.autonomous.proof_pruning_evidence import (
    EVIDENCE_POLICY_VERSION,
    dependency_fingerprint,
    descriptor_fingerprint,
    estimated_context_tokens,
    evidence_fingerprint,
    role_in_objective,
)
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


ProofLike = Union[ProofRecord, Mapping[str, Any]]


def _proof_live_context_value(proof: ProofLike, field: str, default: Any = None) -> Any:
    if isinstance(proof, Mapping):
        return proof.get(field, default)
    return getattr(proof, field, default)


def is_live_context_pruned(proof: ProofLike, requesting_run_id: str) -> bool:
    """Return whether an occurrence is unavailable to this run's model context.

    Missing legacy status is active. Unknown/malformed state fails closed for
    model context, while canonical and human-facing callers remain unfiltered.
    A valid prune is local to its owning run and therefore remains available to
    future runs.
    """
    raw_status = _proof_live_context_value(proof, "live_context_status", None)
    if raw_status is None or str(raw_status).strip() == "":
        return False
    status = str(raw_status).strip().lower()
    if status == "active":
        return False
    if status != "pruned":
        return True

    owner_run_id = str(
        _proof_live_context_value(proof, "live_context_owner_run_id", "") or ""
    ).strip()
    requester = str(requesting_run_id or "").strip()
    if not owner_run_id or not requester:
        return True
    return owner_run_id == requester


def is_live_context_active(proof: ProofLike, requesting_run_id: str) -> bool:
    """Return whether an occurrence may enter the requesting run's context."""
    return not is_live_context_pruned(proof, requesting_run_id)


def filter_live_context_records(
    proofs: List[ProofLike],
    requesting_run_id: str,
) -> List[ProofLike]:
    """Filter records for model use without mutating canonical occurrences."""
    return [
        proof
        for proof in proofs
        if is_live_context_active(proof, requesting_run_id)
    ]


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

    def _resolve_storage_path(self, filename: str) -> Path:
        """Resolve one generated filename beneath the active proof-store root."""
        self._refresh_runtime_root()
        return resolve_filename_within_root(
            self._base_dir,
            filename,
            "proof storage filename",
        )

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

    def _get_revision_path(self) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / "proof_set_revision.json"

    def _get_record_path(self, proof_id: str) -> Path:
        safe_id = self._safe_proof_id(proof_id)
        return self._resolve_storage_path(f"proof_{safe_id}.json")

    def _get_lean_path(self, proof_id: str) -> Path:
        safe_id = self._safe_proof_id(proof_id)
        return self._resolve_storage_path(f"proof_{safe_id}_lean.lean")

    def _get_failed_dir(self) -> Path:
        self._refresh_runtime_root()
        return self._base_dir / "failed"

    def _get_failed_candidates_path(self, source_brainstorm_id: str) -> Path:
        safe_id = validate_single_path_component(source_brainstorm_id, "brainstorm ID")
        failed_dir = self._get_failed_dir()
        return resolve_filename_within_root(
            failed_dir,
            f"{safe_id}.json",
            "failed candidate filename",
        )

    def _default_index(self) -> Dict[str, Any]:
        return {
            "next_proof_id": 1,
            "proof_set_revision": 0,
            "proofs": [],
        }

    @staticmethod
    def _atomic_write_text_sync(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    async def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        await asyncio.to_thread(
            self._atomic_write_text_sync,
            path,
            json.dumps(payload, indent=2),
        )

    def _load_durable_revision_sync(self) -> int:
        path = self._get_revision_path()
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return max(0, int(payload.get("proof_set_revision", 0)))
        except Exception as exc:
            logger.warning("Failed to load proof-set revision at %s: %s", path, exc)
            return 0

    def _save_durable_revision_sync(self, revision: int) -> None:
        self._atomic_write_text_sync(
            self._get_revision_path(),
            json.dumps({"proof_set_revision": max(0, int(revision))}, indent=2),
        )

    async def _save_durable_revision(self) -> None:
        await self._atomic_write_json(
            self._get_revision_path(),
            {"proof_set_revision": int(self._index_data.get("proof_set_revision", 0))},
        )

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
            "proof_set_revision": self._load_durable_revision_sync(),
            "proofs": proofs,
        }

    def _reconcile_index_from_record_files_sync(self) -> bool:
        """Refresh the derived index when authoritative record files differ."""
        if not any(self._base_dir.glob("proof_*.json")):
            # Legacy stores may contain only the embedded index. Preserve them
            # until an individual record file is first published.
            return False
        rebuilt = self._rebuild_index_from_record_files_sync()
        current_proofs = self._index_data.get("proofs", []) if self._index_data else []
        if current_proofs == rebuilt["proofs"]:
            return False
        metadata = {
            key: value
            for key, value in (self._index_data or {}).items()
            if key not in {"proofs", "next_proof_id", "proof_set_revision"}
        }
        self._index_data = {
            **metadata,
            **rebuilt,
            "proof_set_revision": max(
                int((self._index_data or {}).get("proof_set_revision", 0)),
                int(rebuilt.get("proof_set_revision", 0)),
            ) + 1,
        }
        self._save_durable_revision_sync(self._index_data["proof_set_revision"])
        return True

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
        self._index_data["proof_set_revision"] = max(
            int(self._index_data.get("proof_set_revision", 0)),
            await asyncio.to_thread(self._load_durable_revision_sync),
        )
        reconciled = await asyncio.to_thread(self._reconcile_index_from_record_files_sync)
        self._rebuild_reverse_indexes()
        if reconciled:
            await self._save_index()

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
        self._index_data["proof_set_revision"] = max(
            int(self._index_data.get("proof_set_revision", 0)),
            self._load_durable_revision_sync(),
        )
        if self._reconcile_index_from_record_files_sync():
            self._atomic_write_text_sync(
                self._get_index_path(),
                json.dumps(self._index_data, indent=2),
            )
        self._rebuild_reverse_indexes()

    async def _save_index(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        await self._atomic_write_json(self._get_index_path(), self._index_data)

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

    async def _persist_record_files(
        self,
        stored_record: ProofRecord,
        serialized: Dict[str, Any],
    ) -> None:
        """Persist one proof's Lean source before publishing its metadata record."""
        record_path = self._get_record_path(stored_record.proof_id)
        lean_path = self._get_lean_path(stored_record.proof_id)
        lean_existed = await asyncio.to_thread(lean_path.exists)
        await asyncio.to_thread(self._atomic_write_text_sync, lean_path, stored_record.lean_code)
        try:
            await self._atomic_write_json(record_path, serialized)
        except Exception:
            if not lean_existed:
                try:
                    await asyncio.to_thread(lean_path.unlink, True)
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to roll back unpublished Lean proof source %s: %s",
                        lean_path,
                        cleanup_exc,
                    )
            raise

    async def _increment_proof_set_revision(self) -> int:
        revision = int(self._index_data.get("proof_set_revision", 0)) + 1
        self._index_data["proof_set_revision"] = revision
        await self._save_durable_revision()
        return revision

    async def add_proof_occurrence(self, record: ProofRecord) -> ProofRecord:
        """Persist a full record for each newly verified current-run occurrence."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()

            proof_id = record.proof_id or f"proof_{self._index_data['next_proof_id']:03d}"
            stored_record = record.model_copy(update={"proof_id": proof_id})
            serialized = self._serialize_record(stored_record)
            await self._persist_record_files(stored_record, serialized)

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
            await self._increment_proof_set_revision()
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
            await self._persist_record_files(stored_record, serialized)

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
            await self._increment_proof_set_revision()
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

    async def get_all_proofs_for_live_context(
        self,
        requesting_run_id: str,
        novel_only: Optional[bool] = None,
    ) -> List[ProofRecord]:
        """Return model-visible proofs for one owning run.

        This deliberately wraps, rather than changes, canonical enumeration.
        Registration, graph, archive, export, and human-view code must continue
        to use ``get_all_proofs``.
        """
        proofs = await self.get_all_proofs(novel_only=novel_only)
        return [
            proof
            for proof in proofs
            if is_live_context_active(proof, requesting_run_id)
        ]

    async def capture_pruning_snapshot(
        self,
        *,
        proof_store_id: str,
        owning_run_id: str,
        proof_run_id: str,
        proof_run_lifecycle_generation: int,
        owning_lifecycle_generation: Optional[int] = None,
        scope: str,
        source_type: str,
        source_id: str,
        canonical_user_prompt: str,
        session_id: str = "",
        trigger_reasons: Optional[List[str]] = None,
        accepted_prompt_novel_total: int = 0,
        context_pressure: Optional[ProofPruneContextPressure] = None,
    ) -> ProofPruneSnapshot:
        """Atomically capture a deterministic, non-mutating pruning snapshot.

        Every active occurrence is represented in ``whole_set``. Detailed
        descriptors are bounded deterministically to dependency-safe semantic
        review candidates. This method never calls a model and never changes
        live-context state.
        """
        normalized_run_id = str(owning_run_id or "").strip()
        normalized_owning_generation = int(
            owning_lifecycle_generation or proof_run_lifecycle_generation
        )
        normalized_prompt = str(canonical_user_prompt or "")
        if not normalized_run_id:
            raise ValueError("A stable owning run ID is required.")
        if not normalized_prompt.strip():
            raise ValueError("The canonical user prompt is required.")
        async with self._lock:
            if self._index_data is None:
                await self._load_index()
            revision = int(self._index_data.get("proof_set_revision", 0))
            records = [
                self._deserialize_record(data)
                for data in self._index_data.get("proofs", [])
            ]
            active_records = [
                record
                for record in records
                if is_live_context_active(record, normalized_run_id)
            ]

        records_by_id = {record.proof_id: record for record in active_records}
        canonical_identity_by_id = {
            record.proof_id: canonical_proof_identity(
                record.theorem_statement,
                record.lean_code,
            )
            for record in active_records
        }
        dependent_counts: Dict[str, int] = {
            record.proof_id: 0 for record in active_records
        }
        for record in active_records:
            for dependency in record.dependencies:
                if dependency.kind != "moto":
                    continue
                referenced_id = str(dependency.source_ref or dependency.name or "").strip()
                if referenced_id in dependent_counts:
                    dependent_counts[referenced_id] += 1

        protected_by_id: Dict[str, List[str]] = {}
        eligible_ids: List[str] = []
        aggregate: List[ProofPruneAggregateEntry] = []
        protected_support_ids = {
            support_id
            for record in records
            if record.live_context_status == "pruned"
            and record.live_context_owner_run_id == normalized_run_id
            for support_id in record.live_context_prune_supporting_proof_ids
        }

        for record in sorted(active_records, key=lambda item: item.proof_id):
            canonical_identity = canonical_identity_by_id[record.proof_id]
            theorem_hash = str(
                record.canonical_theorem_statement_hash
                or canonical_identity.theorem_statement_hash
            )
            lean_hash = str(
                record.canonical_lean_code_hash
                or canonical_identity.lean_code_hash
            )
            protected_reasons: List[str] = []
            if record.dependency_extraction_status != "complete":
                protected_reasons.append("dependency_extraction_incomplete")
            if dependent_counts.get(record.proof_id, 0) > 0:
                protected_reasons.append("active_dependency_root")
            if record.proof_id in protected_support_ids:
                protected_reasons.append("retained_prune_support")
            eligible = not protected_reasons
            protected_by_id[record.proof_id] = protected_reasons
            if eligible:
                eligible_ids.append(record.proof_id)

            aggregate.append(
                ProofPruneAggregateEntry(
                    proof_id=record.proof_id,
                    theorem_name=record.theorem_name,
                    canonical_theorem_hash=theorem_hash,
                    canonical_lean_hash=lean_hash,
                    novelty_tier=record.novelty_tier,
                    independent_novelty_tier=record.independent_novelty_tier,
                    source_type=record.source_type,
                    source_id=record.source_id,
                    created_at=record.created_at,
                    dependency_extraction_status=record.dependency_extraction_status,
                    dependency_count=len(record.dependencies),
                    dependent_count=dependent_counts.get(record.proof_id, 0),
                    dependency_fingerprint=dependency_fingerprint(record),
                    descriptor_fingerprint=descriptor_fingerprint(record),
                    estimated_context_tokens=estimated_context_tokens(record),
                    protected_reasons=protected_reasons,
                    eligible_candidate=eligible,
                )
            )

        selected_candidate_ids = sorted(
            set(eligible_ids),
            key=lambda item: (
                -estimated_context_tokens(records_by_id[item]),
                records_by_id[item].created_at,
                item,
            ),
        )
        # Every active proof must be available as retained semantic evidence,
        # including protected dependency roots that cannot themselves be targets.
        for record in sorted(active_records, key=lambda item: item.proof_id):
            if record.proof_id not in selected_candidate_ids:
                selected_candidate_ids.append(record.proof_id)

        def descriptor(record: ProofRecord) -> ProofPruneProofDescriptor:
            canonical_identity = canonical_identity_by_id[record.proof_id]
            theorem_hash = str(
                record.canonical_theorem_statement_hash
                or canonical_identity.theorem_statement_hash
            )
            lean_hash = str(
                record.canonical_lean_code_hash or canonical_identity.lean_code_hash
            )
            return ProofPruneProofDescriptor(
                proof_id=record.proof_id,
                theorem_name=record.theorem_name,
                theorem_statement=record.theorem_statement,
                canonical_theorem_hash=theorem_hash,
                canonical_lean_hash=lean_hash,
                novelty_tier=record.novelty_tier,
                novelty_reasoning=record.novelty_reasoning,
                independent_novelty_tier=record.independent_novelty_tier,
                independent_novelty_reasoning=record.independent_novelty_reasoning,
                source_type=record.source_type,
                source_id=record.source_id,
                source_title=str(record.source_title or "")[:1000],
                created_at=record.created_at,
                dependencies=list(record.dependencies),
                dependency_extraction_status=record.dependency_extraction_status,
                dependency_fingerprint=dependency_fingerprint(record),
                descriptor_fingerprint=descriptor_fingerprint(record),
                protected_reasons=protected_by_id.get(record.proof_id, []),
                role_in_user_objective=role_in_objective(record),
                lean_code=record.lean_code,
                lean_code_included=bool(record.lean_code),
            )

        candidate_descriptors = [
            descriptor(records_by_id[proof_id])
            for proof_id in selected_candidate_ids
        ]
        evidence_digest = evidence_fingerprint(
            whole_set=[
                entry.model_dump(mode="json")
                for entry in aggregate
            ],
            descriptor_fingerprints=[
                item.descriptor_fingerprint for item in candidate_descriptors
            ],
        )
        identity_payload = {
            "proof_set_revision": revision,
            "proof_store_id": str(proof_store_id),
            "owning_run_id": normalized_run_id,
            "proof_run_id": str(proof_run_id),
            "proof_run_lifecycle_generation": proof_run_lifecycle_generation,
            "owning_lifecycle_generation": normalized_owning_generation,
            "proof_ids": [entry.proof_id for entry in aggregate],
            "theorem_hashes": [entry.canonical_theorem_hash for entry in aggregate],
            "lean_hashes": [entry.canonical_lean_hash for entry in aggregate],
            "evidence_policy_version": EVIDENCE_POLICY_VERSION,
            "evidence_fingerprint": evidence_digest,
        }
        snapshot_id = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ProofPruneSnapshot(
            snapshot_id=snapshot_id,
            proof_set_revision=revision,
            proof_store_id=str(proof_store_id),
            owning_run_id=normalized_run_id,
            proof_run_id=str(proof_run_id),
            proof_run_lifecycle_generation=proof_run_lifecycle_generation,
            owning_lifecycle_generation=normalized_owning_generation,
            scope=scope,
            source_type=source_type,
            source_id=str(source_id),
            session_id=str(session_id or ""),
            canonical_user_prompt=normalized_prompt,
            trigger_reasons=list(trigger_reasons or []),
            accepted_prompt_novel_total=accepted_prompt_novel_total,
            context_pressure=context_pressure or ProofPruneContextPressure(),
            whole_set=aggregate,
            candidate_descriptors=candidate_descriptors,
            evidence_bounded=False,
            evidence_policy_version=EVIDENCE_POLICY_VERSION,
            evidence_fingerprint=evidence_digest,
        )

    async def get_proof_set_revision(self) -> int:
        async with self._lock:
            if self._index_data is None:
                await self._load_index()
            return int(self._index_data.get("proof_set_revision", 0))

    async def set_live_context_status(
        self,
        *,
        proof_id: str,
        status: str,
        expected_run_id: str,
        expected_proof_set_revision: int,
        actor: str,
        reason: str,
        validator_reasoning: str = "",
        snapshot_revision: Optional[int] = None,
        trigger_reasons: Optional[List[str]] = None,
        expected_theorem_hash: str = "",
        expected_lean_hash: str = "",
    ) -> tuple[ProofRecord, int]:
        """Atomically update owning-run live-context state without touching Lean."""
        async with self._lock:
            if self._index_data is None:
                await self._load_index()
            current_revision = int(self._index_data.get("proof_set_revision", 0))
            if current_revision != int(expected_proof_set_revision):
                raise RuntimeError("Proof set changed; refresh and retry.")

            proof_index = next(
                (
                    index
                    for index, proof in enumerate(self._index_data.get("proofs", []))
                    if proof.get("proof_id") == proof_id
                ),
                None,
            )
            if proof_index is None:
                raise KeyError(proof_id)
            record = self._deserialize_record(self._index_data["proofs"][proof_index])
            normalized_run_id = str(record.run_id or f"legacy:{record.source_type}:{record.source_id}")
            if normalized_run_id != str(expected_run_id or "").strip():
                raise RuntimeError("Proof run changed; refresh and retry.")
            if expected_theorem_hash and expected_theorem_hash != record.canonical_theorem_statement_hash:
                raise RuntimeError("Proof theorem identity changed; refresh and retry.")
            if expected_lean_hash and expected_lean_hash != record.canonical_lean_code_hash:
                raise RuntimeError("Proof Lean identity changed; refresh and retry.")
            if status not in {"active", "pruned"}:
                raise ValueError("Live-context status must be active or pruned.")
            if actor not in {"user", "automatic_proof_pruning"}:
                raise ValueError("Unsupported live-context actor.")

            if status == record.live_context_status:
                if status == "active" or (
                    record.live_context_owner_run_id == normalized_run_id
                    and record.live_context_pruned_by == actor
                ):
                    return record, current_revision
            if (
                status == "active"
                and record.live_context_status == "pruned"
                and record.live_context_owner_run_id == normalized_run_id
                and record.live_context_pruned_by == "automatic_proof_pruning"
            ):
                raise RuntimeError("Validator-approved automatic pruning is immutable in its owning run.")

            if status == "active":
                updated = record.model_copy(
                    update={
                        "live_context_status": "active",
                        "live_context_owner_run_id": "",
                        "live_context_pruned_at": None,
                        "live_context_pruned_by": None,
                        "live_context_prune_reason": "",
                        "live_context_prune_validator_reasoning": "",
                        "live_context_prune_snapshot_revision": None,
                        "live_context_prune_trigger_reasons": [],
                        "live_context_prune_supporting_proof_ids": [],
                    }
                )
            else:
                bounded_reason = str(reason or "").strip()[:2000]
                if not bounded_reason:
                    raise ValueError("A non-empty prune reason is required.")
                updated = record.model_copy(
                    update={
                        "live_context_status": "pruned",
                        "live_context_owner_run_id": normalized_run_id,
                        "live_context_pruned_at": datetime.now(),
                        "live_context_pruned_by": actor,
                        "live_context_prune_reason": bounded_reason,
                        "live_context_prune_validator_reasoning": str(
                            validator_reasoning or ""
                        ).strip()[:4000],
                        "live_context_prune_snapshot_revision": snapshot_revision,
                        "live_context_prune_trigger_reasons": list(trigger_reasons or []),
                        "live_context_prune_supporting_proof_ids": [],
                    }
                )

            serialized = self._serialize_record(updated)
            previous_serialized = self._serialize_record(record)
            previous_revision = current_revision
            record_path = self._get_record_path(proof_id)
            await self._atomic_write_json(record_path, serialized)
            self._index_data["proofs"][proof_index] = serialized
            try:
                new_revision = await self._increment_proof_set_revision()
                self._rebuild_reverse_indexes()
                await self._save_index()
            except Exception:
                self._index_data["proofs"][proof_index] = previous_serialized
                self._index_data["proof_set_revision"] = previous_revision
                self._rebuild_reverse_indexes()
                await self._atomic_write_json(record_path, previous_serialized)
                await asyncio.to_thread(
                    self._save_durable_revision_sync,
                    previous_revision,
                )
                raise
            return updated, new_revision

    async def commit_pruning_intent(
        self,
        intent: ProofPruneCommitIntent,
        *,
        snapshot: ProofPruneSnapshot,
        expected_proof_store_id: str,
        expected_proof_run_id: str,
        expected_lifecycle_generation: int,
    ) -> tuple[ProofRecord, int]:
        """Commit one accepted prune after rechecking decision-relevant facts.

        Unrelated additions do not stale the decision. The target, its cited
        retained semantic supports, dependency/protection state, hashes, store,
        run and lifecycle must still match the immutable review snapshot.
        """
        if snapshot.proof_store_id != str(expected_proof_store_id):
            raise RuntimeError("Proof store changed; refresh and retry.")
        if snapshot.proof_run_id != str(expected_proof_run_id):
            raise RuntimeError("Proof run changed; refresh and retry.")
        if (
            snapshot.proof_run_lifecycle_generation
            != int(expected_lifecycle_generation)
        ):
            raise RuntimeError("Proof lifecycle changed; refresh and retry.")
        if snapshot.owning_lifecycle_generation != int(expected_lifecycle_generation):
            raise RuntimeError("Owning workflow lifecycle changed; refresh and retry.")
        if intent.snapshot_id != snapshot.snapshot_id:
            raise RuntimeError("Pruning snapshot identity changed; refresh and retry.")
        if intent.evidence_policy_version != snapshot.evidence_policy_version:
            raise RuntimeError("Pruning evidence policy changed; refresh and retry.")
        if intent.evidence_fingerprint != snapshot.evidence_fingerprint:
            raise RuntimeError("Pruning evidence changed; refresh and retry.")

        snapshot_entries = {entry.proof_id: entry for entry in snapshot.whole_set}
        snapshot_target = snapshot_entries.get(intent.proof_id)
        if snapshot_target is None or not snapshot_target.eligible_candidate:
            raise RuntimeError("Pruning target is no longer eligible.")
        descriptor = next(
            (
                item
                for item in snapshot.candidate_descriptors
                if item.proof_id == intent.proof_id
            ),
            None,
        )
        if descriptor is None:
            raise RuntimeError("Pruning target evidence is unavailable.")
        if descriptor.dependency_fingerprint != intent.target_dependency_fingerprint:
            raise RuntimeError("Pruning target dependency evidence changed.")
        if descriptor.descriptor_fingerprint != intent.target_descriptor_fingerprint:
            raise RuntimeError("Pruning target descriptor evidence changed.")

        async with self._lock:
            if self._index_data is None:
                await self._load_index()
            current_revision = int(self._index_data.get("proof_set_revision", 0))
            current_records = {
                str(data.get("proof_id", "")): self._deserialize_record(data)
                for data in self._index_data.get("proofs", [])
                if data.get("proof_id")
            }
            target = current_records.get(intent.proof_id)
            if target is None:
                raise KeyError(intent.proof_id)
            if not is_live_context_active(target, intent.owning_run_id):
                raise RuntimeError("Pruning target is no longer active.")
            target_identity = canonical_proof_identity(
                target.theorem_statement,
                target.lean_code,
            )
            theorem_hash = str(
                target.canonical_theorem_statement_hash
                or target_identity.theorem_statement_hash
            )
            lean_hash = str(
                target.canonical_lean_code_hash or target_identity.lean_code_hash
            )
            if theorem_hash != intent.expected_theorem_hash:
                raise RuntimeError("Pruning target theorem identity changed.")
            if lean_hash != intent.expected_lean_hash:
                raise RuntimeError("Pruning target Lean identity changed.")
            if target.dependency_extraction_status != "complete":
                raise RuntimeError("Pruning target dependency state is incomplete.")
            if dependency_fingerprint(target) != intent.target_dependency_fingerprint:
                raise RuntimeError("Pruning target dependencies changed.")
            if descriptor_fingerprint(target) != intent.target_descriptor_fingerprint:
                raise RuntimeError("Pruning target descriptor changed.")

            active_dependents = []
            for record in current_records.values():
                if not is_live_context_active(record, intent.owning_run_id):
                    continue
                for dependency in record.dependencies:
                    reference = str(
                        dependency.source_ref or dependency.name or ""
                    ).strip()
                    if dependency.kind == "moto" and reference == target.proof_id:
                        active_dependents.append(record.proof_id)
            if active_dependents:
                raise RuntimeError("Pruning target became an active dependency root.")

            if not intent.supporting_proof_ids:
                raise RuntimeError("Semantic pruning requires retained support.")
            for support_id in intent.supporting_proof_ids:
                support = current_records.get(support_id)
                if support is None or not is_live_context_active(
                    support, intent.owning_run_id
                ):
                    raise RuntimeError(
                        "A retained semantic support changed or is unavailable."
                    )
                if support.dependency_extraction_status != "complete":
                    raise RuntimeError(
                        "A retained semantic support has incomplete dependencies."
                    )
                expected_fingerprint = intent.supporting_proof_fingerprints.get(
                    support_id
                )
                if (
                    not expected_fingerprint
                    or descriptor_fingerprint(support) != expected_fingerprint
                ):
                    raise RuntimeError(
                        "A retained semantic support descriptor changed."
                    )
                if any(
                    dependency.kind == "moto"
                    and str(
                        dependency.source_ref or dependency.name or ""
                    ).strip()
                    == target.proof_id
                    for dependency in support.dependencies
                ):
                    raise RuntimeError(
                        "A retained semantic support depends on the pruning target."
                    )

            proof_index = next(
                index
                for index, data in enumerate(self._index_data.get("proofs", []))
                if data.get("proof_id") == target.proof_id
            )
            updated = target.model_copy(
                update={
                    "live_context_status": "pruned",
                    "live_context_owner_run_id": intent.owning_run_id,
                    "live_context_pruned_at": datetime.now(),
                    "live_context_pruned_by": "automatic_proof_pruning",
                    "live_context_prune_reason": intent.proposer_reasoning[:2000],
                    "live_context_prune_validator_reasoning": (
                        intent.validator_reasoning[:4000]
                    ),
                    "live_context_prune_snapshot_revision": (
                        intent.proof_set_revision
                    ),
                    "live_context_prune_trigger_reasons": list(
                        intent.trigger_reasons
                    ),
                    "live_context_prune_supporting_proof_ids": list(
                        intent.supporting_proof_ids
                    ),
                }
            )
            serialized = self._serialize_record(updated)
            previous_serialized = self._serialize_record(target)
            previous_revision = current_revision
            record_path = self._get_record_path(target.proof_id)
            await self._atomic_write_json(
                record_path,
                serialized,
            )
            self._index_data["proofs"][proof_index] = serialized
            try:
                new_revision = await self._increment_proof_set_revision()
                self._rebuild_reverse_indexes()
                await self._save_index()
            except Exception:
                self._index_data["proofs"][proof_index] = previous_serialized
                self._index_data["proof_set_revision"] = previous_revision
                self._rebuild_reverse_indexes()
                await self._atomic_write_json(record_path, previous_serialized)
                await asyncio.to_thread(
                    self._save_durable_revision_sync,
                    previous_revision,
                )
                raise
            logger.info(
                "Applied proof live-context prune %s from snapshot revision %s "
                "at current revision %s",
                target.proof_id,
                snapshot.proof_set_revision,
                current_revision,
            )
            return updated, new_revision

    async def update_proof_dependencies(
        self,
        proof_id: str,
        dependencies,
        *,
        extraction_status: str = "complete",
        extraction_detail: str = "",
    ) -> Optional[ProofRecord]:
        """Persist a new dependency list for an existing proof record."""
        if extraction_status not in {"not_attempted", "complete", "partial", "failed"}:
            raise ValueError("Unsupported dependency extraction status.")
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
                updated_record = record.model_copy(
                    update={
                        "dependencies": list(dependencies or []),
                        "dependency_extraction_status": extraction_status,
                        "dependency_extraction_detail": str(
                            extraction_detail or ""
                        ).strip()[:1000],
                        "dependency_extracted_at": datetime.now(),
                    }
                )
                updated_proofs.append(self._serialize_record(updated_record))

            if updated_record is None:
                return None

            self._index_data["proofs"] = updated_proofs
            self._rebuild_reverse_indexes()

            await self._atomic_write_json(
                self._get_record_path(proof_id),
                self._serialize_record(updated_record),
            )
            await self._increment_proof_set_revision()
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
                "live_context_status": proof.live_context_status,
                "live_context_owner_run_id": proof.live_context_owner_run_id,
                "live_context_pruned_at": (
                    proof.live_context_pruned_at.isoformat()
                    if proof.live_context_pruned_at
                    else None
                ),
                "live_context_pruned_by": proof.live_context_pruned_by,
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
        return self._count_loaded_proofs()

    def count_proofs_cached(self) -> Dict[str, int]:
        """Return counts from preloaded memory without touching proof storage."""
        if self._index_data is None:
            return {
                "total": 0,
                "novel": 0,
                "duplicate_novel": 0,
                "syntheticlib_novel": 0,
                "known": 0,
                "not_novel": 0,
                "live_context_active": 0,
                "live_context_pruned": 0,
            }
        return self._count_loaded_proofs()

    def _count_loaded_proofs(self) -> Dict[str, int]:
        """Count proofs in the already-loaded index projection."""
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
            "live_context_active": sum(
                1 for proof in proofs if proof.get("live_context_status", "active") != "pruned"
            ),
            "live_context_pruned": sum(
                1 for proof in proofs if proof.get("live_context_status", "active") == "pruned"
            ),
        }

    def get_known_proofs_summary_for_browsing(
        self,
        source_id: Optional[str] = None,
        limit: int = 15,
        requesting_run_id: str = "",
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
            if (
                not p.get("novel")
                or is_duplicate_novel_tier(p.get("novelty_tier", ""))
            )
            and is_live_context_active(p, requesting_run_id)
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

    def get_novel_proofs_for_injection(self, requesting_run_id: str = "") -> str:
        """Format the novel proofs block for highest-priority prompt injection."""
        self._ensure_index_loaded_sync()
        proofs = self._index_data.get("proofs", []) if self._index_data else []
        novel_proofs = [
            proof for proof in proofs
            if proof.get("novel") and (
                is_prompt_injection_novel_tier(proof.get("novelty_tier", ""))
                or not str(proof.get("novelty_tier") or "").strip()
            )
            and is_live_context_active(proof, requesting_run_id)
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

    def inject_into_prompt(self, prompt: str, requesting_run_id: str = "") -> str:
        """Prepend the verified novel proofs block when available."""
        proofs_block = self.get_novel_proofs_for_injection(requesting_run_id)
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
                "live_context_status": proof_data.get("live_context_status", "active"),
                "live_context_owner_run_id": proof_data.get("live_context_owner_run_id", ""),
                "live_context_pruned_at": proof_data.get("live_context_pruned_at"),
                "live_context_pruned_by": proof_data.get("live_context_pruned_by"),
                "live_context_prune_reason": proof_data.get("live_context_prune_reason", ""),
                "live_context_prune_validator_reasoning": proof_data.get(
                    "live_context_prune_validator_reasoning", ""
                ),
                "live_context_prune_snapshot_revision": proof_data.get(
                    "live_context_prune_snapshot_revision"
                ),
                "live_context_prune_trigger_reasons": proof_data.get(
                    "live_context_prune_trigger_reasons", []
                ),
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
