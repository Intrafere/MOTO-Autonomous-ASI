"""
Proof database for Lean 4 verified results.

Stores both novel and non-novel verified proofs centrally for UI/API access.
Novel proofs are also formatted for highest-priority direct prompt injection.
"""
import asyncio
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiofiles

from backend.shared.config import system_config
from backend.shared.models import FailedProofCandidate, ProofCandidate, ProofRecord
from backend.shared.path_safety import validate_single_path_component
from backend.autonomous.prompts.proof_prompts import format_failure_hints_for_injection

logger = logging.getLogger(__name__)


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
        self._session_manager = None
        self._index_data: Optional[Dict[str, Any]] = None
        self._mathlib_reverse_index: Dict[str, List[str]] = {}
        self._mathlib_reverse_short_index: Dict[str, List[str]] = {}

    def set_session_manager(self, session_manager) -> None:
        """Switch storage to the active session directory when available."""
        self._session_manager = session_manager
        if session_manager and session_manager.is_session_active:
            self._base_dir = session_manager.get_proofs_dir()
        else:
            self._base_dir = Path(system_config.data_dir) / "proofs"
        self._index_data = None
        logger.info("Proof database using path: %s", self._base_dir)

    def _safe_proof_id(self, proof_id: str) -> str:
        return validate_single_path_component(proof_id, "proof ID")

    def _get_index_path(self) -> Path:
        return self._base_dir / "proofs_index.json"

    def _get_record_path(self, proof_id: str) -> Path:
        return self._base_dir / f"proof_{self._safe_proof_id(proof_id)}.json"

    def _get_lean_path(self, proof_id: str) -> Path:
        return self._base_dir / f"proof_{self._safe_proof_id(proof_id)}_lean.lean"

    def _get_failed_dir(self) -> Path:
        return self._base_dir / "failed"

    def _get_failed_candidates_path(self, source_brainstorm_id: str) -> Path:
        safe_id = validate_single_path_component(source_brainstorm_id, "brainstorm ID")
        return self._get_failed_dir() / f"{safe_id}.json"

    def _default_index(self) -> Dict[str, Any]:
        return {
            "next_proof_id": 1,
            "proofs": [],
        }

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

    async def initialize(self) -> None:
        """Ensure storage exists and load the index."""
        if self._session_manager and self._session_manager.is_session_active:
            self._base_dir = self._session_manager.get_proofs_dir()

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
                self._index_data = self._default_index()
                await self._save_index()
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
                self._index_data = self._default_index()
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

    async def add_proof(self, record: ProofRecord) -> ProofRecord:
        """Persist a proof record and return the stored copy."""
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
            return stored_record

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
                    logger.error("Failed to read Lean file for %s: %s", proof_id, exc)

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
            return [proof for proof in proofs if proof.novel is novel_only]

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
                    logger.error("Failed to read proof %s: %s", proof_id, exc)

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
        novel_count = sum(1 for proof in proofs if proof.get("novel"))
        return {
            "total": len(proofs),
            "novel": novel_count,
            "known": len(proofs) - novel_count,
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
        known_proofs = [p for p in proofs if not p.get("novel")]

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
        novel_proofs = [proof for proof in proofs if proof.get("novel")]

        if not novel_proofs:
            return ""

        lines = [
            "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ===",
            "[These proofs have been formally verified. They represent proven mathematical truths.]",
            "",
        ]
        for index, proof in enumerate(novel_proofs, start=1):
            lines.extend(
                [
                    f"PROOF {index}: {proof.get('theorem_statement', '').strip()}",
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
        if "=== OPEN LEMMA TARGETS LEAN 4 COULD NOT YET CLOSE ===" in prompt:
            return prompt
        if not prompt:
            return hints_block
        return f"{hints_block}\n\n{prompt}"

    async def list_proof_library(self, novel_only: bool = True) -> List[Dict[str, Any]]:
        """List all proofs across all sessions (legacy + session-based) for the proof library.

        Mirrors the cross-session listing pattern used by PaperLibrary.list_history_papers().
        """
        all_proofs: List[Dict[str, Any]] = []

        legacy_proofs_dir = Path(system_config.data_dir) / "proofs"
        if legacy_proofs_dir.exists():
            all_proofs.extend(
                await self._list_proofs_from_directory(legacy_proofs_dir, "legacy", novel_only)
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
                    await self._list_proofs_from_directory(proofs_dir, session_dir.name, novel_only)
                )

        all_proofs.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        return all_proofs

    async def _list_proofs_from_directory(
        self, proofs_dir: Path, session_id: str, novel_only: bool
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
        user_prompt = ""
        if session_metadata_path.exists():
            try:
                async with aiofiles.open(session_metadata_path, "r", encoding="utf-8") as handle:
                    meta = json.loads(await handle.read())
                    user_prompt = meta.get("user_prompt", "")
            except Exception:
                pass

        results: List[Dict[str, Any]] = []
        for proof_data in index_data.get("proofs", []):
            is_novel = proof_data.get("novel", False)
            if novel_only and not is_novel:
                continue

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
                "solver": proof_data.get("solver", "Lean 4"),
                "novel": is_novel,
                "novelty_reasoning": proof_data.get("novelty_reasoning", ""),
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
            proofs_dir = Path(system_config.auto_sessions_base_dir) / validate_single_path_component(session_id, "session ID") / "proofs"

        if not proofs_dir.exists():
            return None

        safe_id = validate_single_path_component(proof_id, "proof ID")
        record_path = proofs_dir / f"proof_{safe_id}.json"
        lean_path = proofs_dir / f"proof_{safe_id}_lean.lean"

        if not record_path.exists():
            return None

        try:
            async with aiofiles.open(record_path, "r", encoding="utf-8") as handle:
                proof_data = json.loads(await handle.read())
        except Exception as exc:
            logger.error("Failed to read proof %s from session %s: %s", proof_id, session_id, exc)
            return None

        lean_code = ""
        if lean_path.exists():
            try:
                async with aiofiles.open(lean_path, "r", encoding="utf-8") as handle:
                    lean_code = await handle.read()
            except Exception:
                lean_code = str(proof_data.get("lean_code", "") or "")
        else:
            lean_code = str(proof_data.get("lean_code", "") or "")

        return {
            "library_id": f"{session_id}:{proof_id}",
            "session_id": session_id,
            **proof_data,
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
