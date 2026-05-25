"""LeanOJ proof-memory persistence and direct/RAG context allocation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles

from backend.aggregator.core.rag_manager import rag_manager
from backend.shared.config import rag_config, system_config
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)


ARTIFACT_ACCEPTED_IDEAS = "accepted_ideas"
ARTIFACT_VERIFIED_SUBPROOFS = "verified_subproofs"
ARTIFACT_PARTIAL_PROOFS = "partial_proofs"
ARTIFACT_FINAL_ATTEMPTS = "final_attempts"
ARTIFACT_FINAL_CYCLE_PACKETS = "final_cycle_packets"
ARTIFACT_FAILED_SUBPROOFS = "failed_subproofs"


def _remove_attempt_count_language(value: Any) -> str:
    text = str(value or "")
    replacements = (
        (
            r"\bfailed\s+\d+\s+consecutive\s+verification/edit\s+attempts?\b",
            "encountered repeated verification/edit failures",
        ),
        (r"\bfailed\s+\d+\s+consecutive\s+attempts?\b", "encountered repeated failures"),
        (r"\bfailed\s+\d+\s+attempts?\b", "encountered repeated failures"),
        (r"\bfailed\s+\d+\s+times\b", "encountered repeated failures"),
        (r"\bafter\s+failed\s+attempts\b", "after recent proof-check failures"),
        (r"\bfailed\s+attempts\b", "proof-check failures"),
        (r"\battempts\s+\d+\s*-\s*\d+\b", "recent final-loop feedback"),
        (r"\bwith\s+exactly\s+\d+\s+failed\s+attempts?\b", "with recent proof-check failures"),
        (r"\bUse this exact failed-attempt count[^.]*\.", ""),
        (r"\bfailed-attempt count\b", "failure context"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()

USEFUL_ARTIFACTS = (
    ARTIFACT_ACCEPTED_IDEAS,
    ARTIFACT_VERIFIED_SUBPROOFS,
    ARTIFACT_PARTIAL_PROOFS,
    ARTIFACT_FINAL_ATTEMPTS,
    ARTIFACT_FINAL_CYCLE_PACKETS,
    ARTIFACT_FAILED_SUBPROOFS,
)


@dataclass
class LeanOJMemoryItem:
    """One optional proof-memory source eligible for direct injection or RAG."""

    artifact: str
    title: str
    text: str
    priority: int
    source_name: str
    rag_only: bool = False


@dataclass
class LeanOJContextAllocation:
    """Prepared context blocks consumed by LeanOJ prompt builders."""

    direct_proof_context: str = ""
    rag_evidence_context: str = ""
    refuted_construction_warnings: str = ""
    capped_rejection_feedback: str = ""
    current_final_cycle_packet: str = ""
    current_working_proof_attempt: str = ""
    direct_sources: list[str] = field(default_factory=list)
    rag_sources: list[str] = field(default_factory=list)

    def as_prompt_blocks(self) -> dict[str, str]:
        return {
            "direct_proof_context": self.direct_proof_context,
            "rag_evidence_context": self.rag_evidence_context,
            "refuted_construction_warnings": self.refuted_construction_warnings,
            "capped_rejection_feedback": self.capped_rejection_feedback,
            "current_final_cycle_packet": self.current_final_cycle_packet,
            "current_working_proof_attempt": self.current_working_proof_attempt,
        }


class LeanOJContextManager:
    """Session-scoped LeanOJ artifact storage and RAG/offload routing."""

    def __init__(self) -> None:
        self._indexed_hashes: dict[str, str] = {}
        self._index_locks: dict[str, asyncio.Lock] = {}
        self._artifact_sync_counts: dict[tuple[str, str], int] = {}
        self._artifact_sync_digests: dict[tuple[str, str], str] = {}

    @staticmethod
    def artifacts_base_dir() -> Path:
        return Path(system_config.data_dir) / "leanoj_artifacts"

    def session_artifact_dir(self, session_id: str) -> Path:
        return self.artifacts_base_dir() / (session_id or "latest")

    @staticmethod
    def source_prefix(session_id: str) -> str:
        return f"leanoj_{session_id or 'latest'}_"

    def source_name(self, session_id: str, artifact: str) -> str:
        return f"{self.source_prefix(session_id)}{artifact}"

    def source_names_for_session(self, session_id: str) -> list[str]:
        return [self.source_name(session_id, artifact) for artifact in USEFUL_ARTIFACTS]

    async def write_session_artifacts(
        self,
        *,
        session_id: str,
        accepted_ideas: list[str],
        accepted_idea_records: list[dict[str, Any]] | None = None,
        verified_subproofs: list[dict[str, Any]],
        partial_proofs: list[dict[str, Any]],
        failed_subproofs: list[dict[str, Any]],
        final_attempts: list[dict[str, Any]],
        final_cycle_packets: list[dict[str, Any]],
    ) -> None:
        """Persist full LeanOJ proof memory independently from trimmed UI state."""
        if not session_id:
            return

        base = self.session_artifact_dir(session_id)
        base.mkdir(parents=True, exist_ok=True)
        accepted_records = [
            dict(record)
            for record in (accepted_idea_records or [])
            if isinstance(record, dict) and str(record.get("content") or "").strip()
        ]
        recorded_contents = {str(record.get("content") or "") for record in accepted_records}
        accepted_records.extend(
            {"content": item}
            for item in accepted_ideas
            if str(item).strip() and str(item) not in recorded_contents
        )
        if not accepted_records:
            accepted_records = [{"content": item} for item in accepted_ideas]
        await self._sync_jsonl(base / f"{ARTIFACT_ACCEPTED_IDEAS}.jsonl", session_id, ARTIFACT_ACCEPTED_IDEAS, accepted_records)
        await self._sync_jsonl(base / f"{ARTIFACT_VERIFIED_SUBPROOFS}.jsonl", session_id, ARTIFACT_VERIFIED_SUBPROOFS, verified_subproofs)
        await self._sync_jsonl(base / f"{ARTIFACT_PARTIAL_PROOFS}.jsonl", session_id, ARTIFACT_PARTIAL_PROOFS, partial_proofs)
        await self._sync_jsonl(base / f"{ARTIFACT_FAILED_SUBPROOFS}.jsonl", session_id, ARTIFACT_FAILED_SUBPROOFS, failed_subproofs)
        await self._sync_jsonl(base / f"{ARTIFACT_FINAL_ATTEMPTS}.jsonl", session_id, ARTIFACT_FINAL_ATTEMPTS, final_attempts)
        await self._sync_jsonl(base / f"{ARTIFACT_FINAL_CYCLE_PACKETS}.jsonl", session_id, ARTIFACT_FINAL_CYCLE_PACKETS, final_cycle_packets)

    async def append_record(self, session_id: str, artifact: str, record: dict[str, Any]) -> None:
        """Append one record to a full-memory artifact log."""
        if not session_id:
            return
        path = self.session_artifact_dir(session_id) / f"{artifact}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(record, ensure_ascii=False) + "\n")
        key = (session_id, artifact)
        self._artifact_sync_counts[key] = self._artifact_sync_counts.get(key, self._count_jsonl_records(path) - 1) + 1
        self._artifact_sync_digests.pop(key, None)

    def load_session_artifacts(self, session_id: str) -> dict[str, list[Any]]:
        """Load full LeanOJ artifact logs for resume."""
        base = self.session_artifact_dir(session_id)
        return {
            ARTIFACT_ACCEPTED_IDEAS: self._records_to_strings(self._read_jsonl(base / f"{ARTIFACT_ACCEPTED_IDEAS}.jsonl")),
            "accepted_idea_records": self._read_jsonl(base / f"{ARTIFACT_ACCEPTED_IDEAS}.jsonl"),
            ARTIFACT_VERIFIED_SUBPROOFS: self._read_jsonl(base / f"{ARTIFACT_VERIFIED_SUBPROOFS}.jsonl"),
            ARTIFACT_PARTIAL_PROOFS: self._read_jsonl(base / f"{ARTIFACT_PARTIAL_PROOFS}.jsonl"),
            ARTIFACT_FAILED_SUBPROOFS: self._read_jsonl(base / f"{ARTIFACT_FAILED_SUBPROOFS}.jsonl"),
            ARTIFACT_FINAL_ATTEMPTS: self._read_jsonl(base / f"{ARTIFACT_FINAL_ATTEMPTS}.jsonl"),
            ARTIFACT_FINAL_CYCLE_PACKETS: self._read_jsonl(base / f"{ARTIFACT_FINAL_CYCLE_PACKETS}.jsonl"),
        }

    async def allocate_context(
        self,
        *,
        session_id: str,
        mode: str,
        user_prompt: str,
        lean_template: str,
        task_request: str,
        context_window: int,
        max_output_tokens: int,
        accepted_ideas: list[str],
        verified_subproofs: list[dict[str, Any]],
        partial_proofs: list[dict[str, Any]],
        failed_subproofs: list[dict[str, Any]],
        final_attempts: list[dict[str, Any]],
        final_cycle_packets: list[dict[str, Any]] | None = None,
        refuted_constructions: list[dict[str, Any]] | None = None,
        current_final_cycle_packet: dict[str, Any] | None = None,
        current_working_proof_attempt: dict[str, Any] | None = None,
        capped_rejection_feedback: str = "",
    ) -> LeanOJContextAllocation:
        """Allocate optional LeanOJ memory direct first, then through scoped RAG."""
        normalized_mode = mode if mode in {"brainstorm", "recursive_brainstorm", "subproof", "final_solver"} else "brainstorm"
        allocation = LeanOJContextAllocation(
            capped_rejection_feedback=capped_rejection_feedback.strip(),
            current_final_cycle_packet=self._format_final_cycle_packet(current_final_cycle_packet)
            if current_final_cycle_packet
            else "",
            current_working_proof_attempt=self._format_working_proof_attempt(current_working_proof_attempt)
            if current_working_proof_attempt
            else "",
            refuted_construction_warnings=self._format_refuted_construction_warnings(refuted_constructions or [])
            if normalized_mode == "final_solver"
            else "",
        )

        available_tokens = rag_config.get_available_input_tokens(context_window, max_output_tokens)
        mandatory_tokens = count_tokens(user_prompt) + count_tokens(lean_template) + count_tokens(task_request)
        mandatory_tokens += rag_config.get_prompt_assembly_overhead_estimate()
        mandatory_tokens += count_tokens(allocation.current_final_cycle_packet)
        mandatory_tokens += count_tokens(allocation.current_working_proof_attempt)
        mandatory_tokens += count_tokens(allocation.refuted_construction_warnings)
        mandatory_tokens += count_tokens(allocation.capped_rejection_feedback)
        remaining_tokens = available_tokens - mandatory_tokens
        if remaining_tokens < 0:
            raise RuntimeError(
                "LeanOJ mandatory context overflow before optional proof memory allocation. "
                f"Mandatory tokens: {mandatory_tokens}. Available input tokens: {available_tokens}. "
                f"Context mode: {normalized_mode}. Increase the role context window or reduce mandatory context."
            )

        direct_parts: list[str] = []
        offloaded_items: list[LeanOJMemoryItem] = []
        minimum_rag_reserve = min(5000, max(1000, int(available_tokens * 0.05)))

        for item in self._memory_items(
            session_id=session_id,
            mode=normalized_mode,
            accepted_ideas=accepted_ideas,
            verified_subproofs=verified_subproofs,
            partial_proofs=partial_proofs,
            failed_subproofs=failed_subproofs,
            final_attempts=final_attempts,
            final_cycle_packets=final_cycle_packets or [],
            current_final_cycle_packet=current_final_cycle_packet,
            has_current_working_proof_attempt=current_working_proof_attempt is not None,
        ):
            formatted = f"{item.title}\n{item.text}".strip()
            tokens = count_tokens(formatted)
            if (
                not item.rag_only
                and tokens <= remaining_tokens
                and remaining_tokens - tokens >= minimum_rag_reserve
            ):
                direct_parts.append(formatted)
                allocation.direct_sources.append(item.source_name)
                remaining_tokens -= tokens
            else:
                offloaded_items.append(item)
                allocation.rag_sources.append(item.source_name)

        allocation.direct_proof_context = "\n\n".join(direct_parts).strip()

        if offloaded_items and remaining_tokens <= 500:
            offloaded_titles = ", ".join(item.artifact for item in offloaded_items)
            raise RuntimeError(
                "LeanOJ context allocation could not preserve useful proof memory. "
                f"Mandatory context left only {remaining_tokens} tokens for RAG/offload; "
                f"offloaded sources would be silently dropped: {offloaded_titles}."
            )

        if offloaded_items:
            for item in offloaded_items:
                await self._ensure_source_indexed(item.source_name, f"{item.title}\n{item.text}".strip())

            rag_pack = await rag_manager.retrieve(
                query="\n\n".join([user_prompt, lean_template, task_request]),
                chunk_size=rag_config.validator_chunk_size,
                max_tokens=max(0, remaining_tokens - 200),
                exclude_sources=allocation.direct_sources or None,
                include_sources=allocation.rag_sources,
                include_source_prefixes=[self.source_prefix(session_id)],
            )
            allocation.rag_evidence_context = rag_pack.text or ""

        return allocation

    async def remove_session(self, session_id: str) -> None:
        """Remove persisted LeanOJ artifacts and their RAG sources for one session."""
        base = self.session_artifact_dir(session_id)
        if base.exists():
            shutil.rmtree(base)
        self._clear_sync_counts(session_id)
        await self.remove_session_rag_sources(session_id)

    async def clear_all(self) -> None:
        """Remove all LeanOJ artifact stores and LeanOJ RAG sources."""
        base = self.artifacts_base_dir()
        session_ids = [path.name for path in base.iterdir() if path.is_dir()] if base.exists() else []
        if base.exists():
            shutil.rmtree(base)
        self._artifact_sync_counts.clear()
        self._artifact_sync_digests.clear()
        await self.remove_all_leanoj_rag_sources(session_ids=session_ids)

    async def remove_session_rag_sources(self, session_id: str) -> None:
        await self._remove_rag_sources(self.source_names_for_session(session_id))

    async def remove_all_leanoj_rag_sources(self, session_ids: list[str] | None = None) -> None:
        sources: set[str] = set(self._indexed_hashes.keys())
        for session_id in session_ids or []:
            sources.update(self.source_names_for_session(session_id))
        if session_ids is None:
            base = self.artifacts_base_dir()
            if base.exists():
                for path in base.iterdir():
                    if path.is_dir():
                        sources.update(self.source_names_for_session(path.name))
        await self._remove_rag_sources(sources)

    async def _remove_rag_sources(self, sources: list[str] | set[str]) -> None:
        for source_name in sorted({source for source in sources if source}):
            try:
                await rag_manager.remove_document(source_name)
            except Exception as exc:
                logger.warning("Failed to remove LeanOJ RAG source %s: %s", source_name, exc)
            self._indexed_hashes.pop(source_name, None)

    async def _ensure_source_indexed(self, source_name: str, text: str) -> None:
        if not text.strip():
            return
        lock = self._index_locks.setdefault(source_name, asyncio.Lock())
        async with lock:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            has_chunks = any(
                chunk.source_file == source_name
                for chunk in rag_manager.chunks_by_size[rag_config.validator_chunk_size]
            )
            if self._indexed_hashes.get(source_name) == digest and has_chunks:
                return

            await rag_manager.remove_document(source_name)

            await rag_manager.add_text(
                text,
                source_name,
                chunk_sizes=rag_config.submitter_chunk_intervals,
                is_permanent=False,
            )
            self._indexed_hashes[source_name] = digest

    async def _sync_jsonl(
        self,
        path: Path,
        session_id: str,
        artifact: str,
        records: list[Any],
    ) -> None:
        """Append new records; rewrite when records shrink or same-length content changes."""
        key = (session_id, artifact)
        new_digest = self._records_digest(records)
        persisted_count = self._artifact_sync_counts.get(key)
        if persisted_count is None:
            persisted_count = self._count_jsonl_records(path)

        if len(records) < persisted_count:
            await self._write_jsonl(path, records)
            self._artifact_sync_counts[key] = len(records)
            self._artifact_sync_digests[key] = new_digest
            return

        if len(records) == persisted_count:
            known_digest = self._artifact_sync_digests.get(key)
            if known_digest == new_digest:
                self._artifact_sync_counts[key] = persisted_count
                return
            if known_digest is None and self._jsonl_digest(path) == new_digest:
                self._artifact_sync_counts[key] = persisted_count
                self._artifact_sync_digests[key] = new_digest
                return
            await self._write_jsonl(path, records)
            self._artifact_sync_counts[key] = persisted_count
            self._artifact_sync_digests[key] = new_digest
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            for record in records[persisted_count:]:
                await f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._artifact_sync_counts[key] = len(records)
        self._artifact_sync_digests[key] = new_digest

    @staticmethod
    def _records_digest(records: list[Any]) -> str:
        return hashlib.sha256(
            "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) for record in records).encode(
                "utf-8"
            )
        ).hexdigest()

    def _jsonl_digest(self, path: Path) -> str:
        if not path.exists():
            return self._records_digest([])
        try:
            records = self._read_jsonl(path)
        except Exception as exc:
            logger.warning("Failed to digest LeanOJ artifact log %s: %s", path, exc)
            return ""
        return self._records_digest(records)

    @staticmethod
    def _count_jsonl_records(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception as exc:
            logger.warning("Failed to count LeanOJ artifact log %s: %s", path, exc)
            return 0

    def _clear_sync_counts(self, session_id: str) -> None:
        stale_keys = [key for key in self._artifact_sync_counts if key[0] == session_id]
        for key in stale_keys:
            self._artifact_sync_counts.pop(key, None)
            self._artifact_sync_digests.pop(key, None)

    def _memory_items(
        self,
        *,
        session_id: str,
        mode: str,
        accepted_ideas: list[str],
        verified_subproofs: list[dict[str, Any]],
        partial_proofs: list[dict[str, Any]],
        failed_subproofs: list[dict[str, Any]],
        final_attempts: list[dict[str, Any]],
        final_cycle_packets: list[dict[str, Any]],
        current_final_cycle_packet: dict[str, Any] | None,
        has_current_working_proof_attempt: bool = False,
    ) -> list[LeanOJMemoryItem]:
        recent_failed_subproofs = failed_subproofs[-10:]
        raw_items = {
            ARTIFACT_FINAL_CYCLE_PACKETS: (
                "HISTORICAL FINAL-CYCLE FAILURE PACKETS",
                self._format_final_cycle_packets(final_cycle_packets),
            ),
            ARTIFACT_VERIFIED_SUBPROOFS: (
                "VERIFIED SUBPROOFS / HELPER LEMMAS",
                self._format_verified_subproofs_for_final(verified_subproofs)
                if mode == "final_solver"
                else self._format_verified_subproofs(verified_subproofs),
            ),
            ARTIFACT_PARTIAL_PROOFS: (
                "LEAN-ACCEPTED PARTIAL PROOF SCAFFOLDS",
                self._format_partial_proofs_for_final(partial_proofs)
                if mode == "final_solver"
                else self._format_partial_proofs(partial_proofs),
            ),
            ARTIFACT_ACCEPTED_IDEAS: (
                "ACTIVE PROOF-PLAN NOTES" if mode == "final_solver" else "ACCEPTED BRAINSTORM IDEAS",
                self._format_strings_for_final(accepted_ideas)
                if mode == "final_solver"
                else self._format_strings(accepted_ideas),
            ),
            ARTIFACT_FAILED_SUBPROOFS: (
                "FAILED SUBPROOF FEEDBACK",
                self._format_attempts(recent_failed_subproofs),
            ),
        }

        brainstorm_priority = [
            ARTIFACT_ACCEPTED_IDEAS,
            ARTIFACT_PARTIAL_PROOFS,
            ARTIFACT_VERIFIED_SUBPROOFS,
            ARTIFACT_FINAL_CYCLE_PACKETS,
        ]
        if has_current_working_proof_attempt:
            brainstorm_priority = [
                ARTIFACT_PARTIAL_PROOFS,
                ARTIFACT_VERIFIED_SUBPROOFS,
                ARTIFACT_ACCEPTED_IDEAS,
                ARTIFACT_FINAL_CYCLE_PACKETS,
            ]

        priority_by_mode = {
            "final_solver": [
                ARTIFACT_VERIFIED_SUBPROOFS,
                ARTIFACT_ACCEPTED_IDEAS,
            ],
            "brainstorm": brainstorm_priority,
            "recursive_brainstorm": brainstorm_priority,
            "subproof": [
                ARTIFACT_FAILED_SUBPROOFS,
                ARTIFACT_VERIFIED_SUBPROOFS,
                ARTIFACT_PARTIAL_PROOFS,
                ARTIFACT_ACCEPTED_IDEAS,
                ARTIFACT_FINAL_CYCLE_PACKETS,
            ],
        }
        order = priority_by_mode.get(mode, priority_by_mode["brainstorm"])

        items: list[LeanOJMemoryItem] = []
        for priority, artifact in enumerate(order):
            title, text = raw_items[artifact]
            if not text:
                continue
            items.append(
                LeanOJMemoryItem(
                    artifact=artifact,
                    title=title,
                    text=text,
                    priority=priority,
                    source_name=self.source_name(session_id, artifact),
                    rag_only=artifact == ARTIFACT_FINAL_CYCLE_PACKETS,
                )
            )
        return items

    @staticmethod
    def _record_key(record: dict[str, Any] | None) -> str:
        if not record:
            return ""
        try:
            return json.dumps(record, sort_keys=True, default=str)
        except TypeError:
            return str(record)

    @staticmethod
    async def _write_jsonl(path: Path, records: list[Any]) -> None:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            for record in records:
                await f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        except Exception as exc:
            logger.warning("Failed to load LeanOJ artifact log %s: %s", path, exc)
        return records

    @staticmethod
    def _records_to_strings(records: list[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        for record in records:
            value = record.get("content", record)
            if isinstance(value, str) and value.strip():
                values.append(value)
        return values

    @staticmethod
    def _format_strings(values: list[str]) -> str:
        clean = [str(value).strip() for value in values if str(value).strip()]
        return "\n".join(f"{index}. {value}" for index, value in enumerate(clean, start=1))

    @staticmethod
    def _format_strings_for_final(values: list[str]) -> str:
        clean = [LeanOJContextManager._final_mode_text(value).strip() for value in values if str(value).strip()]
        return "\n".join(f"{index}. {value}" for index, value in enumerate(clean, start=1))

    @staticmethod
    def _format_attempts(records: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, record in enumerate(records, start=1):
            lean_code = str(record.get("lean_code") or "").strip()
            lean_feedback = _remove_attempt_count_language(record.get("lean_feedback") or "")
            feedback_lines = ["Lean pass feedback:", lean_feedback] if lean_feedback else []
            blocks.append(
                "\n".join(
                    [
                        f"FEEDBACK ITEM {index}: {_remove_attempt_count_language(record.get('request', 'proof feedback'))}",
                        "Error summary: "
                        f"{_remove_attempt_count_language(record.get('error_summary', record.get('error_output', '')))}",
                        *feedback_lines,
                        "Lean code:",
                        lean_code or "[not recorded]",
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @staticmethod
    def _format_refuted_construction_warnings(
        records: list[dict[str, Any]],
        *,
        limit: int = 5,
        max_chars: int = 1500,
    ) -> str:
        """Compact final-mode warnings for failed routes, kept separate from proof evidence."""
        clean_records = [record for record in records if isinstance(record, dict)]
        blocks: list[str] = []
        for record in clean_records[-limit:]:
            content = str(record.get("content") or record.get("summary") or record.get("error_summary") or "").strip()
            if not content:
                continue
            reason = str(
                record.get("reasoning")
                or record.get("validator_summary")
                or record.get("validator_reasoning")
                or record.get("edit_reasoning")
                or ""
            ).strip()
            line = LeanOJContextManager._final_mode_text(content)
            if reason:
                line = f"{line} Reason: {LeanOJContextManager._final_mode_text(reason)}"
            blocks.append(line)

        if not blocks:
            return ""

        text = "\n".join(f"{index}. {block}" for index, block in enumerate(blocks, start=1))
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n[truncated]"

    @staticmethod
    def _format_final_cycle_packets(packets: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, packet in enumerate(packets, start=1):
            attempts = packet.get("attempts") if isinstance(packet.get("attempts"), list) else []
            blocks.append(
                "\n".join(
                    [
                        f"FINAL-CYCLE FEEDBACK {index}",
                        f"Summary: {_remove_attempt_count_language(packet.get('summary', ''))}",
                        "Recent verification/edit feedback:",
                        LeanOJContextManager._format_attempts([dict(item) for item in attempts if isinstance(item, dict)]),
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @staticmethod
    def _format_verified_subproofs(subproofs: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, subproof in enumerate(subproofs, start=1):
            lean_feedback = str(subproof.get("lean_feedback") or "").strip()
            feedback_lines = ["Lean verifier feedback:", lean_feedback] if lean_feedback else []
            blocks.append(
                "\n".join(
                    [
                        f"SUBPROOF {index}: {subproof.get('request', '')}",
                        f"Role: {subproof.get('role', '')}",
                        f"Theorem/Lemma: {subproof.get('theorem_or_lemma', '')}",
                        *feedback_lines,
                        "Verified Lean 4 code:",
                        str(subproof.get("lean_code") or ""),
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @staticmethod
    def _final_mode_text(value: Any) -> str:
        text = str(value or "")
        cleaned = (
            text.replace("need_more_brainstorming", "additional proof context")
            .replace("Brainstorm", "Proof memory")
            .replace("brainstorm", "proof memory")
            .replace("BRAINSTORM", "PROOF MEMORY")
        )
        return _remove_attempt_count_language(cleaned)

    @classmethod
    def _format_verified_subproofs_for_final(cls, subproofs: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, subproof in enumerate(subproofs, start=1):
            lean_feedback = cls._final_mode_text(subproof.get("lean_feedback") or "").strip()
            feedback_lines = ["Lean verifier feedback:", lean_feedback] if lean_feedback else []
            blocks.append(
                "\n".join(
                    [
                        f"SUBPROOF {index}: {cls._final_mode_text(subproof.get('request', ''))}",
                        f"Theorem/Lemma: {cls._final_mode_text(subproof.get('theorem_or_lemma', ''))}",
                        *feedback_lines,
                        "Verified Lean 4 code:",
                        str(subproof.get("lean_code") or ""),
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @staticmethod
    def _format_partial_proofs(partial_proofs: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, proof in enumerate(partial_proofs, start=1):
            placeholders = ", ".join(proof.get("placeholder_tokens") or []) or "unknown"
            blocks.append(
                "\n".join(
                    [
                        f"PARTIAL PROOF {index}: {proof.get('request', '')}",
                        f"Target: {proof.get('target', '')}; placeholders: {placeholders}",
                        f"Summary: {proof.get('summary', '')}",
                        "Lean-accepted incomplete scaffold:",
                        str(proof.get("lean_code") or ""),
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @classmethod
    def _format_partial_proofs_for_final(cls, partial_proofs: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, proof in enumerate(partial_proofs, start=1):
            placeholders = ", ".join(proof.get("placeholder_tokens") or []) or "unknown"
            blocks.append(
                "\n".join(
                    [
                        f"PARTIAL PROOF {index}: {cls._final_mode_text(proof.get('request', ''))}",
                        f"Placeholders: {placeholders}",
                        f"Summary: {cls._final_mode_text(proof.get('summary', ''))}",
                        "Lean-accepted incomplete scaffold:",
                        str(proof.get("lean_code") or ""),
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    @staticmethod
    def _format_final_cycle_packet(packet: dict[str, Any] | None) -> str:
        if not packet:
            return ""
        attempts = packet.get("attempts") if isinstance(packet.get("attempts"), list) else []
        partial_proofs = packet.get("partial_proofs") if isinstance(packet.get("partial_proofs"), list) else []
        lines = [
            "CURRENT FINAL-CYCLE FEEDBACK",
            "This is the immediate final-loop feedback to use for repairing the current proof.",
            LeanOJContextManager._format_attempts([dict(item) for item in attempts if isinstance(item, dict)]),
            "Partial final scaffolds captured during this cycle:",
            LeanOJContextManager._format_partial_proofs(
                [dict(item) for item in partial_proofs if isinstance(item, dict)]
            )
            or "[none recorded]",
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _format_working_proof_attempt(packet: dict[str, Any] | None) -> str:
        if not packet:
            return ""
        verified = packet.get("verified_subproofs") if isinstance(packet.get("verified_subproofs"), list) else []
        partials = packet.get("partial_final_proofs") if isinstance(packet.get("partial_final_proofs"), list) else []
        parts = [
            "CURRENT WORKING PROOF ATTEMPT",
            "This is the proof attempt the next LeanOJ brainstorm must repair or complete directly.",
            f"Trigger: {packet.get('trigger', '')}",
            f"Requested path: {packet.get('requested_path', '')}",
            f"Stuck reason: {_remove_attempt_count_language(packet.get('stuck_reason', ''))}",
            (
                "Master proof metadata: "
                f"version={packet.get('master_proof_version', 0)}, "
                f"lines={packet.get('master_proof_line_count', 0)}, "
                f"sha256={packet.get('master_proof_hash', '')}"
            ),
            f"Last edit summary: {packet.get('master_proof_last_edit_summary', '')}",
            "Latest master_proof.lean:",
            str(packet.get("master_proof") or "[not initialized]").strip(),
            "Recent final solver feedback:",
            str(packet.get("recent_final_attempts") or "[none recorded]").strip(),
            "Verified helper subproofs available to reuse:",
            LeanOJContextManager._format_verified_subproofs([dict(item) for item in verified if isinstance(item, dict)])
            or "[none recorded]",
            "Lean-accepted partial final scaffolds:",
            LeanOJContextManager._format_partial_proofs([dict(item) for item in partials if isinstance(item, dict)])
            or "[none recorded]",
        ]
        old_attempt = str(packet.get("old_attempt_before_redo") or "").strip()
        if old_attempt:
            validator_justification = str(
                packet.get("old_attempt_before_redo_validator_justification") or ""
            ).strip()
            apparent_issue = str(packet.get("old_attempt_before_redo_apparent_issue") or "").strip()
            parts += [
                "",
                "OLD ATTEMPT THE SUBMITTER DECIDED TO REDO (preserved for reference only; do NOT revert to this):",
                f"Original version: v{packet.get('old_attempt_before_redo_version', '?')}",
                (
                    "Old attempt metadata: "
                    f"lines={packet.get('old_attempt_before_redo_line_count', 0)}, "
                    f"chars={packet.get('old_attempt_before_redo_char_count', 0)}, "
                    f"sha256={packet.get('old_attempt_before_redo_hash', '')}"
                ),
                f"Summary: {packet.get('old_attempt_before_redo_summary', '')}",
                "WHY THE VALIDATOR ALLOWED THIS REDO/SHORTENING:",
                validator_justification or "[No validator justification was recorded.]",
                "APPARENT ISSUE WITH THIS OLD LONGER ATTEMPT:",
                apparent_issue or "[No apparent issue was recorded.]",
                "Old proof content:",
                old_attempt,
            ]
        return "\n".join(parts).strip()


leanoj_context_manager = LeanOJContextManager()