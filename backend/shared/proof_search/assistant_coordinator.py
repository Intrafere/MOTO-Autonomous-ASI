"""Non-blocking Assistant proof-support coordinator."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from backend.shared.config import system_config
from backend.shared.json_parser import parse_json
from backend.shared.proof_search.assistant_cache import AssistantRankCache, goal_hash_for_snapshot
from backend.shared.proof_search.assistant_models import AssistantProofPack, AssistantProofSupport, AssistantTargetSnapshot
from backend.shared.proof_search.assistant_ranker import ranked_candidates_to_cache_rows, score_assistant_proof_candidates, select_assistant_proof_supports
from backend.shared.proof_search.models import ProofSearchRequest, UnifiedProofSearchRecord, default_proof_search_corpora
from backend.shared.proof_search.search_service import ProofSearchService, proof_search_service
from backend.shared.response_extraction import extract_response_text

logger = logging.getLogger(__name__)

_ASSISTANT_CANDIDATE_POOL_TARGET = 64
_ASSISTANT_SHORTLIST_TARGET = 21
_ASSISTANT_FINAL_PACK_LIMIT = 7
_ASSISTANT_SELECTION_MAX_OUTPUT_TOKENS = 4096
_ASSISTANT_SELECTION_CODE_PREVIEW_CHARS = 1200
_CURRENT_RUN_CORPUS_SCOPES = {"active", "current"}

AssistantSelector = Callable[
    [AssistantTargetSnapshot, list[AssistantProofSupport], str, str, str],
    Awaitable[tuple[list[str], str]],
]


class _AssistantSelectionOutputError(ValueError):
    """Raised when the Assistant selection call returns malformed output."""


class AssistantProofSearchCoordinator:
    """Maintains latest Assistant proof-support packs without blocking solvers."""

    def __init__(
        self,
        service: ProofSearchService | None = None,
        cache: AssistantRankCache | None = None,
        assistant_selector: AssistantSelector | None = None,
    ) -> None:
        self._service = service or proof_search_service
        self._cache = cache or AssistantRankCache()
        self._assistant_selector = assistant_selector
        self._packs: dict[str, AssistantProofPack] = {}
        self._goal_target_hashes: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._task_sequence = 0
        self._latest_pack_target_hash = ""
        self._latest_pack_consumed = True

    @property
    def enabled(self) -> bool:
        return bool(system_config.agent_conversation_memory_enabled)

    def get_latest_pack(self, target_hash: str | None = None) -> AssistantProofPack | None:
        if target_hash:
            return _drop_current_run_supports_from_pack(self._packs.get(target_hash))
        if not self._packs:
            return None
        return _drop_current_run_supports_from_pack(next(reversed(self._packs.values())))

    def get_status(self) -> dict[str, Any]:
        latest_pack = self.get_latest_pack()
        enabled_corpora = default_proof_search_corpora() if self.enabled else []
        disabled_reason = ""
        if not self.enabled:
            disabled_reason = "Session History Memory is disabled."
        elif not enabled_corpora:
            disabled_reason = "No proof-search corpora are enabled."
        return {
            "enabled": self.enabled,
            "running_tasks": sum(1 for task in self._tasks.values() if not task.done()),
            "cached_pack_count": len(self._packs),
            "latest_target_hash": latest_pack.target_hash if latest_pack else "",
            "latest_workflow_mode": latest_pack.workflow_mode if latest_pack else "",
            "latest_target_kind": latest_pack.target_kind if latest_pack else "",
            "latest_result_count": len(latest_pack.results) if latest_pack else 0,
            "latest_freshness": latest_pack.freshness if latest_pack else "",
            "latest_warnings": latest_pack.warnings[:3] if latest_pack else [],
            "enabled_corpora": enabled_corpora,
            "disabled_reason": disabled_reason,
        }

    def submit_target(self, snapshot: AssistantTargetSnapshot) -> str:
        target_hash = snapshot.stable_hash()
        snapshot = snapshot.model_copy(update={"target_hash": target_hash})
        if not self.enabled:
            self._packs.pop(target_hash, None)
            logger.info(
                "Assistant memory search skipped for %s/%s: Agent Conversation Memory is disabled",
                snapshot.workflow_mode,
                snapshot.target_kind,
            )
            return target_hash
        cached_pack = self._load_cached_pack(snapshot)
        if cached_pack is not None:
            self._packs[target_hash] = cached_pack
            logger.info(
                "Assistant memory loaded cached pack for %s/%s (target=%s, results=%s, freshness=%s)",
                snapshot.workflow_mode,
                snapshot.target_kind,
                target_hash[:12],
                len(cached_pack.results),
                cached_pack.freshness,
            )
            if not cached_pack.results and cached_pack.freshness == "cached":
                logger.info(
                    "Assistant memory refresh skipped for %s/%s (target=%s already has an empty Assistant selection)",
                    snapshot.workflow_mode,
                    snapshot.target_kind,
                    target_hash[:12],
                )
                return target_hash
            if cached_pack.freshness == "cached":
                self._latest_pack_target_hash = target_hash
                self._latest_pack_consumed = not bool(cached_pack.results)
                logger.info(
                    "Assistant memory refresh skipped for %s/%s (target=%s exact cached pack is current)",
                    snapshot.workflow_mode,
                    snapshot.target_kind,
                    target_hash[:12],
                )
                return target_hash
        running_target_hash = self._running_target_hash()
        if running_target_hash:
            logger.info(
                "Assistant memory refresh already running for %s/%s (running_target=%s, requested_target=%s)",
                snapshot.workflow_mode,
                snapshot.target_kind,
                running_target_hash[:12],
                target_hash[:12],
            )
            return target_hash
        if self._latest_pack_target_hash and not self._latest_pack_consumed:
            if cached_pack is None:
                latest_pack = self.get_latest_pack(self._latest_pack_target_hash) or self.get_latest_pack()
                if latest_pack is not None:
                    self._packs[target_hash] = latest_pack.model_copy(
                        update={
                            "target_hash": target_hash,
                            "freshness": "stale-but-best-known",
                            "selection_mode": "stale-but-best-known",
                        }
                    )
            logger.info(
                "Assistant memory refresh deferred for %s/%s (latest_target=%s has not been consumed by a solver response)",
                snapshot.workflow_mode,
                snapshot.target_kind,
                self._latest_pack_target_hash[:12],
            )
            return target_hash
        existing = self._tasks.get(target_hash)
        if existing and not existing.done():
            logger.info(
                "Assistant memory refresh already running for %s/%s (target=%s)",
                snapshot.workflow_mode,
                snapshot.target_kind,
                target_hash[:12],
            )
            return target_hash
        logger.info(
            "Assistant memory refresh scheduled for %s/%s (target=%s, phase=%s, source=%s:%s)",
            snapshot.workflow_mode,
            snapshot.target_kind,
            target_hash[:12],
            snapshot.workflow_phase or "unknown",
            snapshot.source_type or "unknown",
            snapshot.source_id or "unknown",
        )
        task = asyncio.create_task(self._refresh_pack(snapshot))
        task.add_done_callback(lambda completed: self._on_task_done(target_hash, completed))
        self._tasks[target_hash] = task
        return target_hash

    async def refresh_now(self, snapshot: AssistantTargetSnapshot) -> AssistantProofPack | None:
        target_hash = snapshot.stable_hash()
        snapshot = snapshot.model_copy(update={"target_hash": target_hash})
        if not self.enabled:
            return None
        cached_pack = self._load_cached_pack(snapshot)
        if cached_pack is not None:
            self._packs[target_hash] = cached_pack
        await self._refresh_pack(snapshot)
        return self._packs.get(target_hash)

    async def stop_all(self, *, clear_packs: bool = True, broadcast: bool = False, reason: str = "parent_stopped") -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if clear_packs:
            self._packs.clear()
            self._goal_target_hashes.clear()
            self._latest_pack_target_hash = ""
            self._latest_pack_consumed = True
            await asyncio.to_thread(_delete_if_exists, _assistant_pack_path())
        if broadcast:
            await self._broadcast_event("assistant_proof_pack_stopped", {"reason": reason, "cleared": clear_packs})

    def mark_pack_consumed_by_solver(self, target_hash: str, *, role_id: str = "", task_id: str = "") -> None:
        """Allow the next Assistant refresh after a solver used a published pack."""
        if not target_hash or target_hash not in self._packs:
            return
        self._latest_pack_consumed = True
        logger.info(
            "Assistant memory pack consumed by solver role=%s task=%s target=%s",
            role_id or "unknown",
            task_id or "unknown",
            target_hash[:12],
        )

    async def _refresh_pack(self, snapshot: AssistantTargetSnapshot) -> None:
        async with self._lock:
            warnings: list[str] = []
            corpora = default_proof_search_corpora()
            if not corpora:
                corpora = ["moto", "manual", "leanoj"]

            records: list[UnifiedProofSearchRecord] = []
            seen_ids: set[str] = set()
            excluded_session_ids = [session_id for session_id in [_active_autonomous_session_id()] if session_id]
            for query in _build_query_variants(snapshot):
                if len(records) >= _ASSISTANT_CANDIDATE_POOL_TARGET:
                    break
                try:
                    query_records = await self._service.search_candidate_pool(
                        ProofSearchRequest(
                            query=query,
                            goal_statement=snapshot.target_statement or snapshot.lean_template,
                            imports=snapshot.imports or ["Mathlib"],
                            dependency_names=snapshot.dependency_names,
                            corpora=corpora,
                            verified_only=True,
                            include_partial=False,
                            include_failed=False,
                            limit=_ASSISTANT_CANDIDATE_POOL_TARGET,
                            hydrate_lean_code=True,
                        ),
                        pool_limit=_ASSISTANT_CANDIDATE_POOL_TARGET - len(records),
                        exclude_corpus_scopes=sorted(_CURRENT_RUN_CORPUS_SCOPES),
                        exclude_session_ids=excluded_session_ids,
                    )
                    query_records = _filter_current_run_records(query_records)
                except Exception as exc:
                    logger.debug("Assistant proof search query failed: %s", exc)
                    warnings.append(f"Search query failed: {exc}")
                    continue
                for record in query_records:
                    if record.search_id in seen_ids:
                        continue
                    seen_ids.add(record.search_id)
                    records.append(record)

            ranked_candidates = score_assistant_proof_candidates(records, snapshot)
            await asyncio.to_thread(self._cache.upsert_candidates, target_hash=snapshot.target_hash, candidates=ranked_candidates_to_cache_rows(ranked_candidates))
            candidate_stats = await asyncio.to_thread(self._cache.load_candidate_stats, snapshot.target_hash)
            shortlist = select_assistant_proof_supports(ranked_candidates, limit=_ASSISTANT_SHORTLIST_TARGET, candidate_stats=candidate_stats)
            if not shortlist:
                warnings.append("Assistant found no verified proof supports for the current target.")
                await self._publish_pack(snapshot, [], warnings=warnings, selection_mode="no_candidates", candidate_count=len(records), shortlist_count=0, selection_reasoning="No verified candidate supports were found.")
                return
            await self._select_and_publish_assistant_pack(snapshot=snapshot, shortlist=shortlist, warnings=warnings, candidate_count=len(records))

    async def _select_and_publish_assistant_pack(self, *, snapshot: AssistantTargetSnapshot, shortlist: list[AssistantProofSupport], warnings: list[str], candidate_count: int) -> None:
        assistant_role_id = _assistant_role_id_for_snapshot(snapshot)
        assistant_model_id = "injected-assistant" if self._assistant_selector is not None else _assistant_model_id(assistant_role_id)
        if not assistant_model_id:
            warnings.append(f"Assistant role '{assistant_role_id}' is not configured.")
            await self._publish_pack(snapshot, [], warnings=warnings, selection_mode="unavailable", assistant_role_id=assistant_role_id, assistant_model_id="", candidate_count=candidate_count, shortlist_count=len(shortlist), selection_reasoning="Configured Assistant role was unavailable.")
            return

        task_id = self._next_assistant_task_id(snapshot.workflow_mode)
        await self._broadcast_event("assistant_proof_pack_refresh_started", {"target_hash": snapshot.target_hash, "workflow_mode": snapshot.workflow_mode, "target_kind": snapshot.target_kind, "workflow_phase": snapshot.workflow_phase, "source_type": snapshot.source_type, "source_id": snapshot.source_id, "assistant_role_id": assistant_role_id, "assistant_model_id": assistant_model_id, "candidate_count": candidate_count, "shortlist_count": len(shortlist), "max_result_count": _ASSISTANT_FINAL_PACK_LIMIT})
        try:
            selected_search_ids, selection_reasoning = await self._select_with_assistant(snapshot, shortlist, assistant_role_id=assistant_role_id, assistant_model_id=assistant_model_id, task_id=task_id)
        except Exception as exc:
            warnings.append(f"Assistant LLM selection failed: {exc}")
            await self._broadcast_event("assistant_proof_pack_warning", {"target_hash": snapshot.target_hash, "workflow_mode": snapshot.workflow_mode, "target_kind": snapshot.target_kind, "workflow_phase": snapshot.workflow_phase, "source_type": snapshot.source_type, "source_id": snapshot.source_id, "warnings": warnings[-3:], "assistant_role_id": assistant_role_id, "assistant_model_id": assistant_model_id, "candidate_count": candidate_count, "shortlist_count": len(shortlist)})
            await self._publish_pack(snapshot, [], warnings=warnings, selection_mode="unavailable", assistant_role_id=assistant_role_id, assistant_model_id=assistant_model_id, candidate_count=candidate_count, shortlist_count=len(shortlist), selection_reasoning="Assistant LLM selection failed.")
            return

        selected_supports = _supports_for_selected_ids(shortlist, selected_search_ids)
        if selected_search_ids and not selected_supports:
            warnings.append("Assistant selected only IDs that were not present in the candidate shortlist.")
        await self._publish_pack(snapshot, selected_supports, warnings=warnings, selection_mode="assistant_llm", assistant_role_id=assistant_role_id, assistant_model_id=assistant_model_id, candidate_count=candidate_count, shortlist_count=len(shortlist), selection_reasoning=selection_reasoning)

    def _next_assistant_task_id(self, workflow_mode: str) -> str:
        self._task_sequence += 1
        mode = "".join(char if char.isalnum() else "_" for char in workflow_mode or "assistant")
        return f"assistant_pack_{mode}_{self._task_sequence:03d}"

    async def _select_with_assistant(self, snapshot: AssistantTargetSnapshot, shortlist: list[AssistantProofSupport], *, assistant_role_id: str, assistant_model_id: str, task_id: str) -> tuple[list[str], str]:
        if self._assistant_selector is not None:
            return await self._assistant_selector(snapshot, shortlist, assistant_role_id, assistant_model_id, task_id)
        from backend.shared.api_client_manager import api_client_manager
        role_config = api_client_manager.get_role_config(assistant_role_id)
        if role_config is None:
            raise RuntimeError(f"Assistant role '{assistant_role_id}' is not configured")
        max_tokens = _assistant_selection_max_tokens(role_config.max_output_tokens)
        prompt = _build_assistant_selection_prompt(snapshot, shortlist)
        try:
            payload = await _generate_assistant_selection_payload(
                prompt=prompt,
                task_id=task_id,
                assistant_role_id=assistant_role_id,
                assistant_model_id=assistant_model_id,
                max_tokens=max_tokens,
            )
            selected_ids = _extract_selected_search_ids(payload)
        except _AssistantSelectionOutputError as first_error:
            repair_prompt = _build_assistant_selection_repair_prompt(
                snapshot,
                shortlist,
                error=str(first_error),
            )
            try:
                payload = await _generate_assistant_selection_payload(
                    prompt=repair_prompt,
                    task_id=f"{task_id}_retry",
                    assistant_role_id=assistant_role_id,
                    assistant_model_id=assistant_model_id,
                    max_tokens=max_tokens,
                )
                selected_ids = _extract_selected_search_ids(payload)
            except _AssistantSelectionOutputError as retry_error:
                raise _AssistantSelectionOutputError(
                    f"{first_error}; retry failed: {retry_error}"
                ) from retry_error
        clean_ids = [str(item).strip() for item in selected_ids if str(item).strip()]
        reasoning = str(payload.get("reasoning") or payload.get("selection_reasoning") or "").strip() or "Assistant selected proof supports for the current target."
        reasoning = _compact_for_assistant_selection(reasoning, 300)
        return clean_ids[:_ASSISTANT_FINAL_PACK_LIMIT], reasoning

    async def _publish_pack(self, snapshot: AssistantTargetSnapshot, supports: list[AssistantProofSupport], *, warnings: list[str], selection_mode: str, assistant_role_id: str = "", assistant_model_id: str = "", candidate_count: int, shortlist_count: int, selection_reasoning: str = "") -> None:
        pack = AssistantProofPack(workflow_mode=snapshot.workflow_mode, target_kind=snapshot.target_kind, target_hash=snapshot.target_hash, query_summary=_compact_query_summary(snapshot), results=supports[:_ASSISTANT_FINAL_PACK_LIMIT], warnings=warnings, selection_mode=selection_mode, assistant_role_id=assistant_role_id, assistant_model_id=assistant_model_id, candidate_count=candidate_count, shortlist_count=shortlist_count, selection_reasoning=selection_reasoning)
        source_counts: dict[str, int] = {}
        for support in pack.results:
            source_counts[support.corpus] = source_counts.get(support.corpus, 0) + 1
        logger.info("Assistant memory pack refreshed for %s/%s (target=%s, mode=%s, results=%s, local=%s, syntheticlib4=%s)", snapshot.workflow_mode, snapshot.target_kind, snapshot.target_hash[:12], selection_mode, len(pack.results), sum(count for corpus, count in source_counts.items() if corpus != "syntheticlib4"), source_counts.get("syntheticlib4", 0))
        self._packs[snapshot.target_hash] = pack
        self._latest_pack_target_hash = snapshot.target_hash
        self._latest_pack_consumed = not bool(pack.results)
        goal_hash = goal_hash_for_snapshot(snapshot)
        if goal_hash:
            self._goal_target_hashes[goal_hash] = snapshot.target_hash
        await asyncio.to_thread(self._cache.record_pack, snapshot=snapshot, pack=pack, selected_search_ids=[support.search_id for support in pack.results])
        await self._persist_pack(pack)
        await self._broadcast_event("assistant_proof_pack_updated", {"target_hash": pack.target_hash, "workflow_mode": pack.workflow_mode, "target_kind": pack.target_kind, "result_count": len(pack.results), "local_result_count": sum(count for corpus, count in source_counts.items() if corpus != "syntheticlib4"), "syntheticlib4_result_count": source_counts.get("syntheticlib4", 0), "source_counts": source_counts, "max_result_count": _ASSISTANT_FINAL_PACK_LIMIT, "workflow_phase": snapshot.workflow_phase, "source_type": snapshot.source_type, "source_id": snapshot.source_id, "warnings": pack.warnings[:3], "selection_mode": pack.selection_mode, "assistant_role_id": pack.assistant_role_id, "assistant_model_id": pack.assistant_model_id, "candidate_count": pack.candidate_count, "shortlist_count": pack.shortlist_count})

    def _on_task_done(self, target_hash: str, task: asyncio.Task) -> None:
        if self._tasks.get(target_hash) is task:
            self._tasks.pop(target_hash, None)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Assistant proof-search refresh failed")

    def _running_target_hash(self) -> str:
        for target_hash, task in self._tasks.items():
            if not task.done():
                return target_hash
        return ""

    async def _persist_pack(self, pack: AssistantProofPack) -> None:
        await asyncio.to_thread(_write_json, _assistant_pack_path(), pack.metadata_only_dump())

    async def _broadcast_event(self, event_type: str, payload: dict[str, Any]) -> None:
        try:
            from backend.api.routes import websocket
            await websocket.broadcast_event(event_type, payload)
        except Exception:
            logger.debug("Assistant proof-search event broadcast failed", exc_info=True)

    def _load_cached_pack(self, snapshot: AssistantTargetSnapshot) -> AssistantProofPack | None:
        try:
            goal_hash = goal_hash_for_snapshot(snapshot)
            previous_target_hash = self._goal_target_hashes.get(goal_hash) if goal_hash else ""
            if previous_target_hash:
                in_memory_pack = self._packs.get(previous_target_hash)
                if in_memory_pack is not None:
                    freshness = "cached" if previous_target_hash == snapshot.target_hash else "stale-but-best-known"
                    return _drop_current_run_supports_from_pack(in_memory_pack.model_copy(update={"target_hash": snapshot.target_hash, "freshness": freshness, "selection_mode": freshness}))
            cached = self._cache.load_cached_pack(target_hash=snapshot.target_hash, goal_hash=goal_hash)
            if cached is not None:
                cached = cached.model_copy(update={"selection_mode": cached.freshness})
            return _drop_current_run_supports_from_pack(cached)
        except Exception:
            logger.debug("Assistant proof-search cache lookup failed", exc_info=True)
            return None


def _build_query_variants(snapshot: AssistantTargetSnapshot) -> list[str]:
    variants = [snapshot.search_text(), "\n\n".join(part for part in [snapshot.user_prompt, snapshot.current_prompt_or_topic, snapshot.writing_goal, snapshot.outline_summary] if part), "\n\n".join(part for part in [snapshot.user_prompt, snapshot.target_statement] if part), "\n\n".join(part for part in [snapshot.lean_template, snapshot.lean_error] if part), "\n\n".join(part for part in [snapshot.rejection_feedback, snapshot.proof_attempt_feedback] if part), "\n\n".join(part for part in [snapshot.accepted_memory_summary, snapshot.paper_or_proof_draft_summary, snapshot.recent_activity_summary] if part), " ".join([*snapshot.dependency_names, *snapshot.imports]), " ".join(snapshot.source_titles), snapshot.source_title]
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in variants:
        text = " ".join((value or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned or [snapshot.target_statement or snapshot.user_prompt or snapshot.lean_template]


def _assistant_role_id_for_snapshot(snapshot: AssistantTargetSnapshot) -> str:
    workflow_mode = (snapshot.workflow_mode or "").lower()
    if workflow_mode == "manual_proof_check":
        return "manual_proof_assistant"
    if workflow_mode == "aggregator":
        return "aggregator_assistant"
    if workflow_mode == "compiler":
        return "compiler_assistant"
    if workflow_mode == "leanoj":
        return "leanoj_assistant"
    return "autonomous_assistant"


def _assistant_model_id(role_id: str) -> str:
    try:
        from backend.shared.api_client_manager import api_client_manager
        config = api_client_manager.get_role_config(role_id)
    except Exception:
        return ""
    if config is None:
        return ""
    return config.openrouter_model_id or config.model_id


async def _generate_assistant_selection_payload(
    *,
    prompt: str,
    task_id: str,
    assistant_role_id: str,
    assistant_model_id: str,
    max_tokens: int,
) -> dict[str, Any]:
    from backend.shared.api_client_manager import api_client_manager

    response = await api_client_manager.generate_completion(
        task_id=task_id,
        role_id=assistant_role_id,
        model=assistant_model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        _moto_disable_supercharge=True,
        _moto_reasoning_effort_override="none",
    )
    try:
        payload = parse_json(extract_response_text(response, context="assistant_proof_search"))
    except Exception as exc:
        raise _AssistantSelectionOutputError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise _AssistantSelectionOutputError("Assistant response was not a JSON object")
    return payload


def _extract_selected_search_ids(payload: dict[str, Any]) -> list[Any]:
    selected_ids = payload.get("selected_search_ids")
    if selected_ids is None:
        selected_ids = payload.get("selected_ids")
    if not isinstance(selected_ids, list):
        raise _AssistantSelectionOutputError("Assistant response missing selected_search_ids array")
    return selected_ids


def _assistant_selection_max_tokens(configured_max_tokens: int | None) -> int:
    configured = int(configured_max_tokens or 0)
    if configured <= 0:
        return _ASSISTANT_SELECTION_MAX_OUTPUT_TOKENS
    return min(configured, _ASSISTANT_SELECTION_MAX_OUTPUT_TOKENS)


def _build_assistant_selection_prompt(snapshot: AssistantTargetSnapshot, shortlist: list[AssistantProofSupport]) -> str:
    return _assistant_selection_prompt(
        snapshot,
        shortlist,
        prefix=(
            "You are the configured MOTO Assistant memory role. "
            "Return one compact JSON object only."
        ),
    )


def _build_assistant_selection_repair_prompt(
    snapshot: AssistantTargetSnapshot,
    shortlist: list[AssistantProofSupport],
    *,
    error: str,
) -> str:
    safe_error = " ".join(error.split())[:240]
    return _assistant_selection_prompt(
        snapshot,
        shortlist,
        prefix=(
            "Your previous Assistant proof-support selection was invalid: "
            f"{safe_error}. Return corrected JSON only."
        ),
    )


def _assistant_selection_prompt(
    snapshot: AssistantTargetSnapshot,
    shortlist: list[AssistantProofSupport],
    *,
    prefix: str,
) -> str:
    ids = "\n".join(_format_assistant_candidate(support) for support in shortlist)
    target = _compact_for_assistant_selection(snapshot.search_text(), 2400)
    return (
        f"{prefix}\n"
        'Required schema: {"selected_search_ids":["<exact listed id>"],"reasoning":"<=160 chars"}\n'
        f"Rules: select at most {_ASSISTANT_FINAL_PACK_LIMIT}; use only exact listed IDs; use [] if no listed proof support is genuinely useful for the target; no markdown.\n\n"
        f"TARGET:\n{target}\n\n"
        f"CANDIDATES:\n{ids}\n"
    )


def _format_assistant_candidate(support: AssistantProofSupport) -> str:
    label = support.theorem_name or support.theorem_statement or support.proof_id
    statement = "" if label == support.theorem_statement else support.theorem_statement
    parts = [f"- id: {support.search_id}", f"  label: {_compact_for_assistant_selection(label, 180)}"]
    if statement:
        parts.append(f"  statement: {_compact_for_assistant_selection(statement, 220)}")
    return "\n".join(parts)


def _compact_for_assistant_selection(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _supports_for_selected_ids(shortlist: list[AssistantProofSupport], selected_search_ids: list[str]) -> list[AssistantProofSupport]:
    by_id = {support.search_id: support for support in shortlist}
    selected: list[AssistantProofSupport] = []
    seen: set[str] = set()
    for search_id in selected_search_ids:
        if search_id in seen:
            continue
        support = by_id.get(search_id)
        if support is None:
            continue
        selected.append(support)
        seen.add(search_id)
        if len(selected) >= _ASSISTANT_FINAL_PACK_LIMIT:
            break
    return selected


def _compact_query_summary(snapshot: AssistantTargetSnapshot) -> str:
    summary = " ".join(part for part in [snapshot.workflow_phase, snapshot.current_prompt_or_topic, snapshot.writing_goal, snapshot.outline_summary, snapshot.paper_or_proof_draft_summary, snapshot.target_statement, snapshot.lean_template, snapshot.lean_error, snapshot.rejection_feedback, snapshot.source_title] if part)
    summary = " ".join(summary.split())
    return summary[:600] + ("..." if len(summary) > 600 else "")


def _filter_current_run_records(records: list[UnifiedProofSearchRecord]) -> list[UnifiedProofSearchRecord]:
    return [record for record in records if not _is_current_run_record(record)]


def _drop_current_run_supports_from_pack(pack: AssistantProofPack | None) -> AssistantProofPack | None:
    if pack is None or not pack.results:
        return pack
    filtered_results = [support for support in pack.results if not _is_current_run_support(support)]
    if len(filtered_results) == len(pack.results):
        return pack
    return pack.model_copy(update={"results": filtered_results})


def _is_current_run_record(record: UnifiedProofSearchRecord) -> bool:
    if record.corpus == "syntheticlib4":
        return False
    if (record.corpus_scope or "").strip().lower() in _CURRENT_RUN_CORPUS_SCOPES:
        return True
    active_session_id = _active_autonomous_session_id()
    return bool(active_session_id and record.session_id == active_session_id)


def _is_current_run_support(support: AssistantProofSupport) -> bool:
    if support.corpus == "syntheticlib4":
        return False
    if (support.corpus_scope or "").strip().lower() in _CURRENT_RUN_CORPUS_SCOPES:
        return True
    active_session_id = _active_autonomous_session_id()
    if not active_session_id:
        return False
    if support.session_id:
        return support.session_id == active_session_id
    parts = support.search_id.split(":")
    return len(parts) >= 3 and parts[1] == active_session_id


def _active_autonomous_session_id() -> str:
    try:
        from backend.autonomous.memory.session_manager import session_manager
        if session_manager.is_session_active:
            return str(session_manager.session_id or "").strip()
    except Exception:
        logger.debug("Assistant could not inspect active autonomous session", exc_info=True)
    return ""


def _assistant_pack_path() -> Path:
    return Path(system_config.data_dir) / "proof_search" / "assistant_latest_pack.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _delete_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


assistant_proof_search_coordinator = AssistantProofSearchCoordinator()
