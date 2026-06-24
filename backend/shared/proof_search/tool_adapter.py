"""OpenAI-compatible proof-search tool adapter."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.shared.config import system_config
from backend.shared.proof_search.models import (
    ProofSearchCorpus,
    ProofSearchRequest,
    UnifiedProofSearchRecord,
    default_proof_search_corpora,
)
from backend.shared.proof_search.search_service import (
    ProofSearchService,
    proof_search_service,
)

MAX_PROOF_SEARCH_RESULTS = 7

SEARCH_LEAN_PROOFS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_lean_proofs",
        "description": (
            "Search MOTO local proof history and SyntheticLib4 proof records for "
            "prompt-relevant Lean proof patterns. Use this only for active proof "
            "formalization or proof repair work. Results are capped at 7 combined "
            "proofs and include provenance plus theorem/Lean-code hashes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["overview", "search", "hydrate", "attest_usage"],
                },
                "query": {"type": "string"},
                "goal_statement": {"type": "string"},
                "lean_template": {"type": "string"},
                "imports": {"type": "array", "items": {"type": "string"}},
                "dependency_names": {"type": "array", "items": {"type": "string"}},
                "corpora": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["moto", "manual", "autonomous", "leanoj", "syntheticlib4"],
                    },
                },
                "verified_only": {"type": "boolean"},
                "include_partial": {"type": "boolean"},
                "include_failed": {"type": "boolean"},
                "novelty_filters": {"type": "array", "items": {"type": "string"}},
                "module_filters": {"type": "array", "items": {"type": "string"}},
                "source_filters": {"type": "array", "items": {"type": "string"}},
                "exclude_ids": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_PROOF_SEARCH_RESULTS},
                "cursor": {"type": "string"},
                "hydrate_lean_code": {"type": "boolean"},
                "search_mode": {
                    "type": "string",
                    "enum": ["auto", "exact", "lexical", "text", "semantic", "hybrid"],
                },
                "source": {
                    "type": "string",
                    "enum": ["moto", "manual", "autonomous", "leanoj", "syntheticlib4"],
                },
                "proof_id": {"type": "string"},
                "fingerprint": {"type": "string"},
                "session_id": {"type": "string"},
                "usage_attestation": {
                    "type": "object",
                    "properties": {
                        "retrieval_batch_id": {"type": "string"},
                        "used_fingerprints": {"type": "array", "items": {"type": "string"}},
                        "unused_fingerprints": {"type": "array", "items": {"type": "string"}},
                        "used_proofs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "fingerprint": {"type": "string"},
                                    "theorem_statement_hash": {"type": "string"},
                                    "lean_code_hash": {"type": "string"},
                                },
                            },
                        },
                        "entire_code_used": {"type": "boolean"},
                        "moto_artifact_hash": {"type": "string"},
                        "usage_type": {"type": "string"},
                    },
                },
            },
            "required": ["action"],
        },
    },
}

_VALID_CORPORA: set[str] = {"moto", "manual", "leanoj", "syntheticlib4"}
_CORPUS_ALIASES = {"autonomous": "moto"}


async def execute_search_lean_proofs(
    arguments: dict[str, Any] | str,
    *,
    service: ProofSearchService | None = None,
    usage_root: Path | None = None,
) -> dict[str, Any]:
    """Execute one `search_lean_proofs` tool call and return JSON-safe output."""
    active_service = service or proof_search_service
    try:
        args = _coerce_arguments(arguments)
        action = str(args.get("action") or "").strip()
        if action == "overview":
            overview = await active_service.overview()
            return _tool_success(action, overview=overview.model_dump(mode="json"))
        if action == "search":
            request = _build_search_request(args)
            response = await active_service.search(request)
            return _tool_success(
                action,
                results=[_record_to_tool_result(record) for record in response.results],
                next_cursor=response.next_cursor,
                searched_corpora=response.searched_corpora,
                corpus_counts=response.corpus_counts,
                ranking_notes=response.ranking_notes,
                weak_result_warning=response.weak_result_warning,
            )
        if action == "hydrate":
            record = await _hydrate_record(active_service, args)
            if record is None:
                return _tool_error(action, "Proof record not found.")
            return _tool_success(action, results=[_record_to_tool_result(record)])
        if action == "attest_usage":
            attestation = await _persist_usage_attestation(args, usage_root=usage_root)
            return _tool_success(action, usage_attestation=attestation)
        return _tool_error(action or "unknown", "Unsupported search_lean_proofs action.")
    except Exception as exc:
        return _tool_error("error", str(exc))


def _coerce_arguments(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        parsed = json.loads(arguments or "{}")
    else:
        parsed = dict(arguments or {})
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _build_search_request(args: dict[str, Any]) -> ProofSearchRequest:
    goal_parts = [_string(args.get("goal_statement"))]
    lean_template = _string(args.get("lean_template"))
    if lean_template:
        goal_parts.append(lean_template)
    return ProofSearchRequest(
        query=_string(args.get("query")),
        goal_statement="\n\n".join(part for part in goal_parts if part),
        imports=_string_list(args.get("imports")),
        dependency_names=_string_list(args.get("dependency_names")),
        corpora=_normalize_corpora(args.get("corpora")),
        verified_only=bool(args.get("verified_only", True)),
        include_partial=bool(args.get("include_partial", False)),
        include_failed=bool(args.get("include_failed", False)),
        novelty_filters=_string_list(args.get("novelty_filters")),
        module_filters=_string_list(args.get("module_filters")),
        source_filters=_string_list(args.get("source_filters")),
        limit=_normalize_limit(args.get("limit")),
        cursor=_optional_string(args.get("cursor")),
        exclude_ids=_string_list(args.get("exclude_ids")),
        hydrate_lean_code=bool(args.get("hydrate_lean_code", True)),
        search_mode=_normalize_search_mode(args.get("search_mode")),
    )


async def _hydrate_record(
    service: ProofSearchService,
    args: dict[str, Any],
) -> UnifiedProofSearchRecord | None:
    source = _normalize_corpus(args.get("source") or _first(args.get("corpora")) or "")
    proof_id = _string(args.get("proof_id") or args.get("fingerprint"))
    if not source:
        raise ValueError("Hydrate action requires 'source' or a single corpus.")
    if not proof_id:
        raise ValueError("Hydrate action requires 'proof_id' or 'fingerprint'.")
    return await service.get_record(
        corpus=source,
        proof_id=proof_id,
        session_id=_optional_string(args.get("session_id")),
    )


async def _persist_usage_attestation(
    args: dict[str, Any],
    *,
    usage_root: Path | None = None,
) -> dict[str, Any]:
    attestation = dict(args.get("usage_attestation") or {})
    now = datetime.now(timezone.utc).isoformat()
    used_proofs = _normalize_used_proofs(attestation)
    payload = {
        "schema_version": "moto.proof_search_usage_attestation.v1",
        "created_at": now,
        "retrieval_batch_id": _string(attestation.get("retrieval_batch_id")),
        "used_fingerprints": [proof["fingerprint"] for proof in used_proofs],
        "used_proofs": used_proofs,
        "unused_fingerprints": _string_list(attestation.get("unused_fingerprints")),
        "usage_type": _string(attestation.get("usage_type") or "whole_proof_dependency"),
        "entire_code_used": bool(attestation.get("entire_code_used", False)),
        "moto_artifact_hash": _string(attestation.get("moto_artifact_hash")),
        "submitted": False,
    }
    if not used_proofs:
        raise ValueError("Usage attestation requires at least one used fingerprint.")
    if payload["entire_code_used"] and any(
        not proof["theorem_statement_hash"] or not proof["lean_code_hash"]
        for proof in used_proofs
    ):
        raise ValueError(
            "Whole-code usage attestations require theorem_statement_hash and lean_code_hash for every used proof."
        )
    root = usage_root or Path(system_config.data_dir) / "proof_search"
    path = root / "usage_attestations.jsonl"

    def _append() -> None:
        root.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    await asyncio.to_thread(_append)
    return {**payload, "persisted": True}


def _normalize_used_proofs(attestation: dict[str, Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    raw_used_proofs = attestation.get("used_proofs") or []
    if isinstance(raw_used_proofs, list):
        for item in raw_used_proofs:
            if not isinstance(item, dict):
                continue
            fingerprint = _string(item.get("fingerprint"))
            if not fingerprint or fingerprint in seen:
                continue
            normalized.append(
                {
                    "fingerprint": fingerprint,
                    "theorem_statement_hash": _string(item.get("theorem_statement_hash")),
                    "lean_code_hash": _string(item.get("lean_code_hash")),
                }
            )
            seen.add(fingerprint)
    for fingerprint in _string_list(attestation.get("used_fingerprints")):
        if fingerprint in seen:
            continue
        normalized.append(
            {
                "fingerprint": fingerprint,
                "theorem_statement_hash": "",
                "lean_code_hash": "",
            }
        )
        seen.add(fingerprint)
    return normalized


def _record_to_tool_result(record: UnifiedProofSearchRecord) -> dict[str, Any]:
    return {
        "search_id": record.search_id,
        "corpus": record.corpus,
        "corpus_scope": record.corpus_scope,
        "source_kind": record.source_kind,
        "proof_id": record.proof_id,
        "fingerprint": record.external_fingerprint,
        "release_id": record.release_id,
        "session_id": record.session_id,
        "source_type": record.source_type,
        "source_id": record.source_id,
        "source_title": record.source_title,
        "display_title": record.display_title,
        "theorem_name": record.theorem_name,
        "theorem_statement": record.theorem_statement,
        "informal_statement": record.informal_statement,
        "proof_description": record.proof_description,
        "formal_sketch": record.formal_sketch,
        "imports": record.imports,
        "dependency_names": record.dependency_names,
        "topic_tags": record.topic_tags,
        "domain_tags": record.domain_tags,
        "module": record.module,
        "source_path": record.source_path,
        "novelty_tier": record.novelty_tier,
        "novelty_reasoning": record.novelty_reasoning,
        "lean_code": record.lean_code,
        "theorem_statement_hash": record.theorem_statement_hash,
        "lean_code_hash": record.lean_code_hash,
        "canonical_uri": record.canonical_uri,
        "metadata": record.metadata,
    }


def _tool_success(action: str, **payload: Any) -> dict[str, Any]:
    return {
        "success": True,
        "action": action,
        "overview": payload.pop("overview", None),
        "results": payload.pop("results", []),
        "next_cursor": payload.pop("next_cursor", None),
        "searched_corpora": payload.pop("searched_corpora", []),
        "corpus_counts": payload.pop("corpus_counts", {}),
        "ranking_notes": payload.pop("ranking_notes", ""),
        "weak_result_warning": payload.pop("weak_result_warning", None),
        "usage_attestation": payload.pop("usage_attestation", None),
        "error": None,
        **payload,
    }


def _tool_error(action: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "action": action,
        "overview": None,
        "results": [],
        "next_cursor": None,
        "searched_corpora": [],
        "corpus_counts": {},
        "ranking_notes": "",
        "weak_result_warning": None,
        "usage_attestation": None,
        "error": message,
    }


def _normalize_corpora(value: Any) -> list[ProofSearchCorpus]:
    corpora = [_normalize_corpus(item) for item in _string_list(value)]
    valid = [corpus for corpus in corpora if corpus]
    if valid:
        from backend.shared.config import system_config

        return [
            corpus
            for corpus in valid
            if (
                (corpus == "syntheticlib4" and system_config.syntheticlib4_enabled)
                or (corpus != "syntheticlib4" and system_config.agent_conversation_memory_enabled)
            )
        ]
    return default_proof_search_corpora()


def _normalize_corpus(value: Any) -> ProofSearchCorpus | None:
    corpus = _CORPUS_ALIASES.get(_string(value), _string(value))
    if corpus not in _VALID_CORPORA:
        return None
    return corpus  # type: ignore[return-value]


def _normalize_search_mode(value: Any) -> str:
    mode = _string(value) or "hybrid"
    if mode in {"lexical", "text"}:
        return "text"
    if mode == "exact":
        return "exact"
    return "hybrid"


def _normalize_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = MAX_PROOF_SEARCH_RESULTS
    return min(max(limit, 1), MAX_PROOF_SEARCH_RESULTS)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    text = _string(value)
    return text or None


def _string(value: Any) -> str:
    return str(value or "").strip()


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None
