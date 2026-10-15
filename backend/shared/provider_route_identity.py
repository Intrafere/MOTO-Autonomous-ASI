"""Route identity for user-facing provider/API-error Live Activity.

Every provider/API error shown to the user names the workflow mode, role,
model, provider and (for OpenRouter routes) the configured and upstream host.
Identity is carried in structured fields plus one composed ``route_identity``
prefix that is applied to display messages only. It must never be added to
exception text, ``error_summary``, backoff keys, correlation hashes or
notification keys, because those feed text classifiers and stable identities.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from backend.shared.log_redaction import redact_log_text

PROVIDER_ERROR_ACTIVITY_EVENTS = frozenset({
    "provider_retry_cooldown",
    "openai_codex_oauth_error",
    "sakana_fugu_error",
    "oauth_provider_error",
    "oauth_provider_usage_limited",
    "provider_usage_limit_resumed",
    "hung_connection_alert",
    "openrouter_rate_limit",
    "openrouter_privacy_error",
    "account_credits_exhausted",
    "boost_credits_exhausted",
    "openrouter_fallback",
    "openrouter_fallback_failed",
    "free_model_rotated",
    "free_model_auto_selector_used",
    "free_models_exhausted",
    "provider_repair_required",
    "provider_role_disabled",
    "autonomous_proof_provider_paused",
    "autonomous_proof_provider_resumed",
    "proof_run_provider_paused",
    "proof_run_provider_resumed",
    "proof_retry_scheduled",
    "proof_retry_started",
    "proof_prune_provider_paused",
    "proof_prune_repair_required",
    "assistant_proof_pack_failed",
    "solution_path_proposal_user_repair_required",
    "leanoj_model_call_failed",
    "leanoj_provider_paused",
    "leanoj_provider_resumed",
    "proof_competition_unavailable",
    "proof_competitor_unavailable",
    "auto_research_stopped",
})

VALID_WORKFLOW_MODES = frozenset({
    "autonomous",
    "aggregator",
    "compiler",
    "leanoj",
    "manual_proof_check",
})
WORKFLOW_LABELS = {
    "autonomous": "Autonomous",
    "aggregator": "Aggregator",
    "compiler": "Compiler",
    "leanoj": "LeanOJ",
    "manual_proof_check": "Manual proof check",
}
PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "lm_studio": "LM Studio",
    "openai_codex_oauth": "OpenAI Codex",
    "xai_grok_oauth": "xAI Grok",
    "sakana_fugu": "Sakana Fugu",
}
MODEL_PROVIDERS = frozenset(PROVIDER_LABELS)
OPENROUTER_FAMILY_PROVIDERS = frozenset({"openrouter"})
PLACEHOLDER_PROVIDERS = frozenset({"", "unknown", "configured_role", "provider"})
# Retry activity whose provider is a placeholder or a non-model service is a
# workflow-loop or storage retry: it names mode and role only, never a model.
NON_MODEL_PROVIDERS = PLACEHOLDER_PROVIDERS | frozenset({"chromadb"})
HOST_ROUTING_VALUES = frozenset({"pinned", "auto", "auto_fallback", "direct"})
ROUTE_KIND_LABELS = {
    "boost": "boost",
    "fallback": "LM Studio fallback",
    "lm_studio_fallback": "LM Studio fallback",
    "free_rotation": "free rotation",
    "auto_selector": "auto-selector",
    "embeddings": "embeddings",
}
ROUTE_IDENTITY_MAX_CHARS = 480
ROUTE_IDENTITY_FIELD_NAMES = (
    "workflow_mode",
    "role_id",
    "model_role_id",
    "role_label",
    "model",
    "provider",
    "provider_label",
    "route_kind",
    "task_id",
    "configured_model",
    "configured_provider",
    "configured_host_provider",
    "effective_host_provider",
    "host_routing",
    "provider_request_id",
    "route_identity",
)

_SEPARATOR = " · "
_HOST_ARROW = " → "
_FIELD_LIMITS = {
    "workflow_mode": 40,
    "role_id": 160,
    "model_role_id": 160,
    "role_label": 120,
    "model": 200,
    "provider": 60,
    "provider_label": 60,
    "route_kind": 40,
    "task_id": 120,
    "configured_model": 200,
    "configured_provider": 60,
    "configured_host_provider": 80,
    "effective_host_provider": 80,
    "host_routing": 20,
    "provider_request_id": 80,
}
_EXACT_ROLE_LABELS = {
    "aggregator_validator": "Validator",
    "compiler_validator": "Validator",
    "compiler_writer": "Writing Submitter",
    "compiler_high_param": "Rigor & Proofs",
    "compiler_rigor_formalization": "Rigor Formalization",
    "compiler_rigor_novelty": "Rigor Novelty",
    "paper_critic": "Paper Critic",
    "critique_validator": "Critique Validator",
    "critique_cleanup": "Critique Cleanup",
    "aggregator_single_model": "Single-Model Submitter",
    "embeddings": "Embeddings",
    "rag_vector_search": "RAG Vector Search",
    "manual_proof_assistant": "Assistant (manual proof)",
    "aggregator_assistant": "Assistant (aggregator)",
    "autonomous_assistant": "Assistant (autonomous)",
    "compiler_assistant": "Assistant (compiler)",
    "leanoj_assistant": "Assistant (LeanOJ)",
}
_PROOF_ROLE_KINDS = (
    "candidate_list_validator",
    "identification",
    "formalization",
    "lemma_search",
    "novelty",
    "prune_proposer",
    "prune_validator",
    "framing_gate",
)
_PROOF_CONTEXTS = (
    ("compiler_manual_paper", "manual paper"),
    ("compiler_aggregator", "manual aggregator"),
    ("manual_brainstorm", "manual brainstorm"),
    ("manual_paper", "manual paper"),
    ("brainstorm", "brainstorm"),
    ("paper", "paper"),
)
_MODE_PREFIXES = ("aggregator_", "compiler_", "autonomous_", "leanoj_", "manual_")
_RUN_DIGEST_SUFFIX = re.compile(r"_[0-9a-f]{10}$")
_SUBMITTER_ROLE = re.compile(r"^(?:aggregator|agg)_(?:submitter|sub)_?(\d+)$")
_RIGOR_SECONDARY_ROLE = re.compile(r"^compiler_rigor_secondary_.*_(\d+)$")
_PROOF_COMPETITOR_SUFFIX = re.compile(r"_competition_(\d+)_[0-9a-f]+$")


def _clean(value: Any, field: str = "") -> str:
    if value is None:
        return ""
    limit = _FIELD_LIMITS.get(field)
    return redact_log_text(str(value), limit).strip()


def _title(words: str) -> str:
    title = " ".join(part.capitalize() for part in words.replace("_", " ").split())
    return title.replace("Leanoj", "LeanOJ").replace("Lm Studio", "LM Studio")


def workflow_label(mode: Any) -> str:
    """Return the user-facing label for a workflow mode."""
    key = _clean(mode).lower()
    return WORKFLOW_LABELS.get(key, _clean(mode))


def provider_label(provider: Any, fallback: Any = "") -> str:
    """Return the user-facing label for a provider id."""
    label = _clean(fallback, "provider_label")
    if label:
        return label
    key = _clean(provider, "provider").lower()
    return PROVIDER_LABELS.get(key, _clean(provider, "provider"))


def is_placeholder_provider(provider: Any) -> bool:
    return _clean(provider).lower() in PLACEHOLDER_PROVIDERS


def is_model_provider(provider: Any) -> bool:
    """Return whether a provider id names a model-serving provider route."""
    return _clean(provider).lower() in MODEL_PROVIDERS


def _base_role_label(role_id: str) -> str:
    role = role_id.strip()
    lowered = role.lower()
    if not lowered:
        return ""
    if lowered in _EXACT_ROLE_LABELS:
        return _EXACT_ROLE_LABELS[lowered]
    submitter = _SUBMITTER_ROLE.match(lowered)
    if submitter:
        return f"Submitter {submitter.group(1)}"
    if lowered.endswith("_direct"):
        return "direct call"
    secondary = _RIGOR_SECONDARY_ROLE.match(lowered)
    if secondary:
        return f"Rigor Secondary Formalizer {secondary.group(1)}"
    if lowered.startswith("autonomous_proof_"):
        rest = lowered[len("autonomous_proof_"):]
        competitor = _PROOF_COMPETITOR_SUFFIX.search(rest)
        if competitor:
            rest = rest[: competitor.start()]
        for kind in _PROOF_ROLE_KINDS:
            if rest == kind or rest.startswith(f"{kind}_"):
                context = rest[len(kind):].lstrip("_")
                details = [
                    context_label
                    for prefix, context_label in _PROOF_CONTEXTS
                    if context == prefix or context.startswith(f"{prefix}_")
                ][:1]
                if competitor:
                    details.append(f"competitor {competitor.group(1)}")
                label = f"Proof {_title(kind)}"
                return f"{label} ({', '.join(details)})" if details else label
    if lowered.endswith("_assistant"):
        owner = lowered[: -len("_assistant")]
        return f"Assistant ({_title(owner)})" if owner else "Assistant"
    trimmed = _RUN_DIGEST_SUFFIX.sub("", lowered)
    for prefix in _MODE_PREFIXES:
        if trimmed.startswith(prefix) and len(trimmed) > len(prefix):
            trimmed = trimmed[len(prefix):]
            break
    return _title(trimmed)


def role_label(role_id: Any, route_kind: Any = "") -> str:
    """Return a distinguishing user-facing role label, noting non-primary routes."""
    base = _base_role_label(_clean(role_id, "role_id"))
    if not base:
        return ""
    kind_label = ROUTE_KIND_LABELS.get(_clean(route_kind, "route_kind").lower(), "")
    if not kind_label or kind_label.lower() == base.lower():
        return base
    if base.endswith(")"):
        return f"{base[:-1]}, {kind_label})"
    return f"{base} ({kind_label})"


def host_fields(
    provider: Any,
    *,
    configured_host: Optional[str] = None,
    upstream_host: Any = "",
    auto_fallback_from: Any = "",
) -> dict[str, str]:
    """Return host routing fields for one provider call.

    ``configured_host`` is the host pinned in settings; ``""`` means OpenRouter
    auto routing and ``None`` means the configured host is unknown (no host
    routing is reported then, so legacy-shaped data is never mislabelled auto).
    """
    provider_key = _clean(provider, "provider").lower()
    if provider_key not in OPENROUTER_FAMILY_PROVIDERS:
        if is_placeholder_provider(provider_key):
            return {}
        return {"host_routing": "direct"}
    upstream = _clean(upstream_host, "effective_host_provider")
    fallback_from = _clean(auto_fallback_from, "configured_host_provider")
    if fallback_from:
        fields = {
            "configured_host_provider": fallback_from,
            "host_routing": "auto_fallback",
        }
        if upstream:
            fields["effective_host_provider"] = upstream
        return fields
    if configured_host is None and not upstream:
        return {}
    pinned = _clean(configured_host or "", "configured_host_provider")
    if pinned:
        return {
            "configured_host_provider": pinned,
            "effective_host_provider": upstream or pinned,
            "host_routing": "pinned",
        }
    fields = {"configured_host_provider": "", "host_routing": "auto"}
    if upstream:
        fields["effective_host_provider"] = upstream
    return fields


def format_host_segment(fields: Mapping[str, Any]) -> str:
    """Render the host segment of the route identity prefix."""
    routing = _clean(fields.get("host_routing")).lower()
    configured = _clean(fields.get("configured_host_provider"), "configured_host_provider")
    effective = _clean(fields.get("effective_host_provider"), "effective_host_provider")
    request_id = _clean(fields.get("provider_request_id"), "provider_request_id")
    not_reported = (
        f"upstream not reported; request {request_id}"
        if request_id
        else "upstream not reported"
    )
    if routing == "pinned":
        host = configured or effective
        if not host:
            return ""
        if effective and configured and effective != configured:
            return f"host {configured} (pinned){_HOST_ARROW}{effective}"
        return f"host {host} (pinned)"
    if routing == "auto":
        return f"host auto{_HOST_ARROW}{effective}" if effective else f"host auto ({not_reported})"
    if routing == "auto_fallback":
        unavailable = f"pinned {configured} unavailable" if configured else "pinned host unavailable"
        if effective:
            return f"host auto ({unavailable}){_HOST_ARROW}{effective}"
        return f"host auto ({unavailable}; {not_reported})"
    return ""


def compose_route_identity(fields: Mapping[str, Any]) -> str:
    """Compose the bounded ``[mode · role · model via provider · host]`` prefix."""
    segments = []
    mode = workflow_label(fields.get("workflow_mode"))
    if mode:
        segments.append(mode)
    label = _clean(fields.get("role_label"), "role_label") or role_label(
        fields.get("model_role_id") or fields.get("role_id"),
        fields.get("route_kind"),
    )
    if label:
        segments.append(label)
    model = _clean(fields.get("model"), "model")
    provider = ""
    if _clean(fields.get("provider")).lower() not in NON_MODEL_PROVIDERS:
        provider = provider_label(fields.get("provider"), fields.get("provider_label"))
    if model and provider:
        segments.append(f"{model} via {provider}")
    elif model or provider:
        segments.append(model or provider)
    host = format_host_segment(fields)
    if host:
        segments.append(host)
    if not segments:
        return ""
    body = _SEPARATOR.join(segments)
    if len(body) > ROUTE_IDENTITY_MAX_CHARS - 2:
        body = body[: ROUTE_IDENTITY_MAX_CHARS - 5].rstrip() + "..."
    return f"[{body}]"


def route_identity_fields(
    *,
    workflow_mode: Any = "",
    role_id: Any = "",
    model_role_id: Any = "",
    model: Any = "",
    provider: Any = "",
    provider_label_text: Any = "",
    route_kind: Any = "",
    task_id: Any = "",
    configured_model: Any = "",
    configured_provider: Any = "",
    configured_host_provider: Optional[str] = None,
    upstream_host_provider: Any = "",
    auto_fallback_from_host: Any = "",
    provider_request_id: Any = "",
) -> dict[str, str]:
    """Return non-empty structured identity fields plus the composed prefix."""
    fields: dict[str, str] = {}
    mode = _clean(workflow_mode, "workflow_mode").lower()
    if mode in VALID_WORKFLOW_MODES:
        fields["workflow_mode"] = mode
    for key, value in (
        ("role_id", role_id),
        ("model_role_id", model_role_id),
        ("model", model),
        ("route_kind", route_kind),
        ("task_id", task_id),
        ("configured_model", configured_model),
        ("configured_provider", configured_provider),
        ("provider_request_id", provider_request_id),
    ):
        cleaned = _clean(value, key)
        if cleaned:
            fields[key] = cleaned
    provider_id = _clean(provider, "provider")
    if provider_id and not is_placeholder_provider(provider_id):
        fields["provider"] = provider_id
        fields["provider_label"] = provider_label(provider_id, provider_label_text)
    elif _clean(provider_label_text):
        fields["provider_label"] = _clean(provider_label_text, "provider_label")
    if fields.get("model_role_id") == fields.get("role_id"):
        fields.pop("model_role_id", None)
    label = role_label(fields.get("model_role_id") or fields.get("role_id"), fields.get("route_kind"))
    if label:
        fields["role_label"] = label
    fields.update(host_fields(
        fields.get("provider", ""),
        configured_host=configured_host_provider,
        upstream_host=upstream_host_provider,
        auto_fallback_from=auto_fallback_from_host,
    ))
    if (
        fields.get("model_role_id")
        and fields.get("model_role_id") != fields.get("role_id")
        and "role_id" not in fields
    ):
        fields["role_id"] = fields["model_role_id"]
    identity = compose_route_identity(fields)
    if identity:
        fields["route_identity"] = identity
    return fields


def route_identity_from_error(exc: Optional[BaseException], *, max_depth: int = 5) -> dict[str, Any]:
    """Extract raw route identity keyword arguments from a typed provider error.

    Uses duck typing so this module never imports the API client manager. The
    outermost error wins; causes only fill fields that are still missing.
    """
    found: dict[str, Any] = {}
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    depth = 0
    while current is not None and depth < max_depth and id(current) not in seen:
        seen.add(id(current))
        depth += 1
        route = getattr(current, "route", None)
        candidates: list[tuple[str, Any]] = []
        if route is not None:
            candidates.extend((
                ("provider", getattr(route, "provider", "")),
                ("model", getattr(route, "model", "")),
                ("role_id", getattr(route, "role_id", "")),
                ("task_id", getattr(route, "task_id", "")),
                ("route_kind", getattr(route, "route_kind", "")),
                ("configured_provider", getattr(route, "configured_provider", "")),
                ("configured_model", getattr(route, "configured_model", "")),
                ("upstream_host_provider", getattr(route, "upstream_host_provider", "")),
                ("auto_fallback_from_host", getattr(route, "auto_fallback_from_host", "")),
                ("provider_request_id", getattr(route, "provider_request_id", "")),
            ))
            if "configured_host_provider" not in found:
                found["configured_host_provider"] = _clean(
                    getattr(route, "host_provider", ""),
                    "configured_host_provider",
                )
        candidates.extend((
            ("provider", getattr(current, "provider", "")),
            ("provider_label_text", getattr(current, "provider_label", "")),
            ("model", getattr(current, "model", "")),
            ("role_id", getattr(current, "role_id", "")),
            ("task_id", getattr(current, "task_id", "")),
            ("route_kind", getattr(current, "route_kind", "")),
            ("configured_provider", getattr(current, "configured_provider", "")),
            ("configured_model", getattr(current, "configured_model", "")),
            ("upstream_host_provider", getattr(current, "upstream_host_provider", "")),
            ("auto_fallback_from_host", getattr(current, "auto_fallback_from_host", "")),
            ("provider_request_id", getattr(current, "provider_request_id", "")),
        ))
        explicit_host = getattr(current, "configured_host_provider", None)
        if isinstance(explicit_host, str) and "configured_host_provider" not in found:
            found["configured_host_provider"] = _clean(explicit_host, "configured_host_provider")
        for key, value in candidates:
            if not isinstance(value, str):
                continue
            cleaned = _clean(value)
            if not cleaned:
                continue
            if key == "provider" and is_placeholder_provider(cleaned):
                continue
            found.setdefault(key, cleaned)
        current = getattr(current, "cause", None) or current.__cause__
    return found


def merge_route_identity(payload: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing identity fields without overriding producer-supplied values."""
    merged = dict(payload)
    for key, value in fields.items():
        if value in (None, ""):
            continue
        existing = merged.get(key)
        if existing in (None, "") or (key == "provider" and is_placeholder_provider(existing)):
            merged[key] = value
    return merged


def with_route_identity(message: Any, fields: Any) -> str:
    """Prefix a display message with route identity exactly once."""
    text = str(message or "").strip()
    identity = (
        str(fields.get("route_identity") or "")
        if isinstance(fields, Mapping)
        else str(fields or "")
    ).strip()
    if not identity:
        return text
    if text.startswith(identity) or identity in text:
        return text
    return f"{identity} {text}".strip()
