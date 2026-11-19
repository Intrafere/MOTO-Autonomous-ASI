"""Typed model-output contract failures and visible output normalizations.

A model-output contract failure means a model answered, but its answer did not
match the JSON contract its role requires. It is never a provider failure, a
context overflow, or a semantic decision; callers retry it on the role's own
bounded contract-retry path.

An output normalization is one of the few visible repairs the shared parser and
``json_key_recovery`` may apply (a stray-character drop, a trailing comma, or one
relabeled key). Each applied normalization is logged as metadata and published
as the additive ``model_output_normalized`` Live Activity event.

Top-level imports stay standard-library only: ``json_parser`` imports this module
and is itself imported very early.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_OUTPUT_NORMALIZED_EVENT = "model_output_normalized"
# Longest stop-aware wait between tries after repeated model-output contract
# failures. It bounds one wait, never the number of tries.
CONTRACT_RETRY_MAX_BACKOFF_SECONDS = 30

CONTRACT_STAGES = frozenset({"syntax", "shape", "missing_key", "invalid_value", "variant"})
OUTPUT_REPAIR_KINDS = frozenset({"key_label", "stray_characters", "trailing_comma"})

_MODEL_TOKEN_LIMIT = 64
_ITEM_PATH_LIMIT = 120
_MESSAGE_LIMIT = 1_600
_NON_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")
_NON_PATH_CHARS = re.compile(r"[^A-Za-z0-9_.\-\[\]]")


def safe_model_token(value: Any, limit: int = _MODEL_TOKEN_LIMIT) -> str:
    """Return a model-authored key or ID reduced to key characters and capped."""
    text = _NON_KEY_CHARS.sub("", str(value if value is not None else ""))
    return text[:limit]


def safe_item_path(value: Any) -> str:
    """Return an item path such as ``theorems[2]`` reduced to path characters."""
    text = _NON_PATH_CHARS.sub("", str(value if value is not None else ""))
    return text[:_ITEM_PATH_LIMIT]


def _quoted(value: Any) -> str:
    token = safe_model_token(value)
    return f"'{token}'" if token else "''"


def _quoted_list(values: Iterable[Any]) -> str:
    return ", ".join(_quoted(value) for value in values)


@dataclass(frozen=True)
class OutputRepair:
    """One applied visible normalization of a model's JSON output.

    ``kind`` is ``key_label`` (``from_key`` relabeled as ``to_key``),
    ``stray_characters`` (``removed_chars`` characters dropped outside strings),
    or ``trailing_comma``. Removed characters themselves are never recorded.
    """

    kind: str
    item_path: str = ""
    item_id: str = ""
    from_key: str = ""
    to_key: str = ""
    similarity: Optional[float] = None
    removed_chars: int = 0

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_REPAIR_KINDS:
            raise ValueError(f"Unknown output repair kind: {self.kind}")
        object.__setattr__(self, "item_path", safe_item_path(self.item_path))
        object.__setattr__(self, "item_id", safe_model_token(self.item_id))
        object.__setattr__(self, "from_key", safe_model_token(self.from_key))
        object.__setattr__(self, "to_key", safe_model_token(self.to_key))

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "item_path": self.item_path}
        if self.item_id:
            payload["item_id"] = self.item_id
        if self.kind == "key_label":
            payload["from_key"] = self.from_key
            payload["to_key"] = self.to_key
        if self.kind == "stray_characters":
            payload["removed_chars"] = self.removed_chars
        return payload

    def _location(self) -> str:
        if self.item_path and self.item_id:
            return f" ({self.item_path} '{self.item_id}')"
        if self.item_path:
            return f" ({self.item_path})"
        if self.item_id:
            return f" ('{self.item_id}')"
        return ""

    def describe(self, model_label: str) -> str:
        """Return one display sentence naming what was accepted or ignored."""
        owner = f"{model_label}'s" if model_label else "the model's"
        if self.kind == "key_label":
            return (
                f"Accepted {owner} mislabeled field '{self.from_key}' as "
                f"'{self.to_key}'{self._location()}."
            )
        if self.kind == "stray_characters":
            count = max(1, int(self.removed_chars or 0))
            noun = "character" if count == 1 else "characters"
            return (
                f"Ignored {count} stray {noun} outside JSON strings in {owner} "
                f"reply{self._location()}."
            )
        return f"Ignored a trailing comma in {owner} reply{self._location()}."


@dataclass(frozen=True)
class ItemContractProblem:
    """The contract problems of one item in a model-authored list."""

    item_path: str = ""
    item_id: str = ""
    missing_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    invalid_fields: tuple[str, ...] = ()
    detail: str = ""

    def describe(self) -> str:
        location = safe_item_path(self.item_path) or "item"
        if self.item_id:
            location = f"{location} {_quoted(self.item_id)}"
        parts = []
        if self.missing_keys:
            noun = "field" if len(self.missing_keys) == 1 else "fields"
            parts.append(f"missing required {noun} {_quoted_list(self.missing_keys)}")
        if self.invalid_fields:
            noun = "field" if len(self.invalid_fields) == 1 else "fields"
            parts.append(f"invalid {noun} {_quoted_list(self.invalid_fields)}")
        if self.unknown_keys:
            noun = "key" if len(self.unknown_keys) == 1 else "keys"
            parts.append(f"unexpected {noun} {_quoted_list(self.unknown_keys)}")
        if self.detail:
            parts.append(self.detail.strip().rstrip("."))
        return f"{location}: {'; '.join(parts) or 'not a valid entry'}"


class ModelOutputContractError(ValueError):
    """A model's answer did not match its role's JSON contract.

    ``summary`` must be system-authored text. Model-authored names (keys, IDs)
    belong in the structured fields, which are quoted, reduced to key characters,
    and capped before they reach the message. The message never contains raw
    output excerpts, so it is safe as model-facing retry text.
    """

    def __init__(
        self,
        summary: str,
        *,
        role: str = "",
        stage: str = "shape",
        item_path: str = "",
        item_id: str = "",
        missing_keys: Sequence[str] = (),
        unknown_keys: Sequence[str] = (),
        invalid_fields: Sequence[str] = (),
        problems: Sequence[ItemContractProblem] = (),
        repairs_applied: Sequence[OutputRepair] = (),
        retryable: bool = True,
    ) -> None:
        if stage not in CONTRACT_STAGES:
            raise ValueError(f"Unknown model-output contract stage: {stage}")
        self.role = str(role or "")
        self.stage = stage
        self.item_path = safe_item_path(item_path)
        self.item_id = safe_model_token(item_id)
        self.missing_keys = tuple(safe_model_token(key) for key in missing_keys)
        self.unknown_keys = tuple(safe_model_token(key) for key in unknown_keys)
        self.invalid_fields = tuple(safe_model_token(key) for key in invalid_fields)
        self.problems = tuple(problems)
        self.repairs_applied = tuple(repairs_applied)
        self.retryable = bool(retryable)
        self.summary = str(summary or "").strip()
        super().__init__(self._compose_message())

    def _compose_message(self) -> str:
        parts = [self.summary.rstrip(".") or "The response did not match the required JSON format"]
        own = ItemContractProblem(
            item_path=self.item_path,
            item_id=self.item_id,
            missing_keys=self.missing_keys,
            unknown_keys=self.unknown_keys,
            invalid_fields=self.invalid_fields,
        )
        if self.missing_keys or self.unknown_keys or self.invalid_fields:
            parts.append(own.describe())
        parts.extend(problem.describe() for problem in self.problems)
        message = "; ".join(parts) + "."
        if len(message) > _MESSAGE_LIMIT:
            message = message[: _MESSAGE_LIMIT - 15].rstrip() + "...[truncated]"
        return message

    @property
    def field_summary(self) -> str:
        """Short field-level summary for activity lines (no role or route text)."""
        names = [*self.missing_keys, *self.invalid_fields]
        for problem in self.problems:
            names.extend(problem.missing_keys)
            names.extend(problem.invalid_fields)
        unique = list(dict.fromkeys(name for name in names if name))
        return ", ".join(unique[:8])

    @classmethod
    def from_item_problems(
        cls,
        *,
        role: str = "",
        item_noun_plural: str,
        total: int,
        problems: Sequence[ItemContractProblem],
        stage: str = "shape",
        repairs_applied: Sequence[OutputRepair] = (),
    ) -> "ModelOutputContractError":
        """Build ``"N of M <items> were invalid: ..."`` naming each bad item."""
        count = len(problems)
        verb = "was" if count == 1 else "were"
        return cls(
            f"{count} of {max(int(total), count)} {item_noun_plural} {verb} invalid",
            role=role,
            stage=stage,
            problems=problems,
            repairs_applied=repairs_applied,
        )

    @classmethod
    def syntax(cls, error: BaseException, *, role: str = "") -> "ModelOutputContractError":
        """Wrap a parse failure without any raw output excerpt."""
        lineno = getattr(error, "lineno", None)
        colno = getattr(error, "colno", None)
        reason = getattr(error, "msg", None)
        if reason is not None and lineno is not None:
            summary = (
                "The response was not valid JSON "
                f"({str(reason)[:120]} at line {lineno}, column {colno})"
            )
        else:
            summary = f"The response was not valid JSON ({type(error).__name__})"
        return cls(summary, role=role, stage="syntax")


def is_model_output_contract_error(exc: Optional[BaseException], *, max_depth: int = 5) -> bool:
    """Return true for a contract error or an error explicitly raised from one."""
    current = exc
    seen: set[int] = set()
    depth = 0
    while current is not None and depth < max_depth and id(current) not in seen:
        if isinstance(current, ModelOutputContractError):
            return True
        seen.add(id(current))
        depth += 1
        current = getattr(current, "cause", None) or current.__cause__
    return False


def _model_label(fields: dict[str, Any]) -> str:
    return str(fields.get("model") or fields.get("configured_model") or "").strip()


async def report_model_output_normalizations(
    role_id: str,
    repairs: Sequence[OutputRepair],
    *,
    task_id: str = "",
    model: str = "",
    provider: str = "",
    workflow_mode: str = "",
    source_type: str = "",
    source_id: str = "",
    proof_run_id: str = "",
    run_id: str = "",
) -> Optional[dict[str, Any]]:
    """Log and publish every applied normalization as ``model_output_normalized``.

    Workflow mode, run/source scope and route identity resolve from ``role_id``
    exactly as provider retry activity resolves them; explicit arguments win.
    Publication failure is logged and never fails the caller's work. Returns the
    published (normalized, redacted) payload, or None when nothing was repaired.
    """
    repairs = [repair for repair in repairs if isinstance(repair, OutputRepair)]
    if not repairs:
        return None

    from backend.shared.activity import normalize_activity_payload
    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.log_redaction import bound_single_line
    from backend.shared.provider_route_identity import merge_route_identity, with_route_identity

    role = str(role_id or "").strip()
    logger.info(
        "Model output normalized for role %s task %s: %s",
        bound_single_line(safe_model_token(role, 160)),
        bound_single_line(safe_model_token(task_id, 120)),
        [repair.as_payload() for repair in repairs],
    )
    try:
        fields = api_client_manager.route_identity_for_role(
            role,
            workflow_mode=workflow_mode,
            model=model,
            provider=provider,
            task_id=task_id,
        )
        scope = {
            key: value
            for key, value in (
                ("source_type", source_type),
                ("source_id", source_id),
                ("proof_run_id", proof_run_id),
            )
            if value
        } or api_client_manager._role_activity_source_fields(role)
        message = " ".join(repair.describe(_model_label(fields)) for repair in repairs)
        payload: dict[str, Any] = {
            "event_id": f"{MODEL_OUTPUT_NORMALIZED_EVENT}:{uuid.uuid4().hex}",
            "role_id": role,
            "task_id": str(task_id or ""),
            "workflow_mode": str(fields.get("workflow_mode") or ""),
            "repairs": [repair.as_payload() for repair in repairs],
            "repair_count": len(repairs),
            "message": with_route_identity(message, fields),
            "activity_classification": "informational",
            "activity_category": "model_output",
            "activity_severity": "info",
            "activity_outcome": "normalized",
            **scope,
        }
        if run_id:
            payload["run_id"] = str(run_id)
        payload = merge_route_identity(payload, fields)
        published = normalize_activity_payload(MODEL_OUTPUT_NORMALIZED_EVENT, payload)
        await api_client_manager._broadcast(MODEL_OUTPUT_NORMALIZED_EVENT, published)
        return published
    except Exception as exc:  # visibility must never fail the role's work
        logger.warning(
            "Could not publish model output normalization activity for role %s: %s",
            safe_model_token(role, 160),
            type(exc).__name__,
        )
        return None


def contract_retry_delay_seconds(consecutive_failures: int) -> int:
    """Return the pause before the next try after ``consecutive_failures`` in a row.

    The first failure is retried at once (0); later ones wait 1, 2, 4, ... seconds,
    capped at ``CONTRACT_RETRY_MAX_BACKOFF_SECONDS``. Tries themselves are never capped.
    """
    if consecutive_failures < 2:
        return 0
    return min(2 ** (consecutive_failures - 2), CONTRACT_RETRY_MAX_BACKOFF_SECONDS)


async def wait_after_repeated_contract_failure(
    role_id: str,
    consecutive_failures: int,
    *,
    should_stop: Callable[[], bool],
    workflow_mode: str = "",
) -> None:
    """Stop-aware backoff before another try after repeated contract failures.

    The first failure is retried at once. Each further consecutive failure waits
    a little longer through the shared emit-and-wait seam, so a role that fails
    instantly cannot spin its workflow. Tries are never capped; Stop ends the wait.
    """
    delay = contract_retry_delay_seconds(consecutive_failures)
    if not delay or should_stop():
        return

    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.provider_route_identity import merge_route_identity, with_route_identity

    fields = api_client_manager.route_identity_for_role(role_id, workflow_mode=workflow_mode)
    message = with_route_identity(
        f"The reply broke its JSON format {consecutive_failures} times in a row; "
        f"waiting {delay} seconds before trying again. This is not a provider error "
        "and not a rejection.",
        fields,
    )
    await api_client_manager.emit_and_wait_for_retry(
        merge_route_identity(
            {
                "role_id": role_id,
                "workflow_mode": str(workflow_mode or fields.get("workflow_mode") or ""),
                "retry_attempt": consecutive_failures,
                "retry_after_seconds": delay,
                "retry_layer": "workflow",
                "reason": "model_output_contract_retry",
                "message": message,
            },
            fields,
        ),
        role_id=role_id,
        should_stop=should_stop,
    )
