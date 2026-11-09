"""Auto Top-p Translate background service.

The service translates user-readable outputs into the selected languages. One
job is one document in one language. A job works through the document's
paragraph-level units in order, one "next logical part" at a time: the single
translator submitter sees the source document and the translation built so
far, chooses the longest natural part of the next untranslated units it can
translate without error, translates it, and may correct units translated
earlier by editing them. The
single translator validator judges every attempt, and its feedback reaches the
next attempt. Attempts run strictly in series at temperature 0.0. A part is
final once the validator has accepted three attempts, or once at least one is
accepted and the translator then answers "keep" twice in a row. Rejected text
never enters the translation, and rounds have no cap.

It is a background service, not a top-level workflow: it never takes a
workflow start lease or a sleep inhibitor, never blocks a workflow, and marks
every model call with ``background_service_call`` so workflow token totals and
paper author credits skip it. One job runs at a time app-wide (plus Translate
all's lane, below) unless the opt-in batch translation setting allows as many
jobs as the configured models support. Jobs wait in a priority queue and give
their slot back after every finalized part, so the highest-priority job always
runs next; when the job limit drops, the lowest-priority running jobs yield at
their next attempt boundary.

A document is translated when it belongs to a checked mode's current run
(found by discovery), when a user requested it in a language (saved, so it
keeps translating after the viewer closes), while a viewer shows it, or, while
Translate all is on, when it is an older displayed output (found by history
enumeration). Viewed, requested, and current-run documents rank above Translate all's background documents,
and a running background job in a normal slot yields at its next attempt
boundary when one of them waits for a slot. While Translate all is on and the
translator connection can run more than one job at once, one of its jobs, the
lane, starts background jobs only: one job beyond the job limit without batch
translation, and one of the limit's jobs with it. Older outputs therefore keep
translating beside current-run work. A job keeps the lane until it ends, and
current-run jobs are never preempted. Provider, parser, and context failures
are operational and never become validator rejections; every retry wait goes
through the shared API client manager seam.

Settings changes, disable, pause, cache clearing, repair, and shutdown cancel
and fence in-flight work so stale results are never saved. Store I/O, plan
building, and prompt fitting run in worker threads; ``status_snapshot`` and
``connectivity_summary`` read in-memory state only.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from backend.shared.activity import fill_missing_activity_metadata, normalize_activity_payload
from backend.shared.api_client_manager import ProviderCooldownError, RetryableProviderError, api_client_manager
from backend.shared.background_call_scope import background_service_call
from backend.shared.config import rag_config, system_config
from backend.shared.json_parser import sanitize_model_output_for_retry_context
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.log_redaction import redact_log_text
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_provider_context_length_error,
)
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.prompt_feedback_budget import fit_prompt_with_feedback
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRepairRequiredError
from backend.shared.provider_pause import (
    get_provider_reset_generation,
    is_provider_credit_pause_error,
    wait_for_provider_reset,
)
from backend.shared.provider_route_identity import (
    ROUTE_IDENTITY_FIELD_NAMES,
    ROUTE_IDENTITY_MAX_CHARS,
    with_route_identity,
)
from backend.shared.response_extraction import choice_hit_output_limit, extract_response_text
from backend.shared.translation.contracts import (
    ACTION_KEEP,
    ACTION_TRANSLATE,
    STATUS_ACCEPTED,
    STATUS_CHECK_FAILED,
    STATUS_KEPT,
    STATUS_REJECTED,
    AttemptDecision,
    TranslationContractError,
    TranslatorAttempt,
    attempt_from_record,
    attempt_record,
    disputing_keep,
    judgment_order,
    occurrence_texts,
    parse_probe_response,
    parse_translator_attempt,
    parse_validator_decisions,
    precheck_attempt,
)
from backend.shared.translation.discovery import (
    DiscoveryResult,
    DocumentSpec,
    canonical_parts,
    discover_current_outputs,
    make_document_spec,
)
from backend.shared.translation.history import (
    SOURCE_LABELS,
    enumerate_displayed_outputs,
    history_sources,
)
from backend.shared.translation.languages import (
    PINNED_LANGUAGE_KEYS,
    STANDARD_LANGUAGES,
    LanguageEntry,
    custom_language_key,
    language_descriptor,
    proficiency_from_probe,
    resolve_language,
)
from backend.shared.translation.prompts import (
    PROBE_JSON_SCHEMA,
    TRANSLATOR_RESPONSE_FORMS,
    VALIDATOR_FORM,
    DocumentViews,
    build_json_repair_prompt,
    build_language_probe_prompt,
    build_translator_prompt,
    build_validator_prompt,
    fit_document_views,
    render_accepted_block,
    render_attempts_block,
    render_document_views,
    render_earlier_feedback,
    render_judge_block,
)
from backend.shared.translation.segmentation import (
    SEGMENTATION_VERSION,
    DocumentPlan,
    UnitView,
    assemble_occurrences,
    build_document_plan,
    build_unit_views,
    plan_from_json,
    plan_to_json,
    to_global,
    to_local,
)
from backend.shared.translation.settings import (
    AutoTranslateSettings,
    TranslationRoleConfig,
    load_auto_translate_settings,
    normalize_settings,
    save_auto_translate_settings,
    settings_ready,
)
from backend.shared.translation.store import DocumentRecord, SegmentTranslation, translation_store
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

SERVICE_NAME = "translation"
WORKFLOW_MODE = "translation"
TRANSLATOR_ROLE_ID = "translation_submitter"
VALIDATOR_ROLE_ID = "translation_validator"
SERVICE_ROLE_ID = "translation_service"
# The name users see; "Auto Top-p Translate" stays the internal name.
DISPLAY_NAME = "Foreign Language Translation w/ Top-p"
PROBE_ROLE_PREFIX = "translation_language_probe_"
JOB_STATE_VERSION = 1

# Pacing wait (not a retry): coalesces bursts of output events into one scan.
SCAN_DEBOUNCE_SECONDS = 5.0
VIEWING_WINDOW_SECONDS = 600.0
# A viewing lookup saves a document's viewed time only once the saved one is this old.
VIEWED_SAVE_INTERVAL_SECONDS = 60.0
# Pacing timer (not a retry): the queue is re-evaluated this long after the
# earliest viewing window ends, so the expiry is certain to have passed.
VIEWING_EXPIRY_MARGIN_SECONDS = 0.5
QUEUE_EVENT_MIN_INTERVAL_SECONDS = 1.0

# The part stop rule.
ACCEPTED_ATTEMPTS_TO_FINALIZE = 3
KEEPS_TO_FINALIZE = 2

# Choosing the next part: the estimated translation of the chosen units must
# fit the translator's output budget, and the part must fit every prompt that
# shows it in full. Within that limit the translator picks the part's length.
OUTPUT_TOKENS_PER_SOURCE_TOKEN = 2.5
OUTPUT_TOKENS_PER_UNIT = 40
OUTPUT_BUDGET_FRACTION = 0.8
OUTPUT_BUDGET_RESERVE_TOKENS = 300
# A translator or validator prompt shows the part's source with up to this many
# translations of it at once: its accepted attempts, the newest rejected or
# judged attempt, and the translation built so far.
PART_TRANSLATIONS_PER_PROMPT = 4
# Input room kept beside the part for instructions, notes, and feedback text.
PART_INPUT_RESERVE_TOKENS = 4000

TRANSLATOR_FEEDBACK_PROMPT_ENTRIES = 5
VALIDATOR_EARLIER_FEEDBACK_ENTRIES = 3
# Rejected, failed, and kept attempt records kept per part (accepted ones are always kept).
STORED_NON_ACCEPTED_ATTEMPTS = 10
STORED_FEEDBACK_MAX_CHARS = 2000
STORED_REASONING_MAX_CHARS = 2000
# A provider context rejection refits the prompt into this share of the rejected prompt.
CONTEXT_RETRY_SHRINK = 0.85
REPAIR_ASSISTANT_MAX_CHARS = 1500
REASONING_EVENT_CHARS = 500
FEEDBACK_EVENT_CHARS = 600
CHECK_PROBLEMS_EVENT_LIMIT = 5
VALIDATOR_CONTRACT_RETRY_SECONDS = 5
LOOP_RETRY_SECONDS = 10
FREE_MODEL_RETRY_SECONDS = 120
FREE_MODELS_EXHAUSTED_REASON = "free_models_exhausted"
TRUNCATION_REPAIR_THRESHOLD = 2
ACTIVITY_LIMIT = 60
STATUS_QUEUE_LIMIT = 25
STATS_TIME_BUCKET_SECONDS = 15.0

TRUNCATION_FEEDBACK = (
    "Your response was cut off at the max output token limit. Return the complete "
    "JSON object, and keep translator_notes short."
)
INVALID_JSON_FEEDBACK = "Your response was not valid JSON in the required format"
VALIDATOR_TRUNCATION_PROBLEM = "response was cut off at the max output token limit"

FINALIZE_THREE_ACCEPTED = "three_accepted"
FINALIZE_KEPT_TWICE = "kept_twice"

# Run lifecycle events: a run starting, stopping (by Stop or a fatal error), or
# being cleared changes which outputs are current or still being written, so
# Translate all enumerates the older ones again too.
LIFECYCLE_EVENTS = frozenset({
    "auto_research_started",
    "auto_research_resumed",
    "auto_research_stopped",
    "system_started",
    "system_stopped",
    "system_reset",
    "compiler_started",
    "compiler_stopped",
    "compiler_error",
    "context_overflow_error",
    "provider_repair_required",
    "lean_workspace_repair_required",
    "paper_cleared",
    "leanoj_started",
    "leanoj_stopped",
    "leanoj_cleared",
})

SCAN_TRIGGER_EVENTS = LIFECYCLE_EVENTS | frozenset({
    "paper_completed",
    "final_answer_complete",
    "tier3_complete",
    "tier3_short_form_complete",
    "tier3_long_form_complete",
    "novel_proof_discovered",
    "known_proof_verified",
    "proof_verified",
    "proof_registration_duplicate",
    "submission_accepted",
    "brainstorm_completed",
    "research_papers_disabled_brainstorm_complete",
    "session_changed",
    "paper_critique_completed",
    "leanoj_final_verified",
    "leanoj_brainstorm_proof_verified",
})

# Targeting tiers, best first: a checked mode's current run (or a document
# being viewed), then Translate all's background documents.
TIER_CURRENT = 0
TIER_BACKGROUND = 1
# A document key's origin, strongest first; a key keeps its strongest origin.
ORIGIN_RANK = {"auto": 0, "backlog": 1, "viewer": 2}
# Translate all registers enumerated documents in chunks this size.
BACKLOG_REGISTRATION_CHUNK = 100

PRIORITY_CLASS_BY_KIND = {
    "proof": 0,
    "paper": 1,
    "final_answer": 1,
    "live_paper": 1,
    "brainstorm": 2,
    "aggregator_results": 2,
    "other": 2,
}

STATUS_LABELS = {
    "disabled": "disabled",
    "not_configured": "not ready",
    "idle": "ready",
    "translating": "translating",
    "paused": "paused",
    "waiting_for_credits": "waiting",
    "repair_required": "repair required",
}

CONTROL_ACTIONS = ("enable", "disable", "pause", "resume", "retry", "translate_all", "cancel_translate_all")
TRANSLATE_ALL_STATES = ("off", "preparing", "translating", "waiting")
TRANSLATE_ALL_OFF, TRANSLATE_ALL_PREPARING, TRANSLATE_ALL_TRANSLATING, TRANSLATE_ALL_WAITING = TRANSLATE_ALL_STATES

ATTEMPT_ACCEPTED_EVENT = "translation_attempt_accepted"
ATTEMPT_REJECTED_EVENT = "translation_attempt_rejected"
ATTEMPT_KEPT_EVENT = "translation_attempt_kept"
ATTEMPT_CHECK_FAILED_EVENT = "translation_attempt_check_failed"
PART_FINALIZED_EVENT = "translation_part_finalized"
PROGRESS_EVENT = "translation_progress"
QUEUE_CHANGED_EVENT = "translation_queue_changed"
PROVIDER_RETRY_EVENT = "translation_provider_retry"
TRANSLATE_ALL_EVENT = "translation_translate_all"

# Events kept in the status ``recent_activity`` buffer: decisions, finalized
# parts, check failures, repairs, waits, and Translate all starting and
# ending. Status, progress, and queue events are left out.
RECENT_ACTIVITY_EVENTS = frozenset({
    TRANSLATE_ALL_EVENT,
    ATTEMPT_ACCEPTED_EVENT,
    ATTEMPT_REJECTED_EVENT,
    ATTEMPT_KEPT_EVENT,
    PART_FINALIZED_EVENT,
    ATTEMPT_CHECK_FAILED_EVENT,
    "translation_validator_contract_retry",
    "translation_repair_required",
    PROVIDER_RETRY_EVENT,
    "translation_waiting_for_credits",
})
_RETRY_REASON_TEXT = {
    "translation_validator_contract": "the validator reply was not in the required format",
    "translation_loop_retry": "an unexpected error interrupted the attempt",
    FREE_MODELS_EXHAUSTED_REASON: "every free model is rate-limited",
}

_ACTIVITY_TEXT_KEYS = (
    "language_key",
    "language_name",
    "doc_hash",
    "doc_title",
    "kind",
    "translator_model",
    "validator_model",
    "reason",
    "message",
    "feedback",
    "guidance",
    "role_id",
    "problem",
    "state",
    "status_label",
    "route_identity",
    "target_hash",
    "candidate_id",
)
_ACTIVITY_NUMBER_KEYS = (
    "round_index",
    "attempt",
    "part_start",
    "part_end",
    "final_attempt",
    "accepted_count",
    "keep_streak",
    "translated_units",
    "total_units",
    "retry_after_seconds",
    "generation",
)
_ACTIVITY_TEXT_CHARS = 300
# Provider waits and repairs keep their route identity fields, a message long
# enough for its route prefix, and the bounded provider error detail.
_ROUTED_ACTIVITY_EVENTS = frozenset(
    {PROVIDER_RETRY_EVENT, "translation_waiting_for_credits", "translation_repair_required"}
)
_ROUTED_MESSAGE_CHARS = ROUTE_IDENTITY_MAX_CHARS + _ACTIVITY_TEXT_CHARS
# A Translate all finish message keeps every older-output source it could not
# list and the instruction to retry them.
_TRANSLATE_ALL_MESSAGE_CHARS = len(", ".join(SOURCE_LABELS.values())) + _ACTIVITY_TEXT_CHARS
_ERROR_DETAIL_CHARS = 1800
_ATTEMPT_STATUSES = frozenset({STATUS_ACCEPTED, STATUS_REJECTED, STATUS_CHECK_FAILED, STATUS_KEPT})
_FEEDBACK_STATUSES = (STATUS_REJECTED, STATUS_CHECK_FAILED)
# Status of an attempt while the validator judges it; never persisted.
_STATUS_PENDING = "pending"


class _JobStopped(Exception):
    """The job was fenced, cancelled by a control change, or entered repair."""


class _ContextRejected(Exception):
    """The provider rejected a prompt because it exceeded the context window."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("provider context limit")
        self.error = error


class _EditsNotShown(Exception):
    """The validator's context cannot show the part together with the units the attempts edit."""

    def __init__(self, units: Sequence[int]) -> None:
        super().__init__("edited units do not fit the validator context")
        self.units = tuple(units)  # 1-based unit numbers that could not be shown


@dataclass
class _DocKey:
    """One doc key registered on a document, with its own metadata and requests."""

    record: DocumentRecord
    viewed_ts: float = 0.0
    requested_all: bool = False
    requested_languages: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _DocMaterial:
    """A planned document: its plan, the plan JSON, and its unit views."""

    plan: DocumentPlan
    plan_json: str
    views: Tuple[UnitView, ...]


@dataclass
class _Doc:
    """One document plan (content hash) and every doc key registered on it.

    Several outputs can show identical text under different doc keys (a
    viewer's key and discovery's key for the same final answer, for example),
    so targeting, priority, and viewing combine every key: a document targets
    a language at the best tier any key gives it, its priority class is the
    lowest of its keys, and it is being viewed when any key was viewed recently.
    """

    doc_hash: str
    plan: DocumentPlan
    plan_json: str
    order: int
    views: Tuple[UnitView, ...]
    token_counts: Tuple[int, ...]
    keys: Dict[str, _DocKey] = field(default_factory=dict)

    @property
    def unit_count(self) -> int:
        return len(self.views)

    @property
    def unit_hashes(self) -> Set[str]:
        return {view.source_hash for view in self.views}

    @property
    def priority_class(self) -> int:
        return min((entry.record.priority_class for entry in self.keys.values()), default=2)

    @property
    def viewed_ts(self) -> float:
        return max((entry.viewed_ts for entry in self.keys.values()), default=0.0)

    @property
    def primary(self) -> DocumentRecord:
        """The key shown for the document: lowest priority class, then strongest origin, then key order."""
        return min(
            self.keys.values(),
            key=lambda entry: (
                entry.record.priority_class,
                ORIGIN_RANK.get(entry.record.origin, len(ORIGIN_RANK)),
                entry.record.doc_key,
            ),
        ).record

    def requested(self, language_key: str) -> bool:
        return any(
            entry.requested_all or language_key in entry.requested_languages
            for entry in self.keys.values()
        )


@dataclass
class _Job:
    """One running document x language job and what the status shows about it."""

    doc_hash: str
    language_key: str
    epoch: int
    doc_title: str = ""
    task: Optional[asyncio.Task] = None
    cancelled: bool = False
    # A retry or credit wait happened; the job yields after the current attempt.
    waited: bool = False
    # The job gives its slot back at its next attempt boundary: the job limit
    # dropped, or it runs background work while a higher-tier job waits.
    yield_requested: bool = False
    # The job started in Translate all's lane and holds it until it ends, even
    # once its document is viewed or belongs to a current run.
    lane: bool = False
    part_start: int = 0
    part_end: int = 0
    attempt: int = 0
    accepted_count: int = 0
    keep_streak: int = 0
    translator_model: str = ""
    validator_model: str = ""


# Dispatch rank of a document x language job (see ``TranslationService._priority``).
_JobRank = Tuple[int, int, int, int, int]

# The job owning the current task.
_CURRENT_JOB: ContextVar[Optional[_Job]] = ContextVar("moto_translation_job", default=None)


@dataclass
class _PartContext:
    """Everything one attempt at the current part needs."""

    doc: _Doc
    language: LanguageEntry
    descriptor: str
    state: Dict[str, Any]
    part: Dict[str, Any]
    cursor: int
    core: Tuple[int, int]
    max_end: int
    locked: bool

    @property
    def target_hash(self) -> str:
        return f"{self.doc.doc_hash}:{self.language.key}:{self.part['start'] + 1}"


@dataclass(frozen=True)
class _CallResult:
    text: str
    truncated: bool
    effective_model: str


@dataclass(frozen=True)
class _FittedPrompt:
    prompt: str
    tokens: int
    # Occurrence indexes a windowed view shows; None when the full document is shown.
    shown: Optional[frozenset]


@dataclass(frozen=True)
class _TranslatorTurn:
    attempt: Optional[TranslatorAttempt]
    problems: Tuple[str, ...]
    shown: Optional[frozenset]


@dataclass(frozen=True)
class _Stats:
    languages: Tuple[Dict[str, Any], ...]
    queue: Tuple[Dict[str, Any], ...]
    positions: Dict[Tuple[str, str], int]
    # Units left in active languages: every tier, and the background tier alone.
    pending_total: int
    background_pending: int
    # Units translated in active languages at the current tier.
    current_translated: int
    # Translate all's documents (a backlog key and no current-run key), counted over active languages.
    translate_all: Dict[str, int]
    # A background job is running now.
    background_running: bool


def progress_percent(done: int, total: int) -> int:
    """Whole percent of ``done`` out of ``total``, held below 100 until all is done."""
    if total <= 0 or done >= total:
        return 100
    return min(99, done * 100 // total)


def _older_items_left(stats: _Stats) -> int:
    return stats.translate_all["documents"] - stats.translate_all["complete_documents"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_ts(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _bounded(value: Any, limit: int) -> str:
    return redact_log_text(" ".join(str(value or "").split()), limit)


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _match_count(text: str, find: str) -> int:
    """Count possibly overlapping matches of ``find`` in ``text``."""
    count = 0
    position = text.find(find)
    while position != -1:
        count += 1
        position = text.find(find, position + 1)
    return count


def role_label(role_id: str) -> str:
    if role_id == TRANSLATOR_ROLE_ID:
        return "Translator Submitter"
    if role_id == VALIDATOR_ROLE_ID:
        return "Translator Validator"
    return DISPLAY_NAME


def expected_doc_hash(profile: str, parts: Sequence[Tuple[str, str]]) -> str:
    """The doc hash ``build_document_plan`` assigns, computed without planning.

    Used only as a cache key to skip re-planning unchanged documents, so a
    divergence from segmentation simply means the plan is rebuilt.
    """
    payload = json.dumps(
        {
            "version": SEGMENTATION_VERSION,
            "profile": profile,
            "parts": [[name, text] for name, text in canonical_parts(parts)],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def coerce_document_input(item: Any) -> DocumentSpec:
    """Return a validated DocumentSpec from a spec, pydantic model, or mapping."""
    if isinstance(item, DocumentSpec):
        return item
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, Mapping):
        raise ValueError("Each document must be an object.")
    parts: List[Tuple[str, str]] = []
    for part in item.get("parts") or ():
        if isinstance(part, Mapping):
            parts.append((part.get("name"), part.get("text")))
        elif isinstance(part, (list, tuple)) and len(part) == 2:
            parts.append((part[0], part[1]))
        else:
            raise ValueError("Each document part must have a name and text.")
    return make_document_spec(
        doc_key=str(item.get("doc_key") or ""),
        kind=str(item.get("kind") or "other"),
        title=str(item.get("title") or ""),
        profile=str(item.get("profile") or ""),
        parts=parts,
    )


# ----------------------------------------------------------------------
# Job state (persisted JSON) helpers
# ----------------------------------------------------------------------


def new_job_state(doc_hash: str, language_key: str) -> Dict[str, Any]:
    return {
        "version": JOB_STATE_VERSION,
        "doc_hash": doc_hash,
        "language_key": language_key,
        "finals": {},
        "part": None,
    }


def new_part(cursor: int) -> Dict[str, Any]:
    return {
        "start": cursor,
        "end": cursor,
        "locked": False,
        "attempts": [],
        "keep_streak": 0,
        # The last judgment sequence number given to one of the part's attempt records.
        "seq": 0,
        "truncations": {TRANSLATOR_ROLE_ID: 0, VALIDATOR_ROLE_ID: 0},
    }


def job_cursor(unit_count: int, finals: Collection[Any]) -> int:
    """The first occurrence index without a final translation (``unit_count`` when none)."""
    done = {_as_int(key) for key in finals}
    return next((index for index in range(unit_count) if index not in done), unit_count)


def _normalize_attempt(raw: Any, unit_count: int) -> Optional[Dict[str, Any]]:
    """Return a well-formed attempt record, or None when it cannot be trusted."""
    if not isinstance(raw, Mapping):
        return None
    number = _int_or_none(raw.get("number"))
    status = raw.get("status")
    action = raw.get("action")
    if number is None or number < 1 or status not in _ATTEMPT_STATUSES:
        return None
    if action not in (ACTION_TRANSLATE, ACTION_KEEP):
        return None
    units: Dict[str, str] = {}
    for key, text in (raw.get("units") or {}).items() if isinstance(raw.get("units"), Mapping) else ():
        occurrence = _int_or_none(key)
        if occurrence is None or occurrence < 0 or not isinstance(text, str):
            return None
        if occurrence >= unit_count and status != STATUS_CHECK_FAILED:
            return None
        units[str(occurrence)] = text
    edits: List[Dict[str, Any]] = []
    for edit in raw.get("edits") or []:
        if not isinstance(edit, Mapping):
            return None
        occurrence = _int_or_none(edit.get("occurrence"))
        find, replacement = edit.get("find"), edit.get("replace")
        if occurrence is None or occurrence < 0 or not isinstance(find, str) or not isinstance(replacement, str):
            return None
        if occurrence >= unit_count and status != STATUS_CHECK_FAILED:
            return None
        edits.append({"occurrence": occurrence, "find": find, "replace": replacement})
    keep_attempts = [_int_or_none(item) for item in raw.get("keep_attempts") or []]
    if any(item is None for item in keep_attempts):
        return None
    accepted_order = _int_or_none(raw.get("accepted_order"))
    if status == STATUS_ACCEPTED and (accepted_order is None or not units):
        return None
    seq = _int_or_none(raw.get("seq"))
    return {
        "number": number,
        "action": action,
        "units": units,
        "edits": edits,
        "keep_attempts": keep_attempts,
        "notes": str(raw.get("notes") or ""),
        "status": status,
        "feedback": str(raw.get("feedback") or ""),
        "reasoning": str(raw.get("reasoning") or ""),
        "accepted_order": accepted_order if status == STATUS_ACCEPTED else None,
        # A record stored without a judgment sequence keeps its attempt number as one.
        "seq": seq if seq is not None and seq >= 1 else number,
    }


def _normalize_part(raw: Any, unit_count: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    start, end = _int_or_none(raw.get("start")), _int_or_none(raw.get("end"))
    if start is None or end is None or not 0 <= start <= end < unit_count:
        return None
    attempts = []
    for item in raw.get("attempts") or []:
        record = _normalize_attempt(item, unit_count)
        if record is None:
            return None
        attempts.append(record)
    numbers = [record["number"] for record in attempts]
    if len(numbers) != len(set(numbers)):
        return None
    truncations = raw.get("truncations") if isinstance(raw.get("truncations"), Mapping) else {}
    return {
        "start": start,
        "end": end,
        "locked": bool(raw.get("locked")),
        "attempts": sorted(attempts, key=lambda record: record["number"]),
        "keep_streak": max(0, _as_int(raw.get("keep_streak"))),
        "seq": max([0, _as_int(raw.get("seq")), *(record["seq"] for record in attempts)]),
        "truncations": {
            role_id: max(0, _as_int(truncations.get(role_id)))
            for role_id in (TRANSLATOR_ROLE_ID, VALIDATOR_ROLE_ID)
        },
    }


def normalize_job_state(
    raw: Any,
    *,
    doc_hash: str,
    language_key: str,
    unit_count: int,
) -> Tuple[Dict[str, Any], bool]:
    """Return ``(state, changed)``: a well-formed job state and whether it differs from ``raw``.

    Unreadable states start over. Finals outside the document are dropped. A
    part is dropped when it is malformed, overlaps a final translation, or
    (before any attempt was accepted) no longer starts at the cursor.
    """
    if (
        not isinstance(raw, Mapping)
        or raw.get("version") != JOB_STATE_VERSION
        or raw.get("doc_hash") != doc_hash
        or raw.get("language_key") != language_key
    ):
        return new_job_state(doc_hash, language_key), raw is not None
    finals: Dict[str, str] = {}
    for key, text in (raw.get("finals") or {}).items() if isinstance(raw.get("finals"), Mapping) else ():
        occurrence = _int_or_none(key)
        if occurrence is not None and 0 <= occurrence < unit_count and isinstance(text, str):
            finals[str(occurrence)] = text
    part = _normalize_part(raw.get("part"), unit_count) if raw.get("part") is not None else None
    if part is not None:
        cursor = job_cursor(unit_count, finals)
        overlaps = any(str(index) in finals for index in range(part["start"], part["end"] + 1))
        if overlaps or (not part["locked"] and part["start"] != cursor):
            part = None
    state = {
        "version": JOB_STATE_VERSION,
        "doc_hash": doc_hash,
        "language_key": language_key,
        "finals": finals,
        "part": part,
    }
    return state, state != raw


def next_attempt_number(part: Mapping[str, Any]) -> int:
    return max((record["number"] for record in part["attempts"]), default=0) + 1


def stamp_judgment(part: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Give ``record`` the part's next judgment sequence number.

    Called when an attempt record is created and whenever its status or
    feedback changes, so judgment order lists a re-judged attempt after the
    keep that disputed it.
    """
    part["seq"] += 1
    record["seq"] = part["seq"]


def _accepted_records(part: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The part's accepted attempts, oldest acceptance first."""
    return sorted(
        (record for record in part["attempts"] if record["status"] == STATUS_ACCEPTED),
        key=lambda record: record["accepted_order"],
    )


def trailing_kept_records(part: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The kept attempts judged after every other attempt of the part, oldest first.

    These are the refusals of the current keep streak: a dispute is always
    followed by the judgment of the attempts it named.
    """
    trailing: List[Dict[str, Any]] = []
    for record in reversed(judgment_order(part["attempts"])):
        if record["status"] != STATUS_KEPT:
            break
        trailing.append(record)
    return trailing[::-1]


def part_stop_decision(part: Mapping[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(reason, final attempt record)`` when the part's stop rule holds, else None.

    Three accepted attempts finalize with the latest accepted one. Kept twice
    finalizes with the latest accepted attempt named by both of the last two
    keeps, else the latest accepted attempt named by the last keep.
    """
    accepted = _accepted_records(part)
    if not accepted:
        return None
    if len(accepted) >= ACCEPTED_ATTEMPTS_TO_FINALIZE:
        return FINALIZE_THREE_ACCEPTED, accepted[-1]
    if part["keep_streak"] >= KEEPS_TO_FINALIZE:
        named = [set(record["keep_attempts"]) for record in trailing_kept_records(part)[-KEEPS_TO_FINALIZE:]]
        for choice in (set.intersection(*named), named[-1]) if named else ():
            chosen = [record for record in accepted if record["number"] in choice]
            if chosen:
                return FINALIZE_KEPT_TWICE, chosen[-1]
        return FINALIZE_KEPT_TWICE, accepted[-1]
    return None


def part_max_end(
    token_counts: Sequence[int],
    finals: Collection[Any],
    cursor: int,
    max_output_tokens: int,
    input_tokens: Optional[int] = None,
) -> Optional[int]:
    """The last occurrence an unlocked part starting at ``cursor`` may reach.

    The part stays inside the contiguous untranslated run from the cursor and
    its estimated translation fits the translator's output budget. With
    ``input_tokens`` (the smaller of the two roles' input budgets), the part's
    source plus ``PART_TRANSLATIONS_PER_PROMPT`` estimated translations of it
    also fits that budget less ``PART_INPUT_RESERVE_TOKENS``; the first unit
    is always allowed that budget, and a unit too large for it reaches context
    repair as before. Returns None when even the first unit's estimated output
    does not fit.
    """
    done = {_as_int(key) for key in finals}
    budget = OUTPUT_BUDGET_FRACTION * max_output_tokens - OUTPUT_BUDGET_RESERVE_TOKENS
    input_budget = None if input_tokens is None else input_tokens - PART_INPUT_RESERVE_TOKENS
    end: Optional[int] = None
    output = 0
    source = 0
    for occurrence in range(cursor, len(token_counts)):
        if occurrence in done:
            break
        tokens = token_counts[occurrence]
        cost = math.ceil(tokens * OUTPUT_TOKENS_PER_SOURCE_TOKEN) + OUTPUT_TOKENS_PER_UNIT
        if output + cost > budget:
            break
        shown = source + tokens + PART_TRANSLATIONS_PER_PROMPT * (output + cost)
        if end is not None and input_budget is not None and shown > input_budget:
            break
        output += cost
        source += tokens
        end = occurrence
    return end


def seed_job_finals(
    state: Dict[str, Any],
    views: Sequence[UnitView],
    translations: Mapping[str, str],
) -> List[int]:
    """Fill untranslated occurrences outside the current part from final unit translations.

    ``translations`` maps source hashes to unit-local final translations (the
    cross-document cache). Returns the seeded occurrence indexes.
    """
    part = state.get("part")
    skip = range(part["start"], part["end"] + 1) if part else range(0)
    finals = state["finals"]
    seeded: List[int] = []
    for view in views:
        key = str(view.occurrence)
        if key in finals or view.occurrence in skip:
            continue
        text = translations.get(view.source_hash)
        if isinstance(text, str):
            finals[key] = text
            seeded.append(view.occurrence)
    return seeded


def edit_key(edit: Mapping[str, Any]) -> Tuple[int, str, str]:
    return (int(edit["occurrence"]), str(edit["find"]), str(edit["replace"]))


def apply_record_edits(
    state: Dict[str, Any],
    views: Sequence[UnitView],
    record: Mapping[str, Any],
    *,
    already_applied: Collection[Tuple[int, str, str]] = (),
) -> List[Dict[str, Any]]:
    """Apply an attempt's edits to the finals in order; returns the edits that applied.

    Edits are stored in document-wide token form against each unit's
    translation at submission time; an edit that no longer matches exactly
    once is skipped. An edit identical to one in ``already_applied`` (made by
    an earlier attempt) or to an earlier edit of this record counts as
    applied without changing the text again.
    """
    finals = state["finals"]
    applied: List[Dict[str, Any]] = []
    done = set(already_applied)
    for edit in record.get("edits") or []:
        if edit_key(edit) in done:
            applied.append(dict(edit))
            continue
        done.add(edit_key(edit))
        occurrence = edit["occurrence"]
        key = str(occurrence)
        if key not in finals or occurrence >= len(views) or not edit["find"]:
            continue
        view = views[occurrence]
        current = to_global(finals[key], view)
        if _match_count(current, edit["find"]) != 1:
            logger.info("Auto Top-p Translate skipped an edit that no longer matches unit U%s.", occurrence + 1)
            continue
        finals[key] = to_local(current.replace(edit["find"], edit["replace"], 1), view)
        applied.append(dict(edit))
    return applied


def applied_part_edit_keys(
    part: Mapping[str, Any], finals: Mapping[str, str], views: Sequence[UnitView]
) -> Set[Tuple[int, str, str]]:
    """Edits the part's accepted attempts made whose result is still in the translation.

    An edit is still in effect while its replacement is in the unit and no
    later accepted edit of the unit rewrote it (a deletion: while its find
    text is gone). A later attempt repeating one of them asks for a change
    already made, so it counts as applied rather than being applied again.
    """
    made = [edit for record in _accepted_records(part) for edit in record.get("edits") or []]
    keys: Set[Tuple[int, str, str]] = set()
    for index, edit in enumerate(made):
        occurrence = int(edit["occurrence"])
        text = finals.get(str(occurrence))
        if text is None or occurrence >= len(views):
            continue
        current = to_global(text, views[occurrence])
        replacement = edit["replace"]
        if not replacement:
            in_effect = edit["find"] not in current
        else:
            in_effect = replacement in current and not any(
                int(later["occurrence"]) == occurrence
                and (later["find"] in replacement or replacement in later["find"])
                for later in made[index + 1:]
            )
        if in_effect:
            keys.add(edit_key(edit))
    return keys


def without_applied_edits(
    attempt: TranslatorAttempt, applied: Collection[Tuple[int, str, str]]
) -> TranslatorAttempt:
    """``attempt`` without the edits that ``applied`` says are already in the translation."""
    edits = tuple(edit for edit in attempt.edits if (edit.unit - 1, edit.find, edit.replace) not in applied)
    return attempt if len(edits) == len(attempt.edits) else replace(attempt, edits=edits)


def finalize_part_in_state(
    state: Dict[str, Any],
    views: Sequence[UnitView],
    record: Mapping[str, Any],
) -> List[int]:
    """Write the final attempt's units (and same-hash untranslated occurrences); clears the part."""
    texts = occurrence_texts(record["units"])
    finals = state["finals"]
    finalized: List[int] = []
    by_hash: Dict[str, str] = {}
    for occurrence, text in texts.items():
        finals[str(occurrence)] = text
        finalized.append(occurrence)
        by_hash[views[occurrence].source_hash] = text
    for view in views:
        key = str(view.occurrence)
        if key not in finals and view.source_hash in by_hash:
            finals[key] = by_hash[view.source_hash]
            finalized.append(view.occurrence)
    state["part"] = None
    return sorted(finalized)


def _retoken(text: str, old_view: UnitView, new_view: UnitView) -> str:
    return to_global(to_local(text, old_view), new_view)


def reconcile_job_state(
    old_state: Mapping[str, Any],
    old_views: Sequence[UnitView],
    new_views: Sequence[UnitView],
    *,
    doc_hash: str,
    language_key: str,
) -> Dict[str, Any]:
    """Carry a job's progress over to the changed content of its document.

    Occurrences are matched by their source-hash sequence. Finals of matching
    occurrences are copied; the current part survives only when its range and
    every unit its accepted attempts cover or edit map to the new document
    unchanged and in place. Other attempts whose units or edits no longer map
    are dropped, with the keeps that name them.
    """
    mapping: Dict[int, int] = {}
    matcher = SequenceMatcher(
        None,
        [view.source_hash for view in old_views],
        [view.source_hash for view in new_views],
        autojunk=False,
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    state = new_job_state(doc_hash, language_key)
    for key, text in (old_state.get("finals") or {}).items():
        occurrence = _as_int(key)
        if occurrence in mapping:
            state["finals"][str(mapping[occurrence])] = text
    part = old_state.get("part")
    if part:
        state["part"] = _reconcile_part(part, mapping, old_views, new_views)
    return state


def _record_occurrences(record: Mapping[str, Any]) -> Set[int]:
    """The occurrences an attempt record translates or edits."""
    occurrences = {_as_int(key) for key in record["units"]}
    occurrences.update(edit["occurrence"] for edit in record["edits"])
    return occurrences


def _reconcile_part(
    part: Mapping[str, Any],
    mapping: Mapping[int, int],
    old_views: Sequence[UnitView],
    new_views: Sequence[UnitView],
) -> Optional[Dict[str, Any]]:
    start, end = part["start"], part["end"]
    required = set(range(start, end + 1))
    for record in part["attempts"]:
        if record["status"] == STATUS_ACCEPTED:
            required.update(_record_occurrences(record))
    if any(occurrence not in mapping for occurrence in required):
        return None
    if mapping[end] - mapping[start] != end - start:
        return None
    survivors = [
        record
        for record in part["attempts"]
        if all(occurrence in mapping for occurrence in _record_occurrences(record))
    ]
    numbers = {record["number"] for record in survivors}
    survivors = [
        record
        for record in survivors
        if record["status"] != STATUS_KEPT or numbers.issuperset(record["keep_attempts"])
    ]
    numbers = {record["number"] for record in survivors}
    surviving_keeps = sum(1 for record in trailing_kept_records(part) if record["number"] in numbers)
    attempts = []
    for record in survivors:
        verbatim = record["status"] == STATUS_CHECK_FAILED
        units: Dict[str, str] = {}
        for key, text in record["units"].items():
            occurrence = _as_int(key)
            target = mapping[occurrence]
            units[str(target)] = (
                _retoken(text, old_views[occurrence], new_views[target]) if verbatim else text
            )
        edits = []
        for edit in record["edits"]:
            occurrence = edit["occurrence"]
            old_view, new_view = old_views[occurrence], new_views[mapping[occurrence]]
            edits.append(
                {
                    "occurrence": mapping[occurrence],
                    "find": _retoken(edit["find"], old_view, new_view),
                    "replace": _retoken(edit["replace"], old_view, new_view),
                }
            )
        attempts.append({**record, "units": units, "edits": edits})
    return {
        **part,
        "start": mapping[start],
        "end": mapping[end],
        "attempts": attempts,
        "keep_streak": min(part["keep_streak"], surviving_keeps),
    }


def prune_attempt_records(part: Dict[str, Any]) -> None:
    """Drop attempt records neither the stop rule nor the translator prompt can need.

    Kept: every accepted attempt; the newest ``STORED_NON_ACCEPTED_ATTEMPTS``
    rejected and failed attempts by judgment order (kept attempts never take
    their place); the newest kept attempts that can form a keep streak; the
    dispute behind each re-judged attempt among the newest feedback the
    translator sees; and every attempt a stored keep names.
    """
    ordered = judgment_order(part["attempts"])
    feedback = [record for record in ordered if record["status"] in _FEEDBACK_STATUSES]
    kept = [record for record in ordered if record["status"] == STATUS_KEPT][-KEEPS_TO_FINALIZE:]
    for record in feedback[-TRANSLATOR_FEEDBACK_PROMPT_ENTRIES:]:
        dispute = disputing_keep(record, part["attempts"])
        if dispute is not None:
            kept.append(dispute)
    stored = {record["number"] for record in ordered if record["status"] == STATUS_ACCEPTED}
    stored.update(record["number"] for record in feedback[-STORED_NON_ACCEPTED_ATTEMPTS:])
    for record in kept:
        stored.add(record["number"])
        stored.update(record["keep_attempts"])
    part["attempts"] = [record for record in part["attempts"] if record["number"] in stored]


# ----------------------------------------------------------------------
# Thread-side store helpers
# ----------------------------------------------------------------------


def _material_sync(plan: DocumentPlan, plan_json: str) -> _DocMaterial:
    return _DocMaterial(plan=plan, plan_json=plan_json, views=tuple(build_unit_views(plan)))


def _make_doc(doc_hash: str, material: _DocMaterial, order: int) -> _Doc:
    tokens_by_hash = {segment.source_hash: segment.token_count for segment in material.plan.segments}
    return _Doc(
        doc_hash=doc_hash,
        plan=material.plan,
        plan_json=material.plan_json,
        order=order,
        views=material.views,
        token_counts=tuple(tokens_by_hash.get(view.source_hash, 0) for view in material.views),
    )


def _load_store_state_sync() -> Tuple[
    List[Tuple[DocumentRecord, _DocMaterial]],
    Dict[str, Set[str]],
    Dict[Tuple[str, str], int],
    Dict[str, Set[str]],
]:
    loaded: List[Tuple[DocumentRecord, _DocMaterial]] = []
    materials: Dict[str, Optional[_DocMaterial]] = {}
    for record in translation_store.load_documents():
        if record.doc_hash not in materials:
            try:
                materials[record.doc_hash] = _material_sync(plan_from_json(record.plan_json), record.plan_json)
            except (ValueError, TypeError, KeyError) as exc:
                materials[record.doc_hash] = None
                logger.warning(
                    "Skipping unreadable stored translation plan %s: %s",
                    redact_log_text(record.doc_key, 120),
                    type(exc).__name__,
                )
        material = materials[record.doc_hash]
        if material is not None:
            loaded.append((record, material))
    counts: Dict[Tuple[str, str], int] = {}
    for doc_hash, language_key in translation_store.load_job_keys():
        material = materials.get(doc_hash)
        if material is None:
            continue
        state, _changed = normalize_job_state(
            translation_store.get_job_state(doc_hash, language_key),
            doc_hash=doc_hash,
            language_key=language_key,
            unit_count=len(material.views),
        )
        counts[(doc_hash, language_key)] = len(state["finals"])
    return loaded, translation_store.translated_hashes_by_language(), counts, translation_store.load_requests()


def _build_material_sync(spec: DocumentSpec) -> _DocMaterial:
    plan = build_document_plan(spec.profile, canonical_parts(spec.parts))
    return _material_sync(plan, plan_to_json(plan))


def _plan_specs_sync(
    specs: Sequence[DocumentSpec],
    known_hashes: Set[str],
) -> List[Tuple[DocumentSpec, str, Optional[_DocMaterial], str]]:
    """Return (spec, doc_hash, material, error) per spec.

    ``material`` is None when the document is already known or invalid.
    """
    results: List[Tuple[DocumentSpec, str, Optional[_DocMaterial], str]] = []
    for spec in specs:
        doc_hash = expected_doc_hash(spec.profile, spec.parts)
        if doc_hash in known_hashes:
            results.append((spec, doc_hash, None, ""))
            continue
        try:
            material = _build_material_sync(spec)
        except (ValueError, TypeError) as exc:
            results.append((spec, "", None, str(exc)[:200]))
            continue
        results.append((spec, material.plan.doc_hash, material, ""))
    return results


def _seed_from_cache_sync(state: Dict[str, Any], views: Sequence[UnitView], language_key: str) -> List[int]:
    finals = state["finals"]
    missing = sorted({view.source_hash for view in views if str(view.occurrence) not in finals})
    if not missing:
        return []
    return seed_job_finals(state, views, translation_store.get_translations(missing, language_key))


def _persist_registration_sync(
    records: Sequence[DocumentRecord],
    viewed_keys: Sequence[str],
    now: str,
    moves: Sequence[Tuple[str, str, Tuple[UnitView, ...], Tuple[UnitView, ...]]],
) -> Dict[Tuple[str, str], int]:
    """Persist registrations; returns final counts of job states carried to moved content.

    Moves are ``(old_hash, new_hash, old_views, new_views)``. Their job states
    are read before registering, because the store deletes a plan's job states
    once its last doc key leaves it.
    """
    carried: List[Tuple[str, str, Dict[str, Any]]] = []
    if moves:
        job_keys = translation_store.load_job_keys()
        for old_hash, new_hash, old_views, new_views in moves:
            for doc_hash, language_key in job_keys:
                if doc_hash != old_hash:
                    continue
                old_state, _changed = normalize_job_state(
                    translation_store.get_job_state(old_hash, language_key),
                    doc_hash=old_hash,
                    language_key=language_key,
                    unit_count=len(old_views),
                )
                carried.append(
                    (
                        new_hash,
                        language_key,
                        reconcile_job_state(
                            old_state,
                            old_views,
                            new_views,
                            doc_hash=new_hash,
                            language_key=language_key,
                        ),
                    )
                )
    for record in records:
        translation_store.register_document(record)
    translation_store.mark_documents_viewed(viewed_keys, now)
    counts: Dict[Tuple[str, str], int] = {}
    views_by_hash = {new_hash: new_views for _old, new_hash, _old_views, new_views in moves}
    for new_hash, language_key, state in carried:
        key = (new_hash, language_key)
        if key in counts or translation_store.get_job_state(new_hash, language_key) is not None:
            continue
        _seed_from_cache_sync(state, views_by_hash[new_hash], language_key)
        translation_store.save_job_state(new_hash, language_key, state)
        counts[key] = len(state["finals"])
    return counts


def _load_job_state_sync(
    doc_hash: str,
    language_key: str,
    views: Tuple[UnitView, ...],
    seed: bool,
) -> Tuple[Dict[str, Any], bool]:
    state, changed = normalize_job_state(
        translation_store.get_job_state(doc_hash, language_key),
        doc_hash=doc_hash,
        language_key=language_key,
        unit_count=len(views),
    )
    if seed and _seed_from_cache_sync(state, views, language_key):
        changed = True
    return state, changed


def _commit_job_sync(
    doc_hash: str,
    language_key: str,
    state: Dict[str, Any],
    translations: Sequence[SegmentTranslation],
) -> None:
    for item in translations:
        translation_store.save_translation(item)
    translation_store.save_job_state(doc_hash, language_key, state)


def _assemble_sync(
    items: Sequence[Tuple[DocumentPlan, str, int]],
    language_key: str,
) -> List[Tuple[Optional[Dict[str, str]], int]]:
    """Return (assembled parts or None, final count) per (plan, doc_hash, unit_count)."""
    results: List[Tuple[Optional[Dict[str, str]], int]] = []
    for plan, doc_hash, unit_count in items:
        if unit_count == 0:
            results.append((assemble_occurrences(plan, {}), 0))
            continue
        state, _changed = normalize_job_state(
            translation_store.get_job_state(doc_hash, language_key),
            doc_hash=doc_hash,
            language_key=language_key,
            unit_count=unit_count,
        )
        finals = occurrence_texts(state["finals"])
        results.append((assemble_occurrences(plan, finals), len(finals)))
    return results


def _messages_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(count_tokens(str(message.get("content") or "")) for message in messages)


def _fit_prompt_sync(
    doc: _Doc,
    finals: Mapping[str, str],
    core: Tuple[int, int],
    required: Collection[int],
    entries: Sequence[Dict[str, Any]],
    max_entries: int,
    build: Callable[[DocumentViews, Tuple[Dict[str, Any], ...]], str],
    available: int,
) -> Optional[_FittedPrompt]:
    """Fit a prompt around the mandatory document views.

    ``entries`` is rolling feedback, oldest first. The full views come first
    while the feedback sheds its oldest whole entries (the newest is never
    dropped). When even the newest entry does not fit, the views are windowed
    around ``core`` (always showing the ``required`` occurrences): the most
    feedback entries that fit with a window are kept, again shedding the
    oldest first, each with the widest window that fits. Returns None when the
    part and the required occurrences cannot fit with the newest entry.
    """
    if available <= 0:
        return None
    full = render_document_views(doc.plan, doc.views, finals)
    fit = fit_prompt_with_feedback(
        list(entries),
        build_prompt=lambda kept: build(full, kept),
        available_tokens=available,
        max_entries=max_entries,
    )
    if fit.fits:
        return _FittedPrompt(prompt=fit.prompt, tokens=fit.prompt_tokens, shown=None)
    projection = tuple(entries[-max_entries:])
    # Newest feedback suffixes, longest first; each keeps the newest entry.
    for kept in [projection[index:] for index in range(len(projection))] or [()]:
        windowed = fit_document_views(
            doc.plan,
            doc.views,
            finals,
            part_start=core[0],
            part_end=core[1],
            required=required,
            fits=lambda views, kept=kept: count_tokens(build(views, kept)) <= available,
        )
        if windowed is not None:
            prompt = build(windowed, kept)
            return _FittedPrompt(prompt=prompt, tokens=count_tokens(prompt), shown=windowed.shown)
    return None


class TranslationService:
    """Singleton background service behind ``/api/translation``."""

    def __init__(self) -> None:
        self._settings = AutoTranslateSettings()
        self._language_entries: Dict[str, LanguageEntry] = {}
        self._started = False
        self._stopping = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake: Optional[asyncio.Event] = None
        self._scan_event: Optional[asyncio.Event] = None
        self._backlog_event: Optional[asyncio.Event] = None
        self._control_lock: Optional[asyncio.Lock] = None
        self._write_lock: Optional[asyncio.Lock] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._scan_task: Optional[asyncio.Task] = None
        self._backlog_task: Optional[asyncio.Task] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._queue_event_handle: Optional[asyncio.TimerHandle] = None
        self._viewing_expiry_handle: Optional[asyncio.TimerHandle] = None
        self._discover: Callable[[FrozenSet[str]], Awaitable[DiscoveryResult]] = discover_current_outputs
        self._enumerate_backlog: Callable[[FrozenSet[str]], Awaitable[DiscoveryResult]] = enumerate_displayed_outputs
        self._reset_runtime_state()

    # ------------------------------------------------------------------
    # Runtime state
    # ------------------------------------------------------------------

    def _reset_runtime_state(self) -> None:
        self._generation = 0
        self._work_epoch = 0
        self._repair: Optional[Dict[str, Any]] = None
        self._credit_waiters = 0
        self._wait_note_seq = 0
        self._slot_limit = 1
        # Loaded same-base copies per LM Studio model, as last counted.
        self._lm_studio_copies: Dict[str, int] = {}
        # Identifies the latest loaded-copy count; a count started for older settings is discarded.
        self._capacity_seq = 0
        # A cache clear is running; no job may start until it ends.
        self._clearing = False
        self._docs: Dict[str, _Doc] = {}
        self._doc_by_key: Dict[str, str] = {}
        self._translated: Dict[str, Set[str]] = {}
        self._final_counts: Dict[Tuple[str, str], int] = {}
        # Doc key -> the source that found it: current-run discovery, and
        # Translate all's enumeration of older outputs (rebuilt only by a fresh
        # enumeration, so nothing older is targeted before one finishes).
        self._auto_keys: Dict[str, str] = {}
        self._backlog_keys: Dict[str, str] = {}
        # Doc key -> languages a user requested it in; requested documents
        # translate at the current-run tier until done, viewed or not.
        self._requests: Dict[str, Set[str]] = {}
        # An enumeration finished since Translate all started or its modes
        # changed, and the sources the latest one could not list.
        self._backlog_enumerated = False
        self._backlog_failed_sources: Tuple[str, ...] = ()
        self._backlog_scanning = False
        self._backlog_check_task: Optional[asyncio.Task] = None
        self._doc_seq = 0
        self._jobs: Dict[Tuple[str, str], _Job] = {}
        self._task_seq: Dict[str, int] = {}
        self._activity: deque = deque(maxlen=ACTIVITY_LIMIT)
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "accepted_attempts": 0,
            "rejected_attempts": 0,
            "finalized_parts": 0,
            "check_failures": 0,
        }
        self._probe: Optional[Dict[str, Any]] = None
        self._stats_version = 0
        self._stats_cache: Optional[Tuple[Tuple[int, int], _Stats]] = None
        self._last_status_key: Optional[Tuple[Any, ...]] = None
        self._queue_version = 0
        self._queue_signature_sent: Optional[Tuple[Any, ...]] = None
        self._queue_event_at = 0.0

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop and self._wake is not None:
            return
        self._loop = loop
        self._wake = asyncio.Event()
        self._scan_event = asyncio.Event()
        self._backlog_event = asyncio.Event()
        self._control_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    def _wake_worker(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def _touch(self) -> None:
        self._stats_version += 1

    def _selected_keys(self) -> List[str]:
        return [language.key for language in self._settings.languages]

    def _active_language_keys(self) -> List[str]:
        """Selected languages that resolve, in the user's priority order."""
        return [key for key in self._selected_keys() if key in self._language_entries]

    def _refresh_language_entries(self) -> None:
        entries: Dict[str, LanguageEntry] = {}
        for language in self._settings.languages:
            try:
                entries[language.key] = resolve_language(language.key, language.name)
            except ValueError:
                continue
        self._language_entries = entries

    def _can_dispatch(self) -> bool:
        ready, _reason = settings_ready(self._settings)
        return (
            self._started
            and not self._stopping
            and not self._clearing
            and self._settings.enabled
            and not self._settings.paused
            and ready
            and self._repair is None
        )

    def _job_fenced(self, job: _Job) -> bool:
        return (
            self._stopping
            or not self._started
            or job.cancelled
            or job.epoch != self._work_epoch
            or not self._settings.enabled
            or self._settings.paused
            or self._repair is not None
            or job.language_key not in self._language_entries
            or job.doc_hash not in self._docs
        )

    def _raise_if_stopped(self, job: _Job) -> None:
        if self._job_fenced(job):
            raise _JobStopped()

    def _next_task_id(self, prefix: str) -> str:
        sequence = self._task_seq.get(prefix, 0) + 1
        self._task_seq[prefix] = sequence
        return f"{prefix}_{sequence:05d}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load settings and stored state, register the event observer, and start work."""
        self._bind_loop()
        async with self._control_lock:
            if self._started:
                return
            self._reset_runtime_state()
            self._stopping = False
            try:
                self._settings = await asyncio.to_thread(load_auto_translate_settings)
            except Exception as exc:
                logger.warning(
                    "Auto Top-p Translate settings could not be loaded: %s",
                    type(exc).__name__,
                )
                self._settings = AutoTranslateSettings()
            self._refresh_language_entries()
            try:
                loaded, translated, counts, requests = await asyncio.to_thread(_load_store_state_sync)
            except Exception as exc:
                logger.warning(
                    "Auto Top-p Translate store could not be loaded: %s: %s",
                    type(exc).__name__,
                    redact_log_text(exc, 240),
                )
                loaded, translated, counts, requests = [], {}, {}, {}
            for record, material in loaded:
                self._add_doc(record, material)
            self._translated = {key: set(values) for key, values in translated.items()}
            self._final_counts = {key: count for key, count in counts.items() if key[0] in self._docs}
            self._requests = {key: set(languages) for key, languages in requests.items()}
            self._started = True
            from backend.api.routes.websocket import register_event_observer

            register_event_observer(self._observe_event)
            self._configure_roles()
            self._set_slot_limit(self._known_slot_limit())
            self._sync_background_tasks()
            self._refresh_lm_studio_capacity()
            self._schedule_viewing_expiry()
            self._request_scan()
            self._request_backlog()
            self._wake_worker()
        await self._publish_status_if_changed()

    async def shutdown(self) -> None:
        """Cancel the worker, scans, and jobs, then unregister the observer."""
        if self._loop is None:
            return
        self._bind_loop()
        self._stopping = True
        with contextlib.suppress(Exception):
            from backend.api.routes.websocket import unregister_event_observer

            unregister_event_observer(self._observe_event)
        for handle in (self._queue_event_handle, self._viewing_expiry_handle):
            if handle is not None:
                handle.cancel()
        self._queue_event_handle = None
        self._viewing_expiry_handle = None
        tasks = [job.task for job in self._jobs.values() if job.task is not None]
        for job in self._jobs.values():
            job.cancelled = True
        tasks.extend(
            task for task in (self._worker_task, self._scan_task, self._backlog_task) if task is not None
        )
        tasks.extend(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._worker_task = None
        self._scan_task = None
        self._backlog_task = None
        self._jobs.clear()
        self._started = False

    def _sync_background_tasks(self) -> None:
        if not self._started or self._stopping:
            return
        if self._settings.enabled:
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = self._track(asyncio.create_task(self._worker_loop(), name="auto-translate-worker"))
            if self._scan_task is None or self._scan_task.done():
                self._scan_task = self._track(asyncio.create_task(self._scan_loop(), name="auto-translate-scan"))
            if self._backlog_task is None or self._backlog_task.done():
                self._backlog_task = self._track(
                    asyncio.create_task(self._backlog_loop(), name="auto-translate-backlog")
                )
            return
        for task in (self._worker_task, self._scan_task, self._backlog_task):
            if task is not None and not task.done():
                task.cancel()
        self._worker_task = None
        self._scan_task = None
        self._backlog_task = None

    def _track(self, task: asyncio.Task) -> asyncio.Task:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _observe_event(self, event_type: str, _data: Dict[str, Any]) -> None:
        """Websocket broadcast observer: requests a debounced rescan after output and run lifecycle events."""
        if event_type in SCAN_TRIGGER_EVENTS:
            self._request_scan()
        if event_type in LIFECYCLE_EVENTS:
            self._request_backlog()

    def note_outputs_changed(self, reason: str) -> None:
        """Rescan outputs after a route cleared or pruned displayed content; never raises.

        Current-run discovery runs again, and so does Translate all's
        enumeration while it is on, so cleared content stops being targeted.
        """
        try:
            self._request_scan()
            self._request_backlog()
        except Exception as exc:
            logger.debug(
                "Auto Top-p Translate could not request a rescan after %s: %s",
                redact_log_text(reason, 80),
                type(exc).__name__,
            )

    def _request_scan(self) -> None:
        if (
            self._started
            and not self._stopping
            and self._settings.enabled
            and self._scan_event is not None
        ):
            self._scan_event.set()

    def _translate_all_on(self) -> bool:
        return self._settings.enabled and self._settings.translate_all

    def _request_backlog(self) -> None:
        if self._started and not self._stopping and self._translate_all_on() and self._backlog_event is not None:
            self._backlog_event.set()

    def _backlog_busy(self) -> bool:
        """An enumeration of older outputs is pending or running."""
        return self._backlog_scanning or (self._backlog_event is not None and self._backlog_event.is_set())

    @staticmethod
    async def _debounce(event: asyncio.Event) -> None:
        """Wait for ``event``; requests arriving during a short pause coalesce into one run."""
        await event.wait()
        # NON_RETRY pacing wait: requests arriving during it coalesce into one run.
        await asyncio.sleep(SCAN_DEBOUNCE_SECONDS)
        event.clear()

    async def _scan_loop(self) -> None:
        while not self._stopping:
            await self._debounce(self._scan_event)
            if not self._settings.enabled or self._stopping:
                continue
            modes = frozenset(self._settings.modes)
            try:
                result = await self._discover(modes)
                # A mode change during the scan has requested the next scan already.
                if modes == frozenset(self._settings.modes):
                    await self._apply_discovery(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Auto Top-p Translate output scan failed: %s: %s",
                    type(exc).__name__,
                    redact_log_text(exc, 240),
                )

    async def _apply_discovery(self, result: DiscoveryResult) -> None:
        """Target a scan's current-run documents; sources that failed keep their earlier documents."""
        failed = set(result.failed_sources)
        kept = {doc_key: source for doc_key, source in self._auto_keys.items() if source in failed}
        self._auto_keys = {**kept, **{spec.doc_key: spec.source for spec in result.documents}}
        self._touch()
        await self._register_specs(
            list(result.documents),
            language_key=None,
            viewing=False,
            origin="auto",
        )
        self._wake_worker()
        await self._publish_status_if_changed()

    async def _backlog_loop(self) -> None:
        while not self._stopping:
            await self._debounce(self._backlog_event)
            if not self._translate_all_on() or self._stopping:
                continue
            modes = frozenset(self._settings.modes)
            self._backlog_scanning = True
            self._touch()
            try:
                result = await self._enumerate_backlog(modes)
                # A cancel or mode change during the enumeration makes its result stale.
                if self._translate_all_on() and modes == frozenset(self._settings.modes):
                    await self._apply_backlog(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Auto Top-p Translate could not list older outputs: %s: %s",
                    type(exc).__name__,
                    redact_log_text(exc, 240),
                )
                if self._translate_all_on() and modes == frozenset(self._settings.modes):
                    # Every older output the enumeration covers counts as not listed.
                    self._finish_enumeration(history_sources(modes))
            finally:
                self._backlog_scanning = False
                self._touch()
            self._schedule_backlog_check()
            await self._publish_status_if_changed()

    async def _apply_backlog(self, result: DiscoveryResult) -> None:
        """Target the enumerated older outputs in the background tier.

        Documents register in chunks so no write-lock section grows with the
        library. Sources that failed keep their earlier documents; they are
        not listed again until the next enumeration, so a failing source never
        retries on its own.
        """
        failed = set(result.failed_sources)
        kept = {doc_key: source for doc_key, source in self._backlog_keys.items() if source in failed}
        self._backlog_keys = {**kept, **{spec.doc_key: spec.source for spec in result.documents}}
        self._touch()
        specs = list(result.documents)
        for start in range(0, len(specs), BACKLOG_REGISTRATION_CHUNK):
            if not self._translate_all_on():
                return
            await self._register_specs(
                specs[start:start + BACKLOG_REGISTRATION_CHUNK],
                language_key=None,
                viewing=False,
                origin="backlog",
            )
            self._wake_worker()
        self._finish_enumeration(result.failed_sources)

    def _finish_enumeration(self, failed_sources: Sequence[str]) -> None:
        """An enumeration finished: Translate all's state follows its remaining work from now on."""
        self._backlog_enumerated = True
        self._backlog_failed_sources = tuple(failed_sources)
        self._touch()

    def _schedule_backlog_check(self) -> None:
        if (
            not self._started
            or self._stopping
            or not self._translate_all_on()
            or (self._backlog_check_task is not None and not self._backlog_check_task.done())
        ):
            return
        self._backlog_check_task = self._track(
            asyncio.create_task(self._check_backlog_finished(), name="auto-translate-backlog-check")
        )

    async def _check_backlog_finished(self) -> None:
        """Turn Translate all off once every older output is translated in every active language.

        It finishes only after an enumeration has finished, and never while
        another enumeration is pending or running. When the latest enumeration
        could not list some sources, the finish message names them and says
        that Translate all lists them again when it is started again.
        """
        async with self._control_lock:
            if not self._translate_all_on() or not self._backlog_enumerated or self._backlog_busy():
                return
            progress = self._stats().translate_all
            if progress["pending_units"]:
                return
            failed = self._backlog_failed_sources
            try:
                await self._save_translate_all(False)
            except Exception as exc:
                # Translate all stays on until a later check (after a job or an enumeration) saves it.
                logger.warning(
                    "Auto Top-p Translate could not turn Translate all off: %s",
                    type(exc).__name__,
                )
                return
            documents = progress["documents"]
            languages = len(self._active_language_keys())
            message = (
                f"Translate all finished: {_count_label(documents, 'item')} in "
                f"{_count_label(languages, 'language')}."
                if documents
                else "Nothing else to translate for the checked modes."
            )
            if failed:
                labels = ", ".join(SOURCE_LABELS.get(source, source) for source in failed)
                message += (
                    f" These older items could not be listed: {labels}. "
                    "Press Translate all again to retry them."
                )
        await self._record_translate_all("finished", message)
        await self._publish_status_if_changed()

    async def _save_translate_all(self, requested: bool) -> None:
        """Persist the Translate all flag outside a control action."""
        settings = self._settings.model_copy(update={"translate_all": requested})
        await asyncio.to_thread(save_auto_translate_settings, settings)
        self._settings = settings
        self._reset_backlog()

    def _reset_backlog(self) -> None:
        """Translate all started, stopped, or changed modes: older outputs are enumerated afresh.

        While it is off, no document is targeted as an older output.
        """
        self._backlog_enumerated = False
        self._backlog_failed_sources = ()
        if not self._settings.translate_all:
            self._backlog_keys = {}
        self._touch()

    async def _record_translate_all(self, change: str, message: str) -> None:
        await self._broadcast(
            TRANSLATE_ALL_EVENT,
            {"state": change, "message": message},
            classification="informational",
            category="workflow",
            outcome="info",
        )

    async def _worker_loop(self) -> None:
        while not self._stopping:
            await self._wake.wait()
            self._wake.clear()
            self._dispatch()
            await self._publish_status_if_changed()

    # ------------------------------------------------------------------
    # Settings and control
    # ------------------------------------------------------------------

    def get_settings(self) -> AutoTranslateSettings:
        return self._settings.model_copy(deep=True)

    async def update_settings(self, settings: AutoTranslateSettings) -> dict:
        """Normalize, persist, and apply settings; ValueError means invalid settings.

        ``enabled``, ``paused``, and ``translate_all`` keep the service's
        current values: they change only through ``control`` actions, so a
        settings save that raced an Enable, Disable, Pause, Resume, or
        Translate all never reverts it. ``modes`` keeps its current value when
        the request leaves it out, so a client that does not know the field
        never resets it. ``batch_translation`` applies.
        """
        self._bind_loop()
        normalized = normalize_settings(settings, generic_mode=bool(system_config.generic_mode))
        async with self._control_lock:
            previous = self._settings
            kept: Dict[str, Any] = {
                "enabled": previous.enabled,
                "paused": previous.paused,
                "translate_all": previous.translate_all,
            }
            if "modes" not in settings.model_fields_set:
                kept["modes"] = list(previous.modes)
            normalized = normalized.model_copy(update=kept)
            await asyncio.to_thread(save_auto_translate_settings, normalized)
            roles_changed = self._roles_dump(previous) != self._roles_dump(normalized)
            modes_changed = previous.modes != normalized.modes
            removed = set(self._selected_keys()) - {language.key for language in normalized.languages}
            self._settings = normalized
            self._refresh_language_entries()
            # Roles are reconfigured before any job is cancelled, so a job the
            # worker starts while the cancellation settles already uses them.
            self._configure_roles()
            # The new job limit is known before anything is cancelled; until the
            # cancellations settle, the worker runs no more jobs than the lower
            # of the old and new limits. Counts started for older settings no
            # longer apply.
            limit = self._known_slot_limit()
            self._capacity_seq += 1
            self._set_slot_limit(min(self._slot_limit, limit))
            self._generation += 1
            self._repair = None
            if roles_changed or not self._can_run_settings(normalized):
                self._work_epoch += 1
                await self._cancel_jobs(lambda job: True)
            elif removed:
                await self._cancel_jobs(lambda job: job.language_key in removed)
            self._set_slot_limit(limit)
            self._refresh_lm_studio_capacity()
            if modes_changed:
                # Jobs of an unchecked mode stop at their next attempt boundary
                # after the rescan; Translate all enumerates the new modes afresh.
                self._reset_backlog()
                self._request_backlog()
            self._touch()
            self._sync_background_tasks()
            self._request_scan()
            self._wake_worker()
        await self._publish_status_if_changed()
        return self.status_snapshot()

    async def control(self, action: str) -> dict:
        """Apply one of ``CONTROL_ACTIONS``; returns status.

        "translate_all" starts translating older displayed outputs in the
        background (ValueError while translation is off; while paused it runs
        on resume). "cancel_translate_all" stops it: jobs nothing else targets
        are cancelled (their attempt in flight is discarded) and finished parts
        stay. Disable also turns Translate all off; Pause keeps it.
        """
        self._bind_loop()
        if action not in CONTROL_ACTIONS:
            raise ValueError(f"Unknown {DISPLAY_NAME} control action.")
        activity: Optional[Tuple[str, str]] = None
        async with self._control_lock:
            previous = self._settings
            if action == "retry":
                self._repair = None
                self._generation += 1
            else:
                if action == "translate_all" and not previous.enabled:
                    raise ValueError(f"Turn on {DISPLAY_NAME} before using Translate all.")
                update = {
                    "enable": {"enabled": True},
                    "disable": {"enabled": False, "translate_all": False},
                    "pause": {"paused": True},
                    "resume": {"paused": False},
                    "translate_all": {"translate_all": True},
                    "cancel_translate_all": {"translate_all": False},
                }[action]
                settings = previous.model_copy(update=update)
                await asyncio.to_thread(save_auto_translate_settings, settings)
                self._settings = settings
                self._generation += 1
                if action in ("disable", "pause"):
                    self._work_epoch += 1
                    await self._cancel_jobs(lambda job: True)
                if action == "enable":
                    self._repair = None
                if settings.translate_all != previous.translate_all:
                    self._reset_backlog()
                    if action == "translate_all":
                        activity = ("started", "Translate all started.")
                        # Loaded LM Studio copies decide whether Translate all's lane opens.
                        self._refresh_lm_studio_capacity()
                    elif action == "cancel_translate_all":
                        activity = ("cancelled", "Translate all cancelled.")
                        await self._cancel_jobs(lambda job: not self._job_targeted(job))
            self._touch()
            self._sync_background_tasks()
            if action in ("enable", "resume", "retry"):
                self._request_scan()
            if action == "translate_all":
                self._request_backlog()
            self._wake_worker()
        if activity is not None:
            await self._record_translate_all(*activity)
        await self._publish_status_if_changed()
        return self.status_snapshot()

    async def clear_cache(self) -> dict:
        """Drop every stored translation and job state, then translate known documents again.

        No job starts while the clear runs: one started now would read job
        states the clear is about to drop and save them back afterwards.
        Translate all turns off, so a clear never queues the whole library.
        """
        self._bind_loop()
        stopped_translate_all = False
        async with self._control_lock:
            self._generation += 1
            self._clearing = True
            try:
                self._work_epoch += 1
                await self._cancel_jobs(lambda job: True)
                async with self._write_lock:
                    records = [entry.record for doc in self._docs.values() for entry in doc.keys.values()]
                    await self._settled_thread(self._clear_and_restore_sync, records)
                    self._translated = {}
                    self._final_counts = {}
                    # Anything that read job states before the store was cleared
                    # is fenced at its first commit, and viewers that registered
                    # during the clear register again against the cleared store.
                    self._work_epoch += 1
                    self._generation += 1
            finally:
                self._clearing = False
            if self._settings.translate_all:
                await self._save_translate_all(False)
                stopped_translate_all = True
            self._touch()
            self._wake_worker()
        if stopped_translate_all:
            await self._record_translate_all(
                "cancelled", "Translate all was turned off because stored translations were cleared."
            )
        await self._publish_status_if_changed()
        return self.status_snapshot()

    @staticmethod
    def _clear_and_restore_sync(records: Sequence[DocumentRecord]) -> None:
        translation_store.clear_all()
        for record in records:
            translation_store.register_document(record)

    @staticmethod
    def _roles_dump(settings: AutoTranslateSettings) -> str:
        return json.dumps(
            {
                "translator": settings.translator.model_dump(mode="json") if settings.translator else None,
                "validator": settings.validator.model_dump(mode="json") if settings.validator else None,
            },
            sort_keys=True,
        )

    @staticmethod
    def _can_run_settings(settings: AutoTranslateSettings) -> bool:
        return settings.enabled and not settings.paused and settings_ready(settings)[0]

    def _configure_roles(self) -> None:
        if self._settings.translator is not None:
            self._configure_role(TRANSLATOR_ROLE_ID, self._settings.translator)
        if self._settings.validator is not None:
            self._configure_role(VALIDATOR_ROLE_ID, self._settings.validator)

    @staticmethod
    def _configure_role(role_id: str, role: TranslationRoleConfig) -> None:
        try:
            api_client_manager.configure_role(role_id, role.to_model_config())
        except ValueError as exc:
            logger.warning(
                "Auto Top-p Translate could not configure %s: %s",
                role_id,
                redact_log_text(exc, 200),
            )

    @staticmethod
    def _counts_loaded_copies(role: TranslationRoleConfig) -> bool:
        return role.provider == "lm_studio" and not system_config.generic_mode

    def _connection_capacity(self) -> int:
        """How many jobs the translator connection can run at once, without waiting for an LM Studio lookup.

        The smaller capacity of the translator and validator models: the
        per-model concurrency limit for OpenRouter and cloud providers, and for
        LM Studio the loaded same-base copies of the model as last counted (one
        until ``_refresh_lm_studio_capacity`` has counted them).
        """
        settings = self._settings
        roles = [role for role in (settings.translator, settings.validator) if role is not None]
        if not roles:
            return 1
        return min(
            self._lm_studio_copies.get(role.model_id, 1)
            if self._counts_loaded_copies(role)
            else max(1, int(system_config.max_model_concurrency_per_model or 1))
            for role in roles
        )

    def _known_slot_limit(self) -> int:
        """How many jobs may run at once: one without batch translation, else the connection's capacity."""
        return self._connection_capacity() if self._settings.batch_translation else 1

    def _background_lane(self) -> int:
        """Translate all's lane, the job only a background job may start in: 1 or 0.

        It is open while Translate all is on and the translator connection can
        run more than one job at once, so older outputs keep translating beside
        current-run and viewed work. Without batch translation it is one job
        beyond the job limit; with batch translation it is one of the limit's
        jobs, which the normal slots leave to it.
        """
        return int(self._translate_all_on() and self._connection_capacity() > 1)

    def _normal_slots(self) -> int:
        """How many jobs the normal slots hold: the job limit, within what the lane leaves of the connection."""
        return min(self._slot_limit, self._connection_capacity() - self._background_lane())

    def _refresh_lm_studio_capacity(self) -> None:
        """Count the loaded LM Studio copies for the current settings in a tracked background task.

        They are counted while batch translation or Translate all can use more
        than one job. The lookup runs the LM Studio CLI, so startup, settings
        saves, and control actions never wait for it. The count applies once
        it arrives; a count started for older settings is discarded.
        """
        self._capacity_seq += 1
        settings = self._settings
        models = sorted(
            {
                role.model_id
                for role in (settings.translator, settings.validator)
                if role is not None and self._counts_loaded_copies(role)
            }
        )
        if (
            not (settings.batch_translation or settings.translate_all)
            or not models
            or not self._started
            or self._stopping
        ):
            return
        self._track(
            asyncio.create_task(
                self._count_lm_studio_copies(self._capacity_seq, models),
                name="auto-translate-lm-studio-capacity",
            )
        )

    async def _count_lm_studio_copies(self, sequence: int, models: Sequence[str]) -> None:
        try:
            loaded = await lm_studio_client.get_loaded_models()
        except Exception as exc:
            logger.debug("LM Studio loaded-model lookup failed: %s", type(exc).__name__)
            loaded = []
        if sequence != self._capacity_seq or self._stopping:
            return
        for model_id in models:
            self._lm_studio_copies[model_id] = max(
                1, lm_studio_client.count_sibling_instances_from_loaded(model_id, loaded)
            )
        self._set_slot_limit(self._known_slot_limit())
        self._wake_worker()

    def _set_slot_limit(self, limit: int) -> None:
        """Apply a job limit; running jobs beyond it yield at their next attempt boundary.

        A raised limit lets a job that has not yielded yet keep running. The
        worker starts no job while the running jobs fill the limit.
        """
        self._slot_limit = limit
        self._refresh_yields()
        self._touch()

    def _job_rank(self, job: _Job, now: float) -> _JobRank:
        languages = self._active_language_keys()
        index = languages.index(job.language_key) if job.language_key in languages else len(languages)
        return self._priority(self._docs[job.doc_hash], job.language_key, index, now)

    @staticmethod
    def _foreground(rank: _JobRank) -> bool:
        """The rank is a viewed or current-run job's, which starts only in the normal job slots."""
        return rank[0] == 0 or rank[1] == TIER_CURRENT

    def _holds_lane(self, job: _Job) -> bool:
        """The job started in Translate all's lane, which is still open.

        It holds the lane until it ends, including while it stops. Once the
        lane closes, it holds a normal slot.
        """
        return job.lane and bool(self._background_lane())

    def _lane_jobs(self) -> int:
        """How many jobs hold Translate all's lane; every other job holds a normal slot."""
        return sum(1 for job in self._jobs.values() if self._holds_lane(job))

    def _refresh_yields(self) -> None:
        """Ask running jobs that must give their slot back to yield at their next attempt boundary.

        The job holding Translate all's lane keeps it. The other running jobs
        are ranked in dispatch order and keep the job limit's slots, within
        what that lane job leaves of the connection, viewed and current-run
        jobs first, and the jobs past those yield. Only a running lane job
        counts, so opening the lane never asks a job to yield. A background job
        in a normal slot also yields, worst-ranked first, for each viewed or
        current-run job that waits with no free normal slot; current-run jobs
        are never preempted. State is saved after every attempt, so nothing is
        lost, and requests are recomputed on every call, so one clears once its
        cause is gone.
        """
        now = time.time()
        running = sorted(
            (
                job
                for job in self._jobs.values()
                if not job.cancelled and job.doc_hash in self._docs and job.task is not None and not job.task.done()
            ),
            key=lambda job: self._job_rank(job, now),
        )
        limit = min(self._slot_limit, self._connection_capacity() - self._lane_jobs())
        kept = 0
        kept_background: List[_Job] = []
        for job in running:
            if self._holds_lane(job):
                job.yield_requested = False
                continue
            job.yield_requested = kept >= limit
            if not job.yield_requested:
                kept += 1
                if not self._foreground(self._job_rank(job, now)):
                    kept_background.append(job)
        # Background jobs in normal slots, worst-ranked first.
        preemptible = [
            job for job in reversed(kept_background) if self._job_rank(job, now)[:2] == (1, TIER_BACKGROUND)
        ]
        if not preemptible:
            return
        waiting_ahead = sum(1 for rank, _key in self._ranked_waiting(now) if self._foreground(rank))
        free = max(0, self._normal_slots() - (len(self._jobs) - self._lane_jobs()))
        for job in preemptible[: max(0, waiting_ahead - free)]:
            job.yield_requested = True

    async def _cancel_jobs(self, predicate: Callable[[_Job], bool]) -> None:
        tasks = []
        for job in list(self._jobs.values()):
            if predicate(job):
                job.cancelled = True
                if job.task is not None and job.task is not asyncio.current_task():
                    job.task.cancel()
                    tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def _add_doc(self, record: DocumentRecord, material: _DocMaterial) -> _Doc:
        """Track ``record.doc_key`` on its document; a key moving to new content leaves its old one."""
        existing_hash = self._doc_by_key.get(record.doc_key)
        if existing_hash and existing_hash != record.doc_hash:
            self._remove_key(record.doc_key)
        doc = self._docs.get(record.doc_hash)
        if doc is None:
            self._doc_seq += 1
            doc = _make_doc(record.doc_hash, material, self._doc_seq)
            self._docs[record.doc_hash] = doc
        viewed_ts = _iso_to_ts(record.last_viewed_at)
        entry = doc.keys.get(record.doc_key)
        if entry is None:
            doc.keys[record.doc_key] = _DocKey(record=record, viewed_ts=viewed_ts)
        else:
            entry.record = record
            entry.viewed_ts = max(entry.viewed_ts, viewed_ts)
        self._doc_by_key[record.doc_key] = record.doc_hash
        self._touch()
        return doc

    def _remove_key(self, doc_key: str) -> None:
        """Forget one doc key; its document goes only when no other key remains."""
        doc_hash = self._doc_by_key.pop(doc_key, None)
        doc = self._docs.get(doc_hash) if doc_hash else None
        if doc is None:
            return
        doc.keys.pop(doc_key, None)
        if not doc.keys:
            self._remove_doc(doc_hash)
        self._touch()

    def _remove_doc(self, doc_hash: str) -> None:
        doc = self._docs.pop(doc_hash, None)
        if doc is None:
            return
        for doc_key in doc.keys:
            if self._doc_by_key.get(doc_key) == doc_hash:
                del self._doc_by_key[doc_key]
        for key in [key for key in self._final_counts if key[0] == doc_hash]:
            del self._final_counts[key]
        self._fence_doc_jobs(doc_hash)
        self._touch()

    def _fence_doc_jobs(self, doc_hash: str) -> None:
        """Stop the jobs of a document that is going away; they can no longer save."""
        for job in self._jobs.values():
            if job.doc_hash == doc_hash:
                job.cancelled = True
                if job.task is not None and job.task is not asyncio.current_task():
                    job.task.cancel()

    def _doc_tier(self, doc: _Doc, language_key: str, now: Optional[float] = None) -> Optional[int]:
        """The tier ``doc`` is translated at in ``language_key``, or None when nothing targets it.

        A checked mode's current run, a document a user requested in that
        language, and a document a viewer shows in that language while it is
        being viewed are ``TIER_CURRENT``; an older output Translate all found
        is ``TIER_BACKGROUND``.
        """
        if any(doc_key in self._auto_keys for doc_key in doc.keys):
            return TIER_CURRENT
        if self._user_requested(doc, language_key):
            return TIER_CURRENT
        if doc.requested(language_key) and self._viewing(doc, now):
            return TIER_CURRENT
        if any(doc_key in self._backlog_keys for doc_key in doc.keys):
            return TIER_BACKGROUND
        return None

    def _user_requested(self, doc: _Doc, language_key: str) -> bool:
        return any(language_key in self._requests.get(doc_key, ()) for doc_key in doc.keys)

    def _doc_targets(self, doc: _Doc, language_key: str) -> bool:
        return self._doc_tier(doc, language_key) is not None

    def _job_targeted(self, job: _Job) -> bool:
        doc = self._docs.get(job.doc_hash)
        return doc is not None and self._doc_targets(doc, job.language_key)

    def _translate_all_only(self, doc: _Doc) -> bool:
        """Only Translate all found the document: it has an older-output key and no current-run key."""
        return any(doc_key in self._backlog_keys for doc_key in doc.keys) and not any(
            doc_key in self._auto_keys for doc_key in doc.keys
        )

    def _viewing(self, doc: _Doc, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else now
        return bool(doc.viewed_ts) and current - doc.viewed_ts <= VIEWING_WINDOW_SECONDS

    def _schedule_viewing_expiry(self) -> None:
        """Re-evaluate the queue when the earliest active viewing window ends.

        Viewing priority expires with time alone, so this pacing timer (not a
        retry wait) lets waiting documents' queue positions follow a viewed
        document leaving the viewing tier.
        """
        if self._viewing_expiry_handle is not None:
            self._viewing_expiry_handle.cancel()
            self._viewing_expiry_handle = None
        if not self._started or self._stopping or self._loop is None:
            return
        now = time.time()
        expiries = [doc.viewed_ts + VIEWING_WINDOW_SECONDS for doc in self._docs.values() if self._viewing(doc, now)]
        if expiries:
            self._viewing_expiry_handle = self._loop.call_later(
                min(expiries) - now + VIEWING_EXPIRY_MARGIN_SECONDS, self._viewing_window_expired
            )

    def _viewing_window_expired(self) -> None:
        self._touch()
        self._wake_worker()
        self._note_queue_change()
        self._schedule_viewing_expiry()

    async def _refresh_viewed(self, docs: Sequence[_Doc]) -> None:
        """Mark documents a viewer still shows as viewed now, as a viewing registration does.

        The in-memory viewed time, which orders the queue, refreshes every
        time. The saved one, which restores viewing priority after a restart,
        is written only once it is ``VIEWED_SAVE_INTERVAL_SECONDS`` old, so the
        lookups a viewer sends on every queue change stay read-only.
        """
        now = _now_iso()
        now_ts = _iso_to_ts(now)
        due: List[str] = []
        for doc in docs:
            entry = doc.keys[doc.primary.doc_key]
            entry.viewed_ts = max(entry.viewed_ts, now_ts)
            if now_ts - _iso_to_ts(entry.record.last_viewed_at) >= VIEWED_SAVE_INTERVAL_SECONDS:
                entry.record = replace(entry.record, last_seen_at=now, last_viewed_at=now)
                due.append(entry.record.doc_key)
        if due:
            async with self._write_lock:
                await self._settled_thread(translation_store.mark_documents_viewed, due, now)
        self._schedule_viewing_expiry()

    async def register_documents(
        self,
        language_key: Optional[str],
        viewing: bool,
        documents: List[Any],
        request: bool = False,
    ) -> List[dict]:
        """Register viewer documents and return their status for ``language_key``.

        Documents are requested in that language, or in every selected
        language when ``language_key`` is None. A document no current run,
        Translate all, or user request covers translates only while it is
        being viewed: within ``VIEWING_WINDOW_SECONDS`` of a viewing
        registration or lookup. With ``request``, the user asked for the
        translation, which is saved and translates at the current-run tier
        until done, whether or not the document is still viewed.
        """
        self._bind_loop()
        specs = [coerce_document_input(item) for item in documents]
        if not self._settings.enabled:
            hashes = await asyncio.to_thread(
                lambda: [expected_doc_hash(spec.profile, spec.parts) for spec in specs]
            )
            return [
                self._status_row(spec.doc_key, doc_hash, language_key or "", "disabled")
                for spec, doc_hash in zip(specs, hashes)
            ]
        rows = await self._register_specs(
            specs,
            language_key=language_key,
            viewing=viewing,
            origin="viewer",
        )
        if request:
            await self._save_requests(rows, language_key)
        if viewing:
            self._schedule_viewing_expiry()
        self._wake_worker()
        await self._publish_status_if_changed()
        target_language = self._status_language(language_key)
        if target_language is None:
            return [
                self._status_row(spec.doc_key, doc_hash, language_key or "", "unavailable")
                for spec, doc_hash in rows
            ]
        return await self._document_statuses(
            [(spec.doc_key, doc_hash) for spec, doc_hash in rows],
            target_language,
        )

    async def _save_requests(self, rows: Sequence[Tuple[DocumentSpec, str]], language_key: Optional[str]) -> None:
        """Save a user's request for registered documents in ``language_key`` (None: every active language)."""
        languages = [language_key] if language_key else self._active_language_keys()
        languages = [key for key in languages if key in self._language_entries]
        pairs = list(dict.fromkeys(
            (spec.doc_key, key)
            for spec, doc_hash in rows
            if doc_hash in self._docs and spec.doc_key in self._docs[doc_hash].keys
            for key in languages
            if key not in self._requests.get(spec.doc_key, ())
        ))
        if not pairs:
            return
        async with self._write_lock:
            await self._settled_thread(translation_store.save_requests, pairs, _now_iso())
            for doc_key, key in pairs:
                self._requests.setdefault(doc_key, set()).add(key)
            self._touch()

    def _status_language(self, language_key: Optional[str]) -> Optional[str]:
        selected = self._selected_keys()
        if language_key is None:
            return selected[0] if selected else None
        return language_key if language_key in self._language_entries else None

    async def _register_specs(
        self,
        specs: Sequence[DocumentSpec],
        *,
        language_key: Optional[str],
        viewing: bool,
        origin: str,
    ) -> List[Tuple[DocumentSpec, str]]:
        """Plan, persist, and track documents; returns (spec, doc_hash) per spec.

        A doc key that moves to new content carries its job progress over
        (``reconcile_job_state``) for every language.
        """
        if not specs:
            return []
        planned = await asyncio.to_thread(_plan_specs_sync, list(specs), set(self._docs))
        now = _now_iso()
        now_ts = _iso_to_ts(now)
        records: List[DocumentRecord] = []
        viewed_keys: List[str] = []
        new_docs: List[Tuple[DocumentRecord, _DocMaterial]] = []
        moves: List[Tuple[str, str, Tuple[UnitView, ...], Tuple[UnitView, ...]]] = []
        moving_keys: Dict[str, Set[str]] = {}
        rows: List[Tuple[DocumentSpec, str]] = []
        for spec, doc_hash, material, error in planned:
            rows.append((spec, doc_hash))
            if error or not doc_hash:
                logger.warning(
                    "Auto Top-p Translate skipped invalid document %s: %s",
                    redact_log_text(spec.doc_key, 120),
                    redact_log_text(error, 200),
                )
                continue
            existing = self._docs.get(doc_hash)
            entry = existing.keys.get(spec.doc_key) if existing is not None else None
            # A key keeps the strongest origin it has had: auto, then backlog, then viewer.
            origins = [origin]
            if spec.doc_key in self._auto_keys:
                origins.append("auto")
            if spec.doc_key in self._backlog_keys:
                origins.append("backlog")
            if entry is not None:
                origins.append(entry.record.origin)
            effective_origin = min(origins, key=lambda item: ORIGIN_RANK.get(item, len(ORIGIN_RANK)))
            if entry is not None:
                updates: Dict[str, Any] = {"last_seen_at": now}
                if viewing:
                    updates["last_viewed_at"] = now
                changed = (
                    entry.record.origin != effective_origin
                    or entry.record.title != spec.title
                    or entry.record.kind != spec.kind
                )
                if changed:
                    updates.update(
                        origin=effective_origin,
                        title=spec.title,
                        kind=spec.kind,
                        priority_class=PRIORITY_CLASS_BY_KIND.get(spec.kind, 2),
                    )
                entry.record = replace(entry.record, **updates)
                if changed:
                    records.append(entry.record)
                elif viewing:
                    # Unviewed re-registrations (such as every rescan) skip the store write.
                    viewed_keys.append(spec.doc_key)
                continue
            if material is None and existing is not None:
                material = _DocMaterial(plan=existing.plan, plan_json=existing.plan_json, views=existing.views)
            elif material is None:
                # The known document was replaced while planning ran; plan it now.
                material = await asyncio.to_thread(_build_material_sync, spec)
            previous_hash = self._doc_by_key.get(spec.doc_key)
            previous = self._docs.get(previous_hash) if previous_hash else None
            if previous is not None and previous_hash != doc_hash:
                moves.append((previous_hash, doc_hash, previous.views, material.views))
                moving_keys.setdefault(previous_hash, set()).add(spec.doc_key)
            record = DocumentRecord(
                doc_hash=doc_hash,
                doc_key=spec.doc_key,
                kind=spec.kind,
                title=spec.title,
                priority_class=PRIORITY_CLASS_BY_KIND.get(spec.kind, 2),
                plan_json=material.plan_json,
                origin=effective_origin,
                created_at=now,
                last_seen_at=now,
                last_viewed_at=now if viewing else "",
            )
            records.append(record)
            new_docs.append((record, material))

        async with self._write_lock:
            # A document losing its last key stops its jobs before its job
            # states are read, so no later save is missed by the carry-over.
            for old_hash, keys in moving_keys.items():
                old_doc = self._docs.get(old_hash)
                if old_doc is not None and set(old_doc.keys) <= keys:
                    self._fence_doc_jobs(old_hash)
            carried = await self._settled_thread(
                _persist_registration_sync, records, viewed_keys, now, moves
            )
            for record, material in new_docs:
                self._add_doc(record, material)
            for key, count in carried.items():
                if key[0] in self._docs:
                    self._final_counts[key] = count
            for spec, doc_hash in rows:
                doc = self._docs.get(doc_hash) if doc_hash else None
                entry = doc.keys.get(spec.doc_key) if doc is not None else None
                if entry is None:
                    # Invalid, or the key moved to other content later in this batch.
                    continue
                if viewing:
                    entry.viewed_ts = max(entry.viewed_ts, now_ts)
                if origin == "viewer":
                    if language_key is None:
                        entry.requested_all = True
                    else:
                        entry.requested_languages.add(language_key)
            self._touch()
        return rows

    async def document_progress(self, documents: List[Any]) -> List[dict]:
        """Each document's finished share in every selected language, without side effects.

        Unlike registration and lookup, this never registers, views, or
        requests a document, so a viewer can show which languages are ready
        for content it is not translating. Content the service has not seen
        reports ``unknown``; every language reports ``disabled`` while
        translation is off.
        """
        specs = [coerce_document_input(item) for item in documents]
        hashes = await asyncio.to_thread(
            lambda: [expected_doc_hash(spec.profile, spec.parts) for spec in specs]
        )
        languages = self._active_language_keys()
        enabled = self._settings.enabled
        rows = []
        for spec, doc_hash in zip(specs, hashes):
            doc = self._docs.get(doc_hash) if enabled else None
            rows.append({
                "doc_key": spec.doc_key,
                "doc_hash": doc_hash,
                "languages": [self._language_progress(doc, key, enabled) for key in languages],
            })
        return rows

    def _language_progress(self, doc: Optional[_Doc], language_key: str, enabled: bool) -> dict:
        if doc is None:
            status, done, total = ("unknown" if enabled else "disabled"), 0, 0
        else:
            total = doc.unit_count
            done = min(self._final_counts.get((doc.doc_hash, language_key), 0), total)
            status = "complete" if done >= total else "pending"
        return {"language": language_key, "status": status, "translated_units": done, "total_units": total}

    async def lookup_documents(self, language_key: str, doc_hashes: List[str], viewing: bool = False) -> List[dict]:
        """Return the status of registered documents for ``language_key``.

        A lookup requests the language for documents nothing else targets; with
        ``viewing``, the documents a viewer still shows keep their viewing
        priority (and so their targeting): their viewed time is refreshed as a
        viewing registration refreshes it, and the queue is re-evaluated.
        """
        self._bind_loop()
        unique = list(dict.fromkeys(str(item) for item in doc_hashes))
        if not self._settings.enabled:
            return [self._lookup_row(doc_hash, language_key, "disabled") for doc_hash in unique]
        if language_key not in self._language_entries:
            return [self._lookup_row(doc_hash, language_key, "unavailable") for doc_hash in unique]
        known = [self._docs[doc_hash] for doc_hash in unique if doc_hash in self._docs]
        changed = False
        for doc in known:
            entry = doc.keys[doc.primary.doc_key]
            if not self._doc_targets(doc, language_key) and language_key not in entry.requested_languages:
                entry.requested_languages.add(language_key)
                changed = True
        if viewing and known:
            await self._refresh_viewed(known)
            changed = True
        if changed:
            self._touch()
            self._wake_worker()
            await self._publish_status_if_changed()
        # A lookup names only content hashes, so rows carry each document's primary key.
        return await self._document_statuses(
            [
                (self._docs[doc_hash].primary.doc_key if doc_hash in self._docs else "", doc_hash)
                for doc_hash in unique
            ],
            language_key,
        )

    def _lookup_row(self, doc_hash: str, language_key: str, status: str) -> dict:
        doc = self._docs.get(doc_hash)
        return self._status_row(doc.primary.doc_key if doc else "", doc_hash, language_key, status)

    @staticmethod
    def _status_row(
        doc_key: str,
        doc_hash: str,
        language_key: str,
        status: str,
        *,
        translated: int = 0,
        total: int = 0,
        position: Optional[int] = None,
        parts: Optional[Dict[str, str]] = None,
        requested: bool = False,
        automatic: bool = False,
    ) -> dict:
        return {
            "doc_key": doc_key,
            "doc_hash": doc_hash,
            "language": language_key,
            "status": status,
            "translated_units": translated,
            "total_units": total,
            "queue_position": position,
            "parts": parts,
            "requested": requested,
            "automatic": automatic,
        }

    def _target_flags(self, doc: _Doc, language_key: str) -> Dict[str, bool]:
        """Whether a user requested ``doc`` in ``language_key``, and whether a checked mode's current run holds it."""
        return {
            "requested": self._user_requested(doc, language_key),
            "automatic": any(doc_key in self._auto_keys for doc_key in doc.keys),
        }

    async def _document_statuses(
        self,
        entries: Sequence[Tuple[str, str]],
        language_key: str,
    ) -> List[dict]:
        """Status rows for (doc_key, doc_hash) entries; each row echoes its entry's doc_key."""
        positions = self._stats().positions
        rows: List[Optional[dict]] = []
        complete: List[Tuple[int, str, _Doc]] = []
        for index, (doc_key, doc_hash) in enumerate(entries):
            doc = self._docs.get(doc_hash) if doc_hash else None
            if doc is None:
                rows.append(self._status_row(doc_key, doc_hash, language_key, "unknown"))
                continue
            total = doc.unit_count
            done = self._final_counts.get((doc_hash, language_key), 0)
            if done >= total:
                rows.append(None)
                complete.append((index, doc_key, doc))
            else:
                rows.append(
                    self._status_row(
                        doc_key,
                        doc_hash,
                        language_key,
                        "pending",
                        translated=done,
                        total=total,
                        position=positions.get((doc_hash, language_key)),
                        **self._target_flags(doc, language_key),
                    )
                )
        if complete:
            distinct = list({doc.doc_hash: doc for _index, _doc_key, doc in complete}.values())
            assembled = await asyncio.to_thread(
                _assemble_sync,
                [(doc.plan, doc.doc_hash, doc.unit_count) for doc in distinct],
                language_key,
            )
            results = {doc.doc_hash: result for doc, result in zip(distinct, assembled)}
            repaired = False
            for index, doc_key, doc in complete:
                parts, count = results[doc.doc_hash]
                total = doc.unit_count
                if parts is None:
                    # The store no longer has every final; translate the rest again.
                    if doc.doc_hash in self._docs:
                        self._final_counts[(doc.doc_hash, language_key)] = count
                    repaired = True
                    rows[index] = self._status_row(
                        doc_key,
                        doc.doc_hash,
                        language_key,
                        "pending",
                        translated=count,
                        total=total,
                        **self._target_flags(doc, language_key),
                    )
                    continue
                rows[index] = self._status_row(
                    doc_key,
                    doc.doc_hash,
                    language_key,
                    "empty" if total == 0 else "complete",
                    translated=total,
                    total=total,
                    parts=parts,
                    **self._target_flags(doc, language_key),
                )
            if repaired:
                self._touch()
                self._wake_worker()
        return [row for row in rows if row is not None]

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def _priority(self, doc: _Doc, language_key: str, language_index: int, now: float) -> _JobRank:
        """Dispatch rank of one document x language job; lower runs first.

        Order: documents viewed within the viewing window, then the tier
        (current runs before Translate all's background documents), then
        priority class (proofs, then papers / final answers / live papers, then
        the rest), then the user's language order, then document order.
        """
        tier = self._doc_tier(doc, language_key, now)
        viewing = tier == TIER_CURRENT and self._viewing(doc, now)
        # A running job that lost its target ranks last; it stops at its next attempt boundary.
        return (
            0 if viewing else 1,
            TIER_BACKGROUND + 1 if tier is None else tier,
            doc.priority_class,
            language_index,
            doc.order,
        )

    def _ranked_waiting(self, now: Optional[float] = None) -> List[Tuple[_JobRank, Tuple[str, str]]]:
        """(rank, job key) of jobs that need work and are not running, in dispatch order (``_priority``).

        It is evaluated fresh, so an expired viewing boost or a newly viewed
        document takes effect at the next dispatch.
        """
        current = time.time() if now is None else now
        languages = self._active_language_keys()
        items: List[Tuple[_JobRank, Tuple[str, str]]] = []
        for doc in self._docs.values():
            if doc.unit_count == 0:
                continue
            for index, language_key in enumerate(languages):
                key = (doc.doc_hash, language_key)
                if key in self._jobs or self._doc_tier(doc, language_key, current) is None:
                    continue
                if self._final_counts.get(key, 0) >= doc.unit_count:
                    continue
                items.append((self._priority(doc, language_key, index, current), key))
        items.sort()
        return items

    def _waiting_jobs(self, now: Optional[float] = None) -> List[Tuple[str, str]]:
        """Jobs that need work and are not running, in dispatch order."""
        return [key for _rank, key in self._ranked_waiting(now)]

    def _dispatch(self) -> None:
        """Start waiting jobs while a slot is free.

        Translate all's lane, while free and while the running jobs leave the
        translator connection a free job, takes the best waiting background
        job; a free normal slot takes the best-ranked waiting job.
        """
        while self._can_dispatch():
            ranked = self._ranked_waiting()
            lane_jobs = self._lane_jobs()
            background = [key for rank, key in ranked if not self._foreground(rank)]
            if (
                background
                and lane_jobs < self._background_lane()
                and len(self._jobs) < self._connection_capacity()
            ):
                self._start_job(*background[0], lane=True)
            elif ranked and len(self._jobs) - lane_jobs < self._normal_slots():
                self._start_job(*ranked[0][1])
            else:
                break
        self._refresh_yields()
        self._note_queue_change()

    def _start_job(self, doc_hash: str, language_key: str, *, lane: bool = False) -> None:
        job = _Job(doc_hash=doc_hash, language_key=language_key, epoch=self._work_epoch, lane=lane)
        job.doc_title = self._docs[doc_hash].primary.title
        self._jobs[(doc_hash, language_key)] = job
        job.task = asyncio.create_task(self._run_job(job), name=f"auto-translate-{doc_hash[:10]}-{language_key}")
        job.task.add_done_callback(lambda task, current=job: self._job_done(current, task))
        self._touch()

    def _job_done(self, job: _Job, task: asyncio.Task) -> None:
        key = (job.doc_hash, job.language_key)
        if self._jobs.get(key) is job:
            del self._jobs[key]
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("Auto Top-p Translate job failed: %s", type(error).__name__)
        self._touch()
        self._wake_worker()
        if not self._stopping:
            self._note_queue_change()
            self._schedule_backlog_check()

    def _queue_signature(self) -> Tuple[Any, ...]:
        return (tuple(sorted(self._jobs)), tuple(self._waiting_jobs()))

    def _note_queue_change(self) -> None:
        """Broadcast ``translation_queue_changed`` when the queue changed, at most once per second."""
        if not self._started or self._stopping or self._loop is None:
            return
        if self._queue_signature() == self._queue_signature_sent:
            return
        wait = QUEUE_EVENT_MIN_INTERVAL_SECONDS - (time.monotonic() - self._queue_event_at)
        if wait > 0:
            if self._queue_event_handle is None:
                self._queue_event_handle = self._loop.call_later(wait, self._flush_queue_change)
            return
        self._flush_queue_change()

    def _flush_queue_change(self) -> None:
        self._queue_event_handle = None
        if not self._started or self._stopping:
            return
        signature = self._queue_signature()
        if signature == self._queue_signature_sent:
            return
        self._queue_signature_sent = signature
        self._queue_version += 1
        self._queue_event_at = time.monotonic()
        self._track(
            asyncio.create_task(
                self._broadcast(
                    QUEUE_CHANGED_EVENT,
                    {
                        "queue_version": self._queue_version,
                        "activity_id": f"{QUEUE_CHANGED_EVENT}:{self._queue_version}",
                    },
                    classification="informational",
                    category="workflow",
                    outcome="info",
                )
            )
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def _run_job(self, job: _Job) -> None:
        """Translate parts of one document until a part finalizes, a wait happens, it must yield, or it stops.

        The job then ends and the worker dispatches the highest-priority job
        next, which may be this one again.
        """
        _CURRENT_JOB.set(job)
        try:
            doc = self._docs.get(job.doc_hash)
            if doc is None:
                return
            try:
                state = await self._load_state(job, doc)
                while True:
                    self._raise_if_stopped(job)
                    if (
                        job.yield_requested
                        or len(state["finals"]) >= doc.unit_count
                        or not self._doc_targets(doc, job.language_key)
                    ):
                        return
                    finalized = await self._run_attempt(job, doc, state)
                    if finalized or job.waited:
                        return
            except (_JobStopped, asyncio.CancelledError):
                raise
            except Exception as exc:
                logger.error(
                    "Auto Top-p Translate attempt failed: %s: %s",
                    type(exc).__name__,
                    redact_log_text(exc, 300),
                )
                await self._wait_after_unexpected_error(job, SERVICE_ROLE_ID, exc)
        except _JobStopped:
            return

    async def _load_state(self, job: _Job, doc: _Doc) -> Dict[str, Any]:
        """Load (or start) the job state, seeding units already translated elsewhere."""
        key = (doc.doc_hash, job.language_key)
        previous = self._final_counts.get(key, 0)
        seed = bool(doc.unit_hashes & self._translated.get(job.language_key, set()))
        state, changed = await asyncio.to_thread(
            _load_job_state_sync, doc.doc_hash, job.language_key, doc.views, seed
        )
        if changed:
            await self._commit(job, doc, state)
        else:
            self._final_counts[key] = len(state["finals"])
            self._touch()
        if len(state["finals"]) != previous:
            await self._broadcast_progress(job, doc, len(state["finals"]))
        self._update_job_info(job, doc, state)
        return state

    def _update_job_info(self, job: _Job, doc: _Doc, state: Mapping[str, Any]) -> None:
        part = state.get("part")
        if part:
            job.part_start = part["start"] + 1
            job.part_end = part["end"] + 1
            job.attempt = next_attempt_number(part)
            job.accepted_count = len(_accepted_records(part))
            job.keep_streak = part["keep_streak"]
        else:
            job.part_start = job.part_end = job_cursor(doc.unit_count, state["finals"]) + 1
            job.attempt = 1
            job.accepted_count = 0
            job.keep_streak = 0

    async def _run_attempt(self, job: _Job, doc: _Doc, state: Dict[str, Any]) -> bool:
        """Run one attempt at the current part; returns True when the part finalized."""
        settings = self._settings
        language = self._language_entries.get(job.language_key)
        if language is None or settings.translator is None or settings.validator is None:
            raise _JobStopped()
        descriptor = language_descriptor(language)
        part = state.get("part")
        cursor = job_cursor(doc.unit_count, state["finals"])
        if part is not None and part_stop_decision(part) is not None:
            # The deciding attempt was saved, but its part was not finalized yet.
            span = (part["start"], part["end"])
            ctx = _PartContext(doc, language, descriptor, state, part, cursor, span, span[1], bool(part["locked"]))
            return await self._after_attempt(job, ctx, [])
        if part is None:
            part = new_part(cursor)
            state["part"] = part
        locked = bool(part["locked"])
        if locked:
            core = (part["start"], part["end"])
            max_end = part["end"]
        else:
            inputs = [self._available_input(settings.translator), self._available_input(settings.validator)]
            chosen_end = part_max_end(
                doc.token_counts,
                state["finals"],
                cursor,
                settings.translator.max_output_tokens,
                min((value for value in inputs if value), default=None),
            )
            if chosen_end is None:
                await self._enter_output_budget_repair()
                raise _JobStopped()
            max_end = chosen_end
            core = (cursor, max_end)
        ctx = _PartContext(doc, language, descriptor, state, part, cursor, core, max_end, locked)
        number = next_attempt_number(part)
        self._update_job_info(job, doc, state)
        turn = await self._translator_turn(job, ctx)
        if turn.attempt is None:
            return await self._record_check_failure(job, ctx, number, TranslatorAttempt(ACTION_TRANSLATE), turn.problems)
        applied = applied_part_edit_keys(part, state["finals"], doc.views)
        attempt = self._keep_if_identical(turn.attempt, ctx, applied)
        if attempt.action == ACTION_KEEP:
            return await self._handle_keep(job, ctx, number, attempt)
        attempt = without_applied_edits(attempt, applied)
        problems = precheck_attempt(
            doc.views, state["finals"], part, attempt, max_end, plan=doc.plan, visible=turn.shown
        )
        if problems:
            return await self._record_check_failure(job, ctx, number, attempt, problems)
        record = attempt_record(attempt, doc.views, number=number, status=_STATUS_PENDING)
        try:
            decisions = await self._judge(job, ctx, [record], round_index=number)
        except _EditsNotShown as hidden:
            return await self._record_check_failure(
                job, ctx, number, attempt, [self._edits_not_shown_problem(hidden.units, dispute=False)]
            )
        # The part changes only once the attempt is judged, so a stop while
        # judging leaves the saved part as it was.
        if not locked:
            part["end"] = max(_as_int(key) for key in record["units"])
        part["keep_streak"] = 0
        part["attempts"].append(record)
        edited, events = self._apply_decisions(job, ctx, [record], decisions, round_index=number, rejudged=False)
        return await self._after_attempt(job, ctx, events, edited=edited)

    def _keep_if_identical(
        self, attempt: TranslatorAttempt, ctx: _PartContext, applied: Collection[Tuple[int, str, str]]
    ) -> TranslatorAttempt:
        """An attempt repeating an earlier attempt keeps that attempt instead of being judged.

        It repeats record N when its unit texts equal N's (whitespace
        normalized) and, leaving out on both sides the edits ``applied`` says
        are already in the translation, either its edits are exactly N's, or N
        was accepted and it has no other edits. Records are searched newest
        first.
        """
        if attempt.action != ACTION_TRANSLATE:
            return attempt

        def normalized(units: Sequence[Tuple[int, str]]) -> List[Tuple[int, str]]:
            return sorted((number, " ".join(text.split())) for number, text in units)

        units = normalized(attempt.units)
        edits = without_applied_edits(attempt, applied).edits
        for record in reversed(ctx.part["attempts"]):
            if record["status"] not in (STATUS_ACCEPTED, STATUS_REJECTED) or record["action"] != ACTION_TRANSLATE:
                continue
            earlier = without_applied_edits(attempt_from_record(record, ctx.doc.views), applied)
            if normalized(earlier.units) != units:
                continue
            if edits == earlier.edits or (record["status"] == STATUS_ACCEPTED and not edits):
                return TranslatorAttempt(
                    action=ACTION_KEEP,
                    keep_attempts=(record["number"],),
                    translator_notes=attempt.translator_notes,
                )
        return attempt

    async def _handle_keep(
        self,
        job: _Job,
        ctx: _PartContext,
        number: int,
        attempt: TranslatorAttempt,
    ) -> bool:
        """A keep naming only accepted attempts is a refusal; naming rejected ones is a dispute."""
        views, part, state = ctx.doc.views, ctx.part, ctx.state
        problems = precheck_attempt(views, state["finals"], part, attempt, ctx.max_end, plan=ctx.doc.plan)
        if problems:
            return await self._record_check_failure(job, ctx, number, attempt, problems)
        by_number = {record["number"]: record for record in part["attempts"]}
        rejected = [
            by_number[kept] for kept in attempt.keep_attempts if by_number[kept]["status"] == STATUS_REJECTED
        ]
        record = attempt_record(attempt, views, number=number, status=STATUS_KEPT)
        if not rejected:
            stamp_judgment(part, record)
            part["keep_streak"] += 1
            part["attempts"].append(record)
            return await self._after_attempt(
                job, ctx, [self._kept_event(job, ctx, record, dispute=False)]
            )
        problems, judged = self._dispute_check(ctx, rejected)
        if problems:
            return await self._record_check_failure(job, ctx, number, attempt, problems)
        try:
            decisions = await self._judge(
                job, ctx, judged, round_index=number, resubmission_notes=attempt.translator_notes
            )
        except _EditsNotShown as hidden:
            return await self._record_check_failure(
                job, ctx, number, attempt, [self._edits_not_shown_problem(hidden.units, dispute=True)]
            )
        # Stamped before the re-judged attempts, so they follow this keep in judgment order.
        stamp_judgment(part, record)
        part["keep_streak"] = 0
        part["attempts"].append(record)
        edited, events = self._apply_decisions(job, ctx, rejected, decisions, round_index=number, rejudged=True)
        return await self._after_attempt(
            job, ctx, [self._kept_event(job, ctx, record, dispute=True), *events], edited=edited
        )

    @staticmethod
    def _dispute_check(
        ctx: _PartContext, rejected: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Re-check disputed attempts in order; returns (problems, the attempts to judge).

        Each attempt is checked against the translation as the edits of the
        attempts named before it would leave it, the order in which accepted
        edits apply. An edit identical to one an earlier named attempt or an
        accepted attempt of the part makes is already satisfied, and the
        validator judges each attempt without the edits accepted attempts
        already made. Edits are not limited to the translator's current
        window: the validator is shown every edited unit or the dispute fails
        its own check.
        """
        ranges = {tuple(sorted(_as_int(key) for key in record["units"])) for record in rejected}
        if len(ranges) > 1:
            return [
                "The rejected attempts you named cover different units. Name rejected attempts that "
                'cover the same units, or use the "translate" form.'
            ], []
        views = ctx.doc.views
        applied = applied_part_edit_keys(ctx.part, ctx.state["finals"], views)
        judged = [
            {**record, "edits": [edit for edit in record.get("edits") or [] if edit_key(edit) not in applied]}
            for record in rejected
        ]
        working = {"finals": dict(ctx.state["finals"])}
        made = set(applied)
        edited_by: List[str] = []
        problems: List[str] = []
        for record in rejected:
            after = f" after the edits of attempt(s) {', '.join(edited_by)}" if edited_by else ""
            pending = {
                **record,
                "edits": [edit for edit in record.get("edits") or [] if edit_key(edit) not in made],
            }
            record_problems = precheck_attempt(
                views,
                working["finals"],
                ctx.part,
                attempt_from_record(pending, views),
                ctx.max_end,
                plan=ctx.doc.plan,
            )
            problems.extend(
                f"Attempt {record['number']} cannot be judged again as it is{after}: {problem}"
                for problem in record_problems
            )
            if record_problems:
                continue
            newly = apply_record_edits(working, views, pending)
            if newly:
                made.update(edit_key(edit) for edit in newly)
                edited_by.append(str(record["number"]))
        return problems, judged

    @staticmethod
    def _edits_not_shown_problem(units: Sequence[int], *, dispute: bool) -> str:
        listed = ", ".join(f"U{unit}" for unit in units)
        if dispute:
            return (
                f"The validator's context window cannot show the part together with {listed}, which the "
                "attempts you named edit, so they could not be judged again. Write a new attempt in the "
                '"translate" form with fewer edits, or with edits only to units closer to the part.'
            )
        return (
            f"The validator's context window cannot show the part together with {listed}, which this "
            "attempt edits, so it could not be judged. Make fewer edits, or edit only units closer to "
            "the part."
        )

    def _apply_decisions(
        self,
        job: _Job,
        ctx: _PartContext,
        records: Sequence[Dict[str, Any]],
        decisions: Sequence[AttemptDecision],
        *,
        round_index: int,
        rejudged: bool,
    ) -> Tuple[List[int], List[Tuple[str, Dict[str, Any], str, str, str]]]:
        """Record validator decisions in order; accepted edits apply at once. Returns (edited, events).

        An accepted record keeps only the edits that applied, so its events
        and later prompts report the edits actually made. An edit identical to
        one this judgment or an accepted attempt of the part already made
        counts as applied once.
        """
        part, state = ctx.part, ctx.state
        edited: List[int] = []
        made = applied_part_edit_keys(part, state["finals"], ctx.doc.views)
        events: List[Tuple[str, Dict[str, Any], str, str, str]] = []
        for record, decision in zip(records, decisions):
            record["reasoning"] = decision.reasoning[:STORED_REASONING_MAX_CHARS]
            if decision.decision == "accept":
                orders = [item["accepted_order"] for item in _accepted_records(part)]
                record["status"] = STATUS_ACCEPTED
                record["feedback"] = ""
                record["accepted_order"] = max(orders, default=0) + 1
                record["edits"] = apply_record_edits(state, ctx.doc.views, record, already_applied=made)
                made.update(edit_key(edit) for edit in record["edits"])
                for edit in record["edits"]:
                    if edit["occurrence"] not in edited:
                        edited.append(edit["occurrence"])
                if not part["locked"]:
                    occurrences = sorted(_as_int(key) for key in record["units"])
                    part["locked"] = True
                    part["start"], part["end"] = occurrences[0], occurrences[-1]
            else:
                record["status"] = STATUS_REJECTED
                record["feedback"] = decision.feedback[:STORED_FEEDBACK_MAX_CHARS]
            stamp_judgment(part, record)
            events.append(self._decision_event(job, ctx, record, round_index=round_index, rejudged=rejudged))
        return edited, events

    async def _record_check_failure(
        self,
        job: _Job,
        ctx: _PartContext,
        number: int,
        attempt: TranslatorAttempt,
        problems: Sequence[str],
    ) -> bool:
        """An attempt that failed the automatic checks: feedback for the translator, never a decision."""
        record = attempt_record(attempt, ctx.doc.views, number=number, status=STATUS_CHECK_FAILED)
        record["feedback"] = " ".join(problems)[:STORED_FEEDBACK_MAX_CHARS]
        stamp_judgment(ctx.part, record)
        ctx.part["keep_streak"] = 0
        ctx.part["attempts"].append(record)
        payload = {
            **self._event_base(job, ctx),
            "round_index": number,
            "candidate_id": f"{ctx.target_hash}:attempt{number}",
            "role_id": TRANSLATOR_ROLE_ID,
            "translator_model": job.translator_model,
            "problems": [_bounded(problem, 300) for problem in list(problems)[:CHECK_PROBLEMS_EVENT_LIMIT]],
            "message": f"Attempt {number} at units {self._units_label(ctx)} failed the automatic checks.",
        }
        return await self._after_attempt(
            job, ctx, [(ATTEMPT_CHECK_FAILED_EVENT, payload, "operational", "workflow", "rejected")]
        )

    async def _after_attempt(
        self,
        job: _Job,
        ctx: _PartContext,
        events: Sequence[Tuple[str, Dict[str, Any], str, str, str]],
        *,
        edited: Sequence[int] = (),
    ) -> bool:
        """Persist the attempt (finalizing the part when its stop rule holds), then broadcast."""
        state, part, views = ctx.state, ctx.part, ctx.doc.views
        prune_attempt_records(part)
        decision = part_stop_decision(part)
        last_number = next_attempt_number(part) - 1
        translations: Dict[str, SegmentTranslation] = {}
        for occurrence in edited:
            # An accepted edit corrects that unit's final translation, so the cache follows it.
            source_hash = views[occurrence].source_hash
            translations[source_hash] = self._cache_item(
                job, source_hash, state["finals"][str(occurrence)], "", rounds=last_number
            )
        final_payload: Optional[Dict[str, Any]] = None
        if decision is not None:
            reason, final_record = decision
            final_payload = {
                **self._event_base(job, ctx),
                "round_index": last_number,
                "final_attempt": final_record["number"],
                "reason": reason,
                "accepted_count": len(_accepted_records(part)),
                "role_id": VALIDATOR_ROLE_ID,
                "translator_model": job.translator_model,
                "validator_model": job.validator_model,
                "message": (
                    f"Units {self._units_label(ctx)} are final with attempt {final_record['number']} "
                    + ("(three accepted attempts)." if reason == FINALIZE_THREE_ACCEPTED else "(kept twice).")
                ),
            }
            for occurrence in finalize_part_in_state(state, views, final_record):
                source_hash = views[occurrence].source_hash
                translations[source_hash] = self._cache_item(
                    job, source_hash, state["finals"][str(occurrence)], final_record["notes"], rounds=last_number
                )
        previous = self._final_counts.get((ctx.doc.doc_hash, job.language_key), 0)
        await self._commit(job, ctx.doc, state, list(translations.values()))
        for name, payload, classification, category, outcome in events:
            self._count_usage(name)
            await self._broadcast(name, payload, classification=classification, category=category, outcome=outcome)
        self._update_job_info(job, ctx.doc, state)
        if final_payload is None:
            return False
        self._usage["finalized_parts"] += 1
        count = len(state["finals"])
        final_payload.update(translated_units=count, total_units=ctx.doc.unit_count)
        await self._broadcast(
            PART_FINALIZED_EVENT, final_payload, classification="informational", category="workflow", outcome="success"
        )
        if count != previous:
            await self._broadcast_progress(job, ctx.doc, count)
        return True

    def _count_usage(self, event_type: str) -> None:
        counter = {
            ATTEMPT_ACCEPTED_EVENT: "accepted_attempts",
            ATTEMPT_REJECTED_EVENT: "rejected_attempts",
            ATTEMPT_CHECK_FAILED_EVENT: "check_failures",
        }.get(event_type)
        if counter:
            self._usage[counter] += 1

    def _cache_item(self, job: _Job, source_hash: str, text: str, notes: str, *, rounds: int) -> SegmentTranslation:
        """A final unit translation for the cross-document cache, in unit-local tokens."""
        settings = self._settings
        return SegmentTranslation(
            source_hash=source_hash,
            language_key=job.language_key,
            translation=text,
            translator_notes=notes,
            submitter_role=TRANSLATOR_ROLE_ID,
            submitter_model=job.translator_model or (settings.translator.model_id if settings.translator else ""),
            validator_model=job.validator_model or (settings.validator.model_id if settings.validator else ""),
            rounds=max(1, rounds),
            created_at=_now_iso(),
        )

    # ------------------------------------------------------------------
    # Translator and validator turns
    # ------------------------------------------------------------------

    @staticmethod
    def _available_input(role: TranslationRoleConfig) -> Optional[int]:
        try:
            return rag_config.get_available_input_tokens(role.context_window, role.max_output_tokens)
        except ValueError:
            return None

    async def _translator_turn(self, job: _Job, ctx: _PartContext) -> _TranslatorTurn:
        role = self._settings.translator
        role_id = TRANSLATOR_ROLE_ID
        available = self._available_input(role)
        if not available or available <= 0:
            await self._enter_context_repair(role_id)
            raise _JobStopped()
        attempts = ctx.part["attempts"]
        ordered = judgment_order(attempts)
        accepted = [record for record in ordered if record["status"] == STATUS_ACCEPTED]
        # Rolling feedback, oldest judgment first: a fresh rejection of a disputed attempt is the newest.
        projected = [record for record in ordered if record["status"] in _FEEDBACK_STATUSES]
        trailing_kept = trailing_kept_records(ctx.part)
        disputes = {record["number"]: disputing_keep(record, attempts) for record in projected}
        start_unit = (ctx.part["start"] if ctx.locked else ctx.cursor) + 1
        end_unit = ctx.max_end + 1
        views = ctx.doc.views

        def build(document: DocumentViews, retained: Tuple[Dict[str, Any], ...]) -> str:
            # A re-judged attempt is shown after the keep that disputed it.
            shown = [*accepted, *trailing_kept, *retained]
            shown.extend(disputes[record["number"]] for record in retained if disputes[record["number"]])
            unique = list({record["number"]: record for record in shown}.values())
            return build_translator_prompt(
                language_descriptor=ctx.descriptor,
                start_unit=start_unit,
                end_unit=end_unit,
                locked=ctx.locked,
                attempts_block=render_attempts_block(unique, views),
                source_view=document.source,
                translation_view=document.translation,
            )

        def fit(limit: int) -> Optional[_FittedPrompt]:
            return _fit_prompt_sync(
                ctx.doc,
                ctx.state["finals"],
                ctx.core,
                (),
                projected,
                TRANSLATOR_FEEDBACK_PROMPT_ENTRIES,
                build,
                limit,
            )

        fitted = await asyncio.to_thread(fit, available)
        if fitted is None:
            await self._enter_context_repair(role_id)
            raise _JobStopped()
        task_id = self._next_task_id("translation_sub")
        while True:
            try:
                result = await self._complete(
                    job,
                    role_id=role_id,
                    role=role,
                    task_id=task_id,
                    messages=[{"role": "user", "content": fitted.prompt}],
                    temperature=0.0,
                )
                break
            except _ContextRejected as rejected:
                available = min(available, int(fitted.tokens * CONTEXT_RETRY_SHRINK))
                refitted = await asyncio.to_thread(fit, available)
                if refitted is None:
                    await self._enter_context_repair(role_id, error=rejected.error)
                    raise _JobStopped() from None
                fitted = refitted
        job.translator_model = result.effective_model
        if result.truncated:
            await self._note_truncation(job, ctx, role_id)
            return _TranslatorTurn(attempt=None, problems=(TRUNCATION_FEEDBACK,), shown=fitted.shown)
        ctx.part["truncations"][role_id] = 0
        try:
            attempt = parse_translator_attempt(result.text)
        except TranslationContractError as exc:
            attempt, problem, _model = await self._repair_json(
                job,
                role_id=role_id,
                role=role,
                prompt=fitted.prompt,
                raw_text=result.text,
                problem=str(exc),
                schema=TRANSLATOR_RESPONSE_FORMS,
                task_id=task_id,
                available=available,
                parse=parse_translator_attempt,
            )
            if attempt is None:
                return _TranslatorTurn(
                    attempt=None,
                    problems=(f"{INVALID_JSON_FEEDBACK} ({problem}). Respond with only the JSON object.",),
                    shown=fitted.shown,
                )
        return _TranslatorTurn(attempt=attempt, problems=(), shown=fitted.shown)

    async def _judge(
        self,
        job: _Job,
        ctx: _PartContext,
        records: Sequence[Dict[str, Any]],
        *,
        round_index: int,
        resubmission_notes: Optional[str] = None,
    ) -> Tuple[AttemptDecision, ...]:
        """Ask the validator for one decision per attempt in ``records``; never a provider-failure decision.

        ``resubmission_notes`` is set for a dispute. The validator always sees
        the units the attempts edit; ``_EditsNotShown`` is raised instead of a
        call when its context cannot show them together with the part.
        """
        role = self._settings.validator
        role_id = VALIDATOR_ROLE_ID
        available = self._available_input(role)
        if not available or available <= 0:
            await self._enter_context_repair(role_id)
            raise _JobStopped()
        views = ctx.doc.views
        numbers = [record["number"] for record in records]
        judged = set(numbers)
        earlier = judgment_order(
            record
            for record in ctx.part["attempts"]
            if record["status"] in _FEEDBACK_STATUSES and record["number"] not in judged
        )
        accepted = [record for record in _accepted_records(ctx.part) if record["number"] not in judged]
        judge_block = render_judge_block(records, views, resubmission_notes=resubmission_notes)
        accepted_block = render_accepted_block(accepted[-1] if accepted else None, views)
        occurrences = sorted({_as_int(key) for record in records for key in record["units"]})
        core = (occurrences[0], occurrences[-1]) if occurrences else ctx.core
        edited = tuple(sorted({edit["occurrence"] for record in records for edit in record["edits"]}))

        def build(document: DocumentViews, retained: Tuple[Dict[str, Any], ...]) -> str:
            return build_validator_prompt(
                language_descriptor=ctx.descriptor,
                attempt_numbers=numbers,
                judge_block=judge_block,
                source_view=document.source,
                translation_view=document.translation,
                accepted_block=accepted_block,
                earlier_feedback=render_earlier_feedback(retained),
            )

        def fit(limit: int, required: Sequence[int]) -> Optional[_FittedPrompt]:
            return _fit_prompt_sync(
                ctx.doc,
                ctx.state["finals"],
                core,
                required,
                earlier,
                VALIDATOR_EARLIER_FEEDBACK_ENTRIES,
                build,
                limit,
            )

        fitted = await self._fit_validator_prompt(fit, available, edited)
        contract_attempt = 0
        while True:
            self._raise_if_stopped(job)
            task_id = self._next_task_id("translation_val")
            try:
                result = await self._complete(
                    job,
                    role_id=role_id,
                    role=role,
                    task_id=task_id,
                    messages=[{"role": "user", "content": fitted.prompt}],
                    temperature=0.0,
                )
            except _ContextRejected as rejected:
                available = min(available, int(fitted.tokens * CONTEXT_RETRY_SHRINK))
                fitted = await self._fit_validator_prompt(fit, available, edited, error=rejected.error)
                continue
            job.validator_model = result.effective_model
            if result.truncated:
                await self._note_truncation(job, ctx, role_id)
                problem = VALIDATOR_TRUNCATION_PROBLEM
            else:
                ctx.part["truncations"][role_id] = 0
                try:
                    return parse_validator_decisions(result.text, numbers)
                except TranslationContractError as exc:
                    decisions, problem, model = await self._repair_json(
                        job,
                        role_id=role_id,
                        role=role,
                        prompt=fitted.prompt,
                        raw_text=result.text,
                        problem=str(exc),
                        schema=VALIDATOR_FORM,
                        task_id=task_id,
                        available=available,
                        parse=lambda text: parse_validator_decisions(text, numbers),
                    )
                    if decisions is not None:
                        job.validator_model = model or job.validator_model
                        return decisions
            contract_attempt += 1
            await self._validator_contract_retry(job, ctx, problem, round_index, contract_attempt, core)

    async def _fit_validator_prompt(
        self,
        fit: Callable[[int, Sequence[int]], Optional[_FittedPrompt]],
        available: int,
        edited: Sequence[int],
        error: Optional[BaseException] = None,
    ) -> _FittedPrompt:
        """Fit the validator prompt showing the part and every unit the attempts edit.

        When only the part fits, raises ``_EditsNotShown`` naming the edited
        units outside the widest window that fits around the part; when even
        the part does not fit, enters context repair.
        """
        fitted = await asyncio.to_thread(fit, available, edited)
        if fitted is not None:
            return fitted
        part_only = await asyncio.to_thread(fit, available, ()) if edited else None
        if part_only is not None:
            shown = part_only.shown or frozenset()
            hidden = [occurrence for occurrence in edited if occurrence not in shown] or list(edited)
            raise _EditsNotShown([occurrence + 1 for occurrence in hidden])
        await self._enter_context_repair(VALIDATOR_ROLE_ID, error=error)
        raise _JobStopped()

    async def _validator_contract_retry(
        self,
        job: _Job,
        ctx: _PartContext,
        problem: str,
        round_index: int,
        attempt: int,
        judged: Tuple[int, int],
    ) -> None:
        """Operational retry: re-validate the same attempts; never a validator decision.

        ``judged`` is the range of the attempts being judged, which the saved
        part reaches only once they are judged, so the event names it.
        """
        await self._broadcast(
            "translation_validator_contract_retry",
            {
                **self._event_base(job, ctx),
                "part_start": judged[0] + 1,
                "part_end": judged[1] + 1,
                "round_index": round_index,
                "attempt": attempt,
                "role_id": VALIDATOR_ROLE_ID,
                "problem": _bounded(problem, 300),
            },
            classification="operational",
            category="workflow",
            outcome="error",
        )
        job.waited = True
        self._note_retry_wait(
            VALIDATOR_ROLE_ID,
            "translation_validator_contract",
            VALIDATOR_CONTRACT_RETRY_SECONDS,
            route=self._non_model_route(VALIDATOR_ROLE_ID),
        )
        await api_client_manager.emit_and_wait_for_retry(
            {
                "provider": "unknown",
                "provider_label": "Translator Validator",
                "role_id": VALIDATOR_ROLE_ID,
                "reason": "translation_validator_contract",
                "retry_after_seconds": VALIDATOR_CONTRACT_RETRY_SECONDS,
                "workflow_mode": WORKFLOW_MODE,
                "retry_layer": "workflow",
                "error_summary": _bounded(problem, 300),
            },
            role_id=VALIDATOR_ROLE_ID,
            should_stop=lambda: self._job_fenced(job),
        )
        self._raise_if_stopped(job)

    async def _repair_json(
        self,
        job: _Job,
        *,
        role_id: str,
        role: TranslationRoleConfig,
        prompt: str,
        raw_text: str,
        problem: str,
        schema: str,
        task_id: str,
        available: int,
        parse: Callable[[str], Any],
    ) -> Tuple[Any, str, str]:
        """One bounded sanitized JSON repair; returns (parsed or None, problem, model)."""
        repair_prompt = build_json_repair_prompt(problem=problem, schema=schema)
        assistant_turn = sanitize_model_output_for_retry_context(raw_text, max_chars=REPAIR_ASSISTANT_MAX_CHARS)
        attempts: List[List[Dict[str, str]]] = []
        if assistant_turn:
            attempts.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_turn},
                    {"role": "user", "content": repair_prompt},
                ]
            )
        attempts.append([{"role": "user", "content": f"{prompt}\n\n{repair_prompt}"}])
        for messages in attempts:
            if await asyncio.to_thread(_messages_tokens, messages) > available:
                continue
            try:
                result = await self._complete(
                    job,
                    role_id=role_id,
                    role=role,
                    task_id=f"{task_id}_repair",
                    messages=messages,
                    temperature=0.0,
                )
            except _ContextRejected:
                continue
            if result.truncated:
                return None, VALIDATOR_TRUNCATION_PROBLEM, result.effective_model
            try:
                return parse(result.text), "", result.effective_model
            except TranslationContractError as exc:
                return None, str(exc), result.effective_model
        return None, problem, ""

    # ------------------------------------------------------------------
    # Model calls and provider recovery
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        *,
        task_id: str,
        role_id: str,
        role: TranslationRoleConfig,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> _CallResult:
        with background_service_call(SERVICE_NAME):
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=role_id,
                model=role.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=role.max_output_tokens,
                _moto_disable_supercharge=True,
                _moto_force_boost_mode="__none__",
            )
        self._record_usage(response)
        choices = response.get("choices") if isinstance(response, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        metadata = api_client_manager.extract_call_metadata(response)
        return _CallResult(
            text=extract_response_text(response, context="auto_top_p_translate"),
            truncated=choice_hit_output_limit(first_choice),
            effective_model=str(metadata.get("effective_model") or role.model_id),
        )

    def _record_usage(self, response: Any) -> None:
        self._usage["calls"] += 1
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, Mapping):
            self._usage["prompt_tokens"] += _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            self._usage["completion_tokens"] += _as_int(
                usage.get("completion_tokens") or usage.get("output_tokens")
            )

    @staticmethod
    def _is_context_rejection(error: BaseException) -> bool:
        return isinstance(error, ProviderContextLengthError) or (
            isinstance(error, Exception) and is_provider_context_length_error(error)
        )

    async def _complete(
        self,
        job: _Job,
        *,
        role_id: str,
        role: TranslationRoleConfig,
        task_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> _CallResult:
        """Call the model, recovering from provider failures until it succeeds or the job stops."""
        while True:
            self._raise_if_stopped(job)
            reset_generation = get_provider_reset_generation()
            try:
                return await self._invoke(
                    task_id=task_id,
                    role_id=role_id,
                    role=role,
                    messages=messages,
                    temperature=temperature,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._is_context_rejection(error):
                    raise _ContextRejected(error) from None
                await self._recover_from_call_error(job, role_id, error, reset_generation)

    async def _recover_from_call_error(
        self,
        job: _Job,
        role_id: str,
        error: Exception,
        reset_generation: int,
    ) -> None:
        if isinstance(error, RetryableProviderError):
            job.waited = True
            cooldown = isinstance(error, ProviderCooldownError)
            if cooldown:
                # The cooldown seam waits in chunks until the usage window resets.
                retry_after = error.retry_after_seconds
                self._note_retry_wait(
                    role_id,
                    str(error.reason or "retryable_provider_error"),
                    retry_after if isinstance(retry_after, int) and not isinstance(retry_after, bool) else None,
                    route=api_client_manager.route_identity_for_role(
                        role_id, workflow_mode=WORKFLOW_MODE, error=error
                    ),
                    detail=error.error_summary,
                )
            await api_client_manager.wait_for_retryable_provider_error(
                error,
                role_id=role_id,
                should_stop=lambda: self._job_fenced(job),
                activity_callback=None if cooldown else self._record_backoff_wait,
                workflow_mode=WORKFLOW_MODE,
            )
        elif (
            isinstance(error, ProviderRepairRequiredError) and error.reason == "credit_exhaustion"
        ) or is_provider_credit_pause_error(error):
            await self._wait_for_credit_reset(job, role_id, reset_generation, error)
        elif isinstance(error, FreeModelExhaustedError):
            await self._wait_after_free_models_exhausted(job, role_id, error)
        elif isinstance(error, ProviderRepairRequiredError) or is_non_retryable_model_error(error):
            await self._enter_provider_repair(role_id, error)
            raise _JobStopped()
        else:
            await self._wait_after_unexpected_error(job, role_id, error)
        self._raise_if_stopped(job)

    async def _wait_for_credit_reset(
        self, job: _Job, role_id: str, reset_generation: int, error: Exception
    ) -> None:
        job.waited = True
        self._credit_waiters += 1
        self._touch()
        route = api_client_manager.route_identity_for_role(role_id, workflow_mode=WORKFLOW_MODE, error=error)
        self._append_activity(
            "translation_waiting_for_credits",
            self._activity_payload(
                "translation_waiting_for_credits",
                {
                    **route,
                    "role_id": role_id,
                    "retry_attempt": self._next_wait_note(),
                    "message": with_route_identity("Waiting for provider credits to be reset.", route),
                },
                classification="operational",
                category="provider",
                outcome="info",
            ),
        )
        await self._publish_status_if_changed()
        try:
            await wait_for_provider_reset(reset_generation, lambda: self._job_fenced(job))
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            raise _JobStopped() from None
        finally:
            self._credit_waiters -= 1
            self._touch()
            self._wake_worker()

    async def _wait_after_free_models_exhausted(self, job: _Job, role_id: str, error: Exception) -> None:
        from backend.aggregator.agents.submitter import free_model_exhaustion_route_identity

        job.waited = True
        route = free_model_exhaustion_route_identity(role_id, error=error, workflow_mode=WORKFLOW_MODE)
        self._note_retry_wait(role_id, FREE_MODELS_EXHAUSTED_REASON, FREE_MODEL_RETRY_SECONDS, route=route)
        await api_client_manager.emit_and_wait_for_retry(
            {
                "provider": "openrouter",
                "provider_label": "OpenRouter free models",
                **route,
                "role_id": role_id,
                "retry_after_seconds": FREE_MODEL_RETRY_SECONDS,
                "reason": FREE_MODELS_EXHAUSTED_REASON,
                "workflow_mode": WORKFLOW_MODE,
                "retry_layer": "workflow",
            },
            role_id=role_id,
            should_stop=lambda: self._job_fenced(job),
            error=error,
        )

    async def _wait_after_unexpected_error(self, job: _Job, role_id: str, error: Exception) -> None:
        job.waited = True
        detail = _bounded(f"{type(error).__name__}: {error}", 500)
        self._note_retry_wait(
            role_id,
            "translation_loop_retry",
            LOOP_RETRY_SECONDS,
            route=self._non_model_route(role_id, error),
            detail=detail,
        )
        await api_client_manager.emit_and_wait_for_retry(
            {
                "provider": "unknown",
                "provider_label": DISPLAY_NAME,
                "role_id": role_id,
                "reason": "translation_loop_retry",
                "retry_after_seconds": LOOP_RETRY_SECONDS,
                "workflow_mode": WORKFLOW_MODE,
                "retry_layer": "workflow",
                "error_summary": detail,
            },
            role_id=role_id,
            should_stop=lambda: self._job_fenced(job),
            error=error,
        )
        self._raise_if_stopped(job)

    # ------------------------------------------------------------------
    # Repair state
    # ------------------------------------------------------------------

    async def _enter_provider_repair(self, role_id: str, error: Exception) -> None:
        if isinstance(error, ProviderRepairRequiredError):
            reason = error.reason
            message = error.safe_message
            guidance = error.terminal_guidance
        else:
            reason = "provider_configuration"
            message = _bounded(str(error), 500)
            guidance = ""
        await self._enter_repair(
            reason=reason or "provider_repair_required",
            message=message or f"{role_label(role_id)} cannot continue without a settings or provider fix.",
            guidance=guidance
            or (
                f"Check the provider, model, and credentials for {role_label(role_id)} in "
                f"{DISPLAY_NAME} settings, then press Retry."
            ),
            role_id=role_id,
            error=error,
        )

    async def _enter_context_repair(self, role_id: str, error: Optional[BaseException] = None) -> None:
        label = role_label(role_id)
        await self._enter_repair(
            reason="context_window_too_small",
            message=f"The context window of {label} is too small to show the part being translated.",
            guidance=(
                f"Raise the context window or lower Max Output Tokens for {label} in "
                f"{DISPLAY_NAME} settings, then press Retry."
            ),
            role_id=role_id,
            error=error,
        )

    async def _enter_output_budget_repair(self) -> None:
        label = role_label(TRANSLATOR_ROLE_ID)
        await self._enter_repair(
            reason="max_output_tokens_too_small",
            message=f"The max output tokens of {label} are too small to translate the next paragraph.",
            guidance=f"Raise Max Output Tokens for {label} in {DISPLAY_NAME} settings, then press Retry.",
            role_id=TRANSLATOR_ROLE_ID,
        )

    async def _note_truncation(self, job: _Job, ctx: _PartContext, role_id: str) -> None:
        """Count and save an output truncation by ``role_id`` on the part.

        The second in a row enters repair. Its count is reset before it is
        saved, so after Retry or a settings change the role starts afresh.
        """
        truncations = ctx.part["truncations"]
        truncations[role_id] = truncations.get(role_id, 0) + 1
        repair = truncations[role_id] >= TRUNCATION_REPAIR_THRESHOLD
        if repair:
            truncations[role_id] = 0
        await self._commit(job, ctx.doc, ctx.state)
        if repair:
            await self._enter_truncation_repair(role_id)
            raise _JobStopped()

    async def _enter_truncation_repair(self, role_id: str) -> None:
        label = role_label(role_id)
        await self._enter_repair(
            reason="output_truncated",
            message=f"{label} hit its max output token limit twice in a row on the same part.",
            guidance=f"Raise Max Output Tokens for {label}, then press Retry.",
            role_id=role_id,
        )

    async def _enter_repair(
        self,
        *,
        reason: str,
        message: str,
        guidance: str,
        role_id: str,
        error: Optional[BaseException] = None,
    ) -> None:
        if self._repair is not None or self._stopping:
            return
        base = {
            "reason": _bounded(reason, 160),
            "message": _bounded(message, 1000),
            "guidance": _bounded(guidance, 1000),
            "role_id": role_id,
        }
        identity = api_client_manager.route_identity_for_role(role_id, workflow_mode=WORKFLOW_MODE, error=error)
        self._repair = {**base, "route_identity": str(identity.get("route_identity") or "")}
        self._generation += 1
        self._work_epoch += 1
        # Fence every job; cancel the others now. The job raising this repair
        # stops itself after the broadcast below.
        current_job = _CURRENT_JOB.get()
        for job in list(self._jobs.values()):
            job.cancelled = True
            if job is not current_job and job.task is not None and job.task is not asyncio.current_task():
                job.task.cancel()
        self._touch()
        self._wake_worker()
        # Shielded so a cancellation cannot drop the repair notice.
        await asyncio.shield(
            self._broadcast(
                "translation_repair_required",
                {**base, **identity, "message": with_route_identity(base["message"], identity)},
                classification="operational",
                category="provider",
                outcome="error",
            )
        )

    # ------------------------------------------------------------------
    # Store writes
    # ------------------------------------------------------------------

    @staticmethod
    async def _settled_thread(
        function: Callable[..., Any],
        *args: Any,
        on_success: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Run blocking work in a thread and let it finish even if the caller is cancelled.

        Callers hold the write lock across this call, so a cancelled caller
        never releases the lock while its write is still running.
        ``on_success`` runs on the event loop once the work succeeded, even
        when the caller was cancelled meanwhile.
        """
        future = asyncio.ensure_future(asyncio.to_thread(function, *args))
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await future
            if on_success is not None and future.done() and not future.cancelled() and future.exception() is None:
                on_success()
            raise
        if on_success is not None:
            on_success()
        return result

    async def _commit(
        self,
        job: _Job,
        doc: _Doc,
        state: Dict[str, Any],
        translations: Sequence[SegmentTranslation] = (),
    ) -> None:
        """Persist the job state and new final unit translations; a fenced job saves nothing.

        The in-memory final count follows a write that completed, even when the
        job was cancelled while the write ran.
        """
        items = list(translations)
        async with self._write_lock:
            if self._job_fenced(job):
                raise _JobStopped()

            def applied() -> None:
                self._final_counts[(doc.doc_hash, job.language_key)] = len(state["finals"])
                self._translated.setdefault(job.language_key, set()).update(item.source_hash for item in items)
                self._touch()

            await self._settled_thread(
                _commit_job_sync, doc.doc_hash, job.language_key, state, items, on_success=applied
            )

    async def _broadcast_progress(self, job: _Job, doc: _Doc, count: int) -> None:
        await self._broadcast(
            PROGRESS_EVENT,
            {
                "language_key": job.language_key,
                "doc_hashes": [doc.doc_hash],
                "translated_units": count,
                "total_units": doc.unit_count,
            },
            classification="informational",
            category="workflow",
            outcome="info",
        )

    # ------------------------------------------------------------------
    # Activity and broadcasts
    # ------------------------------------------------------------------

    @staticmethod
    def _units_label(ctx: _PartContext) -> str:
        start, end = ctx.part["start"] + 1, ctx.part["end"] + 1
        return f"U{start}" if start == end else f"U{start}–U{end}"

    def _event_base(self, job: _Job, ctx: _PartContext) -> Dict[str, Any]:
        return {
            "language_key": job.language_key,
            "language_name": ctx.language.name,
            "doc_hash": ctx.doc.doc_hash,
            "doc_title": job.doc_title,
            "kind": ctx.doc.primary.kind if ctx.doc.keys else "",
            "target_hash": ctx.target_hash,
            "part_start": ctx.part["start"] + 1,
            "part_end": ctx.part["end"] + 1,
        }

    def _decision_event(
        self,
        job: _Job,
        ctx: _PartContext,
        record: Mapping[str, Any],
        *,
        round_index: int,
        rejudged: bool,
    ) -> Tuple[str, Dict[str, Any], str, str, str]:
        number = record["number"]
        candidate_id = f"{ctx.target_hash}:attempt{number}"
        if rejudged:
            candidate_id = f"{candidate_id}:rejudge{round_index}"
        payload = {
            **self._event_base(job, ctx),
            "round_index": round_index,
            "candidate_id": candidate_id,
            "attempt_number": number,
            "rejudged": rejudged,
            "role_id": TRANSLATOR_ROLE_ID,
            "translator_model": job.translator_model,
            "validator_model": job.validator_model,
            "has_translator_notes": bool(record["notes"]),
            "edit_count": len(record["edits"]),
            "accepted_count": len(_accepted_records(ctx.part)),
        }
        if record["status"] == STATUS_ACCEPTED:
            payload["reasoning"] = _bounded(record["reasoning"], REASONING_EVENT_CHARS)
            return ATTEMPT_ACCEPTED_EVENT, payload, "semantic", "validator", "success"
        payload["feedback"] = _bounded(record["feedback"], FEEDBACK_EVENT_CHARS)
        return ATTEMPT_REJECTED_EVENT, payload, "semantic", "validator", "rejected"

    def _kept_event(
        self, job: _Job, ctx: _PartContext, record: Mapping[str, Any], *, dispute: bool
    ) -> Tuple[str, Dict[str, Any], str, str, str]:
        number = record["number"]
        kept = ", ".join(str(item) for item in record["keep_attempts"])
        message = (
            f"Translator Submitter asked the validator to judge attempt(s) {kept} again."
            if dispute
            else f"Translator Submitter kept attempt(s) {kept} for units {self._units_label(ctx)}."
        )
        payload = {
            **self._event_base(job, ctx),
            "round_index": number,
            "candidate_id": f"{ctx.target_hash}:attempt{number}",
            "role_id": TRANSLATOR_ROLE_ID,
            "translator_model": job.translator_model,
            "keep_attempts": list(record["keep_attempts"]),
            "keep_streak": ctx.part["keep_streak"],
            "dispute": dispute,
            "has_translator_notes": bool(record["notes"]),
            "message": message,
        }
        return ATTEMPT_KEPT_EVENT, payload, "informational", "workflow", "info"

    def _append_activity(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Keep a bounded copy of a normalized activity payload for ``recent_activity``.

        The copy keeps the payload's ``activity_id`` and ``activity_message``, so
        a client can match it to the live event it already received.
        """
        if event_type not in RECENT_ACTIVITY_EVENTS:
            return
        entry: Dict[str, Any] = {"event_type": event_type, "timestamp": _now_iso()}
        activity_id = payload.get("activity_id")
        if isinstance(activity_id, str) and activity_id:
            entry["activity_id"] = activity_id
        entry["activity_message"] = str(payload.get("activity_message") or "")
        for key in _ACTIVITY_TEXT_KEYS:
            value = payload.get(key)
            if value not in (None, ""):
                entry[key] = _bounded(value, _ACTIVITY_TEXT_CHARS)
        if event_type in _ROUTED_ACTIVITY_EVENTS:
            for key in (*ROUTE_IDENTITY_FIELD_NAMES, "message", "error_summary"):
                value = payload.get(key)
                if value not in (None, ""):
                    limit = _ERROR_DETAIL_CHARS if key == "error_summary" else _ROUTED_MESSAGE_CHARS
                    entry[key] = _bounded(value, limit)
        if event_type == TRANSLATE_ALL_EVENT and payload.get("message"):
            entry["message"] = _bounded(payload["message"], _TRANSLATE_ALL_MESSAGE_CHARS)
        for key in _ACTIVITY_NUMBER_KEYS:
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry[key] = value
        problems = payload.get("problems")
        if isinstance(problems, list):
            entry["problems"] = [_bounded(problem, 200) for problem in problems[:CHECK_PROBLEMS_EVENT_LIMIT]]
        keep_attempts = payload.get("keep_attempts")
        if isinstance(keep_attempts, list):
            entry["keep_attempts"] = [_as_int(item) for item in keep_attempts[:10]]
        self._activity.append(entry)

    @staticmethod
    def _non_model_route(role_id: str, error: Optional[BaseException] = None) -> Dict[str, str]:
        """Route identity of a wait no provider classified: the route ``error`` carries, else mode and role."""
        return api_client_manager.route_identity_for_role(
            role_id, workflow_mode=WORKFLOW_MODE, provider="unknown", error=error
        )

    def _next_wait_note(self) -> int:
        """Distinguish repeated waits so each keeps its own activity identity."""
        self._wait_note_seq += 1
        return self._wait_note_seq

    async def _record_backoff_wait(self, event: str, payload: Dict[str, Any]) -> None:
        """Record a wait of the shared backoff seam as it starts, from the seam's own payload."""
        if event != "waiting":
            return
        seconds, retry = payload.get("retry_after_seconds"), payload.get("retry_attempt")
        self._note_retry_wait(
            str(payload.get("role_id") or ""),
            str(payload.get("reason") or "retryable_provider_error"),
            seconds if isinstance(seconds, int) and not isinstance(seconds, bool) else None,
            route={key: str(payload[key]) for key in ROUTE_IDENTITY_FIELD_NAMES if payload.get(key)},
            detail=str(payload.get("error_summary") or ""),
            retry=retry if isinstance(retry, int) and not isinstance(retry, bool) else None,
        )

    def _note_retry_wait(
        self,
        role_id: str,
        reason: str,
        seconds: Optional[int],
        *,
        route: Mapping[str, str],
        detail: str = "",
        retry: Optional[int] = None,
    ) -> None:
        """Record a retry wait the service starts in ``recent_activity``.

        The live retry event comes from the shared retry seam; this entry gives
        the Progress tab a readable line after a status refresh. ``route`` is
        the route identity of the call that failed (workflow mode and role
        only for a non-model wait), shown once at the start of the line;
        ``detail`` is the bounded error text and ``retry`` the seam's retry
        count, when known.
        """
        cause = _RETRY_REASON_TEXT.get(reason) or reason.replace("_", " ")
        timing = f"retrying in {seconds}s" if seconds is not None else "retrying after the provider cooldown"
        count = f" (retry {retry})" if retry else ""
        payload: Dict[str, Any] = {
            **route,
            "role_id": role_id,
            "reason": reason,
            "retry_attempt": self._next_wait_note(),
            "message": with_route_identity(f"{cause[:1].upper()}{cause[1:]}; {timing}{count}.", route),
        }
        if seconds is not None:
            payload["retry_after_seconds"] = seconds
        if detail:
            payload["error_summary"] = _bounded(detail, _ERROR_DETAIL_CHARS)
        self._append_activity(
            PROVIDER_RETRY_EVENT,
            self._activity_payload(
                PROVIDER_RETRY_EVENT,
                payload,
                classification="operational",
                category="provider",
                outcome="info",
            ),
        )

    def _activity_payload(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        classification: str,
        category: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """Fill translation activity metadata and normalize it as the websocket does."""
        enriched = fill_missing_activity_metadata(
            payload,
            workflow=WORKFLOW_MODE,
            generation=self._generation,
            source_type="translation",
            source_id=str(payload.get("doc_hash") or ""),
            role_id=str(payload.get("role_id") or ""),
            classification=classification,
            category=category,
            outcome=outcome,
            feedback=payload.get("feedback") or "",
            message=payload.get("message") or "",
            guidance=payload.get("guidance") or "",
        )
        return normalize_activity_payload(event_type, enriched)

    async def _broadcast(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        classification: str,
        category: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """Record and broadcast one bounded activity event; never raises.

        The payload is normalized once here, so the ``recent_activity`` copy and
        the live event share one ``activity_id``.
        """
        normalized = self._activity_payload(
            event_type,
            payload,
            classification=classification,
            category=category,
            outcome=outcome,
        )
        self._append_activity(event_type, normalized)
        try:
            from backend.api.routes.websocket import broadcast_event

            await broadcast_event(event_type, normalized)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Auto Top-p Translate broadcast failed: %s", type(exc).__name__)
        return normalized

    async def _publish_status_if_changed(self) -> None:
        self._note_queue_change()
        core = self._status_core()
        translate_all = self._translate_all_status(core["stats"])
        key = (
            core["state"],
            self._settings.enabled,
            self._settings.paused,
            self._generation,
            translate_all["requested"],
            translate_all["state"],
        )
        if key == self._last_status_key:
            return
        self._last_status_key = key
        await self._broadcast(
            "translation_status_changed",
            {
                "state": core["state"],
                "status_label": core["status_label"],
                "enabled": self._settings.enabled,
                "paused": self._settings.paused,
                "pending_units": core["pending"],
                "generation": self._generation,
            },
            classification="informational",
            category="workflow",
            outcome="info",
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _stats(self) -> _Stats:
        key = (self._stats_version, int(time.time() // STATS_TIME_BUCKET_SECONDS))
        if self._stats_cache is not None and self._stats_cache[0] == key:
            return self._stats_cache[1]
        stats = self._compute_stats()
        self._stats_cache = (key, stats)
        return stats

    def _compute_stats(self) -> _Stats:
        """Progress per language and for Translate all, and the queue.

        A language's translated and pending counts cover the documents it
        translates at ``TIER_CURRENT`` (current runs and documents being
        viewed); its background counts cover Translate all's documents.
        """
        now = time.time()
        active_keys = set(self._active_language_keys())
        counts = {
            language.key: dict.fromkeys(
                ("translated", "pending", "pending_docs", "complete_docs", "background", "background_docs"), 0
            )
            for language in self._settings.languages
        }
        translate_all = dict.fromkeys(("documents", "complete_documents", "translated_units", "pending_units"), 0)
        for doc in self._docs.values():
            translate_all_only = self._translate_all_only(doc)
            doc_pending = 0
            for language_key, language_counts in counts.items():
                done = min(self._final_counts.get((doc.doc_hash, language_key), 0), doc.unit_count)
                remaining = doc.unit_count - done
                if translate_all_only and language_key in active_keys:
                    translate_all["translated_units"] += done
                    translate_all["pending_units"] += remaining
                    doc_pending += remaining
                tier = self._doc_tier(doc, language_key, now)
                if tier == TIER_CURRENT:
                    language_counts["translated"] += done
                    language_counts["pending"] += remaining
                    language_counts["pending_docs" if remaining else "complete_docs"] += 1
                elif tier == TIER_BACKGROUND and remaining:
                    language_counts["background"] += remaining
                    language_counts["background_docs"] += 1
            if translate_all_only:
                translate_all["documents"] += 1
                if not doc_pending:
                    translate_all["complete_documents"] += 1
        languages = tuple(
            {
                "key": language.key,
                "name": language.name,
                "custom": language.custom,
                "translated_units": counts[language.key]["translated"],
                "pending_units": counts[language.key]["pending"],
                "pending_documents": counts[language.key]["pending_docs"],
                "complete_documents": counts[language.key]["complete_docs"],
                "background_pending_units": counts[language.key]["background"],
                "background_pending_documents": counts[language.key]["background_docs"],
            }
            for language in self._settings.languages
        )
        positions: Dict[Tuple[str, str], int] = {key: 0 for key in self._jobs}
        ordered = sorted(self._jobs, key=lambda key: (self._docs[key[0]].order if key[0] in self._docs else 0, key))
        for index, key in enumerate(self._waiting_jobs(now), start=1):
            positions[key] = index
            ordered.append(key)
        queue_rows = []
        for doc_hash, language_key in ordered[:STATUS_QUEUE_LIMIT]:
            doc = self._docs.get(doc_hash)
            if doc is None:
                continue
            primary = doc.primary
            tier = self._doc_tier(doc, language_key, now)
            queue_rows.append(
                {
                    "doc_hash": doc_hash,
                    "doc_key": primary.doc_key,
                    "kind": primary.kind,
                    "title": primary.title,
                    "language_key": language_key,
                    "position": positions[(doc_hash, language_key)],
                    "viewing": tier == TIER_CURRENT and self._viewing(doc, now),
                    "background": tier == TIER_BACKGROUND,
                    "translated_units": min(self._final_counts.get((doc_hash, language_key), 0), doc.unit_count),
                    "total_units": doc.unit_count,
                }
            )
        background_pending = sum(counts[key]["background"] for key in counts if key in active_keys)
        return _Stats(
            languages=languages,
            queue=tuple(queue_rows),
            positions=positions,
            pending_total=background_pending + sum(counts[key]["pending"] for key in counts if key in active_keys),
            background_pending=background_pending,
            current_translated=sum(counts[key]["translated"] for key in counts if key in active_keys),
            translate_all=translate_all,
            background_running=any(
                not job.cancelled
                and job.doc_hash in self._docs
                and self._doc_tier(self._docs[job.doc_hash], job.language_key, now) == TIER_BACKGROUND
                for job in self._jobs.values()
            ),
        )

    def _status_core(self) -> Dict[str, Any]:
        settings = self._settings
        stats = self._stats()
        ready, reason = settings_ready(settings)
        pending = stats.pending_total
        if not settings.enabled:
            state = "disabled"
            message = f"{DISPLAY_NAME} is off."
        elif not ready:
            state = "not_configured"
            message = reason
        elif self._repair is not None:
            state = "repair_required"
            message = str(self._repair.get("message") or "")
        elif settings.paused:
            state = "paused"
            message = "Translation is paused."
        elif self._credit_waiters > 0:
            state = "waiting_for_credits"
            message = "Waiting for provider credits to be reset. Use Retry OpenRouter after adding credits."
        elif self._jobs or pending:
            state = "translating"
            current = pending - stats.background_pending
            done = progress_percent(stats.current_translated, stats.current_translated + current)
            if not stats.background_pending:
                message = f"Translating: {done}% done."
            elif current:
                message = (
                    f"Translating: {done}% of current runs done; "
                    f"Translate all: {_older_items_left(stats)} older item(s) left."
                )
            else:
                message = f"Translate all: {_older_items_left(stats)} older item(s) left in the background."
        else:
            state = "idle"
            message = "All selected outputs are translated."
        return {
            "state": state,
            "status_label": STATUS_LABELS[state],
            "message": message,
            "pending": pending,
            "stats": stats,
        }

    def _translate_all_blocked(self) -> str:
        """Why no translation job can run now, worded for Translate all, or "" when one can."""
        settings = self._settings
        ready, reason = settings_ready(settings)
        if not ready:
            return f"Translate all waits until translation is set up: {reason}"
        if self._repair is not None:
            return "Translate all waits until the translation repair is resolved."
        if settings.paused:
            return "Translation is paused."
        if self._credit_waiters > 0:
            return "Translate all waits for provider credits to be reset."
        if self._clearing:
            return "Translate all waits while stored translations are cleared."
        if not self._can_dispatch():
            return "Translate all waits until translation can run again."
        return ""

    def _background_waits_for_slots(self) -> bool:
        """Translate all has no lane (one job at a time), and viewed or current-run work takes the job."""
        if self._background_lane():
            return False
        if len(self._jobs) >= self._slot_limit:
            return True
        # A free normal slot goes to a waiting viewed or current-run job first.
        return any(self._foreground(rank) for rank, _key in self._ranked_waiting())

    def _translate_all_status(self, stats: _Stats) -> Dict[str, Any]:
        """The status ``translate_all`` block: whether it is on, its state, and its progress.

        "preparing" means older outputs are still being listed and none is
        left to translate yet. "waiting" means Translate all is blocked, and
        the message says why: translation is paused, needs setup or repair, or
        waits for credits, or the translator connection runs one job at a time
        and current-run or viewed work takes it. Once an enumeration has
        finished, the state follows the remaining work, even when a source
        failed.
        """
        settings = self._settings
        progress = stats.translate_all
        pending = progress["pending_units"]
        blocked = self._translate_all_blocked() if settings.translate_all else ""
        if not settings.translate_all:
            state, message = TRANSLATE_ALL_OFF, ""
        elif blocked:
            state, message = TRANSLATE_ALL_WAITING, blocked
        elif pending and not stats.background_running and self._background_waits_for_slots():
            state = TRANSLATE_ALL_WAITING
            message = (
                "Translate all waits for current-run and viewed translations: "
                "the translator connection runs one job at a time."
            )
        elif pending:
            state = TRANSLATE_ALL_TRANSLATING
            message = f"Translate all: {_older_items_left(stats)} older item(s) left in the background."
        elif self._backlog_busy() or not self._backlog_enumerated:
            state, message = TRANSLATE_ALL_PREPARING, "Finding older items to translate."
        else:
            state, message = TRANSLATE_ALL_TRANSLATING, "Translate all is finishing."
        return {"requested": settings.translate_all, "state": state, **progress, "message": message}

    def status_snapshot(self) -> dict:
        """Fast in-memory status for ``GET /api/translation/status``."""
        core = self._status_core()
        stats: _Stats = core["stats"]
        repair = None
        if self._repair is not None:
            repair = {
                "reason": self._repair.get("reason", ""),
                "message": self._repair.get("message", ""),
                "guidance": self._repair.get("guidance", ""),
                "role_id": self._repair.get("role_id", ""),
                "route_identity": self._repair.get("route_identity", ""),
            }
        return {
            "success": True,
            "enabled": self._settings.enabled,
            "paused": self._settings.paused,
            "batch_translation": self._settings.batch_translation,
            "modes": list(self._settings.modes),
            "max_parallel_jobs": self._slot_limit,
            "state": core["state"],
            "status_label": core["status_label"],
            "message": core["message"],
            "generation": self._generation,
            "languages": [dict(row) for row in stats.languages],
            "queue": [dict(row) for row in stats.queue],
            "active": [
                {
                    "doc_hash": job.doc_hash,
                    "doc_title": job.doc_title,
                    "language_key": job.language_key,
                    "part_start": job.part_start,
                    "part_end": job.part_end,
                    "attempt": job.attempt,
                    "accepted_count": job.accepted_count,
                    "keep_streak": job.keep_streak,
                }
                for job in self._jobs.values()
            ],
            "repair": repair,
            "translate_all": self._translate_all_status(stats),
            "usage": dict(self._usage),
            "recent_activity": [dict(item) for item in reversed(self._activity)],
        }

    def connectivity_summary(self) -> dict:
        """Fast in-memory summary for ``GET /api/connectivity/status`` skills."""
        core = self._status_core()
        return {
            "status": core["status_label"],
            "enabled": self._settings.enabled,
            "active": core["state"] == "translating",
            "message": core["message"],
            "languages": [language.name for language in self._settings.languages],
            "pending_units": core["pending"],
        }

    # ------------------------------------------------------------------
    # Language catalog and proficiency probe
    # ------------------------------------------------------------------

    @staticmethod
    def _model_set_key(settings: AutoTranslateSettings) -> Tuple[Any, ...]:
        return tuple(
            (role.provider, role.model_id) if role is not None else None
            for role in (settings.validator, settings.translator)
        )

    def _probe_usability(self) -> Tuple[bool, bool]:
        probe = self._probe
        if probe is None:
            return False, False
        stale = probe["model_set_key"] != self._model_set_key(self._settings)
        usable = (
            not stale
            and probe.get("validator_names") is not None
            and probe.get("translator_names") is not None
        )
        return usable, stale

    async def language_catalog(self) -> dict:
        """Catalog languages with probe-based proficiency for ``GET /languages``."""
        usable, stale = self._probe_usability()
        proficient: Set[str] = set()
        extras: List[str] = []
        if usable:
            # A language is proficient when both the translator and the validator list it.
            proficient, extras = proficiency_from_probe(
                self._probe["validator_names"],
                [self._probe["translator_names"]],
            )
        languages = [
            {
                "key": entry.key,
                "name": entry.name,
                "native_name": entry.native_name,
                "aliases": list(entry.aliases),
                "pinned": entry.pinned,
                "proficiency": (
                    "proficient" if entry.key in proficient else "uncertain" if usable else "unknown"
                ),
            }
            for entry in STANDARD_LANGUAGES
        ]
        probe = self._probe
        return {
            "success": True,
            "pinned_keys": list(PINNED_LANGUAGE_KEYS),
            "languages": languages,
            "extras": [
                {"key": custom_language_key(name), "name": name, "proficiency": "proficient", "reported": True}
                for name in extras
            ],
            "probe": {
                "status": probe["status"] if probe else "never",
                "probed_at": probe["probed_at"] if probe else None,
                "models": [dict(model) for model in probe["models"]] if probe else [],
                "stale": stale,
            },
        }

    async def probe_languages(self) -> dict:
        """Ask each distinct configured model which languages it can translate into."""
        self._bind_loop()
        settings = self._settings
        rows: List[Tuple[str, TranslationRoleConfig]] = [
            (label, role)
            for label, role in (("validator", settings.validator), ("translator", settings.translator))
            if role is not None
        ]
        if not rows:
            raise ValueError("Configure translator models before checking language proficiency.")
        distinct: Dict[Tuple[str, str], Tuple[str, TranslationRoleConfig]] = {}
        for _label, role in rows:
            model_key = (role.provider, role.model_id)
            if model_key not in distinct:
                distinct[model_key] = (f"{PROBE_ROLE_PREFIX}{len(distinct) + 1}", role)
        for probe_role_id, role in distinct.values():
            self._configure_role(probe_role_id, role)
        results = await asyncio.gather(
            *(self._probe_model(probe_role_id, role) for probe_role_id, role in distinct.values()),
            return_exceptions=True,
        )
        names_by_model: Dict[Tuple[str, str], List[str]] = {}
        errors_by_model: Dict[Tuple[str, str], str] = {}
        for model_key, result in zip(distinct, results):
            if isinstance(result, BaseException):
                # Returned to the client, so redact credential shapes from provider error text.
                errors_by_model[model_key] = _bounded(redact_log_text(f"{type(result).__name__}: {result}"), 300)
            else:
                names_by_model[model_key] = result
        models = [
            {
                "role": label,
                "provider": role.provider,
                "model_id": role.model_id,
                "ok": (role.provider, role.model_id) in names_by_model,
                "error": errors_by_model.get((role.provider, role.model_id), ""),
            }
            for label, role in rows
        ]

        def names(role: Optional[TranslationRoleConfig]) -> Optional[List[str]]:
            return names_by_model.get((role.provider, role.model_id)) if role is not None else None

        validator_names = names(settings.validator)
        translator_names = names(settings.translator)
        usable = validator_names is not None and translator_names is not None
        status = "complete" if all(model["ok"] for model in models) and usable else "partial" if usable else "failed"
        self._probe = {
            "status": status,
            "probed_at": _now_iso(),
            "models": models,
            # The settings snapshot the probe asked, not settings saved during it.
            "model_set_key": self._model_set_key(settings),
            "validator_names": validator_names,
            "translator_names": translator_names,
        }
        return await self.language_catalog()

    async def _probe_model(self, role_id: str, role: TranslationRoleConfig) -> List[str]:
        prompt = build_language_probe_prompt()
        task_id = self._next_task_id("translation_probe")
        result = await self._invoke(
            task_id=task_id,
            role_id=role_id,
            role=role,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        try:
            return parse_probe_response(result.text)
        except TranslationContractError as exc:
            messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
            assistant_turn = sanitize_model_output_for_retry_context(
                result.text, max_chars=REPAIR_ASSISTANT_MAX_CHARS
            )
            if assistant_turn:
                messages.append({"role": "assistant", "content": assistant_turn})
            messages.append(
                {"role": "user", "content": build_json_repair_prompt(problem=str(exc), schema=PROBE_JSON_SCHEMA)}
            )
            repaired = await self._invoke(
                task_id=f"{task_id}_repair",
                role_id=role_id,
                role=role,
                messages=messages,
                temperature=0.0,
            )
            return parse_probe_response(repaired.text)


translation_service = TranslationService()
