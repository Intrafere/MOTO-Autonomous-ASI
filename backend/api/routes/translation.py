"""Auto Top-p Translate REST routes (``/api/translation/*``).

Auto Top-p Translate is a background service, not a top-level workflow: these
routes never take a workflow start lease and work while any workflow runs.
Status, settings, and language-catalog reads return in-memory service state
only. Request validation errors return 422 with bounded field locations and
messages, never the submitted values; unexpected failures return a generic
detail that never echoes exception text. Shared middleware authenticates every
endpoint.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from backend.shared.log_redaction import redact_log_text
from backend.shared.runtime_settings import RuntimeSettingsError
from backend.shared.translation.discovery import TITLE_MAX_CHARS
from backend.shared.translation.service import DISPLAY_NAME, STATUS_QUEUE_LIMIT, translation_service
from backend.shared.translation.settings import AutoTranslateSettings, TranslationMode

logger = logging.getLogger(__name__)

MAX_DOCUMENTS_PER_REQUEST = 200
MAX_PARTS_PER_DOCUMENT = 64
MAX_TOTAL_TEXT_CHARS = 4_000_000
MAX_DOC_KEY_CHARS = 300
MAX_PART_NAME_CHARS = 64
MAX_LOOKUP_HASHES = 500
LANGUAGE_KEY_MIN_CHARS = 2
LANGUAGE_KEY_MAX_CHARS = 40
_ERROR_MESSAGE_CHARS = 300
_ERROR_LOCATION_PART_CHARS = 64
_ERROR_LIMIT = 20

DocumentKind = Literal["paper", "final_answer", "proof", "brainstorm", "aggregator_results", "live_paper", "other"]
DocumentProfile = Literal["document", "fields"]
ControlAction = Literal["enable", "disable", "pause", "resume", "retry", "translate_all", "cancel_translate_all"]
TranslateAllState = Literal["off", "preparing", "translating", "waiting"]
TranslationState = Literal[
    "disabled",
    "not_configured",
    "idle",
    "translating",
    "paused",
    "waiting_for_credits",
    "repair_required",
]
TranslationStatusLabel = Literal[
    "disabled",
    "not ready",
    "ready",
    "translating",
    "paused",
    "waiting",
    "repair required",
]
DocumentTranslationStatus = Literal["complete", "pending", "empty", "unavailable", "disabled", "unknown"]
LanguageProgressStatus = Literal["complete", "pending", "unknown", "disabled"]
LanguageKey = Annotated[
    str,
    StringConstraints(min_length=LANGUAGE_KEY_MIN_CHARS, max_length=LANGUAGE_KEY_MAX_CHARS),
]
DocHash = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[0-9a-f]+$")]

def _safe_error_location(location: Any) -> List[Any]:
    parts: List[Any] = []
    for item in location if isinstance(location, (list, tuple)) else (location,):
        if isinstance(item, int) and not isinstance(item, bool):
            parts.append(item)
        else:
            parts.append(redact_log_text(item, _ERROR_LOCATION_PART_CHARS))
    return parts


def _safe_error_message(message: Any) -> str:
    text = str(message or "")
    if text.startswith("Value error, "):
        text = text[len("Value error, "):]
    return redact_log_text(text, _ERROR_MESSAGE_CHARS)


def safe_validation_errors(errors: Any) -> List[Dict[str, Any]]:
    """Return validation errors as bounded ``{loc, msg, type}`` items without submitted values."""
    items: List[Dict[str, Any]] = []
    for error in list(errors or ())[:_ERROR_LIMIT]:
        if not isinstance(error, dict):
            continue
        items.append(
            {
                "loc": _safe_error_location(error.get("loc", ())),
                "msg": _safe_error_message(error.get("msg")),
                "type": redact_log_text(error.get("type") or "value_error", 80),
            }
        )
    return items or [{"loc": [], "msg": "Invalid request.", "type": "value_error"}]


class _SafeValidationRoute(APIRoute):
    """Return request validation errors without echoing submitted values.

    FastAPI's default 422 body repeats each invalid input, which could return
    megabytes of document text; this keeps only field locations and messages.
    """

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                return JSONResponse(status_code=422, content={"detail": safe_validation_errors(exc.errors())})

        return safe_handler


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(
    prefix="/api/translation",
    tags=["translation"],
    route_class=_SafeValidationRoute,
    dependencies=[Depends(_no_store)],
)


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------


class TranslationControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ControlAction


class TranslationDocumentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=MAX_PART_NAME_CHARS)
    text: str = Field(max_length=MAX_TOTAL_TEXT_CHARS)


class TranslationDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_key: str = Field(min_length=1, max_length=MAX_DOC_KEY_CHARS)
    kind: DocumentKind = "other"
    title: str = Field(default="", max_length=TITLE_MAX_CHARS)
    profile: DocumentProfile = "document"
    parts: List[TranslationDocumentPart] = Field(default_factory=list, max_length=MAX_PARTS_PER_DOCUMENT)

    @field_validator("doc_key")
    @classmethod
    def _printable_key(cls, value: str) -> str:
        if not value.strip() or not value.isprintable():
            raise ValueError("doc_key must contain printable characters only.")
        return value

    @model_validator(mode="after")
    def _unique_part_names(self) -> "TranslationDocumentInput":
        names = [part.name for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("Document part names must be unique.")
        return self


class TranslationDocumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Optional[LanguageKey] = None
    viewing: bool = True
    request: bool = Field(
        default=False,
        description=(
            "True when the user asked for these translations: the request is saved, and the documents translate "
            "in the language (every selected language when language is null) until done, viewed or not."
        ),
    )
    documents: List[TranslationDocumentInput] = Field(min_length=1, max_length=MAX_DOCUMENTS_PER_REQUEST)

    @model_validator(mode="after")
    def _bounded_total_text(self) -> "TranslationDocumentsRequest":
        _check_total_text(self.documents)
        return self


class TranslationProgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: List[TranslationDocumentInput] = Field(min_length=1, max_length=MAX_DOCUMENTS_PER_REQUEST)

    @model_validator(mode="after")
    def _bounded_total_text(self) -> "TranslationProgressRequest":
        _check_total_text(self.documents)
        return self


def _check_total_text(documents: List[TranslationDocumentInput]) -> None:
    total = sum(len(part.text) for document in documents for part in document.parts)
    if total > MAX_TOTAL_TEXT_CHARS:
        raise ValueError(f"Documents may contain at most {MAX_TOTAL_TEXT_CHARS:,} characters of text in total.")


class TranslationLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: LanguageKey
    doc_hashes: List[DocHash] = Field(min_length=1, max_length=MAX_LOOKUP_HASHES)
    viewing: bool = Field(
        default=False,
        description="True while a viewer still shows these documents, so they keep their viewing priority.",
    )


# ----------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------


class TranslationLanguageProgress(BaseModel):
    """Progress of one language; the background counts are Translate all's older outputs."""

    key: str
    name: str
    custom: bool
    translated_units: int
    pending_units: int
    pending_documents: int
    complete_documents: int
    background_pending_units: int
    background_pending_documents: int


class TranslationQueueItem(BaseModel):
    """One document x language job; running jobs come first at position 0."""

    doc_hash: str
    doc_key: str
    kind: str
    title: str
    language_key: str
    position: int = Field(ge=0, description="0 while translating now; n for the n-th job in line.")
    viewing: bool
    background: bool = Field(description="True when only Translate all targets the document.")
    translated_units: int
    total_units: int


class TranslationActiveItem(BaseModel):
    """A running job and the part it is translating (1-based unit numbers)."""

    doc_hash: str
    doc_title: str
    language_key: str
    part_start: int
    part_end: int
    attempt: int
    accepted_count: int
    keep_streak: int


class TranslationRepairState(BaseModel):
    reason: str
    message: str
    guidance: str
    role_id: str
    route_identity: str


class TranslationTranslateAllStatus(BaseModel):
    """Translate all: older displayed outputs translating in the background."""

    requested: bool
    state: TranslateAllState = Field(
        description=(
            "preparing while older outputs are listed and none is left to translate; waiting while "
            "translation cannot run or the translator connection has no job slot left beside "
            "current-run and viewed work, with the reason in message."
        )
    )
    documents: int
    complete_documents: int
    translated_units: int
    pending_units: int
    message: str


class TranslationUsage(BaseModel):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    accepted_attempts: int
    rejected_attempts: int
    finalized_parts: int
    check_failures: int


class TranslationStatusResponse(BaseModel):
    success: bool = True
    enabled: bool
    paused: bool
    batch_translation: bool
    modes: List[TranslationMode] = Field(description="Saved modes whose current runs translate automatically.")
    max_parallel_jobs: int = Field(ge=1)
    state: TranslationState
    status_label: TranslationStatusLabel
    message: str
    generation: int
    languages: List[TranslationLanguageProgress]
    queue: List[TranslationQueueItem] = Field(max_length=STATUS_QUEUE_LIMIT)
    active: List[TranslationActiveItem]
    repair: Optional[TranslationRepairState] = None
    translate_all: TranslationTranslateAllStatus
    usage: TranslationUsage
    recent_activity: List[Dict[str, Any]]


class TranslationSettingsResponse(BaseModel):
    success: bool = True
    settings: AutoTranslateSettings


class TranslationSettingsUpdateResponse(BaseModel):
    success: bool = True
    settings: AutoTranslateSettings
    status: TranslationStatusResponse


class TranslationStatusEnvelope(BaseModel):
    success: bool = True
    status: TranslationStatusResponse


class TranslationCatalogLanguage(BaseModel):
    key: str
    name: str
    native_name: str
    aliases: List[str]
    pinned: bool
    proficiency: Literal["proficient", "uncertain", "unknown"]


class TranslationReportedLanguage(BaseModel):
    key: str
    name: str
    proficiency: Literal["proficient"] = "proficient"
    reported: bool = True


class TranslationProbeModel(BaseModel):
    role: Literal["translator", "validator"]
    provider: str
    model_id: str
    ok: bool
    error: str


class TranslationProbeState(BaseModel):
    status: Literal["never", "complete", "partial", "failed"]
    probed_at: Optional[str] = None
    models: List[TranslationProbeModel]
    stale: bool


class TranslationLanguageCatalogResponse(BaseModel):
    success: bool = True
    pinned_keys: List[str]
    languages: List[TranslationCatalogLanguage]
    extras: List[TranslationReportedLanguage]
    probe: TranslationProbeState


class TranslationDocumentStatus(BaseModel):
    doc_key: str
    doc_hash: str
    language: str
    status: DocumentTranslationStatus
    translated_units: int
    total_units: int
    queue_position: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "0 while translating now; n for the n-th job in line; null when complete, "
            "empty, unavailable, disabled, or unknown."
        ),
    )
    parts: Optional[Dict[str, str]] = None
    requested: bool = Field(default=False, description="A user requested this document in this language.")
    automatic: bool = Field(
        default=False,
        description="A checked mode's current run includes this document, so it translates automatically.",
    )


class TranslationDocumentsResponse(BaseModel):
    success: bool = True
    documents: List[TranslationDocumentStatus]


class TranslationDocumentLanguageProgress(BaseModel):
    language: str
    status: LanguageProgressStatus = Field(
        description="unknown: the service has not seen this content; disabled: translation is off."
    )
    translated_units: int
    total_units: int


class TranslationDocumentProgress(BaseModel):
    doc_key: str
    doc_hash: str
    languages: List[TranslationDocumentLanguageProgress] = Field(
        description="One entry per selected language, in the user's language order."
    )


class TranslationDocumentProgressResponse(BaseModel):
    success: bool = True
    documents: List[TranslationDocumentProgress]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _unprocessable(exc: ValueError) -> HTTPException:
    """422 for invalid input; pydantic errors keep locations and messages only."""
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=422, detail=safe_validation_errors(exc.errors()))
    return HTTPException(status_code=422, detail=_safe_error_message(exc) or "Invalid request.")


def _server_error(action: str, exc: BaseException) -> HTTPException:
    logger.warning(
        "Auto Top-p Translate route failed to %s: %s: %s",
        action,
        type(exc).__name__,
        redact_log_text(exc, 240),
    )
    return HTTPException(status_code=500, detail=f"{DISPLAY_NAME} could not {action}.")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@router.get("/status", response_model=TranslationStatusResponse)
async def get_translation_status() -> Dict[str, Any]:
    """In-memory service status; never touches disk or models."""
    return translation_service.status_snapshot()


@router.get("/settings", response_model=TranslationSettingsResponse)
async def get_translation_settings() -> Dict[str, Any]:
    return {"success": True, "settings": translation_service.get_settings()}


@router.put("/settings", response_model=TranslationSettingsUpdateResponse)
async def put_translation_settings(settings: AutoTranslateSettings) -> Dict[str, Any]:
    """Normalize, persist, and apply settings (hosted mode accepts OpenRouter roles only).

    ``enabled``, ``paused``, and ``translate_all`` in the body are ignored: they
    keep the service's current values and change only through
    ``POST /control``, so a save that races an Enable, Pause, or Translate all
    never reverts it. An omitted ``modes`` keeps the saved modes.
    """
    try:
        status = await translation_service.update_settings(settings)
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except RuntimeSettingsError as exc:
        raise _server_error("save its settings", exc) from None
    except Exception as exc:
        raise _server_error("apply its settings", exc) from None
    return {"success": True, "settings": translation_service.get_settings(), "status": status}


@router.post("/control", response_model=TranslationStatusEnvelope)
async def post_translation_control(request: TranslationControlRequest) -> Dict[str, Any]:
    """Apply a control action; ``translate_all`` is 422 while translation is off."""
    try:
        status = await translation_service.control(request.action)
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except RuntimeSettingsError as exc:
        raise _server_error("save its settings", exc) from None
    except Exception as exc:
        raise _server_error("apply that action", exc) from None
    return {"success": True, "status": status}


@router.get("/languages", response_model=TranslationLanguageCatalogResponse)
async def get_translation_languages() -> Dict[str, Any]:
    """Language catalog with cached probe proficiency; never calls models."""
    return await translation_service.language_catalog()


@router.post("/languages/probe", response_model=TranslationLanguageCatalogResponse)
async def post_translation_language_probe() -> Dict[str, Any]:
    """Ask the translator submitter and translator validator models which languages they can translate into."""
    try:
        return await translation_service.probe_languages()
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except Exception as exc:
        raise _server_error("check translator model languages", exc) from None


@router.post("/documents", response_model=TranslationDocumentsResponse)
async def post_translation_documents(request: TranslationDocumentsRequest) -> Dict[str, Any]:
    """Register viewer documents and return their status; ``request`` saves a user's translation request."""
    documents = [document.model_dump() for document in request.documents]
    try:
        rows = await translation_service.register_documents(
            request.language, request.viewing, documents, request=request.request
        )
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except Exception as exc:
        raise _server_error("register those documents", exc) from None
    return {"success": True, "documents": rows}


@router.post("/documents/lookup", response_model=TranslationDocumentsResponse)
async def post_translation_documents_lookup(request: TranslationLookupRequest) -> Dict[str, Any]:
    """Return the status of registered documents; ``viewing`` refreshes their viewing priority."""
    try:
        rows = await translation_service.lookup_documents(
            request.language, list(request.doc_hashes), viewing=request.viewing
        )
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except Exception as exc:
        raise _server_error("look up those documents", exc) from None
    return {"success": True, "documents": rows}


@router.post("/documents/progress", response_model=TranslationDocumentProgressResponse)
async def post_translation_documents_progress(request: TranslationProgressRequest) -> Dict[str, Any]:
    """Each document's progress in every selected language; never registers, views, or requests it."""
    documents = [document.model_dump() for document in request.documents]
    try:
        rows = await translation_service.document_progress(documents)
    except ValueError as exc:
        raise _unprocessable(exc) from None
    except Exception as exc:
        raise _server_error("read translation progress for those documents", exc) from None
    return {"success": True, "documents": rows}


@router.delete("/cache", response_model=TranslationStatusEnvelope)
async def delete_translation_cache(
    confirm: bool = Query(False, description="Must be true to clear every stored translation."),
) -> Dict[str, Any]:
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to clear every stored translation.")
    try:
        status = await translation_service.clear_cache()
    except Exception as exc:
        raise _server_error("clear stored translations", exc) from None
    return {"success": True, "status": status}
