"""Discover the current runs' user-readable outputs for Auto Top-p Translate.

Each scan reads the current run of every checked mode: the active autonomous
session's complete papers, completed final answer, proofs, and complete
brainstorms; the manual Aggregator's accepted submissions and proofs; the
manual Compiler's Live Paper and proofs; and the loaded Proof Solver run's
proofs. A manual mode's run is current from Start until Clear, which its saved
prompt marks. Specs are built exactly as the viewers that show them build
theirs, so a viewer and discovery share one document identity, and every spec
carries the source that produced it. Discovery is read-only: it never creates
stores or changes which files a memory module points at. Every source is
guarded on its own, so a failing source is skipped and logged and the other
sources still return their documents.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Collection, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.translation.segmentation import (
    PROFILE_DOCUMENT,
    PROFILE_FIELDS,
    PROFILES,
    PROOF_TRANSLATION_FIELDS,
)

logger = logging.getLogger(__name__)

DOCUMENT_KINDS = (
    "paper",
    "final_answer",
    "proof",
    "brainstorm",
    "aggregator_results",
    "live_paper",
    "other",
)
TITLE_MAX_CHARS = 300

MODE_AUTONOMOUS = "autonomous"
MODE_AGGREGATOR = "aggregator"
MODE_COMPILER = "compiler"
MODE_LEANOJ = "leanoj"
MANUAL_MODES = frozenset({MODE_AGGREGATOR, MODE_COMPILER})

# Current-run discovery sources. A source that fails during a scan keeps its
# previously discovered documents.
SOURCE_AUTONOMOUS_PAPERS = "autonomous_papers"
SOURCE_AUTONOMOUS_FINAL_ANSWER = "autonomous_final_answer"
SOURCE_AUTONOMOUS_PROOFS = "autonomous_proofs"
SOURCE_AUTONOMOUS_BRAINSTORMS = "autonomous_brainstorms"
SOURCE_AGGREGATOR_RESULTS = "aggregator_results"
SOURCE_AGGREGATOR_PROOFS = "aggregator_proofs"
SOURCE_COMPILER_LIVE_PAPER = "compiler_live_paper"
SOURCE_COMPILER_PROOFS = "compiler_proofs"
SOURCE_LEANOJ_PROOFS = "leanoj_proofs"

AGGREGATOR_RESULTS_DOC_KEY = "aggregator:results"
COMPILER_LIVE_PAPER_DOC_KEY = "compiler:live-paper"
MANUAL_AGGREGATOR_FILE_NAME = "rag_shared_training.txt"

# Manual-store proofs name their mode through their source: Aggregator inline
# proofs, Aggregator "Search For More", and Single Paper Writer proofs-only runs
# check the Aggregator database; Compiler live-paper, saved-paper, and rigor
# proofs use the manual Compiler source ids.
MANUAL_AGGREGATOR_PROOF_SOURCE_ID = "manual_aggregator"
MANUAL_COMPILER_PROOF_SOURCE_PREFIXES = ("manual_compiler_", "compiler_manual_")
# Proof Solver runs also store their proofs in the autonomous proof store.
LEANOJ_SOURCE_TYPE_PREFIX = "leanoj_"

# Prose fields the Proof Solver proof view translates; a statement that repeats
# the run prompt is never shown there, so it is left out.
LEANOJ_TRANSLATION_FIELDS = ("theorem_statement", "novelty_reasoning", "novelty_group_claim")

# ECMAScript WhiteSpace and LineTerminator characters: what
# ``String.prototype.trim`` removes.
JS_TRIM_CHARACTERS = (
    "\t\n\v\f\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


@dataclass(frozen=True)
class DocumentSpec:
    """One translatable document: stable key, kind, title, profile, and named parts.

    ``source`` names the discovery source that produced the spec; it is not
    part of the document's identity.
    """

    doc_key: str
    kind: str
    title: str
    profile: str
    parts: Tuple[Tuple[str, str], ...]
    source: str = ""


@dataclass(frozen=True)
class DiscoveryResult:
    """A scan's documents and the sources that failed during it."""

    documents: Tuple[DocumentSpec, ...]
    failed_sources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CurrentRunState:
    """The current run of each mode.

    ``autonomous_session`` is the active autonomous session, or "" when none
    is active and the legacy roots are current. A manual mode's run is live
    from Start until Clear, which its saved prompt marks. ``compiler_writing``
    is true while a manual or autonomous compiler may be writing the shared
    compiler paper. ``leanoj_session`` is the loaded Proof Solver run, or "".
    """

    autonomous_session: str
    aggregator_live: bool
    compiler_live: bool
    compiler_writing: bool
    leanoj_session: str

    @property
    def autonomous_run(self) -> str:
        return self.autonomous_session or "legacy"


def js_trim(text: str) -> str:
    """``text`` trimmed as JavaScript's ``String.prototype.trim`` trims it.

    Viewers drop parts whose trimmed text is empty, and Python's ``str.strip``
    removes a different set of characters, so blank checks use this.
    """
    return text.strip(JS_TRIM_CHARACTERS)


def bounded_title(value: Any, fallback: str = "") -> str:
    """Return a one-line title of at most ``TITLE_MAX_CHARS`` characters."""
    text = " ".join(str(value or "").split())
    return (text or fallback)[:TITLE_MAX_CHARS]


def canonical_parts(parts: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    """Return document parts sorted by part name.

    The document hash covers part order, so viewers and discovery must list the
    same parts in the same order to share one document identity.
    """
    return tuple(sorted(parts, key=lambda part: part[0]))


def make_document_spec(
    *,
    doc_key: str,
    kind: str,
    title: str,
    profile: str,
    parts: Iterable[Tuple[str, str]],
) -> DocumentSpec:
    """Build a validated ``DocumentSpec`` with parts sorted by name.

    Raises ValueError for invalid fields.
    """
    key = str(doc_key or "").strip()
    if not key or len(key) > 300 or not key.isprintable():
        raise ValueError("Document key must be 1-300 printable characters.")
    if kind not in DOCUMENT_KINDS:
        raise ValueError("Unknown document kind.")
    if profile not in PROFILES:
        raise ValueError("Unknown document profile.")
    normalized_parts: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for name, text in parts:
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("Document part names and texts must be strings.")
        if not name or len(name) > 64:
            raise ValueError("Document part names must be 1-64 characters.")
        if name in seen:
            raise ValueError("Document part names must be unique.")
        seen.add(name)
        normalized_parts.append((name, text))
    return DocumentSpec(
        doc_key=key,
        kind=kind,
        title=bounded_title(title),
        profile=profile,
        parts=canonical_parts(normalized_parts),
    )


async def _guarded(
    source: str,
    scan: Callable[[], Awaitable[List[DocumentSpec]]],
    failed: List[str],
) -> List[DocumentSpec]:
    try:
        return await scan()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failed.append(source)
        logger.warning(
            "Auto Top-p Translate skipped output source %s: %s: %s",
            source,
            type(exc).__name__,
            redact_log_text(exc, 240),
        )
        return []


async def collect_specs(
    scans: Iterable[Tuple[str, Callable[[], Awaitable[List[DocumentSpec]]]]],
) -> DiscoveryResult:
    """Run each (source, scan) in order; specs are tagged with their source.

    The first spec for a doc key wins. A failing source is skipped and named
    in ``failed_sources``.
    """
    failed: List[str] = []
    documents: List[DocumentSpec] = []
    seen_keys: set[str] = set()
    for source, scan in scans:
        for spec in await _guarded(source, scan, failed):
            if spec.doc_key in seen_keys:
                continue
            seen_keys.add(spec.doc_key)
            documents.append(replace(spec, source=source))
    return DiscoveryResult(documents=tuple(documents), failed_sources=tuple(failed))


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _record_text(record: Any, name: str) -> str:
    value = _record_value(record, name)
    return value if isinstance(value, str) else ""


def _active_session_id() -> str:
    from backend.autonomous.memory.session_manager import session_manager

    if session_manager.is_session_active:
        return str(session_manager.session_id or "").strip()
    return ""


def manual_aggregator_results_path() -> Path:
    """The canonical manual Aggregator database under the active data root.

    Autonomous runs temporarily repoint ``system_config.shared_training_file``
    at their own databases, so the manual file is resolved from the data root.
    """
    return Path(system_config.data_dir) / MANUAL_AGGREGATOR_FILE_NAME


def _read_text_if_present(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


async def current_run_state() -> CurrentRunState:
    """Read which run of each mode is current, without changing any store."""
    from backend.aggregator.memory.shared_training import get_manual_aggregator_prompt_path
    from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
    from backend.compiler.core.compiler_coordinator import compiler_coordinator
    from backend.compiler.memory.manual_prompt import get_manual_compiler_prompt_path
    from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator

    aggregator_path = get_manual_aggregator_prompt_path()
    compiler_path = get_manual_compiler_prompt_path()
    aggregator_prompt, compiler_prompt = await asyncio.to_thread(
        lambda: (_read_text_if_present(aggregator_path), _read_text_if_present(compiler_path))
    )
    return CurrentRunState(
        autonomous_session=_active_session_id(),
        aggregator_live=bool(aggregator_prompt.strip()),
        compiler_live=bool(compiler_prompt.strip()),
        compiler_writing=bool(compiler_coordinator.is_running or autonomous_coordinator.is_active),
        leanoj_session=str(leanoj_coordinator.get_state().session_id or "").strip(),
    )


def manual_proof_mode(record: Any) -> str:
    """The manual mode a manual-store proof belongs to, or "" when its source names none."""
    source_type = str(_record_value(record, "source_type") or "").strip()
    source_id = str(_record_value(record, "source_id") or "").strip()
    if source_type == "brainstorm" and source_id == MANUAL_AGGREGATOR_PROOF_SOURCE_ID:
        return MODE_AGGREGATOR
    if source_type == "paper" and source_id.startswith(MANUAL_COMPILER_PROOF_SOURCE_PREFIXES):
        return MODE_COMPILER
    return ""


def manual_proof_is_live(record: Any, state: CurrentRunState) -> bool:
    """Whether a manual-store proof belongs to a manual mode's current run."""
    mode = manual_proof_mode(record)
    return (mode == MODE_AGGREGATOR and state.aggregator_live) or (
        mode == MODE_COMPILER and state.compiler_live
    )


def autonomous_proof_is_live(record: Any, state: CurrentRunState) -> bool:
    """Whether an autonomous-store proof belongs to the active autonomous session.

    Proof Solver proofs stored there belong to Proof Solver runs, and the
    legacy proof store outlives autonomous Clear, so neither is current.
    """
    source_type = str(_record_value(record, "source_type") or "")
    return bool(state.autonomous_session) and not source_type.startswith(LEANOJ_SOURCE_TYPE_PREFIX)


def _proof_title(record: Any) -> str:
    name = str(_record_value(record, "theorem_name") or "").strip()
    if name:
        return bounded_title(name)
    statement = _record_text(record, "theorem_statement").strip()
    first_line = statement.splitlines()[0] if statement else ""
    return bounded_title(first_line, str(_record_value(record, "proof_id") or "Proof"))


def proof_spec(
    record: Any,
    doc_key: str,
    fields: Sequence[str] = PROOF_TRANSLATION_FIELDS,
) -> Optional[DocumentSpec]:
    """A PROFILE_FIELDS spec of a proof's non-blank prose ``fields``, or None when none has text."""
    parts = [
        (field, text)
        for field in fields
        for text in (_record_text(record, field),)
        if js_trim(text)
    ]
    if not parts:
        return None
    return make_document_spec(
        doc_key=doc_key,
        kind="proof",
        title=_proof_title(record),
        profile=PROFILE_FIELDS,
        parts=parts,
    )


def proof_document_specs(records: Sequence[Any], scope: str) -> List[DocumentSpec]:
    """Build ``proof:{scope}:{run}:{proof_id}`` specs from proof records' prose fields."""
    specs: List[DocumentSpec] = []
    for record in records:
        proof_id = str(_record_value(record, "proof_id") or "").strip()
        if not proof_id:
            continue
        run_id = str(_record_value(record, "run_id") or "").strip() or "current"
        spec = proof_spec(record, f"proof:{scope}:{run_id}:{proof_id}")
        if spec is not None:
            specs.append(spec)
    return specs


def _leanoj_statement_is_prompt(proof: Mapping[str, Any]) -> bool:
    prompt = js_trim(str(proof.get("user_prompt") or ""))
    return bool(prompt) and js_trim(str(proof.get("theorem_statement") or "")) == prompt


def leanoj_proof_specs(
    proofs: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    hide_prompt_statement: bool,
) -> List[DocumentSpec]:
    """Build ``proof:leanoj:{library_id}`` specs from Proof Solver proof views.

    With ``hide_prompt_statement``, a statement that repeats the run prompt is
    left out, as the current-run proof view leaves it out.
    """
    specs: List[DocumentSpec] = []
    for proof in proofs:
        key_id = str(proof.get("library_id") or proof.get("proof_id") or "")
        if not key_id:
            continue
        shown = fields
        if hide_prompt_statement and _leanoj_statement_is_prompt(proof):
            shown = [field for field in fields if field != "theorem_statement"]
        spec = proof_spec(proof, f"proof:leanoj:{key_id}", shown)
        if spec is not None:
            specs.append(spec)
    return specs


def aggregator_results_specs(content: str) -> List[DocumentSpec]:
    """The Aggregator Live Results document, when the database has text."""
    if not js_trim(content):
        return []
    return [
        make_document_spec(
            doc_key=AGGREGATOR_RESULTS_DOC_KEY,
            kind="aggregator_results",
            title="Aggregator accepted submissions",
            profile=PROFILE_DOCUMENT,
            parts=(("content", content),),
        )
    ]


async def read_manual_aggregator_results() -> str:
    """The manual Aggregator database text, as the Live Results view shows it."""
    return await asyncio.to_thread(_read_text_if_present, manual_aggregator_results_path())


def compiler_live_paper_specs(content: str) -> List[DocumentSpec]:
    """The Compiler Live Paper document, when it has text."""
    if not js_trim(content):
        return []
    return [
        make_document_spec(
            doc_key=COMPILER_LIVE_PAPER_DOC_KEY,
            kind="live_paper",
            title="Compiler Paper",
            profile=PROFILE_DOCUMENT,
            parts=(("content", content),),
        )
    ]


async def read_compiler_live_paper() -> str:
    """The Live Paper text exactly as the compiler paper route shows it."""
    from backend.compiler.memory.outline_memory import outline_memory
    from backend.compiler.memory.paper_memory import format_live_paper_display, paper_memory

    outline = await outline_memory.get_outline()
    paper = await paper_memory.get_paper()
    return format_live_paper_display(outline, paper)


async def _scan_autonomous_papers(state: CurrentRunState) -> List[DocumentSpec]:
    from backend.autonomous.memory.paper_library import paper_library

    specs: List[DocumentSpec] = []
    for metadata in await paper_library.get_all_papers():
        content = await paper_library.get_paper_content(metadata.paper_id)
        if not js_trim(content):
            continue
        specs.append(
            make_document_spec(
                doc_key=f"paper:{state.autonomous_run}:{metadata.paper_id}",
                kind="paper",
                title=bounded_title(metadata.title, metadata.paper_id),
                profile=PROFILE_DOCUMENT,
                parts=(("content", content),),
            )
        )
    return specs


async def _scan_autonomous_final_answer(state: CurrentRunState) -> List[DocumentSpec]:
    from backend.autonomous.memory.final_answer_memory import final_answer_memory
    from backend.autonomous.memory.paper_library import paper_library

    if not state.autonomous_session:
        return []
    answer = final_answer_memory.get_state()
    if answer.is_active or answer.status != "complete":
        return []
    content = ""
    title = ""
    if answer.answer_format == "short_form" and answer.short_form_paper_id:
        content = await paper_library.get_paper_content(answer.short_form_paper_id)
        metadata = await paper_library.get_metadata(answer.short_form_paper_id)
        title = metadata.title if metadata else ""
    elif answer.answer_format == "long_form":
        content = await final_answer_memory.get_final_volume()
        volume = answer.volume_organization
        title = volume.volume_title if volume else ""
    if not js_trim(content):
        return []
    return [
        make_document_spec(
            doc_key=f"final_answer:{state.autonomous_session}",
            kind="final_answer",
            title=bounded_title(title, "Final Answer"),
            profile=PROFILE_DOCUMENT,
            parts=(("content", content),),
        )
    ]


async def _scan_autonomous_proofs(state: CurrentRunState) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import proof_database

    records = await proof_database.get_all_proofs()
    return proof_document_specs(
        [record for record in records if autonomous_proof_is_live(record, state)], "autonomous"
    )


async def _scan_autonomous_brainstorms(state: CurrentRunState) -> List[DocumentSpec]:
    from backend.autonomous.memory.brainstorm_memory import brainstorm_memory

    specs: List[DocumentSpec] = []
    for metadata in await brainstorm_memory.get_all_brainstorms():
        if metadata.status != "complete":
            continue
        content = await brainstorm_memory.get_database_content(metadata.topic_id)
        if not js_trim(content):
            continue
        specs.append(
            make_document_spec(
                doc_key=f"brainstorm:current:{metadata.topic_id}",
                kind="brainstorm",
                title=bounded_title(metadata.topic_prompt, metadata.topic_id),
                profile=PROFILE_DOCUMENT,
                parts=(("content", content),),
            )
        )
    return specs


async def _scan_aggregator_results(state: CurrentRunState) -> List[DocumentSpec]:
    if not state.aggregator_live:
        return []
    return aggregator_results_specs(await read_manual_aggregator_results())


async def _scan_manual_proofs(mode: str) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import manual_proof_database

    records = await manual_proof_database.get_all_proofs()
    return proof_document_specs([record for record in records if manual_proof_mode(record) == mode], "manual")


async def _scan_aggregator_proofs(state: CurrentRunState) -> List[DocumentSpec]:
    if not state.aggregator_live:
        return []
    return await _scan_manual_proofs(MODE_AGGREGATOR)


async def _scan_compiler_live_paper(state: CurrentRunState) -> List[DocumentSpec]:
    if not state.compiler_live or state.compiler_writing:
        return []
    return compiler_live_paper_specs(await read_compiler_live_paper())


async def _scan_compiler_proofs(state: CurrentRunState) -> List[DocumentSpec]:
    if not state.compiler_live:
        return []
    return await _scan_manual_proofs(MODE_COMPILER)


async def _scan_leanoj_proofs(state: CurrentRunState) -> List[DocumentSpec]:
    from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
    from backend.leanoj.core.leanoj_proof_views import _extract_leanoj_proofs

    if not state.leanoj_session:
        return []
    return leanoj_proof_specs(
        _extract_leanoj_proofs(leanoj_coordinator.get_status()),
        LEANOJ_TRANSLATION_FIELDS,
        hide_prompt_statement=True,
    )


# (mode, source, scanner name); scanners are looked up by name at scan time.
_LIVE_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    (MODE_AUTONOMOUS, SOURCE_AUTONOMOUS_PAPERS, "_scan_autonomous_papers"),
    (MODE_AUTONOMOUS, SOURCE_AUTONOMOUS_FINAL_ANSWER, "_scan_autonomous_final_answer"),
    (MODE_AUTONOMOUS, SOURCE_AUTONOMOUS_PROOFS, "_scan_autonomous_proofs"),
    (MODE_AUTONOMOUS, SOURCE_AUTONOMOUS_BRAINSTORMS, "_scan_autonomous_brainstorms"),
    (MODE_AGGREGATOR, SOURCE_AGGREGATOR_RESULTS, "_scan_aggregator_results"),
    (MODE_AGGREGATOR, SOURCE_AGGREGATOR_PROOFS, "_scan_aggregator_proofs"),
    (MODE_COMPILER, SOURCE_COMPILER_LIVE_PAPER, "_scan_compiler_live_paper"),
    (MODE_COMPILER, SOURCE_COMPILER_PROOFS, "_scan_compiler_proofs"),
    (MODE_LEANOJ, SOURCE_LEANOJ_PROOFS, "_scan_leanoj_proofs"),
)


async def discover_current_outputs(modes: Collection[str]) -> DiscoveryResult:
    """Scan the current run of every mode in ``modes``.

    A failing source is skipped and named in ``failed_sources``; when the
    current runs cannot be read, every checked source counts as failed.
    """
    sources = [(source, name) for mode, source, name in _LIVE_SOURCES if mode in modes]
    try:
        state = await current_run_state()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Auto Top-p Translate could not read the current runs: %s: %s",
            type(exc).__name__,
            redact_log_text(exc, 240),
        )
        return DiscoveryResult(documents=(), failed_sources=tuple(source for source, _name in sources))
    return await collect_specs((source, partial(globals()[name], state)) for source, name in sources)
