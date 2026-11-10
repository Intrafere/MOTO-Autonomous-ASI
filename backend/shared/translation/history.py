"""Enumerate the older outputs Auto Top-p Translate's Translate all translates.

Translate all covers what a user can still open outside the checked modes'
current runs: autonomous papers, pruned papers, final answers, and proofs in
the libraries and history; proofs in the autonomous and manual proof stores
that no current run owns; archived manual proof runs; a manual Aggregator
database or Compiler Live Paper whose run is not current; and saved Proof
Solver runs. Each spec is built exactly as the viewer that shows the item
builds its own, so the background translation is the one the viewer shows.

Enumeration reads only what the listing routes list now, never the
translation store, so content the user cleared is never included. Each
source leaves the checked modes' current runs out by itself. It is read-only, and every source is guarded on its own: a failing source is
skipped and named in ``failed_sources``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Collection, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from backend.shared.config import system_config
from backend.shared.log_redaction import redact_log_text
from backend.shared.translation.discovery import (
    MANUAL_MODES,
    MODE_AGGREGATOR,
    MODE_AUTONOMOUS,
    MODE_COMPILER,
    MODE_LEANOJ,
    CurrentRunState,
    DiscoveryResult,
    DocumentSpec,
    aggregator_results_specs,
    autonomous_proof_is_live,
    bounded_title,
    collect_specs,
    compiler_live_paper_specs,
    current_run_state,
    js_trim,
    leanoj_proof_specs,
    make_document_spec,
    manual_proof_is_live,
    manual_proof_mode,
    proof_document_specs,
    proof_spec,
    read_compiler_live_paper,
    read_manual_aggregator_results,
)
from backend.shared.translation.segmentation import PROFILE_DOCUMENT

logger = logging.getLogger(__name__)

SOURCE_HISTORY_PAPERS = "history_papers"
SOURCE_HISTORY_PRUNED_PAPERS = "history_pruned_papers"
SOURCE_HISTORY_FINAL_ANSWERS = "history_final_answers"
SOURCE_HISTORY_PROOF_LIBRARY = "history_proof_library"
SOURCE_AUTONOMOUS_STORE_OTHER_PROOFS = "autonomous_store_other_proofs"
SOURCE_MANUAL_STORE_OTHER_PROOFS = "manual_store_other_proofs"
SOURCE_MANUAL_PROOF_HISTORY = "manual_proof_history"
SOURCE_AGGREGATOR_RESULTS_INACTIVE = "aggregator_results_inactive"
SOURCE_COMPILER_LIVE_PAPER_INACTIVE = "compiler_live_paper_inactive"
SOURCE_LEANOJ_LIBRARY = "leanoj_library"

# What each source lists, named as the app names it, for messages about sources that failed.
SOURCE_LABELS: Dict[str, str] = {
    SOURCE_HISTORY_PAPERS: "Stage 2 Papers History",
    SOURCE_HISTORY_PRUNED_PAPERS: "pruned papers",
    SOURCE_HISTORY_FINAL_ANSWERS: "Stage 3 Final Answers History",
    SOURCE_HISTORY_PROOF_LIBRARY: "Proof Library",
    SOURCE_AUTONOMOUS_STORE_OTHER_PROOFS: "Autonomous Mathematical Proofs",
    SOURCE_MANUAL_STORE_OTHER_PROOFS: "Manual Mathematical Proofs",
    SOURCE_MANUAL_PROOF_HISTORY: "Completed Proof Works",
    SOURCE_AGGREGATOR_RESULTS_INACTIVE: "Aggregator Live Results",
    SOURCE_COMPILER_LIVE_PAPER_INACTIVE: "Single Paper Writer Live Paper",
    SOURCE_LEANOJ_LIBRARY: "Proof Solver Completed Proof Works",
}

MANUAL_PROOF_HISTORY_DIR_NAME = "manual_proof_runs"
# The Proof Solver library view shows only these prose fields, with no prompt exclusion.
LEANOJ_LIBRARY_TRANSLATION_FIELDS = ("theorem_statement", "novelty_group_claim")

# Legacy archived papers already reported as skipped, so each is logged once.
_skipped_archives: Set[str] = set()


@dataclass(frozen=True)
class _Scope:
    """What one enumeration covers: the current runs and the checked modes."""

    state: CurrentRunState
    modes: FrozenSet[str]


def _paper_spec(session_id: str, paper_id: str, paper: Optional[Mapping[str, Any]]) -> Optional[DocumentSpec]:
    content = paper.get("content") if paper else None
    if not isinstance(content, str) or not js_trim(content):
        return None
    return make_document_spec(
        doc_key=f"paper:{session_id}:{paper_id}",
        kind="paper",
        title=bounded_title(paper.get("title"), paper_id),
        profile=PROFILE_DOCUMENT,
        parts=(("content", content),),
    )


def _history_ids(entry: Mapping[str, Any]) -> Tuple[str, str]:
    return str(entry.get("session_id") or ""), str(entry.get("paper_id") or "")


async def _enumerate_history_papers(scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.paper_library import paper_library

    specs: List[DocumentSpec] = []
    for entry in await paper_library.list_history_papers():
        session_id, paper_id = _history_ids(entry)
        if session_id == scope.state.autonomous_run:
            # Current-run discovery owns the current run's papers.
            continue
        spec = _paper_spec(session_id, paper_id, await paper_library.get_history_paper(session_id, paper_id))
        if spec is not None:
            specs.append(spec)
    return specs


def _archive_has_banner_time_sync(metadata_path: Path) -> bool:
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return isinstance(metadata, dict) and bool(metadata.get("pruned_at") or metadata.get("created_at"))


async def _legacy_archive_is_stable(session_id: str, paper_id: str) -> bool:
    """Whether a legacy archived paper's pruned banner reads the same every time.

    The banner of a legacy archive is built when the paper is read, from its
    stored pruned or creation time, or from the current time when neither is
    stored; such a paper has no stable identity, so it is not enumerated.
    """
    from backend.autonomous.memory.paper_library import paper_library

    papers_dir = paper_library.get_history_papers_dir(session_id)
    if papers_dir is None:
        return False
    metadata_path = paper_library._build_scoped_library(papers_dir)._get_archive_metadata_path(paper_id)
    if await asyncio.to_thread(_archive_has_banner_time_sync, metadata_path):
        return True
    history_id = f"{session_id}:{paper_id}"
    if history_id not in _skipped_archives:
        _skipped_archives.add(history_id)
        logger.info(
            "Translate all skips legacy archived paper %s: it has no stored time, so its pruned banner "
            "changes on every read.",
            redact_log_text(history_id, 200),
        )
    return False


async def _enumerate_history_pruned_papers(_scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.paper_library import paper_library

    specs: List[DocumentSpec] = []
    for entry in await paper_library.list_pruned_history_papers():
        session_id, paper_id = _history_ids(entry)
        if entry.get("status") == "archived" and not await _legacy_archive_is_stable(session_id, paper_id):
            continue
        spec = _paper_spec(session_id, paper_id, await paper_library.get_pruned_history_paper(session_id, paper_id))
        if spec is not None:
            specs.append(spec)
    return specs


async def _enumerate_history_final_answers(_scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.final_answer_memory import final_answer_memory

    specs: List[DocumentSpec] = []
    for entry in await final_answer_memory.list_all_final_answers():
        answer_id = str(entry.get("answer_id") or "")
        answer = await final_answer_memory.get_final_answer_by_id(answer_id)
        content = answer.get("content") if answer else None
        if not isinstance(content, str) or not js_trim(content):
            continue
        specs.append(
            make_document_spec(
                doc_key=f"final:{answer_id}",
                kind="final_answer",
                title=bounded_title(entry.get("title"), "Final Answer"),
                profile=PROFILE_DOCUMENT,
                parts=(("content", content),),
            )
        )
    return specs


def _library_proof_specs(entries: List[Dict[str, Any]]) -> List[DocumentSpec]:
    specs: List[DocumentSpec] = []
    for entry in entries:
        spec = proof_spec(entry, f"proof:library:{entry.get('library_id') or ''}")
        if spec is not None:
            specs.append(spec)
    return specs


async def _enumerate_history_proof_library(_scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import proof_database

    return _library_proof_specs(await proof_database.list_proof_library(novel_only=None, category="all"))


async def _enumerate_autonomous_store_other_proofs(scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import proof_database

    records = await proof_database.get_all_proofs()
    return proof_document_specs(
        [record for record in records if not autonomous_proof_is_live(record, scope.state)], "autonomous"
    )


def _manual_record_checked(record: Any, modes: FrozenSet[str]) -> bool:
    """A manual proof's mode is checked; one no mode owns shows in the manual workspace."""
    mode = manual_proof_mode(record)
    return mode in modes if mode else bool(MANUAL_MODES & modes)


async def _enumerate_manual_store_other_proofs(scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import manual_proof_database

    records = await manual_proof_database.get_all_proofs()
    return proof_document_specs(
        [
            record
            for record in records
            if not manual_proof_is_live(record, scope.state) and _manual_record_checked(record, scope.modes)
        ],
        "manual",
    )


async def _enumerate_manual_proof_history(scope: _Scope) -> List[DocumentSpec]:
    from backend.autonomous.memory.proof_database import manual_proof_database

    entries = await manual_proof_database.list_proof_library_from_history(
        Path(system_config.data_dir) / MANUAL_PROOF_HISTORY_DIR_NAME, None, "all"
    )
    return _library_proof_specs([entry for entry in entries if _manual_record_checked(entry, scope.modes)])


async def _enumerate_aggregator_results_inactive(scope: _Scope) -> List[DocumentSpec]:
    if scope.state.aggregator_live:
        return []
    return aggregator_results_specs(await read_manual_aggregator_results())


async def _enumerate_compiler_live_paper_inactive(scope: _Scope) -> List[DocumentSpec]:
    if scope.state.compiler_live or scope.state.compiler_writing:
        return []
    return compiler_live_paper_specs(await read_compiler_live_paper())


async def _enumerate_leanoj_library(scope: _Scope) -> List[DocumentSpec]:
    from backend.leanoj.core.leanoj_proof_views import _extract_leanoj_proofs, _iter_leanoj_state_payloads

    payloads = {
        str(payload.get("session_id") or ""): payload
        for payload in await asyncio.to_thread(_iter_leanoj_state_payloads)
        if payload.get("session_id")
    }
    payloads.pop(scope.state.leanoj_session, None)
    proofs = [proof for payload in payloads.values() for proof in _extract_leanoj_proofs(payload)]
    return leanoj_proof_specs(proofs, LEANOJ_LIBRARY_TRANSLATION_FIELDS, hide_prompt_statement=False)


# (modes, source, enumerator name); a source runs when any of its modes is
# checked, and enumerators are looked up by name at enumeration time.
_HISTORY_SOURCES: Tuple[Tuple[FrozenSet[str], str, str], ...] = (
    (frozenset({MODE_AUTONOMOUS}), SOURCE_HISTORY_PAPERS, "_enumerate_history_papers"),
    (frozenset({MODE_AUTONOMOUS}), SOURCE_HISTORY_PRUNED_PAPERS, "_enumerate_history_pruned_papers"),
    (frozenset({MODE_AUTONOMOUS}), SOURCE_HISTORY_FINAL_ANSWERS, "_enumerate_history_final_answers"),
    (frozenset({MODE_AUTONOMOUS}), SOURCE_HISTORY_PROOF_LIBRARY, "_enumerate_history_proof_library"),
    (frozenset({MODE_AUTONOMOUS}), SOURCE_AUTONOMOUS_STORE_OTHER_PROOFS, "_enumerate_autonomous_store_other_proofs"),
    (MANUAL_MODES, SOURCE_MANUAL_STORE_OTHER_PROOFS, "_enumerate_manual_store_other_proofs"),
    (MANUAL_MODES, SOURCE_MANUAL_PROOF_HISTORY, "_enumerate_manual_proof_history"),
    (frozenset({MODE_AGGREGATOR}), SOURCE_AGGREGATOR_RESULTS_INACTIVE, "_enumerate_aggregator_results_inactive"),
    (frozenset({MODE_COMPILER}), SOURCE_COMPILER_LIVE_PAPER_INACTIVE, "_enumerate_compiler_live_paper_inactive"),
    (frozenset({MODE_LEANOJ}), SOURCE_LEANOJ_LIBRARY, "_enumerate_leanoj_library"),
)


def _checked_sources(modes: FrozenSet[str]) -> List[Tuple[str, str]]:
    return [(source, name) for source_modes, source, name in _HISTORY_SOURCES if source_modes & modes]


def history_sources(modes: Collection[str]) -> Tuple[str, ...]:
    """The sources an enumeration of ``modes`` reads, in enumeration order."""
    return tuple(source for source, _name in _checked_sources(frozenset(modes)))


async def enumerate_displayed_outputs(modes: Collection[str]) -> DiscoveryResult:
    """Enumerate every displayed output of ``modes`` outside their current runs.

    When the current runs cannot be read, every checked source counts as failed.
    """
    checked = frozenset(modes)
    sources = _checked_sources(checked)
    try:
        state = await current_run_state()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Translate all could not read the current runs: %s: %s",
            type(exc).__name__,
            redact_log_text(exc, 240),
        )
        return DiscoveryResult(documents=(), failed_sources=tuple(source for source, _name in sources))
    scope = _Scope(state=state, modes=checked)
    return await collect_specs((source, partial(globals()[name], scope)) for source, name in sources)
