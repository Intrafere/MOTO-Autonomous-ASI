"""JSON contracts, deterministic pre-checks, and attempt records for Auto Top-p Translate.

Parsers route model output through the shared sanitizing JSON parser and raise
``TranslationContractError`` with fixed, safe messages: a message never embeds
raw model output, so it may be reused as the ``problem`` of a JSON repair turn.
Pre-checks run before the semantic validator and return actionable feedback for
the translator; they are never recorded as validator decisions.

Units are numbered from 1 (``UnitView.number``) in model-facing text and in
parsed responses; job state, finals, and part ranges use occurrence indexes
(``UnitView.occurrence``, which is the unit number minus 1).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.shared.json_parser import parse_json

from .segmentation import (
    DocumentPlan,
    UnitView,
    paragraph_markers,
    split_paragraphs,
    to_global,
    to_local,
)

TRANSLATOR_NOTES_MAX_CHARS = 1200
REASONING_MAX_CHARS = 2000
FEEDBACK_MAX_CHARS = 900
PROBE_NAME_MAX_CHARS = 60
PROBE_MAX_LANGUAGES = 300
LENGTH_RATIO_MIN = 0.2
LENGTH_RATIO_MAX = 5.0
LENGTH_RATIO_MIN_SOURCE_CHARS = 80
PROMPT_DELIMITERS = ("<<<SOURCE", "SOURCE>>>", "<<<TRANSLATION", "TRANSLATION>>>")
ACTION_TRANSLATE = "translate"
ACTION_KEEP = "keep"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_CHECK_FAILED = "check_failed"
STATUS_KEPT = "kept"

_PLACEHOLDER_SHAPE_RE = re.compile(r"\[\[(?:MOTO|M)\d+\]\]")
# Markup of the model-facing document views; a translation must not copy it.
_VIEW_MARKUP_RE = re.compile(r"<<U\d+>>|\[KEPT AS IS")
_NUMERIC_RE = re.compile(r"^\d+$", re.ASCII)
_MAX_LISTED_TOKENS = 12
_MAX_LISTED_MARKER_MISMATCHES = 5


class TranslationContractError(ValueError):
    """A model response that does not satisfy a translation JSON contract.

    The message is safe: it never contains raw model output.
    """


@dataclass(frozen=True)
class TranslatorEdit:
    """Replace the exact text ``find`` in an already translated unit with ``replace``."""

    unit: int
    find: str
    replace: str


@dataclass(frozen=True)
class TranslatorAttempt:
    """One parsed translator response; text uses document-wide placeholder tokens."""

    action: str  # "translate" | "keep"
    units: Tuple[Tuple[int, str], ...] = ()  # (unit number, translation) in response order
    edits: Tuple[TranslatorEdit, ...] = ()
    keep_attempts: Tuple[int, ...] = ()
    translator_notes: str = ""


@dataclass(frozen=True)
class AttemptDecision:
    attempt: int
    decision: str  # "accept" | "reject"
    reasoning: str
    feedback: str


def _load_json_object(raw_text: str) -> dict:
    try:
        # Translation text carries no raw LaTeX math (it is masked as
        # placeholders), so a JSON "\n" before a word such as "using", "otro",
        # or "egy" is a line break, not the LaTeX command \nu, \not, or \neg.
        parsed = parse_json(raw_text, escape_newline_latex_commands=False)
    except (ValueError, TypeError, RecursionError) as exc:
        detail = str(exc)
        if "truncated" in detail:
            message = "response JSON is incomplete; it looks cut off before the object was closed"
        elif "No JSON found" in detail:
            message = "response contains no JSON object"
        elif "Empty or whitespace-only" in detail or "too short" in detail:
            message = "response is empty or too short to contain the JSON object"
        elif isinstance(exc, json.JSONDecodeError):
            message = "response is not valid JSON"
        else:
            message = "response could not be parsed as JSON"
        raise TranslationContractError(message) from None
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
    if not isinstance(parsed, dict):
        raise TranslationContractError("response must be a single JSON object")
    return parsed


def _optional_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()[:limit].strip()


def _as_number(value: Any) -> Optional[int]:
    """Return a positive integer given as a JSON integer or numeric string, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and _NUMERIC_RE.match(value.strip()):
        number = int(value.strip())
    else:
        return None
    return number if number >= 1 else None


def _is_present(value: Any) -> bool:
    return value is not None and value != [] and value != {} and value != ""


def _parse_units(raw: Any) -> Tuple[Tuple[int, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise TranslationContractError(
            '"units" must be a non-empty JSON list with one entry per translated unit'
        )
    units: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise TranslationContractError(f'entry {index} of "units" must be a JSON object')
        number = _as_number(item.get("unit"))
        if number is None:
            raise TranslationContractError(
                f'entry {index} of "units" needs a "unit" number of 1 or more'
            )
        translation = item.get("translation")
        if not isinstance(translation, str):
            raise TranslationContractError(f"the translation of unit U{number} must be a JSON string")
        if number in seen:
            raise TranslationContractError(
                f'unit U{number} appears more than once in "units"; give each unit once'
            )
        seen.add(number)
        units.append((number, translation.strip()))
    return tuple(units)


def _parse_edits(raw: Any) -> Tuple[TranslatorEdit, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TranslationContractError('"edits" must be a JSON list of edits')
    edits: list[TranslatorEdit] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise TranslationContractError(
                f'edit {index} must be a JSON object with "unit", "find", and "replace"'
            )
        number = _as_number(item.get("unit"))
        if number is None:
            raise TranslationContractError(f'edit {index} needs a "unit" number of 1 or more')
        find = item.get("find")
        replace = item.get("replace")
        if not isinstance(find, str) or not isinstance(replace, str):
            raise TranslationContractError(
                f'edit {index} must give "find" and "replace" as JSON strings'
            )
        edits.append(TranslatorEdit(unit=number, find=find, replace=replace))
    return tuple(edits)


def _parse_keep_attempts(raw: Any) -> Tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TranslationContractError('"keep_attempts" must be a JSON list of attempt numbers')
    numbers: list[int] = []
    for item in raw:
        number = _as_number(item)
        if number is None:
            raise TranslationContractError(
                '"keep_attempts" must contain only attempt numbers of 1 or more'
            )
        if number not in numbers:
            numbers.append(number)
    return tuple(numbers)


def parse_translator_attempt(raw_text: str) -> TranslatorAttempt:
    """Parse one translator response in the "translate" or "keep" form.

    An empty unit translation or an empty ``keep_attempts`` list parses; the
    pre-check reports it as actionable feedback.
    """
    payload = _load_json_object(raw_text)
    action = payload.get("action")
    action = action.strip().casefold() if isinstance(action, str) else ""
    if action not in {ACTION_TRANSLATE, ACTION_KEEP}:
        raise TranslationContractError('action must be "translate" or "keep"')
    notes = _optional_text(payload.get("translator_notes"), TRANSLATOR_NOTES_MAX_CHARS)
    if action == ACTION_KEEP:
        if _is_present(payload.get("units")) or _is_present(payload.get("edits")):
            raise TranslationContractError(
                'a "keep" response must not contain units or edits; use the "translate" form to '
                "write new text"
            )
        return TranslatorAttempt(
            action=ACTION_KEEP,
            keep_attempts=_parse_keep_attempts(payload.get("keep_attempts")),
            translator_notes=notes,
        )
    if _is_present(payload.get("keep_attempts")):
        raise TranslationContractError(
            'a "translate" response must not name keep_attempts; use the "keep" form to keep '
            "earlier attempts"
        )
    return TranslatorAttempt(
        action=ACTION_TRANSLATE,
        units=_parse_units(payload.get("units")),
        edits=_parse_edits(payload.get("edits")),
        translator_notes=notes,
    )


def parse_validator_decisions(
    raw_text: str, attempt_numbers: Sequence[int]
) -> Tuple[AttemptDecision, ...]:
    """Strictly parse one decision per listed attempt, in the listed order."""
    expected = [int(number) for number in attempt_numbers]
    if not expected:
        raise ValueError("at least one attempt number is required")
    payload = _load_json_object(raw_text)
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise TranslationContractError('"decisions" must be a JSON list with one entry per attempt')
    if len(raw_decisions) != len(expected):
        raise TranslationContractError(
            f'"decisions" must contain exactly {len(expected)} entries, one per attempt; '
            f"found {len(raw_decisions)}"
        )

    decisions: list[AttemptDecision] = []
    for index, (item, number) in enumerate(zip(raw_decisions, expected), start=1):
        if not isinstance(item, dict):
            raise TranslationContractError(f"decision {index} must be a JSON object")
        if _as_number(item.get("attempt")) != number:
            raise TranslationContractError(
                f"decision {index} must be for attempt {number}; give one decision per attempt "
                "in the order shown"
            )
        decision = item.get("decision")
        decision = decision.strip().casefold() if isinstance(decision, str) else ""
        if decision not in {"accept", "reject"}:
            raise TranslationContractError(
                f'decision for attempt {number} must be "accept" or "reject"'
            )
        reasoning = _optional_text(item.get("reasoning"), REASONING_MAX_CHARS)
        feedback = _optional_text(item.get("feedback"), FEEDBACK_MAX_CHARS)
        if decision == "reject" and not feedback:
            feedback = reasoning[:FEEDBACK_MAX_CHARS].strip()
            if not feedback:
                raise TranslationContractError(f"feedback is required for rejected attempt {number}")
        decisions.append(
            AttemptDecision(attempt=number, decision=decision, reasoning=reasoning, feedback=feedback)
        )
    return tuple(decisions)


def occurrence_texts(mapping: Mapping[Any, Any]) -> Dict[int, str]:
    """Return a finals or units mapping keyed by int occurrence index.

    Persisted job state uses string keys; string values only.
    """
    return {int(key): text for key, text in mapping.items() if isinstance(text, str)}


def attempt_record(
    attempt: TranslatorAttempt, views: Sequence[UnitView], *, number: int, status: str
) -> Dict[str, Any]:
    """Return the persisted job-state entry for ``attempt``.

    Unit text is stored with each unit's own placeholder tokens, keyed by
    occurrence index. An attempt that failed the automatic checks keeps its
    text exactly as submitted (document-wide tokens): it is only ever shown
    back to the translator, and a stray token cannot survive ``to_local``.
    Edits are stored in document-wide token form, keyed by occurrence index.
    """
    by_occurrence = {view.occurrence: view for view in views}
    verbatim = status == STATUS_CHECK_FAILED
    units: Dict[str, str] = {}
    for unit_number, text in attempt.units:
        occurrence = unit_number - 1
        if verbatim:
            units[str(occurrence)] = text
            continue
        view = by_occurrence.get(occurrence)
        if view is None:
            raise ValueError(f"unit U{unit_number} is not in the document")
        units[str(occurrence)] = to_local(text, view)
    return {
        "number": number,
        "action": attempt.action,
        "units": units,
        "edits": [
            {"occurrence": edit.unit - 1, "find": edit.find, "replace": edit.replace}
            for edit in attempt.edits
        ],
        "keep_attempts": list(attempt.keep_attempts),
        "notes": attempt.translator_notes,
        "status": status,
        "feedback": "",
        "reasoning": "",
        "accepted_order": None,
    }


def attempt_from_record(record: Mapping[str, Any], views: Sequence[UnitView]) -> TranslatorAttempt:
    """Rebuild a stored attempt with document-wide tokens and unit numbers."""
    by_occurrence = {view.occurrence: view for view in views}
    verbatim = record.get("status") == STATUS_CHECK_FAILED
    units: list[tuple[int, str]] = []
    for occurrence, text in occurrence_texts(record.get("units") or {}).items():
        view = by_occurrence.get(occurrence)
        units.append((occurrence + 1, text if verbatim or view is None else to_global(text, view)))
    edits = tuple(
        TranslatorEdit(
            unit=int(edit["occurrence"]) + 1,
            find=str(edit.get("find") or ""),
            replace=str(edit.get("replace") or ""),
        )
        for edit in record.get("edits") or []
    )
    return TranslatorAttempt(
        action=str(record.get("action") or ACTION_TRANSLATE),
        units=tuple(units),
        edits=edits,
        keep_attempts=tuple(int(number) for number in record.get("keep_attempts") or []),
        translator_notes=str(record.get("notes") or ""),
    )


def _judgment_key(record: Mapping[str, Any]) -> Tuple[int, int]:
    number = int(record["number"])
    seq = record.get("seq")
    return (seq if isinstance(seq, int) and not isinstance(seq, bool) else number, number)


def judgment_order(records: Iterable[Mapping[str, Any]]) -> List[Any]:
    """Return attempt records oldest judgment first.

    Records order by their judgment sequence ``seq``, which is refreshed
    whenever a record's status or feedback changes, so an attempt judged again
    after a dispute comes after the keep that disputed it. A record without a
    sequence orders by its attempt number.
    """
    return sorted(records, key=_judgment_key)


def disputing_keep(
    record: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """Return the newest kept record in ``records`` that named ``record`` before its latest judgment.

    A keep naming an attempt judged after it was a dispute: the validator
    judged that attempt again. None when ``record`` was never judged again.
    """
    key = _judgment_key(record)
    disputes = [
        other
        for other in records
        if other.get("status") == STATUS_KEPT
        and record["number"] in (other.get("keep_attempts") or ())
        and _judgment_key(other) < key
    ]
    return max(disputes, key=_judgment_key, default=None)


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _list_tokens(tokens: list[str]) -> str:
    shown = ", ".join(tokens[:_MAX_LISTED_TOKENS])
    hidden = len(tokens) - _MAX_LISTED_TOKENS
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def _describe_marker(marker: str) -> str:
    if not marker:
        return "no Markdown marker"
    if marker == "ul":
        return 'a bullet marker such as "-"'
    if marker == "quote":
        return 'a quote marker ">"'
    if marker.startswith("ol:"):
        return f'the numbered-list marker "{marker[3:]}."'
    if marker.startswith("h") and marker[1:].isdigit():
        level = int(marker[1:])
        return f'a level-{level} heading marker "{"#" * level}"'
    return f'the marker "{marker}"'


def _placeholder_problems(source: str, expected_tokens: Sequence[str], text: str) -> list[str]:
    source_counts = Counter(_PLACEHOLDER_SHAPE_RE.findall(source))
    text_counts = Counter(_PLACEHOLDER_SHAPE_RE.findall(text))
    expected = list(dict.fromkeys(expected_tokens))
    missing = [token for token in expected if text_counts[token] == 0]
    repeated = [token for token in expected if text_counts[token] > max(1, source_counts[token])]
    unknown = [
        token
        for token in dict.fromkeys(_PLACEHOLDER_SHAPE_RE.findall(text))
        if token not in source_counts and token not in expected
    ]
    problems: list[str] = []
    if missing:
        problems.append(
            f"Placeholder(s) {_list_tokens(missing)} from the source are missing from your "
            "translation. Copy every placeholder exactly as written, once each, at the "
            "grammatically correct position."
        )
    if repeated:
        problems.append(
            f"Placeholder(s) {_list_tokens(repeated)} appear more than once in your translation. "
            "Copy each placeholder exactly once, at the position matching the source."
        )
    if unknown:
        problems.append(
            f"Your translation contains placeholder(s) {_list_tokens(unknown)} that do not exist "
            "in the source of this unit. Use only the placeholders that appear in this unit's "
            "source, exactly as written."
        )
    return problems


def _unit_problems(source: str, expected_tokens: Sequence[str], text: str) -> list[str]:
    """Return the deterministic problems of one unit's translation; [] when it passes.

    ``source`` and ``text`` use the same (document-wide) tokens, so returning
    the source unchanged (the already-in-target-language case) always passes.
    """
    if not text.strip():
        return ["Your translation is empty. Translate the complete unit."]

    problems = _placeholder_problems(source, expected_tokens, text)

    source_paragraphs = split_paragraphs(source)
    text_paragraphs = split_paragraphs(text)
    if len(source_paragraphs) != len(text_paragraphs):
        problems.append(
            f"The source has {_plural(len(source_paragraphs), 'paragraph', 'paragraphs')} "
            "separated by blank lines; your translation has "
            f"{len(text_paragraphs)}. Translate each source paragraph as exactly one "
            "paragraph, in the same order, and separate paragraphs with one blank line."
        )
    else:
        mismatches = [
            (index, source_marker, text_marker)
            for index, (source_marker, text_marker) in enumerate(
                zip(paragraph_markers(source), paragraph_markers(text)), start=1
            )
            if source_marker != text_marker
        ]
        if mismatches:
            details = "; ".join(
                f"paragraph {index} should start with {_describe_marker(source_marker)} but starts "
                f"with {_describe_marker(text_marker)}"
                for index, source_marker, text_marker in mismatches[:_MAX_LISTED_MARKER_MISMATCHES]
            )
            hidden = len(mismatches) - _MAX_LISTED_MARKER_MISMATCHES
            if hidden > 0:
                details += f"; and {_plural(hidden, 'more paragraph', 'more paragraphs')}"
            problems.append(
                f"Paragraph-start Markdown markers differ from the source: {details}. Keep each "
                "paragraph's leading marker exactly as in the source."
            )

    leaked = [
        delimiter for delimiter in PROMPT_DELIMITERS if text.count(delimiter) > source.count(delimiter)
    ]
    if leaked:
        problems.append(
            f"Your translation contains the prompt delimiter(s) {', '.join(leaked)}. Return only "
            "the translated text, without the <<<SOURCE / SOURCE>>> or <<<TRANSLATION / "
            "TRANSLATION>>> markers."
        )
    if len(_VIEW_MARKUP_RE.findall(text)) > len(_VIEW_MARKUP_RE.findall(source)):
        problems.append(
            "Your translation contains document view markup such as <<U12>> or [KEPT AS IS]. "
            "Give only the translated text of the unit; the unit number belongs in \"unit\"."
        )

    if len(source) >= LENGTH_RATIO_MIN_SOURCE_CHARS:
        ratio = len(text) / len(source)
        if ratio < LENGTH_RATIO_MIN:
            problems.append(
                f"Your translation is only {ratio:.0%} of the source length (expected at least "
                f"{LENGTH_RATIO_MIN:.0%}), so it is probably incomplete or summarized. Translate "
                "every sentence of the unit."
            )
        elif ratio > LENGTH_RATIO_MAX:
            problems.append(
                f"Your translation is {ratio:.0%} of the source length (expected at most "
                f"{LENGTH_RATIO_MAX:.0%}), so it probably contains additions, explanations, or "
                "repeated text. Translate the unit only, without adding content."
            )
    return problems


def _match_count(text: str, find: str) -> int:
    """Count possibly overlapping matches of ``find`` in ``text``."""
    count = 0
    position = text.find(find)
    while position != -1:
        count += 1
        position = text.find(find, position + 1)
    return count


def _keep_problems(part_state: Optional[Mapping[str, Any]], attempt: TranslatorAttempt) -> list[str]:
    if not attempt.keep_attempts:
        return [
            'Your "keep" response names no attempts. Name the earlier attempt or attempts to keep '
            'in "keep_attempts", or use the "translate" form.'
        ]
    statuses = {
        _as_number(entry.get("number")): str(entry.get("status") or "")
        for entry in ((part_state or {}).get("attempts") or [])
        if isinstance(entry, Mapping)
    }
    problems: list[str] = []
    for number in attempt.keep_attempts:
        status = statuses.get(number)
        if status in {STATUS_ACCEPTED, STATUS_REJECTED}:
            continue
        if status == STATUS_CHECK_FAILED:
            problems.append(
                f"Attempt {number} failed the automatic checks, so it cannot be kept. Keep an "
                'accepted or rejected attempt, or use the "translate" form.'
            )
        elif status == STATUS_KEPT:
            problems.append(
                f"Attempt {number} was itself a keep response and has no translation to keep. "
                "Name the accepted or rejected attempt instead."
            )
        else:
            problems.append(
                f"Attempt {number} does not exist for this part. Keep only attempts shown under "
                "EARLIER ATTEMPTS AT THIS PART."
            )
    return problems


def _range_problems(
    occurrences: Sequence[int],
    locked_range: Optional[Tuple[int, int]],
    cursor: int,
    max_end: int,
) -> list[str]:
    if locked_range is not None:
        start, end = locked_range
        if list(occurrences) == list(range(start, end + 1)):
            return []
        return [
            f"This part is units U{start + 1} to U{end + 1}, which already have an accepted "
            "translation. Translate exactly those units, in order, with none added or left out."
        ]
    problems: list[str] = []
    if occurrences[0] != cursor:
        problems.append(
            f"Your part must start at unit U{cursor + 1}, the first unit that is not translated "
            f"yet; it starts at U{occurrences[0] + 1}."
        )
    for previous, current in zip(occurrences, occurrences[1:]):
        if current != previous + 1:
            problems.append(
                f"The units of a part must be consecutive and in order: after U{previous + 1} "
                f"comes U{previous + 2}, but your response gives U{current + 1}. Translate each "
                "unit of the part exactly once, in order."
            )
            break
    last = max(occurrences)
    if last > max_end:
        problems.append(
            f"Your part reaches U{last + 1}, but this attempt may end no later than "
            f"U{max_end + 1}. Stop at a natural break by U{max_end + 1}."
        )
    return problems


def _edit_problems(
    attempt: TranslatorAttempt,
    views: Mapping[int, UnitView],
    sources: Mapping[str, str],
    finals: Mapping[int, str],
    part: Collection[int],
    visible: Optional[Collection[int]],
) -> list[str]:
    """Check edits in order; later edits of a unit apply to the text left by earlier ones."""
    problems: list[str] = []
    edited: Dict[int, str] = {}
    for index, edit in enumerate(attempt.edits, start=1):
        occurrence = edit.unit - 1
        label = f"Edit {index} (U{edit.unit})"
        view = views.get(occurrence)
        if view is None:
            problems.append(f"{label} names a unit that does not exist in the document.")
            continue
        if occurrence in part:
            problems.append(
                f"{label} targets a unit of this attempt. Put the correction in that unit's "
                "translation instead of an edit."
            )
            continue
        if occurrence not in finals:
            problems.append(
                f"{label} targets a unit that is not translated yet. Edit only units that already "
                "have a translation."
            )
            continue
        if visible is not None and occurrence not in visible:
            problems.append(
                f"{label} targets a unit that is not shown in the translation so far. Edit only "
                "units shown there."
            )
            continue
        if not edit.find:
            problems.append(
                f'{label} has an empty "find". Copy the exact text to replace from that unit\'s '
                "current translation."
            )
            continue
        if edit.find == edit.replace:
            problems.append(
                f'{label} replaces "find" with identical text. Remove the edit or give the '
                'corrected text in "replace".'
            )
            continue
        current = edited.get(occurrence)
        if current is None:
            current = to_global(finals[occurrence], view)
        matches = _match_count(current, edit.find)
        if matches == 0:
            problems.append(
                f'{label}: the "find" text does not appear in the current translation of '
                f"U{edit.unit}. Copy it exactly from that unit, including spacing and punctuation."
            )
            continue
        if matches > 1:
            problems.append(
                f'{label}: the "find" text appears {matches} times in the current translation of '
                f"U{edit.unit}. Include more of the surrounding text so it matches exactly once."
            )
            continue
        edited[occurrence] = current.replace(edit.find, edit.replace, 1)

    # An edited unit must still pass every check a translated unit gets.
    for occurrence, text in edited.items():
        view = views[occurrence]
        source = to_global(sources[view.source_hash], view)
        problems.extend(
            f"Your edits to U{view.number} would break that unit: {problem}"
            for problem in _unit_problems(source, view.global_tokens, text)
        )
    return problems


def precheck_attempt(
    views: Sequence[UnitView],
    finals: Mapping[Any, str],
    part_state: Optional[Mapping[str, Any]],
    attempt: TranslatorAttempt,
    max_end: int,
    *,
    plan: DocumentPlan,
    visible: Optional[Collection[int]] = None,
) -> list[str]:
    """Return deterministic, actionable problems with ``attempt``; [] when it passes.

    ``finals`` maps occurrence indexes to unit-local final translations,
    ``part_state`` is the job's persisted part (or None before the first
    attempt), ``max_end`` is the last occurrence index an unlocked part may
    reach, and ``visible`` optionally lists the occurrence indexes shown in a
    windowed view (edits may only target those).
    """
    if attempt.action == ACTION_KEEP:
        return _keep_problems(part_state, attempt)

    by_occurrence = {view.occurrence: view for view in views}
    sources = {segment.source_hash: segment.masked_text for segment in plan.segments}
    final_texts = occurrence_texts(finals)
    occurrences = [number - 1 for number, _ in attempt.units]
    if not occurrences:
        return ['Your "translate" response has no units. Translate every unit of the part.']

    locked_range: Optional[Tuple[int, int]] = None
    if part_state and part_state.get("locked"):
        locked_range = (int(part_state["start"]), int(part_state["end"]))
    cursor = next(
        (view.occurrence for view in views if view.occurrence not in final_texts), len(views)
    )
    problems = _range_problems(occurrences, locked_range, cursor, max_end)

    for number, text in attempt.units:
        view = by_occurrence.get(number - 1)
        if view is None:
            continue
        source = to_global(sources[view.source_hash], view)
        problems.extend(
            f"U{number}: {problem}" for problem in _unit_problems(source, view.global_tokens, text)
        )

    part = set(occurrences)
    if locked_range is not None:
        part.update(range(locked_range[0], locked_range[1] + 1))
    problems.extend(
        _edit_problems(attempt, by_occurrence, sources, final_texts, part, visible)
    )
    return problems


def _sanitize_language_name(value: str) -> str:
    cleaned = "".join(
        " " if character.isspace() else character
        for character in value
        if character.isspace() or not unicodedata.category(character).startswith("C")
    )
    cleaned = " ".join(cleaned.split())[:PROBE_NAME_MAX_CHARS].strip()
    return cleaned if any(character.isalpha() for character in cleaned) else ""


def parse_probe_response(raw_text: str) -> list[str]:
    """Parse a language probe response into sanitized, deduplicated language names."""
    payload = _load_json_object(raw_text)
    languages = payload.get("languages")
    if not isinstance(languages, list):
        raise TranslationContractError('"languages" must be a JSON list of language names')
    names: list[str] = []
    seen: set[str] = set()
    for item in languages:
        if not isinstance(item, str):
            continue
        name = _sanitize_language_name(item)
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= PROBE_MAX_LANGUAGES:
            break
    return names
