"""Deterministic segmentation, masking, and assembly for Auto Top-p Translate.

A document is split into protected blocks (copied verbatim) and translatable
segments. A segment is one paragraph-level unit: a translatable paragraph, or a
sentence- or word-boundary piece of a paragraph longer than
``SEGMENT_TARGET_TOKENS``. Mathematics, code, references, and links inside
translatable text are masked with placeholders so translators never see or
alter them. Segment identity is the hash of the masked text, so identical prose
is translated once. Unit views number every segment occurrence across the
document and renumber its placeholders document-wide for model prompts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from backend.shared.utils import count_tokens

SEGMENTATION_VERSION = "moto-translate-seg-v2"
SEGMENT_TARGET_TOKENS = 600
PROFILE_DOCUMENT = "document"
PROFILE_FIELDS = "fields"
PROFILES = (PROFILE_DOCUMENT, PROFILE_FIELDS)
PROOF_TRANSLATION_FIELDS = (
    "theorem_statement",
    "formal_sketch",
    "novelty_reasoning",
    "novelty_group_claim",
)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\r?\n(?:[ \t]*\r?\n)*")
_HEADING_RE = re.compile(r"^(#{1,6})\s")
_BULLET_RE = re.compile(r"^[-*+]\s")
_ORDERED_RE = re.compile(r"^(\d+)[.)]\s")
_QUOTE_RE = re.compile(r"^>")

_LINE_RE = re.compile(r"[^\n]*\n|[^\n]+")

# Structural lines of papers, final-answer volumes, brainstorm databases, and
# proof appendices. The backend and frontend proof parsers read these, so a
# translated document must keep them verbatim to parse the same way.
_PROTECTED_LINE_RES = (
    re.compile(r"^\s*\[HARD CODED[^\]]*\]\s*$"),
    re.compile(r"^\s*\[Theorems appendix -[^\]]*\]\s*$"),
    re.compile(r"^\s*([=#*_~-])\1{2,}\s*$"),
    re.compile(r"^\s*===.*===\s*$"),
    re.compile(r"^SUBMISSION #\d+ \| Accepted:.*$"),
    re.compile(r"^#\s*CHAPTER \d+:.*$"),
    re.compile(r"^#\s*Type:.*$"),
    re.compile(r"^\s*MODEL CREDITS\s*$", re.IGNORECASE),
    re.compile(r"^\s*END OF VOLUME\s*$"),
    re.compile(r"^\s*(?:FINAL ANSWER VOLUME|TABLE OF CONTENTS)\s*$"),
    re.compile(r"^(?:\s*#{1,6}\s*)?AI Self-Review and Limitations\s*$", re.IGNORECASE),
    re.compile(r"^\s*Theorem\s*\([^)\r\n]*\).*$", re.IGNORECASE),
    re.compile(r"^\s*(?:Status|Source):.*$"),
    # Brainstorm and Aggregator proof appendices identify entries by these labels.
    re.compile(r"^\s*(?:Proof ID|Theorem Name):.*$", re.IGNORECASE),
)
# Paper appendices write "Lean 4 proof:"; brainstorm and Aggregator appendices
# write "Lean 4 Code:". Both blocks run through the entry's "---" terminator.
_LEAN_START_RE = re.compile(r"^\s*Lean 4 (?:proof|code):\s*$", re.IGNORECASE)
_LEAN_END_RE = re.compile(r"^\s*---\s*$")
# A Lean block missing its terminator still ends before structure that never
# appears in Lean code, so the prose after a malformed entry stays translatable.
_LEAN_BOUNDARY_RES = (
    re.compile(r"^\s*\[HARD CODED[^\]]*\]\s*$"),
    re.compile(r"^\s*===.*===\s*$"),
    re.compile(r"^(?:\s*#{1,6}\s*)?AI Self-Review and Limitations\s*$", re.IGNORECASE),
    re.compile(r"^\s*([=#])\1{19,}\s*$"),
)
# CommonMark fences use backticks or tildes. A bare "~~~" line also matches
# the separator rule; the fence check runs first, so it opens a fence.
_FENCE_RE = re.compile(r"^\s*(?P<fence>```|~~~)")
# As in CommonMark, only a bare fence line of the opening character closes a
# fence; an info string such as "```lean" always opens one.
_FENCE_CLOSE_RES = {
    "`": re.compile(r"^\s*```+\s*$"),
    "~": re.compile(r"^\s*~~~+\s*$"),
}
# Labels kept verbatim while the statement after them is translated. Brainstorm
# and Aggregator appendices write the statement on the "Proof N:" header line.
_STATEMENT_LABEL_RE = re.compile(
    r"^(\s*(?:(?:Theorem\s+)?Statement:|Proof\s+\d+\s*:)[ \t]*)",
    re.IGNORECASE,
)

_MATH_ENVIRONMENTS = (
    "equation", "align", "gather", "multline", "eqnarray", "displaymath", "math",
    "split", "cases", "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
    "aligned", "gathered", "alignat", "flalign",
)
# Display delimiters may span lines but never a blank line, so a stray opener
# cannot swallow the following paragraphs.
_WITHIN_PARAGRAPH = r"(?:(?!\n[ \t]*\r?\n)[\s\S])"
# Latin-script word characters. An inline code span never opens right after
# one or closes right before one, so backtick apostrophes ("It`s") and LaTeX
# quotes (``word'') stay prose. CJK text, which writes code spans directly
# against neighboring characters, is unaffected.
_CODE_WORD_CHAR = r"[0-9A-Za-z_À-ɏ]"
# Alternatives are tried left to right at each position; the leftmost match wins.
_MASK_RE = re.compile(
    r"\[\[(?:MOTO|M)\d+\]\]"
    r"|\\begin\{(?P<env>" + "|".join(_MATH_ENVIRONMENTS) + r")(?P<star>\*?)\}"
    r"[\s\S]*?\\end\{(?P=env)(?P=star)\}"
    r"|(?<!\\)\$\$" + _WITHIN_PARAGRAPH + r"+?\$\$"
    r"|(?<!\\)\\\[" + _WITHIN_PARAGRAPH + r"+?\\\]"
    r"|(?<!\\)\\\(" + _WITHIN_PARAGRAPH + r"+?\\\)"
    r"|(?<![\\$])\$(?![\s$])(?:\\.|[^\\$\n])+?(?<!\s)\$(?!\d)"
    r"|(?<!" + _CODE_WORD_CHAR + r")(?P<tick>`{1,2})(?!`)(?:(?!'')[^\n])+?(?<!`)(?P=tick)"
    r"(?!`)(?!" + _CODE_WORD_CHAR + r")"
    # URLs end at the first non-ASCII character, so CJK text (and full-width
    # punctuation) after a link stays translatable.
    r"|(?P<url>https?://[^\s)\]>\"'\u0080-\U0010FFFF]+)"
    r"|\\(?:cite|ref|eqref|label|url|href)\{[^}]*\}"
)
_URL_TRAILING_PUNCTUATION = ".,;:!?"
_PLACEHOLDER_SHAPE_RE = re.compile(r"\[\[(?:MOTO|M)\d+\]\]")

# Blank-line paragraph separators, matching split_paragraphs, widened to take
# the trailing spaces of one paragraph and the indentation of the next.
_PARAGRAPH_GAP_RE = re.compile(r"[ \t]*\r?\n(?:[ \t]*\r?\n)+[ \t]*")
_SENTENCE_GAP_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WHITESPACE_RE = re.compile(r"\s+")

_PLAN_KEY_VERSION = "segmentation_version"


@dataclass(frozen=True)
class SegmentSource:
    """One unique paragraph-level unit, identified by its masked text."""

    source_hash: str
    masked_text: str
    placeholder_tokens: Tuple[str, ...]
    paragraph_count: int
    paragraph_markers: Tuple[str, ...]
    token_count: int


@dataclass(frozen=True)
class PlanBlock:
    """A protected span or one occurrence of a segment inside a document part."""

    kind: str  # "protected" | "segment"
    text: str = ""
    source_hash: str = ""
    placeholders: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DocumentPlan:
    """The full block structure of one document."""

    profile: str
    doc_hash: str
    parts: Tuple[Tuple[str, Tuple[PlanBlock, ...]], ...]
    segments: Tuple[SegmentSource, ...]


def split_paragraphs(text: str) -> list[str]:
    """Return blank-line-separated, non-empty paragraphs with outer whitespace removed."""
    return [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(text or "") if part.strip()]


def _paragraph_marker(paragraph: str) -> str:
    first_line = paragraph.lstrip().splitlines()[0] if paragraph.strip() else ""
    heading = _HEADING_RE.match(first_line)
    if heading:
        return f"h{len(heading.group(1))}"
    if _BULLET_RE.match(first_line):
        return "ul"
    ordered = _ORDERED_RE.match(first_line)
    if ordered:
        return f"ol:{int(ordered.group(1))}"
    if _QUOTE_RE.match(first_line):
        return "quote"
    return ""


def paragraph_markers(text: str) -> Tuple[str, ...]:
    """Return the normalized leading Markdown marker of every paragraph."""
    return tuple(_paragraph_marker(paragraph) for paragraph in split_paragraphs(text))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _line_content(line: str) -> str:
    """Return a line without its line terminator, for classification only."""
    if line.endswith("\n"):
        line = line[:-1]
    return line[:-1] if line.endswith("\r") else line


def _matches_any(patterns: Sequence[re.Pattern[str]], content: str) -> bool:
    return any(pattern.match(content) for pattern in patterns)


def _fenced_block_end(lines: Sequence[str], start: int, fence_char: str) -> int:
    """Return the exclusive end of a fenced code block opened at ``start`` with ``fence_char``."""
    close_re = _FENCE_CLOSE_RES[fence_char]
    for index in range(start + 1, len(lines)):
        if close_re.match(_line_content(lines[index])):
            return index + 1
    return len(lines)


def _lean_block_end(lines: Sequence[str], start: int) -> int:
    """Return the exclusive end of a Lean block opened at ``start``."""
    for index in range(start + 1, len(lines)):
        content = _line_content(lines[index])
        if _LEAN_END_RE.match(content):
            return index + 1
        if _matches_any(_LEAN_BOUNDARY_RES, content):
            return index
    return len(lines)


def _classify_part(profile: str, text: str) -> list[tuple[bool, str]]:
    """Split one part into ``(translatable, text)`` chunks that cover it exactly."""
    chunks: list[tuple[bool, list[str]]] = []

    def add(translatable: bool, piece: str) -> None:
        if not piece:
            return
        if chunks and chunks[-1][0] == translatable:
            chunks[-1][1].append(piece)
        else:
            chunks.append((translatable, [piece]))

    document = profile == PROFILE_DOCUMENT
    lines = _LINE_RE.findall(text)
    index = 0
    while index < len(lines):
        line = lines[index]
        content = _line_content(line)
        fence = _FENCE_RE.match(content)
        if fence:
            end = _fenced_block_end(lines, index, fence.group("fence")[0])
            add(False, "".join(lines[index:end]))
            index = end
            continue
        if document and _LEAN_START_RE.match(content):
            end = _lean_block_end(lines, index)
            add(False, "".join(lines[index:end]))
            index = end
            continue
        label = _STATEMENT_LABEL_RE.match(content) if document else None
        if document and _matches_any(_PROTECTED_LINE_RES, content):
            add(False, line)
        elif label is not None:
            add(False, line[: label.end()])
            add(True, line[label.end():])
        else:
            add(True, line)
        index += 1
    return [(translatable, "".join(pieces)) for translatable, pieces in chunks]


def _sentinel_for(text: str) -> str:
    """Pick a private-use character absent from ``text`` to mark masked spans."""
    for codepoint in itertools.chain(range(0xE000, 0xF900), range(0xF0000, 0xFFFFE)):
        candidate = chr(codepoint)
        if candidate not in text:
            return candidate
    raise ValueError("Text uses every private-use character; it cannot be masked.")


class _MaskedRun:
    """One translatable run with masked spans held under temporary markers.

    Temporary markers contain no whitespace or sentence punctuation, so
    paragraph, sentence, and whitespace splitting never cut through a masked
    span. ``finalize`` renumbers the markers of one segment from 1.
    """

    def __init__(self, core: str, prefix: str) -> None:
        self.prefix = prefix
        self.originals: list[str] = []
        self._token_cache: dict[str, int] = {}
        sentinel = ""
        pieces: list[str] = []
        position = 0
        match = _MASK_RE.search(core)
        while match is not None:
            start, end = match.span()
            url = match.group("url")
            if url:
                end = start + len(url.rstrip(_URL_TRAILING_PUNCTUATION))
            if not sentinel:
                sentinel = _sentinel_for(core)
            pieces.append(core[position:start])
            pieces.append(f"{sentinel}{len(self.originals)}{sentinel}")
            self.originals.append(core[start:end])
            position = end
            match = _MASK_RE.search(core, end)
        pieces.append(core[position:])
        self.masked = "".join(pieces)
        self._marker_re = (
            re.compile(re.escape(sentinel) + r"(\d+)" + re.escape(sentinel)) if sentinel else None
        )

    def finalize(self, text: str) -> tuple[str, Tuple[Tuple[str, str], ...]]:
        """Return the segment's masked text and its (token, original) pairs."""
        if self._marker_re is None:
            return text, ()
        placeholders: list[tuple[str, str]] = []

        def replace(match: re.Match[str]) -> str:
            token = f"[[{self.prefix}{len(placeholders) + 1}]]"
            placeholders.append((token, self.originals[int(match.group(1))]))
            return token

        return self._marker_re.sub(replace, text), tuple(placeholders)

    def count(self, text: str) -> int:
        """Return the token count of ``text`` once its markers are renumbered."""
        cached = self._token_cache.get(text)
        if cached is None:
            cached = count_tokens(self.finalize(text)[0])
            self._token_cache[text] = cached
        return cached


def _split_with_gaps(text: str, pattern: re.Pattern[str]) -> tuple[list[str], list[str]]:
    """Split ``text`` at ``pattern`` into non-empty items and the gaps between them.

    ``gaps[i]`` sits between ``items[i]`` and ``items[i + 1]``. Matches at the
    very start or end of ``text`` stay inside the adjacent item.
    """
    items: list[str] = []
    gaps: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        if match.start() == 0 or match.end() == len(text) or match.start() == position:
            continue
        items.append(text[position:match.start()])
        gaps.append(match.group())
        position = match.end()
    items.append(text[position:])
    return items, gaps


def _paragraphs_with_gaps(masked: str) -> tuple[list[str], list[str]]:
    """Split stripped run text into paragraphs, folding whitespace-only ones into gaps."""
    raw_items, raw_gaps = _split_with_gaps(masked, _PARAGRAPH_GAP_RE)
    paragraphs = [raw_items[0]]
    gaps: list[str] = []
    pending_gap = ""
    for gap, item in zip(raw_gaps, raw_items[1:]):
        pending_gap += gap
        if item.strip():
            gaps.append(pending_gap)
            paragraphs.append(item)
            pending_gap = ""
        else:
            pending_gap += item
    if pending_gap:
        paragraphs[-1] += pending_gap
    return paragraphs, gaps


def _join(items: Sequence[str], gaps: Sequence[str], start: int, end: int) -> str:
    pieces = [items[start]]
    for index in range(start + 1, end):
        pieces.append(gaps[index - 1])
        pieces.append(items[index])
    return "".join(pieces)


def _group_ranges(
    items: Sequence[str],
    gaps: Sequence[str],
    item_tokens: Sequence[int],
    count: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Greedily group consecutive items into ``[lo, hi)`` ranges within the target.

    Additive per-item estimates choose the ranges cheaply. Each multi-item range
    is then counted exactly and regrouped exactly in the rare case the joined
    text tokenizes longer than its estimate. Every range keeps at least one item.
    """
    estimated: list[tuple[int, int]] = []
    low = 0
    total = item_tokens[0]
    for index in range(1, len(items)):
        extended = total + count(gaps[index - 1]) + item_tokens[index]
        if extended <= SEGMENT_TARGET_TOKENS:
            total = extended
        else:
            estimated.append((low, index))
            low = index
            total = item_tokens[index]
    estimated.append((low, len(items)))

    ranges: list[tuple[int, int]] = []
    for low, high in estimated:
        if high - low == 1 or count(_join(items, gaps, low, high)) <= SEGMENT_TARGET_TOKENS:
            ranges.append((low, high))
            continue
        exact_low = low
        for index in range(low + 1, high):
            if count(_join(items, gaps, exact_low, index + 1)) > SEGMENT_TARGET_TOKENS:
                ranges.append((exact_low, index))
                exact_low = index
        ranges.append((exact_low, high))
    return ranges


def _split_oversized_paragraph(paragraph: str, run: _MaskedRun) -> list[tuple[bool, str]]:
    """Split one oversized paragraph at sentences, then whitespace, into target-sized pieces."""
    sentences, sentence_gaps = _split_with_gaps(paragraph, _SENTENCE_GAP_RE)
    items: list[str] = []
    gaps: list[str] = []
    for index, sentence in enumerate(sentences):
        if run.count(sentence) > SEGMENT_TARGET_TOKENS:
            pieces, piece_gaps = _split_with_gaps(sentence, _WHITESPACE_RE)
        else:
            pieces, piece_gaps = [sentence], []
        if index:
            gaps.append(sentence_gaps[index - 1])
        items.append(pieces[0])
        for gap, piece in zip(piece_gaps, pieces[1:]):
            gaps.append(gap)
            items.append(piece)
    tokens = [run.count(item) for item in items]
    units: list[tuple[bool, str]] = []
    for position, (low, high) in enumerate(_group_ranges(items, gaps, tokens, run.count)):
        if position:
            units.append((False, gaps[low - 1]))
        units.append((True, _join(items, gaps, low, high)))
    return units


def _run_units(run: _MaskedRun) -> list[tuple[bool, str]]:
    """Return ``(is_segment, masked text)`` units covering the run's masked text.

    Every paragraph is its own unit; paragraphs are never grouped. A paragraph
    whose masked token count exceeds the target is split into pieces.
    """
    paragraphs, gaps = _paragraphs_with_gaps(run.masked)
    units: list[tuple[bool, str]] = []
    for index, paragraph in enumerate(paragraphs):
        if index:
            units.append((False, gaps[index - 1]))
        if run.count(paragraph) > SEGMENT_TARGET_TOKENS:
            units.extend(_split_oversized_paragraph(paragraph, run))
        else:
            units.append((True, paragraph))
    return units


def _has_translatable_letters(masked_text: str) -> bool:
    """Return whether any letter remains once placeholders are removed."""
    return any(character.isalpha() for character in _PLACEHOLDER_SHAPE_RE.sub("", masked_text))


class _BlockBuilder:
    """Collects one part's blocks, merging adjacent protected text."""

    def __init__(self) -> None:
        self.blocks: list[PlanBlock] = []
        self._protected: list[str] = []

    def protect(self, text: str) -> None:
        if text:
            self._protected.append(text)

    def segment(self, source_hash: str, placeholders: Tuple[Tuple[str, str], ...]) -> None:
        self._flush()
        self.blocks.append(
            PlanBlock(kind="segment", source_hash=source_hash, placeholders=placeholders)
        )

    def finish(self) -> Tuple[PlanBlock, ...]:
        self._flush()
        return tuple(self.blocks)

    def _flush(self) -> None:
        if self._protected:
            self.blocks.append(PlanBlock(kind="protected", text="".join(self._protected)))
            self._protected = []


def _plan_run(
    run_text: str,
    builder: _BlockBuilder,
    segments: dict[str, SegmentSource],
) -> None:
    """Add one translatable run's protected and segment blocks to ``builder``."""
    core_start = len(run_text) - len(run_text.lstrip())
    core_end = len(run_text.rstrip())
    if core_start >= core_end:
        builder.protect(run_text)
        return
    builder.protect(run_text[:core_start])
    run = _MaskedRun(run_text[core_start:core_end], "MOTO" if "[[M" in run_text else "M")
    for is_segment, text in _run_units(run):
        if not is_segment:
            builder.protect(text)
            continue
        masked_text, placeholders = run.finalize(text)
        if not _has_translatable_letters(masked_text):
            builder.protect(_unmask_pairs(masked_text, placeholders))
            continue
        source_hash = _sha256_hex(f"{SEGMENTATION_VERSION}\n{masked_text}")
        if source_hash not in segments:
            segments[source_hash] = SegmentSource(
                source_hash=source_hash,
                masked_text=masked_text,
                placeholder_tokens=tuple(token for token, _ in placeholders),
                paragraph_count=len(split_paragraphs(masked_text)),
                paragraph_markers=paragraph_markers(masked_text),
                token_count=run.count(text),
            )
        builder.segment(source_hash, placeholders)
    builder.protect(run_text[core_end:])


def _validated_parts(parts: Sequence[Tuple[str, str]]) -> list[tuple[str, str]]:
    validated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, text in parts:
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("Document part names and texts must be strings.")
        if name in seen:
            raise ValueError(f"Duplicate document part name: {name!r}.")
        seen.add(name)
        validated.append((name, text))
    return validated


def build_document_plan(profile: str, parts: Sequence[Tuple[str, str]]) -> DocumentPlan:
    """Split document parts into protected blocks and deduplicated translatable segments.

    Raises ``ValueError`` for an unknown profile, non-string parts, or duplicate
    part names. Identity assembly of the result reproduces every part exactly.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown segmentation profile: {profile!r}.")
    validated = _validated_parts(parts)
    doc_hash = _sha256_hex(
        json.dumps(
            {
                "version": SEGMENTATION_VERSION,
                "profile": profile,
                "parts": [[name, text] for name, text in validated],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    segments: dict[str, SegmentSource] = {}
    plan_parts: list[tuple[str, Tuple[PlanBlock, ...]]] = []
    for name, text in validated:
        builder = _BlockBuilder()
        for translatable, chunk in _classify_part(profile, text):
            if translatable:
                _plan_run(chunk, builder, segments)
            else:
                builder.protect(chunk)
        plan_parts.append((name, builder.finish()))
    return DocumentPlan(
        profile=profile,
        doc_hash=doc_hash,
        parts=tuple(plan_parts),
        segments=tuple(segments.values()),
    )


def _unmask_pairs(masked: str, placeholders: Sequence[Tuple[str, str]]) -> str:
    if not placeholders:
        return masked
    originals = dict(placeholders)
    pattern = re.compile(
        "|".join(re.escape(token) for token in sorted(originals, key=len, reverse=True))
    )
    return pattern.sub(lambda match: originals[match.group()], masked)


def unmask(masked: str, placeholders: Sequence[Tuple[str, str]]) -> str:
    """Replace every placeholder token in ``masked`` with its original text in one pass."""
    return _unmask_pairs(masked, placeholders)


def assemble_parts(plan: DocumentPlan, translations: Mapping[str, str]) -> Optional[dict[str, str]]:
    """Rebuild every part from protected blocks and masked translations.

    ``translations`` maps source hashes to masked translated text. Returns None
    when any segment of the plan has no translation.
    """
    assembled: dict[str, str] = {}
    for name, blocks in plan.parts:
        pieces: list[str] = []
        for block in blocks:
            if block.kind != "segment":
                pieces.append(block.text)
                continue
            translation = translations.get(block.source_hash)
            if not isinstance(translation, str):
                return None
            pieces.append(_unmask_pairs(translation, block.placeholders))
        assembled[name] = "".join(pieces)
    return assembled


def _block_to_dict(block: PlanBlock) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": block.kind}
    if block.text:
        payload["text"] = block.text
    if block.source_hash:
        payload["source_hash"] = block.source_hash
    if block.placeholders:
        payload["placeholders"] = [[token, original] for token, original in block.placeholders]
    return payload


def _segment_to_dict(segment: SegmentSource) -> dict[str, Any]:
    return {
        "source_hash": segment.source_hash,
        "masked_text": segment.masked_text,
        "placeholder_tokens": list(segment.placeholder_tokens),
        "paragraph_count": segment.paragraph_count,
        "paragraph_markers": list(segment.paragraph_markers),
        "token_count": segment.token_count,
    }


def plan_to_json(plan: DocumentPlan) -> str:
    """Serialize a plan losslessly for storage."""
    return json.dumps(
        {
            _PLAN_KEY_VERSION: SEGMENTATION_VERSION,
            "profile": plan.profile,
            "doc_hash": plan.doc_hash,
            "parts": [
                [name, [_block_to_dict(block) for block in blocks]] for name, blocks in plan.parts
            ],
            "segments": [_segment_to_dict(segment) for segment in plan.segments],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Stored translation plan is invalid: {message}.")


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str), f"{field} must be a string")
    return value


def _string_tuple(value: Any, field: str) -> Tuple[str, ...]:
    _require(isinstance(value, list), f"{field} must be a list")
    return tuple(_string(item, field) for item in value)


def _int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _block_from_dict(value: Any) -> PlanBlock:
    _require(isinstance(value, dict), "a block must be an object")
    kind = _string(value.get("kind"), "block kind")
    _require(kind in {"protected", "segment"}, "block kind is unknown")
    placeholders = value.get("placeholders", [])
    _require(isinstance(placeholders, list), "placeholders must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in placeholders:
        _require(isinstance(pair, list) and len(pair) == 2, "a placeholder must be a pair")
        pairs.append((_string(pair[0], "placeholder token"), _string(pair[1], "placeholder original")))
    return PlanBlock(
        kind=kind,
        text=_string(value.get("text", ""), "block text"),
        source_hash=_string(value.get("source_hash", ""), "block source_hash"),
        placeholders=tuple(pairs),
    )


def _segment_from_dict(value: Any) -> SegmentSource:
    _require(isinstance(value, dict), "a segment must be an object")
    return SegmentSource(
        source_hash=_string(value.get("source_hash"), "segment source_hash"),
        masked_text=_string(value.get("masked_text"), "segment masked_text"),
        placeholder_tokens=_string_tuple(value.get("placeholder_tokens"), "placeholder_tokens"),
        paragraph_count=_int(value.get("paragraph_count"), "paragraph_count"),
        paragraph_markers=_string_tuple(value.get("paragraph_markers"), "paragraph_markers"),
        token_count=_int(value.get("token_count"), "token_count"),
    )


def plan_from_json(text: str) -> DocumentPlan:
    """Load a plan written by ``plan_to_json``.

    Raises ``ValueError`` for malformed JSON, an invalid structure, or a plan
    written by another segmentation version; callers rebuild from the source.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stored translation plan is not valid JSON.") from exc
    _require(isinstance(payload, dict), "the plan must be an object")
    _require(
        payload.get(_PLAN_KEY_VERSION) == SEGMENTATION_VERSION,
        "it was written by another segmentation version",
    )
    raw_parts = payload.get("parts")
    _require(isinstance(raw_parts, list), "parts must be a list")
    parts: list[tuple[str, Tuple[PlanBlock, ...]]] = []
    for raw_part in raw_parts:
        _require(
            isinstance(raw_part, list) and len(raw_part) == 2 and isinstance(raw_part[1], list),
            "a part must be a [name, blocks] pair",
        )
        parts.append(
            (
                _string(raw_part[0], "part name"),
                tuple(_block_from_dict(block) for block in raw_part[1]),
            )
        )
    raw_segments = payload.get("segments")
    _require(isinstance(raw_segments, list), "segments must be a list")
    return DocumentPlan(
        profile=_string(payload.get("profile"), "profile"),
        doc_hash=_string(payload.get("doc_hash"), "doc_hash"),
        parts=tuple(parts),
        segments=tuple(_segment_from_dict(segment) for segment in raw_segments),
    )


@dataclass(frozen=True)
class UnitView:
    """One unit occurrence as the translator and validator see it."""

    number: int
    part: str
    occurrence: int
    source_hash: str
    local_tokens: Tuple[str, ...]
    global_tokens: Tuple[str, ...]


def _occurrences(plan: DocumentPlan) -> list[tuple[str, PlanBlock]]:
    return [
        (name, block)
        for name, blocks in plan.parts
        for block in blocks
        if block.kind == "segment"
    ]


def build_unit_views(plan: DocumentPlan) -> list[UnitView]:
    """Number every unit occurrence and give its placeholders document-wide tokens.

    Units keep their own placeholder numbering for identity and reuse; the
    views renumber them across the document so a translator never sees the
    same token in two units.
    """
    occurrences = _occurrences(plan)
    # Masking turns every token-shaped string in the source into a placeholder,
    # so "[[M" can only appear around a unit's own tokens (for example
    # "[[Main Page]]") or inside a placeholder original.
    texts = [_PLACEHOLDER_SHAPE_RE.sub("", segment.masked_text) for segment in plan.segments]
    texts.extend(original for _, block in occurrences for _, original in block.placeholders)
    prefix = "MOTO" if any("[[M" in text for text in texts) else "M"
    views: list[UnitView] = []
    counter = itertools.count(1)
    for index, (name, block) in enumerate(occurrences):
        local = tuple(token for token, _ in block.placeholders)
        views.append(
            UnitView(
                number=index + 1,
                part=name,
                occurrence=index,
                source_hash=block.source_hash,
                local_tokens=local,
                global_tokens=tuple(f"[[{prefix}{next(counter)}]]" for _ in local),
            )
        )
    return views


def _retoken(text: str, mapping: Mapping[str, str]) -> str:
    if not mapping:
        return text
    return _PLACEHOLDER_SHAPE_RE.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def to_global(text: str, view: UnitView) -> str:
    """Rewrite a unit's own placeholder tokens as its document-wide tokens, in one pass."""
    return _retoken(text, dict(zip(view.local_tokens, view.global_tokens)))


def to_local(text: str, view: UnitView) -> str:
    """Rewrite a unit's document-wide placeholder tokens back to its own tokens, in one pass."""
    return _retoken(text, dict(zip(view.global_tokens, view.local_tokens)))


def assemble_occurrences(plan: DocumentPlan, translations: Mapping[int, str]) -> Optional[dict[str, str]]:
    """Rebuild every part from protected blocks and per-occurrence masked translations.

    ``translations`` maps occurrence indexes (``UnitView.occurrence``) to masked
    translated text with the unit's own tokens. Returns None when any
    occurrence has no translation.
    """
    assembled: dict[str, str] = {}
    occurrence = 0
    for name, blocks in plan.parts:
        pieces: list[str] = []
        for block in blocks:
            if block.kind != "segment":
                pieces.append(block.text)
                continue
            translation = translations.get(occurrence)
            occurrence += 1
            if not isinstance(translation, str):
                return None
            pieces.append(_unmask_pairs(translation, block.placeholders))
        assembled[name] = "".join(pieces)
    return assembled
