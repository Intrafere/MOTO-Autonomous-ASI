"""Canonical parsing and metrics for papers with appended Lean proof sections."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


THEOREMS_APPENDIX_START = (
    "[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]"
)
THEOREMS_APPENDIX_END = (
    "[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]"
)

_LEGACY_HEADER_RE = re.compile(
    r"(?im)^=== PROOFS (?:GENERATED FROM|ATTACHED TO) THIS PAPER"
    r"(?: \(Lean 4 Verified\))? ===[ \t]*(?:\r?\n|$)"
)
_SELF_REVIEW_RE = re.compile(
    r"(?im)^(?:[ \t]*#{1,6}[ \t]*)?AI Self-Review and Limitations[ \t]*(?:\r?\n|$)"
)
_VOLUME_CHAPTER_RE = re.compile(
    r"(?m)^#{20,}[ \t]*\r?\n# CHAPTER \d+:[^\n]*(?:\r?\n|$)"
)
_MODEL_CREDITS_RE = re.compile(r"(?im)^={20,}\r?\nMODEL CREDITS[ \t]*(?:\r?\n|$)")
_THEOREM_ENTRY_RE = re.compile(r"(?im)^Theorem[ \t]*\(([^)\r\n]+)\)[^\r\n]*")


@dataclass(frozen=True)
class PaperProof:
    """One displayable proof in source order."""

    order: int
    title: str
    content: str

    def as_dict(self) -> Dict[str, Any]:
        return {"order": self.order, "title": self.title, "content": self.content}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _find_proof_ranges(content: str) -> List[Tuple[int, int, str]]:
    """Return non-overlapping proof-section ranges and their format."""
    candidates: List[Tuple[int, int, str]] = []

    cursor = 0
    while True:
        start = content.find(THEOREMS_APPENDIX_START, cursor)
        if start < 0:
            break
        end_marker = content.find(THEOREMS_APPENDIX_END, start + len(THEOREMS_APPENDIX_START))
        if end_marker < 0:
            break
        end = end_marker + len(THEOREMS_APPENDIX_END)
        if end < len(content) and content[end] == "\r":
            end += 1
        if end < len(content) and content[end] == "\n":
            end += 1
        candidates.append((start, end, "theorems_appendix"))
        cursor = end

    legacy_matches = list(_LEGACY_HEADER_RE.finditer(content))
    for index, match in enumerate(legacy_matches):
        end_candidates = [
            boundary.start()
            for pattern in (_SELF_REVIEW_RE, _VOLUME_CHAPTER_RE, _MODEL_CREDITS_RE)
            if (boundary := pattern.search(content, match.end())) is not None
        ]
        if index + 1 < len(legacy_matches):
            end_candidates.append(legacy_matches[index + 1].start())
        next_appendix = content.find(THEOREMS_APPENDIX_START, match.end())
        if next_appendix >= 0:
            end_candidates.append(next_appendix)
        end = min(end_candidates) if end_candidates else len(content)
        candidates.append((match.start(), end, "legacy"))

    ranges: List[Tuple[int, int, str]] = []
    for start, end, section_format in sorted(candidates):
        if ranges and start < ranges[-1][1]:
            continue
        ranges.append((start, end, section_format))
    return ranges


def _entry_title(header: str, proof_id: str) -> str:
    suffix = header.split(" - ", 1)
    if len(suffix) == 2 and suffix[1].strip():
        return suffix[1].strip()
    return proof_id.strip() or header.strip()


def _parse_entries(section: str, section_format: str, next_order: int) -> List[PaperProof]:
    body = section
    if section_format == "theorems_appendix":
        body = body.replace(THEOREMS_APPENDIX_START, "", 1)
        end_at = body.rfind(THEOREMS_APPENDIX_END)
        if end_at >= 0:
            body = body[:end_at]
    else:
        body = _LEGACY_HEADER_RE.sub("", body, count=1)
    body = body.strip()
    if not body:
        return []

    matches = list(_THEOREM_ENTRY_RE.finditer(body))
    if not matches:
        first_line = next((line.strip(" #=\t") for line in body.splitlines() if line.strip()), "Lean 4 proof")
        return [PaperProof(order=next_order, title=first_line, content=body)]

    proofs: List[PaperProof] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        entry = body[match.start():end].strip()
        entry = re.sub(r"(?:\r?\n)?---[ \t]*$", "", entry).strip()
        proofs.append(
            PaperProof(
                order=next_order + index,
                title=_entry_title(match.group(0), match.group(1)),
                content=entry,
            )
        )
    return proofs


def analyze_paper_content(content: str) -> Dict[str, Any]:
    """Split proof sections from prose and return canonical display metrics."""
    text = content or ""
    ranges = _find_proof_ranges(text)
    proofs: List[PaperProof] = []
    paper_parts: List[str] = []
    proof_parts: List[str] = []
    cursor = 0

    for start, end, section_format in ranges:
        paper_parts.append(text[cursor:start])
        section = text[start:end]
        proof_parts.append(section)
        proofs.extend(_parse_entries(section, section_format, len(proofs) + 1))
        cursor = end
    paper_parts.append(text[cursor:])

    paper_text = "".join(paper_parts)
    proof_text = "".join(proof_parts)
    return {
        "paper_content": paper_text,
        "proofs": [proof.as_dict() for proof in proofs],
        "proof_count": len(proofs),
        "paper_word_count": _word_count(paper_text),
        "proof_word_count": _word_count(proof_text),
        "total_word_count": _word_count(text),
        "paper_character_count": len(paper_text),
        "proof_character_count": len(proof_text),
        "total_character_count": len(text),
    }


def strip_paper_proofs(content: str, *, preserve_appendix_markers: bool = True) -> str:
    """Remove every proof section while retaining surrounding paper/volume content."""
    text = content or ""
    ranges = _find_proof_ranges(text)
    if not ranges:
        return text.rstrip()

    parts: List[str] = []
    cursor = 0
    empty_appendix = (
        f"{THEOREMS_APPENDIX_START}\n"
        "[Theorems appendix - verified Lean 4 theorems not placed inline will appear here]\n"
        f"{THEOREMS_APPENDIX_END}"
    )
    for start, end, section_format in ranges:
        parts.append(text[cursor:start])
        if section_format == "theorems_appendix" and preserve_appendix_markers:
            parts.append(empty_appendix)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts).rstrip()


def paper_metric_fields(content: str) -> Dict[str, int]:
    """Return only persisted/public scalar metric fields."""
    analysis = analyze_paper_content(content)
    return {
        key: analysis[key]
        for key in (
            "proof_count",
            "paper_word_count",
            "proof_word_count",
            "total_word_count",
            "paper_character_count",
            "proof_character_count",
            "total_character_count",
        )
    }
