"""Deterministic structure checks for Lean 4 master proof scaffolds.

A scaffold is one Lean file whose unproved obligations are top-level
``theorem``/``lemma`` declarations that still contain ``sorry``. These helpers
split a file into top-level declaration blocks, list open obligations, and
splice a replacement for one obligation. They never run Lean; Lean remains the
authority on whether a scaffold or a proof is accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.shared.lean_proof_integrity import strip_lean_comments_and_strings

_NAME = r"(?:[A-Za-z_][A-Za-z0-9_'.!?]*|«[^»]+»)"
_MODIFIERS = (
    r"(?:(?:private|protected|noncomputable|unsafe|partial|nonrec|local|scoped)\s+)*"
)
_ATTRIBUTES = r"(?:@\[[^\]]*\]\s*)*"
_COMMAND_KEYWORDS = (
    "theorem|lemma|def|abbrev|instance|structure|class|inductive|example|axiom|"
    "opaque|constant|namespace|section|end|open|variable|variables|universe|"
    "universes|import|set_option|macro|macro_rules|syntax|notation|infix|infixl|"
    "infixr|prefix|postfix|attribute|mutual|elab|export|initialize|deriving"
)
_COMMAND_START_RE = re.compile(
    rf"^{_ATTRIBUTES}{_MODIFIERS}(?P<kind>{_COMMAND_KEYWORDS})\b"
)
_THEOREM_NAME_RE = re.compile(rf"\b(?:theorem|lemma)\s+(?P<name>{_NAME})")
_PREFIX_START_RE = re.compile(r"^(?:/-|--|@\[|#)")
# Same tokens the Lean client treats as placeholder proofs.
_PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9_'])(?:sorry|admit)(?![A-Za-z0-9_'])")
_SORRY_WARNING_RE = re.compile(r":(\d+):\d+:\s*warning:[^\n]*\bsorry\b", re.IGNORECASE)
_PROVABLE_KINDS = {"theorem", "lemma", "example"}
_THEOREM_KINDS = {"theorem", "lemma"}


@dataclass(frozen=True)
class LeanBlock:
    """One top-level command of a Lean file, with any doc comment or attributes."""

    kind: str
    name: str
    signature: str
    text: str
    has_placeholder: bool
    start_line: int
    end_line: int

    @property
    def is_theorem(self) -> bool:
        return self.kind in _THEOREM_KINDS


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def _signature(stripped_block: str) -> str:
    """Return the declaration header up to its top-level ``:=`` or first ``|`` arm."""
    match = re.search(r"\b(?:theorem|lemma)\b", stripped_block)
    if not match:
        return ""
    text = stripped_block[match.start():]
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{⟨":
            depth += 1
        elif char in ")]}⟩":
            depth = max(0, depth - 1)
        elif depth == 0:
            if text.startswith(":=", index):
                return _normalize(text[:index])
            if char == "\n" and text[index + 1:].lstrip(" \t").startswith("|"):
                return _normalize(text[:index])
        index += 1
    return _normalize(text)


def _build_block(lines: list[str], start_line: int) -> LeanBlock:
    text = "\n".join(lines)
    stripped = strip_lean_comments_and_strings(text)
    kind = ""
    for line in stripped.splitlines():
        match = _COMMAND_START_RE.match(line)
        if match:
            kind = match.group("kind")
            break
    name = ""
    signature = ""
    if kind in _THEOREM_KINDS:
        name_match = _THEOREM_NAME_RE.search(stripped)
        name = name_match.group("name") if name_match else ""
        signature = _signature(stripped)
    return LeanBlock(
        kind=kind,
        name=name,
        signature=signature,
        text=text,
        has_placeholder=bool(_PLACEHOLDER_RE.search(stripped)),
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
    )


def parse_blocks(code: str) -> list[LeanBlock]:
    """Split Lean source into top-level command blocks, in file order."""
    lines = (code or "").replace("\r\n", "\n").split("\n")
    blocks: list[LeanBlock] = []
    current: list[str] = []
    current_start = 0
    prefix_only = False
    comment_depth = 0

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(_build_block(current, current_start))
        current = []

    for index, line in enumerate(lines):
        at_top_level = comment_depth == 0 and bool(line) and not line[0].isspace()
        if at_top_level:
            is_command = bool(_COMMAND_START_RE.match(line))
            is_prefix = not is_command and bool(_PREFIX_START_RE.match(line))
            if is_command:
                if current and prefix_only:
                    prefix_only = False
                else:
                    flush()
                    current_start = index
                    prefix_only = False
            elif is_prefix and not (current and prefix_only):
                flush()
                current_start = index
                prefix_only = True
        if not current:
            current_start = index
        current.append(line)
        without_line_comment = re.sub(r"--.*$", "", line) if comment_depth == 0 else line
        comment_depth += without_line_comment.count("/-") - without_line_comment.count("-/")
        comment_depth = max(0, comment_depth)
    flush()
    return blocks


def open_obligations(code: str) -> list[LeanBlock]:
    """Return every top-level theorem/lemma that still contains a placeholder."""
    return [
        block
        for block in parse_blocks(code)
        if block.is_theorem and block.name and block.has_placeholder
    ]


def find_theorem(code: str, name: str) -> Optional[LeanBlock]:
    for block in parse_blocks(code):
        if block.is_theorem and block.name == name:
            return block
    return None


def resolve_theorem_name(code: str, name: str) -> str:
    """Return the name ``name`` is declared under in ``code``.

    A theorem written inside ``namespace A.B`` is declared by its short name, while
    a model may report it fully qualified (``A.B.thm``). When exactly one top-level
    theorem matches the qualified name's suffix, that declared name is returned;
    otherwise ``name`` is returned unchanged.
    """
    wanted = (name or "").strip()
    declared = [block.name for block in parse_blocks(code) if block.is_theorem and block.name]
    if not wanted or wanted in declared:
        return wanted
    matches = [item for item in declared if wanted.endswith("." + item)]
    return matches[0] if len(matches) == 1 else wanted


def scaffold_structure_problem(
    code: str,
    root_name: str,
    *,
    required_root_signature: str = "",
) -> str:
    """Return why ``code`` is not a usable scaffold, or an empty string."""
    if not (code or "").strip():
        return "SCAFFOLD REJECTED: no Lean 4 code was provided."
    blocks = parse_blocks(code)
    theorem_names = [block.name for block in blocks if block.is_theorem and block.name]
    duplicates = sorted({name for name in theorem_names if theorem_names.count(name) > 1})
    if duplicates:
        return (
            "SCAFFOLD REJECTED: top-level theorem/lemma names must be unique. "
            f"Duplicated: {', '.join(duplicates[:8])}."
        )
    stray = [block for block in blocks if block.has_placeholder and not (block.is_theorem and block.name)]
    if stray:
        return (
            "SCAFFOLD REJECTED: every `sorry` must be the proof of a named top-level "
            "`theorem` or `lemma` written at the start of a line. A placeholder was found "
            f"in another declaration near line {stray[0].start_line + 1}."
        )
    root = next((block for block in blocks if block.is_theorem and block.name == root_name), None)
    if root is None:
        return (
            f"SCAFFOLD REJECTED: the master theorem `{root_name}` was not found as a "
            "top-level `theorem` or `lemma` written at the start of a line."
        )
    provable = [block for block in blocks if block.kind in _PROVABLE_KINDS]
    if provable[-1] is not root:
        return (
            f"SCAFFOLD REJECTED: the master theorem `{root_name}` must be the last "
            "`theorem`/`lemma`/`example` in the file. Put every lemma and helper before it."
        )
    if required_root_signature and root.signature != required_root_signature:
        return (
            f"SCAFFOLD REJECTED: the statement of the master theorem `{root_name}` was "
            "changed. Its name and statement must stay exactly as they were:\n"
            f"{required_root_signature}"
        )
    return ""


def declarations_with_sorry_warning(checked_source: str, diagnostics: str) -> set[str]:
    """Names of theorems/lemmas whose own proof Lean reported as using `sorry`.

    Lean prints one positioned warning per declaration that elaborates with
    `sorry`, however the placeholder was written. Positions refer to the
    source Lean actually checked.
    """
    lines = {int(match.group(1)) for match in _SORRY_WARNING_RE.finditer(diagnostics or "")}
    if not lines:
        return set()
    return {
        block.name
        for block in parse_blocks(checked_source)
        if block.is_theorem
        and block.name
        and any(block.start_line + 1 <= line <= block.end_line + 1 for line in lines)
    }


@dataclass(frozen=True)
class ObligationReplacement:
    """Result of planning the replacement of one open obligation."""

    problem: str = ""
    code: str = ""
    new_obligations: tuple[LeanBlock, ...] = ()


def plan_obligation_replacement(
    code: str,
    name: str,
    signature: str,
    replacement: str,
) -> ObligationReplacement:
    """Splice ``replacement`` over obligation ``name`` after structural checks.

    The replacement must restate ``name`` with the same statement and a
    placeholder-free proof. It may add helper declarations; new theorem/lemma
    helpers that contain a placeholder become new open obligations.
    """
    target = find_theorem(code, name)
    if target is None:
        return ObligationReplacement(
            problem=f"SUBMISSION REJECTED: obligation `{name}` is no longer in the scaffold."
        )
    new_blocks = parse_blocks(replacement or "")
    restated = [block for block in new_blocks if block.is_theorem and block.name == name]
    if len(restated) != 1:
        return ObligationReplacement(
            problem=(
                f"SUBMISSION REJECTED: lean_code must contain the declaration `{name}` "
                "exactly once as a top-level `theorem` or `lemma` written at the start of a line."
            )
        )
    if restated[0].signature != signature:
        return ObligationReplacement(
            problem=(
                f"SUBMISSION REJECTED: the statement of `{name}` was changed. Keep its name "
                f"and statement exactly:\n{signature}\n"
                "To change a statement, use action \"change_scaffold\"."
            )
        )
    if restated[0].has_placeholder:
        return ObligationReplacement(
            problem=(
                f"SUBMISSION REJECTED: the proof of `{name}` still contains `sorry`/`admit`. "
                f"Prove `{name}` itself without a placeholder; put any remaining gap in a new "
                "top-level helper `theorem`/`lemma` whose proof is `sorry`."
            )
        )
    if any(block.kind == "import" for block in new_blocks):
        return ObligationReplacement(
            problem=(
                "SUBMISSION REJECTED: lean_code replaces one declaration inside the scaffold "
                "and must not contain `import` lines. Use action \"change_scaffold\" to change imports."
            )
        )
    stray = [block for block in new_blocks if block.has_placeholder and not (block.is_theorem and block.name)]
    if stray:
        return ObligationReplacement(
            problem=(
                "SUBMISSION REJECTED: every `sorry` must be the proof of a named top-level "
                "`theorem` or `lemma` written at the start of a line."
            )
        )
    existing_names = {
        block.name
        for block in parse_blocks(code)
        if block.is_theorem and block.name and block.name != name
    }
    added_names = [block.name for block in new_blocks if block.is_theorem and block.name and block.name != name]
    clashes = sorted(
        {added for added in added_names if added in existing_names or added_names.count(added) > 1}
    )
    if clashes:
        return ObligationReplacement(
            problem=(
                "SUBMISSION REJECTED: new helper names must not repeat names already in the "
                f"scaffold: {', '.join(clashes[:8])}."
            )
        )
    lines = code.replace("\r\n", "\n").split("\n")
    replacement_lines = (replacement or "").replace("\r\n", "\n").strip("\n").split("\n")
    spliced = lines[: target.start_line] + replacement_lines + [""] + lines[target.end_line + 1:]
    return ObligationReplacement(
        code="\n".join(spliced),
        new_obligations=tuple(
            block for block in new_blocks if block.is_theorem and block.name != name and block.has_placeholder
        ),
    )
