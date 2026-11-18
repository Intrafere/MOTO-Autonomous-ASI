"""Model-context handling for verified master proofs.

A verified master proof is one large Lean file. Injecting that file into every
prompt crowds out everything else, so a master proof enters model context as its
theorem statement plus a *solved template*: the scaffold steps shown as proved,
with their proofs and supporting declarations hidden.

A model that needs the hidden Lean code asks for it with a one-object JSON reply.
The shared router then runs an isolated review step in which the model sees the
expanded proof, chooses the declarations to carry forward, and receives its
original task again with those declarations added verbatim. When the review step
does not fit, other context makes room in a fixed order: the proofs list, then
papers, then brainstorms. Each is first moved to retrieval, and only cut if the
prompt still does not fit.

This module holds the pure pieces (template, registry, request and review
contracts, fit planning). It performs no model, storage, or retrieval calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable, Iterable, Literal, Optional

from backend.shared.json_parser import parse_json
from backend.shared.models import MasterProofStep, MasterProofSummary

PROOFS_BLOCK_START = "=== VERIFIED NOVEL MATHEMATICAL PROOFS (Lean 4 Verified) ==="
PROOFS_BLOCK_END = "=== END VERIFIED PROOFS ==="
EXPANSION_REQUEST_KEY = "expand_master_proof"
DEFINITIONS_SELECTOR = "definitions"
EPHEMERAL_RAG_SOURCE_PREFIX = "master_proof_expansion_"
# Evidence kept for each context kind that is moved to retrieval; matches the
# shared allocator's RAG/evidence reserve.
RETRIEVAL_EVIDENCE_TOKENS = 5000
MAX_NOTES_CHARS = 1200
MAX_REGISTERED_MASTER_PROOFS = 512
MAX_NOTED_CONTEXT_PARTS = 24
MIN_NOTED_CONTEXT_CHARS = 200

ContextKind = Literal["proofs", "papers", "brainstorms"]
# Order in which other context makes room for an expanded master proof.
CONTEXT_DROP_ORDER: tuple[ContextKind, ...] = ("proofs", "papers", "brainstorms")
_CONTEXT_LABELS: dict[str, str] = {
    "proofs": "The verified proofs list",
    "papers": "Paper content",
    "brainstorms": "Brainstorm content",
}

_DEFINITION_KINDS = {"def", "abbrev", "structure", "class", "inductive", "instance"}
_DECLARATION_NAME_RE = re.compile(
    r"\b(?:theorem|lemma|def|abbrev|structure|class|inductive|instance)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_'.!?]*|«[^»]+»)"
)
_HANDLE_RE = re.compile(r"^mp_[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# Summary of how a master proof was built
# ---------------------------------------------------------------------------


def summary_from_state_payload(payload: dict[str, Any]) -> MasterProofSummary:
    """Build the stored summary from a master proof state document."""
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), dict) else {}
    steps = [
        MasterProofStep(
            name=str(node.get("name") or name),
            signature=str(node.get("signature") or ""),
            parent=str(node.get("parent") or ""),
            depth=int(node.get("depth") or 0),
            failed_attempts=len(node.get("attempts") or []),
        )
        for name, node in nodes.items()
        if isinstance(node, dict)
    ]
    return MasterProofSummary(
        master_key=str(payload.get("master_key") or ""),
        root_theorem_name=str(payload.get("root_name") or ""),
        scaffold_revision=int(payload.get("scaffold_revision") or 0),
        total_attempts=int(payload.get("total_attempts") or 0),
        scaffold_failed_attempts=len(payload.get("scaffold_attempts") or []),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Declarations and the solved template
# ---------------------------------------------------------------------------


def parse_blocks(lean_code: str) -> list[Any]:
    # Imported on use: the scaffold parser's own imports reach the model router,
    # which imports this module.
    from backend.shared.lean_scaffold import parse_blocks as _parse_blocks

    return _parse_blocks(lean_code)


@dataclass(frozen=True)
class LeanDeclaration:
    name: str
    kind: str
    text: str


def list_declarations(lean_code: str) -> list[LeanDeclaration]:
    """Return every named top-level declaration of a Lean file, in file order."""
    declarations: list[LeanDeclaration] = []
    for block in parse_blocks(lean_code):
        name = block.name
        if not name and block.kind in _DEFINITION_KINDS:
            match = _DECLARATION_NAME_RE.search(block.text)
            name = match.group("name") if match else ""
        if name:
            declarations.append(LeanDeclaration(name=name, kind=block.kind, text=block.text.strip()))
    return declarations


def _step_names(summary: Optional[MasterProofSummary]) -> list[str]:
    return [step.name for step in (summary.steps if summary else [])]


def _root_block(blocks: list[Any], root_name: str) -> Optional[Any]:
    theorems = [block for block in blocks if block.is_theorem and block.name]
    for block in theorems:
        if root_name and block.name == root_name:
            return block
    return theorems[-1] if theorems else None


def build_solved_template(lean_code: str, summary: Optional[MasterProofSummary]) -> str:
    """Render the scaffold steps as solved, with every proof hidden."""
    return _solved_template(
        lean_code or "",
        summary.root_theorem_name if summary else "",
        tuple(_step_names(summary)),
    )


@lru_cache(maxsize=256)
def _solved_template(lean_code: str, root_name: str, step_names: tuple[str, ...]) -> str:
    blocks = parse_blocks(lean_code)
    root = _root_block(blocks, root_name)
    steps = set(step_names)
    theorems = [block for block in blocks if block.is_theorem and block.name]
    if not steps:
        # No recorded scaffold: every lemma other than the master theorem is a step.
        steps = {block.name for block in theorems if block is not root}
    lines: list[str] = []
    for block in theorems:
        if block is root or block.name not in steps:
            continue
        lines.append(f"{block.signature} := by")
        lines.append("  sorry  -- SOLVED: Lean 4 verified, proof hidden")
    if root is not None:
        lines.append(f"{root.signature} := by")
        lines.append("  ...  -- VERIFIED: proved from the solved steps above, proof hidden")
    shown = {block.name for block in theorems if block.name in steps}
    if root is not None:
        shown.add(root.name)
    hidden_lemmas = sum(1 for block in theorems if block.name not in shown)
    hidden_definitions = sum(1 for block in blocks if block.kind in _DEFINITION_KINDS)
    lines.append(
        f"-- Hidden: {hidden_definitions} definitions, {hidden_lemmas} supporting lemmas, and "
        f"every proof ({len(lean_code)} characters of Lean 4 in total)."
    )
    return "\n".join(lines)


def select_declarations(lean_code: str, names: Iterable[str]) -> tuple[list[LeanDeclaration], list[str]]:
    """Resolve requested names to declarations; returns (found, unknown names)."""
    declarations = list_declarations(lean_code)
    by_name: dict[str, LeanDeclaration] = {}
    for declaration in declarations:
        by_name.setdefault(declaration.name, declaration)
    found: list[LeanDeclaration] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for raw in names:
        name = str(raw or "").strip().strip("`")
        if not name:
            continue
        if name.lower() == DEFINITIONS_SELECTOR:
            for declaration in declarations:
                if declaration.kind in _DEFINITION_KINDS and declaration.name not in seen:
                    seen.add(declaration.name)
                    found.append(declaration)
            continue
        declaration = by_name.get(name) or by_name.get(name.rsplit(".", 1)[-1])
        if declaration is None:
            unknown.append(name)
        elif declaration.name not in seen:
            seen.add(declaration.name)
            found.append(declaration)
    order = {declaration.name: index for index, declaration in enumerate(declarations)}
    found.sort(key=lambda declaration: order.get(declaration.name, 0))
    return found, unknown


# ---------------------------------------------------------------------------
# Registry of master proofs currently offered to models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpandableMasterProof:
    handle: str
    proof_id: str
    theorem_name: str
    theorem_statement: str
    lean_code: str


_REGISTRY: "OrderedDict[str, ExpandableMasterProof]" = OrderedDict()


def master_proof_handle(lean_code: str) -> str:
    digest = hashlib.sha256((lean_code or "").encode("utf-8")).hexdigest()
    return f"mp_{digest[:12]}"


def register_master_proof(
    *, proof_id: str, theorem_name: str, theorem_statement: str, lean_code: str
) -> str:
    """Make one master proof expandable and return its prompt handle."""
    handle = master_proof_handle(lean_code)
    _REGISTRY[handle] = ExpandableMasterProof(
        handle=handle,
        proof_id=str(proof_id or ""),
        theorem_name=str(theorem_name or ""),
        theorem_statement=str(theorem_statement or ""),
        lean_code=lean_code or "",
    )
    _REGISTRY.move_to_end(handle)
    while len(_REGISTRY) > MAX_REGISTERED_MASTER_PROOFS:
        _REGISTRY.popitem(last=False)
    return handle


def lookup_master_proof(handle: str) -> Optional[ExpandableMasterProof]:
    return _REGISTRY.get(str(handle or "").strip())


def clear_registered_master_proofs() -> None:
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

MASTER_PROOF_LIST_NOTICE = (
    "[A master proof below is shown as its statement and solved template only; its Lean 4 "
    "proofs and supporting declarations are hidden. If, and only if, your task needs that "
    'hidden Lean 4 code, reply with ONLY this JSON object and nothing else: '
    '{"expand_master_proof": "<master proof id>"}. To view just part of it, add '
    '"declarations": ["<declaration name>", ...]; the name "definitions" selects every '
    "definition. You will then review the expanded proof in a separate step, choose what to "
    "carry into your task, and receive this task again with it added.]"
)


def _entry_end_marker(handle: str) -> str:
    return f"[end of master proof {handle}]"


def render_master_proof_entry(
    *,
    heading: str,
    proof_id: str,
    theorem_name: str,
    theorem_statement: str,
    lean_code: str,
    summary: Optional[MasterProofSummary],
    source_line: str = "",
) -> list[str]:
    """Render one master proof for the verified proofs list and register it."""
    handle = register_master_proof(
        proof_id=proof_id,
        theorem_name=theorem_name,
        theorem_statement=theorem_statement,
        lean_code=lean_code,
    )
    lines = [f"{heading} (MASTER PROOF, id {handle}): {theorem_statement.strip()}"]
    if source_line:
        lines.append(source_line)
    lines.extend(
        [
            "Solved master proof template (Lean 4 verified the complete proof with no `sorry` "
            "or other placeholder; each `sorry` below only marks where a verified proof is hidden):",
            build_solved_template(lean_code, summary),
            _entry_end_marker(handle),
        ]
    )
    return lines


def offered_handles(text: str) -> list[str]:
    """Return the master proof ids still offered for expansion in a prompt."""
    return [
        handle
        for handle in dict.fromkeys(re.findall(r"\[end of master proof (mp_[0-9a-f]{12})\]", text or ""))
    ]


def apply_expansion_result(
    prompt: str, handle: str, result_text: str, *, keep_offer: bool = False
) -> str:
    """Attach an expansion result to its master proof entry.

    The offer to expand that master proof is withdrawn unless ``keep_offer`` is set.
    """
    marker = _entry_end_marker(handle)
    if marker not in prompt:
        return prompt
    closed = (
        marker
        if keep_offer
        else f"[end of master proof {handle}; its expansion was already used for this task]"
    )
    return prompt.replace(marker, f"{result_text.rstrip()}\n{closed}", 1)


def withdraw_expansion_offers(prompt: str) -> str:
    """Close master proof viewing for a prompt without making it any longer.

    Used when a task is too full to take an expansion result: the notice is
    removed and every open entry marker is replaced by a closed one of the same
    length, so the model answers its task instead of asking again.
    """
    text = prompt.replace(MASTER_PROOF_LIST_NOTICE + "\n", "", 1)
    for handle in offered_handles(text):
        text = text.replace(_entry_end_marker(handle), f"[closed master proof {handle}]")
    return text


def format_carried_declarations(
    handle: str, declarations: list[LeanDeclaration], notes: str = "", omitted: int = 0
) -> str:
    if not declarations:
        text = (
            f"Expansion of {handle}: you reviewed its hidden Lean 4 code for this task and "
            "carried nothing forward."
        )
    else:
        body = "\n\n".join(declaration.text for declaration in declarations)
        text = (
            f"Expanded context from {handle} (verbatim Lean 4 declarations you chose to carry "
            f"forward for this task):\n{body}"
        )
    if omitted:
        text += (
            f"\n[{omitted} further chosen declaration(s) were left out because they do not fit "
            "this role's context.]"
        )
    if notes.strip():
        text += f"\nYour review note: {notes.strip()[:MAX_NOTES_CHARS]}"
    return text


def format_unviewable(handle: str, *, partial: bool) -> str:
    if partial:
        return (
            f"Expansion of {handle}: the requested declarations do not fit this role's context "
            "even after other context was removed, so nothing was added."
        )
    return (
        f"Expansion of {handle}: its full Lean 4 code does not fit this role's context even "
        "after other context was removed. You may still request specific declarations of it "
        'by name with {"expand_master_proof": "' + handle + '", "declarations": ["<name>", ...]}.'
    )


# ---------------------------------------------------------------------------
# Expansion request and review contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionRequest:
    handle: str
    declarations: tuple[str, ...] = ()


def parse_expansion_request(text: str) -> Optional[ExpansionRequest]:
    """Return the expansion request when a reply is nothing but that request."""
    if EXPANSION_REQUEST_KEY not in (text or ""):
        return None
    try:
        payload = parse_json(text)
    except Exception:
        return None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict) or EXPANSION_REQUEST_KEY not in payload:
        return None
    if set(payload) - {EXPANSION_REQUEST_KEY, "declarations", "reason", "reasoning"}:
        # A real role answer that merely mentions the key is not an expansion request.
        return None
    handle = str(payload.get(EXPANSION_REQUEST_KEY) or "").strip()
    if not _HANDLE_RE.match(handle):
        return None
    raw = payload.get("declarations")
    names = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) and raw.strip() else [])
    return ExpansionRequest(
        handle=handle,
        declarations=tuple(str(name).strip() for name in names if str(name or "").strip()),
    )


@dataclass(frozen=True)
class ReviewSelection:
    keep: tuple[str, ...] = ()
    notes: str = ""


def parse_review_selection(text: str) -> ReviewSelection:
    """Parse the review step's reply; raises ValueError on a contract failure."""
    payload = parse_json(text)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict) or "keep_declarations" not in payload:
        raise ValueError("review reply is missing keep_declarations")
    keep = payload.get("keep_declarations")
    if not isinstance(keep, list):
        raise ValueError("keep_declarations must be a list")
    return ReviewSelection(
        keep=tuple(str(name).strip() for name in keep if str(name or "").strip()),
        notes=str(payload.get("notes") or "").strip()[:MAX_NOTES_CHARS],
    )


REVIEW_JSON_REMINDER = (
    'Respond with ONLY this JSON object: {"keep_declarations": ["exact_declaration_name", ...], '
    '"notes": "optional short note on how the kept declarations help your task"}'
)


def build_review_prompt(*, handle: str, expanded_code: str, task_text: str, partial: bool) -> str:
    """Build the isolated review step shown after a model asks to expand a master proof."""
    scope = "the declarations you requested from" if partial else "the full Lean 4 code of"
    return f"""MASTER PROOF EXPANSION REVIEW

You asked to view hidden Lean 4 code of the verified master proof {handle} before answering your task. This is a separate review step. Below is {scope} that master proof. Read it and choose which of its top-level declarations should be carried, verbatim, into your task's context. You will then receive your original task again with exactly those declarations added. Do not answer the original task in this step.

{REVIEW_JSON_REMINDER}

Rules:
- Use the exact names of top-level declarations that appear in the EXPANDED MASTER PROOF. Unknown names are ignored.
- Keep only what your task actually needs. An empty list is correct when the solved template was already enough.
- The system copies kept declarations verbatim. Never retype Lean code in your reply.

YOUR ORIGINAL TASK (reference only; part of its context may have been moved to retrieval or removed to make room for the expanded proof):
<<<ORIGINAL TASK
{task_text}
ORIGINAL TASK>>>

EXPANDED MASTER PROOF {handle} (Lean 4 verified):
{expanded_code}

{REVIEW_JSON_REMINDER}
"""


# ---------------------------------------------------------------------------
# Context that may make room for an expanded master proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextPart:
    kind: ContextKind
    text: str


_NOTED_PARTS: ContextVar[tuple[ContextPart, ...]] = ContextVar(
    "moto_master_proof_droppable_context", default=()
)


def note_context_part(kind: ContextKind, text: str) -> None:
    """Record direct paper or brainstorm context used by the current prompt.

    Prompt builders call this for text they inject verbatim. The record is only
    read if a model later expands a master proof, to decide what can make room.
    """
    if kind not in CONTEXT_DROP_ORDER or not text or len(text) < MIN_NOTED_CONTEXT_CHARS:
        return
    parts = tuple(part for part in _NOTED_PARTS.get() if part.text != text)
    _NOTED_PARTS.set((*parts[-(MAX_NOTED_CONTEXT_PARTS - 1):], ContextPart(kind=kind, text=text)))


def noted_context_parts() -> tuple[ContextPart, ...]:
    return _NOTED_PARTS.get()


def kind_for_source_type(source_type: str) -> ContextKind:
    return "papers" if str(source_type or "").strip().lower() == "paper" else "brainstorms"


def proofs_list_part(text: str) -> Optional[ContextPart]:
    """Return the verified proofs list found in a prompt, if any."""
    start = (text or "").find(PROOFS_BLOCK_START)
    if start < 0:
        return None
    end = text.find(PROOFS_BLOCK_END, start)
    if end < 0:
        return None
    return ContextPart(kind="proofs", text=text[start:end + len(PROOFS_BLOCK_END)])


@dataclass
class ReviewFit:
    """Outcome of fitting a review prompt; ``prompt`` is None when nothing fits."""

    prompt: Optional[str]
    actions: list[str] = field(default_factory=list)


RetrieveEvidence = Callable[[ContextKind, str, int], Awaitable[str]]


def _moved_notice(kind: ContextKind, evidence: str) -> str:
    label = _CONTEXT_LABELS[kind]
    return (
        f"[{label} was moved to retrieval to make room for the expanded master proof. "
        f"Most relevant excerpts:]\n{evidence}"
    )


def _cut_notice(kind: ContextKind) -> str:
    return f"[{_CONTEXT_LABELS[kind]} was removed to make room for the expanded master proof.]"


async def fit_review_prompt(
    *,
    task_text: str,
    parts: Iterable[ContextPart],
    build: Callable[[str], str],
    fits: Callable[[str], bool],
    count_tokens: Callable[[str], int],
    retrieve: RetrieveEvidence,
) -> ReviewFit:
    """Fit the review step, making room in the fixed proofs → papers → brainstorms order.

    Every kind is first moved to retrieval, in order, until the prompt fits. Only
    if it still does not fit are the retrieved excerpts cut, in the same order.
    """
    result = ReviewFit(prompt=None)

    async def prompt_fits(candidate: str) -> bool:
        # Counting a large prompt is CPU work; keep it off the event loop.
        return await asyncio.to_thread(fits, candidate)

    prompt = build(task_text)
    if await prompt_fits(prompt):
        result.prompt = prompt
        return result

    present: dict[str, list[ContextPart]] = {kind: [] for kind in CONTEXT_DROP_ORDER}
    for part in parts:
        if part.kind in present and part.text and part.text in task_text:
            if all(part.text != other.text for other in present[part.kind]):
                present[part.kind].append(part)

    # Nothing can fit if the prompt overflows even with all of this context cut.
    bare = task_text
    for kind in CONTEXT_DROP_ORDER:
        for index, part in enumerate(present[kind]):
            bare = bare.replace(part.text, _cut_notice(kind) if index == 0 else "", 1)
    if not await prompt_fits(build(bare)):
        return result

    current = task_text
    moved: dict[str, str] = {}
    for kind in CONTEXT_DROP_ORDER:
        kind_parts = present[kind]
        if not kind_parts:
            continue
        combined = "\n\n".join(part.text for part in kind_parts)
        budget = min(RETRIEVAL_EVIDENCE_TOKENS, await asyncio.to_thread(count_tokens, combined))
        evidence = (await retrieve(kind, combined, budget)).strip() if budget > 0 else ""
        notice = _moved_notice(kind, evidence) if evidence else _cut_notice(kind)
        for index, part in enumerate(kind_parts):
            current = current.replace(part.text, notice if index == 0 else "", 1)
        if evidence:
            moved[kind] = notice
            result.actions.append(f"{kind}:retrieval")
        else:
            result.actions.append(f"{kind}:cut")
        prompt = build(current)
        if await prompt_fits(prompt):
            result.prompt = prompt
            return result

    for kind in CONTEXT_DROP_ORDER:
        notice = moved.get(kind)
        if not notice:
            continue
        current = current.replace(notice, _cut_notice(kind), 1)
        result.actions[result.actions.index(f"{kind}:retrieval")] = f"{kind}:cut"
        prompt = build(current)
        if await prompt_fits(prompt):
            result.prompt = prompt
            return result
    return result
