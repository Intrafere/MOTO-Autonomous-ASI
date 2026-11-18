"""Runs master proof expansion requests for the shared model router.

A model that sees a master proof's solved template may reply with nothing but an
expansion request. The router hands that reply here: this module runs the
isolated review step, copies the declarations the model chose into the original
prompt, and re-issues the original call. Expansion is optional context, so every
failure here falls back to the unexpanded prompt instead of stopping a workflow.

Parsing large Lean files and counting large prompts are CPU work and run in
worker threads so the event loop keeps serving the rest of the app.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend.shared.config import rag_config
from backend.shared.json_parser import sanitize_model_output_for_retry_context
from backend.shared.master_proof_context import (
    EPHEMERAL_RAG_SOURCE_PREFIX,
    PROOFS_BLOCK_START,
    REVIEW_JSON_REMINDER,
    ContextKind,
    ExpandableMasterProof,
    ExpansionRequest,
    LeanDeclaration,
    ReviewSelection,
    apply_expansion_result,
    build_review_prompt,
    fit_review_prompt,
    format_carried_declarations,
    format_unviewable,
    lookup_master_proof,
    noted_context_parts,
    offered_handles,
    parse_expansion_request,
    parse_review_selection,
    proofs_list_part,
    select_declarations,
    withdraw_expansion_offers,
)
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

MASTER_PROOF_EXPANSION_EVENT = "master_proof_context_expanded"
_RETRIEVAL_CHUNK_SIZE = 512
_QUERY_MAX_CHARS = 2000
_EVENT_NAME_LIMIT = 40

Messages = list[dict[str, Any]]


@dataclass
class ExpansionCall:
    """What the router provides for one role call that may expand master proofs."""

    task_id: str
    role_id: str
    model: str
    messages: Messages
    generate: Callable[[Messages], Awaitable[dict[str, Any]]]
    generate_review: Callable[[Messages], Awaitable[dict[str, Any]]]
    fits: Callable[[str], bool]
    response_text: Callable[[dict[str, Any]], str]
    broadcast: Callable[[str, dict[str, Any]], Awaitable[None]]
    route_fields: dict[str, Any] = field(default_factory=dict)


def offers_master_proof_expansion(messages: Messages) -> bool:
    """Cheap pre-check: does any message carry an expandable master proof?"""
    return any(
        PROOFS_BLOCK_START in str(message.get("content") or "")
        and offered_handles(str(message.get("content") or ""))
        for message in messages
        if isinstance(message.get("content"), str)
    )


def _index_has_room(manager: Any, text: str) -> bool:
    """True when indexing ``text`` cannot push real content out of the shared index.

    The index trims its oldest chunks once a size bucket is full. A one-off source
    must never cause that, so it is only indexed with clear headroom (the chunk
    estimate is doubled to stay on the safe side of the chunker).
    """
    estimated_chunks = 2 * (len(text) // _RETRIEVAL_CHUNK_SIZE + 1)
    existing = len(getattr(manager, "chunks_by_size", {}).get(_RETRIEVAL_CHUNK_SIZE, []))
    documents = int(getattr(manager, "document_count", 0))
    return (
        existing + estimated_chunks <= rag_config.max_chunks_per_size
        and documents + 1 <= rag_config.max_documents
    )


async def retrieve_context_evidence(kind: ContextKind, text: str, budget_tokens: int, query: str) -> str:
    """Move one kind of context to retrieval and return its most relevant excerpts.

    The text is indexed under a reserved one-off source that only a retrieval
    naming it can read, then removed again. When the shared index has no spare
    room, nothing is indexed and the caller cuts that context instead.
    """
    if budget_tokens <= 0 or not text.strip():
        return ""
    import backend.aggregator.core.rag_manager as rag_module

    manager = rag_module.rag_manager
    if not _index_has_room(manager, text):
        logger.info(
            "Master proof expansion skipped retrieval of %s context: the shared index has no spare room",
            kind,
        )
        return ""
    source = f"{EPHEMERAL_RAG_SOURCE_PREFIX}{kind}_{uuid.uuid4().hex}"
    try:
        await manager.add_text(text, source, chunk_sizes=[_RETRIEVAL_CHUNK_SIZE], is_permanent=False)
        pack = await manager.retrieve(
            query,
            chunk_size=_RETRIEVAL_CHUNK_SIZE,
            max_tokens=budget_tokens,
            include_sources=[source],
        )
        return str(getattr(pack, "text", "") or "")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Master proof expansion could not retrieve %s context; it will be cut instead (%s)",
            kind,
            type(exc).__name__,
        )
        return ""
    finally:
        try:
            await asyncio.shield(manager.remove_document(source))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Could not remove one-off retrieval source: %s", type(exc).__name__)


def _prompt_message_index(messages: Messages, handle: str) -> int | None:
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, str) and handle in offered_handles(content):
            return index
    return None


def _joined(messages: Messages) -> str:
    return "\n\n".join(str(message.get("content") or "") for message in messages)


def _with_content(messages: Messages, index: int, content: str) -> Messages:
    return [
        {**message, "content": content} if position == index else message
        for position, message in enumerate(messages)
    ]


async def _fits(call: ExpansionCall, messages: Messages) -> bool:
    return await asyncio.to_thread(call.fits, _joined(messages))


async def _review_selection(call: ExpansionCall, prompt: str) -> ReviewSelection:
    """Run the review step; one bounded repair retry, then carry nothing."""
    messages: Messages = [{"role": "user", "content": prompt}]
    response = await call.generate_review(messages)
    text = call.response_text(response)
    try:
        return parse_review_selection(text)
    except Exception:
        visible = sanitize_model_output_for_retry_context(text, max_chars=1200)
        retry: Messages = [
            *messages,
            {"role": "assistant", "content": visible or "[no visible reply]"},
            {
                "role": "user",
                "content": f"Your reply did not match the required JSON. {REVIEW_JSON_REMINDER}",
            },
        ]
        if not await _fits(call, retry):
            return ReviewSelection()
        repaired = call.response_text(await call.generate_review(retry))
        try:
            return parse_review_selection(repaired)
        except Exception:
            return ReviewSelection()


async def _carry_declarations(
    call: ExpansionCall,
    messages: Messages,
    index: int,
    task_text: str,
    handle: str,
    kept: list[LeanDeclaration],
    notes: str,
) -> tuple[str | None, int]:
    """Return the re-issued prompt and how many chosen declarations it carries.

    Declarations that do not fit are dropped whole, last first. The largest
    prefix that fits is found by bisection so a long selection costs few counts.
    Returns ``(None, 0)`` when not even the bare result line fits.
    """

    def render(count: int) -> str:
        text = format_carried_declarations(handle, kept[:count], notes, len(kept) - count)
        return apply_expansion_result(task_text, handle, text)

    async def fits(count: int) -> bool:
        return await _fits(call, _with_content(messages, index, render(count)))

    if await fits(len(kept)):
        return render(len(kept)), len(kept)
    best = -1
    low, high = 0, len(kept) - 1
    while low <= high:
        middle = (low + high) // 2
        if await fits(middle):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < 0:
        return None, 0
    return render(best), best


def _describe_actions(actions: list[str]) -> str:
    labels = {"proofs": "the proofs list", "papers": "papers", "brainstorms": "brainstorms"}
    described = []
    for action in actions:
        kind, _, mode = action.partition(":")
        described.append(
            f"{labels.get(kind, kind)} {'moved to retrieval' if mode == 'retrieval' else 'removed'}"
        )
    return ", ".join(described)


async def _announce(
    call: ExpansionCall,
    target: ExpandableMasterProof,
    request: ExpansionRequest,
    *,
    outcome: str,
    kept: list[LeanDeclaration],
    omitted: int,
    actions: list[str],
) -> None:
    name = target.theorem_name or target.proof_id or target.handle
    if outcome == "too_large":
        sentence = (
            f"wanted to view master proof `{name}`, but it does not fit this role's context "
            "even with other context removed."
        )
    elif outcome == "unknown_declarations":
        sentence = f"asked for declarations of master proof `{name}` that do not exist in it."
    elif outcome == "no_room":
        sentence = (
            f"viewed master proof `{name}`, but its task has no room left to carry anything "
            "back, so master proof viewing was closed for this task."
        )
    elif kept:
        sentence = (
            f"viewed master proof `{name}` and carried {len(kept)} declaration"
            f"{'' if len(kept) == 1 else 's'} into its task."
        )
    else:
        sentence = f"viewed master proof `{name}` and carried nothing into its task."
    if actions:
        sentence += f" To make room for the view: {_describe_actions(actions)}."
    payload = {
        **call.route_fields,
        # Each expansion is its own Live Activity row, even within one role call.
        "event_id": f"{MASTER_PROOF_EXPANSION_EVENT}:{call.task_id}:{target.handle}:{uuid.uuid4().hex[:12]}",
        "role_id": call.role_id,
        "task_id": call.task_id,
        "model": call.model,
        "master_proof_id": target.handle,
        "proof_id": target.proof_id,
        "theorem_name": target.theorem_name,
        "outcome": outcome,
        "requested_declarations": list(request.declarations)[:_EVENT_NAME_LIMIT],
        "kept_declarations": [declaration.name for declaration in kept][:_EVENT_NAME_LIMIT],
        "kept_count": len(kept),
        "omitted_count": omitted,
        "context_actions": list(actions),
        "message": f"Master proof expansion: the model {sentence}",
    }
    try:
        await call.broadcast(MASTER_PROOF_EXPANSION_EVENT, payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Master proof expansion activity was not published: %s", type(exc).__name__)


async def resolve_master_proof_expansions(call: ExpansionCall, response: dict[str, Any]) -> dict[str, Any]:
    """Serve expansion requests until the model gives its real answer.

    Each master proof can be viewed once per call; one that is too large to view
    in full can be requested once more by declaration name. Any other repeat is
    returned to the caller unchanged, where the role's own contract handling applies.
    """
    messages = call.messages
    served: dict[str, str] = {}
    while True:
        try:
            request = parse_expansion_request(call.response_text(response))
        except Exception:
            return response
        if request is None:
            return response
        target = lookup_master_proof(request.handle)
        index = _prompt_message_index(messages, request.handle)
        if target is None or index is None:
            return response
        partial = bool(request.declarations)
        prior = served.get(request.handle)
        if prior == "done" or (prior == "too_large" and not partial):
            return response

        task_text = str(messages[index].get("content") or "")
        expanded_code = target.lean_code
        if partial:
            requested, _unknown = await asyncio.to_thread(
                select_declarations, target.lean_code, request.declarations
            )
            expanded_code = "\n\n".join(declaration.text for declaration in requested)

        kept: list[LeanDeclaration] = []
        omitted = 0
        actions: list[str] = []
        keep_offer = False
        updated: str | None
        if not expanded_code.strip():
            outcome = "unknown_declarations"
            updated = apply_expansion_result(
                task_text,
                request.handle,
                f"Expansion of {request.handle}: none of the requested declaration names exist "
                "in that master proof, so nothing was added.",
            )
        else:
            query = (target.theorem_statement or target.theorem_name)[:_QUERY_MAX_CHARS]
            parts = [part for part in (proofs_list_part(task_text),) if part is not None]
            parts.extend(noted_context_parts())

            async def retrieve(kind: ContextKind, text: str, budget: int) -> str:
                return await retrieve_context_evidence(kind, text, budget, query)

            fit = await fit_review_prompt(
                task_text=task_text,
                parts=parts,
                build=lambda task: build_review_prompt(
                    handle=request.handle,
                    expanded_code=expanded_code,
                    task_text=task,
                    partial=partial,
                ),
                fits=call.fits,
                count_tokens=count_tokens,
                retrieve=retrieve,
            )
            actions = fit.actions
            if fit.prompt is None:
                outcome = "too_large"
                keep_offer = not partial
                updated = apply_expansion_result(
                    task_text,
                    request.handle,
                    format_unviewable(request.handle, partial=partial),
                    keep_offer=keep_offer,
                )
            else:
                selection = await _review_selection(call, fit.prompt)
                chosen, _unknown = await asyncio.to_thread(
                    select_declarations, expanded_code, selection.keep
                )
                updated, carried = await _carry_declarations(
                    call, messages, index, task_text, request.handle, chosen, selection.notes
                )
                kept = chosen[:carried]
                omitted = len(chosen) - carried
                outcome = "carried" if kept else "nothing_carried"

        served[request.handle] = "too_large" if keep_offer else "done"
        if updated is None or not await _fits(call, _with_content(messages, index, updated)):
            # The task is too full to take even the result line. Close master proof
            # viewing for this task so the model answers instead of asking again.
            outcome = "no_room"
            kept, omitted = [], 0
            updated = withdraw_expansion_offers(task_text)
        messages = _with_content(messages, index, updated)
        await _announce(
            call, target, request, outcome=outcome, kept=kept, omitted=omitted, actions=actions
        )
        response = await call.generate(messages)
