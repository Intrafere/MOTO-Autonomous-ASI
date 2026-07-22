"""Fault-tolerant hooks for attaching solution-path state to existing LLM calls."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Protocol

from pydantic import ValidationError

from backend.shared.utils import count_tokens

from .models import SolutionPlan, SolutionRoute

logger = logging.getLogger(__name__)
PLAN_UPDATE_KEYS = ("solution_path_update", "progressive_solution_path")
_SUPERVISED_UPDATE_TASKS: set[asyncio.Task[Any]] = set()


def _supervise_optional_update(task: asyncio.Task[Any]) -> None:
    """Retain background persistence and observe every terminal exception."""
    _SUPERVISED_UPDATE_TASKS.add(task)

    def done(completed: asyncio.Task[Any]) -> None:
        _SUPERVISED_UPDATE_TASKS.discard(completed)
        if completed.cancelled():
            logger.warning("Optional solution-path update persistence was cancelled")
            return
        try:
            completed.result()
        except Exception:
            logger.exception("Unable to persist optional solution-path update")

    task.add_done_callback(done)


class SolutionPathManager(Protocol):
    @property
    def active(self) -> bool: ...

    @property
    def state(self) -> Any: ...

    async def propose(
        self,
        route: SolutionRoute,
        *,
        lifecycle_generation: int,
        rationale: str = "",
        main_route: str = "",
        proposer_role: str = "validator",
        source_task_id: str | None = None,
        source_phase: str | None = None,
        source_decision: str | None = None,
    ) -> Any: ...

    async def set_acceptance_count(self, count: int) -> None: ...


def validator_instruction(*, batch: bool = False) -> str:
    placement = (
        "For a batch response, include at most one `solution_path_update` at the "
        "top level beside `decisions`, never inside an individual decision. "
        if batch
        else "Include it only as a top-level field beside the primary decision fields. "
    )
    return (
        "\n\nPROGRESSIVE SOLUTION PATH (OPTIONAL): Alongside your normal decision, "
        "you MAY include `solution_path_update` when this same semantic review "
        "reveals a material development: a stronger route or goal, evidence that "
        "requires a substantial revision, or completion of a meaningful plan item. "
        "If no approved solution path is shown above and the available context now "
        "supports a clear, useful solution route, use `solution_path_update` to "
        "propose the initial path; the absence of an existing path is not a reason "
        "to omit a clear initial proposal. If an approved path is shown, continue "
        "to use `solution_path_update` for material revisions or meaningful progress. "
        "Do not update it for small nuances, wording changes, or routine validation. "
        + placement
        + "This is the sole permitted extension to the otherwise exact primary JSON "
        "schema. Its compact shape is "
        '`"solution_path_update":{"main_route":"non-empty distilled overall route",'
        '"route":{"ordering":"ordered|parallel|mixed","steps":[...]},'
        '"rationale":"optional"}`. `main_route` must be non-empty for an initial path; '
        "for a later update it may be omitted to preserve the approved main route. "
        "Omit the entire extension otherwise. This field is advisory and must "
        "not alter your primary decision. Do not use it for deterministic, tool, "
        "syntax, proof-checker, or integrity checks."
    )


def advisory_plan_block(manager: SolutionPathManager | None) -> str:
    if manager is None or not manager.active:
        return ""
    plan: SolutionPlan | None = getattr(manager.state, "plan", None)
    if plan is None:
        return ""
    return (
        "\n\n[ADVISORY PROGRESSIVE SOLUTION PATH]\n"
        "This is a distillation attempt at the best currently known path from "
        "the available context. It may be wrong or incomplete, is guidance only, "
        "and may be deviated from whenever a better route serves the user prompt "
        "or the current subgoal. It is not authoritative evidence.\n"
        f"{plan.model_dump_json(indent=2)}\n"
        "[END ADVISORY PROGRESSIVE SOLUTION PATH]"
    )


def with_validator_hook(
    prompt: str,
    manager: SolutionPathManager | None,
    *,
    batch: bool = False,
) -> str:
    if manager is None or not manager.active:
        return prompt
    return prompt + advisory_plan_block(manager) + validator_instruction(batch=batch)


def with_solver_plan(prompt: str, manager: SolutionPathManager | None) -> str:
    return prompt + advisory_plan_block(manager)


def with_budgeted_solver_plan(
    prompt: str,
    manager: SolutionPathManager | None,
    max_input_tokens: int,
) -> str:
    """Append the advisory plan only when the complete prompt still fits."""
    block = advisory_plan_block(manager)
    if not block:
        return prompt
    candidate = prompt + block
    return candidate if count_tokens(candidate) <= max_input_tokens else prompt


async def enqueue_optional_update(
    payload: Any,
    manager: SolutionPathManager | None,
    *,
    proposer_role: str = "validator",
    source_task_id: str | None = None,
    source_phase: str | None = None,
    source_decision: str | None = None,
) -> Any | None:
    """Validate then supervise durable enqueue without blocking primary validation."""
    if manager is None or not manager.active or not isinstance(payload, dict):
        return None
    raw = next((payload.get(key) for key in PLAN_UPDATE_KEYS if key in payload), None)
    if not isinstance(raw, dict):
        return None
    try:
        route = SolutionRoute.model_validate(raw.get("route", raw))
    except (ValidationError, TypeError, ValueError):
        logger.debug("Ignoring malformed optional solution-path update")
        return None
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""
    current_plan: SolutionPlan | None = getattr(manager.state, "plan", None)
    if "main_route" in raw:
        main_route = raw.get("main_route")
        if not isinstance(main_route, str) or not main_route.strip():
            logger.debug("Ignoring optional solution-path update with invalid main_route")
            return None
        main_route = main_route.strip()
    elif current_plan is not None and current_plan.main_route.strip():
        main_route = current_plan.main_route
    else:
        logger.debug("Ignoring initial solution-path update without main_route")
        return None
    lifecycle_generation = getattr(manager.state, "lifecycle_generation", None)
    async def persist() -> Any | None:
        try:
            propose_kwargs = {
                "rationale": rationale,
                "main_route": main_route,
                "proposer_role": proposer_role,
                "source_task_id": source_task_id,
                "source_phase": source_phase,
                "source_decision": source_decision,
            }
            if lifecycle_generation is not None:
                propose_kwargs["lifecycle_generation"] = int(lifecycle_generation)
            return await manager.propose(
                route,
                **propose_kwargs,
            )
        except Exception:
            logger.exception("Unable to persist optional solution-path update")
            return None

    task = asyncio.create_task(
        persist(),
        name=f"solution-path-update:{source_task_id or proposer_role}",
    )
    _supervise_optional_update(task)
    # Give cheap/in-memory managers one turn to finish while never waiting on disk I/O.
    await asyncio.sleep(0)
    if task.done() and not task.cancelled():
        try:
            return task.result()
        except Exception:
            return None
    return task


async def note_acceptances(
    manager: SolutionPathManager | None, count: int
) -> None:
    if manager is None:
        return
    await manager.set_acceptance_count(max(0, count))
