"""
API routes for workflow management.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class SolutionPathStepResponse(BaseModel):
    step_id: str
    title: str
    description: str = ""
    status: Literal["pending", "active", "complete", "blocked"] = "pending"


class SolutionPathRepairResponse(BaseModel):
    proposal_id: str
    reason: str = "unknown_reviewer_failure"
    detail: str = ""
    lifecycle_generation: int


class SolutionPathResponse(BaseModel):
    success: bool = True
    enabled: bool = False
    ownership: Literal["none", "active", "resumable"] = "none"
    mode: Literal["idle", "aggregator", "compiler", "autonomous", "leanoj"] = "idle"
    run_id: str | None = None
    lifecycle_generation: int | None = None
    acceptance_count: int = 0
    revision: int | None = None
    main_route: str = ""
    ordering: Literal["ordered", "unordered"] = "ordered"
    steps: List[SolutionPathStepResponse] = Field(default_factory=list)
    pending_proposals: int = 0
    queued_proposals: int = 0
    reviewing_proposals: int = 0
    repair_required_proposals: int = 0
    repairs: List[SolutionPathRepairResponse] = Field(default_factory=list)
    repair_reason: str | None = None
    repair_detail: str = ""
    message: str = "No solution path is available for the active workflow."


class ResumeSolutionPathProposalRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    proposal_id: str = Field(min_length=1, max_length=256)
    lifecycle_generation: int = Field(ge=1)


class ResumeSolutionPathProposalResponse(BaseModel):
    success: bool = True
    run_id: str
    proposal_id: str
    proposal_status: Literal["queued"]
    lifecycle_generation: int
    message: str


def _solution_path_owner():
    """Resolve one explicit active or resumable owner without guessing by load order."""
    from backend.aggregator.core.coordinator import coordinator
    from backend.compiler.core.compiler_coordinator import compiler_coordinator
    from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
    from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
    from backend.shared.solution_path import solution_path_registry

    coordinators = (
        ("leanoj", leanoj_coordinator, bool(getattr(leanoj_coordinator, "is_active", False))),
        ("autonomous", autonomous_coordinator, bool(getattr(autonomous_coordinator, "is_active", False))),
        ("compiler", compiler_coordinator, bool(getattr(compiler_coordinator, "is_running", False))),
        ("aggregator", coordinator, bool(getattr(coordinator, "is_running", False))),
    )
    attached = []
    for mode, owner, active in coordinators:
        manager = next(
            (
                getattr(owner, name, None)
                for name in ("solution_path_manager", "_solution_path_manager")
                if getattr(owner, name, None) is not None
            ),
            None,
        )
        if manager is not None:
            attached.append((mode, manager, active))

    active = [(mode, manager) for mode, manager, is_active in attached if is_active]
    if len({id(manager) for _, manager in active}) == 1 and active:
        return active[0][0], active[0][1], "active"
    if active:
        return "idle", None, "none"

    loaded = solution_path_registry.loaded_managers()
    if len(loaded) == 1:
        manager = loaded[0]
        workflow_mode = str(getattr(manager.state, "workflow_mode", "") or "")
        mode = workflow_mode if workflow_mode in {"autonomous", "leanoj"} else "aggregator"
        return mode, manager, "resumable"
    if len(loaded) > 1:
        manager = solution_path_registry.latest_loaded_manager()
        if manager is not None:
            workflow_mode = str(getattr(manager.state, "workflow_mode", "") or "")
            mode = workflow_mode if workflow_mode in {"autonomous", "leanoj"} else "aggregator"
            return mode, manager, "resumable"
        return "idle", None, "none"

    unique_attached = {}
    for mode, manager, _ in attached:
        unique_attached.setdefault(id(manager), (mode, manager))
    if len(unique_attached) == 1:
        mode, manager = next(iter(unique_attached.values()))
        return mode, manager, "resumable"
    return "idle", None, "none"


def _apply_boost_state(tasks: List[Dict]) -> List[Dict]:
    """Apply current boost state to tasks before returning to frontend."""
    from backend.shared.boost_manager import boost_manager
    
    for task in tasks:
        task_id = task.get('task_id', '')
        task['using_boost'] = boost_manager.should_use_boost(task_id)
    
    return tasks


@router.get("/api/workflow/predictions")
async def get_workflow_predictions() -> Dict[str, Any]:
    """
    Get predicted next 20 API calls.
    
    Returns:
        List of predicted workflow tasks
    """
    try:
        # Import global coordinator instances
        from backend.aggregator.core.coordinator import coordinator
        from backend.compiler.core.compiler_coordinator import compiler_coordinator
        from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
        from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
        
        # Determine which coordinator is active and return its workflow
        tasks = []
        mode = "idle"
        
        if leanoj_coordinator.is_active:
            mode = "leanoj"
            tasks = [task.model_dump(mode="json") for task in leanoj_coordinator.workflow_tasks]
            logger.debug(f"Returning {len(tasks)} tasks from LeanOJ coordinator")
        elif autonomous_coordinator._running:
            mode = "autonomous"
            # For autonomous mode, check which sub-coordinator is active
            if autonomous_coordinator._brainstorm_aggregator and autonomous_coordinator._brainstorm_aggregator.is_running:
                # Brainstorm aggregation active
                tasks = [task.dict() for task in autonomous_coordinator._brainstorm_aggregator.workflow_tasks]
                logger.debug(f"Returning {len(tasks)} tasks from autonomous brainstorm aggregator")
            elif autonomous_coordinator._paper_compiler and autonomous_coordinator._paper_compiler.is_running:
                # Paper compilation active
                tasks = [task.dict() for task in autonomous_coordinator._paper_compiler.workflow_tasks]
                logger.debug(f"Returning {len(tasks)} tasks from autonomous paper compiler")
            else:
                # Topic selection or idle - return autonomous coordinator's own tasks
                tasks = [task.dict() for task in autonomous_coordinator.workflow_tasks]
                logger.debug(f"Returning {len(tasks)} tasks from autonomous coordinator")
        elif compiler_coordinator.is_running:
            mode = "compiler"
            tasks = [task.dict() for task in compiler_coordinator.workflow_tasks]
            logger.debug(f"Returning {len(tasks)} tasks from compiler coordinator")
        elif coordinator.is_running:
            mode = "aggregator"
            tasks = [task.dict() for task in coordinator.workflow_tasks]
            logger.debug(f"Returning {len(tasks)} tasks from aggregator coordinator")
        
        # CRITICAL: Always apply current boost state before returning
        # This ensures frontend always gets the latest boost state
        tasks = _apply_boost_state(tasks)
        
        return {
            "success": True,
            "mode": mode,
            "tasks": tasks
        }
    except Exception as e:
        logger.error(f"Failed to get workflow predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")


@router.get("/api/workflow/solution-path", response_model=SolutionPathResponse)
async def get_solution_path() -> SolutionPathResponse:
    """Return an in-memory solution-path snapshot without filesystem or model work."""
    try:
        mode, manager, ownership = _solution_path_owner()

        state = getattr(manager, "state", None)
        plan = getattr(state, "plan", None)
        route = getattr(plan, "route", None)
        if manager is None or state is None:
            return SolutionPathResponse(
                mode=mode,
                ownership=ownership,
                message="Solution path tracking is not available for this workflow yet.",
            )

        statuses = [
            str(getattr(proposal, "status", "")).split(".")[-1].lower()
            for proposal in getattr(state, "proposals", [])
        ]
        queued = sum(status in {"queued", "followup"} for status in statuses)
        reviewing = sum(status == "reviewing" for status in statuses)
        repair_proposals = [
            proposal
            for proposal, status in zip(getattr(state, "proposals", []), statuses)
            if status == "user_repair_required"
        ]
        repairs = [
            SolutionPathRepairResponse(
                proposal_id=str(proposal.proposal_id),
                reason=str(getattr(proposal, "repair_reason", None) or "unknown_reviewer_failure").split(".")[-1].lower(),
                detail=str(getattr(proposal, "repair_detail", "") or getattr(proposal, "feedback", "")),
                lifecycle_generation=int(
                    getattr(proposal, "repair_generation", None)
                    or getattr(state, "lifecycle_generation", 1)
                ),
            )
            for proposal in repair_proposals
        ]
        pending = sum(
            1
            for status in statuses
            if status in {"queued", "reviewing", "followup"}
        )
        if plan is None or route is None:
            return SolutionPathResponse(
                enabled=bool(getattr(manager, "active", False)),
                ownership=ownership,
                mode=mode,
                run_id=getattr(state, "run_id", None),
                lifecycle_generation=getattr(state, "lifecycle_generation", None),
                acceptance_count=getattr(state, "acceptance_count", 0),
                pending_proposals=pending,
                queued_proposals=queued,
                reviewing_proposals=reviewing,
                repair_required_proposals=len(repairs),
                repairs=repairs,
                repair_reason=repairs[0].reason if repairs else None,
                repair_detail=repairs[0].detail if repairs else "",
                message="Solution path tracking is loaded; no approved plan is available yet. This is normal and some runs may never generate a solution path.",
            )
        return SolutionPathResponse(
            enabled=True,
            ownership=ownership,
            mode=mode,
            run_id=getattr(state, "run_id", None),
            lifecycle_generation=getattr(state, "lifecycle_generation", None),
            acceptance_count=getattr(state, "acceptance_count", 0),
            revision=getattr(plan, "revision", None),
            main_route=str(getattr(plan, "main_route", "")),
            ordering=str(getattr(route, "ordering", "ordered")).split(".")[-1].lower(),
            steps=[
                SolutionPathStepResponse(
                    step_id=str(step.step_id),
                    title=str(step.title),
                    description=str(getattr(step, "description", "")),
                    status=str(getattr(step, "status", "pending")),
                )
                for step in getattr(route, "steps", [])
            ],
            pending_proposals=pending,
            queued_proposals=queued,
            reviewing_proposals=reviewing,
            repair_required_proposals=len(repairs),
            repairs=repairs,
            repair_reason=repairs[0].reason if repairs else None,
            repair_detail=repairs[0].detail if repairs else "",
            message="Current Main Submitter 1-approved solution path.",
        )
    except Exception:
        logger.exception("Failed to read solution path snapshot")
        return SolutionPathResponse(message="Solution path is temporarily unavailable.")


@router.post(
    "/api/workflow/solution-path/resume",
    response_model=ResumeSolutionPathProposalResponse,
)
async def resume_solution_path_proposal(
    request: ResumeSolutionPathProposalRequest,
) -> ResumeSolutionPathProposalResponse:
    """Retry one repair-blocked proposal owned by the active/resumable workflow."""
    mode, manager, ownership = _solution_path_owner()
    if manager is None or ownership == "none":
        raise HTTPException(status_code=404, detail="No active or resumable solution path was found")
    state = manager.state
    if state.run_id != request.run_id:
        raise HTTPException(status_code=409, detail="Solution path run ownership changed")
    if state.lifecycle_generation != request.lifecycle_generation:
        raise HTTPException(status_code=409, detail="Solution path lifecycle generation changed")
    try:
        proposal = await manager.resume_proposal(
            request.proposal_id,
            lifecycle_generation=request.lifecycle_generation,
        )
    except ValueError as exc:
        detail = str(exc)
        status = 404 if "not found" in detail else 409
        raise HTTPException(status_code=status, detail=detail) from exc
    return ResumeSolutionPathProposalResponse(
        run_id=request.run_id,
        proposal_id=proposal.proposal_id,
        proposal_status="queued",
        lifecycle_generation=request.lifecycle_generation,
        message=f"Solution-path update resumed for {mode} settings.",
    )


@router.get("/api/token-stats")
async def get_token_stats() -> Dict[str, Any]:
    """Return cumulative token usage stats and elapsed research time."""
    from backend.shared.token_tracker import token_tracker
    return {"success": True, **token_tracker.get_stats()}

