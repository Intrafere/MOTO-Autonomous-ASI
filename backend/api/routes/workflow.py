"""
API routes for workflow management.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


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
        
        # Determine which coordinator is active and return its workflow
        tasks = []
        mode = "idle"
        
        if autonomous_coordinator._running:
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
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@router.get("/api/workflow/history")
async def get_workflow_history(limit: int = 50) -> Dict[str, Any]:
    """
    Get completed workflow tasks.
    
    Args:
        limit: Maximum number of tasks to return
        
    Returns:
        List of completed tasks
    """
    try:
        # This would fetch from a persistent history log
        # For now, return empty list
        history = []
        
        return {
            "success": True,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get workflow history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/api/token-stats")
async def get_token_stats() -> Dict[str, Any]:
    """Return cumulative token usage stats and elapsed research time."""
    from backend.shared.token_tracker import token_tracker
    return {"success": True, **token_tracker.get_stats()}

