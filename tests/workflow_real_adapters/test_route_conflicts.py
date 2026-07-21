from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.api.routes import aggregator as aggregator_route
from backend.api.routes import autonomous as autonomous_route
from backend.api.routes import compiler as compiler_route
from backend.api.routes import leanoj as leanoj_route
from backend.shared.workflow_start_guard import WorkflowStartGuard
from tests.workflow_real_adapters.coverage_records import ROUTE_CONFLICT_COVERAGE
from tests.workflow_real_adapters.helpers import inactive_workflow_flags, patch_attributes
from tests.workflow_harness.invariants import assert_single_active_workflow
from tests.workflow_harness.model import WorkflowMode, WorkflowPhase
from tests.workflow_harness.real_adapters import RealWorkflowObservation, route_workflow_state


class _RunningFlag:
    def __init__(self, is_running: bool = False) -> None:
        self.is_running = is_running


class _ActiveFlag:
    def __init__(self, is_active: bool = False) -> None:
        self.is_active = is_active


class _AutonomousFlag:
    def __init__(self, *, state_running: bool = False, active: bool = False) -> None:
        self.is_active = active
        self._state_running = state_running

    def get_state(self):
        return route_workflow_state(is_running=self._state_running, current_tier="tier1_aggregation")


@pytest.mark.parametrize(
    ("route_module", "route_label"),
    [
        (aggregator_route, "Aggregator"),
        (compiler_route, "Compiler"),
        (autonomous_route, "Autonomous Research"),
        (leanoj_route, "Proof Solver"),
    ],
)
@pytest.mark.parametrize(
    ("active_mode", "expected_fragment"),
    [
        (WorkflowMode.MANUAL_AGGREGATOR, "Aggregator"),
        (WorkflowMode.MANUAL_COMPILER, "Compiler"),
        (WorkflowMode.AUTONOMOUS, "Autonomous Research"),
        (WorkflowMode.LEANOJ, "Proof Solver"),
    ],
)
def test_real_route_start_conflicts_preserve_single_workflow_invariant(
    monkeypatch,
    tmp_path,
    route_module,
    route_label,
    active_mode,
    expected_fragment,
):
    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=active_mode,
        phase=WorkflowPhase.TIER1_AGGREGATION if active_mode is WorkflowMode.AUTONOMOUS else WorkflowPhase.PAPER_WRITING,
    )
    observation.record("route_start_conflict", route=route_module.__name__, active_mode=active_mode.value)

    monkeypatch.setattr(route_module, "coordinator", _RunningFlag(active_mode is WorkflowMode.MANUAL_AGGREGATOR))
    monkeypatch.setattr(route_module, "compiler_coordinator", _RunningFlag(active_mode is WorkflowMode.MANUAL_COMPILER))
    monkeypatch.setattr(
        route_module,
        "autonomous_coordinator",
        _AutonomousFlag(
            state_running=active_mode is WorkflowMode.AUTONOMOUS,
            active=active_mode is WorkflowMode.AUTONOMOUS,
        ),
    )
    monkeypatch.setattr(route_module, "leanoj_coordinator", _ActiveFlag(active_mode is WorkflowMode.LEANOJ))

    conflict = route_module._get_start_conflict()

    if route_label == expected_fragment:
        assert conflict is not None
        assert "already running" in conflict
    else:
        assert conflict is not None
        assert expected_fragment in conflict
    observation.emit("start_blocked", active_mode=active_mode.value, route=route_module.__name__)
    assert_single_active_workflow(observation)


def test_real_autonomous_route_counts_pending_child_activity_as_active(monkeypatch, tmp_path):
    observation = RealWorkflowObservation(
        runtime_root=tmp_path,
        mode=WorkflowMode.AUTONOMOUS,
        phase=WorkflowPhase.TIER1_AGGREGATION,
    )
    observation.record("autonomous_child_pending_conflict")

    route_flags = inactive_workflow_flags()
    route_flags["autonomous_coordinator"] = SimpleNamespace(
            is_active=True,
            get_state=lambda: route_workflow_state(is_running=False, current_tier="tier1_aggregation"),
    )
    patch_attributes(monkeypatch, autonomous_route, route_flags)

    conflict = autonomous_route._get_start_conflict()

    assert conflict == "Autonomous research is already running"
    observation.emit("start_blocked", active_mode=WorkflowMode.AUTONOMOUS.value)
    assert_single_active_workflow(observation)


def test_compiler_proof_only_guard_owner_blocks_every_other_route(monkeypatch):
    guard = WorkflowStartGuard()
    monkeypatch.setattr(
        "backend.shared.workflow_start_guard.sleep_inhibitor.acquire",
        lambda _owner: None,
    )
    guard.commit(compiler_route.COMPILER_PROOF_ONLY_OWNER)
    for module in (
        aggregator_route,
        autonomous_route,
        compiler_route,
        leanoj_route,
    ):
        monkeypatch.setattr(module, "workflow_start_guard", guard)

    assert aggregator_route._get_start_conflict()
    assert autonomous_route._get_start_conflict()
    assert compiler_route._get_start_conflict()
    assert leanoj_route._get_start_conflict()
