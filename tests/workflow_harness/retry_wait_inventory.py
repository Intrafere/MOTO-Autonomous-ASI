"""Test-only inventory of production retry and blocking waits.

The inventory is deliberately bounded to modules that own provider retries or
workflow retry/backoff loops.  Every ``sleep`` in those modules must match one
explicit selector below.  This keeps the testing overlay removable while making
new waits fail closed as unclassified evidence.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class WaitClassification(str, Enum):
    ASSURED_RETRY = "assured_retry"
    UNASSURED_RETRY = "unassured_retry"
    NON_RETRY = "non_retry"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class WaitSelector:
    path: str
    scope: str
    call: str
    occurrence: int


@dataclass(frozen=True)
class WaitPolicy:
    selector: WaitSelector
    classification: WaitClassification
    rationale: str
    assurance: str | None = None


@dataclass(frozen=True)
class WaitSite:
    selector: WaitSelector
    line: int
    classification: WaitClassification
    rationale: str
    assurance: str | None


@dataclass(frozen=True)
class RetryWaitInventory:
    sites: tuple[WaitSite, ...]
    stale_policies: tuple[WaitPolicy, ...]

    @property
    def unclassified(self) -> tuple[WaitSite, ...]:
        return tuple(site for site in self.sites if site.classification is WaitClassification.UNCLASSIFIED)

    @property
    def unassured_retries(self) -> tuple[WaitSite, ...]:
        return tuple(site for site in self.sites if site.classification is WaitClassification.UNASSURED_RETRY)

    @property
    def assured_retries(self) -> tuple[WaitSite, ...]:
        return tuple(site for site in self.sites if site.classification is WaitClassification.ASSURED_RETRY)

    @property
    def complete_and_assured(self) -> bool:
        return not self.unclassified and not self.unassured_retries and not self.stale_policies


def _production_source_paths(repo_root: Path) -> tuple[str, ...]:
    roots = (
        repo_root / "backend" / "aggregator",
        repo_root / "backend" / "api",
        repo_root / "backend" / "autonomous",
        repo_root / "backend" / "compiler",
        repo_root / "backend" / "leanoj",
        repo_root / "backend" / "shared",
    )
    return tuple(
        path.relative_to(repo_root).as_posix()
        for root in roots
        for path in sorted(root.rglob("*.py"))
    )


def _selector(path: str, scope: str, occurrence: int, call: str = "asyncio.sleep") -> WaitSelector:
    return WaitSelector(path=path, scope=scope, call=call, occurrence=occurrence)


def _policy(
    path: str,
    scope: str,
    occurrence: int,
    classification: WaitClassification,
    rationale: str,
    *,
    call: str = "asyncio.sleep",
    assurance: str | None = None,
) -> WaitPolicy:
    return WaitPolicy(
        selector=_selector(path, scope, occurrence, call),
        classification=classification,
        rationale=rationale,
        assurance=assurance,
    )


# Explicit policies are populated below from the production Python AST surface. A retry is
# "assured" only when the wait is preceded by the shared provider retry callback/
# broadcaster seam.  Workflow sleeps that merely log a retry remain deliberately
# unassured until production routes them through that seam.
def _scope_policies(
    path: str,
    scope: str,
    count: int,
    classification: WaitClassification,
    rationale: str,
    *,
    call: str = "asyncio.sleep",
    assurance: str | None = None,
) -> tuple[WaitPolicy, ...]:
    return tuple(
        _policy(
            path,
            scope,
            occurrence,
            classification,
            rationale,
            call=call,
            assurance=assurance,
        )
        for occurrence in range(1, count + 1)
    )


WAIT_POLICIES: tuple[WaitPolicy, ...] = (
    *_scope_policies(
        "backend/aggregator/agents/submitter.py",
        "SubmitterAgent._run_loop",
        3,
        WaitClassification.NON_RETRY,
        "Queue backpressure or normal pacing.",
    ),
    _policy(
        "backend/api/routes/cloud_access.py",
        "_expire_pending_after_ttl",
        1,
        WaitClassification.NON_RETRY,
        "OAuth pending-state expiration timer.",
    ),
    _policy(
        "backend/shared/api_client_manager.py",
        "APIClientManager._with_hung_connection_watchdog._watchdog",
        1,
        WaitClassification.NON_RETRY,
        "Hung-call observation timer; it neither retries nor blocks the provider call.",
    ),
    _policy(
        "backend/shared/openai_codex_client.py",
        "OpenAICodexClient._monitor_event_loop_stalls",
        1,
        WaitClassification.NON_RETRY,
        "Event-loop stall sampling cadence for websocket diagnostics; it neither retries nor blocks the provider call.",
    ),
    _policy(
        "backend/shared/provider_recovery/runtime.py",
        "ProviderRecoveryRuntime.sleep_stop_aware",
        1,
        WaitClassification.ASSURED_RETRY,
        "Shared retry runtime sleeps only after its observer has been called; APIClientManager.emit_and_wait_for_retry supplies the assured activity observer.",
        assurance="APIClientManager.emit_and_wait_for_retry/provider_retry_cooldown",
    ),
    _policy(
        "backend/shared/lean4_client.py",
        "Lean4Client._wipe_lake_directory",
        1,
        WaitClassification.NON_RETRY,
        "Bounded Windows filesystem cleanup wait, not a model/provider retry.",
        call="time.sleep",
    ),
    _policy(
        "backend/shared/provider_pause.py",
        "wait_for_provider_reset",
        1,
        WaitClassification.NON_RETRY,
        "Manual-reset provider pause polling has no fixed retry duration.",
    ),
    _policy(
        "backend/shared/sleep_inhibitor.py",
        "SleepInhibitor._run_worker",
        1,
        WaitClassification.NON_RETRY,
        "Native power-request worker shutdown synchronization.",
        call="time.sleep",
    ),
    _policy(
        "backend/shared/solution_path/engine.py",
        "SolutionPathEngine._review_loop",
        1,
        WaitClassification.NON_RETRY,
        "Persisted proposal scheduling delay before the next review turn.",
    ),
    _policy(
        "backend/shared/solution_path/integration.py",
        "enqueue_optional_update",
        1,
        WaitClassification.NON_RETRY,
        "Cooperative event-loop yield after non-blocking enqueue.",
    ),
    _policy(
        "backend/aggregator/core/coordinator.py",
        "Coordinator._validator_loop",
        1,
        WaitClassification.NON_RETRY,
        "Empty-queue polling cadence.",
    ),
    _policy(
        "backend/aggregator/core/coordinator.py",
        "Coordinator._single_model_workflow",
        1,
        WaitClassification.NON_RETRY,
        "Normal inter-round pacing.",
    ),
    _policy(
        "backend/compiler/core/compiler_coordinator.py",
        "CompilerCoordinator._run_critique_aggregation",
        1,
        WaitClassification.NON_RETRY,
        "Normal delay between critique attempts.",
    ),
    *_scope_policies(
        "backend/compiler/core/compiler_coordinator.py",
        "CompilerCoordinator._monitor_aggregator_for_rerag",
        2,
        WaitClassification.NON_RETRY,
        "Periodic Aggregator monitoring cadence, including recovery to the next poll.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator._topic_exploration_phase",
        1,
        WaitClassification.NON_RETRY,
        "Exploration progress polling cadence.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator._brainstorm_aggregation_loop",
        1,
        WaitClassification.NON_RETRY,
        "Brainstorm progress polling cadence.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator.force_tier3_final_answer",
        1,
        WaitClassification.NON_RETRY,
        "Cooperative event-loop yield during parent phase handoff.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator._paper_title_exploration_phase",
        1,
        WaitClassification.NON_RETRY,
        "Title exploration progress polling cadence.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator._compile_paper",
        1,
        WaitClassification.NON_RETRY,
        "Child compiler progress polling cadence.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator._compile_tier3_paper",
        1,
        WaitClassification.NON_RETRY,
        "Tier 3 child compiler progress polling cadence.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator.clear_all_data",
        1,
        WaitClassification.NON_RETRY,
        "Pre-cleanup drain delay.",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator.clear_all_data.safe_rmtree",
        1,
        WaitClassification.NON_RETRY,
        "Filesystem cleanup retry, outside provider/API Live Activity scope.",
        call="time.sleep",
    ),
    _policy(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator.clear_all_data",
        2,
        WaitClassification.NON_RETRY,
        "Post-cleanup RAG synchronization delay.",
    ),
    _policy(
        "backend/leanoj/core/leanoj_coordinator.py",
        "LeanOJCoordinator._wait_for_brainstorm_queue_turn",
        1,
        WaitClassification.NON_RETRY,
        "Queue backpressure polling cadence.",
    ),
    _policy(
        "backend/api/routes/proofs.py",
        "_run_manual_proof_check.execute_round",
        1,
        WaitClassification.NON_RETRY,
        "Cooperative event-loop yield between continuous proof rounds.",
    ),
)


def _call_name(node: ast.Call) -> str | None:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    name = ".".join(reversed(parts))
    if name in {"asyncio.sleep", "time.sleep"}:
        return name
    return None


class _WaitVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.found: list[tuple[str, int]] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name is not None:
            self.found.append((name, node.lineno))
        self.generic_visit(node)


def _discover(repo_root: Path, relative_path: str) -> Iterable[tuple[WaitSelector, int]]:
    source_path = repo_root / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8-sig"), filename=str(source_path))
    visitor = _WaitVisitor()
    visitor.visit(tree)

    # Re-walk by scope so selectors remain stable when unrelated lines move.
    counters: dict[tuple[str, str], int] = {}

    class ScopedVisitor(_WaitVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name = _call_name(node)
            if name is not None:
                scope = ".".join(self.scope) or "<module>"
                key = (scope, name)
                counters[key] = counters.get(key, 0) + 1
                yield_value.append(
                    (
                        WaitSelector(relative_path, scope, name, counters[key]),
                        node.lineno,
                    )
                )
            self.generic_visit(node)

    yield_value: list[tuple[WaitSelector, int]] = []
    scoped = ScopedVisitor()
    scoped.visit(tree)
    return tuple(yield_value)


def build_retry_wait_inventory(
    repo_root: Path,
    *,
    policies: Iterable[WaitPolicy] = WAIT_POLICIES,
) -> RetryWaitInventory:
    policy_by_selector = {policy.selector: policy for policy in policies}
    discovered: list[WaitSite] = []
    seen: set[WaitSelector] = set()
    for relative_path in _production_source_paths(repo_root):
        for selector, line in _discover(repo_root, relative_path):
            seen.add(selector)
            policy = policy_by_selector.get(selector)
            if policy is None:
                discovered.append(
                    WaitSite(
                        selector=selector,
                        line=line,
                        classification=WaitClassification.UNCLASSIFIED,
                        rationale="No explicit retry/non-retry policy covers this bounded wait.",
                        assurance=None,
                    )
                )
            else:
                discovered.append(
                    WaitSite(
                        selector=selector,
                        line=line,
                        classification=policy.classification,
                        rationale=policy.rationale,
                        assurance=policy.assurance,
                    )
                )
    stale = tuple(policy for policy in policies if policy.selector not in seen)
    return RetryWaitInventory(tuple(discovered), stale)


def format_wait_site(site: WaitSite) -> str:
    selector = site.selector
    return (
        f"{selector.path}:{site.line} "
        f"{selector.scope} {selector.call}#{selector.occurrence} "
        f"[{site.classification.value}]"
    )
