"""Build F maturity registry for the test-only real-adapter overlay."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class AdapterMaturity(str, Enum):
    NORMAL = "normal"
    DEEP = "deep"
    BLOCKED = "blocked"


class RepeatContract(str, Enum):
    """What a repeated deep run can truthfully compare."""

    NORMALIZED_PROCESS_RESULT = "normalized_process_result"
    NORMALIZED_PROCESS_AND_RUNTIME_STATE = "normalized_process_and_runtime_state"


@dataclass(frozen=True)
class CleanupContract:
    uses_temporary_roots: bool
    removes_runtime_state: bool
    forbids_workspace_escape: bool
    notes: str


@dataclass(frozen=True)
class RealAdapterDescriptor:
    scenario_id: str
    adapter: str
    maturity: AdapterMaturity
    timeout_seconds: int
    repeat_safe: bool
    fields: tuple[str, ...]
    invariants: tuple[str, ...]
    cleanup: CleanupContract
    test_selectors: tuple[str, ...]
    repeat_contract: RepeatContract | None
    repeat_contract_reason: str | None
    blocked_reason: str | None = None

    @property
    def executable(self) -> bool:
        return bool(self.test_selectors)

    def pytest_args(self) -> tuple[str, ...]:
        if not self.test_selectors:
            raise ValueError(f"{self.scenario_id} is metadata-only")
        return ("-q", *self.test_selectors)


_DEEP_IDS = {
    "real_actual_route_start_conflict_matrix_no_side_effects",
    "real_shared_start_guard_representative_race",
    "real_proof_scope_matrix_isolated_under_temp_root",
    "real_direct_source_rag_exclusion_matrix",
    "real_model_visible_prompts_strip_generated_proof_appendices",
    "real_rigor_mandatory_source_overflow_fails_before_model_or_rag",
}

_RUNTIME_STATE_REPEAT_IDS = {
    "real_model_visible_prompts_strip_generated_proof_appendices",
    "real_rigor_mandatory_source_overflow_fails_before_model_or_rag",
}

_BLOCKED_IDS = {
    "real_parent_action_fencing_unavailable_without_production_seam",
    "real_provider_stop_reset_checkpoint_unavailable_without_wait_seam",
    "real_leanoj_full_final_loop_not_safely_bounded",
}

_DEEP_CLEANUP = CleanupContract(
    uses_temporary_roots=True,
    removes_runtime_state=True,
    forbids_workspace_escape=True,
    notes=(
        "Each repeated subprocess receives isolated data/log/temp roots; the matrix "
        "checks guarded workspace paths before and after execution and removes the variant root."
    ),
)
_NORMAL_CLEANUP = CleanupContract(
    uses_temporary_roots=True,
    removes_runtime_state=True,
    forbids_workspace_escape=True,
    notes="The selected Build B-E test owns cleanup through pytest tmp_path and adapter fixtures.",
)
_BLOCKED_CLEANUP = CleanupContract(
    uses_temporary_roots=False,
    removes_runtime_state=False,
    forbids_workspace_escape=True,
    notes="Metadata-only: no production transition owner or external dependency is invoked.",
)


def _build_registry() -> tuple[RealAdapterDescriptor, ...]:
    # Imported lazily to keep the removable adapter package independent of coverage reporting.
    from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE

    descriptors: list[RealAdapterDescriptor] = []
    for record in REAL_ADAPTER_COVERAGE:
        blocked = record.result == "blocked"
        deep = record.scenario_id in _DEEP_IDS
        maturity = (
            AdapterMaturity.BLOCKED
            if blocked
            else AdapterMaturity.DEEP
            if deep
            else AdapterMaturity.NORMAL
        )
        diagnostics = record.diagnostics or {}
        descriptors.append(
            RealAdapterDescriptor(
                scenario_id=record.scenario_id,
                adapter=record.adapter,
                maturity=maturity,
                timeout_seconds=0 if blocked else 180 if deep else 60,
                repeat_safe=not blocked,
                fields=tuple(record.fields),
                invariants=tuple(record.invariants),
                cleanup=(
                    _BLOCKED_CLEANUP
                    if blocked
                    else _DEEP_CLEANUP
                    if deep
                    else _NORMAL_CLEANUP
                ),
                test_selectors=tuple(record.test_selectors),
                repeat_contract=(
                    (
                        RepeatContract.NORMALIZED_PROCESS_AND_RUNTIME_STATE
                        if record.scenario_id in _RUNTIME_STATE_REPEAT_IDS
                        else RepeatContract.NORMALIZED_PROCESS_RESULT
                    )
                    if deep
                    else None
                ),
                repeat_contract_reason=(
                    (
                        "The scenario promises deterministic logical runtime artifacts."
                        if record.scenario_id in _RUNTIME_STATE_REPEAT_IDS
                        else "The scenario creates timestamped or run-identified state; repeatability is the normalized process result."
                    )
                    if deep
                    else None
                ),
                blocked_reason=str(diagnostics.get("reason", "")) or None,
            )
        )
    return tuple(descriptors)


REAL_ADAPTER_MATURITY_REGISTRY = _build_registry()


def descriptors_for(
    maturity: AdapterMaturity,
) -> tuple[RealAdapterDescriptor, ...]:
    return tuple(
        descriptor
        for descriptor in REAL_ADAPTER_MATURITY_REGISTRY
        if descriptor.maturity is maturity
    )


def validate_maturity_registry(
    descriptors: Iterable[RealAdapterDescriptor] = REAL_ADAPTER_MATURITY_REGISTRY,
    *,
    workspace_root: Path | None = None,
) -> None:
    records = tuple(descriptors)
    ids = [record.scenario_id for record in records]
    assert len(ids) == len(set(ids)), "real-adapter scenario IDs must be unique"
    from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE

    coverage_by_id = {record.scenario_id: record for record in REAL_ADAPTER_COVERAGE}
    assert set(ids) == set(coverage_by_id), "registry must exactly cover coverage results"
    assert _DEEP_IDS.isdisjoint(_BLOCKED_IDS)
    assert set(_DEEP_IDS) == {
        record.scenario_id
        for record in records
        if record.maturity is AdapterMaturity.DEEP
    }
    assert set(_BLOCKED_IDS) == {
        record.scenario_id
        for record in records
        if record.maturity is AdapterMaturity.BLOCKED
    }
    for record in records:
        coverage = coverage_by_id[record.scenario_id]
        assert (record.maturity is AdapterMaturity.BLOCKED) == (
            coverage.result == "blocked"
        )
        assert coverage.result in {"passed", "blocked"}
        assert record.adapter in {"real_route", "real_coordinator"}
        assert record.fields and record.invariants
        assert record.timeout_seconds >= 0
        assert record.cleanup.forbids_workspace_escape
        if record.maturity is AdapterMaturity.BLOCKED:
            assert not record.executable
            assert not record.repeat_safe
            assert record.blocked_reason
            assert record.repeat_contract is None
            assert record.repeat_contract_reason is None
            continue
        assert record.executable and record.repeat_safe
        assert record.timeout_seconds > 0
        if record.maturity is AdapterMaturity.DEEP:
            assert record.repeat_contract is not None
            assert record.repeat_contract_reason
        assert record.test_selectors
        assert len(record.test_selectors) == len(set(record.test_selectors))
        for selector in record.test_selectors:
            assert selector.count("::") == 1, selector
            assert selector.split("::", 1)[1].startswith("test_"), selector
        if workspace_root is not None:
            for selector in record.test_selectors:
                selector_path = selector.split("::", 1)[0]
                assert (workspace_root / selector_path).is_file(), selector_path
