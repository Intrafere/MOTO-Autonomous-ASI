from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .invariant_catalog import INVARIANTS_BY_ID


AdapterType = Literal["model", "real_route", "real_coordinator", "browser_smoke"]
ResultStatus = Literal["passed", "failed", "skipped", "blocked"]
CoverageRunner = Literal["pytest", "playwright"]


@dataclass(frozen=True)
class InteractionCoverage:
    scenario_id: str
    fields: tuple[str, ...]
    invariants: tuple[str, ...]
    adapter: AdapterType
    result: ResultStatus
    test_file: str
    seed: int | None = None
    replay: tuple[str, ...] = ()
    diagnostics: Mapping[str, str] | None = None
    evidence: tuple[str, ...] = ()
    runner: CoverageRunner | None = None
    test_selectors: tuple[str, ...] = ()
    asserted_invariants: tuple[str, ...] = ()


def assert_coverage_metadata_valid(record: InteractionCoverage) -> None:
    if not record.scenario_id:
        raise AssertionError("Coverage metadata is missing scenario_id.")
    if not record.fields:
        raise AssertionError(f"{record.scenario_id} does not declare crossed fields.")
    if not record.invariants:
        raise AssertionError(f"{record.scenario_id} does not declare invariant ids.")
    if not record.test_file.startswith("tests/"):
        raise AssertionError(f"{record.scenario_id} has non-test file path {record.test_file!r}.")
    if record.seed is not None and record.seed < 0:
        raise AssertionError(f"{record.scenario_id} has a negative seed.")
    if record.seed is not None and not record.replay:
        raise AssertionError(f"{record.scenario_id} has a seed but no replay.")
    if record.result == "passed" and not record.evidence:
        raise AssertionError(f"{record.scenario_id} passed without evidence.")
    executable_claim = record.adapter in {"real_route", "real_coordinator", "browser_smoke"}
    if record.result == "passed" and executable_claim and not record.runner:
        raise AssertionError(f"{record.scenario_id} passed without an executable runner.")
    if record.result == "passed" and executable_claim and not record.test_selectors:
        raise AssertionError(f"{record.scenario_id} passed without exact test selectors.")
    if (
        record.result == "passed"
        and executable_claim
        and set(record.asserted_invariants) != set(record.invariants)
    ):
        raise AssertionError(
            f"{record.scenario_id} must link every claimed invariant to an asserted invariant."
        )
    if record.result == "blocked" and (
        record.runner or record.test_selectors or record.asserted_invariants or record.evidence
    ):
        raise AssertionError(f"{record.scenario_id} is blocked but is not metadata-only.")
    if record.result in {"failed", "skipped", "blocked"} and not record.diagnostics:
        raise AssertionError(
            f"{record.scenario_id} is {record.result} without diagnostics."
        )
    if any(not item.strip() for item in record.replay):
        raise AssertionError(f"{record.scenario_id} has an empty replay step.")
    if any(not item.strip() for item in record.evidence):
        raise AssertionError(f"{record.scenario_id} has empty evidence.")
    if any(not item.strip() for item in record.test_selectors):
        raise AssertionError(f"{record.scenario_id} has an empty test selector.")
    if len(record.test_selectors) != len(set(record.test_selectors)):
        raise AssertionError(f"{record.scenario_id} has duplicate test selectors.")
    if len(record.asserted_invariants) != len(set(record.asserted_invariants)):
        raise AssertionError(f"{record.scenario_id} has duplicate asserted invariants.")
    if record.runner == "pytest":
        for selector in record.test_selectors:
            if selector.count("::") != 1 or not selector.split("::", 1)[1].startswith("test_"):
                raise AssertionError(
                    f"{record.scenario_id} has non-exact pytest selector {selector!r}."
                )
    if record.runner == "playwright":
        for selector in record.test_selectors:
            if "::" not in selector or not selector.split("::", 1)[0].endswith(".spec.js"):
                raise AssertionError(
                    f"{record.scenario_id} has non-exact Playwright selector {selector!r}."
                )

    for invariant_id in record.invariants:
        if invariant_id not in INVARIANTS_BY_ID:
            raise AssertionError(f"{record.scenario_id} references unknown invariant {invariant_id!r}.")
        invariant = INVARIANTS_BY_ID[invariant_id]
        if record.result != "blocked" and record.adapter not in invariant.adapters:
            raise AssertionError(
                f"{record.scenario_id} uses adapter {record.adapter!r} for "
                f"invariant {invariant_id!r}, but the catalog does not list that adapter."
            )


def assert_coverage_records_valid(records: tuple[InteractionCoverage, ...]) -> None:
    scenario_ids = [record.scenario_id for record in records]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise AssertionError("Coverage metadata scenario ids must be unique.")
    for record in records:
        assert_coverage_metadata_valid(record)
