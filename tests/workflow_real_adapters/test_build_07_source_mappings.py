from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID
from tests.workflow_harness.source_mappings import BUILD_07_SOURCE_MAPPINGS


def _selector_exists(selector: str) -> bool:
    parts = selector.split("::")
    path = Path(parts[0])
    if not path.is_file():
        return False
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    if len(parts) == 2:
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[1]
            for node in module.body
        )
    class_node = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == parts[1]
        ),
        None,
    )
    return class_node is not None and any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[2]
        for node in class_node.body
    )


def _vitest_selector_exists(selector: str) -> bool:
    test_path, suite_name, test_name = selector.split("::")
    source = Path(test_path).read_text(encoding="utf-8-sig")
    suite_pattern = rf"\bdescribe\(\s*['\"]{re.escape(suite_name)}['\"]"
    test_pattern = rf"\b(?:test|it)\(\s*['\"]{re.escape(test_name)}['\"]"
    return bool(re.search(suite_pattern, source) and re.search(test_pattern, source))


def _mapping_selector_exists(mapping) -> bool:
    if mapping.runner == "pytest":
        return _selector_exists(mapping.test_selector)
    if mapping.runner == "vitest":
        return _vitest_selector_exists(mapping.test_selector)
    return False


def test_source_mappings_cover_stable_invariants() -> None:
    invariant_ids = [mapping.invariant_id for mapping in BUILD_07_SOURCE_MAPPINGS]

    assert set(invariant_ids).issubset(INVARIANTS_BY_ID)
    scenario_ids = [mapping.scenario_id for mapping in BUILD_07_SOURCE_MAPPINGS]
    assert len(scenario_ids) == len(set(scenario_ids))
    for mapping in BUILD_07_SOURCE_MAPPINGS:
        assert mapping.production_sources
        invariant = INVARIANTS_BY_ID[mapping.invariant_id]
        assert mapping.coverage_record().fields == (
            invariant.field,
            *invariant.crossed_fields,
        )
        for source in mapping.production_sources:
            assert Path(source).is_file(), source


def test_build_07_passes_link_exact_nodes_and_gaps_are_blocked() -> None:
    for mapping in BUILD_07_SOURCE_MAPPINGS:
        if mapping.result == "passed":
            assert mapping.test_selector
            assert _mapping_selector_exists(mapping), mapping.test_selector
            assert mapping.evidence
            assert mapping.observed_transition
            assert mapping.blocked_reason is None
            assert mapping.blocked_dependencies == ()
            assert mapping.adapter in {
                "focused_production",
                "real_route",
                "real_coordinator",
            }
        else:
            assert mapping.result == "blocked"
            assert mapping.test_selector is None
            assert mapping.blocked_reason
            assert mapping.blocked_dependencies
