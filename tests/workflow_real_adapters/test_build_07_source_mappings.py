from __future__ import annotations

import ast
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


def test_build_07_source_mappings_cover_ten_stable_invariants() -> None:
    invariant_ids = [mapping.invariant_id for mapping in BUILD_07_SOURCE_MAPPINGS]

    assert len(invariant_ids) == 10
    assert len(invariant_ids) == len(set(invariant_ids))
    assert set(invariant_ids).issubset(INVARIANTS_BY_ID)
    assert all(mapping.production_sources for mapping in BUILD_07_SOURCE_MAPPINGS)


def test_build_07_passes_link_exact_nodes_and_gaps_are_blocked() -> None:
    for mapping in BUILD_07_SOURCE_MAPPINGS:
        if mapping.result == "passed":
            assert mapping.test_selector
            assert _selector_exists(mapping.test_selector), mapping.test_selector
            assert mapping.evidence
            assert mapping.blocked_reason is None
        else:
            assert mapping.result == "blocked"
            assert mapping.test_selector is None
            assert mapping.blocked_reason
