"""Every applied model-output repair must reach Live Activity.

The shared parser repairs syntax only for callers that collect the repairs in a
``repairs`` list, and key recovery always returns its repairs. This inventory
walks the production backend and fails when a function collects repairs but
neither publishes them (``report_model_output_normalizations``) nor hands them
to its caller (a ``repairs`` parameter, or a reviewed propagating helper).
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

PARSE_FUNCTIONS = {"parse_json", "sanitize_json_response"}
REPORTERS = {"report_model_output_normalizations"}

# Functions that collect repairs and return them to a caller that publishes
# them. Each entry names the module-relative function and why it is safe.
PROPAGATING_HELPERS = {
    ("backend/autonomous/agents/output_contract.py", "parse_contract_reply"):
        "returns (value, repairs); parse_with_contract_retry publishes them",
    ("backend/compiler/agents/high_param_submitter.py", "_parse_rigor_reply"):
        "returns a _RigorReply; discovery/placement publish it via "
        "_publish_rigor_repairs only after their contract checks accept the reply",
}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _collects_repairs(call: ast.Call) -> bool:
    name = _call_name(call)
    if name == "recover_key_labels":
        return True
    if name not in PARSE_FUNCTIONS:
        return False
    for keyword in call.keywords:
        if keyword.arg == "repairs":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
    return False


def _references_reporter(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and _call_name(node) in REPORTERS:
            return True
    return False


def _has_repairs_parameter(function: ast.AST) -> bool:
    args = function.args
    names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return "repairs" in names


def _innermost_functions(tree: ast.AST):
    """Yield (function, direct calls) with calls attributed to the innermost function."""
    stack: list = []

    def visit(node):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_function:
            stack.append((node, []))
        if isinstance(node, ast.Call) and stack:
            stack[-1][1].append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)
        if is_function:
            yield_list.append(stack.pop())

    yield_list: list = []
    visit(tree)
    return yield_list


def _collect_sites():
    sites = []
    for path in sorted(BACKEND.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("backend/data/", "backend/logs/")):
            continue  # runtime roots (for example the Lean workspace), not production code
        if relative == "backend/shared/json_parser.py" or relative == "backend/shared/json_key_recovery.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for function, calls in _innermost_functions(tree):
            if any(_collects_repairs(call) for call in calls):
                sites.append((relative, function))
    return sites


def test_every_repair_collecting_function_publishes_or_propagates():
    unpublished = []
    for relative, function in _collect_sites():
        if (relative, function.name) in PROPAGATING_HELPERS:
            continue
        if _has_repairs_parameter(function) or _references_reporter(function):
            continue
        unpublished.append(f"{relative}:{function.lineno} {function.name}")
    assert not unpublished, (
        "These functions collect model-output repairs but never publish them "
        "(report_model_output_normalizations) or pass them to their caller: "
        + ", ".join(unpublished)
    )


def test_inventory_finds_the_known_sites_and_has_no_stale_exemptions():
    sites = {(relative, function.name) for relative, function in _collect_sites()}
    assert ("backend/autonomous/agents/proof_identification_agent.py", "_parse_candidate_payload") in sites
    assert ("backend/shared/critique_prompts.py", "parse_critique_response") in sites
    stale = [key for key in PROPAGATING_HELPERS if key not in sites]
    assert not stale, f"Stale propagating-helper exemptions: {stale}"


def test_parse_calls_without_a_repairs_list_get_no_repairs():
    # The parser itself enforces the other half: without a list, no repair runs.
    import json

    import pytest

    from backend.shared.json_parser import parse_json

    with pytest.raises(json.JSONDecodeError):
        parse_json('{"reasoning": "x",\n s "submission": "y"}')
