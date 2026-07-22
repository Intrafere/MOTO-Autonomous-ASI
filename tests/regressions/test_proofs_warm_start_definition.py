import ast
from pathlib import Path


def test_lean4_warm_start_helper_has_single_definition():
    route_path = Path(__file__).resolve().parents[2] / "backend" / "api" / "routes" / "proofs.py"
    module = ast.parse(route_path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_schedule_lean4_warm_start"
    ]
    assert len(definitions) == 1
