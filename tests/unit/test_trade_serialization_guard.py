"""Static guard: proof-trade code never wholesale-dumps proof/library/search models.

The canonical plan forbids feeding ``model_dump()`` output of proof records,
library entries, certificates, or proof-search records into anything that may
leave MOTO. Until the Build 6 allowlisted serializer exists, every dump call in
trade modules is pinned to a reviewed receiver allowlist so a new call site
fails here until it is explicitly reviewed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED = REPO_ROOT / "backend" / "shared"
TRADE_MODULES = (
    "legacy_port_receipt.py",
    "proof_trade_authorization.py",
    "proof_trade_ledger.py",
    "syntheticlib_legacy_port.py",
    "syntheticlib_source_selection.py",
    "syntheticlib_trade_compatibility.py",
    "syntheticlib_trade_eligibility.py",
    "trade_data_boundary.py",
)
DUMP_METHODS = {"model_dump", "model_dump_json", "dict", "json"}

# Reviewed (module, receiver) pairs. All are ledger-owned models or receipt /
# dependency models validated locally; none is a whole proof/library/search record.
ALLOWED_DUMP_RECEIVERS = {
    ("proof_trade_ledger.py", "working"),
    ("proof_trade_ledger.py", "validated"),
    ("proof_trade_ledger.py", "entry"),
    ("legacy_port_receipt.py", "LegacyPortReceipt.model_validate(payload)"),
    ("syntheticlib_trade_eligibility.py", "dependency"),
    ("syntheticlib_trade_eligibility.py", "legacy_receipt"),
    ("syntheticlib_trade_eligibility.py", "novelty_receipt"),
    ("syntheticlib_trade_eligibility.py", "receipt"),
}

# Whole-record models that must never be imported by trade modules.
FORBIDDEN_IMPORTS = {
    "ProofLibraryEntry",
    "ProofLibraryResponse",
    "ProofCertificateResponse",
    "UnifiedProofSearchRecord",
    "ProofSearchResponse",
}
# ProofRecord is read only by the canonical eligibility/source-policy projection.
PROOF_RECORD_READERS = {
    "legacy_port_receipt.py",
    "syntheticlib_legacy_port.py",
    "syntheticlib_trade_eligibility.py",
    "syntheticlib_source_selection.py",
}


def _tree(name: str) -> ast.Module:
    return ast.parse((SHARED / name).read_text(encoding="utf-8"), filename=name)


def _dump_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in DUMP_METHODS
        ):
            yield node


@pytest.mark.parametrize("module", TRADE_MODULES)
def test_trade_modules_only_dump_reviewed_receivers(module) -> None:
    unexpected = []
    for call in _dump_calls(_tree(module)):
        name = ast.unparse(call.func.value)
        if (module, name) not in ALLOWED_DUMP_RECEIVERS:
            unexpected.append(f"{module}:{call.lineno} {name}.{call.func.attr}()")
    assert not unexpected, "Unreviewed model dumps in trade code: " + ", ".join(unexpected)


@pytest.mark.parametrize("module", TRADE_MODULES)
def test_trade_modules_do_not_import_whole_record_models(module) -> None:
    imported = set()
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module and "proof_search" in node.module:
            pytest.fail(f"{module} imports proof-search records ({node.module})")
    assert not imported & FORBIDDEN_IMPORTS
    if module not in PROOF_RECORD_READERS:
        assert "ProofRecord" not in imported


def test_trade_routes_never_dump_models() -> None:
    routes = REPO_ROOT / "backend" / "api" / "routes" / "syntheticlib4.py"
    tree = ast.parse(routes.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and "trade" in node.name:
            offenders.extend(
                f"{node.name}:{call.lineno}"
                for call in _dump_calls(node)
                if call.func.attr in {"model_dump", "model_dump_json"}
            )
    assert not offenders


# Usage attestations are a separate authority from proof trade (plan item 15):
# trade code never imports the attestation writer, the SyntheticLib4 corpus
# client, or names the attestation schema/storage.
ATTESTATION_MODULES = {
    "backend.shared.proof_search.tool_adapter",
    "backend.shared.syntheticlib4_client",
}
ATTESTATION_MARKERS = ("proof_search_usage_attestation", "_persist_usage_attestation")


@pytest.mark.parametrize("module", TRADE_MODULES)
def test_trade_modules_stay_separate_from_usage_attestations(module) -> None:
    source = (SHARED / module).read_text(encoding="utf-8")
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ATTESTATION_MODULES, f"{module} imports {node.module}"
        if isinstance(node, ast.Import):
            assert not {alias.name for alias in node.names} & ATTESTATION_MODULES
    assert not any(marker in source for marker in ATTESTATION_MARKERS)
