"""Route-inventory gate: every SyntheticLib backend route is classified and denied.

New SyntheticLib routes (or new gated proof routes) fail this test until they are
given a sample request and an authority class, so no private route can ship
without denial coverage.
"""
from __future__ import annotations

import ast
import inspect
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.api.routes import proofs, syntheticlib4
from backend.shared.syntheticlib_product import require_syntheticlib_developer_tooling
from tests.unit.syntheticlib_gate_cases import (
    DEVELOPER_TOOLING_DENIED_CONFIGS,
    RUNTIME_DENIED_CONFIGS,
    apply_lifecycle,
    lifecycle_config,  # noqa: F401 - pytest fixture
)

_SNAPSHOT_ID = "moto_selection_" + "a" * 32
_BATCH_ID = "moto_batch_" + "a" * 32

# (method, path) -> (authority class, JSON body, query params)
_ROUTE_SAMPLES: dict[tuple[str, str], tuple[str, dict | None, dict | None]] = {
    ("GET", "/api/syntheticlib4/status"): ("status_stub", None, None),
    ("POST", "/api/syntheticlib4/auth/start"): ("runtime", {}, None),
    ("POST", "/api/syntheticlib4/auth/exchange"): ("runtime", {"code": "c", "state": "s"}, None),
    ("POST", "/api/syntheticlib4/api-key"): ("runtime", {"api_key": "sl4-test-placeholder"}, None),
    ("DELETE", "/api/syntheticlib4/auth"): ("runtime", None, None),
    ("GET", "/api/syntheticlib4/releases"): ("runtime", None, None),
    ("POST", "/api/syntheticlib4/refresh"): ("runtime", None, None),
    ("POST", "/api/syntheticlib4/import-local-snapshot"): ("runtime", {"source_name": "snap"}, None),
    ("POST", "/api/syntheticlib4/reindex"): ("runtime", None, None),
    ("POST", "/api/syntheticlib4/retrieve-batch"): ("runtime", {"query": "finite"}, None),
    ("GET", "/api/syntheticlib4/account/proofs"): ("runtime", None, None),
    ("GET", "/api/syntheticlib4/account/proofs/search"): ("runtime", None, {"q": "x"}),
    ("GET", "/api/syntheticlib4/developer/trade-evidence"): ("runtime", None, None),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/snapshots"): (
        "runtime",
        {"client_proof_ids": ["moto_proof_" + "a" * 32]},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/batches/prepare"): (
        "runtime",
        {"snapshot_id": _SNAPSHOT_ID},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/legacy-port/migrate"): (
        "runtime",
        {"confirm": True, "dry_run": True},
        None,
    ),
    ("GET", "/api/syntheticlib4/developer/trade-ledger"): ("developer", None, None),
    ("GET", "/api/syntheticlib4/developer/trade-authorization"): ("developer", None, None),
    ("GET", "/api/syntheticlib4/developer/trade-ledger/entries"): ("developer", None, None),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/snapshots/reconcile"): (
        "developer",
        {"snapshot_id": _SNAPSHOT_ID},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/snapshots/cancel"): (
        "developer",
        {"snapshot_id": _SNAPSHOT_ID},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/batches/cancel"): (
        "developer",
        {"client_batch_id": _BATCH_ID},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/archive-reset"): (
        "developer",
        {"confirm": True},
        None,
    ),
    ("POST", "/api/syntheticlib4/developer/trade-ledger/archive/delete"): (
        "developer",
        {"confirm": True},
        None,
    ),
}

_GATE_CALLS = {"require_syntheticlib_runtime", "require_syntheticlib_developer_tooling"}
# Ledger routes delegate to a helper that authorizes before touching the ledger.
_GATED_HELPERS = {"_run_ledger_operation"}

# Every name a SyntheticLib handler could reach after its gate.
_SIDE_EFFECT_TARGETS = {
    syntheticlib4: [
        "syntheticlib4_client",
        "proof_search_service",
        "project_installation_proof_stores",
        "get_authorization_readiness",
        "get_ledger_status",
        "list_ledger_entries",
        "create_selection_snapshot",
        "reconcile_selection_snapshot",
        "cancel_selection_snapshot",
        "prepare_batch",
        "cancel_batch",
        "archive_and_reset_local_ledger",
        "delete_archived_local_ledgers",
        "migrate_legacy_proofs",
    ],
    proofs: ["_get_scoped_proof_database"],
}


def _calls_gate(endpoint, *, helpers: frozenset[str] | set[str] = frozenset()) -> bool:
    tree = ast.parse(inspect.getsource(endpoint).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _GATE_CALLS or node.func.id in helpers:
                return True
    return False


def _gated_proof_routes() -> list[APIRoute]:
    return [
        route
        for route in proofs.router.routes
        if isinstance(route, APIRoute) and _calls_gate(route.endpoint)
    ]


def _inventory() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for route in [*syntheticlib4.router.routes, *_gated_proof_routes()]:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods):
            keys.append((method, route.path))
    return sorted(keys)


def _concrete(path: str) -> str:
    return path.replace("{proof_id}", "proof_gate_probe")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(syntheticlib4.router)
    app.include_router(proofs.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def side_effects():
    reached: list[str] = []
    patches = []
    for module, names in _SIDE_EFFECT_TARGETS.items():
        for name in names:
            sentinel = mock.MagicMock(name=name)
            sentinel.side_effect = lambda *a, _n=name, **k: reached.append(_n)
            patches.append(mock.patch.object(module, name, sentinel))
    for patcher in patches:
        patcher.start()
    yield reached
    for patcher in patches:
        patcher.stop()


def test_every_syntheticlib_route_is_classified() -> None:
    inventory = set(_inventory())
    unclassified = sorted(inventory - set(_ROUTE_SAMPLES))
    stale = sorted(set(_ROUTE_SAMPLES) - inventory)
    assert not unclassified, f"classify new SyntheticLib routes: {unclassified}"
    assert not stale, f"remove stale route samples: {stale}"


def test_gated_proof_route_discovery_scans_every_proof_route() -> None:
    # No /api/proofs route is SyntheticLib-gated today; any future one must be
    # discovered here and then classified in _ROUTE_SAMPLES.
    proof_routes = [route for route in proofs.router.routes if isinstance(route, APIRoute)]
    assert proof_routes
    assert _calls_gate(syntheticlib4.get_syntheticlib_trade_ledger_status)


def test_every_non_status_route_calls_a_gate() -> None:
    helper = inspect.signature(syntheticlib4._run_ledger_operation)
    assert helper.parameters["authorize"].default is require_syntheticlib_developer_tooling
    for route in syntheticlib4.router.routes:
        if not isinstance(route, APIRoute) or route.path.endswith("/status"):
            continue
        assert _calls_gate(route.endpoint, helpers=_GATED_HELPERS), route.path


def _request(client: TestClient, key: tuple[str, str]):
    method, path = key
    _, body, params = _ROUTE_SAMPLES[key]
    return client.request(method, _concrete(path), json=body, params=params)


@pytest.mark.parametrize("key", sorted(_ROUTE_SAMPLES))
@pytest.mark.parametrize(("product_state", "authorized", "generic_mode"), DEVELOPER_TOOLING_DENIED_CONFIGS)
def test_route_denied_without_authority_before_any_side_effect(
    lifecycle_config, client, side_effects, key, product_state, authorized, generic_mode
) -> None:
    authority, _, _ = _ROUTE_SAMPLES[key]
    config = (product_state, authorized, generic_mode)
    if authority in {"runtime", "status_stub"} and config not in RUNTIME_DENIED_CONFIGS:
        pytest.skip("product-authority route is allowed after launch")
    apply_lifecycle(lifecycle_config, product_state, authorized, generic_mode)

    response = _request(client, key)

    if authority == "status_stub":
        assert response.status_code == 200
        payload = response.json()
        assert payload["runtime_available"] is False
        assert payload["status"]["credential_configured"] is False
        assert payload["current_release"] is None
    else:
        assert response.status_code == 403, (key, response.status_code)
    assert side_effects == []
