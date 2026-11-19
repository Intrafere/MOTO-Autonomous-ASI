"""Request auth must decide on the routed path, never on a Host-derived URL.

Starlette releases before 1.x rebuild ``request.url`` from the client-controlled
``Host`` header, so a Host such as ``x/api/features?`` makes ``request.url.path``
read as an allowlisted route while the router still dispatches the real,
protected path. The checks here hold whichever Starlette version is installed.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.api import middleware
from backend.api.middleware import DESKTOP_API_TOKEN_HEADER, setup_middleware
from backend.api.proxy_auth import routed_path, routed_query
from backend.api.routes import websocket as websocket_routes

CRAFTED_HOSTS = ["x/api/features?", "x/api/features#", "x/api/health?", "x/health?"]


def _protected_app() -> FastAPI:
    app = FastAPI()
    setup_middleware(app)

    @app.get("/api/secret")
    async def secret():
        return {"secret": "protected"}

    @app.get("/api/proofs/{proof_id}/details")
    async def proof_details(proof_id: str):
        return {"proof_id": proof_id}

    return app


def test_routed_helpers_ignore_the_host_header():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/secret",
        "query_string": b"a=1",
        "headers": [(b"host", b"x/api/features?")],
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
    }
    assert routed_path(scope) == "/api/secret"
    assert routed_query(scope) == b"a=1"


@pytest.mark.parametrize("host", CRAFTED_HOSTS)
def test_desktop_token_is_required_whatever_the_host_header_says(monkeypatch, host):
    monkeypatch.setattr(middleware.system_config, "generic_mode", False)
    monkeypatch.setattr(middleware.system_config, "desktop_api_token", "test-desktop-token")
    client = TestClient(_protected_app())

    assert client.get("/api/secret", headers={"host": host}).status_code == 401
    assert (
        client.get(
            "/api/secret",
            headers={"host": host, DESKTOP_API_TOKEN_HEADER: "test-desktop-token"},
        ).status_code
        == 200
    )


def test_desktop_public_export_match_uses_the_routed_path(monkeypatch):
    monkeypatch.setattr(middleware.system_config, "generic_mode", False)
    monkeypatch.setattr(middleware.system_config, "desktop_api_token", "test-desktop-token")
    client = TestClient(_protected_app())

    response = client.get(
        "/api/proofs/proof_001/details",
        headers={"host": "x/api/proofs/proof_001/certificate?"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize("host", CRAFTED_HOSTS)
def test_generic_proxy_auth_is_required_whatever_the_host_header_says(monkeypatch, host):
    monkeypatch.setattr(middleware.system_config, "generic_mode", True)
    monkeypatch.setattr(middleware.system_config, "instance_id", "inst")
    monkeypatch.setattr(middleware.system_config, "internal_proxy_secret", "test-proxy-secret")
    client = TestClient(_protected_app())

    assert client.get("/api/secret", headers={"host": host}).status_code == 401


@pytest.mark.parametrize("host", CRAFTED_HOSTS)
def test_generic_websocket_rejects_unsigned_connection_with_crafted_host(monkeypatch, host):
    monkeypatch.setattr(websocket_routes.system_config, "generic_mode", True)
    monkeypatch.setattr(websocket_routes.system_config, "instance_id", "inst")
    monkeypatch.setattr(websocket_routes.system_config, "internal_proxy_secret", "test-proxy-secret")
    app = FastAPI()
    app.include_router(websocket_routes.router)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"host": host}):
            pass
    assert websocket_routes.manager.active_connections == []
