"""Hosted requests are authenticated before their body is read, and the read is capped.

The body used to be read in full before the proxy signature was checked, and the
size limit only applied when ``Content-Length`` was present, so an unsigned
chunked upload could fill the sandbox's memory.
"""
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.api import middleware, proxy_auth
from backend.api.middleware import setup_middleware
from backend.api.proxy_auth import (
    PROXY_BODY_SHA256_HEADER,
    PROXY_INSTANCE_HEADER,
    PROXY_SIGNATURE_HEADER,
    PROXY_TIMESTAMP_HEADER,
    build_proxy_signature,
    hash_proxy_body,
)

SECRET = "test-proxy-secret"
INSTANCE = "inst"
PATH = "/api/echo"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(middleware.system_config, "generic_mode", True)
    monkeypatch.setattr(middleware.system_config, "instance_id", INSTANCE)
    monkeypatch.setattr(middleware.system_config, "internal_proxy_secret", SECRET)
    monkeypatch.setattr(middleware.system_config, "generic_max_request_bytes", 64)
    monkeypatch.setattr(proxy_auth, "_SEEN_PROXY_SIGNATURES", {})
    app = FastAPI()
    setup_middleware(app)

    @app.post(PATH)
    async def echo(request: Request):
        return {"received": (await request.body()).decode("utf-8")}

    return TestClient(app)


def _signed_headers(body: bytes, *, claimed_body: bytes | None = None) -> dict[str, str]:
    timestamp = str(int(time.time()))
    body_hash = hash_proxy_body(body if claimed_body is None else claimed_body)
    signature = build_proxy_signature(
        secret=SECRET,
        instance_id=INSTANCE,
        timestamp=timestamp,
        method="POST",
        path=PATH,
        query_string="",
        body_hash=body_hash,
    )
    return {
        PROXY_INSTANCE_HEADER: INSTANCE,
        PROXY_TIMESTAMP_HEADER: timestamp,
        PROXY_SIGNATURE_HEADER: signature,
        PROXY_BODY_SHA256_HEADER: body_hash,
    }


def test_signed_request_reaches_the_route_with_its_body(client):
    body = b'{"hello": "world"}'

    response = client.post(PATH, content=body, headers=_signed_headers(body))

    assert response.status_code == 200
    assert response.json() == {"received": body.decode("utf-8")}


def test_unsigned_request_is_rejected_without_reading_its_body(client, monkeypatch):
    async def must_not_read(request):
        raise AssertionError("the body of an unauthenticated request was read")

    monkeypatch.setattr(middleware, "_read_generic_body", must_not_read)
    body = b"x" * 32

    missing = client.post(PATH, content=body, headers={PROXY_BODY_SHA256_HEADER: hash_proxy_body(body)})
    forged = client.post(
        PATH,
        content=body,
        headers={**_signed_headers(body), PROXY_SIGNATURE_HEADER: "0" * 64},
    )

    assert missing.status_code == 401
    assert forged.status_code == 403


def test_signed_request_with_a_different_body_is_rejected(client):
    response = client.post(
        PATH,
        content=b"actual body",
        headers=_signed_headers(b"actual body", claimed_body=b"claimed body"),
    )

    assert response.status_code == 403


def test_chunked_body_over_the_limit_is_rejected_without_content_length(client):
    chunks = [b"a" * 40, b"b" * 40]
    body = b"".join(chunks)

    response = client.post(PATH, content=iter(chunks), headers=_signed_headers(body))

    assert "content-length" not in {name.lower() for name in response.request.headers}
    assert response.status_code == 413
