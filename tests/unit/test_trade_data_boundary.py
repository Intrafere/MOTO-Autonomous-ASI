from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.shared.trade_data_boundary as boundary
from backend.shared.log_redaction import redact_log_text, redact_sensitive_values
from backend.shared.trade_data_boundary import (
    TRADE_PROHIBITED_FIELDS,
    TradeDataBoundaryViolation,
    apply_trade_data_log_boundary,
    assert_trade_safe,
    classify_trade_field,
    install_trade_data_log_boundary,
    redact_trade_payload,
    redact_trade_text,
    trade_safe_error_detail,
)


HEX = "0123456789abcdef0123456789abcdef"

SECRET_CLASSES = {
    "bearer_token": f"Authorization: Bearer abc.DEF-ghi_{HEX}",
    "openrouter_key": f"key sk-or-v1-{HEX}",
    "credential_assignment": "api_key=supersecretvalue",
    "installation_id": f"moto_installation_{HEX}",
    "root_binding_id": f"moto_root_{HEX}",
    "client_batch_id": f"moto_batch_{HEX}",
    "selection_snapshot_id": f"moto_selection_{HEX}",
    "client_proof_id": f"moto_proof_{HEX}",
    "stable_store_id": f"moto_store_{HEX}",
    "windows_path": r"failed reading C:\Users\alice\moto\backend\data\proofs\proof_1.json now",
    "windows_forward_path": "at D:/research/private/notes.txt",
    "unc_path": r"share \\fileserver\research\private.lean here",
    "posix_home_path": "opened /home/alice/moto/backend/data ok",
    "macos_path": "opened /Users/alice/Library/moto.json ok",
    "tilde_path": "see ~/moto/backend/data please",
    "email": "account alice.researcher@example.org linked",
}

SENSITIVE_FRAGMENTS = {
    "bearer_token": f"abc.DEF-ghi_{HEX}",
    "openrouter_key": f"sk-or-v1-{HEX}",
    "credential_assignment": "supersecretvalue",
    "installation_id": HEX,
    "root_binding_id": HEX,
    "client_batch_id": HEX,
    "selection_snapshot_id": HEX,
    "client_proof_id": HEX,
    "stable_store_id": HEX,
    "windows_path": "alice",
    "windows_forward_path": "research/private",
    "unc_path": "fileserver",
    "posix_home_path": "alice",
    "macos_path": "alice",
    "tilde_path": "moto/backend/data",
    "email": "alice.researcher",
}


@pytest.fixture
def local_identity(monkeypatch):
    monkeypatch.setattr(boundary.getpass, "getuser", lambda: "zelda-researcher")
    monkeypatch.setattr(boundary.socket, "gethostname", lambda: "LAB-WORKSTATION-42.corp")
    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("COMPUTERNAME", raising=False)
    boundary._local_identity_tokens.cache_clear()
    yield
    boundary._local_identity_tokens.cache_clear()


@pytest.mark.parametrize("secret_class", sorted(SECRET_CLASSES))
def test_trade_text_redacts_every_secret_class(secret_class):
    text = SECRET_CLASSES[secret_class]
    fragment = SENSITIVE_FRAGMENTS[secret_class]

    redacted = redact_trade_text(text)

    assert fragment not in redacted
    assert fragment not in json.dumps(redact_trade_payload({"note": text}))
    with pytest.raises(TradeDataBoundaryViolation) as caught:
        assert_trade_safe({"note": text})
    assert fragment not in str(caught.value)
    assert all(code.startswith("sensitive_value:") for code in caught.value.reason_codes)


def test_local_username_and_hostname_are_removed(local_identity):
    text = "user zelda-researcher on LAB-WORKSTATION-42 (LAB-WORKSTATION-42.corp)"

    redacted = redact_trade_text(text)

    assert "zelda" not in redacted.lower()
    assert "workstation" not in redacted.lower()
    with pytest.raises(TradeDataBoundaryViolation) as caught:
        assert_trade_safe([text])
    assert "sensitive_value:local_identity" in caught.value.reason_codes


def test_generic_local_names_are_not_treated_as_identity(monkeypatch):
    monkeypatch.setattr(boundary.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(boundary.socket, "gethostname", lambda: "localhost")
    for name in ("USERNAME", "USER", "COMPUTERNAME"):
        monkeypatch.delenv(name, raising=False)
    boundary._local_identity_tokens.cache_clear()
    try:
        assert redact_trade_text("the user prompt") == "the user prompt"
    finally:
        boundary._local_identity_tokens.cache_clear()


ALL_PROHIBITED_FIELDS = sorted(
    (field_class, name)
    for field_class, names in TRADE_PROHIBITED_FIELDS.items()
    for name in names
)


@pytest.mark.parametrize(("field_class", "name"), ALL_PROHIBITED_FIELDS)
def test_every_prohibited_field_is_excluded_and_rejected(field_class, name):
    secret = "PRIVATE-RESEARCH-CONTENT-7f3a"
    payload = {"kept": 1, "nested": [{name: secret}]}

    redacted = redact_trade_payload(payload)

    assert secret not in json.dumps(redacted)
    assert redacted["kept"] == 1
    with pytest.raises(TradeDataBoundaryViolation) as caught:
        assert_trade_safe(payload)
    assert f"prohibited_field:{field_class}" in caught.value.reason_codes
    assert secret not in str(caught.value)


def test_field_names_are_normalized_across_conventions():
    assert classify_trade_field("userPrompt") == "prompt"
    assert classify_trade_field("Lean-Code") == "proof_body"
    assert classify_trade_field("SOURCE_TITLE") == "source_identity"
    assert classify_trade_field("proof_count") is None


def test_proof_bodies_keep_only_shape_metadata():
    redacted = redact_trade_payload({"lean_code": "theorem x : True := trivial"})

    assert redacted == {"lean_code": {"excluded": True, "length": 27}}


def test_safe_metadata_payload_passes():
    assert_trade_safe(
        {
            "schema_version": "moto-proof-trade-ledger-v1",
            "counts": {"selected": 1},
            "canonical_lean_code_hash": "a" * 64,
            "reason_codes": ["batch_blocked_ineligible"],
        }
    )


def test_payload_depth_is_bounded():
    payload: dict = {}
    cursor = payload
    for _ in range(20):
        cursor["child"] = {}
        cursor = cursor["child"]

    with pytest.raises(TradeDataBoundaryViolation) as caught:
        assert_trade_safe(payload)
    assert "payload_too_deep" in caught.value.reason_codes
    assert "[bounded]" in json.dumps(redact_trade_payload(payload))


def test_error_detail_never_contains_exception_text():
    exc = RuntimeError(r"cannot read C:\Users\alice\secret.json token=abc")

    detail = trade_safe_error_detail("SyntheticLib4 refresh failed", exc)

    assert detail == "SyntheticLib4 refresh failed (RuntimeError)"


def test_base_redaction_is_idempotent_and_keeps_occurrence_ids():
    text = f"token=abc moto_installation_{HEX} search moto:moto_proof_{HEX}\nnext"

    once = redact_sensitive_values(text)

    assert redact_sensitive_values(once) == once
    assert redact_log_text(redact_log_text(text)) == redact_log_text(text)
    assert f"moto_proof_{HEX}" in once
    assert HEX not in once.replace(f"moto_proof_{HEX}", "")
    assert "\n" in once


def _record(name: str, msg: str, args=(), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(name, logging.ERROR, __file__, 1, msg, args, exc_info)


def test_log_boundary_redacts_credentials_for_every_logger():
    record = apply_trade_data_log_boundary(
        _record("backend.compiler.core", "auth %s for %s", ("Bearer abcdef", f"moto_root_{HEX}"))
    )

    assert record.getMessage() == "auth Bearer [redacted] for moto_root_[redacted]"
    assert record.args == ()


def test_log_boundary_leaves_clean_records_untouched():
    record = _record("backend.compiler.core", "processed %d items in %s", (3, r"C:\data"))

    apply_trade_data_log_boundary(record)

    assert record.args == (3, r"C:\data")
    assert record.msg == "processed %d items in %s"


def test_trade_logger_drops_paths_ids_and_tracebacks():
    try:
        raise ValueError(r"bad path C:\Users\alice\proofs")
    except ValueError:
        import sys

        exc_info = sys.exc_info()
    record = _record(
        "backend.shared.syntheticlib_trade_eligibility",
        "store %s proof %s",
        (r"C:\Users\alice\proofs", f"moto_proof_{HEX}"),
        exc_info,
    )

    apply_trade_data_log_boundary(record)
    formatted = logging.Formatter("%(message)s").format(record)

    assert "alice" not in formatted
    assert HEX not in formatted
    assert "ValueError (traceback withheld by the proof-trade data boundary)" in formatted
    assert "Traceback" not in formatted


def test_non_trade_traceback_is_credential_redacted():
    try:
        raise RuntimeError("token=leaked-value")
    except RuntimeError:
        import sys

        exc_info = sys.exc_info()
    record = _record("backend.api.routes.update", "failed", (), exc_info)

    apply_trade_data_log_boundary(record)
    formatted = logging.Formatter("%(message)s").format(record)

    assert "leaked-value" not in formatted
    assert "Traceback" in formatted


def test_install_is_idempotent_and_applies_to_new_records(caplog):
    previous = logging.getLogRecordFactory()
    try:
        install_trade_data_log_boundary()
        installed = logging.getLogRecordFactory()
        install_trade_data_log_boundary()
        assert logging.getLogRecordFactory() is installed

        with caplog.at_level(logging.INFO, logger="backend.shared.proof_trade_ledger"):
            logging.getLogger("backend.shared.proof_trade_ledger").info(
                "ledger at %s", "/home/alice/moto/backend/data"
            )
        assert "alice" not in caplog.text
        assert "[path-redacted]" in caplog.text
    finally:
        logging.setLogRecordFactory(previous)


def test_uncaught_errors_return_generic_detail(monkeypatch):
    from backend.api import middleware
    from backend.api.middleware import DESKTOP_API_TOKEN_HEADER, setup_middleware

    monkeypatch.setattr(middleware.system_config, "generic_mode", False)
    monkeypatch.setattr(middleware.system_config, "desktop_api_token", "test-desktop-token")
    app = FastAPI()
    setup_middleware(app)

    @app.get("/api/boom")
    async def boom():
        raise RuntimeError(r"secret C:\Users\alice\data token=abc")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/boom", headers={DESKTOP_API_TOKEN_HEADER: "test-desktop-token"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "alice" not in response.text
    assert "token" not in response.text
