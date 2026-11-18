"""Tests for the `moto` command-line control surface (moto_cli.py + cli_control)."""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import moto_cli
from backend.shared import cli_control
from backend.shared.config import system_config

TOKEN = "test-token-123"


class _FakeBackend:
    """Minimal stand-in for the MOTO REST surface the CLI uses."""

    def __init__(self, instance_id: str = "default"):
        self.instance_id = instance_id
        self.requests: list[dict] = []
        self.autonomous_running = False
        # (METHOD, path) -> payload, or callable(request) -> (status, payload[, content_type])
        self.routes: dict = {}
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep pytest output quiet
                return

            def _send(self, status: int, payload, content_type: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:  # the client gave up (timeout test)
                    pass

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type") or ""
                body = json.loads(raw) if raw and "json" in content_type else None
                path = self.path.split("?")[0]
                record = {"method": method, "path": self.path, "body": body, "raw": raw,
                          "content_type": content_type, "token": self.headers.get(moto_cli.TOKEN_HEADER)}
                backend.requests.append(record)
                if path == "/api/health":
                    return self._send(200, {"status": "healthy", "instance_id": backend.instance_id, "version": "9.9"})
                if self.headers.get(moto_cli.TOKEN_HEADER) != TOKEN:
                    return self._send(401, {"detail": "Missing or invalid desktop API token."})
                route = backend.routes.get((method, path))
                if route is not None:
                    if callable(route):
                        return self._send(*route(record))
                    return self._send(200, route)
                if path == "/api/auto-research/status":
                    return self._send(200, {"is_running": backend.autonomous_running})
                if path.endswith("/status"):
                    return self._send(200, {"is_running": False})
                if path == "/api/auto-research/start":
                    if backend.autonomous_running:
                        return self._send(400, {"detail": "Aggregator is running. Stop it first."})
                    backend.autonomous_running = True
                    return self._send(200, {"success": True, "message": "started"})
                if path == "/api/auto-research/paper/p1":
                    return self._send(200, {"paper_id": "p1", "content": "PAPER BODY"})
                if path == "/api/token-stats":
                    return self._send(200, {"success": True})
                if path == "/api/slow":
                    import time as _time
                    _time.sleep(2)
                    return self._send(200, {})
                return self._send(404, {"detail": "Not Found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_PATCH(self):
                self._handle("PATCH")

            def do_DELETE(self):
                self._handle("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _write_control(data_root: Path, backend: _FakeBackend) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / moto_cli.CONTROL_FILE_NAME).write_text(json.dumps({
        "schema": "moto-cli-control-v1",
        "instance_id": backend.instance_id,
        "pid": 4242,
        "backend_url": backend.url,
        "frontend_url": "http://localhost:5173",
        "desktop_api_token": TOKEN,
        "data_root": str(data_root),
    }), encoding="utf-8")


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    code = moto_cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def fake_moto(tmp_path):
    data_root = tmp_path / "data"
    with _FakeBackend() as backend:
        _write_control(data_root, backend)
        yield backend, data_root


def test_module_name_never_shadows_pypi_moto():
    assert Path(moto_cli.__file__).name == "moto_cli.py"
    assert not (Path(moto_cli.__file__).parent / "moto.py").exists()
    assert not (Path(moto_cli.__file__).parent / "moto").is_dir()


def test_status_reports_not_running_with_exit_3(tmp_path, capsys):
    code, out, _ = _run(capsys, "status", "--json", "--data-root", str(tmp_path / "empty"))
    assert code == moto_cli.EXIT_NOT_RUNNING
    assert json.loads(out)["running"] is False


def test_stale_control_file_is_not_treated_as_running(tmp_path, capsys):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / moto_cli.CONTROL_FILE_NAME).write_text(json.dumps({
        "instance_id": "default", "backend_url": "http://127.0.0.1:9", "desktop_api_token": TOKEN,
    }), encoding="utf-8")
    code, _, _ = _run(capsys, "status", "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_NOT_RUNNING


def test_status_uses_token_and_reports_active_workflow(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.autonomous_running = True
    code, out, _ = _run(capsys, "status", "--json", "--data-root", str(data_root))
    payload = json.loads(out)
    assert code == 0
    assert payload["active_workflow"] == "autonomous"
    assert payload["instance"]["version"] == "9.9"
    protected = [r for r in backend.requests if r["path"] != "/api/health"]
    assert protected and all(r["token"] == TOKEN for r in protected)


def test_start_reuses_saved_template_with_new_prompt(fake_moto, capsys):
    backend, data_root = fake_moto
    template_dir = data_root / moto_cli.START_TEMPLATE_DIR_NAME
    template_dir.mkdir()
    (template_dir / "autonomous.json").write_text(json.dumps({
        "request": {"user_research_prompt": "old", "validator_model": "m", "allow_research_papers": True},
    }), encoding="utf-8")
    code, out, _ = _run(
        capsys, "autonomous", "start", "--prompt", "new prompt", "--no-papers",
        "--set", "tier3_enabled=true", "--json", "--data-root", str(data_root),
    )
    assert code == 0, out
    start = next(r for r in backend.requests if r["path"] == "/api/auto-research/start")
    assert start["body"]["user_research_prompt"] == "new prompt"
    assert start["body"]["validator_model"] == "m"
    assert start["body"]["allow_research_papers"] is False
    assert start["body"]["tier3_enabled"] is True


def test_start_without_saved_settings_explains_how_to_get_them(fake_moto, capsys):
    _, data_root = fake_moto
    code, out, _ = _run(capsys, "aggregator", "start", "--prompt", "x", "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE
    error = json.loads(out)["error"]
    assert "--model" in error and "start Aggregator once" in error


def test_dry_run_prints_request_without_starting(fake_moto, tmp_path, capsys):
    backend, data_root = fake_moto
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"validator_model": "m"}), encoding="utf-8")
    code, out, _ = _run(capsys, "autonomous", "start", "--prompt", "p", "--config", str(config),
                        "--dry-run", "--json", "--data-root", str(data_root))
    assert code == 0
    assert json.loads(out) == {"validator_model": "m", "user_research_prompt": "p"}
    assert not any(r["path"].endswith("/start") for r in backend.requests)


def test_rejected_request_returns_exit_4_with_json_detail(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.autonomous_running = True
    config_args = ["--set", "validator_model=m", "--prompt", "p"]
    (data_root / moto_cli.START_TEMPLATE_DIR_NAME).mkdir()
    (data_root / moto_cli.START_TEMPLATE_DIR_NAME / "autonomous.json").write_text(
        json.dumps({"request": {"user_research_prompt": "p"}}), encoding="utf-8")
    code, out, _ = _run(capsys, "autonomous", "start", *config_args, "--json", "--data-root", str(data_root))
    payload = json.loads(out)
    assert code == moto_cli.EXIT_REJECTED
    assert payload["detail"]["status"] == 400
    assert "Stop it first" in payload["error"]


def test_paper_prints_plain_text_for_humans(fake_moto, capsys):
    _, data_root = fake_moto
    code, out, _ = _run(capsys, "autonomous", "paper", "p1", "--data-root", str(data_root))
    assert code == 0
    assert out.strip() == "PAPER BODY"


def test_clear_requires_yes(fake_moto, capsys):
    backend, data_root = fake_moto
    code, _, err = _run(capsys, "autonomous", "clear", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE
    assert "--yes" in err
    assert not any("/clear" in r["path"] for r in backend.requests)


def test_several_instances_require_choice(tmp_path, capsys, monkeypatch):
    with _FakeBackend("alpha") as first, _FakeBackend("beta") as second:
        _write_control(tmp_path / "a", first)
        _write_control(tmp_path / "b", second)
        monkeypatch.setattr(moto_cli, "candidate_data_roots", lambda explicit=None: [tmp_path / "a", tmp_path / "b"])
        code, _, err = _run(capsys, "autonomous", "status")
        assert code == moto_cli.EXIT_USAGE and "--instance" in err
        code, out, _ = _run(capsys, "autonomous", "status", "--instance", "beta", "--json")
        assert code == 0 and json.loads(out) == {"is_running": False}


@pytest.mark.parametrize("raw,expected", [
    ("/api/health", "/api/health"),
    ("api/proofs/runs", "/api/proofs/runs"),
    ("C:/Program Files/Git/api/health", "/api/health"),  # Git Bash path rewriting
])
def test_route_paths_are_normalized(raw, expected):
    assert moto_cli.normalize_route_path(raw) == expected


def test_route_path_rejects_non_routes():
    with pytest.raises(moto_cli.CliError):
        moto_cli.normalize_route_path("D:/data/file.txt")


def test_dry_run_works_while_moto_is_closed(tmp_path, capsys):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"request": {"user_prompt": "saved"}}), encoding="utf-8")
    code, out, _ = _run(capsys, "aggregator", "start", "--config", str(config), "--dry-run", "--json",
                        "--data-root", str(tmp_path / "closed"))
    assert code == 0 and json.loads(out) == {"user_prompt": "saved"}


def test_control_file_with_an_old_token_is_ignored(tmp_path, capsys):
    with _FakeBackend() as backend:
        _write_control(tmp_path / "data", backend)
        control = tmp_path / "data" / moto_cli.CONTROL_FILE_NAME
        payload = json.loads(control.read_text(encoding="utf-8"))
        payload["desktop_api_token"] = "token-from-an-earlier-launch"
        control.write_text(json.dumps(payload), encoding="utf-8")
        code, _, _ = _run(capsys, "status", "--json", "--data-root", str(tmp_path / "data"))
        assert code == moto_cli.EXIT_NOT_RUNNING


def test_two_control_files_for_one_backend_count_once(tmp_path, monkeypatch):
    with _FakeBackend() as backend:
        _write_control(tmp_path / "a", backend)
        _write_control(tmp_path / "b", backend)
        monkeypatch.setattr(moto_cli, "candidate_data_roots", lambda explicit=None: [tmp_path / "a", tmp_path / "b"])
        assert len(moto_cli.discover_instances()) == 1


def test_item_ids_cannot_escape_their_route(fake_moto, capsys):
    backend, data_root = fake_moto
    _run(capsys, "autonomous", "paper", "../stop", "--json", "--data-root", str(data_root))
    assert any(r["path"] == "/api/auto-research/paper/..%2Fstop" for r in backend.requests)
    assert not any(r["path"].endswith("/stop") for r in backend.requests)


def test_slow_answers_report_a_timeout_not_a_dead_backend(fake_moto):
    backend, data_root = fake_moto
    instance = moto_cli.read_control_file(data_root)
    with pytest.raises(moto_cli.CliError) as caught:
        moto_cli.MotoClient(instance, timeout=0.5).get("/api/slow")
    assert caught.value.code == moto_cli.EXIT_ERROR
    assert "did not answer" in str(caught.value)


def test_shutdown_only_trusts_the_launcher_record_for_this_backend(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    monkeypatch.setattr(moto_cli, "LAUNCHER_STATE_PATH", state)
    instance = moto_cli.Instance("default", "http://127.0.0.1:1", "t", pid=111)
    state.write_text(json.dumps({"instances": [
        {"instance_id": "default", "backend_window_pid": 999, "frontend_window_pid": 555},
    ]}), encoding="utf-8")
    assert moto_cli.launcher_frontend_pid(instance) is None  # stale record from an older launch
    state.write_text(json.dumps({"instances": [
        {"instance_id": "default", "backend_window_pid": 111, "frontend_window_pid": 555},
    ]}), encoding="utf-8")
    assert moto_cli.launcher_frontend_pid(instance) == 555


def test_global_options_work_before_the_command(fake_moto, capsys):
    _, data_root = fake_moto
    code, out, _ = _run(capsys, "--json", "--data-root", str(data_root), "compiler", "status")
    assert code == 0 and json.loads(out) == {"is_running": False}


# --- installation -----------------------------------------------------------


def _make_foreign_command(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (f"{name}.exe" if sys.platform == "win32" else name)
    target.write_text("not moto", encoding="utf-8")
    if sys.platform != "win32":
        target.chmod(0o755)
    return target


def test_install_skips_moto_when_another_program_owns_it(tmp_path, monkeypatch):
    foreign_dir = tmp_path / "other-tool"
    foreign = _make_foreign_command(foreign_dir, "moto")
    monkeypatch.setenv("PATH", str(foreign_dir))
    bin_dir = tmp_path / "bin"
    result = moto_cli.install_cli(bin_dir, update_path=False)
    assert result["installed"] == ["moto-asi"]
    assert str(foreign) in result["skipped"]["moto"]
    assert not (bin_dir / moto_cli.shim_filename("moto")).exists()
    assert foreign.read_text(encoding="utf-8") == "not moto"


def test_install_withdraws_its_shim_when_another_moto_appears_later(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv("PATH", str(bin_dir))
    moto_cli.install_cli(bin_dir, update_path=False)
    assert (bin_dir / moto_cli.shim_filename("moto")).exists()
    foreign_dir = tmp_path / "other-tool"
    _make_foreign_command(foreign_dir, "moto")
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), str(foreign_dir)]))
    result = moto_cli.install_cli(bin_dir, update_path=False)
    assert "moto" in result["skipped"]
    assert not (bin_dir / moto_cli.shim_filename("moto")).exists()
    assert (bin_dir / moto_cli.shim_filename("moto-asi")).exists()


def test_install_force_still_never_overwrites_foreign_file_in_bin(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    foreign = bin_dir / moto_cli.shim_filename("moto")
    foreign.write_text("someone else's script", encoding="utf-8")
    result = moto_cli.install_cli(bin_dir, force=True, update_path=False)
    assert "moto" in result["skipped"]
    assert foreign.read_text(encoding="utf-8") == "someone else's script"


def test_install_is_idempotent_and_uninstall_removes_only_shims(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv("PATH", str(bin_dir))
    first = moto_cli.install_cli(bin_dir, update_path=False)
    second = moto_cli.install_cli(bin_dir, update_path=False)
    assert first["installed"] == second["installed"] == ["moto", "moto-asi"]
    shim = (bin_dir / moto_cli.shim_filename("moto")).read_text(encoding="utf-8")
    assert moto_cli.SHIM_MARKER in shim and "moto_cli.py" in shim
    keep = bin_dir / "unrelated.txt"
    keep.write_text("x", encoding="utf-8")
    removed = moto_cli.uninstall_cli(bin_dir, update_path=False)
    assert sorted(removed["removed"]) == ["moto", "moto-asi"]
    assert keep.exists()


def test_path_entries_are_appended_never_prepended():
    sep = os.pathsep
    assert moto_cli.path_with_entry(f"A{sep}B", "C") == f"A{sep}B{sep}C"
    assert moto_cli.path_with_entry(f"A{sep}C", "C") is None
    assert moto_cli.path_without_entry(f"A{sep}C{sep}B", "C") == f"A{sep}B"
    assert moto_cli.path_without_entry("A", "C") is None


def test_launcher_cli_install_respects_opt_out(monkeypatch):
    import moto_launcher

    called = []
    monkeypatch.setenv("MOTO_SKIP_CLI_INSTALL", "1")
    monkeypatch.setattr(moto_cli, "install_cli", lambda *a, **k: called.append(1) or {})
    moto_launcher.ensure_cli_command()
    assert called == []


def test_skill_file_is_installable(tmp_path, capsys):
    code, out, _ = _run(capsys, "skill", "install", "--target", str(tmp_path), "--json")
    assert code == 0
    installed = Path(json.loads(out)["installed"][0])
    text = installed.read_text(encoding="utf-8")
    assert text.startswith("---\nname: moto\n")


# --- backend control file and start templates -------------------------------


def test_control_file_written_and_removed_only_by_owner(monkeypatch):
    monkeypatch.setattr(system_config, "desktop_api_token", "tok")
    monkeypatch.setattr(system_config, "backend_host", "0.0.0.0")
    monkeypatch.setattr(system_config, "backend_port", 8123)
    path = cli_control.control_file_path()
    try:
        cli_control.write_control_file(version="1.0")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["backend_url"] == "http://127.0.0.1:8123"
        assert payload["desktop_api_token"] == "tok"
        assert payload["pid"] == os.getpid()
        instance = moto_cli.read_control_file(path.parent)
        assert instance is not None and instance.token == "tok"

        payload["pid"] = os.getpid() + 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        cli_control.remove_control_file()
        assert path.exists()  # another process owns it

        cli_control.write_control_file(version="1.0")
        cli_control.remove_control_file()
        assert not path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_control_file_never_written_in_generic_mode(monkeypatch):
    monkeypatch.setattr(system_config, "generic_mode", True)
    monkeypatch.setattr(system_config, "desktop_api_token", "tok")
    cli_control.write_control_file(version="1.0")
    assert not cli_control.control_file_path().exists()


def test_start_template_drops_credentials(monkeypatch):
    class _Request:
        def model_dump(self, mode="json"):
            return {"user_prompt": "p", "openrouter_api_key": "sk-x",
                    "nested": [{"client_secret": "s", "model": "m"}], "max_output_tokens": 5}

    target = Path(system_config.data_dir) / cli_control.START_TEMPLATE_DIR_NAME / "aggregator.json"
    try:
        cli_control.remember_start_request("aggregator", _Request())
        saved = json.loads(target.read_text(encoding="utf-8"))["request"]
        assert saved == {"user_prompt": "p", "nested": [{"model": "m"}], "max_output_tokens": 5}
        cli_control.remember_start_request("unknown-mode", _Request())
        assert not (target.parent / "unknown-mode.json").exists()
    finally:
        target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Raw API access, route discovery, and request building
# ---------------------------------------------------------------------------


def test_api_supports_patch_with_a_json_body(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("PATCH", "/api/proofs/p1/live-context")] = {"success": True}
    code, out, _ = _run(capsys, "api", "patch", "api/proofs/p1/live-context", "--data", '{"action": "prune"}',
                        "--query", "scope=manual", "--json", "--data-root", str(data_root))
    assert code == 0, out
    call = backend.requests[-1]
    assert call["method"] == "PATCH" and call["body"] == {"action": "prune"}
    assert call["path"].endswith("?scope=manual")


def test_api_rejects_malformed_query_instead_of_dropping_it(fake_moto, capsys):
    _, data_root = fake_moto
    code, _, err = _run(capsys, "api", "GET", "/api/proofs", "--query", "scope", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE and "KEY=VALUE" in err


def test_api_uploads_files_and_saves_binary_responses(fake_moto, tmp_path, capsys):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/aggregator/upload-file")] = {"filename": "notes.txt"}
    backend.routes[("GET", "/api/x.pdf")] = lambda _r: (200, b"%PDF-1.7\x00\xff", "application/pdf")
    source = tmp_path / "notes.txt"
    source.write_bytes(b"hello")
    code, _, _ = _run(capsys, "api", "POST", "/api/aggregator/upload-file", "--file", str(source),
                      "--data-root", str(data_root))
    assert code == 0
    upload = backend.requests[-1]
    assert upload["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="notes.txt"' in upload["raw"] and b"hello" in upload["raw"]
    target = tmp_path / "out.pdf"
    code, out, _ = _run(capsys, "api", "GET", "/api/x.pdf", "-o", str(target), "--json", "--data-root", str(data_root))
    assert code == 0 and json.loads(out)["bytes"] == 10
    assert target.read_bytes() == b"%PDF-1.7\x00\xff"


def test_routes_lists_and_inlines_request_schemas(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/openapi.json")] = {
        "paths": {
            "/api/proofs/check": {"post": {"summary": "Run Manual Proof Check", "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ProofCheckRequest"}}}}}},
            "/api/health": {"get": {"summary": "Health"}},
        },
        "components": {"schemas": {"ProofCheckRequest": {
            "properties": {
                "source_type": {"enum": ["brainstorm", "paper"]},
                "self": {"$ref": "#/components/schemas/ProofCheckRequest"},
            },
        }}},
    }
    code, out, _ = _run(capsys, "routes", "proof", "--schema", "--json", "--data-root", str(data_root))
    rows = json.loads(out)
    assert code == 0 and [row["path"] for row in rows] == ["/api/proofs/check"]
    schema = rows[0]["request_body"]["schema"]
    assert schema["properties"]["source_type"]["enum"] == ["brainstorm", "paper"]
    assert schema["properties"]["self"] == {"$ref": "ProofCheckRequest"}  # cycles stop instead of recursing


def test_set_reaches_nested_fields_and_every_list_item():
    body = {"submitter_configs": [{"model_id": "a"}, {"model_id": "b"}], "final_solver": {"model_id": "c"}}
    result = moto_cli.apply_set_overrides(body, [
        "submitter_configs.*.context_window=1000", "submitter_configs.1.model_id=z", "final_solver.model_id=y",
    ])
    assert result["submitter_configs"] == [
        {"model_id": "a", "context_window": 1000},
        {"model_id": "z", "context_window": 1000},
    ]
    assert result["final_solver"] == {"model_id": "y"}
    assert body["submitter_configs"][0] == {"model_id": "a"}  # the saved template is never mutated
    for bad in ("submitter_configs.5.model_id=x", "submitter_configs.one=x", "final_solver.model_id.deeper=x"):
        with pytest.raises(moto_cli.CliError):
            moto_cli.apply_set_overrides(body, [bad])


@pytest.mark.parametrize("mode", list(moto_cli.MODES))
def test_fresh_settings_match_the_backend_start_request_models(mode):
    from backend.shared.models import (
        AggregatorStartRequest,
        AutonomousResearchStartRequest,
        CompilerStartRequest,
        LeanOJStartRequest,
    )

    model = {"autonomous": AutonomousResearchStartRequest, "aggregator": AggregatorStartRequest,
             "compiler": CompilerStartRequest, "leanoj": LeanOJStartRequest}[mode]
    spec = moto_cli.RoleSpec("openrouter", "vendor/model", 200000, 40000, host="Host", reasoning="high")
    body = moto_cli.fresh_start_request(mode, spec, submitters=2)
    body[moto_cli.MODES[mode].prompt_field] = "prompt"
    if mode == "leanoj":
        body["lean_template"] = "theorem t : True := by sorry"
    dumped = model.model_validate(body).model_dump()

    def kept(parsed, sent) -> bool:  # every field we sent exists and keeps its value (defaults may be added)
        if isinstance(sent, dict):
            return isinstance(parsed, dict) and all(key in parsed and kept(parsed[key], value) for key, value in sent.items())
        if isinstance(sent, list):
            return isinstance(parsed, list) and len(parsed) == len(sent) and all(map(kept, parsed, sent))
        return parsed == sent

    for key, value in body.items():
        assert key in dumped and kept(dumped[key], value), key


def test_start_with_model_builds_settings_from_openrouter_metadata(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/openrouter/models")] = {"models": [
        {"id": "vendor/model", "context_length": 200000, "top_provider": {"max_completion_tokens": 32000}},
    ]}
    code, out, _ = _run(capsys, "autonomous", "start", "--model", "vendor/model", "--submitters", "1",
                        "--prompt", "p", "--json", "--data-root", str(data_root))
    assert code == 0, out
    body = next(r for r in backend.requests if r["path"] == "/api/auto-research/start")["body"]
    assert body["submitter_configs"] == [{"submitter_id": 1, "provider": "openrouter", "model_id": "vendor/model",
                                          "context_window": 200000, "max_output_tokens": 32000}]
    assert body["validator_context_window"] == 200000 and body["assistant_max_tokens"] == 32000


def test_model_limits_are_never_guessed(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/openrouter/models")] = {"models": [{"id": "vendor/model", "context_length": 200000}]}
    code, out, _ = _run(capsys, "config", "init", "autonomous", "--model", "vendor/model", "--json",
                        "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE and "--max-tokens" in json.loads(out)["error"]
    code, out, _ = _run(capsys, "config", "init", "compiler", "--provider", "lm_studio", "--model", "local",
                        "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE
    code, out, _ = _run(capsys, "config", "init", "compiler", "--provider", "lmstudio", "--model", "local",
                        "--context", "32000", "--max-tokens", "8000", "--json", "--data-root", str(data_root))
    assert code == 0 and json.loads(out)["validator_provider"] == "lm_studio"


def test_leanoj_start_requires_a_lean_template(fake_moto, capsys):
    _, data_root = fake_moto
    code, out, _ = _run(capsys, "leanoj", "start", "--model", "m", "--context", "1000", "--max-tokens", "100",
                        "--prompt", "p", "--dry-run", "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE and "--lean-template" in json.loads(out)["error"]


def test_aggregator_start_uploads_reference_files(fake_moto, tmp_path, capsys):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/aggregator/upload-file")] = lambda r: (200, {"filename": "ref.txt"})
    backend.routes[("POST", "/api/aggregator/start")] = {"success": True}
    reference = tmp_path / "ref.txt"
    reference.write_text("facts", encoding="utf-8")
    code, out, _ = _run(capsys, "aggregator", "start", "--model", "m", "--context", "1000", "--max-tokens", "100",
                        "--prompt", "p", "--file", str(reference), "--json", "--data-root", str(data_root))
    assert code == 0, out
    start = next(r for r in backend.requests if r["path"] == "/api/aggregator/start")
    assert start["body"]["uploaded_files"] == ["ref.txt"]


# ---------------------------------------------------------------------------
# Status, waiting, and recovery
# ---------------------------------------------------------------------------


def test_status_explains_why_work_stopped_or_is_blocked(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/auto-research/status")] = {"is_running": False, "terminal_event": {
        "reason": "provider_repair_required", "title": "OpenRouter key rejected", "message": "Key invalid",
        "terminal_guidance": "Save a new key, then start again.", "fatal": True,
    }}
    backend.routes[("GET", "/api/proofs/runs")] = {"runs": [
        {"proof_run_id": "r1", "scope": "manual", "source_type": "paper", "source_id": "manual_compiler_current",
         "status": "provider_paused", "current_round": 2, "lifecycle_generation": 3},
        {"proof_run_id": "r0", "scope": "manual", "source_type": "paper", "source_id": "x", "status": "completed"},
    ]}
    backend.routes[("GET", "/api/workflow/solution-path")] = {
        "run_id": "run", "lifecycle_generation": 1, "repair_required_proposals": 1,
        "repairs": [{"proposal_id": "p1", "lifecycle_generation": 1}], "repair_detail": "context too small",
    }
    code, out, _ = _run(capsys, "status", "--json", "--data-root", str(data_root))
    payload = json.loads(out)
    assert code == 0
    assert [run["proof_run_id"] for run in payload["proof_runs"]] == ["r1"]
    kinds = [item["kind"] for item in payload["attention"]]
    assert kinds == ["autonomous_stopped", "proof_run_provider_paused", "solution_path_repair"]
    assert payload["attention"][0]["hint"] == "Save a new key, then start again."
    code, out, _ = _run(capsys, "status", "--data-root", str(data_root))
    assert "Needs attention: Autonomous Research stopped: OpenRouter key rejected" in out


def test_wait_returns_once_enough_papers_exist(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    backend.autonomous_running = True
    counts = iter([0, 1, 2])
    backend.routes[("GET", "/api/auto-research/papers")] = lambda r: (200, {"papers": [{}] * next(counts)})
    monkeypatch.setattr(moto_cli.time, "sleep", lambda _s: None)
    code, out, _ = _run(capsys, "wait", "--papers", "2", "--json", "--data-root", str(data_root))
    assert code == 0 and json.loads(out)["condition"] == "2 completed papers"


def test_wait_reports_a_run_that_stopped_before_the_goal(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/auto-research/status")] = {
        "is_running": False, "terminal_event": {"message": "Lean broke"},
    }
    backend.routes[("GET", "/api/auto-research/papers")] = {"papers": []}
    monkeypatch.setattr(moto_cli.time, "sleep", lambda _s: None)
    code, out, _ = _run(capsys, "wait", "--papers", "1", "--json", "--data-root", str(data_root))
    payload = json.loads(out)
    assert code == moto_cli.EXIT_REJECTED and "Lean broke" in payload["error"]
    assert payload["detail"]["terminal_event"]["message"] == "Lean broke"


def test_wait_times_out_with_exit_1(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.autonomous_running = True
    code, out, _ = _run(capsys, "wait", "--timeout", "0.2", "--interval", "0.05", "--json",
                        "--data-root", str(data_root))
    assert code == moto_cli.EXIT_ERROR and "Timed out" in json.loads(out)["error"]


def test_retry_solution_path_resumes_each_blocked_proposal(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/workflow/solution-path")] = {
        "run_id": "run-1", "lifecycle_generation": 4, "repairs": [{"proposal_id": "p1", "lifecycle_generation": 4}],
    }
    backend.routes[("POST", "/api/workflow/solution-path/resume")] = {"success": True}
    code, _, _ = _run(capsys, "retry", "solution-path", "--json", "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"] == {"run_id": "run-1", "proposal_id": "p1", "lifecycle_generation": 4}


# ---------------------------------------------------------------------------
# Credentials, proofs, settings, updates
# ---------------------------------------------------------------------------


def test_keys_are_read_from_stdin_and_never_from_arguments(fake_moto, capsys, monkeypatch):
    import io

    backend, data_root = fake_moto
    backend.routes[("POST", "/api/openrouter/set-api-key")] = {"success": True, "message": "saved"}
    monkeypatch.setattr(sys, "stdin", io.StringIO("sk-or-secret\n"))
    code, out, _ = _run(capsys, "keys", "set", "openrouter", "--json", "--data-root", str(data_root))
    assert code == 0, out
    assert backend.requests[-1]["body"] == {"api_key": "sk-or-secret"}
    assert "sk-or-secret" not in out
    code, _, _ = _run(capsys, "keys", "set", "openrouter", "sk-or-inline", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE
    code, _, _ = _run(capsys, "keys", "clear", "codex", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE  # needs --yes


def test_vllm_set_test_status_and_clear_use_the_setup_routes(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    probe = {"success": True, "chat_models": [{"id": "Qwen/Qwen2.5"}], "embedding_model": "nomic-embed"}
    backend.routes[("POST", "/api/cloud-access/vllm/test")] = probe
    backend.routes[("POST", "/api/cloud-access/vllm/config")] = {**probe, "status": {"configured": True}}
    backend.routes[("GET", "/api/cloud-access/vllm/availability")] = {
        "configured": True, "available": True, "servers": ["http://127.0.0.1:8001/v1"],
        "models": ["Qwen/Qwen2.5"], "embedding_model": "nomic-embed",
    }
    backend.routes[("DELETE", "/api/cloud-access/vllm")] = {"success": True}

    code, out, _ = _run(capsys, "vllm", "test", "http://127.0.0.1:8001", "--data-root", str(data_root))
    assert code == 0 and "nothing saved" in out and "Qwen/Qwen2.5" in out
    assert backend.requests[-1]["body"] == {"servers": ["http://127.0.0.1:8001"], "embedding_model": ""}

    monkeypatch.setenv("MY_VLLM_KEY", "vllm-secret")
    code, out, _ = _run(capsys, "vllm", "set", "http://127.0.0.1:8001", "http://127.0.0.1:8002",
                        "--embedding-model", "nomic-embed", "--key-from-env", "MY_VLLM_KEY",
                        "--data-root", str(data_root))
    assert code == 0 and "vLLM setup saved" in out and "vllm-secret" not in out
    assert backend.requests[-1]["body"] == {
        "servers": ["http://127.0.0.1:8001", "http://127.0.0.1:8002"],
        "embedding_model": "nomic-embed",
        "api_key": "vllm-secret",
        "clear_api_key": False,
    }

    code, out, _ = _run(capsys, "vllm", "status", "--data-root", str(data_root))
    assert code == 0 and "reachable" in out and "nomic-embed" in out
    assert backend.requests[-1]["path"] == "/api/cloud-access/vllm/availability?refresh=true"

    code, _, _ = _run(capsys, "vllm", "clear", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE  # needs --yes
    code, out, _ = _run(capsys, "vllm", "clear", "--yes", "--data-root", str(data_root))
    assert code == 0 and backend.requests[-1]["method"] == "DELETE"


def test_vllm_model_limits_come_from_the_detected_context_window(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/cloud-access/vllm/models")] = {"models": [
        {"id": "Qwen/Qwen2.5", "context_length": 32768},
        {"id": "no/window"},
    ]}
    code, out, _ = _run(capsys, "config", "init", "compiler", "--provider", "vllm", "--model", "Qwen/Qwen2.5",
                        "--json", "--data-root", str(data_root))
    assert code == 0, out
    settings = json.loads(out)
    assert settings["validator_provider"] == "vllm" and settings["validator_model"] == "Qwen/Qwen2.5"
    assert settings["validator_context_size"] == 32768 and settings["validator_max_output_tokens"] == 6553

    # A server that reports no window, or a model no server serves, is never guessed.
    code, out, _ = _run(capsys, "config", "init", "compiler", "--provider", "vllm", "--model", "no/window",
                        "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE and "--max-tokens" in json.loads(out)["error"]
    code, out, _ = _run(capsys, "config", "init", "compiler", "--provider", "vllm", "--model", "missing",
                        "--json", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE and "vllm status" in json.loads(out)["error"]


def test_login_finishes_with_a_pasted_callback(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/cloud-access/openai-codex/oauth/exchange")] = {"success": True}
    code, _, _ = _run(capsys, "login", "codex", "--callback-url",
                      "http://localhost:1455/auth/callback?code=c&state=s", "--json", "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"]["redirect_url"].endswith("state=s")


def test_proof_check_and_stop_use_the_run_lifecycle(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/proofs/check")] = {"proof_run_id": "r9", "queued": True}
    backend.routes[("GET", "/api/proofs/runs/r9")] = {"proof_run_id": "r9", "lifecycle_generation": 7}
    backend.routes[("POST", "/api/proofs/runs/r9/stop")] = {"proof_run_id": "r9"}
    code, _, _ = _run(capsys, "proofs", "check", "aggregator", "--loop", "--json", "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"] == {"source_type": "brainstorm", "source_id": "manual_aggregator",
                                            "run_mode": "loop_with_pruning"}
    code, _, _ = _run(capsys, "proofs", "stop", "r9", "--json", "--data-root", str(data_root))
    assert code == 0 and backend.requests[-1]["body"] == {"expected_lifecycle_generation": 7}
    code, _, _ = _run(capsys, "proofs", "check", "paper", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE  # a paper check needs its ID


def test_proof_settings_keep_values_that_were_not_changed(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/proofs/status")] = {"lean4_user_enabled": True, "lean4_proof_timeout": 900}
    backend.routes[("POST", "/api/proofs/settings")] = {"lean4_enabled": True}
    code, _, _ = _run(capsys, "proofs", "settings", "--smt", "on", "--json", "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"] == {"enabled": True, "timeout": 900, "smt_enabled": True}


def test_proof_settings_change_proof_types(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/proofs/status")] = {"lean4_user_enabled": False, "lean4_proof_timeout": 2000}
    backend.routes[("POST", "/api/proofs/settings")] = {"proof_master_proofs_enabled": False}
    code, _, _ = _run(capsys, "proofs", "settings", "--master", "off", "--single-shot", "on", "--json",
                      "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"] == {
        "enabled": False, "timeout": 2000,
        "single_shot_proofs_enabled": True, "master_proofs_enabled": False,
    }


def test_free_model_settings_merge_with_current_values(fake_moto, capsys):
    backend, data_root = fake_moto
    backend.routes[("GET", "/api/openrouter/free-model-settings")] = {
        "looping_enabled": True, "auto_selector_enabled": False,
    }
    backend.routes[("POST", "/api/openrouter/free-model-settings")] = {"success": True}
    code, _, _ = _run(capsys, "settings", "free-models", "--auto-selector", "on", "--json",
                      "--data-root", str(data_root))
    assert code == 0
    assert backend.requests[-1]["body"] == {"looping_enabled": True, "auto_selector_enabled": True}


def test_update_apply_needs_yes_and_waits_for_the_result(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    code, _, _ = _run(capsys, "update", "apply", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_USAGE
    backend.routes[("POST", "/api/update/pull")] = {"started": True, "install_kind": "git"}
    states = iter(["idle", "running", "done"])
    backend.routes[("GET", "/api/update/pull-status")] = lambda r: (200, {"status": next(states), "output_lines": []})
    monkeypatch.setattr(moto_cli.time, "sleep", lambda _s: None)
    code, out, _ = _run(capsys, "update", "apply", "--yes", "--json", "--data-root", str(data_root))
    assert code == 0 and json.loads(out)["restart_required"] is True


# ---------------------------------------------------------------------------
# Shutdown and live events
# ---------------------------------------------------------------------------


def test_shutdown_stops_the_workflow_then_exits_gracefully(fake_moto, capsys, monkeypatch, tmp_path):
    backend, data_root = fake_moto
    backend.autonomous_running = True

    def stop(_record):
        backend.autonomous_running = False
        return 200, {"success": True}

    backend.routes[("POST", "/api/auto-research/stop")] = stop
    calls: list[str] = []
    alive = {"backend": True}

    def graceful(pid):
        assert not backend.autonomous_running  # the workflow stopped before the backend was asked to exit
        calls.append(f"graceful:{pid}")
        alive["backend"] = False
        return True

    monkeypatch.setattr(moto_cli, "LAUNCHER_STATE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(moto_cli, "request_graceful_exit", graceful)
    monkeypatch.setattr(moto_cli, "_pid_alive", lambda pid: alive["backend"])
    monkeypatch.setattr(moto_cli, "_terminate_tree", lambda pid: calls.append(f"kill:{pid}"))
    code, out, _ = _run(capsys, "shutdown", "--json", "--data-root", str(data_root))
    payload = json.loads(out)
    assert code == 0, out
    assert calls == ["graceful:4242"]
    assert payload["graceful"] is True and payload["forced"] is False
    assert payload["stopped_workflows"] == ["autonomous"]
    assert not (data_root / moto_cli.CONTROL_FILE_NAME).exists()


def test_shutdown_force_kills_when_a_clean_exit_is_impossible(fake_moto, capsys, monkeypatch, tmp_path):
    _, data_root = fake_moto
    alive = {"backend": True}
    killed: list[int] = []
    monkeypatch.setattr(moto_cli, "LAUNCHER_STATE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(moto_cli, "request_graceful_exit", lambda pid: False)
    monkeypatch.setattr(moto_cli, "_pid_alive", lambda pid: alive["backend"])

    def kill(pid):
        killed.append(pid)
        alive["backend"] = False

    monkeypatch.setattr(moto_cli, "_terminate_tree", kill)
    code, out, _ = _run(capsys, "shutdown", "--json", "--data-root", str(data_root))
    assert code == 0 and killed == [4242] and json.loads(out)["forced"] is True


class _FakeClosed(Exception):
    pass


class _FakeWsError(Exception):
    pass


class _FakeSocket:
    def __init__(self, messages, then_close=False):
        self.messages = list(messages)
        self.then_close = then_close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def recv(self, timeout=None):
        if self.messages:
            return self.messages.pop(0)
        if self.then_close:
            raise _FakeClosed("server restarted")
        raise TimeoutError

    def send(self, _message):
        return None


def test_events_reconnect_after_the_stream_drops(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/ws-ticket")] = {"ticket": "t"}
    sockets = iter([
        OSError("connection refused"),
        _FakeSocket([json.dumps({"type": "a"})], then_close=True),
        _FakeSocket([json.dumps({"type": "b"})]),
    ])

    def connect(_url, **_kw):
        item = next(sockets)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(moto_cli, "_load_websockets", lambda: (connect, _FakeClosed, _FakeWsError))
    monkeypatch.setattr(moto_cli.time, "sleep", lambda _s: None)
    code, out, err = _run(capsys, "events", "--json", "--limit", "2", "--data-root", str(data_root))
    assert code == 0
    assert [json.loads(line)["type"] for line in out.splitlines()] == ["a", "b"]
    assert "retrying" in err and "reconnected" in err


def test_events_no_reconnect_exits_3(fake_moto, capsys, monkeypatch):
    backend, data_root = fake_moto
    backend.routes[("POST", "/api/ws-ticket")] = {"ticket": "t"}
    monkeypatch.setattr(moto_cli, "_load_websockets", lambda: (
        lambda _url, **_kw: _FakeSocket([], then_close=True), _FakeClosed, _FakeWsError))
    code, _, _ = _run(capsys, "events", "--no-reconnect", "--data-root", str(data_root))
    assert code == moto_cli.EXIT_NOT_RUNNING
