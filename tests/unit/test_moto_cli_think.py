"""Tests for `moto think`: handing MOTO one question and checking back for its answer."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import moto_cli

TOKEN = "test-token-think"
# What the user last started in the app: papers on, final answer stage off.
SAVED_SETTINGS = {"user_research_prompt": "old question", "validator_model": "m", "tier3_enabled": False,
                  "allow_research_papers": True, "allow_mathematical_proofs": True}
# What an earlier `moto think` start saved for itself.
SAVED_THINK_SETTINGS = {"user_research_prompt": "think question", "validator_model": "think-m", "tier3_enabled": False,
                        "allow_research_papers": False, "allow_mathematical_proofs": True, "direct_answer_enabled": True}


class _ThinkBackend:
    """The slice of the MOTO REST surface `moto think` uses."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.status: dict = {"is_running": False, "run_id": "legacy", "current_tier": "idle", "resume_available": False}
        self.other_running: str | None = None
        self.paused_prompt = ""
        self.direct_answer: dict | None = None
        self.papers: list[dict] = []
        self.proofs: list[dict] = []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def _send(self, status: int, payload) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                path = self.path.split("?")[0]
                body = json.loads(raw) if raw else None
                backend.requests.append({"method": method, "path": self.path, "body": body})
                if path == "/api/health":
                    return self._send(200, {"status": "healthy", "instance_id": "default", "version": "9.9"})
                if self.headers.get(moto_cli.TOKEN_HEADER) != TOKEN:
                    return self._send(401, {"detail": "Missing or invalid desktop API token."})
                if path == "/api/token-stats":
                    return self._send(200, {"success": True})
                if path == "/api/auto-research/status":
                    return self._send(200, backend.status)
                if path.endswith("/status"):
                    return self._send(200, {"is_running": backend.other_running is not None and backend.other_running in path})
                if path == "/api/auto-research/prompt":
                    return self._send(200, {"prompt": backend.paused_prompt})
                if path == "/api/auto-research/start":
                    backend.status = {**backend.status, "is_running": True, "run_id": "session-1",
                                      "current_tier": "tier1_aggregation", "terminal_event": None}
                    backend.paused_prompt = (body or {}).get("user_research_prompt", "")
                    if (body or {}).get("direct_answer_enabled"):
                        backend.status["direct_answer"] = {"enabled": True, "state": "pending"}
                    if (body or {}).get("max_runtime_seconds"):
                        limit = body["max_runtime_seconds"]
                        backend.status["runtime_limit"] = {"max_runtime_seconds": limit, "remaining_seconds": limit}
                    return self._send(200, {"success": True, "message": "started"})
                if path == "/api/auto-research/stop":
                    backend.status = {**backend.status, "is_running": False, "resume_available": True}
                    return self._send(200, {"success": True})
                if path == "/api/auto-research/clear":
                    backend.status = {**backend.status, "resume_available": False}
                    return self._send(200, {"success": True})
                if path == "/api/auto-research/direct-answer/finish":
                    backend.status["direct_answer"] = {"enabled": True, "state": "writing"}
                    return self._send(200, {"success": True, "message": "the direct answer is being written"})
                if path == "/api/auto-research/direct-answer":
                    return self._send(200, {"has_answer": backend.direct_answer is not None, "answer": backend.direct_answer})
                if path == "/api/auto-research/papers":
                    return self._send(200, {"papers": backend.papers})
                if path.startswith("/api/auto-research/paper/"):
                    return self._send(200, {"content": "FULL " + path.rsplit("/", 1)[-1]})
                if path == "/api/proofs":
                    return self._send(200, {"proofs": backend.proofs})
                if path == "/api/workflow/solution-path":
                    return self._send(200, {"repair_required_proposals": 0})
                return self._send(404, {"detail": "Not Found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def posts(self, path: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST" and r["path"].split("?")[0] == path]

    def post_paths(self) -> list[str]:
        return [r["path"].split("?")[0] for r in self.requests if r["method"] == "POST"]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def think(tmp_path, capsys):
    data_root = tmp_path / "data"
    with _ThinkBackend() as backend:
        data_root.mkdir(parents=True)
        (data_root / moto_cli.CONTROL_FILE_NAME).write_text(json.dumps({
            "schema": "moto-cli-control-v1", "instance_id": "default", "pid": 4242, "backend_url": backend.url,
            "desktop_api_token": TOKEN, "data_root": str(data_root),
        }), encoding="utf-8")
        templates = data_root / moto_cli.START_TEMPLATE_DIR_NAME
        templates.mkdir()
        (templates / "autonomous.json").write_text(json.dumps({"request": SAVED_SETTINGS}), encoding="utf-8")
        (templates / "think.json").write_text(json.dumps({"request": SAVED_THINK_SETTINGS}), encoding="utf-8")

        def run(*argv: str, command: str = "think") -> tuple[int, dict]:
            code = moto_cli.main([command, *argv, "--json", "--data-root", str(data_root)])
            return code, json.loads(capsys.readouterr().out)

        yield backend, run


def test_durations_accept_units_and_reject_nonsense():
    assert moto_cli.parse_duration("90m") == 5400
    assert moto_cli.parse_duration("1h30m") == 5400
    assert moto_cli.parse_duration("1.5h") == 5400
    assert moto_cli.parse_duration("2d") == 172800
    assert moto_cli.parse_duration("600") == 600
    assert moto_cli.format_duration(5400) == "1h 30m"
    for bad in ("soon", "5x", "30", "10s", "h2", "nan"):
        with pytest.raises(moto_cli.CliError) as raised:
            moto_cli.parse_duration(bad)
        assert raised.value.code == moto_cli.EXIT_USAGE


def test_start_asks_for_a_direct_answer_and_does_not_assume_a_paper(think):
    backend, run = think
    code, payload = run("start", "--prompt", "Which design is safest?", "--max-runtime", "2h")
    assert code == 0, payload
    body = backend.posts("/api/auto-research/start")[0]["body"]
    assert body["user_research_prompt"] == "Which design is safest?"
    assert body["validator_model"] == "m"  # the user's saved model settings are reused
    assert body["direct_answer_enabled"] is True
    # The saved settings had papers and proofs on; a think only explores unless asked.
    assert body["allow_research_papers"] is False and body["allow_mathematical_proofs"] is False
    assert body["tier3_enabled"] is False
    assert body["max_runtime_seconds"] == 7200
    assert payload["started"] is True and payload["resumed"] is False
    assert payload["state"] == "thinking" and payload["session_id"] == "session-1"
    assert payload["runtime_limit"]["max_runtime_seconds"] == 7200
    assert payload["question"] == "Which design is safest?"
    assert payload["answer_ready"] is False


def test_start_adds_papers_and_proofs_only_when_asked_and_no_limit_by_default(think):
    backend, run = think
    code, _ = run("start", "--prompt", "Q", "--papers", "--proofs")
    assert code == 0
    body = backend.posts("/api/auto-research/start")[0]["body"]
    assert body["allow_research_papers"] is True and body["allow_mathematical_proofs"] is True
    assert "max_runtime_seconds" not in body


def test_start_needs_a_question(think):
    backend, run = think
    code, payload = run("start")
    assert code == moto_cli.EXIT_USAGE
    assert "--prompt" in payload["error"]
    assert not backend.posts("/api/auto-research/start")


@pytest.mark.parametrize("active", ["autonomous", "compiler"])
def test_start_is_rejected_while_a_workflow_runs(think, active):
    backend, run = think
    if active == "autonomous":
        backend.status["is_running"] = True
    else:
        backend.other_running = "/api/compiler/"
    code, payload = run("start", "--prompt", "Q")
    assert code == moto_cli.EXIT_REJECTED
    assert payload["detail"] == {"reason": "workflow_running", "active_workflow": active}
    assert not backend.posts("/api/auto-research/start")


def test_a_paused_different_question_is_never_retired_without_replace(think):
    backend, run = think
    backend.status["resume_available"] = True
    backend.paused_prompt = "The user's own research"
    code, payload = run("start", "--prompt", "New question")
    assert code == moto_cli.EXIT_REJECTED
    assert payload["detail"]["reason"] == "paused_session_exists"
    assert payload["detail"]["paused_question"] == "The user's own research"
    assert not backend.posts("/api/auto-research/clear")
    assert not backend.posts("/api/auto-research/start")

    code, payload = run("start", "--prompt", "New question", "--replace")
    assert code == 0, payload
    assert backend.posts("/api/auto-research/clear")[0]["path"].endswith("confirm=true")
    assert backend.posts("/api/auto-research/start")[0]["body"]["user_research_prompt"] == "New question"


def test_starting_the_same_paused_question_continues_it(think):
    backend, run = think
    backend.status["resume_available"] = True
    backend.paused_prompt = "Same question"
    code, payload = run("start", "--prompt", "Same question")
    assert code == 0, payload
    assert payload["resumed"] is True
    assert not backend.posts("/api/auto-research/clear")


def test_resume_reuses_the_questions_own_settings_with_a_new_limit(think):
    backend, run = think
    code, payload = run("resume")
    assert code == moto_cli.EXIT_REJECTED and payload["detail"]["reason"] == "nothing_to_resume"

    backend.status["resume_available"] = True
    backend.paused_prompt = "Paused question"
    code, payload = run("resume", "--max-runtime", "30m")
    assert code == 0, payload
    body = backend.posts("/api/auto-research/start")[0]["body"]
    assert body["user_research_prompt"] == "Paused question"
    assert body["validator_model"] == "think-m"  # the think's saved settings, not the app's
    assert body["allow_mathematical_proofs"] is True and body["allow_research_papers"] is False
    assert body["max_runtime_seconds"] == 1800 and body["direct_answer_enabled"] is True
    assert payload["resumed"] is True


@pytest.mark.parametrize("status, expected", [
    ({"is_running": True}, "thinking"),
    ({"is_running": True, "direct_answer": {"enabled": True, "state": "pending"}}, "thinking"),
    ({"is_running": True, "direct_answer": {"enabled": True, "state": "writing"}}, "answering"),
    ({"is_running": True, "direct_answer": {"enabled": True, "state": "ready"}}, "thinking"),
    ({"is_running": False, "direct_answer": {"state": "ready"}}, "answered"),
    ({"is_running": False, "direct_answer": {"state": "ready"},
      "terminal_event": {"reason": "max_runtime_reached"}}, "answered"),
    ({"is_running": False, "direct_answer": {"state": "ready"},
      "terminal_event": {"reason": "direct_answer_complete"}}, "answered"),
    ({"is_running": False, "direct_answer": {"state": "ready"},
      "terminal_event": {"reason": "provider_repair"}}, "needs_attention"),
    ({"is_running": False, "terminal_event": {"reason": "direct_answer_failed"}}, "needs_attention"),
    ({"is_running": False, "terminal_event": {"reason": "max_runtime_reached"}}, "time_limit_reached"),
    ({"is_running": False, "resume_available": True}, "stopped"),
    ({"is_running": False}, "idle"),
    (None, "idle"),
])
def test_state_is_one_word_for_where_the_question_stands(status, expected):
    assert moto_cli.think_state(status) == expected


def test_status_after_a_time_limit_says_the_answer_is_ready_and_can_be_resumed(think):
    backend, run = think
    backend.paused_prompt = "Q"
    backend.status.update({
        "resume_available": True, "run_id": "session-1",
        "stats": {"total_papers_completed": 0, "total_submissions_accepted": 31},
        "direct_answer": {"enabled": False, "state": "ready", "reason": "time_limit:2h"},
        "terminal_event": {"reason": "max_runtime_reached", "message": "Reached 2h and wrote its direct answer."},
    })
    code, payload = run("status")
    assert code == 0
    assert payload["state"] == "answered" and payload["running"] is False
    assert payload["answer_ready"] is True and payload["answer_reason"] == "time_limit"
    assert payload["can_resume"] is True
    assert payload["progress"]["ideas_accepted"] == 31
    assert "think answer" in payload["next"] and "think resume" in payload["next"]
    assert payload["attention"] == []  # a chosen limit is not a problem to repair

    # The general status command does not report a chosen ending as a problem either.
    code, general = run(command="status")
    assert code == 0 and general["attention"] == []


def test_status_surfaces_a_problem_stop_as_attention(think):
    backend, run = think
    backend.status.update({"resume_available": True, "terminal_event": {
        "reason": "provider_repair", "title": "Model configuration requires repair", "message": "bad key",
        "terminal_guidance": "Sign in again.",
    }})
    code, payload = run("status")
    assert code == 0 and payload["state"] == "needs_attention"
    assert payload["attention"][0]["kind"] == "autonomous_stopped"
    assert payload["attention"][0]["hint"] == "Sign in again."
    assert "think finish" in payload["next"]


def test_answer_returns_motos_one_answer_to_the_question(think):
    backend, run = think
    backend.status.update({"run_id": "session-1", "direct_answer": {"state": "ready"}})
    backend.direct_answer = {
        "answer_id": "a1", "question": "Which design is safest?", "answer": "Design A, because ...",
        "reason": "time_limit:2h", "exploration_complete": False, "created_at": "2026-11-17T10:00:00",
        "word_count": 4, "model": "a/b", "basis": {"topics": ["t1"]},
    }
    backend.papers = [{"paper_id": "p1", "title": "Paper 1", "abstract": "A1", "word_count": 10}]
    code, payload = run("answer")
    assert code == 0
    assert payload["answer_ready"] is True and payload["state"] == "answered"
    assert payload["question"] == "Which design is safest?"
    assert payload["answer"] == {
        "content": "Design A, because ...", "reason": "time_limit", "exploration_complete": False,
        "created_at": "2026-11-17T10:00:00", "word_count": 4, "model": "a/b",
    }
    assert "maximum runtime" in payload["note"]
    assert payload["papers"] == [] and payload["proofs"] == []  # only on request

    backend.proofs = [{"proof_id": "proof_1", "theorem_statement": "T", "novelty_tier": "novel", "lean_code": "x"}]
    code, payload = run("answer", "--details")
    assert payload["papers"][0]["content"] == "FULL p1"
    assert payload["proofs"] == [{"proof_id": "proof_1", "theorem_statement": "T", "novelty_tier": "novel", "proof_kind": None}]


def test_answer_before_one_exists_says_how_to_get_it(think):
    backend, run = think
    backend.status.update({"is_running": True, "run_id": "session-1",
                           "direct_answer": {"enabled": True, "state": "pending"}})
    code, payload = run("answer")
    assert code == 0
    assert payload["answer_ready"] is False and payload["answer"] is None
    assert "think finish" in payload["note"]

    backend.status["direct_answer"] = {"enabled": True, "state": "writing"}
    code, payload = run("answer")
    assert payload["state"] == "answering" and "writing its answer" in payload["note"]


def test_finish_asks_a_running_think_to_answer_now(think):
    backend, run = think
    backend.status.update({"is_running": True, "direct_answer": {"enabled": True, "state": "pending"}})
    code, payload = run("finish")
    assert code == 0, payload
    assert backend.post_paths() == ["/api/auto-research/direct-answer/finish"]
    assert payload["finishing"] is True and payload["state"] == "answering"


def test_finish_refuses_a_run_that_was_not_started_by_think(think):
    backend, run = think
    backend.status["is_running"] = True  # the user's own research, no direct answer
    code, payload = run("finish")
    assert code == moto_cli.EXIT_REJECTED and payload["detail"]["reason"] == "not_a_think_run"
    assert backend.post_paths() == []


def test_finish_on_a_stopped_question_continues_it_only_to_answer(think):
    backend, run = think
    code, payload = run("finish")
    assert code == moto_cli.EXIT_REJECTED and payload["detail"]["reason"] == "nothing_to_resume"

    backend.status["resume_available"] = True
    backend.paused_prompt = "Paused question"
    code, payload = run("finish")
    assert code == 0, payload
    assert backend.post_paths() == ["/api/auto-research/start", "/api/auto-research/direct-answer/finish"]
    body = backend.posts("/api/auto-research/start")[0]["body"]
    assert body["direct_answer_enabled"] is True and "max_runtime_seconds" not in body
    assert payload["state"] == "answering"


def test_wait_returns_the_status_when_time_runs_out_instead_of_failing(think, monkeypatch):
    backend, run = think
    backend.status["is_running"] = True
    monkeypatch.setattr(moto_cli.time, "sleep", lambda _s: None)
    code, payload = run("wait", "--timeout", "0.01", "--interval", "0.01")
    assert code == 0
    assert payload["timed_out"] is True and payload["state"] == "thinking"

    backend.status["is_running"] = False
    code, payload = run("wait")
    assert code == 0 and payload["timed_out"] is False


def test_stop_keeps_progress_and_points_to_finish_for_an_answer(think):
    backend, run = think
    backend.status["is_running"] = True
    code, payload = run("stop")
    assert code == 0
    assert backend.posts("/api/auto-research/stop")
    assert payload["state"] == "stopped" and "think finish" in payload["next"]


def test_plain_autonomous_start_takes_the_limit_and_final_answer_flags(tmp_path):
    templates = tmp_path / moto_cli.START_TEMPLATE_DIR_NAME
    templates.mkdir()
    (templates / "autonomous.json").write_text(json.dumps({"request": SAVED_SETTINGS}), encoding="utf-8")
    (templates / "think.json").write_text(json.dumps({"request": SAVED_THINK_SETTINGS}), encoding="utf-8")
    instance = moto_cli.Instance("offline", "", "", data_root=tmp_path)
    parser = moto_cli.build_parser()

    args = parser.parse_args(["autonomous", "start", "--prompt", "Q", "--max-runtime", "45m", "--final-answer"])
    body = moto_cli.build_start_request(args, moto_cli.MODES["autonomous"], instance)
    assert body["max_runtime_seconds"] == 2700
    assert body["tier3_enabled"] is True

    # Without the flags a plain start keeps the saved setting, adds no limit, and never
    # picks up a think's settings.
    args = parser.parse_args(["autonomous", "start", "--prompt", "Q"])
    body = moto_cli.build_start_request(args, moto_cli.MODES["autonomous"], instance)
    assert "max_runtime_seconds" not in body and "direct_answer_enabled" not in body
    assert body["tier3_enabled"] is False and body["validator_model"] == "m"
