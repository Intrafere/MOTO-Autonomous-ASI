"""Real-code adapters for the desktop `moto` CLI control surface.

The production start routes write the CLI start template; these tests feed that
output back through the CLI request builder and into the next real start, and
check the hosted, conflict, and control-file boundaries around it.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import moto_cli
from backend.api.routes import aggregator as aggregator_route
from backend.shared import cli_control
from backend.shared.config import system_config
from backend.shared.models import AggregatorStartRequest
from tests.workflow_harness.invariants import assert_observed_invariant
from tests.workflow_harness.real_adapters import ManualAggregatorAdapter, minimal_aggregator_request
from tests.workflow_harness.real_adapters.observed_state import RealWorkflowObservation
from tests.workflow_real_adapters.helpers import inactive_workflow_flags, patch_attributes

REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_MARKERS = ("api_key", "secret", "password", "access_token", "refresh_token")


class _FakeSolutionPath:
    async def stop(self) -> None:
        return None


class _FakeAggregatorCoordinator:
    def __init__(self) -> None:
        self.is_running = False
        self.solution_path_manager = _FakeSolutionPath()
        self.initialized: dict = {}

    async def initialize(self, **kwargs) -> None:
        self.initialized = kwargs

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False


def _patch_real_aggregator_start(monkeypatch, tmp_path: Path, **flag_overrides) -> _FakeAggregatorCoordinator:
    """Keep the real start/stop route code; fake only providers, RAG, and the coordinator."""
    fake = _FakeAggregatorCoordinator()
    flags = inactive_workflow_flags()
    flags["coordinator"] = fake
    flags.update(flag_overrides)
    patch_attributes(monkeypatch, aggregator_route, flags)
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "shared_training_file", tmp_path / "rag_shared_training.txt")
    monkeypatch.setattr(aggregator_route, "save_manual_aggregator_prompt", AsyncMock())
    monkeypatch.setattr(aggregator_route, "require_embedding_provider_ready", AsyncMock())
    monkeypatch.setattr(aggregator_route, "_require_openrouter_host_provider_available", AsyncMock())
    monkeypatch.setattr(aggregator_route.api_client_manager, "configure_role", lambda *_args: None)
    monkeypatch.setattr(aggregator_route.context_allocator, "set_context_windows", lambda *_args: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "reset", lambda: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "start_timer", lambda: None)
    monkeypatch.setattr(aggregator_route.token_tracker, "stop_timer", lambda: None)
    monkeypatch.setattr(aggregator_route.assistant_proof_search_coordinator, "stop_all", AsyncMock())

    class _Registry:
        async def acquire(self, *_args, **_kwargs):
            return fake.solution_path_manager

    import backend.shared.solution_path as solution_path_module

    monkeypatch.setattr(solution_path_module, "solution_path_registry", _Registry())
    return fake


def _template_path(root: Path) -> Path:
    return root / cli_control.START_TEMPLATE_DIR_NAME / "aggregator.json"


def _credential_keys(value) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in CREDENTIAL_MARKERS):
                found.append(str(key))
            found.extend(_credential_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_credential_keys(item))
    return found


def _cli_start_args(prompt: str) -> argparse.Namespace:
    return argparse.Namespace(config=None, prompt=prompt, prompt_file=None, set=None)


@pytest.mark.asyncio
async def test_real_start_template_replays_through_cli_into_the_next_real_start(monkeypatch, tmp_path):
    fake = _patch_real_aggregator_start(monkeypatch, tmp_path)
    adapter = ManualAggregatorAdapter()
    observation = RealWorkflowObservation(runtime_root=tmp_path)

    # First real start: its saved template is the output the next step consumes.
    observation.record("real_aggregator_start", prompt="first")
    observation.cli_start_attempts += 1
    assert (await adapter.start(minimal_aggregator_request(user_prompt="first")))["status"] == "started"
    observation.cli_committed_starts += 1
    saved = json.loads(_template_path(tmp_path).read_text(encoding="utf-8"))
    observation.cli_templates_saved += 1
    observation.cli_template_credential_leaks += len(_credential_keys(saved))
    assert saved["mode"] == "aggregator"
    assert saved["request"]["user_prompt"] == "first"
    assert (await adapter.stop())["status"] == "stopped"

    # The CLI turns that template into the next request, and the real route accepts it.
    observation.record("cli_replay_saved_template", prompt="second")
    instance = moto_cli.Instance("default", "", "", data_root=tmp_path)
    body = moto_cli.build_start_request(_cli_start_args("second"), moto_cli.MODES["aggregator"], instance)
    observation.cli_template_replays += 1
    try:
        replayed = AggregatorStartRequest.model_validate(body)
    except Exception:
        observation.cli_template_replays_rejected += 1
        raise
    observation.cli_start_attempts += 1
    assert (await adapter.start(replayed))["status"] == "started"
    observation.cli_committed_starts += 1
    resaved = json.loads(_template_path(tmp_path).read_text(encoding="utf-8"))
    observation.cli_templates_saved += 1
    observation.cli_template_credential_leaks += len(_credential_keys(resaved))
    assert fake.initialized["user_prompt"] == "second"
    assert resaved["request"]["user_prompt"] == "second"
    assert resaved["request"]["validator_model"] == saved["request"]["validator_model"]
    assert (await adapter.stop())["status"] == "stopped"

    for invariant_id in (
        "cli_control.start_template_only_after_committed_start",
        "cli_control.start_template_credentials_stripped",
        "cli_control.template_replays_into_valid_start",
    ):
        observation.observed_invariants.add(invariant_id)
        assert_observed_invariant(observation, invariant_id)


@pytest.mark.asyncio
async def test_real_conflicting_start_saves_no_template(monkeypatch, tmp_path):
    _patch_real_aggregator_start(
        monkeypatch, tmp_path, compiler_coordinator=SimpleNamespace(is_running=True)
    )
    observation = RealWorkflowObservation(runtime_root=tmp_path)
    observation.record("real_aggregator_start_while_compiler_running")
    observation.cli_start_attempts += 1
    with pytest.raises(HTTPException) as rejected:
        await ManualAggregatorAdapter().start(minimal_aggregator_request())
    assert rejected.value.status_code == 400
    if _template_path(tmp_path).exists():
        observation.cli_templates_after_rejected_start += 1
        observation.cli_templates_saved += 1

    invariant_id = "cli_control.start_template_only_after_committed_start"
    observation.observed_invariants.add(invariant_id)
    assert_observed_invariant(observation, invariant_id)


@pytest.mark.asyncio
async def test_real_hosted_start_writes_no_cli_artifacts(monkeypatch, tmp_path):
    _patch_real_aggregator_start(monkeypatch, tmp_path)
    monkeypatch.setattr(system_config, "generic_mode", True)
    monkeypatch.setattr(system_config, "desktop_api_token", "hosted-token-never-written")
    adapter = ManualAggregatorAdapter()
    observation = RealWorkflowObservation(runtime_root=tmp_path)

    observation.record("hosted_real_aggregator_start")
    observation.cli_hosted_attempts += 1
    assert (await adapter.start(minimal_aggregator_request()))["status"] == "started"
    cli_control.write_control_file(version="hosted")
    written = [
        path for path in (_template_path(tmp_path), cli_control.control_file_path(tmp_path)) if path.exists()
    ]
    observation.cli_hosted_artifacts_written += len(written)
    assert (await adapter.stop())["status"] == "stopped"

    invariant_id = "cli_control.hosted_writes_no_artifacts"
    observation.observed_invariants.add(invariant_id)
    assert_observed_invariant(observation, invariant_id)


def _lifespan_call_lines() -> dict[str, int]:
    """Line numbers of the control-file calls and the yield inside the real lifespan."""
    tree = ast.parse((REPO_ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8"))
    lifespan = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_application_lifespan"
    )
    lines: dict[str, int] = {}
    for node in ast.walk(lifespan):
        if isinstance(node, ast.Yield):
            lines["yield"] = node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"write_control_file", "remove_control_file", "_restore_leanoj_session"}:
                lines.setdefault(node.func.id, node.lineno)
    return lines


def test_control_file_published_when_ready_and_removed_only_by_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "desktop_api_token", "launch-token")
    monkeypatch.setattr(system_config, "backend_host", "127.0.0.1")
    monkeypatch.setattr(system_config, "backend_port", 8123)
    observation = RealWorkflowObservation(runtime_root=tmp_path)

    # Startup order in the real lifespan: ready work first, then publish, then serve.
    lines = _lifespan_call_lines()
    observation.record("inspect_lifespan_order", **lines)
    if not (lines["_restore_leanoj_session"] < lines["write_control_file"] < lines["yield"] < lines["remove_control_file"]):
        observation.cli_control_file_written_before_ready += 1

    # The file the backend writes is exactly what the CLI's discovery reads.
    cli_control.write_control_file(version="test")
    discovered = moto_cli.read_control_file(tmp_path)
    assert discovered is not None
    assert discovered.token == "launch-token"
    assert discovered.backend_url == "http://127.0.0.1:8123"
    assert discovered.pid == os.getpid()

    # Another process's file survives this process's shutdown cleanup.
    control = cli_control.control_file_path(tmp_path)
    payload = json.loads(control.read_text(encoding="utf-8"))
    payload["pid"] = os.getpid() + 1
    control.write_text(json.dumps(payload), encoding="utf-8")
    observation.record("remove_control_file", owner="other")
    cli_control.remove_control_file()
    if not control.exists():
        observation.cli_control_file_foreign_removals += 1

    cli_control.write_control_file(version="test")
    observation.record("remove_control_file", owner="self")
    cli_control.remove_control_file()
    assert not control.exists()

    invariant_id = "cli_control.control_file_owner_lifecycle"
    observation.observed_invariants.add(invariant_id)
    assert_observed_invariant(observation, invariant_id)
