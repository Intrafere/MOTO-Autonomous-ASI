"""Measured token-count ratios: thresholds, verification, safe-direction adjustment, routes."""
import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.shared.token_calibration as calibration_module
from backend.api.routes.workflow import router
from backend.shared.api_client_manager import APIClientManager
from backend.shared.config import rag_config, system_config
from backend.shared.models import ModelConfig
from backend.shared.token_calibration import (
    MAX_SAMPLES,
    MIN_SAMPLE_TOKENS,
    REQUIRED_SAMPLES,
    RESAMPLE_INTERVAL,
    TOKEN_CALIBRATION_FILENAME,
    TokenCalibration,
    normalize_model_key,
)

CLAUDE = ("claude_api", "claude-opus-5-5")
CODEX = ("openai_codex_oauth", "gpt-5.5")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store bound to an empty data root and installed as the process store."""
    monkeypatch.setattr(system_config, "data_dir", str(tmp_path))
    fresh = TokenCalibration()
    monkeypatch.setattr(calibration_module, "token_calibration", fresh)
    return fresh


def record_many(store, key, ratio, count=REQUIRED_SAMPLES, moto_tokens=100_000):
    for _ in range(count):
        assert store.record(key, moto_tokens, int(moto_tokens * ratio))


def test_lm_studio_sibling_instances_share_one_measurement():
    assert normalize_model_key("lm_studio", "qwen-3:2") == ("lm_studio", "qwen-3")
    assert normalize_model_key("openrouter", "vendor/model:free") == ("openrouter", "vendor/model:free")
    assert normalize_model_key("", "model") is None
    assert normalize_model_key("openrouter", "") is None


def test_small_prompts_are_never_compared(store):
    assert not store.record(CLAUDE, MIN_SAMPLE_TOKENS - 1, MIN_SAMPLE_TOKENS * 2)
    assert not store.wants_sample(CLAUDE, MIN_SAMPLE_TOKENS - 1)
    # A prompt must also be a meaningful share of the role's window.
    assert not store.record(CLAUDE, 50_000, 65_000, context_window=1_000_000)
    assert store.record(CLAUDE, 100_000, 130_000, context_window=1_000_000)
    assert store.snapshot()["models"][0]["samples"] == 1


def test_implausible_ratios_are_ignored(store):
    assert not store.record(CLAUDE, 100_000, 10_000)
    assert not store.record(CLAUDE, 100_000, 400_000)
    assert not store.record(CLAUDE, 100_000, 0)
    assert store.snapshot()["models"] == []


def test_nothing_changes_until_enough_measurements_agree(store):
    store.register_role("writer", *CLAUDE, 1_000_000)
    record_many(store, CLAUDE, 1.3, count=REQUIRED_SAMPLES - 1)

    assert store.verified_ratio(CLAUDE) is None
    assert store.window_scale(1_000_000) == 1.0
    assert store.snapshot()["models"][0]["state"] == "measuring"

    assert store.record(CLAUDE, 100_000, 130_000)

    assert store.verified_ratio(CLAUDE) == pytest.approx(1.3)
    assert store.adjusted_context_window(1_000_000) == int(1_000_000 / 1.3)
    entry = store.snapshot()["models"][0]
    assert entry["state"] == "counts_more"
    assert entry["applied"] is True
    assert entry["context_windows"] == [{"configured": 1_000_000, "effective": int(1_000_000 / 1.3)}]


def test_high_end_of_recent_measurements_is_used(store):
    for index in range(10):
        store.record(CLAUDE, 100_000, 100_000 + index * 1_000 + 20_000)
    # Ratios 1.20 .. 1.29: the 90th percentile is the second highest.
    assert store.verified_ratio(CLAUDE) == pytest.approx(1.28)


def test_a_matching_model_is_left_alone(store):
    store.register_role("validator", *CODEX, 400_000)
    record_many(store, CODEX, 1.015)

    assert store.window_scale(400_000) == 1.0
    assert store.adjusted_context_window(400_000) == 400_000
    entry = store.snapshot()["models"][0]
    assert entry["state"] == "matches"
    assert entry["applied"] is False


def test_a_model_that_counts_fewer_tokens_never_enlarges_the_window(store):
    store.register_role("validator", *CODEX, 400_000)
    record_many(store, CODEX, 0.8)

    assert store.window_scale(400_000) == 1.0
    assert store.adjusted_context_window(400_000) == 400_000


def test_shared_window_value_uses_the_larger_ratio(store):
    store.register_role("writer", *CLAUDE, 1_000_000)
    store.register_role("validator", *CODEX, 1_000_000)
    record_many(store, CLAUDE, 1.3)
    record_many(store, CODEX, 1.0)

    assert store.window_scale(1_000_000) == pytest.approx(1.3)
    # A window no configured role uses is untouched.
    assert store.window_scale(200_000) == 1.0


def test_reconfiguring_a_role_moves_its_window(store):
    store.register_role("writer", *CLAUDE, 1_000_000)
    record_many(store, CLAUDE, 1.3)
    assert store.window_scale(1_000_000) > 1.0

    store.register_role("writer", *CODEX, 400_000)

    assert store.window_scale(1_000_000) == 1.0
    assert store.window_scale(400_000) == 1.0


def test_adjustment_can_be_turned_off_while_measuring_continues(store):
    store.register_role("writer", *CLAUDE, 1_000_000)
    record_many(store, CLAUDE, 1.3)

    store.set_adjust_enabled(False)

    assert store.window_scale(1_000_000) == 1.0
    assert store.record(CLAUDE, 100_000, 130_000)
    entry = store.snapshot()["models"][0]
    assert entry["state"] == "counts_more"
    assert entry["applied"] is False


def test_reset_discards_one_model_or_all(store):
    record_many(store, CLAUDE, 1.3)
    record_many(store, CODEX, 1.0)

    assert store.reset(*CLAUDE) == 1
    assert [entry["model"] for entry in store.snapshot()["models"]] == [CODEX[1]]
    assert store.reset("claude_api", "unknown") == 0
    assert store.reset() == 1
    assert store.snapshot()["models"] == []


def test_once_the_window_is_full_only_some_calls_are_compared(store):
    record_many(store, CLAUDE, 1.3, count=MAX_SAMPLES)

    decisions = [store.wants_sample(CLAUDE, 500_000) for _ in range(RESAMPLE_INTERVAL * 2)]

    assert decisions.count(True) == 2
    assert store.snapshot()["models"][0]["samples"] == MAX_SAMPLES


def test_measurements_persist_and_reload(store, tmp_path):
    store.register_role("writer", *CLAUDE, 1_000_000)
    record_many(store, CLAUDE, 1.3)
    store.set_adjust_enabled(False)
    asyncio.run(store.save())

    saved = json.loads((tmp_path / TOKEN_CALIBRATION_FILENAME).read_text(encoding="utf-8"))
    assert saved["adjust_enabled"] is False
    assert saved["models"][0]["provider"] == "claude_api"
    assert len(saved["models"][0]["ratios"]) == REQUIRED_SAMPLES

    reloaded = TokenCalibration()
    assert reloaded.adjust_enabled is False
    assert reloaded.verified_ratio(CLAUDE) == pytest.approx(1.3)


def test_unreadable_file_starts_empty(store, tmp_path):
    (tmp_path / TOKEN_CALIBRATION_FILENAME).write_text("{not json", encoding="utf-8")

    assert store.snapshot()["models"] == []
    assert store.adjust_enabled is True


def test_measurements_follow_the_active_data_root(store, tmp_path, monkeypatch):
    record_many(store, CLAUDE, 1.3)
    other_root = tmp_path / "other"
    other_root.mkdir()

    monkeypatch.setattr(system_config, "data_dir", str(other_root))

    assert store.snapshot()["models"] == []


def test_input_budget_shrinks_only_for_a_verified_model(store):
    unadjusted = rag_config.get_available_input_tokens(1_000_000, 64_000)
    store.register_role("writer", *CLAUDE, 1_000_000)
    assert rag_config.get_available_input_tokens(1_000_000, 64_000) == unadjusted

    record_many(store, CLAUDE, 1.3)

    adjusted = rag_config.get_available_input_tokens(1_000_000, 64_000)
    assert adjusted == int(1_000_000 / 1.3) - 64_000 - rag_config.context_buffer_tokens
    assert adjusted < unadjusted
    # Other windows keep their configured budget.
    assert rag_config.get_available_input_tokens(400_000, 64_000) == 400_000 - 64_000 - rag_config.context_buffer_tokens


def test_a_window_too_small_to_shrink_keeps_its_configured_budget(store):
    store.register_role("writer", *CLAUDE, 100_000)
    record_many(store, CLAUDE, 1.3)

    # 100000 / 1.3 leaves no room after a 90000-token output reserve.
    assert rag_config.get_available_input_tokens(100_000, 90_000) == 100_000 - 90_000 - rag_config.context_buffer_tokens


# --------------------------------------------------------------- completed calls


def completed_response(manager, *, provider, model, prompt_tokens, boosted=False):
    return {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5},
        manager.CALL_METADATA_KEY: {
            "effective_provider": provider,
            "effective_model": model,
            "boosted": boosted,
        },
    }


async def measure(manager, response, messages, *, role_id="writer", uses_tools=False):
    manager._schedule_token_measurement(response, messages, role_id=role_id, uses_tools=uses_tools)
    tasks = list(manager._token_measurement_tasks)
    if tasks:
        await asyncio.gather(*tasks)
    return len(tasks)


@pytest.fixture
def manager(store, monkeypatch):
    import backend.shared.api_client_manager as manager_module

    monkeypatch.setattr(manager_module, "token_calibration", store)
    monkeypatch.setattr(system_config, "generic_mode", False)
    instance = APIClientManager()
    instance.configure_role(
        "writer",
        ModelConfig(
            provider="claude_api",
            model_id="claude-opus-5-5",
            context_window=200_000,
            max_output_tokens=16_000,
        ),
    )
    return instance


LARGE_PROMPT = "The quick brown fox jumps over the lazy dog. " * 6_000


def test_configuring_a_role_registers_its_window(manager, store):
    assert store.role_model_key("writer") == CLAUDE


def test_completed_call_is_compared_without_another_request(manager, store, monkeypatch):
    import backend.shared.api_client_manager as manager_module

    monkeypatch.setattr(manager_module, "count_tokens", lambda text: 60_000)
    response = completed_response(manager, provider="claude_api", model="claude-opus-5-5", prompt_tokens=78_000)

    scheduled = asyncio.run(measure(manager, response, [{"role": "user", "content": LARGE_PROMPT}]))

    assert scheduled == 1
    entry = store.snapshot()["models"][0]
    assert (entry["provider"], entry["model"], entry["samples"]) == ("claude_api", "claude-opus-5-5", 1)
    assert Path(system_config.data_dir, TOKEN_CALIBRATION_FILENAME).exists()


def test_tool_conversations_and_small_prompts_are_skipped(manager, store):
    response = completed_response(manager, provider="claude_api", model="claude-opus-5-5", prompt_tokens=78_000)
    large = [{"role": "user", "content": LARGE_PROMPT}]

    async def run():
        assert await measure(manager, response, large, uses_tools=True) == 0
        assert await measure(manager, response, [{"role": "user", "content": "short"}]) == 0
        assert await measure(
            manager,
            response,
            [*large, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}],
        ) == 0
        assert await measure(manager, response, [{"role": "user", "content": [{"type": "text", "text": "x"}]}]) == 0
        assert await measure(manager, {"choices": []}, large) == 0

    asyncio.run(run())
    assert store.snapshot()["models"] == []


def test_boost_and_fallback_calls_are_measured_under_the_answering_model(manager, store, monkeypatch):
    import backend.shared.api_client_manager as manager_module

    monkeypatch.setattr(manager_module, "count_tokens", lambda text: 60_000)
    messages = [{"role": "user", "content": LARGE_PROMPT}]

    async def run():
        await measure(
            manager,
            completed_response(manager, provider="openrouter", model="vendor/boost", prompt_tokens=61_000, boosted=True),
            messages,
        )
        await measure(
            manager,
            completed_response(manager, provider="lm_studio", model="local-model:2", prompt_tokens=70_000),
            messages,
        )

    asyncio.run(run())
    measured = {(entry["provider"], entry["model"]) for entry in store.snapshot()["models"]}
    assert measured == {("openrouter", "vendor/boost"), ("lm_studio", "local-model")}


def test_a_provider_alias_is_measured_under_the_configured_model(manager, store, monkeypatch):
    import backend.shared.api_client_manager as manager_module

    monkeypatch.setattr(manager_module, "count_tokens", lambda text: 60_000)
    response = completed_response(manager, provider="claude_api", model="claude-opus-5-5-20261001", prompt_tokens=78_000)

    asyncio.run(measure(manager, response, [{"role": "user", "content": LARGE_PROMPT}]))

    assert [(entry["provider"], entry["model"]) for entry in store.snapshot()["models"]] == [CLAUDE]


# ------------------------------------------------------------------------ routes


def test_routes_report_toggle_and_reset(store):
    store.register_role("writer", *CLAUDE, 1_000_000)
    record_many(store, CLAUDE, 1.3)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    status = client.get("/api/token-calibration").json()
    assert status["adjust_enabled"] is True
    assert status["models"][0]["state"] == "counts_more"
    assert status["models"][0]["applied"] is True

    disabled = client.post("/api/token-calibration/settings", json={"adjust_enabled": False}).json()
    assert disabled["adjust_enabled"] is False
    assert disabled["models"][0]["applied"] is False
    assert store.window_scale(1_000_000) == 1.0

    assert client.post("/api/token-calibration/reset", json={"provider": "claude_api"}).status_code == 400
    cleared = client.post(
        "/api/token-calibration/reset", json={"provider": "claude_api", "model": "claude-opus-5-5"}
    ).json()
    assert cleared["models"] == []


def test_routes_are_in_the_openapi_schema():
    app = FastAPI()
    app.include_router(router)
    paths = app.openapi()["paths"]

    assert "get" in paths["/api/token-calibration"]
    assert "post" in paths["/api/token-calibration/settings"]
    assert "post" in paths["/api/token-calibration/reset"]
