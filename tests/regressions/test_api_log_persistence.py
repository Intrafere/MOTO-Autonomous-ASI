import json
from pathlib import Path

import pytest

from backend.autonomous.memory.autonomous_api_logger import AutonomousAPILogger
from backend.shared.boost_logger import BoostLogger
from backend.shared.config import system_config


@pytest.fixture
def isolated_loggers(tmp_path, monkeypatch):
    monkeypatch.setattr(system_config, "data_dir", tmp_path)
    monkeypatch.setattr(system_config, "generic_mode", False)
    monkeypatch.setattr(system_config, "api_log_store_full_payloads", True)

    autonomous = AutonomousAPILogger()
    autonomous._prepared_root_identity = None
    autonomous._volatile_payloads.clear()
    boost = BoostLogger()
    boost._prepared_root_identity = None
    boost._volatile_payloads.clear()
    yield autonomous, boost, tmp_path
    autonomous._volatile_payloads.clear()
    boost._volatile_payloads.clear()


@pytest.mark.asyncio
async def test_full_payload_debugging_is_volatile(isolated_loggers):
    autonomous, boost, root = isolated_loggers
    await autonomous.log_api_call(
        task_id="task-1",
        role_id="role-1",
        model="model",
        provider="openrouter",
        prompt="password=prompt-secret",
        response_content="Authorization: Bearer response-secret",
    )
    await boost.log_boost_call(
        task_id="boost-1",
        role_id="role-1",
        model="model",
        prompt_preview="safe preview",
        response_content="token=boost-response-secret",
    )

    autonomous_disk = (root / "auto_api_log.txt").read_text(encoding="utf-8")
    boost_disk = (root / "boost_api_log.txt").read_text(encoding="utf-8")
    assert "prompt-secret" not in autonomous_disk
    assert "response-secret" not in autonomous_disk
    assert "boost-response-secret" not in boost_disk

    [autonomous_detail] = await autonomous.get_logs(include_full=True)
    [boost_detail] = await boost.get_logs(include_full=True)
    assert autonomous_detail["prompt_full"] == "password=prompt-secret"
    assert autonomous_detail["response_full"] == "Authorization: Bearer response-secret"
    assert boost_detail["response_full"] == "token=boost-response-secret"


@pytest.mark.asyncio
async def test_hosted_mode_ignores_full_payload_debug_flag(isolated_loggers, monkeypatch):
    autonomous, boost, root = isolated_loggers
    monkeypatch.setattr(system_config, "generic_mode", True)
    await autonomous.log_api_call(
        task_id="task-2",
        role_id="role-2",
        model="model",
        provider="openrouter",
        prompt="password=hosted-prompt-secret",
        response_content="Authorization: Bearer hosted-response-secret",
    )
    await boost.log_boost_call(
        task_id="boost-2",
        role_id="role-2",
        model="model",
        prompt_preview="safe preview",
        response_content="token=hosted-boost-secret",
    )

    [autonomous_detail] = await autonomous.get_logs(include_full=True)
    [boost_detail] = await boost.get_logs(include_full=True)
    assert "prompt_full" not in autonomous_detail
    assert "response_full" not in autonomous_detail
    assert "response_full" not in boost_detail
    assert "hosted-prompt-secret" not in (root / "auto_api_log.txt").read_text(encoding="utf-8")
    assert "hosted-boost-secret" not in (root / "boost_api_log.txt").read_text(encoding="utf-8")


def test_legacy_full_payloads_are_scrubbed_atomically(isolated_loggers):
    autonomous, _, root = isolated_loggers
    log_path = Path(root) / "auto_api_log.txt"
    log_path.write_text(json.dumps({
        "prompt_full": "password=legacy-secret",
        "response_full": "Authorization: Bearer legacy-response",
    }) + "\n", encoding="utf-8")
    autonomous._prepared_root_identity = None
    autonomous._prepare_active_root()

    persisted = log_path.read_text(encoding="utf-8")
    assert "legacy-secret" not in persisted
    assert "legacy-response" not in persisted
    record = json.loads(persisted)
    assert record["prompt_size"] == len("password=legacy-secret")
    assert record["response_size"] == len("Authorization: Bearer legacy-response")
