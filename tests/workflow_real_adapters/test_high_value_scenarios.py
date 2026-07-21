from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api.routes import proofs as proofs_route
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.shared.config import system_config
from backend.shared.models import ProofSettingsUpdateRequest
from tests.workflow_real_adapters.coverage_records import (
    HIGH_VALUE_GAP_COVERAGE,
    HIGH_VALUE_REAL_COVERAGE,
)


@pytest.mark.asyncio
async def test_disabled_proof_runtime_skips_autonomous_checkpoint(monkeypatch):
    coordinator = AutonomousCoordinator()
    stage = AsyncMock()
    coordinator._proof_verification_stage = stage
    coordinator._allow_mathematical_proofs = True
    monkeypatch.setattr(system_config, "lean4_enabled", False)

    result = await coordinator._run_proof_verification(
        "Proof-bearing source.",
        "brainstorm",
        "disabled-runtime-source",
        source_title="Disabled runtime",
    )

    assert result == "complete"
    stage.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_hosted_proof_settings_returns_desktop_unavailable(monkeypatch):
    monkeypatch.setattr(system_config, "generic_mode", True)
    request = ProofSettingsUpdateRequest(enabled=True, timeout=600)

    with pytest.raises(HTTPException) as raised:
        await proofs_route.update_proof_settings(request)

    assert raised.value.status_code == 501
    assert raised.value.detail["lean4_enabled"] is False
    assert "unavailable in hosted mode" in raised.value.detail["message"]


def test_unavailable_real_seams_are_truthfully_blocked():
    assert {record.result for record in HIGH_VALUE_GAP_COVERAGE} == {"blocked"}
    assert all(record.diagnostics and record.diagnostics.get("reason") for record in HIGH_VALUE_GAP_COVERAGE)
    assert {record.result for record in HIGH_VALUE_REAL_COVERAGE} == {"passed"}
