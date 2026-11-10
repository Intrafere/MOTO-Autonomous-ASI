"""Boost enable/update must refuse limits that would make every boosted call fail."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.api.routes import boost as boost_routes
from backend.shared.models import BoostConfig


def _config(**overrides) -> BoostConfig:
    base = dict(
        enabled=True,
        openrouter_api_key="",
        boost_model_id="test/model",
        boost_context_window=8192,
        boost_max_output_tokens=2048,
    )
    base.update(overrides)
    return BoostConfig(**base)


def test_validate_accepts_positive_limits():
    boost_routes._validate_boost_limits(_config())


@pytest.mark.parametrize("field", ["boost_max_output_tokens", "boost_context_window"])
def test_validate_rejects_zero_limits(field):
    with pytest.raises(HTTPException) as raised:
        boost_routes._validate_boost_limits(_config(**{field: 0}))
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_enable_route_rejects_zero_output_before_key_lookup(monkeypatch):
    def unexpected_key_lookup(_key):
        raise AssertionError("key resolution must not run for an invalid config")

    monkeypatch.setattr(boost_routes, "_resolve_boost_api_key", unexpected_key_lookup)
    with pytest.raises(HTTPException) as raised:
        await boost_routes.enable_boost(_config(boost_max_output_tokens=0))
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_update_route_rejects_zero_output(monkeypatch):
    with pytest.raises(HTTPException) as raised:
        await boost_routes.update_boost_model(_config(boost_max_output_tokens=0))
    assert raised.value.status_code == 400
