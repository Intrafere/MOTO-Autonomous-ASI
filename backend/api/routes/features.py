"""
Build identity and capability metadata routes.
"""
import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from backend.shared.build_info import get_build_info
from backend.shared.config import system_config
from backend.shared.subscription_providers.metadata import CLOUD_PROVIDERS
from backend.shared.syntheticlib_product import get_syntheticlib_product_state
from backend.shared.trade_data_boundary import redact_trade_text

router = APIRouter()
logger = logging.getLogger(__name__)


class SubscriptionProviderCapability(BaseModel):
    """Capability-only description of one desktop subscription provider."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    label: str
    auth_kind: Literal["oauth_pkce", "api_key"]
    available: bool


class FeaturesResponse(BaseModel):
    """Stable public build identity and deployment capability contract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: str
    build_commit: str
    update_channel: str
    api_contract_version: str
    generic_mode: bool
    lm_studio_enabled: bool
    pdf_download_available: bool
    openai_codex_oauth_available: bool
    xai_grok_oauth_available: bool
    sakana_fugu_available: bool
    kimi_code_available: bool
    subscription_providers: List[SubscriptionProviderCapability]
    syntheticlib_product_state: Literal[
        "coming_soon",
        "developer_preview",
        "launched",
    ]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPDATE_NOTICE_PATH = _REPO_ROOT / ".moto_update_notice.json"
_UPDATE_NOTICE_REFRESH_INTERVAL_SECONDS = 4 * 60 * 60
_update_notice_refresh_lock = asyncio.Lock()


class _UpdateNoticeRefreshState:
    def __init__(self) -> None:
        self.last_refresh_at = time.monotonic()


_update_notice_refresh_state = _UpdateNoticeRefreshState()


@router.get("/api/features", response_model=FeaturesResponse)
async def get_features() -> Dict[str, Any]:
    """
    Return the public build-identity and capability contract.

    The identity fields remain stable for update comparison while the capability
    flags expose mode-level behavior without leaking per-user runtime state.
    """
    is_generic = system_config.generic_mode
    return get_build_info().as_features_payload(
        {
            "generic_mode": is_generic,
            "lm_studio_enabled": not is_generic,
            "pdf_download_available": not is_generic,
            **{provider.capability_flag: not is_generic for provider in CLOUD_PROVIDERS},
            "subscription_providers": [
                {
                    "id": provider.provider_id,
                    "label": provider.label,
                    "auth_kind": provider.auth_kind,
                    "available": not (is_generic and provider.desktop_only),
                }
                for provider in CLOUD_PROVIDERS
            ],
            "syntheticlib_product_state": get_syntheticlib_product_state().value,
        }
    )


@router.get("/api/update-notice")
async def get_update_notice() -> Dict[str, Any]:
    """Return an update notice, refreshing it periodically while the app runs."""
    notice = _read_update_notice()
    if notice.get("update_available"):
        return notice

    await _refresh_runtime_update_notice_if_due()
    return _read_update_notice()


def _read_update_notice() -> Dict[str, Any]:
    try:
        payload = json.loads(_UPDATE_NOTICE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("update_available"):
            return payload
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"update_available": False}
    return {"update_available": False}


async def _refresh_runtime_update_notice_if_due() -> None:
    """Check GitHub at most every 4 hours when no launcher notice exists."""
    if system_config.generic_mode:
        return

    now = time.monotonic()
    if now - _update_notice_refresh_state.last_refresh_at < _UPDATE_NOTICE_REFRESH_INTERVAL_SECONDS:
        return

    async with _update_notice_refresh_lock:
        now = time.monotonic()
        if now - _update_notice_refresh_state.last_refresh_at < _UPDATE_NOTICE_REFRESH_INTERVAL_SECONDS:
            return
        _update_notice_refresh_state.last_refresh_at = now

        try:
            from moto_updater import check_for_updates, write_update_notice

            result = await asyncio.to_thread(
                check_for_updates,
                exclude_instance_id=system_config.instance_id,
            )
            await asyncio.to_thread(write_update_notice, result)
        except Exception as exc:
            logger.warning("Runtime update notice refresh failed: %s", redact_trade_text(exc))
