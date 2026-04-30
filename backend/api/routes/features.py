"""
Build identity and capability metadata routes.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from backend.shared.build_info import get_build_info
from backend.shared.config import system_config

router = APIRouter()
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPDATE_NOTICE_PATH = _REPO_ROOT / ".moto_update_notice.json"


@router.get("/api/features")
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
        }
    )


@router.get("/api/update-notice")
async def get_update_notice() -> Dict[str, Any]:
    """Return the launcher-written update notice, if one exists."""
    try:
        payload = json.loads(_UPDATE_NOTICE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("update_available"):
            return payload
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"update_available": False}
