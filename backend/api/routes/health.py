"""
Health and readiness routes.
"""
from fastapi import APIRouter

from backend.shared.build_info import get_build_info
from backend.shared.config import system_config

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def api_health():
    """Return a readiness payload for desktop and hosted runtime probes."""
    build_info = get_build_info()
    return {
        "status": "healthy",
        "instance_id": system_config.instance_id,
        "generic_mode": system_config.generic_mode,
        "version": build_info.version,
        "build_commit": build_info.build_commit,
    }
