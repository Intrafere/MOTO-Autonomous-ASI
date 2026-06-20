"""SyntheticLib4 corpus access and local proof-index control routes."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.shared.path_safety import validate_single_path_component
from backend.shared.proof_search.search_service import proof_search_service
from backend.shared.config import system_config
from backend.shared.syntheticlib4_client import (
    SYNTHETICLIB4_CONTRACT_VERSION,
    syntheticlib4_client,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/syntheticlib4", tags=["syntheticlib4"])


class SyntheticLib4AuthStartRequest(BaseModel):
    redirect_uri: str | None = None


class SyntheticLib4AuthExchangeRequest(BaseModel):
    code: str = ""
    state: str = ""
    redirect_url: str = ""
    redirect_uri: str | None = None


class SyntheticLib4ApiKeyRequest(BaseModel):
    api_key: str


class SyntheticLib4RetrieveBatchRequest(BaseModel):
    contract_version: str | None = None
    query: str = ""
    goal_statement: str = ""
    imports: list[str] = []
    dependency_names: list[str] = []
    module_filters: list[str] = []
    novelty_filters: list[str] = []
    release_id: str | None = None
    channel: str = "stable"
    excluded_fingerprints: list[str] = []
    cursor: str | None = None
    limit: int = Field(default=7, ge=1, le=7)
    include_full_code: bool = True


class SyntheticLib4ImportLocalSnapshotRequest(BaseModel):
    source_name: str
    channel: str = "stable"


async def _snapshot_status() -> dict[str, Any]:
    """Read non-secret SyntheticLib4 fixture/snapshot status off the event loop."""

    def _load() -> dict[str, Any]:
        account_status = syntheticlib4_client.get_status()
        manifest = syntheticlib4_client.get_release_manifest()
        proof_count = len(syntheticlib4_client.load_proof_metadata())
        validation = syntheticlib4_client.validate_local_snapshot()
        return {
            "account_status": account_status,
            "manifest": manifest,
            "proof_count": proof_count,
            "validation": validation,
        }

    return await asyncio.to_thread(_load)


@router.get("/status")
async def get_syntheticlib4_status():
    """Return SyntheticLib4 auth/snapshot/index status without exposing secrets."""
    try:
        snapshot = await _snapshot_status()
        overview = await proof_search_service.overview(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 status failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 status failed: {exc}") from exc

    manifest = snapshot["manifest"]
    syntheticlib4_corpus = next(
        (corpus for corpus in overview.corpora if corpus.get("id") == "syntheticlib4"),
        None,
    )
    return {
        "success": True,
        "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
        "status": snapshot["account_status"],
        "current_release": {
            "release_id": manifest.get("release_id", ""),
            "channel": manifest.get("channel", "stable"),
            "generated_at": manifest.get("generated_at", ""),
            "lean_toolchain": manifest.get("lean_toolchain", ""),
            "mathlib_revision": manifest.get("mathlib_revision", ""),
            "syntheticlib4_revision": manifest.get("syntheticlib4_revision", ""),
            "proof_count": snapshot["proof_count"],
            "schema_version": manifest.get("schema_version", ""),
        },
        "local_snapshot": {
            "available": snapshot["proof_count"] > 0,
            "proof_count": snapshot["proof_count"],
            "freshness": syntheticlib4_corpus.get("freshness") if syntheticlib4_corpus else "not indexed",
            "validation": snapshot["validation"],
        },
        "proof_index": {
            "total_records": overview.total_records,
            "syntheticlib4_records": syntheticlib4_corpus.get("count", 0) if syntheticlib4_corpus else 0,
            "result_cap": overview.result_cap,
        },
    }


@router.post("/auth/start")
async def start_syntheticlib4_auth(_: SyntheticLib4AuthStartRequest):
    """Return a clear placeholder until the production SyntheticLib4 OAuth contract is live."""
    raise HTTPException(
        status_code=501,
        detail=(
            "SyntheticLib4 hosted OAuth is not connected in this mock/offline build. "
            "Use the local snapshot and proof-search routes until SyntheticLib.com auth goes live."
        ),
    )


@router.post("/auth/exchange")
async def exchange_syntheticlib4_auth(_: SyntheticLib4AuthExchangeRequest):
    """Return a clear placeholder until the production SyntheticLib4 OAuth contract is live."""
    raise HTTPException(
        status_code=501,
        detail=(
            "SyntheticLib4 hosted OAuth exchange is not connected in this mock/offline build."
        ),
    )


@router.post("/api-key")
async def set_syntheticlib4_api_key(request: SyntheticLib4ApiKeyRequest):
    """Store a SyntheticLib4 API key through the mode-appropriate secret path."""
    try:
        status = await asyncio.to_thread(syntheticlib4_client.set_api_key, request.api_key)
    except Exception as exc:
        logger.exception("SyntheticLib4 API-key setup failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 API-key setup failed: {exc}") from exc
    return {
        "success": True,
        "message": (
            "SyntheticLib4 API key stored for this MOTO instance. Live SyntheticLib.com "
            "validation will activate when the production service contract is available."
        ),
        "status": status,
    }


@router.delete("/auth")
async def clear_syntheticlib4_auth():
    """Clear SyntheticLib4 auth state without deleting local snapshots."""
    try:
        status = await asyncio.to_thread(syntheticlib4_client.clear_credentials)
    except Exception as exc:
        logger.exception("SyntheticLib4 auth clear status failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 auth clear failed: {exc}") from exc
    return {
        "success": True,
        "message": "SyntheticLib4 credentials cleared. Local snapshots and proof indexes were preserved.",
        "status": status,
    }


@router.get("/releases")
async def list_syntheticlib4_releases(channel: str | None = None):
    """List locally available SyntheticLib4 releases."""
    try:
        releases = await asyncio.to_thread(syntheticlib4_client.list_releases, channel)
    except Exception as exc:
        logger.exception("SyntheticLib4 release listing failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 release listing failed: {exc}") from exc
    return releases


@router.post("/refresh")
async def refresh_syntheticlib4_snapshot():
    """
    Refresh local SyntheticLib4 search state.

    The current client is fixture/snapshot-backed, so refresh validates available
    metadata and rebuilds the unified local proof index.
    """
    try:
        validation = await asyncio.to_thread(syntheticlib4_client.validate_local_snapshot)
        overview = await proof_search_service.rebuild_index(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 snapshot refresh failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 refresh failed: {exc}") from exc
    return {
        "success": True,
        "message": "SyntheticLib4 local snapshot metadata validated and proof index rebuilt.",
        "snapshot_validation": validation,
        "overview": overview.model_dump(mode="json"),
    }


@router.post("/import-local-snapshot")
async def import_syntheticlib4_local_snapshot(request: SyntheticLib4ImportLocalSnapshotRequest):
    """
    Activate a local snapshot staged under `data/syntheticlib4/imports/{source_name}`.

    This route is intentionally path-component based rather than accepting an
    arbitrary host path. It gives the future downloader/control-plane a safe
    activation surface while preserving the existing active snapshot on failure.
    """
    try:
        source_name = validate_single_path_component(request.source_name, "SyntheticLib4 snapshot import name")
        source_dir = Path(system_config.data_dir) / "syntheticlib4" / "imports" / source_name
        result = await asyncio.to_thread(
            syntheticlib4_client.import_snapshot_directory,
            source_dir,
            channel=request.channel,
        )
        overview = await proof_search_service.rebuild_index(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 local snapshot import failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 local snapshot import failed: {exc}") from exc
    return {
        **result,
        "overview": overview.model_dump(mode="json"),
    }


@router.post("/reindex")
async def reindex_syntheticlib4_proofs():
    """Rebuild the unified proof-search index from available local proof corpora."""
    try:
        overview = await proof_search_service.rebuild_index(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 proof index rebuild failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 reindex failed: {exc}") from exc
    return {"success": True, "overview": overview.model_dump(mode="json")}


@router.post("/retrieve-batch")
async def retrieve_syntheticlib4_batch(request: SyntheticLib4RetrieveBatchRequest):
    """Return a bounded mock/offline SyntheticLib4 retrieve-batch response."""
    try:
        response = await asyncio.to_thread(
            syntheticlib4_client.retrieve_batch,
            request.model_dump(mode="json"),
        )
    except Exception as exc:
        logger.exception("SyntheticLib4 retrieve-batch failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 retrieve-batch failed: {exc}") from exc
    return response


@router.get("/account/proofs")
async def list_syntheticlib4_account_proofs(
    cursor: str | None = None,
    limit: int = 50,
    release_id: str | None = None,
    channel: str | None = None,
):
    """Browse accepted SyntheticLib4 account proofs through the mock/offline contract."""
    try:
        return await asyncio.to_thread(
            syntheticlib4_client.list_account_proofs,
            cursor=cursor,
            limit=limit,
            release_id=release_id,
            channel=channel,
        )
    except Exception as exc:
        logger.exception("SyntheticLib4 account proof listing failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 account proof listing failed: {exc}") from exc


@router.get("/account/proofs/search")
async def search_syntheticlib4_account_proofs(
    q: str = "",
    module: str | None = None,
    novelty_rank: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
):
    """Search accepted SyntheticLib4 account proofs through the mock/offline contract."""
    try:
        return await asyncio.to_thread(
            syntheticlib4_client.search_user_proofs,
            query=q,
            module=module,
            novelty_rank=novelty_rank,
            cursor=cursor,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("SyntheticLib4 account proof search failed")
        raise HTTPException(status_code=500, detail=f"SyntheticLib4 account proof search failed: {exc}") from exc
