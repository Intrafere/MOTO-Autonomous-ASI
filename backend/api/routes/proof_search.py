"""Unified proof-search routes for MOTO and SyntheticLib4 corpora."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from backend.shared.proof_search.models import (
    ProofSearchCorpus,
    PublicProofSearchRequest,
    UnifiedProofSearchRecord,
)
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import (
    AssistantSupportLineageResponse,
    LatestAssistantProofPackResponse,
)
from backend.shared.proof_search.search_service import proof_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proof-search", tags=["proof-search"])


@router.get("/overview")
async def get_proof_search_overview():
    """Return a compact proof-corpus map for UI and AI navigation."""
    try:
        overview = await proof_search_service.overview()
    except Exception as exc:
        logger.exception("Failed to build proof-search overview")
        raise HTTPException(status_code=500, detail=f"Proof-search overview failed: {exc}") from exc
    return overview.model_dump(mode="json")


@router.get("/assistant/latest-pack", response_model=LatestAssistantProofPackResponse)
async def get_latest_assistant_proof_pack():
    """Return the latest metadata-only Assistant proof-memory pack for the workflow panel."""
    try:
        return assistant_proof_search_coordinator.get_latest_pack_payload()
    except Exception as exc:
        logger.exception("Failed to load latest Assistant proof pack")
        raise HTTPException(status_code=500, detail=f"Assistant proof pack failed: {exc}") from exc


@router.get(
    "/assistant/targets/{target_hash}/supports/{support_search_id}/lineage",
    response_model=AssistantSupportLineageResponse,
)
async def get_assistant_support_lineage(
    target_hash: str,
    support_search_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
):
    """Return bounded metadata-only provenance for one Assistant support."""
    pack = assistant_proof_search_coordinator.get_latest_pack(target_hash)
    support = next(
        (item for item in pack.results if item.search_id == support_search_id),
        None,
    ) if pack else None
    if support is None:
        raise HTTPException(status_code=404, detail="Assistant target or support not found")
    total, occurrences = await proof_search_service.support_lineage(
        theorem_statement_hash=support.theorem_statement_hash,
        lean_code_hash=support.lean_code_hash,
        corpora=[support.corpus],
        exclude_run_ids=[support.run_id] if support.run_id else [],
        exclude_session_ids=[support.session_id] if support.session_id else [],
        offset=offset,
        limit=limit,
    )
    payload = {
        "target_hash": target_hash,
        "support_search_id": support_search_id,
        "occurrence_total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(occurrences) if offset + len(occurrences) < total else None,
        "occurrences": occurrences,
    }
    return payload


@router.post("/search")
async def search_proofs(request: PublicProofSearchRequest):
    """Search up to seven combined proof records across indexed corpora."""
    try:
        response = await proof_search_service.search(request)
    except Exception as exc:
        logger.exception("Proof search failed")
        raise HTTPException(status_code=500, detail=f"Proof search failed: {exc}") from exc
    return response.model_dump(mode="json")


@router.get("/proofs/{source}/{proof_id}", response_model=UnifiedProofSearchRecord)
async def get_proof_search_record(
    source: ProofSearchCorpus,
    proof_id: str,
    session_id: str | None = None,
    search_id: str | None = None,
    run_id: str | None = None,
):
    """Return one indexed proof record, hydrating SyntheticLib4 code when available."""
    try:
        record = await proof_search_service.get_record(
            corpus=source,
            proof_id=proof_id,
            session_id=session_id,
            search_id=search_id,
            run_id=run_id,
        )
    except ValueError as exc:
        logger.warning("Proof-search record hydration rejected: %s", exc)
        raise HTTPException(status_code=409, detail=f"Proof-search hydration rejected: {exc}") from exc
    except Exception as exc:
        logger.exception("Proof-search record hydration failed")
        raise HTTPException(status_code=500, detail=f"Proof-search hydration failed: {exc}") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Proof record not found")
    return record


@router.post("/reindex")
async def reindex_proofs():
    """Rebuild the local proof-search index from available sources."""
    try:
        overview = await proof_search_service.rebuild_index()
    except Exception as exc:
        logger.exception("Proof-search reindex failed")
        raise HTTPException(status_code=500, detail=f"Proof-search reindex failed: {exc}") from exc
    return {"success": True, "overview": overview.model_dump(mode="json")}

