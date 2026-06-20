"""Unified proof-search routes for MOTO and SyntheticLib4 corpora."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.shared.proof_search.models import (
    ProofSearchCorpus,
    PublicProofSearchRequest,
    UnifiedProofSearchRecord,
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
):
    """Return one indexed proof record, hydrating SyntheticLib4 code when available."""
    try:
        record = await proof_search_service.get_record(
            corpus=source,
            proof_id=proof_id,
            session_id=session_id,
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

