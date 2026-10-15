"""SyntheticLib4 corpus access and local proof-index control routes."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.shared.path_safety import validate_single_path_component
from backend.shared.proof_search.search_service import proof_search_service
from backend.shared.config import system_config
from backend.shared.syntheticlib4_client import (
    SYNTHETICLIB4_CONTRACT_VERSION,
    syntheticlib4_client,
)
from backend.shared.syntheticlib_product import (
    get_syntheticlib_product_state,
    require_syntheticlib_developer_tooling,
    require_syntheticlib_runtime,
    syntheticlib_runtime_allowed,
)
from backend.shared.proof_trade_authorization import get_authorization_readiness
from backend.shared.proof_trade_ledger import (
    ProofTradeLedgerRejected,
    ProofTradeLedgerUnavailable,
    archive_and_reset_local_ledger,
    cancel_batch,
    cancel_selection_snapshot,
    create_selection_snapshot,
    delete_archived_local_ledgers,
    get_ledger_status,
    list_ledger_entries,
    prepare_batch,
    reconcile_selection_snapshot,
)
from backend.shared.syntheticlib_trade_eligibility import project_installation_proof_stores
from backend.shared.syntheticlib_legacy_port import (
    LegacyPortUnavailable,
    migrate_legacy_proofs,
)
from backend.shared.trade_data_boundary import trade_safe_error_detail

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
    if not syntheticlib_runtime_allowed():
        return {
            "success": True,
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "product_state": get_syntheticlib_product_state().value,
            "runtime_available": False,
            "status": {
                "authenticated": False,
                "credential_configured": False,
                "auth_mode": "none",
            },
            "current_release": None,
            "local_snapshot": None,
            "proof_index": None,
        }
    try:
        snapshot = await _snapshot_status()
        overview = await proof_search_service.overview(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 status failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 status failed", exc)) from exc

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
    require_syntheticlib_runtime()
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
    require_syntheticlib_runtime()
    raise HTTPException(
        status_code=501,
        detail=(
            "SyntheticLib4 hosted OAuth exchange is not connected in this mock/offline build."
        ),
    )


@router.post("/api-key")
async def set_syntheticlib4_api_key(request: SyntheticLib4ApiKeyRequest):
    """Store a SyntheticLib4 API key through the mode-appropriate secret path."""
    require_syntheticlib_runtime()
    try:
        status = await asyncio.to_thread(syntheticlib4_client.set_api_key, request.api_key)
    except Exception as exc:
        logger.exception("SyntheticLib4 API-key setup failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 API-key setup failed", exc)) from exc
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
    require_syntheticlib_runtime()
    try:
        status = await asyncio.to_thread(syntheticlib4_client.clear_credentials)
    except Exception as exc:
        logger.exception("SyntheticLib4 auth clear status failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 auth clear failed", exc)) from exc
    return {
        "success": True,
        "message": "SyntheticLib4 credentials cleared. Local snapshots and proof indexes were preserved.",
        "status": status,
    }


@router.get("/releases")
async def list_syntheticlib4_releases(channel: str | None = None):
    """List locally available SyntheticLib4 releases."""
    require_syntheticlib_runtime()
    try:
        releases = await asyncio.to_thread(syntheticlib4_client.list_releases, channel)
    except Exception as exc:
        logger.exception("SyntheticLib4 release listing failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 release listing failed", exc)) from exc
    return releases


@router.post("/refresh")
async def refresh_syntheticlib4_snapshot():
    """
    Refresh local SyntheticLib4 search state.

    The current client is fixture/snapshot-backed, so refresh validates available
    metadata and rebuilds the unified local proof index.
    """
    require_syntheticlib_runtime()
    try:
        validation = await asyncio.to_thread(syntheticlib4_client.validate_local_snapshot)
        overview = await proof_search_service.rebuild_index(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 snapshot refresh failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 refresh failed", exc)) from exc
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
    require_syntheticlib_runtime()
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
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 local snapshot import failed", exc)) from exc
    return {
        **result,
        "overview": overview.model_dump(mode="json"),
    }


@router.post("/reindex")
async def reindex_syntheticlib4_proofs():
    """Rebuild the unified proof-search index from available local proof corpora."""
    require_syntheticlib_runtime()
    try:
        overview = await proof_search_service.rebuild_index(include_disabled=True)
    except Exception as exc:
        logger.exception("SyntheticLib4 proof index rebuild failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 reindex failed", exc)) from exc
    return {"success": True, "overview": overview.model_dump(mode="json")}


@router.post("/retrieve-batch")
async def retrieve_syntheticlib4_batch(request: SyntheticLib4RetrieveBatchRequest):
    """Return a bounded mock/offline SyntheticLib4 retrieve-batch response."""
    require_syntheticlib_runtime()
    try:
        response = await asyncio.to_thread(
            syntheticlib4_client.retrieve_batch,
            request.model_dump(mode="json"),
        )
    except Exception as exc:
        logger.exception("SyntheticLib4 retrieve-batch failed")
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 retrieve-batch failed", exc)) from exc
    return response


@router.get("/account/proofs")
async def list_syntheticlib4_account_proofs(
    cursor: str | None = None,
    limit: int = 50,
    release_id: str | None = None,
    channel: str | None = None,
):
    """Browse accepted SyntheticLib4 account proofs through the mock/offline contract."""
    require_syntheticlib_runtime()
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
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 account proof listing failed", exc)) from exc


@router.get("/account/proofs/search")
async def search_syntheticlib4_account_proofs(
    q: str = "",
    module: str | None = None,
    novelty_rank: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
):
    """Search accepted SyntheticLib4 account proofs through the mock/offline contract."""
    require_syntheticlib_runtime()
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
        raise HTTPException(status_code=500, detail=trade_safe_error_detail("SyntheticLib4 account proof search failed", exc)) from exc


class TradeEvidenceDiagnosticResponse(BaseModel):
    """Metadata-only diagnostic for one canonical occurrence or malformed artifact."""

    record_key: str
    store_key: str
    client_proof_id: str
    canonical_locator_digest: str
    canonical_theorem_statement_hash: str
    canonical_lean_code_hash: str
    created_at: str
    proof_set_revision: int | None
    source_selection_policy_version: str
    source_category: str
    source_selected: bool
    evidence_complete: bool
    eligible_to_offer: Literal[False]
    reason_codes: list[str]
    evidence_class: Literal["full", "legacy_port", "none"]


class TradeEvidenceStoreSummaryResponse(BaseModel):
    """Path-free summary of one canonical proof store."""

    store_key: str
    store_role: str
    stable_store_id: str
    proof_set_revision: int | None
    projection_fingerprint: str
    record_count: int
    evidence_complete_count: int
    source_selected_count: int
    evidence_complete: bool
    reason_codes: list[str]


class TradeEvidenceDiagnosticsResponse(BaseModel):
    """Read-only local proof-trade evidence audit; never an eligibility verdict."""

    success: Literal[True] = True
    product_state: str
    schema_version: str
    source_selection_policy_version: str
    projection_fingerprint: str
    evidence_complete: bool
    eligible_to_offer: Literal[False]
    reason_codes: list[str]
    stores: list[TradeEvidenceStoreSummaryResponse]
    diagnostics: list[TradeEvidenceDiagnosticResponse]


@router.get(
    "/developer/trade-evidence",
    response_model=TradeEvidenceDiagnosticsResponse,
)
async def get_syntheticlib_trade_evidence_diagnostics():
    """Audit canonical proof stores for future proof trade (local preview only).

    Product authority (shared with the proof-history offer panel). The audit is
    read-only and metadata-only: it never contacts SyntheticLib, inspects credentials, reads
    proof-search indexes, or marks any proof eligible.
    """
    require_syntheticlib_runtime()
    sessions_root = system_config.auto_sessions_base_dir
    try:
        projection = await asyncio.to_thread(
            project_installation_proof_stores,
            Path(system_config.data_dir),
            auto_sessions_root=Path(sessions_root) if sessions_root else None,
        )
    except Exception as exc:
        logger.exception("SyntheticLib trade-evidence audit failed")
        raise HTTPException(
            status_code=500,
            detail="SyntheticLib trade-evidence audit failed",
        ) from exc
    return TradeEvidenceDiagnosticsResponse(
        product_state=get_syntheticlib_product_state().value,
        **projection.to_dict(),
    )


class TradeLedgerStatusResponse(BaseModel):
    """Read-only, metadata-only local proof-trade ledger status."""

    success: Literal[True] = True
    product_state: str
    schema_version: str
    available: bool
    reason_code: str
    exists: bool
    ledger_generation: int | None
    snapshot_counts: dict[str, int]
    batch_counts: dict[str, int]
    account_binding_present: bool
    contract_version_present: bool
    archived_ledger_count: int
    upload_enabled: Literal[False]


@router.get(
    "/developer/trade-ledger",
    response_model=TradeLedgerStatusResponse,
)
async def get_syntheticlib_trade_ledger_status():
    """Report the local proof-trade ledger state (local preview only).

    Developer tooling: requires local developer-preview authority even after
    launch. Read-only: it never creates installation identity or ledger files,
    and an unreadable ledger is reported by reason code rather than repaired.
    """
    require_syntheticlib_developer_tooling()
    try:
        status = await asyncio.to_thread(get_ledger_status)
    except Exception as exc:
        logger.error("SyntheticLib trade-ledger status failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=trade_safe_error_detail("SyntheticLib trade-ledger status failed", exc),
        ) from exc
    return TradeLedgerStatusResponse(
        product_state=get_syntheticlib_product_state().value,
        **status,
    )


class TradeAuthorizationReadinessResponse(BaseModel):
    """Read-only, metadata-only proof-trade desktop authorization readiness."""

    success: Literal[True] = True
    product_state: str
    readiness_version: str
    pkce_method: Literal["S256"]
    redirect_policy: Literal["profile_defined"]
    required_scopes: list[str]
    profile_published: bool
    keyring_available: bool
    installation_bound: bool
    authorization_enabled: Literal[False]
    upload_enabled: Literal[False] = False
    reason_codes: list[str]


@router.get(
    "/developer/trade-authorization",
    response_model=TradeAuthorizationReadinessResponse,
)
async def get_syntheticlib_trade_authorization_readiness():
    """Report proof-trade authorization readiness (local preview only).

    Developer tooling: requires local developer-preview authority even after
    launch. Read-only: it never starts a login, opens a listener, exchanges tokens, reads or writes credentials, or
    creates installation identity. Authorization stays disabled until the
    registered native-client profile and frozen contract are ingested.
    """
    require_syntheticlib_developer_tooling()
    try:
        readiness = await asyncio.to_thread(get_authorization_readiness)
    except Exception as exc:
        logger.error(
            "SyntheticLib trade-authorization readiness failed (%s)", type(exc).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=trade_safe_error_detail(
                "SyntheticLib trade-authorization readiness failed", exc
            ),
        ) from exc
    return TradeAuthorizationReadinessResponse(
        product_state=get_syntheticlib_product_state().value,
        readiness_version=readiness.readiness_version,
        pkce_method=readiness.pkce_method,
        redirect_policy=readiness.redirect_policy,
        required_scopes=list(readiness.required_scopes),
        profile_published=readiness.profile_published,
        keyring_available=readiness.keyring_available,
        installation_bound=readiness.installation_bound,
        authorization_enabled=readiness.authorization_enabled,
        reason_codes=list(readiness.reason_codes),
    )


# Ledger controls. Snapshot creation and batch preparation (the offer panel's
# calls) require product runtime authority; every other control is developer
# tooling that needs local preview authority even after launch. Each route
# checks authority before touching the ledger, passes IDs in JSON bodies, and
# reports refusals by reason code only. Nothing here uploads, authorizes, or
# contacts SyntheticLib; batches stay blocked until proofs are canonically
# eligible, which is impossible before the frozen contract exists.

_MAX_SELECTION_SIZE = 500


class TradeLedgerSnapshotSummaryResponse(BaseModel):
    snapshot_id: str
    state: Literal["selected", "invalidated", "cancelled"]
    entry_count: int
    selection_digest: str
    reconciliation_outcome: Literal["ok", "failed"] | None
    reason_codes: list[str]


class TradeLedgerSnapshotListItem(TradeLedgerSnapshotSummaryResponse):
    created_at: str
    updated_at: str
    reconciled_at: str | None
    revision_advanced: bool


class TradeLedgerBatchListItem(BaseModel):
    client_batch_id: str
    snapshot_id: str
    state: Literal["prepared", "cancelled", "invalidated"]
    created_at: str
    updated_at: str


class TradeLedgerEntriesResponse(BaseModel):
    """Metadata-only snapshot and batch summaries; never proof entries or paths."""

    success: Literal[True] = True
    product_state: str
    schema_version: str
    available: bool
    reason_code: str
    exists: bool
    ledger_generation: int | None
    snapshots: list[TradeLedgerSnapshotListItem]
    batches: list[TradeLedgerBatchListItem]
    upload_enabled: Literal[False]


class TradeLedgerSelectionRequest(BaseModel):
    client_proof_ids: list[str] = Field(min_length=1, max_length=_MAX_SELECTION_SIZE)


class TradeLedgerSnapshotRequest(BaseModel):
    snapshot_id: str = Field(min_length=1, max_length=128)


class TradeLedgerBatchRequest(BaseModel):
    client_batch_id: str = Field(min_length=1, max_length=128)


class TradeLedgerArchiveResetRequest(BaseModel):
    confirm: Literal[True]


class TradeLedgerSnapshotMutationResponse(BaseModel):
    success: Literal[True] = True
    snapshot: TradeLedgerSnapshotSummaryResponse
    upload_enabled: Literal[False] = False


class TradeLedgerPrepareBatchResponse(BaseModel):
    success: Literal[True] = True
    outcome: Literal["existing", "prepared", "invalidated", "blocked"]
    client_batch_id: str | None
    reason_codes: list[str]
    upload_enabled: Literal[False] = False


class TradeLedgerBatchMutationResponse(BaseModel):
    success: Literal[True] = True
    client_batch_id: str
    state: Literal["prepared", "cancelled", "invalidated"]
    upload_enabled: Literal[False] = False


class TradeLedgerArchiveResetResponse(BaseModel):
    success: Literal[True] = True
    archived: bool
    server_side_deletion_claimed: Literal[False]


class TradeLedgerArchiveDeleteResponse(BaseModel):
    success: Literal[True] = True
    deleted_count: int
    skipped_count: int
    server_side_deletion_claimed: Literal[False]


async def _run_ledger_operation(
    label: str,
    operation,
    *args,
    authorize=require_syntheticlib_developer_tooling,
):
    """Run a ledger call off the event loop with reason-code-only refusals."""
    authorize()
    try:
        return await asyncio.to_thread(operation, *args)
    except ProofTradeLedgerRejected as exc:
        raise HTTPException(
            status_code=409,
            detail="Proof-trade ledger request rejected: " + ", ".join(exc.reason_codes),
        ) from None
    except ProofTradeLedgerUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Proof-trade ledger unavailable: {exc.reason_code}",
        ) from None
    except Exception as exc:
        logger.error("SyntheticLib %s failed (%s)", label, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=trade_safe_error_detail(f"SyntheticLib {label} failed", exc),
        ) from exc


@router.get(
    "/developer/trade-ledger/entries",
    response_model=TradeLedgerEntriesResponse,
)
async def list_syntheticlib_trade_ledger_entries():
    """List metadata-only snapshot and batch summaries (local preview only).

    Read-only: it never creates installation identity or ledger files.
    """
    entries = await _run_ledger_operation("trade-ledger listing", list_ledger_entries)
    return TradeLedgerEntriesResponse(
        product_state=get_syntheticlib_product_state().value,
        **entries,
    )


@router.post(
    "/developer/trade-ledger/snapshots",
    response_model=TradeLedgerSnapshotMutationResponse,
)
async def create_syntheticlib_trade_selection_snapshot(request: TradeLedgerSelectionRequest):
    """Bind explicitly selected evidence-complete proofs into a local snapshot."""
    summary = await _run_ledger_operation(
        "trade selection snapshot",
        create_selection_snapshot,
        request.client_proof_ids,
        authorize=require_syntheticlib_runtime,
    )
    return TradeLedgerSnapshotMutationResponse(snapshot=summary)


@router.post(
    "/developer/trade-ledger/snapshots/reconcile",
    response_model=TradeLedgerSnapshotMutationResponse,
)
async def reconcile_syntheticlib_trade_selection_snapshot(request: TradeLedgerSnapshotRequest):
    """Re-check a selected snapshot against canonical stores; failure is terminal."""
    summary = await _run_ledger_operation(
        "trade snapshot reconciliation", reconcile_selection_snapshot, request.snapshot_id
    )
    return TradeLedgerSnapshotMutationResponse(snapshot=summary)


@router.post(
    "/developer/trade-ledger/snapshots/cancel",
    response_model=TradeLedgerSnapshotMutationResponse,
)
async def cancel_syntheticlib_trade_selection_snapshot(request: TradeLedgerSnapshotRequest):
    """Cancel a selected snapshot and its prepared local batches."""
    summary = await _run_ledger_operation(
        "trade snapshot cancellation", cancel_selection_snapshot, request.snapshot_id
    )
    return TradeLedgerSnapshotMutationResponse(snapshot=summary)


@router.post(
    "/developer/trade-ledger/batches/prepare",
    response_model=TradeLedgerPrepareBatchResponse,
)
async def prepare_syntheticlib_trade_batch(request: TradeLedgerSnapshotRequest):
    """Reconcile a snapshot and record a local batch only if every proof is eligible."""
    result = await _run_ledger_operation(
        "trade batch preparation",
        prepare_batch,
        request.snapshot_id,
        authorize=require_syntheticlib_runtime,
    )
    return TradeLedgerPrepareBatchResponse(**result)


@router.post(
    "/developer/trade-ledger/batches/cancel",
    response_model=TradeLedgerBatchMutationResponse,
)
async def cancel_syntheticlib_trade_batch(request: TradeLedgerBatchRequest):
    """Cancel one prepared local batch before any upload."""
    result = await _run_ledger_operation("trade batch cancellation", cancel_batch, request.client_batch_id)
    return TradeLedgerBatchMutationResponse(**result)


@router.post(
    "/developer/trade-ledger/archive-reset",
    response_model=TradeLedgerArchiveResetResponse,
)
async def archive_and_reset_syntheticlib_trade_ledger(_: TradeLedgerArchiveResetRequest):
    """Archive the local ledger (even if unreadable) and start empty.

    Only local pending material is cleared; no server-side deletion is claimed.
    """
    result = await _run_ledger_operation("trade-ledger archive/reset", archive_and_reset_local_ledger)
    return TradeLedgerArchiveResetResponse(**result)


@router.post(
    "/developer/trade-ledger/archive/delete",
    response_model=TradeLedgerArchiveDeleteResponse,
)
async def delete_archived_syntheticlib_trade_ledgers(_: TradeLedgerArchiveResetRequest):
    """Permanently delete archived local ledgers (developer tooling only).

    Only regular archived files are removed; the current ledger is untouched and
    no server-side deletion is claimed.
    """
    result = await _run_ledger_operation(
        "trade-ledger archive deletion", delete_archived_local_ledgers
    )
    return TradeLedgerArchiveDeleteResponse(**result)


class LegacyPortMigrateRequest(BaseModel):
    confirm: Literal[True]
    dry_run: bool


class LegacyPortStoreResult(BaseModel):
    store_key: str
    store_role: str
    eligible_count: int
    migrated_count: int
    already_qualified_count: int
    skipped: dict[str, int]


class LegacyPortMigrateResponse(BaseModel):
    """Metadata-only legacy-proof migration result; never uploads or claims server state."""

    success: Literal[True] = True
    migration_version: str
    dry_run: bool
    eligible_count: int
    migrated_count: int
    already_qualified_count: int
    skipped: dict[str, int]
    reason_codes: list[str]
    stores: list[LegacyPortStoreResult]
    upload_enabled: Literal[False]
    server_side_claim: Literal[False]


@router.post(
    "/developer/legacy-port/migrate",
    response_model=LegacyPortMigrateResponse,
)
async def migrate_syntheticlib_legacy_proofs(request: LegacyPortMigrateRequest):
    """Mint port identity for proofs stored by released builds (explicit, one-time).

    Product authority: developer-only before launch, available to desktop users
    at launch. Refused while any workflow or manual proof run is active. Lean
    source and novelty fields are never changed and no proof-time evidence is
    invented; refusals are reason codes only.
    """
    require_syntheticlib_runtime()
    try:
        result = await migrate_legacy_proofs(dry_run=request.dry_run)
    except LegacyPortUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Legacy proof migration refused: {exc.reason_code}",
        ) from None
    except Exception as exc:
        logger.error("SyntheticLib legacy proof migration failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=trade_safe_error_detail("SyntheticLib legacy proof migration failed", exc),
        ) from exc
    return LegacyPortMigrateResponse(**result)
