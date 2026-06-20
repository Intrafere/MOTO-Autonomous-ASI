"""Shared proof-search service used by routes and future AI tool adapters."""
from __future__ import annotations

import asyncio
from pathlib import Path

from backend.shared.config import system_config
from backend.shared.proof_search.indexer import ProofSearchIndexer
from backend.shared.proof_search.models import (
    CorpusOverview,
    ProofSearchRequest,
    ProofSearchResponse,
    UnifiedProofSearchRecord,
    default_proof_search_corpora,
)
from backend.shared.proof_search.moto_sources import load_moto_proof_records
from backend.shared.proof_search.syntheticlib4_sources import (
    load_syntheticlib4_fixture_records,
    normalize_syntheticlib4_record,
)
from backend.shared.syntheticlib4_client import syntheticlib4_client


class ProofSearchService:
    """Coordinates source normalization and the local SQLite proof-search index."""

    def __init__(self, index_path: Path | None = None) -> None:
        self._explicit_index_path = Path(index_path) if index_path else None
        self._lock = asyncio.Lock()

    @property
    def index_path(self) -> Path:
        if self._explicit_index_path is not None:
            return self._explicit_index_path
        return Path(system_config.data_dir) / "proof_search" / "proof_search.sqlite"

    async def rebuild_index(self, *, include_disabled: bool = False) -> CorpusOverview:
        """Rebuild the unified index from currently available local sources."""
        async with self._lock:
            records = await self._load_records()
            indexer = ProofSearchIndexer(self.index_path)
            await asyncio.to_thread(indexer.rebuild, records)
            return await asyncio.to_thread(
                indexer.overview,
                None if include_disabled else default_proof_search_corpora(),
            )

    async def overview(self, *, include_disabled: bool = False) -> CorpusOverview:
        await self._ensure_index()
        return await asyncio.to_thread(
            ProofSearchIndexer(self.index_path).overview,
            None if include_disabled else default_proof_search_corpora(),
        )

    async def search(self, request: ProofSearchRequest) -> ProofSearchResponse:
        request = self._filter_request_corpora(request)
        if not request.corpora:
            return ProofSearchResponse(
                results=[],
                result_count=0,
                searched_corpora=[],
                corpus_counts={},
                ranking_notes="All proof-search memory corpora are disabled.",
                weak_result_warning="No proof-search corpora are enabled for this request.",
            )
        await self._ensure_index()
        return await asyncio.to_thread(ProofSearchIndexer(self.index_path).search, request)

    async def search_candidate_pool(
        self,
        request: ProofSearchRequest,
        *,
        pool_limit: int,
        exclude_corpus_scopes: list[str] | None = None,
        exclude_session_ids: list[str] | None = None,
    ) -> list[UnifiedProofSearchRecord]:
        """Return a wider internal candidate pool for Assistant ranking.

        Public route/tool search remains capped by the indexer's normal result
        cap; this internal path lets Assistant gather enough verified candidates
        for visit-count P-UCB/MMR selection without changing the REST contract.
        """
        request = self._filter_request_corpora(request)
        if not request.corpora:
            return []
        await self._ensure_index()
        return await asyncio.to_thread(
            ProofSearchIndexer(self.index_path).search_candidate_pool,
            request,
            pool_limit=pool_limit,
            exclude_corpus_scopes=exclude_corpus_scopes,
            exclude_session_ids=exclude_session_ids,
        )

    async def get_record(
        self,
        *,
        corpus: str,
        proof_id: str,
        session_id: str | None = None,
    ) -> UnifiedProofSearchRecord | None:
        """Fetch one indexed proof and hydrate SyntheticLib4 fixture code when available."""
        if corpus not in set(default_proof_search_corpora()):
            return None
        await self._ensure_index()
        record = await asyncio.to_thread(
            ProofSearchIndexer(self.index_path).get_record,
            corpus=corpus,
            proof_id=proof_id,
            session_id=session_id,
        )
        if record is None or record.corpus != "syntheticlib4" or record.lean_code:
            return record

        hydrated = await asyncio.to_thread(
            syntheticlib4_client.hydrate_proof,
            record.external_fingerprint or record.proof_id,
        )
        if not hydrated or not str(hydrated.get("lean_code") or "").strip():
            return record

        hydrated_record = normalize_syntheticlib4_record(
            hydrated,
            release_id=record.release_id,
            channel=record.corpus_scope or "stable",
        )
        if hydrated_record.theorem_statement_hash != record.theorem_statement_hash:
            raise ValueError("SyntheticLib4 hydration theorem-statement hash mismatch")
        if hydrated_record.lean_code_hash != record.lean_code_hash:
            raise ValueError("SyntheticLib4 hydration Lean-code hash mismatch")
        return hydrated_record

    async def _ensure_index(self) -> None:
        if self.index_path.exists() and not await asyncio.to_thread(
            self._sources_are_newer_than_index
        ):
            return
        await self.rebuild_index()

    def _filter_request_corpora(self, request: ProofSearchRequest) -> ProofSearchRequest:
        enabled = set(default_proof_search_corpora())
        requested = [corpus for corpus in request.corpora if corpus in enabled]
        if requested == request.corpora:
            return request
        return request.model_copy(update={"corpora": requested})

    def _sources_are_newer_than_index(self) -> bool:
        try:
            index_mtime = self.index_path.stat().st_mtime
        except OSError:
            return True
        data_root = Path(system_config.data_dir)
        source_roots = [
            data_root / "proofs",
            data_root / "manual_proofs",
            data_root / "manual_proof_runs",
            data_root / "auto_sessions",
            data_root / "leanoj_sessions",
            data_root / "leanoj_partial_proofs",
            data_root / "syntheticlib4",
        ]
        for root in source_roots:
            if not root.exists():
                continue
            try:
                files = root.rglob("*")
                for path in files:
                    if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".lean"}:
                        continue
                    if path.stat().st_mtime > index_mtime:
                        return True
            except OSError:
                # If source freshness cannot be determined, rebuild rather than
                # serving stale proof-history records.
                return True
        return False

    async def _load_records(self) -> list[UnifiedProofSearchRecord]:
        records: list[UnifiedProofSearchRecord] = []
        try:
            records.extend(load_syntheticlib4_fixture_records())
        except Exception:
            # SyntheticLib4 is optional; local MOTO proof search should still work.
            records.extend([])
        records.extend(await load_moto_proof_records())
        return records


proof_search_service = ProofSearchService()

