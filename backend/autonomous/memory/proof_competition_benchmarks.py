"""Durable, session-owned competition telemetry (never prompt or API-log storage).

Execution callers await ensure_report() only for enabled competitions, then record()
with cumulative snapshots. revision monotonically increases per competitor/problem;
replays are no-ops. Set started_at only when formalization actually begins, never at
queue time. Effective routes describe observed calls, not the configured shortcut.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.shared.config import system_config
from backend.shared.path_safety import resolve_path_within_root, validate_single_path_component

CAVEAT = (
    "Conditional fallback benchmark, not an overall model ranking: later competitors "
    "see only earlier failures. Pairwise scores use shared actually-started problems, "
    "not queued work. Supporting proof context can evolve between competitors; "
    "compare context revisions and effective routes. Unknown token usage is not zero."
)


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkRoute(BenchmarkModel):
    provider: str = Field(max_length=100)
    model_id: str = Field(max_length=250)
    host: str | None = Field(default=None, max_length=100)
    reasoning_effort: str | None = Field(default=None, max_length=40)
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    boosted: bool = False
    fallback: bool = False
    supercharge: bool = False


class BenchmarkRecord(BenchmarkModel):
    execution_id: str = Field(min_length=1, max_length=250)
    source_type: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=250)
    round_index: int = Field(default=1, ge=1)
    candidate_id: str = Field(min_length=1, max_length=250)
    competitor_id: str = Field(min_length=1, max_length=100)
    competitor_order: int = Field(ge=0)
    route_revision: str = Field(min_length=1, max_length=100)
    configured_route: BenchmarkRoute
    effective_routes: list[BenchmarkRoute] = Field(default_factory=list, max_length=100)
    revision: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts_consumed: int = Field(default=0, ge=0)
    outcome: Literal["queued", "running", "verified", "exhausted", "interrupted", "unavailable", "cancelled", "skipped"] = "queued"
    proof_id: str | None = Field(default=None, max_length=250)
    interruption_kind: str | None = Field(default=None, max_length=100)
    elapsed_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    context_revision: str | None = Field(default=None, max_length=250)
    support_ids: list[str] = Field(default_factory=list, max_length=100)

    @property
    def problem_key(self) -> str:
        return _digest([self.execution_id, self.source_type, self.source_id, self.round_index, self.candidate_id])

    @property
    def competitor_key(self) -> str:
        return _digest([self.competitor_id, self.route_revision])

    @property
    def record_key(self) -> str:
        return _digest([self.problem_key, self.competitor_key])


class CompetitorScore(BenchmarkModel):
    competitor_key: str
    competitor_id: str
    route_revision: str
    configured_route: BenchmarkRoute
    verified: int = 0
    completed_losses: int = 0
    incomplete: int = 0
    solve_rate: float | None = None


class PairwiseComparison(BenchmarkModel):
    shared_problem_count: int
    problem_keys: list[str]
    shared_records: list[BenchmarkRecord] = Field(default_factory=list)
    left: CompetitorScore
    right: CompetitorScore
    conditional_rescue_count: int = 0
    conditional_rescue_opportunities: int = 0
    conditional_rescue_rate: float | None = None


class BenchmarkSummary(BenchmarkModel):
    session_id: str
    report_id: str
    run_id: str
    created_at: str
    updated_at: str
    record_count: int


class BenchmarkReport(BenchmarkSummary):
    records: list[BenchmarkRecord]
    pairwise: list[PairwiseComparison]
    attempted: int
    verified: int
    secondary_rescues: int
    unavailable: int
    interrupted: int
    caveat: str = CAVEAT


class BenchmarkPage(BenchmarkModel):
    reports: list[BenchmarkSummary]
    total: int
    offset: int
    limit: int


def _digest(parts: list) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_report(summary: BenchmarkSummary, records: list[BenchmarkRecord]) -> BenchmarkReport:
    groups: dict[str, dict[str, BenchmarkRecord]] = {}
    for row in records:
        if row.started_at is not None and row.outcome not in {"queued", "skipped"}:
            groups.setdefault(row.competitor_key, {})[row.problem_key] = row
    pairwise = []
    # Include configured but not-yet-started competitors so empty comparisons are explicit.
    competitors = {row.competitor_key: row for row in records}
    ordered = sorted(competitors, key=lambda key: (competitors[key].competitor_order, key))
    for left_key, right_key in combinations(ordered, 2):
        shared = sorted(set(groups.get(left_key, {})) & set(groups.get(right_key, {})))
        scores = []
        for key in (left_key, right_key):
            sample = competitors[key]
            rows = [groups[key][problem] for problem in shared]
            successes = sum(row.outcome == "verified" for row in rows)
            losses = sum(row.outcome == "exhausted" for row in rows)
            scores.append(CompetitorScore(
                competitor_key=key, competitor_id=sample.competitor_id,
                route_revision=sample.route_revision, configured_route=sample.configured_route,
                verified=successes, completed_losses=losses,
                incomplete=len(rows) - successes - losses,
                solve_rate=successes / (successes + losses) if successes + losses else None,
            ))
        opportunities = [problem for problem in shared
                         if groups[left_key][problem].outcome == "exhausted"
                         and groups[right_key][problem].outcome in {"verified", "exhausted"}]
        rescues = sum(groups[right_key][problem].outcome == "verified" for problem in opportunities)
        pairwise.append(PairwiseComparison(shared_problem_count=len(shared), problem_keys=shared,
                                           shared_records=[groups[key][problem] for problem in shared for key in (left_key, right_key)],
                                           left=scores[0], right=scores[1],
                                           conditional_rescue_count=rescues,
                                           conditional_rescue_opportunities=len(opportunities),
                                           conditional_rescue_rate=rescues / len(opportunities) if opportunities else None))
    attempted = [row for row in records if row.started_at is not None and row.outcome not in {"queued", "skipped"}]
    return BenchmarkReport(**summary.model_dump(), records=records, pairwise=pairwise,
        attempted=len(attempted), verified=sum(row.outcome == "verified" for row in attempted),
        secondary_rescues=sum(row.outcome == "verified" and row.competitor_order > 0 for row in attempted),
        unavailable=sum(row.outcome == "unavailable" for row in records),
        interrupted=sum(row.outcome == "interrupted" for row in records))


class ProofCompetitionBenchmarkStore:
    """SQLite transactions serialize snapshots across helper instances and cancellation.

    base_dir is trusted operator/test configuration, never accepted from HTTP callers.
    report_id should be stable across session Stop/Start; execution_id separates manual
    checks and source/round lifecycles; route_revision separates changed configurations.
    """

    def __init__(self, session_id: str, base_dir: Path | str | None = None):
        self.session_id = validate_single_path_component(session_id, "session ID")
        self.base_dir = Path(base_dir or system_config.auto_sessions_base_dir)

    def _path(self) -> Path:
        unresolved = self.base_dir / self.session_id
        if unresolved.is_symlink():
            raise ValueError("Session aliases are not benchmark owners")
        session = resolve_path_within_root(self.base_dir, self.session_id)
        if not session.is_dir():
            raise FileNotFoundError("Autonomous session not found")
        return resolve_path_within_root(session, "proof_competition.sqlite")

    @contextmanager
    def _connect(self, create: bool = False):
        path = self._path()
        if not create and not path.is_file():
            raise FileNotFoundError("Benchmark report not found")
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        if create:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    report_id TEXT NOT NULL, record_key TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (report_id, record_key));
            """)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    async def ensure_report(self, report_id: str, *, run_id: str, enabled: bool = True) -> None:
        if not enabled:
            return
        validate_single_path_component(report_id, "report ID")
        await asyncio.to_thread(self._ensure_report, report_id, run_id)

    def _ensure_report(self, report_id, run_id):
        with self._connect(create=True) as conn:
            existing = conn.execute("SELECT run_id FROM reports WHERE report_id=?", (report_id,)).fetchone()
            if existing and existing[0] != run_id:
                raise ValueError("Report belongs to a different run")
            now = _now()
            conn.execute("INSERT OR IGNORE INTO reports VALUES (?, ?, ?, ?)", (report_id, run_id, now, now))

    async def record(self, report_id: str, record: BenchmarkRecord) -> bool:
        validate_single_path_component(report_id, "report ID")
        row = BenchmarkRecord.model_validate(record).model_copy(deep=True)
        if row.outcome in {"running", "verified", "exhausted"} and row.started_at is None:
            raise ValueError("Attempt outcome requires an actual start")
        if row.outcome in {"queued", "skipped"} and (row.started_at or row.attempts_consumed):
            raise ValueError("Queued/skipped work cannot contain attempts")
        if row.attempts_consumed and row.started_at is None:
            raise ValueError("Consumed attempts require an actual start")
        return await asyncio.to_thread(self._record, report_id, row)

    def _record(self, report_id, row):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("SELECT 1 FROM reports WHERE report_id=?", (report_id,)).fetchone():
                raise FileNotFoundError("Benchmark report not found")
            found = conn.execute("SELECT payload FROM records WHERE report_id=? AND record_key=?", (report_id, row.record_key)).fetchone()
            if found:
                previous = BenchmarkRecord.model_validate_json(found[0])
                if row.revision <= previous.revision:
                    return False
                if row.configured_route != previous.configured_route:
                    raise ValueError("Changed configured route requires a new route revision")
                if row.attempts_consumed < previous.attempts_consumed or (previous.started_at and row.started_at != previous.started_at):
                    raise ValueError("Benchmark attempt state cannot regress")
                if previous.outcome in {"verified", "exhausted", "skipped"} and row.outcome != previous.outcome:
                    raise ValueError("Completed benchmark outcome cannot regress")
            conn.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?)", (report_id, row.record_key, row.model_dump_json()))
            conn.execute("UPDATE reports SET updated_at=? WHERE report_id=?", (_now(), report_id))
            return True

    async def get_report(self, report_id: str) -> BenchmarkReport:
        validate_single_path_component(report_id, "report ID")
        return await asyncio.to_thread(self._get_report, report_id)

    def _get_report(self, report_id):
        with self._connect() as conn:
            conn.execute("BEGIN")
            report = conn.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
            if not report:
                raise FileNotFoundError("Benchmark report not found")
            records = [BenchmarkRecord.model_validate_json(row[0]) for row in conn.execute(
                "SELECT payload FROM records WHERE report_id=? ORDER BY record_key", (report_id,))]
        summary = BenchmarkSummary(session_id=self.session_id, record_count=len(records), **dict(report))
        return build_report(summary, records)

    def _list_reports(self) -> list[BenchmarkSummary]:
        try:
            with self._connect() as conn:
                return [BenchmarkSummary(session_id=self.session_id, **dict(row)) for row in conn.execute(
                    "SELECT r.*, (SELECT COUNT(*) FROM records x WHERE x.report_id=r.report_id) AS record_count FROM reports r")]
        except FileNotFoundError:
            return []


async def list_benchmark_reports(*, session_id: str | None = None, offset: int = 0,
                                 limit: int = 25, base_dir: Path | str | None = None) -> BenchmarkPage:
    root = Path(base_dir or system_config.auto_sessions_base_dir)
    if session_id:
        validate_single_path_component(session_id, "session ID")
    def read():
        sessions = [session_id] if session_id else ([p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()] if root.is_dir() else [])
        reports = []
        for session in sessions:
            reports.extend(ProofCompetitionBenchmarkStore(session, root)._list_reports())
        reports.sort(key=lambda row: (row.updated_at, row.session_id, row.report_id), reverse=True)
        return BenchmarkPage(reports=reports[offset:offset + limit], total=len(reports), offset=offset, limit=limit)
    return await asyncio.to_thread(read)
