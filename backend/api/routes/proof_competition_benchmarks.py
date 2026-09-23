"""Read-only competition reports; shared middleware authenticates all endpoints."""
from fastapi import APIRouter, HTTPException, Query, Response
from backend.autonomous.memory.session_manager import session_manager
from backend.autonomous.memory.proof_competition_benchmarks import (
    BenchmarkPage, BenchmarkReport, ProofCompetitionBenchmarkStore, list_benchmark_reports,
)

router = APIRouter(prefix="/api/proof-competition/benchmarks", tags=["Proof competition benchmarks"])


@router.get("", response_model=BenchmarkPage)
async def list_reports(response: Response, session_id: str | None = None,
                       current: bool = False, offset: int = Query(0, ge=0),
                       limit: int = Query(25, ge=1, le=100)):
    response.headers["Cache-Control"] = "no-store"
    if current:
        session_id = session_manager.session_id
        if not session_id:
            return BenchmarkPage(reports=[], total=0, offset=offset, limit=limit)
    try:
        return await list_benchmark_reports(session_id=session_id, offset=offset, limit=limit)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid benchmark session identity")


@router.get("/{session_id}/{report_id}", response_model=BenchmarkReport)
async def get_report(session_id: str, report_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await ProofCompetitionBenchmarkStore(session_id).get_report(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid benchmark identity")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark report not found")
