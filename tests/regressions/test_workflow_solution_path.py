from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes.workflow import router
from backend.shared.solution_path import (
    ReviewDecision,
    SolutionRoute,
    SolutionPathManagerRegistry,
)


def test_solution_path_route_returns_typed_safe_fallback():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).get("/api/workflow/solution-path")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["enabled"] is False
    assert payload["steps"] == []
    assert payload["ordering"] == "ordered"
    assert payload["queued_proposals"] == 0
    assert payload["reviewing_proposals"] == 0
    assert payload["repair_required_proposals"] == 0
    assert isinstance(payload["message"], str)


def test_solution_path_schema_is_present_in_openapi():
    app = FastAPI()
    app.include_router(router)

    operation = app.openapi()["paths"]["/api/workflow/solution-path"]["get"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
    schema = app.openapi()["components"]["schemas"]["SolutionPathResponse"]
    assert schema["properties"]["mode"]["enum"] == [
        "idle",
        "aggregator",
        "compiler",
        "autonomous",
        "leanoj",
    ]
    assert schema["properties"]["ordering"]["enum"] == ["ordered", "unordered"]
    assert "queued_proposals" in schema["properties"]
    assert "reviewing_proposals" in schema["properties"]
    assert "lifecycle_generation" in schema["properties"]
    assert "repair_required_proposals" in schema["properties"]
    assert "repair_reason" in schema["properties"]
    assert "/api/workflow/solution-path/resume" in app.openapi()["paths"]


def test_solution_path_route_exposes_loaded_manager_before_first_plan(monkeypatch, tmp_path):
    import backend.shared.solution_path.registry as registry_module
    import backend.shared.solution_path as solution_path_package
    import backend.api.routes.workflow as workflow_module

    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    registry = SolutionPathManagerRegistry()
    manager = __import__("asyncio").run(
        registry.acquire(
            tmp_path,
            workflow_mode="manual",
            user_prompt="Persisted prompt",
            stable_run_id="manual",
            reviewer=reviewer,
        )
    )
    __import__("asyncio").run(manager.propose(SolutionRoute(steps=[])))
    __import__("asyncio").run(manager.stop())
    monkeypatch.setattr(registry_module, "solution_path_registry", registry)
    monkeypatch.setattr(solution_path_package, "solution_path_registry", registry)
    monkeypatch.setattr(workflow_module, "solution_path_registry", registry, raising=False)

    app = FastAPI()
    app.include_router(router)
    payload = TestClient(app).get("/api/workflow/solution-path").json()

    assert payload["run_id"] == "manual"
    assert payload["queued_proposals"] == 1
    assert payload["pending_proposals"] == 1
    assert payload["steps"] == []


def test_solution_path_route_selects_latest_of_multiple_resumable_runs(
    monkeypatch, tmp_path
):
    import asyncio
    import backend.shared.solution_path.registry as registry_module
    import backend.shared.solution_path as solution_path_package

    async def reviewer(proposal, plan):
        return ReviewDecision(decision="approve")

    async def prepare():
        registry = SolutionPathManagerRegistry()
        first = await registry.acquire(
            tmp_path / "manual",
            workflow_mode="manual",
            user_prompt="First prompt",
            stable_run_id="first-run",
            reviewer=reviewer,
        )
        await first.stop()
        second = await registry.acquire(
            tmp_path / "autonomous",
            workflow_mode="autonomous",
            user_prompt="Second prompt",
            stable_run_id="second-run",
            reviewer=reviewer,
        )
        await second.stop()
        return registry

    registry = asyncio.run(prepare())
    monkeypatch.setattr(registry_module, "solution_path_registry", registry)
    monkeypatch.setattr(solution_path_package, "solution_path_registry", registry)

    app = FastAPI()
    app.include_router(router)
    payload = TestClient(app).get("/api/workflow/solution-path").json()

    assert payload["ownership"] == "resumable"
    assert payload["mode"] == "autonomous"
    assert payload["run_id"] == "second-run"


def test_resume_route_requeues_only_matching_generation(monkeypatch, tmp_path):
    import asyncio
    import backend.shared.solution_path.registry as registry_module
    import backend.shared.solution_path as solution_path_package

    calls = 0

    async def reviewer(proposal, plan):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("No OpenRouter API key is available")
        return ReviewDecision(decision="approve")

    async def prepare():
        registry = SolutionPathManagerRegistry()
        manager = await registry.acquire(
            tmp_path,
            workflow_mode="manual",
            user_prompt="Repair prompt",
            stable_run_id="repair-run",
            reviewer=reviewer,
        )
        await manager.set_acceptance_count(5)
        proposal = await manager.propose(SolutionRoute(steps=[]))
        await manager.wait_idle()
        await manager.stop()
        return registry, manager, proposal

    registry, manager, proposal = asyncio.run(prepare())
    monkeypatch.setattr(registry_module, "solution_path_registry", registry)
    monkeypatch.setattr(solution_path_package, "solution_path_registry", registry)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    generation = manager.state.lifecycle_generation
    stale = client.post(
        "/api/workflow/solution-path/resume",
        json={
            "run_id": "repair-run",
            "proposal_id": proposal.proposal_id,
            "lifecycle_generation": generation - 1,
        },
    )
    assert stale.status_code == 409

    response = client.post(
        "/api/workflow/solution-path/resume",
        json={
            "run_id": "repair-run",
            "proposal_id": proposal.proposal_id,
            "lifecycle_generation": generation,
        },
    )
    assert response.status_code == 200
    assert response.json()["proposal_status"] == "queued"
