import asyncio

import pytest

from backend.shared.solution_path import (
    ReviewDecision,
    SolutionPathManagerRegistry,
    stable_solution_path_run_id,
)


async def reviewer(proposal, plan):
    return ReviewDecision(decision="approve")


@pytest.mark.asyncio
async def test_registry_reuses_paused_manager_and_preserves_provenance(tmp_path):
    registry = SolutionPathManagerRegistry()
    first = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt="  Prove the target  ",
        stable_run_id="manual",
        reviewer=reviewer,
    )
    await first.set_acceptance_count(7)
    await first.stop()

    resumed = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt="Prove the target",
        stable_run_id="manual",
        reviewer=reviewer,
    )
    assert resumed is first
    assert resumed.state.acceptance_count == 7
    assert resumed.state.workflow_mode == "manual"
    assert resumed.state.user_prompt == "Prove the target"
    assert resumed.state.prompt_hash


@pytest.mark.asyncio
async def test_registry_rejects_prompt_collision_and_clear_deletes(tmp_path):
    registry = SolutionPathManagerRegistry()
    manager = await registry.acquire(
        tmp_path,
        workflow_mode="leanoj",
        user_prompt="Prompt A",
        stable_run_id="run-a",
        reviewer=reviewer,
    )
    state_path = tmp_path / "run-a" / "solution_path_state.json"
    assert state_path.exists()

    with pytest.raises(ValueError, match="user_prompt mismatch"):
        # A fresh registry exercises persisted provenance validation.
        await SolutionPathManagerRegistry().acquire(
            tmp_path,
            workflow_mode="leanoj",
            user_prompt="Prompt B",
            stable_run_id="run-a",
            reviewer=reviewer,
        )

    await registry.clear_manager(manager)
    assert not state_path.exists()
    assert registry.get("run-a") is None


@pytest.mark.asyncio
async def test_clear_run_preserves_unrelated_solution_path_histories(tmp_path):
    registry = SolutionPathManagerRegistry()
    manual = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt="Manual prompt",
        stable_run_id="manual",
        reviewer=reviewer,
    )
    autonomous = await registry.acquire(
        tmp_path,
        workflow_mode="autonomous",
        user_prompt="Autonomous prompt",
        stable_run_id="session-a",
        reviewer=reviewer,
    )

    await registry.clear_run(tmp_path, manual.run_id)

    assert not (tmp_path / "manual").exists()
    assert (tmp_path / "session-a" / "solution_path_state.json").exists()
    assert registry.get("session-a") is autonomous


def test_factory_identity_is_stable_and_mode_scoped():
    assert stable_solution_path_run_id("autonomous", " same ") == stable_solution_path_run_id(
        "autonomous", "same"
    )
    assert stable_solution_path_run_id("autonomous", "same") != stable_solution_path_run_id(
        "leanoj", "same"
    )


@pytest.mark.asyncio
async def test_registry_canonicalizes_root_and_validates_loaded_provenance(tmp_path):
    registry = SolutionPathManagerRegistry()
    first = await registry.acquire(
        tmp_path / "paths" / "..",
        workflow_mode="manual",
        user_prompt="Prompt A",
        stable_run_id="same-run",
        reviewer=reviewer,
    )
    resumed = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt=" Prompt A ",
        stable_run_id="same-run",
        reviewer=reviewer,
    )

    assert resumed is first
    with pytest.raises(ValueError, match="workflow_mode mismatch"):
        await registry.acquire(
            tmp_path,
            workflow_mode="autonomous",
            user_prompt="Prompt A",
            stable_run_id="same-run",
            reviewer=reviewer,
        )
    with pytest.raises(ValueError, match="user_prompt mismatch"):
        await registry.acquire(
            tmp_path,
            workflow_mode="manual",
            user_prompt="Prompt B",
            stable_run_id="same-run",
            reviewer=reviewer,
        )


@pytest.mark.asyncio
async def test_clear_run_linearizes_with_reacquire_and_advances_generation(
    tmp_path, monkeypatch
):
    registry = SolutionPathManagerRegistry()
    first = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt="Prompt",
        stable_run_id="run",
        reviewer=reviewer,
    )
    clear_entered = asyncio.Event()
    release_clear = asyncio.Event()
    original_clear = first.clear

    async def blocked_clear():
        clear_entered.set()
        await release_clear.wait()
        await original_clear()

    monkeypatch.setattr(first, "clear", blocked_clear)
    clear_task = asyncio.create_task(registry.clear_run(tmp_path, "run"))
    await clear_entered.wait()
    acquire_task = asyncio.create_task(
        registry.acquire(
            tmp_path,
            workflow_mode="manual",
            user_prompt="Prompt",
            stable_run_id="run",
            reviewer=reviewer,
        )
    )
    await asyncio.sleep(0)
    assert not acquire_task.done()

    lifecycle = await registry._lifecycle(tmp_path, "run")
    generation_during_clear = lifecycle.generation
    release_clear.set()
    await clear_task
    second = await acquire_task

    assert second is not first
    assert lifecycle.generation == generation_during_clear + 1
    assert registry.get("run", tmp_path) is second
    assert (tmp_path / "run" / "solution_path_state.json").exists()


@pytest.mark.asyncio
async def test_unrelated_run_lifecycle_does_not_share_lock(tmp_path, monkeypatch):
    registry = SolutionPathManagerRegistry()
    run_a = await registry.acquire(
        tmp_path,
        workflow_mode="manual",
        user_prompt="A",
        stable_run_id="run-a",
        reviewer=reviewer,
    )
    clear_entered = asyncio.Event()
    release_clear = asyncio.Event()
    original_clear = run_a.clear

    async def blocked_clear():
        clear_entered.set()
        await release_clear.wait()
        await original_clear()

    monkeypatch.setattr(run_a, "clear", blocked_clear)
    clear_task = asyncio.create_task(registry.clear_run(tmp_path, "run-a"))
    await clear_entered.wait()

    run_b = await asyncio.wait_for(
        registry.acquire(
            tmp_path,
            workflow_mode="autonomous",
            user_prompt="B",
            stable_run_id="run-b",
            reviewer=reviewer,
        ),
        timeout=1,
    )
    assert registry.get("run-b", tmp_path) is run_b

    release_clear.set()
    await clear_task


@pytest.mark.asyncio
async def test_same_run_id_is_independent_across_roots(tmp_path):
    registry = SolutionPathManagerRegistry()
    first = await registry.acquire(
        tmp_path / "one",
        workflow_mode="manual",
        user_prompt="One",
        stable_run_id="shared",
        reviewer=reviewer,
    )
    second = await registry.acquire(
        tmp_path / "two",
        workflow_mode="autonomous",
        user_prompt="Two",
        stable_run_id="shared",
        reviewer=reviewer,
    )

    assert first is not second
    assert registry.get("shared", tmp_path / "one") is first
    assert registry.get("shared", tmp_path / "two") is second
    with pytest.raises(ValueError, match="ambiguous"):
        registry.get("shared")
