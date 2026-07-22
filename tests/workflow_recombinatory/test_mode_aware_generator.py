from __future__ import annotations

from pathlib import Path

import pytest

from tests.workflow_harness import actions
from tests.workflow_harness.recombinatory_generator import (
    ACTION_SPECS,
    DEFAULT_TARGETS,
    ActionSpec,
    GeneratedRunFailure,
    GenerationTarget,
    REPLAY_NOT_REPRODUCED,
    ReplayRejected,
    action_specs_by_id,
    eligible_action_specs,
    format_failure,
    reduce_failing_action_ids,
    replay_action_ids,
    run_generated_scenario,
    validate_action_specs,
    validate_generation_target,
)
from tests.workflow_harness.exercise_evidence import observed_exercise_tokens
from tests.workflow_harness.model import WorkflowEvent, WorkflowMode, WorkflowModel


def _target(target_id: str) -> GenerationTarget:
    return next(target for target in DEFAULT_TARGETS if target.target_id == target_id)


def test_same_seed_and_target_produce_same_action_ids_and_replay(tmp_path):
    target = _target("generated_allowed_outputs_provider_pause_resume")
    first = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path / "first"),
        target,
        seed=41,
    )
    second = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path / "second"),
        target,
        seed=41,
    )

    assert first.action_ids == second.action_ids
    assert first.replay == second.replay


def test_fixed_seed_matrix_preserves_shortest_path_even_when_seed_varies(tmp_path):
    target = _target("generated_assistant_prompt_context")
    runs = {
        run_generated_scenario(
            WorkflowModel(runtime_root=tmp_path / f"seed-{seed}"),
            target,
            seed=seed,
        ).action_ids
        for seed in (7, 19, 41, 73, 101)
    }

    assert runs == {
        (
            "start_autonomous_papers_and_proofs",
            "refresh_assistant_pack",
            "prepare_prompt_context",
            "disable_session_history",
        )
    }


def test_eligible_actions_are_state_aware_and_do_not_include_no_ops(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)
    idle_ids = {spec.action_id for spec in eligible_action_specs(model)}

    assert "start_autonomous_proofs_only" in idle_ids
    assert "complete_brainstorm" not in idle_ids
    assert "resume" not in idle_ids

    actions.start_autonomous_proofs_only(model)
    active_ids = {spec.action_id for spec in eligible_action_specs(model)}

    assert "complete_topic_exploration" in active_ids
    assert "start_autonomous_proofs_only" not in active_ids
    assert "attempt_conflicting_start" in active_ids


def test_default_registry_and_targets_are_valid():
    validate_action_specs()
    for target in DEFAULT_TARGETS:
        validate_generation_target(target)


def test_invalid_registry_field_is_rejected():
    invalid = (
        ActionSpec(
            action_id="invalid",
            execute=actions.stop,
            fields=("not_a_field",),
            eligible_when=lambda model: True,
        ),
    )

    with pytest.raises(AssertionError, match="unknown fields"):
        validate_action_specs(invalid)


def test_target_with_duplicate_fields_is_rejected():
    target = GenerationTarget(
        target_id="duplicate_fields",
        fields=("runtime_exclusivity", "runtime_exclusivity"),
        invariants=("runtime.child_tasks_count_as_active",),
        must_exercise=(),
    )

    with pytest.raises(AssertionError, match="at least two unique fields"):
        validate_generation_target(target)


def test_missing_required_evidence_reports_generation_metadata(tmp_path):
    target = GenerationTarget(
        target_id="impossible_in_one_step",
        fields=("provider_pause_resume", "workflow_filesystem_state"),
        invariants=("provider.pause_preserves_checkpoint",),
        must_exercise=("provider_pause",),
        max_steps=1,
    )

    with pytest.raises(GeneratedRunFailure) as error:
        run_generated_scenario(
            WorkflowModel(runtime_root=tmp_path),
            target,
            seed=41,
        )

    message = str(error.value)
    assert "Seed: 41" in message
    assert "Fields: provider_pause_resume x workflow_filesystem_state" in message
    assert "Action count: 0" in message
    assert "Invariant: generator.target_unreachable" in message
    assert "Replay:" in message


def test_failure_formatter_contains_numbered_replay():
    message = format_failure(
        seed=7,
        fields=("allowed_outputs", "proof_runtime_gating"),
        action_count=2,
        invariant="outputs.at_least_one_output_enabled",
        replay=("first()", "second()"),
        detail="failed",
    )

    assert "Seed: 7" in message
    assert "Fields: allowed_outputs x proof_runtime_gating" in message
    assert "Action count: 2" in message
    assert "Invariant: outputs.at_least_one_output_enabled" in message
    assert "1. first()" in message
    assert "2. second()" in message


def test_every_default_target_can_complete_for_normal_seed(tmp_path):
    for target in DEFAULT_TARGETS:
        result = run_generated_scenario(
            WorkflowModel(runtime_root=tmp_path / target.target_id),
            target,
            seed=41,
        )
        assert set(target.must_exercise).issubset(result.observed_evidence)
        assert set(target.fields).issubset(result.observed_fields)
        assert set(target.invariants) == result.exercised_invariants
        assert result.action_count <= target.max_steps
        assert result.action_ids


@pytest.mark.parametrize(
    ("target_id", "required_actions"),
    [
        (
            "generated_manual_aggregator_lifecycle",
            {
                "start_manual_aggregator",
                "accept_manual_aggregator_submission",
                "clear_manual_aggregator_state",
            },
        ),
        (
            "generated_leanoj_durable_lifecycle",
            {
                "start_leanoj",
                "edit_leanoj_master_proof",
                "skip_leanoj_brainstorm",
                "force_leanoj_brainstorm",
                "stop",
                "resume",
                "clear_leanoj_state",
            },
        ),
        (
            "generated_autonomous_paper_checkpoint",
            {
                "start_autonomous_papers_and_proofs",
                "attempt_conflicting_start",
                "complete_topic_exploration",
                "force_paper_writing",
                "enter_autonomous_paper_checkpoint",
                "complete_autonomous_paper_checkpoint",
            },
        ),
    ],
)
def test_representative_new_action_families_are_generated(
    tmp_path, target_id, required_actions
):
    result = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path / target_id),
        _target(target_id),
        seed=41,
    )
    assert required_actions.issubset(result.action_ids)


def test_all_action_specs_have_stable_unique_ids():
    ids = [spec.action_id for spec in ACTION_SPECS]
    assert ids == list(dict.fromkeys(ids))


def _replay_fixture():
    def start(model):
        model.record("fixture_start")
        model.mode = WorkflowMode.AUTONOMOUS

    def noise_one(model):
        model.record("fixture_noise_one")

    def noise_two(model):
        model.record("fixture_noise_two")

    def trigger(model):
        model.record("fixture_trigger")
        model.terminal_stop_events += 1

    idle = lambda model: model.mode is WorkflowMode.NONE
    active = lambda model: model.mode is WorkflowMode.AUTONOMOUS
    specs = (
        ActionSpec("fixture.start", start, ("runtime_exclusivity",), idle),
        ActionSpec("fixture.noise_one", noise_one, ("runtime_exclusivity",), active),
        ActionSpec("fixture.noise_two", noise_two, ("runtime_exclusivity",), active),
        ActionSpec("fixture.trigger", trigger, ("runtime_exclusivity",), active),
    )
    return specs, lambda: WorkflowModel(runtime_root=Path("."))


def test_action_lookup_and_replay_reproduce_named_failure_category():
    specs, model_factory = _replay_fixture()

    assert tuple(action_specs_by_id(specs)) == tuple(spec.action_id for spec in specs)
    result = replay_action_ids(
        model_factory(),
        ("fixture.start", "fixture.noise_one", "fixture.trigger"),
        action_specs=specs,
        failure_predicate=lambda model: model.terminal_stop_events == 1,
        failure_category="fixture.terminal_stop",
    )

    assert result.failed
    assert result.failure_category == "fixture.terminal_stop"
    assert result.action_ids == (
        "fixture.start",
        "fixture.noise_one",
        "fixture.trigger",
    )
    assert result.replay[-1] == "fixture_trigger()"


def test_replay_rejects_unknown_and_state_ineligible_action_ids():
    specs, model_factory = _replay_fixture()

    with pytest.raises(ReplayRejected, match="Unknown action ID"):
        replay_action_ids(model_factory(), ("fixture.unknown",), action_specs=specs)
    with pytest.raises(ReplayRejected, match="ineligible"):
        replay_action_ids(model_factory(), ("fixture.trigger",), action_specs=specs)


def test_requested_replay_category_reports_not_reproduced_exactly():
    specs, model_factory = _replay_fixture()

    result = replay_action_ids(
        model_factory(),
        ("fixture.start", "fixture.noise_one"),
        action_specs=specs,
        failure_predicate=lambda model: model.terminal_stop_events == 1,
        failure_category="fixture.terminal_stop",
    )

    assert result.failure_category == REPLAY_NOT_REPRODUCED
    assert not result.failed
    assert result.failure_detail == (
        "Requested failure category 'fixture.terminal_stop' was not reproduced."
    )


def test_requested_replay_category_does_not_accept_different_invariant_failure():
    specs, model_factory = _replay_fixture()

    def break_invariant(model):
        model.record("fixture_break_invariant")
        model.prompt_user_direct_injected = False

    breaking_specs = specs + (
        ActionSpec(
            "fixture.break_invariant",
            break_invariant,
            ("runtime_exclusivity",),
            lambda model: model.mode is WorkflowMode.AUTONOMOUS,
        ),
    )
    result = replay_action_ids(
        model_factory(),
        ("fixture.start", "fixture.break_invariant"),
        action_specs=breaking_specs,
        failure_category="fixture.terminal_stop",
    )

    assert result.failure_category == REPLAY_NOT_REPRODUCED
    assert "observed 'prompt.user_prompt_direct_injected'" in result.failure_detail


def test_reducer_finds_shortest_prefix_and_one_minimal_failure():
    specs, model_factory = _replay_fixture()
    original = (
        "fixture.start",
        "fixture.noise_one",
        "fixture.noise_two",
        "fixture.trigger",
        "fixture.noise_one",
    )
    predicate = lambda model: model.terminal_stop_events == 1
    reduction = reduce_failing_action_ids(
        model_factory,
        original,
        action_specs=specs,
        failure_predicate=predicate,
        failure_category="fixture.terminal_stop",
    )

    assert reduction.shortest_failing_prefix == original[:4]
    assert reduction.action_ids == ("fixture.start", "fixture.trigger")
    assert reduction.minimized

    for index in range(len(reduction.action_ids)):
        candidate = reduction.action_ids[:index] + reduction.action_ids[index + 1 :]
        try:
            result = replay_action_ids(
                model_factory(),
                candidate,
                action_specs=specs,
                failure_predicate=predicate,
                failure_category="fixture.terminal_stop",
            )
        except ReplayRejected:
            continue
        assert not result.failed


def test_reducer_returns_original_for_successful_replay_without_minimizing():
    specs, model_factory = _replay_fixture()
    original = ("fixture.start", "fixture.noise_one")

    reduction = reduce_failing_action_ids(
        model_factory,
        original,
        action_specs=specs,
        failure_predicate=lambda model: model.terminal_stop_events == 1,
        failure_category="fixture.terminal_stop",
    )

    assert reduction.action_ids == original
    assert reduction.failure_category is None
    assert not reduction.minimized


def test_replay_reduction_is_deterministic():
    specs, model_factory = _replay_fixture()
    original = (
        "fixture.start",
        "fixture.noise_one",
        "fixture.noise_two",
        "fixture.trigger",
    )
    kwargs = {
        "action_specs": specs,
        "failure_predicate": lambda model: model.terminal_stop_events == 1,
        "failure_category": "fixture.terminal_stop",
    }

    assert reduce_failing_action_ids(model_factory, original, **kwargs) == (
        reduce_failing_action_ids(model_factory, original, **kwargs)
    )


@pytest.mark.parametrize("seed", range(64))
def test_every_seed_reaches_every_default_target_within_budget(tmp_path, seed):
    for target in DEFAULT_TARGETS:
        result = run_generated_scenario(
            WorkflowModel(runtime_root=tmp_path / f"{target.target_id}-{seed}"),
            target,
            seed=seed,
        )
        assert set(target.must_exercise).issubset(result.observed_evidence)
        assert set(target.fields).issubset(result.observed_fields)
        assert result.action_count <= target.max_steps


def test_planner_avoids_high_weight_dead_end_action(tmp_path):
    target = GenerationTarget(
        target_id="dead_end_adversary",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        must_exercise=("start_blocked",),
        max_steps=2,
    )
    specs = (
        ActionSpec(
            action_id="clear_dead_end",
            execute=actions.clear,
            fields=("runtime_exclusivity", "websocket_api_contracts"),
            eligible_when=lambda model: True,
            weight=10_000,
        ),
        next(spec for spec in ACTION_SPECS if spec.action_id == "start_manual_compiler"),
        next(spec for spec in ACTION_SPECS if spec.action_id == "attempt_conflicting_start"),
    )

    result = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path),
        target,
        seed=7,
        action_specs=specs,
    )

    assert result.action_ids == ("start_manual_compiler", "attempt_conflicting_start")


def test_planner_returns_true_shortest_path_not_depth_first_first_hit(tmp_path):
    target = GenerationTarget(
        target_id="shortest_path_adversary",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        must_exercise=("start_blocked",),
        max_steps=3,
    )

    def enter_long(model):
        model.record("enter_long")
        model.checkpoint["route"] = "long-start"

    def continue_long(model):
        model.record("continue_long")
        model.checkpoint["route"] = "long-ready"

    def enter_short(model):
        model.record("enter_short")
        model.checkpoint["route"] = "short-ready"

    def trigger(model):
        model.record("trigger")
        model.emit("start_blocked", active_mode="fixture")

    specs = (
        ActionSpec(
            "fixture.enter_long",
            enter_long,
            target.fields,
            lambda model: "route" not in model.checkpoint,
            weight=10_000,
        ),
        ActionSpec(
            "fixture.continue_long",
            continue_long,
            target.fields,
            lambda model: model.checkpoint.get("route") == "long-start",
        ),
        ActionSpec(
            "fixture.enter_short",
            enter_short,
            target.fields,
            lambda model: "route" not in model.checkpoint,
        ),
        ActionSpec(
            "fixture.trigger",
            trigger,
            target.fields,
            lambda model: model.checkpoint.get("route") in {"long-ready", "short-ready"},
        ),
    )

    result = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path),
        target,
        seed=7,
        action_specs=specs,
    )

    assert result.action_ids == ("fixture.enter_short", "fixture.trigger")


@pytest.mark.parametrize("history_kind", ["events", "persisted_events"])
def test_planning_signature_preserves_relevant_event_history(tmp_path, history_kind):
    target = GenerationTarget(
        target_id=f"{history_kind}_history_adversary",
        fields=("runtime_exclusivity", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        must_exercise=("start_blocked",),
        max_steps=2,
    )

    def record_dead_history(model):
        model.record("record_dead_history")
        getattr(model, history_kind).append(WorkflowEvent("dead_history"))

    def record_live_history(model):
        model.record("record_live_history")
        getattr(model, history_kind).append(WorkflowEvent("live_history"))

    def trigger(model):
        model.record("history_trigger")
        model.emit("start_blocked", active_mode="fixture")

    def live_history(model):
        return any(
            event.event_type == "live_history" for event in getattr(model, history_kind)
        )

    specs = (
        ActionSpec("fixture.dead", record_dead_history, target.fields, lambda model: True),
        ActionSpec("fixture.live", record_live_history, target.fields, lambda model: True),
        ActionSpec("fixture.trigger", trigger, target.fields, live_history),
    )

    result = run_generated_scenario(
        WorkflowModel(runtime_root=tmp_path),
        target,
        seed=19,
        action_specs=specs,
    )

    assert result.action_ids == ("fixture.live", "fixture.trigger")


def test_build_e_evidence_requires_concrete_observations_not_default_flags(tmp_path):
    model = WorkflowModel(runtime_root=tmp_path)

    assert model.prompt_user_direct_injected
    assert model.direct_sources_excluded_from_rag
    assert model.mandatory_source_overflow_visible
    assert not {
        "prompt_context",
        "rag_source_exclusion",
        "mandatory_source_overflow",
    }.intersection(observed_exercise_tokens(model))

    model.prepare_prompt_context()
    model.verify_rag_source_exclusion()
    model.reject_mandatory_source_overflow()

    assert {
        "prompt_context",
        "rag_source_exclusion",
        "mandatory_source_overflow",
    }.issubset(observed_exercise_tokens(model))


def test_unreachable_target_fails_before_executing_real_model(tmp_path):
    target = GenerationTarget(
        target_id="unreachable_adversary",
        fields=("provider_pause_resume", "workflow_filesystem_state"),
        invariants=("provider.pause_preserves_checkpoint",),
        must_exercise=("provider_pause",),
        max_steps=1,
    )
    model = WorkflowModel(runtime_root=tmp_path)

    with pytest.raises(GeneratedRunFailure) as error:
        run_generated_scenario(model, target, seed=73)

    message = str(error.value)
    assert "Action count: 0" in message
    assert "Invariant: generator.target_unreachable" in message
    assert "No path within 1 steps" in message
    assert "missing evidence ['provider_pause']" in message
    assert "visited " in message
    assert model.replay == []
