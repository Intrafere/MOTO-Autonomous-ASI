import importlib
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.prompts.completion_prompts import (
    build_completion_review_prompt,
    build_completion_self_validation_prompt,
    get_completion_review_json_schema,
    get_completion_review_system_prompt,
    get_completion_self_validation_json_schema,
    get_completion_self_validation_system_prompt,
)
from backend.autonomous.prompts.paper_continuation_prompts import (
    build_continuation_decision_prompt,
    build_continuation_validation_prompt,
    get_continuation_decision_json_schema,
    get_continuation_decision_system_prompt,
    get_continuation_validator_json_schema,
    get_continuation_validator_system_prompt,
)
from backend.autonomous.prompts.paper_reference_prompts import (
    build_additional_reference_expansion_prompt,
    build_pre_brainstorm_expansion_prompt,
    build_reference_expansion_prompt,
    build_reference_selection_prompt,
    get_additional_reference_expansion_system_prompt,
    get_pre_brainstorm_expansion_system_prompt,
    get_reference_expansion_json_schema,
    get_reference_expansion_system_prompt,
    get_reference_selection_json_schema,
    get_reference_selection_system_prompt,
)
from backend.autonomous.prompts.paper_title_exploration_prompts import (
    build_title_exploration_user_prompt,
)
from backend.autonomous.prompts.paper_title_prompts import (
    build_paper_title_prompt,
    build_paper_title_validation_prompt,
    get_paper_title_json_schema,
    get_paper_title_system_prompt,
    get_paper_title_validator_json_schema,
    get_paper_title_validator_system_prompt,
)
from backend.autonomous.prompts.topic_exploration_prompts import (
    build_exploration_user_prompt,
)
from backend.autonomous.prompts.topic_prompts import (
    build_topic_selection_prompt,
    build_topic_validation_prompt,
    get_topic_selection_json_schema,
    get_topic_selection_system_prompt,
    get_topic_validator_json_schema,
    get_topic_validator_system_prompt,
)


ENGINEERING_GOAL = (
    "Develop safer low-energy desalination for remote communities while reducing "
    "membrane fouling, toxic cleaning chemicals, and lifecycle cost."
)
SCIENCE_GOAL = (
    "Develop a testable catalyst hypothesis for selective ambient-pressure ammonia "
    "synthesis, including discriminating controls and measurable failure criteria."
)
SOFTWARE_GOAL = (
    "Design a resilient distributed protocol that preserves safety and useful "
    "availability during partitions, Byzantine faults, and rolling upgrades."
)
MATH_GOAL = (
    "Prove a new combinatorial bound for intersecting set systems with a forbidden "
    "configuration, including sharpness or a matching construction."
)
MIXED_GOAL = (
    "Optimize a flood-resilient microgrid under cost and repair constraints while "
    "proving a formal impossibility bound on simultaneous outage and redundancy targets."
)
coordinator_module = importlib.import_module(
    "backend.autonomous.core.autonomous_coordinator"
)


def _schema_fields(schema: str) -> set[str]:
    return set(re.findall(r'^\s*"([a-z_]+)"\s*:', schema, flags=re.MULTILINE))


def _assert_domain_general_strategy(text: str) -> None:
    lower = text.lower()
    assert "direct" in lower
    assert "rigor" in lower
    assert "math" in lower
    assert "proof" in lower
    assert any(
        term in lower
        for term in ("mechanism", "design", "algorithm", "experiment", "constraints")
    )


def _paper() -> dict:
    return {
        "paper_id": "paper_007",
        "title": "Prior Membrane Study",
        "reference_title_display": "Prior Membrane Study [validator ratings]",
        "abstract": "A proposed cleaning mechanism with unresolved provenance.",
        "outline": "I. Mechanism\nII. Constraints\nIII. Proposed tests",
        "word_count": 1400,
        "source_brainstorm_ids": ["topic_002"],
        "content": "Full source content with proposed controls and limitations.",
    }


@pytest.mark.parametrize(
    ("module_name", "active_output"),
    [
        (
            "topic_exploration_prompts",
            build_exploration_user_prompt(ENGINEERING_GOAL, [], []),
        ),
        ("topic_prompts", get_topic_selection_system_prompt()),
        ("completion_prompts", get_completion_review_system_prompt()),
        ("paper_reference_prompts", get_pre_brainstorm_expansion_system_prompt(3)),
        (
            "paper_title_exploration_prompts",
            build_title_exploration_user_prompt(
                ENGINEERING_GOAL,
                "Reduce fouling safely.",
                "A candidate cleaning mechanism.",
                [],
                [],
            ),
        ),
        ("paper_title_prompts", get_paper_title_system_prompt()),
        ("paper_continuation_prompts", get_continuation_decision_system_prompt()),
    ],
)
def test_strategy_modules_do_not_impose_universal_mathematics_policy(
    module_name: str,
    active_output: str,
) -> None:
    forbidden = re.compile(
        r"\b(?:must be mathematical|must use mathematics|must include (?:a )?theorem|"
        r"must include (?:a )?proof|only mathematical|mathematical form is required)\b",
        flags=re.IGNORECASE,
    )

    assert not forbidden.search(active_output), module_name
    assert re.search(r"\b(?:math|mathematical|theorem|proof)\w*\b", active_output, re.IGNORECASE)
    assert re.search(
        r"\b(?:when|whenever|where|if|conditional|appropriate|relevant)\b",
        active_output,
        re.IGNORECASE,
    )


def test_topic_exploration_preserves_full_engineering_goal_and_generalizes_routes() -> None:
    prompt = build_exploration_user_prompt(
        ENGINEERING_GOAL,
        [{"topic_id": "topic_002", "topic_prompt": "Existing route", "status": "in_progress"}],
        [_paper()],
    )

    assert ENGINEERING_GOAL in prompt
    assert "strongest credible, genuinely novel" in prompt
    assert "WHOLE question" in prompt
    assert "easier adjacent" in prompt
    for route in ("mechanism", "design route", "algorithm", "theorem", "experiment"):
        assert route in prompt
    assert "mathematics, theorem discovery, proof, and formalization remain first-class" in prompt
    assert "evidence/provenance" in prompt
    assert "proposed-test" in prompt
    assert "mathematical direction" not in prompt.lower()


@pytest.mark.parametrize(
    "system_prompt",
    [
        get_topic_selection_system_prompt(),
        get_topic_validator_system_prompt(),
    ],
)
def test_topic_system_prompts_are_direct_novel_math_aware_and_non_fabricating(
    system_prompt: str,
) -> None:
    _assert_domain_general_strategy(system_prompt)
    assert "genuinely novel" in system_prompt
    assert "user's exact objective" in system_prompt
    assert "whole question" in system_prompt.lower()
    assert "easier" in system_prompt
    assert "adjacent" in system_prompt or "background-heavy" in system_prompt
    assert "provenance" in system_prompt or "evidence-aware" in system_prompt
    assert "AI-GENERATED" in system_prompt
    assert "NEVER cite internal documents as authoritative" in system_prompt
    assert any(
        phrase in system_prompt
        for phrase in (
            "non-mathematical work is not deficient",
            "Engineering, software, strategic, and causal routes",
            "domain-grounded, evidence-aware",
        )
    )


def test_topic_builders_preserve_goal_candidates_feedback_and_phase_context() -> None:
    brainstorms = [
        {
            "topic_id": "topic_003",
            "topic_prompt": "Partition recovery",
            "status": "in_progress",
            "submission_count": 8,
            "papers_generated": ["paper_002"],
        }
    ]
    action = {
        "action": "continue_existing",
        "topic_id": "topic_003",
        "reasoning": "The recovery mechanism remains unresolved.",
    }
    selection = build_topic_selection_prompt(
        SOFTWARE_GOAL,
        brainstorms,
        [_paper()],
        rejection_context="Reject the broad survey detour.",
        candidate_questions="Candidate 1: quorum repair under partitions",
    )
    validation = build_topic_validation_prompt(SOFTWARE_GOAL, brainstorms, [_paper()], action)

    for prompt in (selection, validation):
        assert SOFTWARE_GOAL in prompt
        assert "topic_003" in prompt
        assert "Prior Membrane Study" in prompt
    assert "TOPIC EXPLORATION RESULTS" in selection
    assert "Candidate 1: quorum repair under partitions" in selection
    assert "Reject the broad survey detour." in selection
    assert "PROPOSED TOPIC SELECTION" in validation
    assert "continue_existing" in validation
    assert "The recovery mechanism remains unresolved." in validation


def test_topic_schemas_preserve_exact_fields_and_enums() -> None:
    selection = get_topic_selection_json_schema()
    validator = get_topic_validator_json_schema()

    assert _schema_fields(selection) == {"action", "topic_id", "topic_prompt", "reasoning"}
    assert '"new_topic", "continue_existing"' in selection
    assert _schema_fields(validator) == {"decision", "reasoning"}
    assert '"accept" or "reject"' in validator


@pytest.mark.asyncio
async def test_topic_execution_rejects_continuing_completed_brainstorm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AutonomousCoordinator.__new__(AutonomousCoordinator)
    coordinator._stop_event = SimpleNamespace(is_set=lambda: False)
    coordinator._current_topic_id = "topic_before"
    coordinator._acceptance_count = 4
    coordinator._consecutive_rejections = 2
    coordinator._broadcast = AsyncMock()

    metadata = SimpleNamespace(
        status="complete",
        submission_count=12,
        papers_generated=["paper_001"],
    )
    get_metadata = AsyncMock(return_value=metadata)
    monkeypatch.setattr(coordinator_module.brainstorm_memory, "get_metadata", get_metadata)

    result = await coordinator._execute_topic_selection(
        SimpleNamespace(action="continue_existing", topic_id="topic_complete")
    )

    assert result is None
    assert coordinator._current_topic_id == "topic_before"
    assert coordinator._acceptance_count == 4
    assert coordinator._consecutive_rejections == 2
    get_metadata.assert_awaited_once_with("topic_complete")
    coordinator._broadcast.assert_awaited_once()
    event_name, payload = coordinator._broadcast.await_args.args
    assert event_name == "topic_selection_rejected"
    assert "already marked complete" in payload["reasoning"]


@pytest.mark.parametrize(
    "system_prompt",
    [
        get_completion_review_system_prompt(),
        get_completion_self_validation_system_prompt(),
    ],
)
def test_completion_prompts_define_synthesis_readiness_without_empirical_overclaim(
    system_prompt: str,
) -> None:
    _assert_domain_general_strategy(system_prompt)
    assert "AI-GENERATED" in system_prompt
    assert "empirical or engineering work" in system_prompt
    assert "synthesis" in system_prompt.lower()
    assert "does NOT mean" in system_prompt or "does not assert" in system_prompt
    assert any(term in system_prompt for term in ("experiments", "experiment"))
    assert any(term in system_prompt for term in ("demonstrated", "demonstration"))


def test_completion_builders_preserve_science_goal_database_and_same_assessment_context() -> None:
    review = build_completion_review_prompt(
        SCIENCE_GOAL,
        "Identify a falsifiable surface-poisoning mechanism.",
        "Submission 1: proposed isotope control.",
        7,
        "Prior review: add discriminating measurements.",
    )
    assessment = {
        "decision": "continue_brainstorm",
        "reasoning": "Controls remain underspecified.",
        "suggested_additions": "Specify isotope and blank controls.",
    }
    self_validation = build_completion_self_validation_prompt(
        SCIENCE_GOAL,
        "Identify a falsifiable surface-poisoning mechanism.",
        "Submission 1: proposed isotope control.",
        assessment,
    )

    for prompt in (review, self_validation):
        assert SCIENCE_GOAL in prompt
        assert "Identify a falsifiable surface-poisoning mechanism." in prompt
        assert "Submission 1: proposed isotope control." in prompt
    assert "Total Accepted Submissions: 7" in review
    assert "Prior review: add discriminating measurements." in review
    assert "YOUR COMPLETION ASSESSMENT (to validate)" in self_validation
    assert "continue_brainstorm" in self_validation
    assert "Specify isotope and blank controls." in self_validation


def test_completion_schemas_preserve_exact_fields_and_enums() -> None:
    review = get_completion_review_json_schema()
    self_validation = get_completion_self_validation_json_schema()

    assert _schema_fields(review) == {"decision", "reasoning", "suggested_additions"}
    assert '"continue_brainstorm" or "write_paper"' in review
    assert _schema_fields(self_validation) == {"validated", "reasoning"}
    assert "true or false" in self_validation


@pytest.mark.parametrize(
    ("system_prompt", "cap_text"),
    [
        (get_pre_brainstorm_expansion_system_prompt(3), "up to 3 papers maximum"),
        (get_additional_reference_expansion_system_prompt(3), "max 3 total"),
        (get_reference_expansion_system_prompt(6), "up to 6 papers"),
        (get_reference_selection_system_prompt(4), "Maximum 4 papers"),
    ],
)
def test_reference_system_prompts_preserve_caps_math_provenance_and_skepticism(
    system_prompt: str, cap_text: str
) -> None:
    assert cap_text in system_prompt
    assert "AI-GENERATED" in system_prompt
    assert "NEVER cite internal documents as authoritative" in system_prompt
    assert "Mathematical reasoning" in system_prompt
    assert "proof" in system_prompt.lower()
    assert "mechanism" in system_prompt.lower()
    assert "evidence" in system_prompt.lower()
    assert (
        "provenance" in system_prompt.lower()
        or "independently re-checking" in system_prompt
        or "verify information independently" in system_prompt
    )
    assert "direct" in system_prompt.lower()


def test_reference_builders_preserve_goal_two_step_context_persistence_and_rendered_caps() -> None:
    paper = _paper()
    pre = build_pre_brainstorm_expansion_prompt(
        ENGINEERING_GOAL, "Reduce fouling safely.", "[Not started]", [paper], 3
    )
    additional = build_additional_reference_expansion_prompt(
        ENGINEERING_GOAL,
        "Reduce fouling safely.",
        "A candidate pulsed-cleaning mechanism.",
        [paper],
        ["paper_001"],
        [{"paper_id": "paper_001", "title": "Already selected"}],
        3,
    )
    legacy = build_reference_expansion_prompt(
        ENGINEERING_GOAL, "Reduce fouling safely.", "Brainstorm summary", [paper], 6
    )
    final = build_reference_selection_prompt(
        ENGINEERING_GOAL,
        "Reduce fouling safely.",
        "Brainstorm summary",
        [paper],
        mode="initial",
        max_papers=3,
        retrieved_context="Retrieved evidence with source provenance.",
    )

    for prompt in (pre, additional, legacy, final):
        assert ENGINEERING_GOAL in prompt
        assert "Reduce fouling safely." in prompt
        assert "paper_007" in prompt
        assert "Prior Membrane Study [validator ratings]" in prompt
    assert "ENTIRE brainstorm exploration AND paper writing" in pre
    assert "abstracts and outlines" in pre
    assert "1 papers, 2 slots remaining" in additional
    assert "can add up to 2 more" in additional
    assert "Already selected" in additional
    assert "Titles, Abstracts, and Outlines" in legacy
    assert "MODE: INITIAL SELECTION" in final
    assert "FULL PAPER CONTENT" in final
    assert "RAG-RETRIEVED FULL-PAPER EVIDENCE" in final
    assert "up to 3 papers maximum" in final
    assert "brainstorm exploration AND paper writing" in final


def test_reference_schemas_preserve_exact_fields_and_caps() -> None:
    expansion = get_reference_expansion_json_schema()
    selection = get_reference_selection_json_schema(3)

    assert _schema_fields(expansion) == {
        "expand_papers",
        "proceed_without_references",
        "reasoning",
    }
    assert _schema_fields(selection) == {"selected_papers", "reasoning"}
    assert "maximum 3" in selection
    assert "up to 3 paper_ids" in selection


def test_title_exploration_preserves_goal_and_uses_five_domain_appropriate_candidates() -> None:
    prompt = build_title_exploration_user_prompt(
        MATH_GOAL,
        "Find the sharp extremal obstruction.",
        "A compression argument suggests a bound.",
        [_paper()],
        [_paper()],
    )

    assert MATH_GOAL in prompt
    assert "collect 5 validated candidate titles" in prompt
    assert "solution-oriented research paper or report" in prompt
    assert "mathematical title forms when the work is mathematical" in prompt
    assert "theorems, and proofs remain first-class" in prompt
    assert "does not imply" in prompt.lower()
    assert "completed experiments" in prompt
    assert "EXISTING RELATED PAPERS (do not duplicate" in prompt


@pytest.mark.parametrize(
    "system_prompt",
    [
        get_paper_title_system_prompt(),
        get_paper_title_validator_system_prompt(),
    ],
)
def test_title_system_prompts_are_domain_general_math_aware_and_anti_fabrication(
    system_prompt: str,
) -> None:
    assert "paper" in system_prompt.lower()
    assert "domain" in system_prompt.lower()
    assert "Mathematical reasoning" in system_prompt
    assert "first-class whenever relevant" in system_prompt
    assert "AI-GENERATED" in system_prompt
    assert "provenance" in system_prompt
    assert any(
        phrase in system_prompt.lower()
        for phrase in (
            "proposed experiments",
            "completed empirical demonstration",
            "implemented, tested, measured, or demonstrated",
        )
    )


def test_title_builders_preserve_mixed_goal_candidates_feedback_and_redundancy_boundary() -> None:
    existing = [{"title": "Completed Reliability Bound", "abstract": "A finished prior result."}]
    selection = build_paper_title_prompt(
        MIXED_GOAL,
        "Joint engineering and impossibility analysis.",
        "Source brainstorm: optimize topology and prove a lower bound.",
        existing,
        [_paper()],
        rejection_feedback="Do not imply the pilot was completed.",
        candidate_titles="1. Cost-Constrained Recovery with an Impossibility Frontier",
    )
    validation = build_paper_title_validation_prompt(
        MIXED_GOAL,
        "Joint engineering and impossibility analysis.",
        "Source brainstorm: optimize topology and prove a lower bound.",
        existing,
        "A Proposed Microgrid Design and Formal Outage Bound",
        "Reflects the source brainstorm without duplicating the completed paper.",
        [_paper()],
    )

    for prompt in (selection, validation):
        assert MIXED_GOAL in prompt
        assert "Source brainstorm: optimize topology and prove a lower bound." in prompt
        assert "Completed Reliability Bound" in prompt
        assert "SELECTED REFERENCE PAPERS" in prompt
    assert "PRE-VALIDATED CANDIDATE TITLES" in selection
    assert "Do not imply the pilot was completed." in selection
    assert "EXISTING PAPERS FROM THIS BRAINSTORM (Differentiate from these)" in selection
    assert "PROPOSED TITLE" in validation
    validator = get_paper_title_validator_system_prompt()
    assert "DO NOT reject a title for being \"similar to brainstorm submissions\"" in validator
    assert "ONLY reject for similarity" in validator
    assert "EXISTING COMPLETED PAPERS" in validator


def test_title_schemas_preserve_exact_fields_and_enums() -> None:
    title = get_paper_title_json_schema()
    validator = get_paper_title_validator_json_schema()

    assert _schema_fields(title) == {"paper_title", "reasoning"}
    assert _schema_fields(validator) == {"decision", "reasoning"}
    assert '"accept" or "reject"' in validator


@pytest.mark.parametrize(
    "system_prompt",
    [
        get_continuation_decision_system_prompt(),
        get_continuation_validator_system_prompt(),
    ],
)
def test_continuation_system_prompts_support_distinct_answer_bearing_contributions(
    system_prompt: str,
) -> None:
    assert "direct answer" in system_prompt
    assert "domain- and claim-appropriate rigor" in system_prompt
    assert "proof" in system_prompt.lower()
    assert "mechanism" in system_prompt.lower()
    assert "evidence" in system_prompt.lower()
    assert "AI-GENERATED" in system_prompt
    assert "provenance" in system_prompt or "without pretending" in system_prompt


def test_continuation_builders_preserve_goal_phase_context_decision_and_three_paper_cap() -> None:
    papers = [
        {
            "title": "Protocol Core",
            "abstract": "Safety mechanism and assumptions.",
            "outline": "I. Model\nII. Protocol",
        }
    ]
    decision = build_continuation_decision_prompt(
        SOFTWARE_GOAL,
        "Partition-tolerant protocol.",
        "Submissions include a distinct recovery design.",
        papers,
        2,
        "Previous feedback: distinguish the recovery contribution.",
    )
    validation = build_continuation_validation_prompt(
        SOFTWARE_GOAL,
        "Partition-tolerant protocol.",
        "Submissions include a distinct recovery design.",
        papers,
        2,
        {
            "decision": "write_another_paper",
            "reasoning": "Recovery is a distinct answer-bearing contribution.",
        },
    )

    for prompt in (decision, validation):
        assert SOFTWARE_GOAL in prompt
        assert "Partition-tolerant protocol." in prompt
        assert "Submissions include a distinct recovery design." in prompt
        assert "Protocol Core" in prompt
        assert "2 of 3 maximum" in prompt
    assert "Previous feedback: distinguish the recovery contribution." in decision
    assert "PROPOSED CONTINUATION DECISION" in validation
    assert "write_another_paper" in validation


def test_continuation_schemas_preserve_exact_fields_and_enums() -> None:
    decision = get_continuation_decision_json_schema()
    validator = get_continuation_validator_json_schema()

    assert _schema_fields(decision) == {"decision", "reasoning"}
    assert '"write_another_paper" or "move_on"' in decision
    assert _schema_fields(validator) == {"decision", "reasoning"}
    assert '"accept" or "reject"' in validator
