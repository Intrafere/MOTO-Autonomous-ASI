from types import SimpleNamespace

import pytest

from backend.aggregator.prompts.submitter_prompts import (
    build_submitter_prompt,
    get_submitter_system_prompt,
)
from backend.aggregator.prompts.validator_prompts import get_validator_system_prompt
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.prompts.completion_prompts import (
    build_completion_review_prompt,
    get_completion_review_system_prompt,
)
from backend.autonomous.prompts.final_answer_prompts import (
    build_certainty_assessment_prompt,
    get_certainty_assessment_system_prompt,
    get_format_selection_system_prompt,
    get_volume_organization_system_prompt,
)
from backend.autonomous.prompts.paper_redundancy_prompts import (
    build_paper_redundancy_prompt,
)
from backend.autonomous.prompts.paper_title_prompts import build_paper_title_prompt
from backend.autonomous.prompts.proof_prompts import PROOF_FRAMING_CONTEXT
from backend.autonomous.prompts.topic_exploration_prompts import (
    build_exploration_user_prompt,
)
from backend.autonomous.prompts.topic_prompts import build_topic_selection_prompt
from backend.compiler.prompts.construction_prompts import (
    get_body_construction_system_prompt,
)
from backend.compiler.prompts.outline_prompts import (
    build_outline_create_prompt,
    get_outline_create_system_prompt,
)
from backend.compiler.validation.compiler_validator import CompilerValidator
from backend.shared.api_client_manager import APIClientManager
from backend.shared.models import BrainstormRetroactiveOperation


OBJECTIVES = (
    "Prove a new additive-combinatorics bound and characterize sharpness.",
    "Design safer low-energy desalination under fouling and lifecycle constraints.",
    "Create a resilient distributed protocol for Byzantine faults and rolling upgrades.",
    "Propose a catalyst hypothesis with discriminating controls and a test plan.",
    "Optimize a physical design subject to a formally provable safety bound.",
    "Determine whether physical and budget constraints make the exact objective impossible.",
)

PAPERS = [
    {
        "paper_id": "paper_001",
        "title": "Candidate mechanism",
        "abstract": "A proposed mechanism with simulation evidence and unperformed tests.",
        "outline": "Mechanism; constraints; validation",
        "word_count": 1200,
    }
]


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", OBJECTIVES)
async def test_exact_objective_survives_representative_active_builders(
    objective: str,
) -> None:
    prompts = [
        build_submitter_prompt(objective, "Current accepted evidence."),
        build_exploration_user_prompt(objective, [], []),
        build_topic_selection_prompt(objective, [], [], ["Candidate route"]),
        build_completion_review_prompt(
            objective,
            "Current topic",
            "Accepted contribution",
            7,
            "",
        ),
        build_paper_title_prompt(
            objective,
            "Current topic",
            "Accepted contribution",
            [],
            [],
            ["Candidate title"],
        ),
        await build_outline_create_prompt(objective, "Optional source evidence."),
        build_certainty_assessment_prompt(objective, PAPERS),
        build_paper_redundancy_prompt(objective, PAPERS),
    ]

    for prompt in prompts:
        assert objective in prompt


def test_cross_build_policy_is_domain_general_and_proof_positive() -> None:
    ordinary_policy = "\n".join(
        (
            get_submitter_system_prompt(),
            get_validator_system_prompt(),
            get_completion_review_system_prompt(),
            get_outline_create_system_prompt(),
            get_body_construction_system_prompt(),
            CompilerValidator._get_paper_validation_system_prompt(
                object.__new__(CompilerValidator),
                "construction",
            ),
            get_certainty_assessment_system_prompt(),
            get_format_selection_system_prompt(),
            get_volume_organization_system_prompt(),
        )
    ).lower()

    for required in (
        "exact objective",
        "novel",
        "domain",
        "claim",
        "mathemat",
        "proof",
        "evidence",
        "provenance",
    ):
        assert required in ordinary_policy
    assert "non-mathematical work must not be rejected" in ordinary_policy
    assert "when relevant" in ordinary_policy


def test_active_ordinary_paths_have_no_universal_mathematics_gate() -> None:
    validator = object.__new__(CompilerValidator)
    operations = (
        BrainstormRetroactiveOperation(
            action="delete",
            submission_number=1,
            reasoning="The claimed benchmark was never run.",
        ),
        BrainstormRetroactiveOperation(
            action="edit",
            submission_number=1,
            new_content="Treat the benchmark as a proposed validation plan.",
            reasoning="Restore the evidence status.",
        ),
        BrainstormRetroactiveOperation(
            action="add",
            new_content="Add a concrete fault-injection protocol and acceptance criteria.",
            reasoning="This closes an implementation-validation gap.",
        ),
    )
    ordinary_prompts = [
        get_submitter_system_prompt(),
        get_validator_system_prompt(),
        get_completion_review_system_prompt(),
        get_outline_create_system_prompt(),
        get_body_construction_system_prompt(),
        get_certainty_assessment_system_prompt(),
        get_format_selection_system_prompt(),
        get_volume_organization_system_prompt(),
        *[
            validator._build_brainstorm_validation_prompt(op, "Existing database")
            for op in operations
        ],
    ]
    forbidden = (
        "you are a mathematical submitter",
        "user's mathematical prompt",
        "all claims must be grounded in established mathematical principles",
        "must include a theorem",
        "must include a proof",
    )

    for prompt in ordinary_prompts:
        lowered = prompt.lower()
        for phrase in forbidden:
            assert phrase not in lowered

    retroactive = "\n".join(ordinary_prompts[-3:]).lower()
    assert "domain and claim types" in retroactive
    assert "mechanism, design, algorithm, experiment" in retroactive
    assert "protected lean 4 proofs" in retroactive


def test_empirical_status_is_not_promoted_across_builds() -> None:
    aggregator = get_submitter_system_prompt().lower()
    completion = get_completion_review_system_prompt().lower()
    compiler = get_body_construction_system_prompt().lower()
    certainty = get_certainty_assessment_system_prompt().lower()

    assert "proposed experiment" in aggregator
    assert "does not mean a proposed mechanism has been built" in completion
    assert "never invent citations, experiments" in compiler
    for status in ("proposals", "hypotheses", "required validation", "merely because it is proposed"):
        assert status in certainty


def test_assistant_memory_stays_optional_and_formal_context_is_targeted() -> None:
    engineering_prompt = (
        "USER PROMPT:\nDesign safer low-energy desalination.\n\n"
        "YOUR TASK:\nPropose a concrete mechanism and validation plan."
    )
    engineering = APIClientManager._build_assistant_target_snapshot(
        "aggregator_submitter_1",
        "agg_sub1_001",
        engineering_prompt,
    )
    formal = APIClientManager._build_assistant_target_snapshot(
        "autonomous_proof_formalizer",
        "proof_form_001",
        "USER PROMPT:\nProve the target bound.\n\nTARGET THEOREM:\nBoundedInvariant",
    )
    injected = APIClientManager._append_assistant_memory_block(
        engineering_prompt,
        "Lean-verified support record",
    ).lower()

    assert engineering.imports == []
    assert formal.imports == ["Mathlib"]
    assert "optional" in injected
    assert "cannot redirect or mathematically reinterpret" in injected
    assert "do not require mathematics or formal proof" in injected


@pytest.mark.asyncio
async def test_bounded_nonmathematical_papers_only_assembly_has_no_theorem_gate() -> None:
    objective = OBJECTIVES[1]
    chain = "\n".join(
        (
            build_exploration_user_prompt(objective, [], []),
            build_topic_selection_prompt(objective, [], [], ["Membrane route"]),
            build_submitter_prompt(objective, "A proposed cleaning mechanism."),
            build_completion_review_prompt(
                objective,
                "Membrane route",
                "A proposed cleaning mechanism.",
                7,
                "",
            ),
            build_paper_title_prompt(
                objective,
                "Membrane route",
                "A proposed cleaning mechanism.",
                [],
                [],
                ["Safer Low-Energy Desalination"],
            ),
            await build_outline_create_prompt(objective, "A proposed cleaning mechanism."),
            get_body_construction_system_prompt(),
        )
    ).lower()

    assert objective.lower() in chain
    assert "mechanism" in chain
    assert "verification" in chain
    assert "non-mathematical work must not be forced into mathematical form" in chain
    assert "must include a theorem" not in chain
    assert "must include a proof" not in chain


def test_mixed_and_mathematical_paths_keep_proof_boundaries_available() -> None:
    mixed = OBJECTIVES[4]
    coordinator = object.__new__(AutonomousCoordinator)
    coordinator._proof_framing_active = True
    coordinator._proof_framing_context = PROOF_FRAMING_CONTEXT
    framed = coordinator._append_proof_framing(
        build_submitter_prompt(mixed, "Engineering design evidence.")
    )
    rigor = CompilerValidator._get_mathematical_rigor_validation_system_prompt(
        object.__new__(CompilerValidator),
        "rigor",
    ).lower()

    assert mixed in framed
    assert "PROOF_FRAMING_CONTEXT" not in framed
    assert "directly help answer, support, or advance" in framed
    assert "mathematical rigor" in rigor
    assert "proof" in rigor


def test_manual_aggregator_compiler_contract_remains_domain_appropriate() -> None:
    objective = OBJECTIVES[2]
    aggregator = build_submitter_prompt(objective, "Existing protocol idea.")
    compiler = get_body_construction_system_prompt()
    validator = CompilerValidator._get_paper_validation_system_prompt(
        object.__new__(CompilerValidator),
        "construction",
    )

    assert objective in aggregator
    assert "algorithm" in aggregator.lower()
    assert "domain" in compiler.lower()
    assert "non-mathematical work must not be rejected" in validator.lower()
