from types import SimpleNamespace

import pytest
import importlib
from backend.autonomous.core.autonomous_coordinator import AutonomousCoordinator
from backend.autonomous.agents.final_answer.certainty_assessor import CertaintyAssessor
from backend.autonomous.agents.final_answer.volume_organizer import VolumeOrganizer
from backend.autonomous.prompts.final_answer_prompts import (
    build_certainty_assessment_prompt,
    build_certainty_validation_prompt,
    build_format_selection_prompt,
    build_volume_organization_prompt,
    get_certainty_assessment_json_schema,
    get_certainty_assessment_system_prompt,
    get_certainty_validator_json_schema,
    get_format_selection_json_schema,
    get_format_selection_system_prompt,
    get_format_validator_json_schema,
    get_volume_organization_json_schema,
    get_volume_organization_system_prompt,
    get_volume_validator_json_schema,
)
from backend.autonomous.prompts.paper_redundancy_prompts import (
    build_paper_redundancy_prompt,
)
from backend.shared.config import system_config
from backend.shared.models import VolumeChapter, VolumeOrganization


PAPERS = [
    {
        "paper_id": "paper_001",
        "title": "Mechanism",
        "abstract": "A proposed mechanism with simulation evidence.",
        "outline": "Mechanism; constraints; validation",
        "word_count": 1200,
    }
]


def test_redundancy_preserves_cross_domain_unique_value() -> None:
    prompt = build_paper_redundancy_prompt(
        "Design a safer low-energy desalination system.",
        PAPERS,
    ).lower()

    for concept in (
        "solution mechanism",
        "evidence",
        "implementation",
        "experimental proposal",
        "failure-mode",
        "theorem",
        "proof",
        "algorithm",
        "impossibility result",
    ):
        assert concept in prompt
    assert "at most one" in prompt
    assert "when in doubt, do not recommend removal" in prompt


def test_certainty_preserves_evidence_status_and_generalized_impossibility() -> None:
    prompt = get_certainty_assessment_system_prompt().lower()

    assert "proposals" in prompt
    assert "hypotheses" in prompt
    assert "required validation" in prompt
    assert "never treat an invention" in prompt
    for impossibility in (
        "mathematically impossible",
        "physically infeasible",
        "internally inconsistent",
        "prohibited by stated constraints",
    ):
        assert impossibility in prompt


def test_certainty_builder_handles_invention_without_claiming_demonstration() -> None:
    prompt = build_certainty_assessment_prompt(
        "Invent a compact atmospheric water harvester.",
        PAPERS,
    ).lower()

    assert "strongest defensible answer" in prompt
    assert "proposed mechanism with simulation evidence" in prompt
    assert "proposals" in prompt
    assert "demonstrated merely because it is proposed" in prompt


def test_certainty_expansion_supplements_complete_library_catalog() -> None:
    papers = PAPERS + [
        {
            "paper_id": "paper_002",
            "title": "Limitation",
            "abstract": "A contradiction and unvalidated assumption.",
            "outline": "Limitations",
            "word_count": 600,
        }
    ]
    prompt = build_certainty_assessment_prompt(
        "Assess the proposed system.",
        papers,
        expanded_papers=[
            {
                "paper_id": "paper_001",
                "title": "Mechanism",
                "content": "Full mechanism evidence",
            }
        ],
    )

    assert "paper_001" in prompt
    assert "paper_002" in prompt
    assert "A contradiction and unvalidated assumption." in prompt
    assert "Full mechanism evidence" in prompt


def test_certainty_validator_receives_evidence_not_only_titles() -> None:
    prompt = build_certainty_validation_prompt(
        "Assess the proposed system.",
        PAPERS,
        {
            "certainty_level": "partial_answer",
            "known_certainties_summary": "The mechanism is proposed.",
            "reasoning": "No prototype exists.",
        },
        expanded_papers=[
            {
                "paper_id": "paper_001",
                "title": "Mechanism",
                "content": "The theorem is proved, but the prototype is unbuilt.",
            }
        ],
    )

    assert "A proposed mechanism with simulation evidence." in prompt
    assert "Mechanism; constraints; validation" in prompt
    assert "The theorem is proved, but the prototype is unbuilt." in prompt


def test_certainty_expansion_recognizes_all_solution_modalities() -> None:
    assessor = CertaintyAssessor("submitter", "validator")
    prompt = assessor._build_expansion_request_prompt("Solve the objective.", PAPERS).lower()

    for concept in (
        "mechanism",
        "evidence",
        "implementation",
        "risk",
        "validation",
        "theorem",
        "proof",
    ):
        assert concept in prompt
    assert '"expand_papers"' in prompt
    assert '"proceed_without_expansion"' in prompt


def test_format_selection_is_domain_neutral_and_not_paper_count_driven() -> None:
    prompt = get_format_selection_system_prompt().lower()

    assert "genuinely independent solution components" in prompt
    assert "number of source papers alone does not justify a volume" in prompt
    assert "mathematical" in prompt
    assert "formal proof" in prompt


def test_volume_supports_cross_domain_dependency_structures_and_gaps() -> None:
    prompt = get_volume_organization_system_prompt().lower()
    schema = get_volume_organization_json_schema().lower()

    for concept in (
        "foundations → results → proofs",
        "constraints → mechanism → implementation → validation",
        "hypothesis → evidence → protocol → limitations",
        "design",
        "evaluation",
        "safety",
        "risk",
    ):
        assert concept in prompt
    assert "mixed engineering and formal analysis" in schema
    assert "mathematical bounds" in schema
    assert "without claiming unperformed tests" in schema


def test_existing_tier3_json_contracts_are_unchanged() -> None:
    certainty = get_certainty_assessment_json_schema()
    assert '"certainty_level"' in certainty
    assert '"known_certainties_summary"' in certainty
    assert '"reasoning"' in certainty
    for value in (
        "total_answer",
        "partial_answer",
        "no_answer_known",
        "appears_impossible",
        "other",
    ):
        assert value in certainty

    fmt = get_format_selection_json_schema()
    assert '"answer_format"' in fmt
    assert "short_form | long_form" in fmt
    assert '"decision"' in get_format_validator_json_schema()
    assert '"decision"' in get_certainty_validator_json_schema()

    volume = get_volume_organization_json_schema()
    for field in (
        '"volume_title"',
        '"chapters"',
        '"chapter_type"',
        '"paper_id"',
        '"title"',
        '"order"',
        '"description"',
        '"outline_complete"',
        '"reasoning"',
    ):
        assert field in volume
    for value in ("existing_paper", "gap_paper", "introduction", "conclusion"):
        assert value in volume
    assert '"decision"' in get_volume_validator_json_schema()


def test_active_builders_keep_schema_before_untrusted_context() -> None:
    certainty = build_certainty_assessment_prompt("UNTRUSTED_GOAL", PAPERS)
    fmt = build_format_selection_prompt(
        "UNTRUSTED_GOAL",
        PAPERS,
        {"certainty_level": "partial_answer", "known_certainties_summary": "supported"},
    )
    volume = build_volume_organization_prompt(
        "UNTRUSTED_GOAL",
        PAPERS,
        {"certainty_level": "partial_answer", "known_certainties_summary": "supported"},
    )

    for prompt in (certainty, fmt, volume):
        assert prompt.index("REQUIRED JSON FORMAT") < prompt.index("UNTRUSTED_GOAL")


@pytest.mark.parametrize(
    ("title", "findings", "context", "expected"),
    [
        (
            "A Testable Atmospheric Water Harvester",
            "Simulation supports the mechanism; prototype performance is unvalidated.",
            "",
            "prototype performance is unvalidated",
        ),
        (
            "A New Extremal Bound",
            "The theorem and Lean-verified proof establish the requested bound.",
            "",
            "lean-verified proof",
        ),
        (
            "A Safe Controller with Formal Limits",
            "Engineering evidence supports the controller; a theorem bounds instability.",
            "Chapter type: gap_paper. Close the implementation and safety-validation gap.",
            "close the implementation and safety-validation gap",
        ),
    ],
)
def test_tier3_compiler_prompt_assembly_is_domain_general_and_proof_aware(
    monkeypatch,
    title,
    findings,
    context,
    expected,
) -> None:
    coordinator = AutonomousCoordinator()
    seen = {}

    def apply_proof_context(prompt: str) -> str:
        seen["prompt"] = prompt
        return f"{prompt}\nPROOF_CONTEXT_APPLIED"

    monkeypatch.setattr(coordinator, "_apply_proof_context", apply_proof_context)
    prompt = coordinator._build_tier3_compiler_prompt(title, findings, context)
    lowered = prompt.lower()

    assert "write a mathematical research paper" not in lowered
    assert "strongest credible and genuinely novel solution" in lowered
    assert "domain" in lowered
    assert "mathematical reasoning" in lowered
    assert "formal proof" in lowered
    assert expected in lowered
    if context:
        assert "directly advances the final answer" in lowered
        assert "directly answers the research question" not in lowered
    else:
        assert "directly answers the research question" in lowered
    assert prompt.endswith("PROOF_CONTEXT_APPLIED")
    assert seen["prompt"] in prompt


@pytest.mark.asyncio
async def test_tier3_reference_selection_passes_six_paper_cap(monkeypatch) -> None:
    coordinator = AutonomousCoordinator()
    seen = {}

    async def select_references(**kwargs):
        seen.update(kwargs)
        return ["paper_001"]

    monkeypatch.setattr(
        coordinator,
        "_get_effective_user_research_prompt",
        lambda: "Solve it.",
    )
    coordinator._reference_selector = SimpleNamespace(
        select_references=select_references
    )

    selected = await coordinator._tier3_reference_selection(PAPERS)

    assert selected == ["paper_001"]
    assert seen["max_total_papers"] == 6
    assert seen["available_papers"] == PAPERS
    assert seen["already_selected"] == []


@pytest.mark.asyncio
async def test_tier3_title_selection_reuses_exploration_candidates(monkeypatch) -> None:
    coordinator = AutonomousCoordinator()
    assessment = SimpleNamespace(
        certainty_level="partial_answer",
        known_certainties_summary="The mechanism is supported; validation remains.",
    )
    seen = {}

    monkeypatch.setattr(
        coordinator,
        "_get_reference_paper_details",
        lambda _ids: _async_result(PAPERS),
    )
    monkeypatch.setattr(
        coordinator,
        "_paper_title_exploration_phase",
        lambda **_kwargs: _async_result(
            "\n".join(f"Candidate {index}" for index in range(1, 6))
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_get_effective_user_research_prompt",
        lambda: "Solve it.",
    )

    async def select_title(**kwargs):
        seen.update(kwargs)
        return "Validated Final Title"

    coordinator._title_selector = SimpleNamespace(select_title=select_title)

    title = await coordinator._tier3_title_selection(assessment, ["paper_001"])

    assert title == "Validated Final Title"
    assert seen["candidate_titles"].count("Candidate") == 5
    assert seen["reference_papers"] == PAPERS
    assert "Defensible Findings and Evidence Status" in seen["brainstorm_summary"]


@pytest.mark.asyncio
async def test_volume_chapter_forwards_role_scope_and_all_existing_references(
    monkeypatch,
) -> None:
    coordinator = AutonomousCoordinator()
    chapter = VolumeChapter(
        chapter_type="gap_paper",
        title="Validation Gap",
        order=3,
        description="Build and validate the fail-safe controller.",
    )
    volume = VolumeOrganization(
        volume_title="Mixed solution",
        chapters=[
            VolumeChapter(
                chapter_type="existing_paper",
                paper_id="paper_001",
                title="Mechanism",
                order=2,
            ),
            chapter,
        ],
    )
    assessment = SimpleNamespace(
        known_certainties_summary="The mechanism is supported; validation remains."
    )
    seen = {}

    monkeypatch.setattr(
        coordinator,
        "_get_reference_paper_details",
        lambda _ids: _async_result(PAPERS),
    )
    monkeypatch.setattr(
        coordinator,
        "_paper_title_exploration_phase",
        lambda **_kwargs: _async_result("Candidate One"),
    )
    coordinator._title_selector = SimpleNamespace(
        select_title=lambda **_kwargs: _async_result("Validated Gap Chapter")
    )

    async def compile_paper(**kwargs):
        seen.update(kwargs)
        return "Abstract\nComplete chapter"

    monkeypatch.setattr(coordinator, "_compile_tier3_paper", compile_paper)
    coordinator_module = importlib.import_module(
        "backend.autonomous.core.autonomous_coordinator"
    )
    monkeypatch.setattr(
        coordinator_module.outline_memory,
        "get_outline",
        lambda: _async_result("Outline"),
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "save_chapter_paper",
        lambda **_kwargs: _async_result(None),
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "get_chapter_paper",
        lambda _order: _async_result(""),
    )

    assert await coordinator._write_volume_chapter(chapter, volume, assessment)
    assert seen["reference_paper_ids"] == ["paper_001"]
    assert "gap_paper" in seen["writing_context"]
    assert "Build and validate the fail-safe controller." in seen["writing_context"]
    assert system_config.autonomous_tier3_short_form_max_reference_papers == 6


@pytest.mark.asyncio
async def test_volume_chapter_requires_validated_final_title(monkeypatch) -> None:
    coordinator = AutonomousCoordinator()
    chapter = VolumeChapter(
        chapter_type="gap_paper",
        title="Organizer Draft",
        order=2,
        description="Close the validation gap.",
    )
    volume = VolumeOrganization(volume_title="Volume", chapters=[chapter])
    assessment = SimpleNamespace(known_certainties_summary="Supported findings")
    compiled = False

    monkeypatch.setattr(
        coordinator,
        "_get_reference_paper_details",
        lambda _ids: _async_result([]),
    )
    monkeypatch.setattr(
        coordinator,
        "_paper_title_exploration_phase",
        lambda **_kwargs: _async_result("Five candidate titles"),
    )
    coordinator._title_selector = SimpleNamespace(
        select_title=lambda **_kwargs: _async_result(None)
    )

    async def compile_paper(**_kwargs):
        nonlocal compiled
        compiled = True
        return "Should not run"

    monkeypatch.setattr(coordinator, "_compile_tier3_paper", compile_paper)

    assert not await coordinator._write_volume_chapter(chapter, volume, assessment)
    assert not compiled


@pytest.mark.asyncio
async def test_generated_gap_content_reaches_later_conclusion(monkeypatch) -> None:
    coordinator = AutonomousCoordinator()
    gap = VolumeChapter(
        chapter_type="gap_paper",
        title="Generated Gap",
        order=2,
        description="Close the implementation gap.",
    )
    conclusion = VolumeChapter(
        chapter_type="conclusion",
        title="Conclusion",
        order=3,
    )
    volume = VolumeOrganization(
        volume_title="Volume",
        chapters=[gap, conclusion],
    )
    assessment = SimpleNamespace(known_certainties_summary="Supported findings")
    seen = {}

    monkeypatch.setattr(
        coordinator,
        "_get_reference_paper_details",
        lambda _ids: _async_result([]),
    )
    monkeypatch.setattr(
        coordinator,
        "_paper_title_exploration_phase",
        lambda **_kwargs: _async_result("Five candidate titles"),
    )
    coordinator._title_selector = SimpleNamespace(
        select_title=lambda **_kwargs: _async_result("Validated Conclusion")
    )
    coordinator_module = importlib.import_module(
        "backend.autonomous.core.autonomous_coordinator"
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "get_chapter_paper",
        lambda order: _async_result(
            "Generated implementation and validation evidence" if order == 2 else ""
        ),
    )
    monkeypatch.setattr(
        coordinator_module.outline_memory,
        "get_outline",
        lambda: _async_result("Outline"),
    )
    monkeypatch.setattr(
        coordinator_module.final_answer_memory,
        "save_chapter_paper",
        lambda **_kwargs: _async_result(None),
    )

    async def compile_paper(**kwargs):
        seen.update(kwargs)
        return "Abstract\nConclusion"

    monkeypatch.setattr(coordinator, "_compile_tier3_paper", compile_paper)

    assert await coordinator._write_volume_chapter(conclusion, volume, assessment)
    assert "Generated implementation and validation evidence" in seen[
        "generated_chapter_context"
    ]


async def _async_result(value):
    return value


def test_volume_writing_order_remains_gap_conclusion_introduction() -> None:
    volume = VolumeOrganization(
        volume_title="Mixed solution",
        chapters=[
            VolumeChapter(
                chapter_type="introduction",
                title="Introduction",
                order=1,
            ),
            VolumeChapter(
                chapter_type="gap_paper",
                title="Validation",
                order=3,
            ),
            VolumeChapter(
                chapter_type="existing_paper",
                paper_id="paper_001",
                title="Mechanism",
                order=2,
            ),
            VolumeChapter(
                chapter_type="conclusion",
                title="Conclusion",
                order=4,
            ),
        ],
    )

    order = VolumeOrganizer("submitter", "validator").get_writing_order(volume)
    assert [chapter.chapter_type for chapter in order] == [
        "gap_paper",
        "conclusion",
        "introduction",
    ]
