from pathlib import Path
import json
from unittest.mock import AsyncMock

import pytest

from backend.compiler.memory.compiler_rejection_log import CompilerRejectionLog
from backend.compiler.memory.paper_memory import (
    PAPER_ANCHOR,
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
)
from backend.compiler.prompts.construction_prompts import (
    get_abstract_construction_system_prompt,
    get_body_construction_system_prompt,
    get_conclusion_construction_system_prompt,
    get_construction_json_schema,
    get_construction_system_prompt,
    get_introduction_construction_system_prompt,
    get_wolfram_tool_guidance,
)
from backend.compiler.prompts.critique_prompts import (
    get_critique_json_schema,
    get_critique_submitter_system_prompt,
    get_critique_validation_json_schema,
    get_critique_validator_system_prompt,
)
from backend.compiler.prompts.outline_prompts import (
    build_outline_create_prompt,
    build_outline_update_prompt,
    get_outline_create_system_prompt,
    get_outline_json_schema,
    get_outline_update_system_prompt,
)
from backend.compiler.prompts.review_prompts import (
    EMPIRICAL_RED_TEAM_REVIEW_FOCUS,
    get_review_json_schema,
    get_review_system_prompt,
)
from backend.compiler.validation.compiler_validator import CompilerValidator
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.compiler.agents.writer_submitter import (
    WOLFRAM_MAX_CALLS_PER_SUBMISSION,
    WritingSubmitter,
)
from backend.shared.critique_prompts import (
    CRITIQUE_JSON_SCHEMA,
    build_critique_prompt as build_shared_critique_prompt,
    get_default_critique_prompt,
)
from backend.shared.models import CompilerValidationResult, ContextPack


def _joined(*parts: str) -> str:
    return "\n".join(parts).lower()


def _make_writer(monkeypatch) -> WritingSubmitter:
    from backend.compiler.agents import writer_submitter

    monkeypatch.setattr(
        writer_submitter.system_config,
        "compiler_writer_context_window",
        32768,
    )
    monkeypatch.setattr(
        writer_submitter.system_config,
        "compiler_writer_max_output_tokens",
        4096,
    )
    return WritingSubmitter("test-model", "Solve it.")


def test_outline_contract_supports_domain_specific_solution_structures() -> None:
    prompt = _joined(
        get_outline_create_system_prompt(),
        get_outline_update_system_prompt(),
    )

    for concept in (
        "algorithm",
        "architecture",
        "engineering",
        "failure modes",
        "empirical",
        "falsifiable",
        "strateg",
        "evaluation",
    ):
        assert concept in prompt
    assert "direct solution" in prompt or "direct answer" in prompt
    assert "non-mathematical work must not be forced into mathematical form" in prompt


def test_all_construction_phase_prompts_allow_empty_new_string_only_for_delete() -> None:
    for prompt in (
        get_construction_system_prompt(),
        get_body_construction_system_prompt(),
        get_conclusion_construction_system_prompt(),
        get_introduction_construction_system_prompt(),
        get_abstract_construction_system_prompt(),
        get_construction_json_schema(),
    ):
        lowered = prompt.lower()
        if "new_string" in lowered and "must" in lowered and "empty" in lowered:
            assert "delete" in lowered


@pytest.mark.parametrize(
    ("outline", "expected_error"),
    [
        (
            "I. Abstract\nII. Introduction\nIII. Method\nIV. Conclusion",
            "INCORRECT_SECTION_NAME - Abstract",
        ),
        (
            "I. Introduction\nAbstract\nII. Method\nIII. Conclusion",
            "INCORRECT_SECTION_ORDER - Abstract",
        ),
        (
            "I. Introduction\nII. Conclusion",
            "MISSING_REQUIRED_SECTION - Body",
        ),
    ],
)
def test_outline_prevalidation_enforces_abstract_and_body_structure(
    outline: str,
    expected_error: str,
) -> None:
    coordinator = object.__new__(CompilerCoordinator)
    assert expected_error in coordinator._pre_validate_outline_structure(outline)


def test_outline_contract_retains_mathematical_progression_conditionally() -> None:
    prompt = _joined(
        get_outline_create_system_prompt(),
        get_outline_update_system_prompt(),
    )

    for concept in ("definitions", "theorems", "proofs"):
        assert concept in prompt
    assert "mathemat" in prompt
    assert "when relevant" in prompt or "when useful" in prompt or "as appropriate" in prompt


def test_outline_contract_allows_only_unnumbered_optional_abstract() -> None:
    prompt = _joined(
        get_outline_create_system_prompt(),
        get_outline_update_system_prompt(),
        get_outline_json_schema(),
    )

    assert 'unnumbered heading "abstract"' in prompt
    assert "i. introduction" in prompt
    assert "i. abstract" not in prompt
    assert "0. abstract" not in prompt


@pytest.mark.asyncio
async def test_active_outline_builders_preserve_abstract_and_update_contracts(
    monkeypatch,
) -> None:
    from backend.compiler.prompts import outline_prompts

    monkeypatch.setattr(
        outline_prompts.compiler_rejection_log,
        "get_rejections_text",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "backend.compiler.memory.outline_memory.outline_memory.get_creation_feedback",
        AsyncMock(return_value=""),
    )

    create_prompt = await build_outline_create_prompt("Solve it.", "Evidence")
    update_prompt = await build_outline_update_prompt(
        "Solve it.",
        "I. Introduction\nII. Body\nIII. Conclusion",
        "Draft",
    )

    assert 'unnumbered heading "Abstract"' in create_prompt
    assert "I. Introduction" in create_prompt
    assert "I. Abstract" not in create_prompt
    assert "0. Abstract" not in create_prompt
    assert '"operation": "insert_after"' in update_prompt
    assert "insert_after | replace" not in update_prompt


@pytest.mark.asyncio
async def test_submit_outline_update_rejects_affirmative_non_insert_operation(
    monkeypatch,
) -> None:
    from backend.compiler.agents import writer_submitter

    submitter = _make_writer(monkeypatch)
    monkeypatch.setattr(writer_submitter.outline_memory, "get_outline", AsyncMock(return_value="I. Introduction"))
    monkeypatch.setattr(writer_submitter.paper_memory, "get_paper", AsyncMock(return_value="Draft"))
    monkeypatch.setattr(
        writer_submitter.compiler_rag_manager,
        "retrieve_for_mode",
        AsyncMock(return_value=ContextPack(text="")),
    )
    monkeypatch.setattr(writer_submitter, "build_outline_update_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(writer_submitter, "count_tokens", lambda _text: 1)
    monkeypatch.setattr(
        writer_submitter.api_client_manager,
        "prewarm_assistant_memory_context",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        writer_submitter.api_client_manager,
        "generate_completion",
        AsyncMock(
            return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "needs_update": True,
                            "operation": "replace",
                            "old_string": "I. Introduction",
                            "new_string": "I. Revised Introduction",
                            "reasoning": "Revise.",
                        })
                    }
                }]
            }
        ),
    )

    with pytest.raises(ValueError, match="operation='insert_after'"):
        await submitter.submit_outline_update()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "old_string", "new_string", "expected_content", "accepted"),
    [
        ("delete", "obsolete paragraph", "", "obsolete paragraph", True),
        ("delete", "", "", None, False),
        ("replace", "old", "new", "new", True),
        ("insert_after", "anchor", "addition", "addition", True),
        ("full_content", "", "complete draft", "complete draft", True),
        ("replace", "old", "", None, False),
        ("insert_after", "anchor", "", None, False),
        ("full_content", "", "", None, False),
    ],
)
async def test_construction_parser_validates_fields_by_operation(
    monkeypatch,
    operation,
    old_string,
    new_string,
    expected_content,
    accepted,
) -> None:
    from backend.compiler.agents import writer_submitter

    submitter = _make_writer(monkeypatch)
    payload = {
        "needs_construction": True,
        "operation": operation,
        "old_string": old_string,
        "new_string": new_string,
        "reasoning": "Edit the draft.",
        "section_complete": False,
    }
    monkeypatch.setattr(writer_submitter.outline_memory, "get_outline", AsyncMock(return_value="I. Introduction"))
    monkeypatch.setattr(writer_submitter.paper_memory, "get_paper", AsyncMock(return_value="Draft"))
    monkeypatch.setattr(
        writer_submitter.compiler_rag_manager,
        "retrieve_for_mode",
        AsyncMock(return_value=ContextPack(text="")),
    )
    monkeypatch.setattr(writer_submitter, "build_construction_prompt", AsyncMock(return_value="prompt"))
    monkeypatch.setattr(writer_submitter, "count_tokens", lambda _text: 1)
    monkeypatch.setattr(
        writer_submitter.api_client_manager,
        "prewarm_assistant_memory_context",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        submitter,
        "_generate_completion_with_wolfram_tool",
        AsyncMock(return_value=(json.dumps(payload), [], {})),
    )

    submission = await submitter.submit_construction()

    assert (submission is not None) is accepted
    if submission is not None:
        assert submission.operation == operation
        assert submission.old_string == old_string
        assert submission.new_string == new_string
        assert submission.content == expected_content


def test_body_construction_supports_all_purpose_solution_content() -> None:
    prompt = get_body_construction_system_prompt().lower()

    for concept in (
        "algorithm",
        "engineering",
        "empirical",
        "strateg",
        "constraints",
        "failure modes",
    ):
        assert concept in prompt
    assert "independently checkable" in prompt
    assert "section_complete" in prompt


def test_construction_retains_mathematical_rigor_and_provenance_guards() -> None:
    prompt = _joined(
        get_body_construction_system_prompt(),
        get_construction_system_prompt(),
    )

    assert "mathematical claims" in prompt
    assert "proof" in prompt
    assert "explicit assumptions" in prompt
    for prohibited in (
        "invent citations",
        "experiments",
        "benchmark numbers",
        "code artifacts",
    ):
        assert prohibited in prompt


def test_review_checks_direct_value_and_domain_specific_failure_modes() -> None:
    prompt = get_review_system_prompt().lower()

    assert "direct" in prompt
    assert "user's" in prompt and "prompt" in prompt
    assert "failure modes" in prompt
    assert "interfaces" in prompt or "implementation" in prompt
    assert "mathematical" in prompt and ("proof gap" in prompt or "proof gaps" in prompt)
    assert "no edit" in prompt or "needs_edit" in prompt


def test_empirical_red_team_remains_conservative_and_non_fabricating() -> None:
    prompt = EMPIRICAL_RED_TEAM_REVIEW_FOCUS.lower()

    for concept in (
        "fabricated experiments",
        "nonexistent code",
        "unsupported benchmark",
        "uncited external",
        "proposed experiment",
    ):
        assert concept in prompt
    assert "do not preserve unsupported benchmark numbers" in prompt


def test_compiler_self_review_accepts_engineering_and_algorithmic_limitations() -> None:
    submitter = get_critique_submitter_system_prompt().lower()
    validator = get_critique_validator_system_prompt().lower()

    for concept in ("engineering", "algorithm", "implementation"):
        assert concept in submitter or concept in validator
    assert "one important point per turn" in submitter
    assert "do not propose direct edits or rewrites" in submitter
    assert "critique_needed=false" in submitter
    assert "non-redundant" in validator


def test_ordinary_validator_uses_domain_rigor_and_keeps_theorem_guard() -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Solve it.")
    outline = validator._get_outline_validation_system_prompt("outline_create").lower()
    construction = validator._get_paper_validation_system_prompt("construction").lower()

    for concept in (
        "domain-appropriate",
        "direct solution value",
        "claim-type provenance",
        "specificity and actionability",
        "novelty without fabrication",
    ):
        assert concept in outline or concept in construction
    assert "does not require mathematics" in construction or "mathematics is irrelevant" in construction
    assert "unsupported theorem claims" in construction
    assert "rigor_check" in construction


def test_prompt_schemas_preserve_existing_field_contracts() -> None:
    outline_schema = get_outline_json_schema()
    construction_schema = get_construction_json_schema()
    review_schema = get_review_json_schema()
    critique_schema = get_critique_json_schema()
    critique_validation_schema = get_critique_validation_json_schema()

    for field in ("content", "outline_complete", "reasoning"):
        assert field in outline_schema
    for field in (
        "needs_construction",
        "operation",
        "old_string",
        "new_string",
        "reasoning",
        "section_complete",
    ):
        assert field in construction_schema
    for field in ("needs_edit", "operation", "old_string", "new_string", "reasoning"):
        assert field in review_schema
    for field in ("critique_needed", "submission", "reasoning"):
        assert field in critique_schema
    for field in ("decision", "reasoning", "summary"):
        assert field in critique_validation_schema


def test_exact_edit_and_protected_markers_remain_visible() -> None:
    prompts = (
        get_body_construction_system_prompt(),
        get_conclusion_construction_system_prompt(),
        get_introduction_construction_system_prompt(),
        get_abstract_construction_system_prompt(),
        get_construction_system_prompt(),
        get_review_system_prompt(),
    )

    assert THEOREMS_APPENDIX_START == (
        "[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]"
    )
    assert THEOREMS_APPENDIX_END == (
        "[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]"
    )
    assert PAPER_ANCHOR == (
        "[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]"
    )
    for prompt in prompts:
        assert "old_string" in prompt
        assert "new_string" in prompt
    marker_prompts = "\n".join(prompts)
    assert PAPER_ANCHOR in marker_prompts
    assert THEOREMS_APPENDIX_START in marker_prompts
    assert THEOREMS_APPENDIX_END in marker_prompts


def test_ordinary_prompt_identities_are_not_universally_mathematical() -> None:
    prompts = (
        get_outline_create_system_prompt(),
        get_outline_update_system_prompt(),
        get_body_construction_system_prompt(),
        get_conclusion_construction_system_prompt(),
        get_introduction_construction_system_prompt(),
        get_abstract_construction_system_prompt(),
        get_construction_system_prompt(),
        get_review_system_prompt(),
        get_critique_submitter_system_prompt(),
        get_critique_validator_system_prompt(),
    )

    for prompt in prompts:
        assert "mathematical document" not in prompt.lower()


def test_shared_default_critique_is_domain_general_and_schema_stable() -> None:
    prompt = get_default_critique_prompt().lower()

    for standard in ("factual", "logical", "mathematical", "technical", "methodological"):
        assert standard in prompt
    for field in (
        "novelty_rating",
        "novelty_feedback",
        "correctness_rating",
        "correctness_feedback",
        "impact_rating",
        "impact_feedback",
        "full_critique",
    ):
        assert field in CRITIQUE_JSON_SCHEMA


def test_shared_custom_critique_prompt_remains_authoritative() -> None:
    custom = "CUSTOM REVIEW AUTHORITY: assess only the supplied deployment criteria."
    built = build_shared_critique_prompt(
        paper_content="Paper body.",
        paper_title="Deployment report",
        custom_prompt=custom,
    )

    assert custom in built
    assert get_default_critique_prompt() not in built
    assert CRITIQUE_JSON_SCHEMA in built
    assert "PAPER TITLE: Deployment report" in built
    assert "Paper body." in built


def test_wolfram_guidance_preserves_tool_and_budget_contract(monkeypatch) -> None:
    from backend.compiler.prompts import construction_prompts
    from backend.shared import wolfram_alpha_client

    monkeypatch.setattr(construction_prompts.system_config, "wolfram_alpha_enabled", True)
    monkeypatch.setattr(construction_prompts.system_config, "wolfram_alpha_api_key", "test-key")
    monkeypatch.setattr(wolfram_alpha_client, "get_wolfram_client", lambda: object())

    guidance = get_wolfram_tool_guidance()

    assert "CONSTRUCTION MODE ONLY" in guidance
    assert "wolfram_alpha_query" in guidance
    assert "up to 20 Wolfram Alpha calls" in guidance
    assert "Lean 4 proof verification" in guidance
    assert WOLFRAM_MAX_CALLS_PER_SUBMISSION == 20


@pytest.mark.asyncio
async def test_rejection_repair_guidance_is_domain_appropriate(tmp_path: Path) -> None:
    log = CompilerRejectionLog()
    log.rejections_file = tmp_path / "rejections.txt"
    log.acceptances_file = tmp_path / "acceptances.txt"
    log.declines_file = tmp_path / "declines.txt"
    await log.initialize()

    await log.add_rejection(
        CompilerValidationResult(
            submission_id="domain-rigor",
            decision="reject",
            reasoning="The mechanism has no feasibility support.",
            summary="Correctness standard failed.",
            coherence_check=True,
            rigor_check=False,
            placement_check=True,
        ),
        mode="construction",
        submission_content="Proposed mechanism.",
    )
    text = await log.get_rejections_text()

    assert "appropriate to their domain and claim type" in text
    assert "Mathematical claims require sound derivation, proof, or explicit assumptions" in text
    assert "factual, logical, technical, empirical, and methodological claims" in text
    assert "SUBMISSION PREVIEW" in text
    assert "VALIDATOR REASONING" in text
