import pytest
from unittest.mock import AsyncMock

from backend.aggregator.agents import submitter as submitter_module
from backend.aggregator.agents.submitter import SubmitterAgent, _requires_lean_retry_schema
from backend.shared.brainstorm_proof_gate import BrainstormProofGateResult
from backend.aggregator.prompts.submitter_prompts import (
    CREATIVITY_EMPHASIS_BOOST_PROMPT,
    build_submitter_prompt,
    get_submitter_json_schema,
    get_submitter_system_prompt,
)
from backend.aggregator.prompts.validator_prompts import (
    get_cleanup_review_system_prompt,
    get_removal_validation_system_prompt,
    get_validator_dual_json_schema,
    get_validator_dual_system_prompt,
    get_validator_system_prompt,
    get_validator_triple_json_schema,
    get_validator_triple_system_prompt,
)


def test_ordinary_submitter_seeks_direct_all_purpose_solutions() -> None:
    prompt = get_submitter_system_prompt(lean4_enabled=False)

    assert "strongest credible and genuinely novel solution" in prompt
    assert "user's exact objective" in prompt
    assert "WHOLE question" in prompt
    assert "next best necessary piece" in prompt
    for contribution in (
        "invention",
        "mechanism",
        "design",
        "algorithm",
        "experimental proposal",
        "falsifiable hypothesis",
        "counterexample",
        "impossibility argument",
        "implementation strategy",
        "risk analysis",
    ):
        assert contribution in prompt
    assert "You are a mathematical submitter" not in prompt
    assert "ALL submissions must be rooted in sound mathematical reasoning" not in prompt


def test_mathematics_and_lean_remain_affirmatively_available() -> None:
    prompt = get_submitter_system_prompt(lean4_enabled=True)
    schema = get_submitter_json_schema(lean4_enabled=True)

    assert "Mathematical reasoning and formal proof remain first-class" in prompt
    assert "Mathematics, theorem discovery, proof, and formalization are explicitly welcome" in prompt
    assert "mathematical claims require sound derivation, proof, or explicit assumptions" in prompt
    for field in (
        "theorem_statement",
        "formal_sketch",
        "expected_novelty_tier",
        "prompt_relevance_rationale",
        "novelty_rationale",
        "why_not_standard_known_result",
        "theorem_name",
        "lean_code",
        "reasoning",
    ):
        assert field in schema
    assert "routine helpers" in schema
    assert "trivial/easy proofs" in schema


def test_lean_schema_is_absent_when_disabled() -> None:
    prompt = build_submitter_prompt("objective", "context", lean4_enabled=False)

    for lean_only_text in (
        "lean_proof",
        "Lean proof candidate",
        "theorem_statement",
        "formal_sketch",
        "expected_novelty_tier",
        "prompt_relevance_rationale",
        "novelty_rationale",
        "why_not_standard_known_result",
        "theorem_name",
        "lean_code",
    ):
        assert lean_only_text not in prompt


def test_claim_type_rigor_covers_engineering_and_provenance() -> None:
    prompt = get_submitter_system_prompt()

    for requirement in (
        "concrete mechanism",
        "constraints",
        "feasibility reasoning",
        "failure modes",
        "verification plan",
        "Literature claims",
        "Empirical claims",
        "Artifact claims",
    ):
        assert requirement in prompt
    assert "NEVER invent experiments" in prompt
    assert "proposed experiment" in prompt
    assert "without claiming that a prototype or test result already exists" in get_submitter_json_schema()


@pytest.mark.parametrize(
    "prompt_builder",
    [
        get_validator_system_prompt,
        get_validator_dual_system_prompt,
        get_validator_triple_system_prompt,
    ],
)
def test_all_validator_batch_sizes_share_the_same_policy(prompt_builder) -> None:
    prompt = prompt_builder()

    for criterion in (
        "SHARED ALL-PURPOSE EVALUATION CRITERIA",
        "Direct impact",
        "Genuine novelty",
        "Correctness",
        "Provenance and uncertainty",
        "Specificity and verification",
        "Feasibility",
        "Non-redundancy",
        "Mathematics remains first-class",
        "non-mathematical work must not be rejected merely for lacking theorem or proof form",
    ):
        assert criterion in prompt
    assert "user's exact objective" in prompt
    assert "[LEAN 4 VERIFIED BRAINSTORM PROOF]" in prompt
    assert "Lean verification only establishes formal validity" in prompt
    for forbidden_universal in (
        "evaluate mathematical submissions",
        "user's mathematical prompt",
        "rather than mathematical solutions",
        "lacks mathematical rigor",
        "invalid proof structure",
    ):
        assert forbidden_universal not in prompt


def test_batch_validators_preserve_independent_redundancy_resolution() -> None:
    dual = get_validator_dual_system_prompt()
    triple = get_validator_triple_system_prompt()

    assert "TWO SEPARATE, INDEPENDENT decisions first" in dual
    assert "Keep ONLY the stronger/more complete one" in dual
    assert "THREE SEPARATE, INDEPENDENT decisions first" in triple
    assert "keep ONLY the strongest" in triple


def test_cleanup_and_removal_preserve_unique_non_mathematical_value() -> None:
    cleanup = get_cleanup_review_system_prompt()
    removal = get_removal_validation_system_prompt()

    assert "AT MOST ONE" in cleanup
    assert "unique mechanism, design, algorithm, evidence plan, risk analysis" in cleanup
    assert "select the WEAKEST one for removal" in cleanup
    assert "When in doubt, DO NOT recommend removal" in cleanup
    assert "ANY unique value" in removal
    assert "If uncertain, REJECT the removal" in removal
    assert "unique value not covered elsewhere" in removal
    for prompt in (cleanup, removal):
        assert "[LEAN 4 VERIFIED BRAINSTORM PROOF]" in prompt
        assert "Lean verification only establishes formal validity" in prompt
        assert "Do NOT re-litigate Lean syntax" in prompt


def test_validator_examples_use_claim_appropriate_mixed_domain_standards() -> None:
    dual = get_validator_dual_json_schema()
    triple = get_validator_triple_json_schema()

    assert "segmented battery-isolation mechanism" in dual
    assert "complete mathematical argument" in dual
    assert "software isolation design" in triple
    assert "rigorous counterexample" in triple
    assert "provide a concrete mechanism and verification plan" in triple
    assert "provide a test that could falsify the claim" in triple


def test_retry_variant_detection_preserves_recognizable_lean_candidates() -> None:
    malformed_lean_outputs = (
        '{"submission_type": "lean_proof", "lean_code": "theorem x',
        '{"theorem_statement": "High-impact result", "formal_sketch": ',
        '{"lean_code": "import Mathlib\\ntheorem target : True := by trivial"',
    )

    for output in malformed_lean_outputs:
        assert _requires_lean_retry_schema(output, lean4_enabled=True)
        assert not _requires_lean_retry_schema(output, lean4_enabled=False)
    assert not _requires_lean_retry_schema(
        '{"submission_type": "idea", "submission": "engineering mechanism"',
        lean4_enabled=True,
    )


def test_retry_source_requires_complete_schema_and_forbids_lean_downgrade() -> None:
    import inspect

    from backend.aggregator.agents import submitter as submitter_module

    source = inspect.getsource(submitter_module.SubmitterAgent._generate_submission)
    assert "get_submitter_json_schema" in source
    assert "do not convert it into an ordinary idea" in source
    assert '"your submission (LaTeX allowed' not in source


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _patch_submitter_dependencies(monkeypatch, agent: SubmitterAgent, responses: list[dict]) -> None:
    monkeypatch.setattr(
        submitter_module.shared_training_memory,
        "get_all_content",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(agent.local_memory, "get_all_content", AsyncMock(return_value=""))
    monkeypatch.setattr(agent.local_memory, "add_rejection", AsyncMock())
    monkeypatch.setattr(
        submitter_module.context_allocator,
        "allocate_submitter_context",
        AsyncMock(return_value={"direct": "", "rag_context": None}),
    )
    monkeypatch.setattr(
        submitter_module.api_client_manager,
        "prewarm_assistant_memory_context",
        AsyncMock(),
    )
    monkeypatch.setattr(
        submitter_module.api_client_manager,
        "generate_completion",
        AsyncMock(side_effect=responses),
    )
    monkeypatch.setattr(
        submitter_module.api_client_manager,
        "extract_call_metadata",
        lambda response: {},
    )
    monkeypatch.setattr(
        submitter_module.lm_studio_client,
        "cache_model_load_config",
        AsyncMock(),
    )


@pytest.mark.asyncio
async def test_ordinary_engineering_idea_parses_without_proof_gate(monkeypatch) -> None:
    agent = SubmitterAgent(
        submitter_id=1,
        model_name="model-a",
        user_prompt="Design a safer battery pack",
        user_files_content={},
        context_window=32_000,
        max_output_tokens=2_000,
    )
    _patch_submitter_dependencies(
        monkeypatch,
        agent,
        [
            _completion(
                '{"submission_type":"idea","submission":"Use segmented isolation with '
                'redundant contactors, explicit sensor-failure handling, and staged '
                'fault-injection tests.","reasoning":"This is a concrete mechanism '
                'with constraints and verification."}'
            )
        ],
    )
    proof_gate = AsyncMock()
    monkeypatch.setattr(submitter_module, "verify_brainstorm_proof_candidate", proof_gate)
    monkeypatch.setattr(submitter_module.system_config, "lean4_enabled", True)

    submission = await agent._generate_submission()

    assert submission is not None
    assert "segmented isolation" in submission.content
    proof_gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_lean_retry_still_passes_proof_gate(monkeypatch) -> None:
    agent = SubmitterAgent(
        submitter_id=1,
        model_name="model-a",
        user_prompt="Prove the target theorem",
        user_files_content={},
        context_window=32_000,
        max_output_tokens=2_000,
    )
    repaired = (
        '{"submission_type":"lean_proof","theorem_statement":"True",'
        '"formal_sketch":"Use True.intro.","expected_novelty_tier":"novel_formulation",'
        '"prompt_relevance_rationale":"Directly proves the requested target.",'
        '"novelty_rationale":"A prompt-specific formulation.",'
        '"why_not_standard_known_result":"The exact target is prompt-specific.",'
        '"theorem_name":"target","lean_code":"theorem target : True := by trivial",'
        '"reasoning":"Complete Lean proof candidate."}'
    )
    _patch_submitter_dependencies(
        monkeypatch,
        agent,
        [
            _completion('{"submission_type":"lean_proof","lean_code":"theorem target'),
            _completion(repaired),
        ],
    )
    gate_result = BrainstormProofGateResult(
        accepted=True,
        submission_content="[LEAN 4 VERIFIED BRAINSTORM PROOF]\nTrue",
        theorem_statement="True",
        theorem_name="target",
        formal_sketch="Use True.intro.",
        expected_novelty_tier="novel_formulation",
        prompt_relevance_rationale="Directly proves the target.",
        novelty_rationale="Prompt-specific formulation.",
        why_not_standard_known_result="Exact prompt target.",
        lean_code="theorem target : True := by trivial",
        reasoning="Lean verified.",
    )
    proof_gate = AsyncMock(return_value=gate_result)
    monkeypatch.setattr(submitter_module, "verify_brainstorm_proof_candidate", proof_gate)
    monkeypatch.setattr(submitter_module.system_config, "lean4_enabled", True)

    submission = await agent._generate_submission()

    assert submission is not None
    proof_gate.assert_awaited_once()
    assert submission.content.startswith("[LEAN 4 VERIFIED BRAINSTORM PROOF]")
    assert submission.metadata["brainstorm_lean_proof"]["lean_code"].startswith("theorem target")


@pytest.mark.asyncio
async def test_rejected_proof_gate_cannot_emit_ordinary_submission(monkeypatch) -> None:
    agent = SubmitterAgent(
        submitter_id=1,
        model_name="model-a",
        user_prompt="Prove the target theorem",
        user_files_content={},
        context_window=32_000,
        max_output_tokens=2_000,
    )
    candidate = (
        '{"submission_type":"lean_proof","theorem_statement":"True",'
        '"formal_sketch":"Use True.intro.","expected_novelty_tier":"novel_formulation",'
        '"prompt_relevance_rationale":"Direct target.","novelty_rationale":"Prompt-specific.",'
        '"why_not_standard_known_result":"Exact prompt target.","theorem_name":"target",'
        '"lean_code":"theorem target : True := by trivial","reasoning":"Candidate."}'
    )
    _patch_submitter_dependencies(monkeypatch, agent, [_completion(candidate)])
    proof_gate = AsyncMock(
        return_value=BrainstormProofGateResult(
            accepted=False,
            failure_feedback="Rejected by the protected proof gate.",
        )
    )
    monkeypatch.setattr(submitter_module, "verify_brainstorm_proof_candidate", proof_gate)
    monkeypatch.setattr(submitter_module.system_config, "lean4_enabled", True)

    submission = await agent._generate_submission()

    assert submission is None
    proof_gate.assert_awaited_once()
    agent.local_memory.add_rejection.assert_awaited_once()


def test_creativity_emphasis_is_optional_rigorous_and_non_fabricating() -> None:
    normal = build_submitter_prompt("objective", "context", creativity_emphasized=False)
    creative = build_submitter_prompt("objective", "context", creativity_emphasized=True)

    assert "CREATIVITY EMPHASIS BOOST" not in normal
    assert CREATIVITY_EMPHASIS_BOOST_PROMPT in creative
    assert "near-solution or adjacent solution" in creative
    assert "not permission to fabricate" in creative
    assert "same JSON schema and rigor requirements" in creative
