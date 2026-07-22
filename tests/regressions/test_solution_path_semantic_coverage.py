import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative_path: str, class_name: str, method_name: str) -> str:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return ast.get_source_segment(text, child) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {relative_path}")


def test_aggregator_batch_has_one_hook_and_one_batch_proposal():
    source = _source(
        "backend/aggregator/agents/validator.py",
        "ValidatorAgent",
        "_assess_batch_quality",
    )
    assert source.count("with_validator_hook(") == 1
    assert source.count("enqueue_optional_update(") == 1
    assert 'source_phase="batch_submission_validation"' in source
    assert "batch=True" in source
    assert source.index("submission_number") < source.index("enqueue_optional_update(")


def test_aggregator_cleanup_semantic_validators_are_integrated():
    cleanup = _source(
        "backend/aggregator/agents/validator.py",
        "ValidatorAgent",
        "perform_cleanup_review",
    )
    removal = _source(
        "backend/aggregator/agents/validator.py",
        "ValidatorAgent",
        "validate_removal",
    )

    assert cleanup.count("with_validator_hook(") == 1
    assert cleanup.count("enqueue_optional_update(") == 1
    assert 'source_phase="cleanup_review_validation"' in cleanup
    assert removal.count("with_validator_hook(") == 1
    assert removal.count("enqueue_optional_update(") == 1
    assert 'source_phase="cleanup_removal_validation"' in removal


def test_compiler_critique_semantic_validators_are_integrated():
    validation = _source(
        "backend/compiler/core/compiler_coordinator.py",
        "CompilerCoordinator",
        "_validate_critique",
    )
    cleanup = _source(
        "backend/compiler/core/compiler_coordinator.py",
        "CompilerCoordinator",
        "_perform_critique_cleanup",
    )

    assert validation.count("with_validator_hook(") == 1
    assert validation.count("enqueue_optional_update(") == 1
    assert 'source_phase="critique_validation"' in validation
    assert cleanup.count("with_validator_hook(") == 1
    assert cleanup.count("enqueue_optional_update(") == 1
    assert 'source_phase="critique_cleanup_validation"' in cleanup
    assert validation.count("_retry") >= 1
    assert cleanup.count("_retry") >= 1


def test_compiler_lean_placement_excludes_solution_path_proposals():
    validation = _source(
        "backend/compiler/validation/compiler_validator.py",
        "CompilerValidator",
        "validate_submission",
    )

    assert "validation_mode = self._get_effective_validation_mode(submission)" in validation
    assert validation.count('validation_mode != "rigor_lean_placement"') == 2
    assert validation.count("with_validator_hook(") == 1
    assert validation.count("enqueue_optional_update(") == 1


def test_new_solver_surfaces_receive_only_budgeted_advisory_path():
    reference_expansion = _source(
        "backend/autonomous/agents/reference_selector.py",
        "ReferenceSelectorAgent",
        "_request_expansion",
    )
    reference_selection = _source(
        "backend/autonomous/agents/reference_selector.py",
        "ReferenceSelectorAgent",
        "_make_final_selection",
    )
    continuation = _source(
        "backend/autonomous/core/autonomous_coordinator.py",
        "AutonomousCoordinator",
        "_brainstorm_continuation_decision",
    )
    proof_identification = _source(
        "backend/autonomous/agents/proof_identification_agent.py",
        "ProofIdentificationAgent",
        "identify_candidates",
    )

    for source in (
        reference_expansion,
        reference_selection,
        continuation,
        proof_identification,
    ):
        assert "with_budgeted_solver_plan(" in source
        assert "with_validator_hook(" not in source
        assert "enqueue_optional_update(" not in source
