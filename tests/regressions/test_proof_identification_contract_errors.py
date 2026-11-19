"""Model-output contract handling at the proof sites.

Covers the Gemma incident (one mislabeled theorem field among valid entries),
per-item contract errors, the consented identification retry shape, typed stage
classification, the novelty and brainstorm-gate key recovery, and the order of
each declared schema against its prompt's JSON example.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.aggregator.prompts import submitter_prompts
from backend.autonomous.agents import proof_identification_agent as identification_module
from backend.autonomous.agents.proof_identification_agent import ProofIdentificationAgent
from backend.autonomous.core import proof_novelty as novelty_module
from backend.autonomous.core.proof_verification_stage import ProofVerificationStage
from backend.autonomous.prompts.proof_prompts import (
    PROOF_IDENTIFICATION_RESPONSE_SCHEMA,
    PROOF_IDENTIFICATION_THEOREM_SCHEMA,
    PROOF_NOVELTY_RESPONSE_SCHEMA,
    build_proof_identification_prompt,
    build_proof_novelty_prompt,
)
from backend.shared import brainstorm_proof_gate as gate_module
from backend.shared.api_client_manager import api_client_manager
from backend.shared.config import system_config
from backend.shared.json_key_recovery import ObjectSchema, recover_key_labels
from backend.shared.model_output_contract import (
    MODEL_OUTPUT_NORMALIZED_EVENT,
    ModelOutputContractError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _theorem(theorem_id: str, **overrides) -> dict:
    item = {
        "theorem_id": theorem_id,
        "statement": f"Statement of {theorem_id}.",
        "formal_sketch": "Induct on n.",
        "expected_novelty_tier": "mathematical_discovery",
        "prompt_relevance_rationale": "Directly advances the prompt.",
        "novelty_rationale": "Not in Mathlib or standard references.",
        "why_not_standard_known_result": "No textbook states this bound.",
        "retry_open_target_id": "",
    }
    item.update(overrides)
    return item


def _relabel(item: dict, old: str, new: str) -> dict:
    return {(new if key == old else key): value for key, value in item.items()}


def _gemma_payload() -> str:
    """The 09:16 shape: one valid list where lemma_b mislabels novelty_rationale."""
    theorems = [
        _theorem("lemma_a"),
        _relabel(_theorem("lemma_b"), "novelty_rationale", "novelty_rationally"),
        _theorem("lemma_c"),
    ]
    return json.dumps({"has_provable_theorems": True, "theorems": theorems}, indent=2)


def _response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def broadcasts(monkeypatch):
    events: list[tuple[str, dict]] = []

    async def capture(event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setattr(api_client_manager, "_broadcast", capture)
    return events


@pytest.fixture
def proof_kinds(monkeypatch):
    def set_kinds(*, single_shot: bool = True, master: bool = True) -> None:
        monkeypatch.setattr(system_config, "proof_single_shot_enabled", single_shot, raising=False)
        monkeypatch.setattr(system_config, "proof_master_proofs_enabled", master, raising=False)

    set_kinds()
    return set_kinds


class _ScriptedApi:
    """Stand-in for the identification module's api_client_manager."""

    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[dict] = []

    async def prewarm_assistant_memory_context(self, **_kwargs):
        return ""

    async def generate_completion(self, **kwargs):
        self.calls.append(kwargs)
        return _response(self.contents[min(len(self.calls), len(self.contents)) - 1])


def _agent() -> ProofIdentificationAgent:
    return ProofIdentificationAgent(
        model_id="gemma-local",
        context_window=32000,
        max_output_tokens=2000,
        role_id="autonomous_proof_identification_test",
    )


async def _identify(agent: ProofIdentificationAgent):
    return await agent.identify_candidates(
        user_research_prompt="Bound the sequence.",
        source_type="brainstorm",
        source_id="topic_1",
        source_content="Source content.",
    )


# --- Proof identification -------------------------------------------------


def test_gemma_payload_yields_every_candidate_and_one_recorded_repair():
    repairs: list = []
    has_candidates, candidates = ProofIdentificationAgent._parse_candidate_payload(
        _gemma_payload(), role="autonomous_proof_identification_test", repairs=repairs
    )

    assert has_candidates is True
    assert [candidate.theorem_id for candidate in candidates] == ["lemma_a", "lemma_b", "lemma_c"]
    assert candidates[1].novelty_rationale == "Not in Mathlib or standard references."
    assert len(repairs) == 1
    repair = repairs[0]
    assert (repair.kind, repair.item_path, repair.item_id) == ("key_label", "theorems[1]", "lemma_b")
    assert (repair.from_key, repair.to_key) == ("novelty_rationally", "novelty_rationale")


@pytest.mark.asyncio
async def test_gemma_payload_needs_no_retry_and_publishes_one_normalization(
    monkeypatch, broadcasts, proof_kinds
):
    api = _ScriptedApi([_gemma_payload()])
    monkeypatch.setattr(identification_module, "api_client_manager", api)

    has_candidates, candidates = await _identify(_agent())

    assert has_candidates is True
    assert len(candidates) == 3
    assert len(api.calls) == 1
    normalized = [payload for event, payload in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT]
    assert len(normalized) == 1
    payload = normalized[0]
    assert payload["role_id"] == "autonomous_proof_identification_test"
    assert payload["repair_count"] == 1
    assert payload["repairs"][0]["to_key"] == "novelty_rationale"
    assert "Accepted" in payload["message"] and "'novelty_rationally'" in payload["message"]


def test_stray_character_before_key_is_dropped_and_recorded():
    content = json.dumps(
        {"has_provable_theorems": True, "theorems": [_theorem("lemma_a")]}, indent=2
    ).replace('      "why_not_standard_known_result"', 's     "why_not_standard_known_result"')
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)

    repairs: list = []
    _has, candidates = ProofIdentificationAgent._parse_candidate_payload(content, repairs=repairs)

    assert [candidate.theorem_id for candidate in candidates] == ["lemma_a"]
    assert [repair.kind for repair in repairs] == ["stray_characters"]


def test_one_bad_entry_names_item_and_field_without_claiming_none_were_valid():
    theorems = [
        _theorem("lemma_a"),
        # Two unknown keys: no relabel is attempted, so the entry stays invalid.
        {
            **{k: v for k, v in _theorem("lemma_b").items() if k != "novelty_rationale"},
            "novelty_note": "x",
            "extra_note": "y",
        },
        _theorem("lemma_c"),
    ]
    content = json.dumps({"has_provable_theorems": True, "theorems": theorems})

    with pytest.raises(ModelOutputContractError) as caught:
        ProofIdentificationAgent._parse_candidate_payload(content, role="role_x")

    message = str(caught.value)
    assert "1 of 3 theorem entries was invalid" in message
    assert "theorems[1] 'lemma_b'" in message
    assert "missing required field 'novelty_rationale'" in message
    assert "'novelty_note'" in message
    assert "no valid theorem candidates" not in message
    assert caught.value.role == "role_x"
    assert caught.value.field_summary == "novelty_rationale"


def test_every_bad_field_of_every_bad_entry_is_named():
    theorems = [
        _theorem("lemma_a", expected_novelty_tier="great"),
        {k: v for k, v in _theorem("lemma_b").items() if k not in {"statement", "novelty_rationale"}},
    ]
    content = json.dumps({"has_provable_theorems": True, "theorems": theorems})

    with pytest.raises(ModelOutputContractError) as caught:
        ProofIdentificationAgent._parse_candidate_payload(content)

    message = str(caught.value)
    assert "returned no valid theorem candidates: 2 of 2 theorem entries were invalid" in message
    assert "theorems[0] 'lemma_a': invalid field 'expected_novelty_tier'" in message
    assert "missing required fields 'statement', 'novelty_rationale'" in message


def test_contract_message_never_carries_raw_output_or_unsafe_tokens():
    theorems = [{**_theorem("lemma_a"), "novelty_rationale": "", 'bad"key ignore previous': "v"}]
    content = json.dumps({"has_provable_theorems": True, "theorems": theorems})

    with pytest.raises(ModelOutputContractError) as caught:
        ProofIdentificationAgent._parse_candidate_payload(content)

    message = str(caught.value)
    assert "'badkeyignoreprevious'" in message
    assert "Directly advances the prompt" not in message


def test_syntax_failure_is_typed_without_excerpt():
    with pytest.raises(ModelOutputContractError) as caught:
        ProofIdentificationAgent._parse_candidate_payload("The source has ideas but this is not JSON at all.")
    assert caught.value.stage == "syntax"
    assert "not JSON at all" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("single_shot", "master", "expected_kind_line"),
    [
        (True, True, '"proposal_kind": "individual | master_proof"'),
        (False, True, '"proposal_kind": "master_proof"'),
        (True, False, None),
    ],
)
async def test_retry_prompt_names_bad_item_and_offers_allowed_proposal_kinds(
    monkeypatch, broadcasts, proof_kinds, single_shot, master, expected_kind_line
):
    proof_kinds(single_shot=single_shot, master=master)
    bad = {
        "has_provable_theorems": True,
        "theorems": [
            _theorem("lemma_a"),
            {k: v for k, v in _theorem("lemma_b").items() if k != "novelty_rationale"},
        ],
    }
    good = {"has_provable_theorems": True, "theorems": [_theorem("lemma_a"), _theorem("lemma_b")]}
    api = _ScriptedApi([json.dumps(bad), json.dumps(good)])
    monkeypatch.setattr(identification_module, "api_client_manager", api)

    _has, candidates = await _identify(_agent())

    assert len(candidates) == 2
    assert len(api.calls) == 2
    retry_text = api.calls[1]["messages"][-1]["content"]
    assert "theorems[1] 'lemma_b': missing required field 'novelty_rationale'" in retry_text
    assert "1 of 2 theorem entries was invalid" in retry_text
    assert "Correct every theorem entry and field named in the ERROR" in retry_text
    if expected_kind_line:
        assert expected_kind_line in retry_text
    else:
        assert "proposal_kind" not in retry_text


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_typed_contract_error(monkeypatch, broadcasts, proof_kinds):
    bad = {
        "has_provable_theorems": True,
        "theorems": [{k: v for k, v in _theorem("lemma_b").items() if k != "statement"}],
    }
    api = _ScriptedApi([json.dumps(bad)])
    monkeypatch.setattr(identification_module, "api_client_manager", api)

    with pytest.raises(ModelOutputContractError) as caught:
        await _identify(_agent())

    assert len(api.calls) == 2
    assert caught.value.role == "autonomous_proof_identification_test"
    assert "'statement'" in str(caught.value)


@pytest.mark.asyncio
async def test_rejected_reply_never_publishes_an_accepted_normalization(
    monkeypatch, broadcasts, proof_kinds
):
    # theorems[0] needs a relabel, theorems[1] is invalid: the reply is rejected,
    # so no "Accepted ..." normalization may be shown for it.
    bad = {
        "has_provable_theorems": True,
        "theorems": [
            _relabel(_theorem("t1"), "novelty_rationale", "novelty_rationally"),
            {k: v for k, v in _theorem("t2").items() if k != "why_not_standard_known_result"},
        ],
    }
    api = _ScriptedApi([json.dumps(bad), json.dumps(bad)])
    monkeypatch.setattr(identification_module, "api_client_manager", api)

    with pytest.raises(ModelOutputContractError):
        await _identify(_agent())

    assert len(api.calls) == 2
    assert [event for event, _ in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT] == []


def test_parsers_without_a_repairs_list_never_repair():
    with pytest.raises(ModelOutputContractError):
        ProofIdentificationAgent._parse_candidate_payload(_gemma_payload())
    content = json.dumps(
        {
            "novelty_tier": "novel_variant",
            "reasonig": "Relabeled only for a publishing caller.",
            "novelty_scope": "standalone",
            "group_claim": "",
            "group_proof_ids": [],
        }
    )
    assessment = novelty_module._parse_novelty_payload(content)
    assert assessment.reasoning != "Relabeled only for a publishing caller."


@pytest.mark.asyncio
async def test_rejected_novelty_reply_publishes_no_normalization(broadcasts):
    content = json.dumps({"novelty_tier": "huge", "reasonig": "x"})
    with pytest.raises(ModelOutputContractError):
        await novelty_module._parse_and_report_novelty(
            content,
            role_id="autonomous_proof_novelty",
            task_id="nov_1",
            allowed_group_proof_ids=frozenset(),
            require_explicit_scope=False,
        )
    assert [event for event, _ in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT] == []


# --- Stage classification -------------------------------------------------


def test_contract_error_mentioning_context_is_not_a_context_error():
    error = ModelOutputContractError(
        "Proof identification claimed provable theorems but 1 of 2 theorem entries was invalid",
        role="autonomous_proof_identification_brainstorm",
        problems=(),
        missing_keys=("source_context_note",),
        item_path="theorems[1]",
    )
    assert "context" in str(error).lower()

    report = ProofVerificationStage._stage_failure_report(
        error, round_phase="identification", workflow_mode="autonomous"
    )

    assert report["error_kind"] == "model_output_contract_error"
    assert report["message"].startswith("Proof verification encountered an error: ")
    assert "proof-identification response did not match the required JSON format" in report["message"]
    assert "No Lean work was attempted" in report["message"]
    assert report["contract_fields"]["contract_role_id"] == "autonomous_proof_identification_brainstorm"
    assert report["contract_fields"]["contract_item_path"] == "theorems[1]"


def test_contract_error_wrapped_by_another_error_is_still_found():
    inner = ModelOutputContractError("bad", role="r")
    try:
        try:
            raise inner
        except ModelOutputContractError as exc:
            raise RuntimeError("wrapper") from exc
    except RuntimeError as outer:
        report = ProofVerificationStage._stage_failure_report(
            outer, round_phase="list_review", workflow_mode="autonomous"
        )
    assert report["error_kind"] == "model_output_contract_error"
    assert "candidate-list review response" in report["message"]


def test_lean_phase_failure_never_claims_no_lean_work():
    contract = ProofVerificationStage._stage_failure_report(
        ModelOutputContractError("bad", role="r"), round_phase="lean", workflow_mode="autonomous"
    )
    internal = ProofVerificationStage._stage_failure_report(
        RuntimeError("boom"), round_phase="lean", workflow_mode="autonomous"
    )
    assert "No Lean work" not in contract["message"]
    assert "No Lean work" not in internal["message"]
    assert internal["error_kind"] == "internal_error"


def test_free_text_context_word_is_internal_and_provider_context_text_is_context():
    internal = ProofVerificationStage._stage_failure_report(
        RuntimeError("source context file missing"),
        round_phase="identification",
        workflow_mode="autonomous",
    )
    provider = ProofVerificationStage._stage_failure_report(
        RuntimeError("context_length_exceeded"),
        round_phase="identification",
        workflow_mode="autonomous",
    )
    assert internal["error_kind"] == "internal_error"
    assert internal["message"].endswith("No Lean work was attempted.")
    assert provider["error_kind"] == "context_error"


@pytest.mark.asyncio
async def test_stage_reports_identification_contract_failure_accurately(
    monkeypatch, broadcasts, proof_kinds
):
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    bad = {
        "has_provable_theorems": True,
        "theorems": [
            _theorem("lemma_a"),
            {k: v for k, v in _theorem("lemma_b").items() if k != "why_not_standard_known_result"},
        ],
    }
    api = _ScriptedApi([json.dumps(bad)])
    monkeypatch.setattr(identification_module, "api_client_manager", api)
    events: list[tuple[str, dict]] = []
    statuses: list[str] = []

    async def broadcast(event_type, payload):
        events.append((event_type, payload))

    async def checkpoint(payload):
        statuses.append(payload["status"])

    result = await ProofVerificationStage().run(
        content="brainstorm source",
        source_type="brainstorm",
        source_id="topic_contract_failure",
        user_prompt="prove things",
        submitter_model="gemma-local",
        submitter_context=32000,
        submitter_max_tokens=2000,
        validator_model="validator",
        validator_context=32000,
        validator_max_tokens=2000,
        broadcast_fn=broadcast,
        novel_proofs_db=SimpleNamespace(),
        append_to_source=False,
        checkpoint_callback=checkpoint,
    )

    assert result.had_error is True
    assert "error" in statuses
    complete = [payload for event, payload in events if event == "proof_check_complete"]
    assert len(complete) == 1
    payload = complete[0]
    assert payload["had_error"] is True
    assert payload["error_kind"] == "model_output_contract_error"
    assert payload["error_stage"] == "identification"
    assert payload["message"].startswith("Proof verification encountered an error: ")
    assert "theorems[1] 'lemma_b'" in payload["message"]
    assert "'why_not_standard_known_result'" in payload["message"]
    assert "No Lean work was attempted" in payload["message"]
    assert "no valid theorem candidates" not in payload["message"]
    assert "proof_check_no_candidates" not in [event for event, _ in events]


# --- Proof novelty ----------------------------------------------------------


def test_novelty_mislabeled_reasoning_is_recovered():
    content = json.dumps(
        {
            "novelty_tier": "novel_variant",
            "reasonig": "The verified bound sharpens the known constant.",
            "novelty_scope": "standalone",
            "group_claim": "",
            "group_proof_ids": [],
        }
    )
    repairs: list = []
    assessment = novelty_module._parse_novelty_payload(content, repairs=repairs)

    assert assessment.reasoning == "The verified bound sharpens the known constant."
    assert [(r.from_key, r.to_key) for r in repairs] == [("reasonig", "reasoning")]


def test_novelty_tier_is_never_relabeled_and_errors_are_typed():
    content = json.dumps(
        {
            "novelty_tierr": "major_mathematical_discovery",
            "reasoning": "x",
            "novelty_scope": "standalone",
            "group_claim": "",
            "group_proof_ids": [],
        }
    )
    with pytest.raises(ModelOutputContractError) as caught:
        novelty_module._parse_novelty_payload(content)
    assert caught.value.missing_keys == ("novelty_tier",)

    with pytest.raises(ModelOutputContractError) as caught:
        novelty_module._parse_novelty_payload(json.dumps({"novelty_tier": "huge", "reasoning": "x"}))
    assert caught.value.invalid_fields == ("novelty_tier",)
    assert "huge" not in str(caught.value)


@pytest.mark.asyncio
async def test_novelty_fallback_is_labeled_without_error_text(monkeypatch, broadcasts):
    class Api:
        calls = 0

        async def generate_completion(self, **_kwargs):
            Api.calls += 1
            return _response(json.dumps({"novelty_tier": "huge", "reasoning": "secret-ish model text"}))

    monkeypatch.setattr(novelty_module, "api_client_manager", Api())
    assessment = await novelty_module.assess_proof_novelty_details(
        user_prompt="Prompt.",
        theorem_statement="A verified theorem.",
        lean_code="theorem t : True := trivial",
        validator_model="validator",
        validator_context=16000,
        validator_max_tokens=1000,
        existing_novel_proofs="",
        task_id="proof_novelty_fallback",
    )

    assert Api.calls == 2
    assert assessment.novelty_tier == "not_novel"
    assert assessment.assessment_origin == "conservative_fallback"
    assert "did not match the required JSON format (novelty_tier)" in assessment.reasoning
    assert "stored conservatively as not_novel" in assessment.reasoning
    assert "secret-ish" not in assessment.reasoning and "huge" not in assessment.reasoning


# --- Brainstorm proof gate ------------------------------------------------


def _lean_submission(**overrides) -> dict:
    item = {
        "submission_type": "lean_proof",
        "theorem_statement": "For all n, f n <= 2 n.",
        "formal_sketch": "Induction.",
        "expected_novelty_tier": "novel_variant",
        "prompt_relevance_rationale": "Bounds the prompt's sequence.",
        "novelty_rationale": "Absent from Mathlib.",
        "why_not_standard_known_result": "Uses the prompt's own recurrence.",
        "theorem_name": "f_le",
        "lean_code": "theorem f_le : True := trivial",
        "reasoning": "Progress.",
    }
    item.update(overrides)
    return item


def _patch_failing_lean(monkeypatch):
    checked: list[str] = []

    class Client:
        async def check_proof(self, code, timeout=None):
            checked.append(code)
            return SimpleNamespace(
                success=False,
                error_output="type mismatch",
                diagnostic_output="",
                goal_states="",
                raw_stderr="",
                compile_receipt=None,
            )

    monkeypatch.setattr(gate_module, "get_lean4_client", lambda: Client())
    monkeypatch.setattr(gate_module, "raise_for_lean_workspace_failure", lambda _result: None)
    return checked


async def _gate(parsed: dict):
    return await gate_module.verify_brainstorm_proof_candidate(
        parsed=parsed,
        user_prompt="Prompt.",
        source_context="Context.",
        model_id="gemma-local",
        role_id="aggregator_submitter_1",
        task_id_prefix="agg_sub1_001_lean",
        max_tokens=1000,
        validator_model="validator",
        validator_context=16000,
        validator_max_tokens=1000,
        validator_role_id="aggregator_validator",
        max_attempts=1,
    )


@pytest.mark.asyncio
async def test_gate_recovers_one_mislabeled_rationale_and_reaches_lean(monkeypatch, broadcasts):
    checked = _patch_failing_lean(monkeypatch)
    parsed = _relabel(_lean_submission(), "novelty_rationale", "novelty_rationally")

    result = await _gate(parsed)

    assert checked == ["theorem f_le : True := trivial"]
    assert result.novelty_rationale == "Absent from Mathlib."
    assert "rationale field" not in result.failure_feedback
    normalized = [payload for event, payload in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT]
    assert len(normalized) == 1 and normalized[0]["role_id"] == "aggregator_submitter_1"
    assert "novelty_rationally" in parsed  # caller's object is never mutated


@pytest.mark.asyncio
async def test_gate_publishes_relabel_only_after_required_field_checks(monkeypatch, broadcasts):
    checked = _patch_failing_lean(monkeypatch)
    refused = _relabel(
        _lean_submission(expected_novelty_tier=""), "novelty_rationale", "novelty_rationally"
    )

    result = await _gate(refused)

    assert checked == []
    assert result.accepted is False
    assert not [payload for event, payload in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT]


@pytest.mark.asyncio
async def test_gate_never_relabels_lean_code_or_uses_aliases_as_sources(monkeypatch, broadcasts):
    checked = _patch_failing_lean(monkeypatch)
    result = await _gate(_relabel(_lean_submission(), "lean_code", "lean_cod"))
    assert checked == []
    assert "both `theorem_statement` and `lean_code` are required" in result.failure_feedback

    aliased = _relabel(_lean_submission(), "theorem_statement", "theorem_or_lemma")
    repaired, repairs = recover_key_labels(aliased, gate_module.BRAINSTORM_LEAN_PROOF_SCHEMA)
    assert repairs == [] and repaired is aliased
    assert not [payload for event, payload in broadcasts if event == MODEL_OUTPUT_NORMALIZED_EVENT]


def test_gate_does_not_coerce_an_idea_into_the_lean_variant():
    idea = {"submission": "An idea about the bound.", "reasoning": "Why.", "lean_cod": "theorem x"}
    assert gate_module.is_lean_proof_submission(idea) is False


# --- Schema vs prompt order ------------------------------------------------


def _key_order_matches(item: dict, schema: ObjectSchema) -> None:
    declared = [spec.name for spec in schema.keys]
    assert all(key in declared for key in item), (list(item), declared)
    assert list(item) == [name for name in declared if name in item]
    required = [spec.name for spec in schema.keys if spec.required]
    assert all(name in item for name in required)


def _example_json(prompt: str) -> dict:
    block = prompt.rsplit("JSON format:\n", 1)[1]
    parsed, _end = json.JSONDecoder().raw_decode(block.strip())
    return parsed


@pytest.mark.parametrize(
    ("single_shot", "master"), [(True, True), (True, False), (False, True)]
)
def test_identification_schema_matches_prompt_example_for_each_proof_kind(single_shot, master):
    prompt = build_proof_identification_prompt(
        user_prompt="Prompt.",
        source_type="brainstorm",
        source_id="topic_1",
        source_content="Content mentioning novelty_rationale and theorem_id early.",
        allow_single_shot_proofs=single_shot,
        allow_master_proofs=master,
    )
    example = _example_json(prompt)

    assert list(example) == [spec.name for spec in PROOF_IDENTIFICATION_RESPONSE_SCHEMA.keys]
    item = example["theorems"][0]
    _key_order_matches(item, PROOF_IDENTIFICATION_THEOREM_SCHEMA)
    assert ("proposal_kind" in item) is master


def test_novelty_schema_matches_prompt_example():
    prompt = build_proof_novelty_prompt(
        user_prompt="Prompt.",
        theorem_statement="T.",
        lean_code="theorem t : True := trivial",
        existing_novel_proofs="",
    )
    example = _example_json(prompt)
    assert list(example) == [spec.name for spec in PROOF_NOVELTY_RESPONSE_SCHEMA.keys]


def _lean_variant_example(text: str) -> dict:
    marker = '"submission_type": "lean_proof"'
    start = text.rindex("{", 0, text.index(marker))
    parsed, _end = json.JSONDecoder().raw_decode(text[start:])
    return parsed


def test_brainstorm_gate_schema_matches_aggregator_and_leanoj_examples():
    examples = [
        _lean_variant_example(submitter_prompts.get_submitter_system_prompt(lean4_enabled=True)),
        _lean_variant_example(submitter_prompts.get_submitter_json_schema(lean4_enabled=True)),
    ]
    leanoj_source = (REPO_ROOT / "backend" / "leanoj" / "prompts.py").read_text(encoding="utf-8")
    leanoj_line = next(
        line for line in leanoj_source.splitlines() if '"submission_type": "lean_proof", "theorem_statement"' in line
    )
    examples.append(_lean_variant_example(leanoj_line.replace("{{", "{").replace("}}", "}")))

    for example in examples:
        _key_order_matches(example, gate_module.BRAINSTORM_LEAN_PROOF_SCHEMA)
