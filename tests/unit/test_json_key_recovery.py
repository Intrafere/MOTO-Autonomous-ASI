"""Guards of schema-aware single-key relabeling (json_key_recovery)."""
import copy

import pytest

from backend.shared.json_key_recovery import KeySpec, ObjectSchema, recover_key_labels
from backend.shared.json_parser import parse_json
from backend.shared.model_output_contract import OutputRepair

THEOREM = ObjectSchema(
    name="test_identification_theorem",
    keys=(
        KeySpec("theorem_id", aliases=("id",)),
        KeySpec("statement", repairable=True),
        KeySpec("formal_sketch", required=False, repairable=True),
        KeySpec("expected_novelty_tier", repairable=True),
        KeySpec("prompt_relevance_rationale", repairable=True),
        KeySpec("novelty_rationale", repairable=True),
        KeySpec("why_not_standard_known_result", repairable=True),
        KeySpec("retry_open_target_id", required=False),
        KeySpec("proposal_kind", required=False),
    ),
    item_id_key="theorem_id",
)
IDENTIFICATION = ObjectSchema(
    name="test_identification",
    keys=(
        KeySpec("has_provable_theorems", value_type="bool"),
        KeySpec("theorems", value_type="list_object", item_schema=THEOREM),
    ),
    allow_extra=frozenset({"solution_path_update"}),
)


def _theorem(**overrides):
    item = {
        "theorem_id": "lemma_b",
        "statement": "Every x satisfies P(x).",
        "formal_sketch": "Induction on x.",
        "expected_novelty_tier": "mathematical_discovery",
        "prompt_relevance_rationale": "Solves the prompt's main case.",
        "novelty_rationale": "No known bound of this form.",
        "why_not_standard_known_result": "Not in Mathlib.",
        "retry_open_target_id": "",
    }
    item.update(overrides)
    return item


def _rename(item, old, new, value=None):
    """Rename one key in place order (optionally replacing its value)."""
    return {
        (new if key == old else key): (value if key == old and value is not None else item_value)
        for key, item_value in item.items()
    }


@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        ("novelty_rationally", "novelty_rationale"),
        ("prompt_reradle_rationale", "prompt_relevance_rationale"),
        ("expected_novelty_int_tier", "expected_novelty_tier"),
    ],
)
def test_observed_gemma_misspellings_recover(wrong, right):
    item = _rename(_theorem(), right, wrong)
    payload = {"has_provable_theorems": True, "theorems": [item]}
    recovered, repairs = recover_key_labels(payload, IDENTIFICATION)
    fixed = recovered["theorems"][0]
    assert fixed == _theorem()
    assert list(fixed) == list(_theorem())
    assert len(repairs) == 1
    repair = repairs[0]
    assert (repair.kind, repair.item_path, repair.item_id) == ("key_label", "theorems[0]", "lemma_b")
    assert (repair.from_key, repair.to_key) == (wrong, right)


def test_recovery_does_not_mutate_its_input_and_is_idempotent():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    payload = {"has_provable_theorems": True, "theorems": [item]}
    snapshot = copy.deepcopy(payload)
    recovered, _repairs = recover_key_labels(payload, IDENTIFICATION)
    assert payload == snapshot
    again, repairs = recover_key_labels(recovered, IDENTIFICATION)
    assert repairs == []
    assert again is recovered


def test_unchanged_objects_are_returned_as_the_same_object():
    payload = {"has_provable_theorems": True, "theorems": [_theorem()]}
    recovered, repairs = recover_key_labels(payload, IDENTIFICATION)
    assert recovered is payload and repairs == []


def test_per_item_repair_is_isolated():
    good = _theorem(theorem_id="lemma_a")
    bad = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    payload = {"has_provable_theorems": True, "theorems": [good, bad]}
    recovered, repairs = recover_key_labels(payload, IDENTIFICATION)
    assert recovered["theorems"][0] is good
    assert recovered["theorems"][1] == _theorem()
    assert [repair.item_path for repair in repairs] == ["theorems[1]"]


def test_extra_novelty_tier_with_missing_rationale_is_refused():
    item = _rename(_theorem(), "novelty_rationale", "novelty_tier", value="major_novel")
    recovered, repairs = recover_key_labels(item, THEOREM)
    assert repairs == [] and recovered is item


def test_best_match_on_a_present_key_is_refused():
    item = _rename(_theorem(), "novelty_rationale", "expected_novelty", value="Some text.")
    assert recover_key_labels(item, THEOREM)[1] == []


def test_dissimilar_name_is_refused():
    item = _rename(_theorem(), "novelty_rationale", "foo")
    assert recover_key_labels(item, THEOREM)[1] == []


@pytest.mark.parametrize("value", ["", "   ", None, ["a list"], 3, True])
def test_empty_or_wrong_typed_value_is_refused(value):
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    item["novelty_rationally"] = value
    assert recover_key_labels(item, THEOREM)[1] == []


def test_two_unknown_keys_are_refused():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    item = _rename(item, "formal_sketch", "formal_skecth")
    assert recover_key_labels(item, THEOREM)[1] == []


def test_order_crossing_is_refused():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    reordered = {
        "theorem_id": item["theorem_id"],
        "formal_sketch": item["formal_sketch"],
        "statement": item["statement"],
        **{key: value for key, value in item.items() if key not in ("theorem_id", "statement", "formal_sketch")},
    }
    assert recover_key_labels(reordered, THEOREM)[1] == []


def test_unknown_key_outside_the_missing_slot_is_refused():
    item = _theorem()
    del item["novelty_rationale"]
    moved = {"novelty_rationally": "No known bound of this form.", **item}
    assert recover_key_labels(moved, THEOREM)[1] == []


def test_swapped_anchor_values_are_never_relabeled():
    edit = ObjectSchema(
        name="test_edit",
        keys=(
            KeySpec("operation"),
            KeySpec("old_string"),
            KeySpec("new_string"),
            KeySpec("reasoning", repairable=True),
            KeySpec("section_complete", value_type="bool"),
        ),
    )
    item = {
        "operation": "replace",
        "old_string": "a",
        "new_strng": "b",
        "reasoning": "fix",
        "section_complete": False,
    }
    assert recover_key_labels(item, edit)[1] == []


def test_bool_decision_target_is_refused():
    schema = ObjectSchema(
        name="test_phase",
        keys=(
            KeySpec("content", repairable=True),
            KeySpec("reasoning", repairable=True),
            KeySpec("summary", repairable=True),
            KeySpec("section_complete", value_type="bool"),
        ),
    )
    item = {"content": "c", "reasoning": "r", "summary": "s", "section_completed": True}
    assert recover_key_labels(item, schema)[1] == []


def test_protected_source_key_is_refused():
    item = _rename(_theorem(), "novelty_rationale", "decision", value="No known bound.")
    assert recover_key_labels(item, THEOREM)[1] == []


@pytest.mark.parametrize(
    "name",
    ["theorem_id", "proposal_kind", "retry_open_target_id", "decision", "operation", "old_string", "new_string"],
)
def test_protected_keys_can_never_be_declared_repairable(name):
    with pytest.raises(ValueError):
        KeySpec(name, repairable=True)


@pytest.mark.parametrize("value_type", ["bool", "int", "number", "list", "object", "any"])
def test_only_string_keys_can_be_repairable(value_type):
    with pytest.raises(ValueError):
        KeySpec("free_text", value_type=value_type, repairable=True)


def test_duplicate_keys_are_refused():
    raw = (
        '{"theorem_id": "lemma_b", "statement": "S", "formal_sketch": "F",'
        ' "expected_novelty_tier": "mathematical_discovery",'
        ' "prompt_relevance_rationale": "P", "novelty_rationally": "N",'
        ' "why_not_standard_known_result": "W", "statement": "S2"}'
    )
    parsed = parse_json(raw)
    assert parsed.duplicate_keys == ("statement",)
    assert recover_key_labels(parsed, THEOREM)[1] == []


def test_parsed_objects_without_duplicates_recover_and_keep_their_type():
    raw = (
        '{"has_provable_theorems": true, "theorems": [{"theorem_id": "lemma_b", "statement": "S",'
        ' "formal_sketch": "F", "expected_novelty_tier": "mathematical_discovery",'
        ' "prompt_relevance_rationale": "P", "novelty_rationally": "N",'
        ' "why_not_standard_known_result": "W", "retry_open_target_id": ""}]}'
    )
    parsed = parse_json(raw)
    recovered, repairs = recover_key_labels(parsed, IDENTIFICATION)
    assert recovered["theorems"][0]["novelty_rationale"] == "N"
    assert type(recovered["theorems"][0]) is type(parsed["theorems"][0])
    assert len(repairs) == 1


def test_declared_alias_is_not_an_unknown_key_and_stays_untouched():
    item = _theorem()
    item = _rename(item, "theorem_id", "id")
    item = _rename(item, "novelty_rationale", "novelty_rationally")
    recovered, repairs = recover_key_labels(item, THEOREM)
    assert recovered["id"] == "lemma_b" and "theorem_id" not in recovered
    assert recovered["novelty_rationale"] == "No known bound of this form."
    assert repairs[0].item_id == "lemma_b"


def test_extension_keys_are_ignored_and_untouched():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    payload = {
        "has_provable_theorems": True,
        "solution_path_update": {"novelty_rationally": "keep me"},
        "theorems": [item],
    }
    recovered, repairs = recover_key_labels(payload, IDENTIFICATION)
    assert recovered["solution_path_update"] == {"novelty_rationally": "keep me"}
    assert len(repairs) == 1


NOTES = ObjectSchema(
    name="test_notes",
    keys=(
        KeySpec("summary", repairable=True),
        KeySpec("details", repairable=True),
        KeySpec("notes", required=False, repairable=True),
        KeySpec("caveats", required=False, repairable=True),
        KeySpec("conclusion", repairable=True),
        KeySpec("references", repairable=True),
        KeySpec("appendix", required=False, repairable=True),
    ),
)


def test_optional_gap_with_both_neighbours_present_is_repaired():
    item = {"summary": "s", "details": "d", "notse": "n", "caveats": "c", "conclusion": "x", "references": "r"}
    recovered, repairs = recover_key_labels(item, NOTES)
    assert list(recovered) == ["summary", "details", "notes", "caveats", "conclusion", "references"]
    assert repairs[0].to_key == "notes"


def test_optional_gap_with_a_missing_neighbour_is_refused():
    item = {"summary": "s", "details": "d", "notse": "n", "conclusion": "x", "references": "r"}
    assert recover_key_labels(item, NOTES)[1] == []


def test_optional_last_key_has_no_second_neighbour_and_is_refused():
    item = {"summary": "s", "details": "d", "notes": "n", "caveats": "c", "conclusion": "x", "references": "r", "apendix": "a"}
    assert recover_key_labels(item, NOTES)[1] == []


def test_fewer_than_three_anchors_is_refused():
    small = ObjectSchema(
        name="test_small",
        keys=(
            KeySpec("statement", repairable=True),
            KeySpec("novelty_rationale", repairable=True),
            KeySpec("reasoning", repairable=True),
        ),
    )
    item = {"statement": "s", "novelty_rationally": "n", "reasoning": "r"}
    assert recover_key_labels(item, small)[1] == []


def test_two_missing_required_keys_are_refused():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    del item["statement"]
    assert recover_key_labels(item, THEOREM)[1] == []


def test_repair_record_is_an_output_repair():
    item = _rename(_theorem(), "novelty_rationale", "novelty_rationally")
    _recovered, repairs = recover_key_labels(item, THEOREM)
    assert isinstance(repairs[0], OutputRepair)
    assert repairs[0].as_payload() == {
        "kind": "key_label",
        "item_path": "",
        "item_id": "lemma_b",
        "from_key": "novelty_rationally",
        "to_key": "novelty_rationale",
    }
