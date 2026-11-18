"""Guarded visible syntax normalizations in the shared model-JSON parser."""
import json

import pytest

from backend.shared.json_parser import JsonObject, parse_json, sanitize_json_response
from backend.shared.model_output_contract import OutputRepair

GEMMA_LINE = 's     "why_not_standard_known_result": "Not a textbook lemma.",'


def _gemma_payload(stray_line: str = GEMMA_LINE) -> str:
    return (
        "{\n"
        '  "has_provable_theorems": true,\n'
        '  "theorems": [\n'
        "    {\n"
        '      "theorem_id": "lemma_b",\n'
        '      "statement": "Every x satisfies P(x).",\n'
        '      "novelty_rationale": "New bound.",\n'
        f"{stray_line}\n"
        '      "retry_open_target_id": ""\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def test_exact_gemma_stray_line_is_dropped_and_reported():
    repairs: list = []
    result = parse_json(_gemma_payload(), repairs=repairs)
    theorem = result["theorems"][0]
    assert theorem["why_not_standard_known_result"] == "Not a textbook lemma."
    assert theorem["retry_open_target_id"] == ""
    assert repairs == [
        OutputRepair(kind="stray_characters", item_path="theorems[0]", removed_chars=1)
    ]


def test_gemma_stray_line_inside_think_and_markdown_wrappers():
    wrapped = "<think>planning the list</think>\n```json\n" + _gemma_payload() + "\n```"
    repairs: list = []
    result = parse_json(wrapped, repairs=repairs)
    assert result["theorems"][0]["theorem_id"] == "lemma_b"
    assert [repair.kind for repair in repairs] == ["stray_characters"]


def test_allow_repairs_false_leaves_the_text_invalid():
    with pytest.raises(json.JSONDecodeError):
        parse_json(_gemma_payload(), allow_repairs=False)
    sanitized = sanitize_json_response(_gemma_payload(), allow_repairs=False)
    with pytest.raises(json.JSONDecodeError):
        json.loads(sanitized)


def test_sanitize_returns_repaired_text_and_fills_repairs():
    repairs: list = []
    sanitized = sanitize_json_response(_gemma_payload(), repairs=repairs)
    assert json.loads(sanitized)["theorems"][0]["statement"] == "Every x satisfies P(x)."
    assert len(repairs) == 1


def test_valid_json_is_never_repaired():
    repairs: list = []
    text = '{"a": "x s y", "b": [1, 2], "c": {"d": null}}'
    assert parse_json(text, repairs=repairs) == {"a": "x s y", "b": [1, 2], "c": {"d": None}}
    assert repairs == []


def test_characters_inside_strings_are_never_touched():
    text = '{"a": "x",\n "b": "keep s  \\"q\\" and , ] here",\n "c": "z",}'
    repairs: list = []
    result = parse_json(text, repairs=repairs)
    assert result["b"] == 'keep s  "q" and , ] here'
    assert [repair.kind for repair in repairs] == ["trailing_comma"]


@pytest.mark.parametrize(
    "stray",
    [
        "-",  # could be a sign
        ".",
        "7",
        "s7",
        "\\",
        "abcd",  # longer than 3
    ],
)
def test_forbidden_or_long_stray_runs_are_not_removed(stray):
    text = f'{{\n  "a": "x",\n{stray}   "b": "y"\n}}'
    with pytest.raises(json.JSONDecodeError):
        parse_json(text)


def test_stray_needs_whitespace_before_the_next_key():
    with pytest.raises(json.JSONDecodeError):
        parse_json('{\n  "a": "x",\n  s"b": "y"\n}')


@pytest.mark.parametrize(
    "text",
    [
        '{"a": s "x"}',  # value position
        '{"a": "x", "b": [1, s "y"]}',  # array element position
        '["x", s "y"]',
        '{"x": - 7}',  # never strip a sign in value position
        '{"x": !false}',
    ],
)
def test_value_and_array_positions_are_never_repaired(text):
    with pytest.raises(json.JSONDecodeError):
        parse_json(text)


def test_stray_after_complete_member_without_comma_is_refused():
    with pytest.raises(json.JSONDecodeError):
        parse_json('{\n  "a": "x"\ns  "b": "y"\n}')


def test_missing_comma_and_single_quotes_remain_failures():
    with pytest.raises(json.JSONDecodeError):
        parse_json('{"a": "x" "b": "y"}')
    with pytest.raises(json.JSONDecodeError):
        parse_json("{'a': 'x', 'b': 'y'}")


def test_trailing_commas_in_objects_and_arrays():
    repairs: list = []
    result = parse_json('{"a": [1, 2, ], "b": {"c": "d",},}', repairs=repairs)
    assert result == {"a": [1, 2], "b": {"c": "d"}}
    assert [(repair.kind, repair.item_path) for repair in repairs] == [
        ("trailing_comma", "a"),
        ("trailing_comma", "b"),
        ("trailing_comma", ""),
    ]


def test_double_comma_is_not_a_trailing_comma():
    with pytest.raises(json.JSONDecodeError):
        parse_json('{"a": [1, , ], "b": "c"}')


def test_repair_count_is_capped():
    members = ",\n".join(f'x  "k{index}": "v"' for index in range(6))
    with pytest.raises(json.JSONDecodeError):
        parse_json("{\n" + members + "\n}")
    five = ",\n".join(f'x  "k{index}": "v"' for index in range(5))
    repairs: list = []
    assert len(parse_json("{\n" + five + "\n}", repairs=repairs)) == 5
    assert len(repairs) == 5


@pytest.mark.parametrize(
    "text",
    [
        '{"a": "x",\ns  "b": "y"',  # unclosed brace
        '{"a": "x",\ns  "b": "unterminated',  # open string
        '{"a": "x",\ns  "b": [1, 2}',
    ],
)
def test_truncated_output_still_raises_and_is_never_repaired(text):
    repairs: list = []
    with pytest.raises(ValueError):
        parse_json(text, repairs=repairs)
    assert repairs == []


def test_newline_spam_truncation_still_raises():
    with pytest.raises(ValueError, match="truncated"):
        sanitize_json_response('{"a": "x",\ns  "b": "y' + "\\n" * 12)


def test_first_of_several_objects_is_used():
    assert parse_json('{"a": "first"} {"a": "second"}') == {"a": "first"}


def test_top_level_array_returns_first_object():
    assert parse_json('[{"a": "x"}, {"a": "y"}]') == {"a": "x"}


def test_parse_json_records_duplicate_keys_but_behaves_like_dict():
    result = parse_json('{"a": "1", "b": {"c": 1, "c": 2}, "a": "3"}')
    assert isinstance(result, JsonObject) and isinstance(result, dict)
    assert result == {"a": "3", "b": {"c": 2}}
    assert list(result) == ["a", "b"]
    assert result.duplicate_keys == ("a",)
    assert result["b"].duplicate_keys == ("c",)
    assert json.loads(json.dumps(result)) == {"a": "3", "b": {"c": 2}}


def test_objects_without_duplicates_report_none():
    result = parse_json('{"a": {"b": "c"}, "d": [ {"e": 1} ]}')
    assert result.duplicate_keys == ()
    assert result["d"][0].duplicate_keys == ()


@pytest.mark.parametrize(
    "text",
    [
        # A member the model commented out must never become a real member.
        '{"theorems": [{"theorem_id": "a", "statement": "x",\n'
        '  // "retry_open_target_id": "open_7"\n'
        "}], \"has_provable_theorems\": true}",
        '{"has_provable_theorems": true,\n # "proposal_kind": "master_proof"\n}',
        '{"has_provable_theorems": true,\n /* "proposal_kind": "master_proof" */\n}',
        '{"has_provable_theorems": true,\n * "proposal_kind": "master_proof"\n}',
    ],
)
def test_commented_out_members_are_never_revived(text):
    repairs: list = []
    with pytest.raises(json.JSONDecodeError):
        parse_json(text, repairs=repairs)
    assert repairs == []


def test_repairs_run_only_when_the_caller_collects_them():
    # A caller that cannot publish repairs keeps strict parsing.
    with pytest.raises(json.JSONDecodeError):
        parse_json(_gemma_payload())
    with pytest.raises(json.JSONDecodeError):
        parse_json('{"a": "x", "b": "y",}')
    with pytest.raises(json.JSONDecodeError):
        json.loads(sanitize_json_response(_gemma_payload()))
    repairs: list = []
    assert parse_json('{"a": "x", "b": "y",}', repairs=repairs) == {"a": "x", "b": "y"}
    assert [repair.kind for repair in repairs] == ["trailing_comma"]
