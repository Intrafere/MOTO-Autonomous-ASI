import json

import pytest

from backend.shared.translation.contracts import (
    FEEDBACK_MAX_CHARS,
    PROBE_MAX_LANGUAGES,
    PROBE_NAME_MAX_CHARS,
    REASONING_MAX_CHARS,
    TRANSLATOR_NOTES_MAX_CHARS,
    AttemptDecision,
    TranslationContractError,
    TranslatorAttempt,
    TranslatorEdit,
    attempt_from_record,
    attempt_record,
    occurrence_texts,
    parse_probe_response,
    parse_translator_attempt,
    parse_validator_decisions,
    precheck_attempt,
)
from backend.shared.translation.segmentation import build_document_plan, build_unit_views


RAW_MARKER = "RAW_MODEL_OUTPUT_MARKER_7f3a"

LONG_SOURCE = (
    "This closing paragraph is long enough to trigger the length ratio check, because it is "
    "clearly over eighty characters."
)
DOCUMENT = (
    "## Introduction\n\n"
    "We bound $a$ by $b$ for every prime $p$.\n\n"
    "The second paragraph uses $c$ only.\n\n"
    "- A bullet with $d$.\n\n"
    f"{LONG_SOURCE}"
)
PLAN = build_document_plan("document", [("body", DOCUMENT)])
VIEWS = build_unit_views(PLAN)
# Unit-local finals for U1..U3; U3's own [[M1]] is the document-wide [[M4]].
FINALS = {
    0: "## Introducción",
    1: "Acotamos [[M1]] por [[M2]] para todo primo [[M3]].",
    2: "El segundo párrafo usa solo [[M1]].",
}
CURSOR = 3  # U4
MAX_END = 4  # U5
U4_OK = "- Una viñeta con [[M5]]."
U5_OK = (
    "Este párrafo final es lo bastante largo para activar la comprobación de longitud, porque "
    "supera claramente los ochenta caracteres."
)


def _translate(*units, edits=(), notes=""):
    return TranslatorAttempt(
        action="translate", units=tuple(units), edits=tuple(edits), translator_notes=notes
    )


def _precheck(attempt, *, finals=FINALS, part_state=None, max_end=MAX_END, visible=None):
    return precheck_attempt(VIEWS, finals, part_state, attempt, max_end, plan=PLAN, visible=visible)


def _part(attempts=(), *, start=CURSOR, end=MAX_END, locked=False):
    return {"start": start, "end": end, "locked": locked, "attempts": list(attempts)}


def _record(number, status):
    return {"number": number, "status": status}


def test_fixture_views_use_document_wide_tokens():
    assert [view.number for view in VIEWS] == [1, 2, 3, 4, 5]
    assert VIEWS[2].local_tokens == ("[[M1]]",) and VIEWS[2].global_tokens == ("[[M4]]",)
    assert VIEWS[3].global_tokens == ("[[M5]]",)


# --- parse_translator_attempt --------------------------------------------------------------


def test_translate_form_parses_in_response_order():
    raw = json.dumps(
        {
            "action": "translate",
            "units": [
                {"unit": 4, "translation": "  - Una viñeta con [[M5]].  "},
                {"unit": "5", "translation": "Final."},
            ],
            "edits": [{"unit": " 3 ", "find": " usa ", "replace": " emplea "}],
            "translator_notes": "  Used 'emplea'.  ",
        },
        ensure_ascii=False,
    )
    assert parse_translator_attempt(raw) == TranslatorAttempt(
        action="translate",
        units=((4, "- Una viñeta con [[M5]]."), (5, "Final.")),
        edits=(TranslatorEdit(unit=3, find=" usa ", replace=" emplea "),),
        keep_attempts=(),
        translator_notes="Used 'emplea'.",
    )


def test_translate_form_list_wrapped_and_wrapped_in_reasoning():
    payload = {"action": " Translate ", "units": [{"unit": 1, "translation": "Hola"}]}
    assert parse_translator_attempt(json.dumps([payload])).units == ((1, "Hola"),)
    raw = "<think>plan</think>\n```json\n" + json.dumps(payload) + "\n```"
    assert parse_translator_attempt(raw).action == "translate"


def test_translate_notes_and_edits_are_optional():
    attempt = parse_translator_attempt(
        json.dumps({"action": "translate", "units": [{"unit": 1, "translation": "Hola"}], "edits": None})
    )
    assert attempt.edits == () and attempt.translator_notes == "" and attempt.keep_attempts == ()
    coerced = parse_translator_attempt(
        json.dumps(
            {
                "action": "translate",
                "units": [{"unit": 1, "translation": "Hola"}],
                "translator_notes": "n" * (TRANSLATOR_NOTES_MAX_CHARS + 50),
                "keep_attempts": [],
            }
        )
    )
    assert coerced.translator_notes == "n" * TRANSLATOR_NOTES_MAX_CHARS


def test_translate_empty_unit_text_parses_for_the_precheck():
    attempt = parse_translator_attempt(
        json.dumps({"action": "translate", "units": [{"unit": 4, "translation": "   "}]})
    )
    assert attempt.units == ((4, ""),)
    assert _precheck(attempt)[0] == "U4: Your translation is empty. Translate the complete unit."


@pytest.mark.parametrize("action", ["translated", "", None, 1, "improve"])
def test_bad_or_missing_action(action):
    payload = {"units": [{"unit": 1, "translation": "x"}]}
    if action is not None:
        payload["action"] = action
    with pytest.raises(TranslationContractError) as excinfo:
        parse_translator_attempt(json.dumps(payload))
    assert str(excinfo.value) == 'action must be "translate" or "keep"'


@pytest.mark.parametrize("units", [[], None, "U1: text", {"unit": 1, "translation": "x"}])
def test_translate_needs_a_non_empty_units_list(units):
    payload = {"action": "translate"}
    if units is not None:
        payload["units"] = units
    with pytest.raises(TranslationContractError, match='"units" must be a non-empty JSON list'):
        parse_translator_attempt(json.dumps(payload))


@pytest.mark.parametrize(
    "unit,match",
    [
        ("text", 'entry 1 of "units" must be a JSON object'),
        ({"translation": "x"}, 'needs a "unit" number'),
        ({"unit": 0, "translation": "x"}, 'needs a "unit" number'),
        ({"unit": True, "translation": "x"}, 'needs a "unit" number'),
        ({"unit": 1.5, "translation": "x"}, 'needs a "unit" number'),
        ({"unit": "one", "translation": "x"}, 'needs a "unit" number'),
        ({"unit": 1}, "translation of unit U1 must be a JSON string"),
        ({"unit": 1, "translation": ["x"]}, "translation of unit U1 must be a JSON string"),
    ],
)
def test_translate_bad_unit_entries(unit, match):
    with pytest.raises(TranslationContractError, match=match):
        parse_translator_attempt(json.dumps({"action": "translate", "units": [unit]}))


def test_translate_duplicate_units():
    raw = json.dumps(
        {
            "action": "translate",
            "units": [{"unit": 4, "translation": "a"}, {"unit": "4", "translation": "b"}],
        }
    )
    with pytest.raises(TranslationContractError) as excinfo:
        parse_translator_attempt(raw)
    assert str(excinfo.value) == 'unit U4 appears more than once in "units"; give each unit once'


@pytest.mark.parametrize(
    "edits,match",
    [
        ("U3: x -> y", '"edits" must be a JSON list of edits'),
        ({"unit": 3, "find": "x", "replace": "y"}, '"edits" must be a JSON list of edits'),
        (["x"], 'edit 1 must be a JSON object with "unit", "find", and "replace"'),
        ([{"find": "x", "replace": "y"}], 'edit 1 needs a "unit" number'),
        ([{"unit": 3, "replace": "y"}], 'edit 1 must give "find" and "replace" as JSON strings'),
        ([{"unit": 3, "find": "x", "replace": 7}], 'edit 1 must give "find" and "replace"'),
    ],
)
def test_translate_bad_edits_shape(edits, match):
    payload = {"action": "translate", "units": [{"unit": 4, "translation": "x"}], "edits": edits}
    with pytest.raises(TranslationContractError, match=match):
        parse_translator_attempt(json.dumps(payload))


def test_edit_text_is_kept_exactly():
    raw = json.dumps(
        {
            "action": "translate",
            "units": [{"unit": 4, "translation": "x"}],
            "edits": [{"unit": 3, "find": "  ", "replace": ""}],
        }
    )
    assert parse_translator_attempt(raw).edits == (TranslatorEdit(3, "  ", ""),)


def test_translate_must_not_name_keep_attempts():
    payload = {"action": "translate", "units": [{"unit": 1, "translation": "x"}], "keep_attempts": [1]}
    with pytest.raises(TranslationContractError, match='must not name keep_attempts'):
        parse_translator_attempt(json.dumps(payload))


def test_keep_form_parses_and_dedupes():
    raw = json.dumps({"action": "keep", "keep_attempts": [2, "1", 2], "translator_notes": "Both fine."})
    assert parse_translator_attempt(raw) == TranslatorAttempt(
        action="keep", keep_attempts=(2, 1), translator_notes="Both fine."
    )


@pytest.mark.parametrize("payload", [{"action": "keep", "keep_attempts": []}, {"action": "keep"}])
def test_keep_with_empty_or_missing_list_parses_for_the_precheck(payload):
    attempt = parse_translator_attempt(json.dumps(payload))
    assert attempt.keep_attempts == ()
    assert _precheck(attempt, part_state=_part([_record(1, "accepted")])) == [
        'Your "keep" response names no attempts. Name the earlier attempt or attempts to keep in '
        '"keep_attempts", or use the "translate" form.'
    ]


@pytest.mark.parametrize(
    "keep_attempts,match",
    [
        ("1", '"keep_attempts" must be a JSON list'),
        ([0], "only attempt numbers of 1 or more"),
        (["first"], "only attempt numbers of 1 or more"),
        ([True], "only attempt numbers of 1 or more"),
    ],
)
def test_keep_bad_attempt_numbers(keep_attempts, match):
    with pytest.raises(TranslationContractError, match=match):
        parse_translator_attempt(json.dumps({"action": "keep", "keep_attempts": keep_attempts}))


@pytest.mark.parametrize(
    "extra", [{"units": [{"unit": 1, "translation": "x"}]}, {"edits": [{"unit": 1, "find": "a", "replace": "b"}]}]
)
def test_keep_must_not_carry_new_text(extra):
    payload = {"action": "keep", "keep_attempts": [1], **extra}
    with pytest.raises(TranslationContractError, match='a "keep" response must not contain units or edits'):
        parse_translator_attempt(json.dumps(payload))


def test_keep_tolerates_empty_units_and_edits():
    raw = json.dumps({"action": "keep", "keep_attempts": [1], "units": [], "edits": []})
    assert parse_translator_attempt(raw).keep_attempts == (1,)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (f"I cannot answer this as JSON because {RAW_MARKER}.", "no JSON object"),
        ('{"action": "translate", "units": [{"unit": 1, "translation": "' + RAW_MARKER, "incomplete"),
        ("", "empty or too short"),
        ('{"action": "keep" "x" ' + RAW_MARKER + "}", "not valid JSON"),
        (json.dumps(["just a string " + RAW_MARKER]), "single JSON object"),
    ],
)
def test_malformed_json_messages_are_safe(raw, expected):
    with pytest.raises(TranslationContractError) as excinfo:
        parse_translator_attempt(raw)
    assert expected in str(excinfo.value)
    assert RAW_MARKER not in str(excinfo.value)


def test_contract_messages_never_echo_model_text():
    raw = json.dumps({"action": RAW_MARKER, "units": [{"unit": RAW_MARKER, "translation": RAW_MARKER}]})
    with pytest.raises(TranslationContractError) as excinfo:
        parse_translator_attempt(raw)
    assert RAW_MARKER not in str(excinfo.value)


def test_latex_backslashes_survive():
    escaped = r'{"action": "translate", "units": [{"unit": 1, "translation": "Le \\textbf{gras} [[M1]]."}]}'
    assert parse_translator_attempt(escaped).units == ((1, r"Le \textbf{gras} [[M1]]."),)
    unescaped = r'{"action": "translate", "units": [{"unit": 1, "translation": "Le \textbf{gras} \frac{a}{b}."}]}'
    assert parse_translator_attempt(unescaped).units == ((1, r"Le \textbf{gras} \frac{a}{b}."),)


@pytest.mark.parametrize(
    "translation",
    [
        "Acotamos el error.\nusando esta cota, la afirmación se sigue.",
        "Obtenemos\n[[M1]]\nusando el lema y otro caso.",
        "Primera parte.\notro renglón.\negy sor.\negalitarian.\nequal sides.",
    ],
)
def test_json_line_breaks_before_words_like_latex_commands_survive(translation):
    raw = json.dumps({"action": "translate", "units": [{"unit": 1, "translation": translation}]})
    assert parse_translator_attempt(raw).units == ((1, translation),)
    feedback = json.dumps(
        {"decisions": [{"attempt": 1, "decision": "reject", "reasoning": "", "feedback": translation}]}
    )
    assert parse_validator_decisions(feedback, [1])[0].feedback == translation


# --- parse_validator_decisions -------------------------------------------------------------


def _decisions(*items):
    return json.dumps({"decisions": list(items)}, ensure_ascii=False)


def _decision(attempt, decision, reasoning="Looks right.", feedback=""):
    return {"attempt": attempt, "decision": decision, "reasoning": reasoning, "feedback": feedback}


def test_decisions_in_listed_order():
    raw = _decisions(
        _decision(4, "ACCEPT "),
        _decision("6", "Reject", "Wrong term.", "Terminology: use 'borne'."),
    )
    assert parse_validator_decisions(raw, [4, 6]) == (
        AttemptDecision(attempt=4, decision="accept", reasoning="Looks right.", feedback=""),
        AttemptDecision(attempt=6, decision="reject", reasoning="Wrong term.", feedback="Terminology: use 'borne'."),
    )


def test_decisions_list_wrapped():
    raw = json.dumps([{"decisions": [_decision(2, "accept")]}])
    assert parse_validator_decisions(raw, (2,))[0].decision == "accept"


@pytest.mark.parametrize("numbers", [[4], [4, 6, 7]])
def test_decisions_exact_count(numbers):
    raw = _decisions(_decision(4, "accept"), _decision(6, "accept"))
    with pytest.raises(TranslationContractError, match=f"exactly {len(numbers)} entries"):
        parse_validator_decisions(raw, numbers)


@pytest.mark.parametrize("given", [(6, 4), (4, 5), (4, None), (4, True), (4, 6.0)])
def test_decisions_exact_attempt_order(given):
    raw = _decisions(_decision(given[0], "accept"), _decision(given[1], "accept"))
    with pytest.raises(TranslationContractError, match="must be for attempt"):
        parse_validator_decisions(raw, [4, 6])


@pytest.mark.parametrize("decision", ["maybe", "", None, 1, "accepted"])
def test_decisions_invalid_decision(decision):
    with pytest.raises(TranslationContractError, match='decision for attempt 3 must be "accept" or "reject"'):
        parse_validator_decisions(_decisions(_decision(3, decision)), [3])


def test_rejection_feedback_falls_back_to_reasoning():
    long_reasoning = "Fidelity: the claim is reversed. " * 100
    decision = parse_validator_decisions(
        _decisions(_decision(1, "reject", reasoning=long_reasoning, feedback="  ")), [1]
    )[0]
    assert decision.reasoning == long_reasoning.strip()[:REASONING_MAX_CHARS].strip()
    assert decision.feedback == decision.reasoning[:FEEDBACK_MAX_CHARS].strip()


def test_rejection_without_feedback_or_reasoning():
    with pytest.raises(TranslationContractError) as excinfo:
        parse_validator_decisions(_decisions(_decision(2, "reject", reasoning="", feedback="")), [2])
    assert str(excinfo.value) == "feedback is required for rejected attempt 2"


def test_feedback_is_capped():
    decision = parse_validator_decisions(_decisions(_decision(1, "reject", feedback="f" * 5000)), [1])[0]
    assert decision.feedback == "f" * FEEDBACK_MAX_CHARS


@pytest.mark.parametrize("payload", [{"decisions": "accept all"}, {}, {"decisions": ["accept"]}])
def test_decisions_structural_errors(payload):
    with pytest.raises(TranslationContractError):
        parse_validator_decisions(json.dumps(payload), [1])


def test_decisions_malformed_json_message_is_safe():
    with pytest.raises(TranslationContractError) as excinfo:
        parse_validator_decisions(f"Attempt 1 is best. {RAW_MARKER}", [1])
    assert RAW_MARKER not in str(excinfo.value)


def test_decisions_need_attempt_numbers():
    with pytest.raises(ValueError):
        parse_validator_decisions(_decisions(), [])


# --- precheck_attempt: translate -----------------------------------------------------------


def test_valid_attempt_with_edit_passes():
    attempt = _translate((4, U4_OK), (5, U5_OK), edits=[TranslatorEdit(3, "usa solo [[M4]]", "emplea solo [[M4]]")])
    assert _precheck(attempt) == []


def test_part_may_end_before_max_end():
    assert _precheck(_translate((4, U4_OK))) == []


def test_identity_translation_passes():
    attempt = _translate((4, "- A bullet with [[M5]]."), (5, LONG_SOURCE))
    assert _precheck(attempt) == []


def test_finals_with_persisted_string_keys():
    finals = {str(key): value for key, value in FINALS.items()}
    attempt = _translate((4, U4_OK), edits=[TranslatorEdit(3, "usa solo", "emplea solo")])
    assert _precheck(attempt, finals=finals) == []


def test_wrong_start():
    problems = _precheck(_translate((5, U5_OK)))
    assert problems == [
        "Your part must start at unit U4, the first unit that is not translated yet; it starts at U5."
    ]


def test_non_consecutive_units():
    finals = {0: FINALS[0]}
    attempt = _translate((2, "Acotamos [[M1]] por [[M2]] para todo primo [[M3]]."), (4, U4_OK))
    problems = _precheck(attempt, finals=finals)
    assert problems == [
        "The units of a part must be consecutive and in order: after U2 comes U3, but your "
        "response gives U4. Translate each unit of the part exactly once, in order."
    ]


def test_units_out_of_order_are_not_consecutive():
    problems = _precheck(_translate((5, U5_OK), (4, U4_OK)))
    assert any("start at unit U4" in problem for problem in problems)
    assert any("after U5 comes U6, but your response gives U4" in problem for problem in problems)


def test_beyond_max_end():
    problems = _precheck(_translate((4, U4_OK), (5, U5_OK)), max_end=3)
    assert problems == ["Your part reaches U5, but this attempt may end no later than U4. Stop at a natural break by U4."]


def test_unit_beyond_the_document():
    problems = _precheck(_translate((4, U4_OK), (5, U5_OK), (6, "Extra.")))
    assert problems == [
        "Your part reaches U6, but this attempt may end no later than U5. Stop at a natural break by U5."
    ]


def test_locked_range_must_match_exactly():
    locked = _part([_record(1, "accepted")], start=3, end=4, locked=True)
    assert _precheck(_translate((4, U4_OK), (5, U5_OK)), part_state=locked) == []
    problems = _precheck(_translate((4, U4_OK)), part_state=locked)
    assert problems == [
        "This part is units U4 to U5, which already have an accepted translation. Translate exactly "
        "those units, in order, with none added or left out."
    ]
    # The locked range wins over max_end.
    assert _precheck(_translate((4, U4_OK), (5, U5_OK)), part_state=locked, max_end=3) == []


def test_unlocked_part_state_does_not_fix_the_range():
    unlocked = _part([_record(1, "rejected")], start=3, end=4)
    assert _precheck(_translate((4, U4_OK)), part_state=unlocked) == []


def test_missing_placeholder_in_unit():
    problems = _precheck(_translate((4, "- Una viñeta.")))
    assert len(problems) == 1
    assert problems[0].startswith("U4: Placeholder(s) [[M5]] from the source are missing")


def test_repeated_placeholder_in_unit():
    problems = _precheck(_translate((4, "- Una viñeta con [[M5]] y [[M5]].")))
    assert len(problems) == 1
    assert problems[0].startswith("U4: Placeholder(s) [[M5]] appear more than once")


def test_unit_local_token_is_unknown_in_document_view():
    # U4's own token is [[M1]] locally, but the translator must use the document-wide [[M5]].
    problems = _precheck(_translate((4, "- Una viñeta con [[M1]].")))
    assert len(problems) == 2
    assert "U4: Placeholder(s) [[M5]] from the source are missing" in problems[0]
    assert problems[1].startswith("U4: Your translation contains placeholder(s) [[M1]] that do not exist")


def test_paragraph_count_mismatch():
    problems = _precheck(_translate((4, "- Una viñeta\n\ncon [[M5]].")))
    assert problems == [
        "U4: The source has 1 paragraph separated by blank lines; your translation has 2. Translate "
        "each source paragraph as exactly one paragraph, in the same order, and separate paragraphs "
        "with one blank line."
    ]


def test_marker_mismatch():
    problems = _precheck(_translate((4, "Una viñeta con [[M5]].")))
    assert len(problems) == 1
    assert 'paragraph 1 should start with a bullet marker such as "-" but starts with no Markdown marker' in problems[0]


@pytest.mark.parametrize("delimiter", ["<<<SOURCE", "SOURCE>>>", "<<<TRANSLATION", "TRANSLATION>>>"])
def test_prompt_delimiter_leakage(delimiter):
    problems = _precheck(_translate((4, f"- Una viñeta con [[M5]]. {delimiter}")))
    assert len(problems) == 1 and delimiter in problems[0] and "prompt delimiter" in problems[0]


@pytest.mark.parametrize("markup", ["<<U4>> ", "[KEPT AS IS] "])
def test_view_markup_leakage(markup):
    problems = _precheck(_translate((4, f"- Una viñeta con [[M5]]. {markup}")))
    assert len(problems) == 1 and "document view markup" in problems[0]


def test_length_ratio():
    too_short = _precheck(_translate((4, U4_OK), (5, "Corto.")))
    assert len(too_short) == 1 and too_short[0].startswith("U5: ") and "incomplete or summarized" in too_short[0]
    too_long = _precheck(_translate((4, U4_OK), (5, "Demasiado largo. " * 60)))
    assert len(too_long) == 1 and "additions" in too_long[0]
    # Units shorter than the ratio threshold skip the ratio check.
    assert _precheck(_translate((4, "- Una viñeta con [[M5]] " + "y más " * 40 + "."))) == []


def test_problems_are_reported_together():
    attempt = _translate((5, "Corto <<<TRANSLATION"), edits=[TranslatorEdit(9, "a", "b")])
    problems = _precheck(attempt)
    assert any("start at unit U4" in problem for problem in problems)
    assert any(problem.startswith("U5: ") and "prompt delimiter" in problem for problem in problems)
    assert any(problem.startswith("Edit 1 (U9)") for problem in problems)


# --- precheck_attempt: edits ---------------------------------------------------------------


def _edit_problems(*edits, finals=FINALS, part_state=None, visible=None):
    return _precheck(_translate((4, U4_OK), edits=edits), finals=finals, part_state=part_state, visible=visible)


def test_edit_to_a_unit_in_the_part():
    problems = _edit_problems(TranslatorEdit(4, "viñeta", "viñetas"))
    assert problems == [
        "Edit 1 (U4) targets a unit of this attempt. Put the correction in that unit's translation "
        "instead of an edit."
    ]


def test_edit_to_a_locked_unit_outside_the_response():
    locked = _part([_record(1, "accepted")], start=3, end=4, locked=True)
    attempt = _translate((4, U4_OK), edits=[TranslatorEdit(5, "a", "b")])
    problems = _precheck(attempt, part_state=locked)
    assert any("Edit 1 (U5) targets a unit of this attempt" in problem for problem in problems)


def test_edit_to_an_untranslated_unit():
    problems = _edit_problems(TranslatorEdit(5, "final", "último"))
    assert problems == [
        "Edit 1 (U5) targets a unit that is not translated yet. Edit only units that already have a "
        "translation."
    ]


def test_edit_to_a_unit_that_does_not_exist():
    assert _edit_problems(TranslatorEdit(40, "a", "b")) == [
        "Edit 1 (U40) names a unit that does not exist in the document."
    ]


def test_edit_to_a_unit_outside_the_window():
    edit = TranslatorEdit(1, "Introducción", "Introducción general")
    assert _edit_problems(edit, visible={1, 2, 3, 4}) == [
        "Edit 1 (U1) targets a unit that is not shown in the translation so far. Edit only units "
        "shown there."
    ]
    assert _edit_problems(edit, visible={0, 3, 4}) == []


def test_edit_with_empty_find():
    problems = _edit_problems(TranslatorEdit(2, "", "x"))
    assert problems == [
        'Edit 1 (U2) has an empty "find". Copy the exact text to replace from that unit\'s current '
        "translation."
    ]


def test_edit_find_equals_replace():
    problems = _edit_problems(TranslatorEdit(2, "Acotamos", "Acotamos"))
    assert problems == [
        'Edit 1 (U2) replaces "find" with identical text. Remove the edit or give the corrected text '
        'in "replace".'
    ]


def test_edit_find_not_found_uses_document_wide_tokens():
    # The stored final has U3's local [[M1]]; the translator sees and must quote [[M4]].
    problems = _edit_problems(TranslatorEdit(3, "solo [[M1]]", "únicamente [[M1]]"))
    assert problems == [
        'Edit 1 (U3): the "find" text does not appear in the current translation of U3. Copy it '
        "exactly from that unit, including spacing and punctuation."
    ]
    assert _edit_problems(TranslatorEdit(3, "solo [[M4]]", "únicamente [[M4]]")) == []


def test_edit_find_not_unique_counts_overlaps():
    finals = {**FINALS, 2: "aaa [[M1]]."}
    problems = _edit_problems(TranslatorEdit(3, "aa", "b"), finals=finals)
    assert problems == [
        'Edit 1 (U3): the "find" text appears 2 times in the current translation of U3. Include more '
        "of the surrounding text so it matches exactly once."
    ]


def test_edit_breaking_placeholders():
    problems = _edit_problems(TranslatorEdit(2, " por [[M2]]", ""))
    assert len(problems) == 1
    assert problems[0].startswith("Your edits to U2 would break that unit: Placeholder(s) [[M2]]")
    unknown = _edit_problems(TranslatorEdit(3, "[[M4]]", "[[M1]]"))
    assert len(unknown) == 2
    assert all(problem.startswith("Your edits to U3 would break that unit: ") for problem in unknown)
    assert "[[M4]]" in unknown[0] and "[[M1]]" in unknown[1]


def test_edited_units_get_the_structural_checks_of_a_translated_unit():
    problems = _edit_problems(TranslatorEdit(3, "solo", "solo\n\n<<U9>> ## SOURCE>>>"))
    assert len(problems) == 3
    assert all(problem.startswith("Your edits to U3 would break that unit: ") for problem in problems)
    assert "The source has 1 paragraph" in problems[0]
    assert "prompt delimiter(s) SOURCE>>>" in problems[1]
    assert "document view markup" in problems[2]
    # Markers and the length ratio are checked against the edited unit's own source.
    [marker] = _edit_problems(TranslatorEdit(3, "El segundo", "- El segundo"))
    assert marker.startswith("Your edits to U3 would break that unit: Paragraph-start Markdown markers")
    [ratio] = _edit_problems(TranslatorEdit(5, U5_OK, "Corto."), finals={**FINALS, 4: U5_OK})
    assert ratio.startswith("Your edits to U5 would break that unit: Your translation is only")
    assert _edit_problems(TranslatorEdit(1, "Introducción", "Introducción general")) == []


def test_sequential_edits_to_one_unit():
    first = TranslatorEdit(2, "Acotamos", "Limitamos")
    assert _edit_problems(first, TranslatorEdit(2, "Limitamos [[M1]]", "Acotamos [[M1]] así")) == []
    # The second edit sees the text left by the first, so the old wording is gone.
    problems = _edit_problems(first, TranslatorEdit(2, "Acotamos", "Restringimos"))
    assert problems == [
        'Edit 2 (U2): the "find" text does not appear in the current translation of U2. Copy it '
        "exactly from that unit, including spacing and punctuation."
    ]


# --- precheck_attempt: keep ----------------------------------------------------------------


def _keep(*numbers):
    return TranslatorAttempt(action="keep", keep_attempts=tuple(numbers), translator_notes="")


ATTEMPTS = [
    _record(1, "rejected"),
    _record(2, "accepted"),
    _record(3, "check_failed"),
    _record(4, "kept"),
]


def test_keep_accepted_and_rejected_attempts_pass():
    assert _precheck(_keep(2), part_state=_part(ATTEMPTS)) == []
    assert _precheck(_keep(1, 2), part_state=_part(ATTEMPTS)) == []


def test_keep_unknown_check_failed_and_kept_attempts():
    problems = _precheck(_keep(9, 3, 4, 2), part_state=_part(ATTEMPTS))
    assert problems == [
        "Attempt 9 does not exist for this part. Keep only attempts shown under EARLIER ATTEMPTS AT "
        "THIS PART.",
        'Attempt 3 failed the automatic checks, so it cannot be kept. Keep an accepted or rejected '
        'attempt, or use the "translate" form.',
        "Attempt 4 was itself a keep response and has no translation to keep. Name the accepted or "
        "rejected attempt instead.",
    ]


def test_keep_without_a_part():
    assert _precheck(_keep(1), part_state=None)[0].startswith("Attempt 1 does not exist for this part.")


# --- attempt records -----------------------------------------------------------------------


def test_attempt_record_stores_unit_local_text_and_round_trips():
    attempt = _translate((4, U4_OK), (5, U5_OK), edits=[TranslatorEdit(3, "[[M4]]", "[[M4]] ")], notes="n")
    record = attempt_record(attempt, VIEWS, number=2, status="rejected")
    assert record == {
        "number": 2,
        "action": "translate",
        "units": {"3": "- Una viñeta con [[M1]].", "4": U5_OK},
        "edits": [{"occurrence": 2, "find": "[[M4]]", "replace": "[[M4]] "}],
        "keep_attempts": [],
        "notes": "n",
        "status": "rejected",
        "feedback": "",
        "reasoning": "",
        "accepted_order": None,
    }
    assert attempt_from_record(json.loads(json.dumps(record)), VIEWS) == attempt


def test_check_failed_record_keeps_text_as_submitted():
    attempt = _translate((4, "- Una viñeta con [[M1]]."), (9, "fuera"))
    record = attempt_record(attempt, VIEWS, number=1, status="check_failed")
    assert record["units"] == {"3": "- Una viñeta con [[M1]].", "8": "fuera"}
    assert attempt_from_record(record, VIEWS) == attempt


def test_attempt_record_rejects_units_outside_the_document():
    with pytest.raises(ValueError):
        attempt_record(_translate((9, "x")), VIEWS, number=1, status="accepted")


def test_keep_record_round_trips():
    attempt = TranslatorAttempt(action="keep", keep_attempts=(2, 1), translator_notes="Fine.")
    record = attempt_record(attempt, VIEWS, number=5, status="kept")
    assert record["keep_attempts"] == [2, 1] and record["units"] == {} and record["edits"] == []
    assert attempt_from_record(record, VIEWS) == attempt


def test_occurrence_texts():
    assert occurrence_texts({"3": "a", 4: "b", "5": None}) == {3: "a", 4: "b"}


# --- parse_probe_response ------------------------------------------------------------------


def test_probe_sanitizes_dedupes_and_skips_junk():
    raw = json.dumps(
        {
            "languages": [
                "  English ",
                "Spanish\u0000",
                "Mandarin\n  Chinese",
                "english",
                "SPANISH",
                "",
                "   ",
                "123",
                "---",
                7,
                None,
                {"name": "French"},
                "L" * 200,
                "Modern Standard Arabic",
            ]
        }
    )
    assert parse_probe_response(raw) == [
        "English",
        "Spanish",
        "Mandarin Chinese",
        "L" * PROBE_NAME_MAX_CHARS,
        "Modern Standard Arabic",
    ]


def test_probe_caps_item_count():
    raw = json.dumps({"languages": [f"Language {index}" for index in range(PROBE_MAX_LANGUAGES + 50)]})
    names = parse_probe_response(raw)
    assert len(names) == PROBE_MAX_LANGUAGES
    assert names[-1] == f"Language {PROBE_MAX_LANGUAGES - 1}"


def test_probe_empty_list_is_valid():
    assert parse_probe_response(json.dumps({"languages": []})) == []


def test_probe_keeps_json_line_breaks_before_latex_like_words():
    assert parse_probe_response(json.dumps({"languages": ["Spanish\nnu"]})) == ["Spanish nu"]


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"languages": "English, Spanish"}),
        json.dumps({"langs": ["English"]}),
        f"English and Spanish {RAW_MARKER}",
    ],
)
def test_probe_contract_errors(raw):
    with pytest.raises(TranslationContractError) as excinfo:
        parse_probe_response(raw)
    assert RAW_MARKER not in str(excinfo.value)
