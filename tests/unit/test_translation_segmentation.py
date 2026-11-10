import hashlib
import json
import random
import re

import pytest

from backend.autonomous.memory.paper_library import PaperLibrary
from backend.autonomous.memory.paper_model_tracker import PaperModelTracker
from backend.shared.paper_proofs import (
    THEOREMS_APPENDIX_END,
    THEOREMS_APPENDIX_START,
    analyze_paper_content,
)
from backend.shared.translation import segmentation
from backend.shared.translation.segmentation import (
    PROFILE_DOCUMENT,
    PROFILE_FIELDS,
    PROOF_TRANSLATION_FIELDS,
    SEGMENT_TARGET_TOKENS,
    SEGMENTATION_VERSION,
    DocumentPlan,
    PlanBlock,
    UnitView,
    assemble_occurrences,
    assemble_parts,
    build_document_plan,
    build_unit_views,
    plan_from_json,
    plan_to_json,
    split_paragraphs,
    to_global,
    to_local,
    unmask,
)
from backend.shared.utils import count_tokens


EMPTY_APPENDIX = "[Theorems appendix - verified Lean 4 theorems not placed inline will appear here]"
LEGACY_HEADER = "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ==="
BRAINSTORM_PROOF_HEADER = "=== PROOFS GENERATED FROM THIS BRAINSTORM (Lean 4 Verified) ==="
PLACEHOLDER_RE = re.compile(r"\[\[(?:MOTO|M)\d+\]\]")

LEAN_ONE = "theorem sum_pos (a b : Nat) (ha : 0 < a) : 0 < a + b := by\n  omega"
LEAN_TWO = "theorem two_lt (n : Nat) (h : 2 ≤ n) : 1 < n := by\n  -- order argument\n  omega"
LEAN_THREE = "theorem legacy_true : True := by\n  trivial"
LEAN_BRAINSTORM = "theorem brainstorm_true : True := by\n  exact True.intro"

PROOF_ONE = {
    "proof_id": "proof_a1",
    "theorem_name": "Positive sums",
    "theorem_statement": "For natural numbers $a, b$ with $0 < a$, the sum $a + b$ is positive.",
    "lean_code": LEAN_ONE,
    "source_type": "paper",
    "source_id": "paper_7",
    "novel": True,
    "novelty_tier": "mathematical_discovery",
}
PROOF_TWO = {
    "proof_id": "proof_a2",
    "theorem_name": "Strict lower bound",
    "theorem_statement": "Every natural number \\(n \\ge 2\\) is strictly greater than one.",
    "lean_code": LEAN_TWO,
    "source_type": "paper",
    "source_id": "paper_7",
    "novel": False,
}
PROOF_THREE = {
    "proof_id": "proof_legacy",
    "theorem_name": "Legacy truth",
    "theorem_statement": "Truth holds.",
    "lean_code": LEAN_THREE,
    "source_type": "brainstorm",
    "source_id": "topic_3",
    "novel": False,
}

PAPER_BODY = (
    "## Abstract\n"
    "\n"
    "We study sums of natural numbers \\cite{knuth1997} and show $a + b > 0$ whenever $a > 0$.\n"
    "\n"
    "## I. Introduction\n"
    "\n"
    "The main identity is\n"
    "$$\n"
    "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}\n"
    "$$\n"
    "and the display form \\[ e^{i\\pi} + 1 = 0 \\] appears in \\(\\mathbb{C}\\). "
    "See https://example.org/notes?id=4 for details.\n"
    "\n"
    "\\begin{equation}\n"
    "f(x) = x^2\n"
    "\\end{equation}\n"
    "\n"
    "\\begin{align*}\n"
    "a &= b \\\\\n"
    "b &= c\n"
    "\\end{align*}\n"
    "\n"
    "Run `lake build` to check the proofs, as described in Section~\\ref{sec:proofs}.\n"
    "\n"
    "- First item with $x$.\n"
    "- Second item.\n"
    "\n"
    "1. Ordered item.\n"
    "\n"
    "> A quoted remark.\n"
    "\n"
    "## II. Conclusion\n"
    "\n"
    "Positive sums stay positive.\n"
    "\n"
    f"{THEOREMS_APPENDIX_START}\n"
    f"{EMPTY_APPENDIX}\n"
    f"{THEOREMS_APPENDIX_END}\n"
    "\n"
    "AI Self-Review and Limitations\n"
    "\n"
    "The argument relies only on standard arithmetic.\n"
)


def _realistic_paper() -> str:
    tracker = PaperModelTracker(user_prompt="Study positive sums", paper_title="Positive Sums")
    for model_id in ("openai/gpt-5", "openai/gpt-5", "deepseek/deepseek-r1"):
        tracker.track_call(model_id)
    tracker.track_wolfram_call("2+2")
    content = PaperLibrary.attach_verified_proofs_to_content(PAPER_BODY, [PROOF_ONE, PROOF_TWO])
    legacy = f"{LEGACY_HEADER}\n\n{PaperLibrary._format_verified_proof_entry(PROOF_THREE)}"
    content = content.replace(
        "AI Self-Review and Limitations", f"{legacy}\n\nAI Self-Review and Limitations", 1
    )
    return (
        tracker.generate_author_attribution()
        + "\n"
        + content
        + tracker.generate_model_credits()
        + "\n"
    )


def _brainstorm_database() -> str:
    separator = "=" * 80
    bodies = (
        "Consider the map $f(n) = n + 1$ on $\\mathbb{N}$.\n\nIt has no fixed point.",
        "## Route\n\n- Bound the sum with \\(a \\le b\\).\n- Conclude positivity.",
        "A counterexample search at https://example.org/search found nothing.",
    )
    sections = [
        f"\n{separator}\nSUBMISSION #{number} | Accepted: 2026-10-0{number}T12:00:00\n{separator}\n\n{body}\n"
        for number, body in enumerate(bodies, start=1)
    ]
    proof_lines = [
        "",
        "",
        BRAINSTORM_PROOF_HEADER,
        "",
        "Proof 1: Truth holds in the brainstorm.",
        "Status: Verified (Novel)",
        "Proof ID: proof_b1",
        "Lean 4 Code:",
        LEAN_BRAINSTORM,
        "---",
    ]
    return "".join(sections) + "\n".join(proof_lines) + "\n"


def _volume(chapter_content: str) -> str:
    lines = [
        "=" * 80,
        "FINAL ANSWER VOLUME",
        "Title: Positive Sums Volume",
        "Generated: 2026-10-31T12:00:00",
        "=" * 80,
        "",
        "TABLE OF CONTENTS",
        "-" * 40,
        "  1. Positive Sums [Existing Paper]",
        "  2. Closing Remarks [Conclusion]",
        "",
        "=" * 80,
        "",
    ]
    chapters = ((1, "Positive Sums", "Existing Paper", chapter_content),
                (2, "Closing Remarks", "Conclusion", "Every positive sum remains positive."))
    for order, title, chapter_type, content in chapters:
        lines.extend(
            [
                "#" * 80,
                f"# CHAPTER {order}: {title}",
                f"# Type: {chapter_type}",
                "#" * 80,
                "",
                content,
                "",
                "",
            ]
        )
    lines.extend(["=" * 80, "END OF VOLUME", "=" * 80])
    return "\n".join(lines)


def _identity(plan: DocumentPlan) -> dict:
    return {segment.source_hash: segment.masked_text for segment in plan.segments}


def _assert_identity(profile: str, parts) -> DocumentPlan:
    plan = build_document_plan(profile, parts)
    assert assemble_parts(plan, _identity(plan)) == dict(parts)
    return plan


def _masked_texts(plan: DocumentPlan) -> list:
    return [segment.masked_text for segment in plan.segments]


def _segment_blocks(plan: DocumentPlan) -> list:
    return [block for _, blocks in plan.parts for block in blocks if block.kind == "segment"]


def _assert_absent_from_segments(plan: DocumentPlan, *needles: str) -> None:
    for needle in needles:
        for masked in _masked_texts(plan):
            assert needle not in masked, f"{needle!r} leaked into a translatable segment"


def _reverse_words(masked: str) -> str:
    """A structure-preserving fake translation: reverse every word, keep placeholders."""
    return re.sub(
        r"(\[\[(?:MOTO|M)\d+\]\])|([A-Za-z]{2,})",
        lambda match: match.group(1) or match.group(2)[::-1],
        masked,
    )


def _lean_lines(*codes: str) -> list:
    return [line for code in codes for line in code.splitlines() if line.strip()]


def _assert_segment_invariants(plan: DocumentPlan) -> None:
    by_hash = {segment.source_hash: segment for segment in plan.segments}
    assert len(by_hash) == len(plan.segments)
    for segment in plan.segments:
        masked = segment.masked_text
        assert masked == masked.strip()
        assert segment.source_hash == hashlib.sha256(
            f"{SEGMENTATION_VERSION}\n{masked}".encode("utf-8")
        ).hexdigest()
        assert tuple(PLACEHOLDER_RE.findall(masked)) == segment.placeholder_tokens
        prefix = "MOTO" if segment.placeholder_tokens and "MOTO" in segment.placeholder_tokens[0] else "M"
        assert segment.placeholder_tokens == tuple(
            f"[[{prefix}{index}]]" for index in range(1, len(segment.placeholder_tokens) + 1)
        )
        assert not re.search(r"\[\[(?:MOTO|M)\d|(?:MOTO|M)\d+\]\]", PLACEHOLDER_RE.sub("", masked))
        assert segment.paragraph_count == len(split_paragraphs(masked)) == 1
        assert len(segment.paragraph_markers) == 1
        assert segment.token_count == count_tokens(masked)
        assert not any("\ue000" <= character <= "\uf8ff" for character in masked)
    for block in _segment_blocks(plan):
        assert block.source_hash in by_hash
        assert tuple(token for token, _ in block.placeholders) == by_hash[block.source_hash].placeholder_tokens


def test_realistic_paper_identity_and_protected_structure():
    paper = _realistic_paper()
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", paper)])
    _assert_segment_invariants(plan)

    _assert_absent_from_segments(
        plan,
        *_lean_lines(LEAN_ONE, LEAN_TWO, LEAN_THREE),
        "Lean 4 proof:",
        THEOREMS_APPENDIX_START,
        THEOREMS_APPENDIX_END,
        EMPTY_APPENDIX,
        LEGACY_HEADER,
        "AI Self-Review and Limitations",
        "MODEL CREDITS",
        "=" * 80,
        "Theorem (proof_a1)",
        "Theorem (proof_a2)",
        "Theorem (proof_legacy)",
        "Status:",
        "Source:",
        "Statement:",
    )
    masked = _masked_texts(plan)
    assert "For natural numbers [[M1]] with [[M2]], the sum [[M3]] is positive." in masked
    assert "Every natural number [[M1]] is strictly greater than one." in masked
    assert any("The argument relies only on standard arithmetic." in text for text in masked)
    assert any("AUTONOMOUS AI SOLUTION" in text for text in masked)
    assert not any("$" in text or "\\begin" in text or "https://" in text for text in masked)

    blocks = plan.parts[0][1]
    for index, block in enumerate(blocks):
        if block.kind == "protected" and block.text.endswith("Statement: "):
            assert blocks[index + 1].kind == "segment"
            break
    else:
        pytest.fail("Statement label was not kept as a protected prefix")


def test_translated_paper_keeps_proof_parser_structure_and_originals():
    paper = _realistic_paper()
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", paper)])
    translations = {segment.source_hash: _reverse_words(segment.masked_text) for segment in plan.segments}
    translated = assemble_parts(plan, translations)["content"]

    assert translated != paper
    original = analyze_paper_content(paper)
    result = analyze_paper_content(translated)
    assert result["proof_count"] == original["proof_count"] == 3
    assert [proof["title"] for proof in result["proofs"]] == [proof["title"] for proof in original["proofs"]]
    for code in (LEAN_ONE, LEAN_TWO, LEAN_THREE):
        assert code in translated
    for original_math in (
        "$a + b > 0$",
        "$$\n\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}\n$$",
        "\\[ e^{i\\pi} + 1 = 0 \\]",
        "\\(\\mathbb{C}\\)",
        "\\begin{align*}\na &= b \\\\\nb &= c\n\\end{align*}",
        "`lake build`",
        "https://example.org/notes?id=4",
        "\\cite{knuth1997}",
        "\\ref{sec:proofs}",
    ):
        assert original_math in translated
    assert "Statement: roF larutan srebmun $a, b$ htiw $0 < a$" in translated


def test_brainstorm_database_identity_keeps_submission_and_proof_structure():
    database = _brainstorm_database()
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", database)])
    _assert_segment_invariants(plan)
    _assert_absent_from_segments(
        plan,
        "SUBMISSION #",
        "Accepted:",
        "=" * 80,
        BRAINSTORM_PROOF_HEADER,
        "Proof 1:",
        "Proof ID:",
        "Lean 4 Code:",
        *_lean_lines(LEAN_BRAINSTORM),
    )
    masked = _masked_texts(plan)
    assert "Consider the map [[M1]] on [[M2]]." in masked
    assert "It has no fixed point." in masked
    assert "## Route" in masked
    assert "- Bound the sum with [[M1]].\n- Conclude positivity." in masked
    assert "A counterexample search at [[M1]] found nothing." in masked
    # The statement after the "Proof N:" label is translated; the label is kept.
    assert any(text.startswith("Truth holds in the brainstorm.") for text in masked)


def test_long_form_volume_identity_keeps_chapter_headers():
    volume = _volume(_realistic_paper())
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", volume)])
    _assert_segment_invariants(plan)
    _assert_absent_from_segments(
        plan,
        "FINAL ANSWER VOLUME",
        "TABLE OF CONTENTS",
        "END OF VOLUME",
        "# CHAPTER",
        "# Type:",
        "#" * 80,
        "-" * 40,
        *_lean_lines(LEAN_ONE, LEAN_TWO, LEAN_THREE),
    )
    translated = assemble_parts(
        plan, {segment.source_hash: _reverse_words(segment.masked_text) for segment in plan.segments}
    )["content"]
    assert analyze_paper_content(translated)["proof_count"] == analyze_paper_content(volume)["proof_count"]
    assert "# CHAPTER 2: Closing Remarks" in translated


def test_crlf_and_mixed_line_endings_are_reproduced_exactly():
    crlf_paper = _realistic_paper().replace("\n", "\r\n")
    mixed = "First line.\r\nSecond line.\n\r\nNext $x$ paragraph.\r\n\r\n   \r\nLast.\n"
    plan = _assert_identity(PROFILE_DOCUMENT, [("paper", crlf_paper), ("mixed", mixed)])
    _assert_segment_invariants(plan)
    _assert_absent_from_segments(plan, *_lean_lines(LEAN_ONE, LEAN_TWO, LEAN_THREE), THEOREMS_APPENDIX_START)


def test_empty_and_whitespace_parts_yield_only_protected_blocks():
    parts = [("empty", ""), ("blank", "   \n\t\n"), ("crlf", "\r\n\r\n"), ("nbsp", "\u00a0 \n")]
    plan = _assert_identity(PROFILE_DOCUMENT, parts)
    assert plan.segments == ()
    assert plan.parts[0] == ("empty", ())
    for _, blocks in plan.parts:
        assert all(block.kind == "protected" for block in blocks)


def test_fields_profile_protects_only_fenced_code():
    fields = {
        "theorem_statement": "For every $n \\ge 1$, the sum is positive.",
        "formal_sketch": (
            "Status: induct on n.\n\n"
            "```lean\ntheorem fenced_marker : True := trivial\n```\n\n"
            "Statement: close with omega."
        ),
        "novelty_reasoning": "Theorem (main) is new because no reference states it.",
        "novelty_group_claim": "",
    }
    parts = [(name, fields[name]) for name in PROOF_TRANSLATION_FIELDS]
    plan = _assert_identity(PROFILE_FIELDS, parts)
    _assert_segment_invariants(plan)
    _assert_absent_from_segments(plan, "fenced_marker", "```")
    masked = _masked_texts(plan)
    assert "For every [[M1]], the sum is positive." in masked
    assert "Status: induct on n." in masked
    assert "Statement: close with omega." in masked
    assert "Theorem (main) is new because no reference states it." in masked
    assert dict(plan.parts)["novelty_group_claim"] == ()


@pytest.mark.parametrize(
    "snippet",
    [
        "$x^2 + 1$",
        "$$\\int_0^1 f(x)\\,dx$$",
        "$$\na = b\n$$",
        "\\[ a = b \\]",
        "\\(a + b\\)",
        "\\begin{equation}\na = b\n\\end{equation}",
        "\\begin{align*}\na &= b \\\\\nc &= d\n\\end{align*}",
        "\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}",
        "\\begin{cases} 1 & x > 0 \\end{cases}",
        "`lake build`",
        "``a ` b``",
        "https://example.org/path?q=1&r=2",
        "http://example.org",
        "\\cite{knuth1997}",
        "\\ref{sec:intro}",
        "\\eqref{eq:main}",
        "\\label{eq:main}",
        "\\url{example.org}",
        "\\href{https://example.org}",
    ],
)
def test_each_maskable_span_becomes_one_placeholder(snippet):
    text = f"Before {snippet} after the span"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    [segment] = plan.segments
    assert segment.masked_text == "Before [[M1]] after the span"
    [block] = _segment_blocks(plan)
    assert block.placeholders == (("[[M1]]", snippet),)


@pytest.mark.parametrize(
    "text",
    [
        "It costs $5 and $10 today.",
        "An escaped \\$x$ price stays.",
        "A stray $ sign and $ another.",
        "Unclosed $$ display opener stays.",
    ],
)
def test_currency_and_unbalanced_dollars_are_not_masked(text):
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    [segment] = plan.segments
    assert segment.masked_text == text
    assert segment.placeholder_tokens == ()


def test_url_trailing_sentence_punctuation_stays_translatable():
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", "Read https://example.org/a. Then stop.")])
    [segment] = plan.segments
    assert segment.masked_text == "Read [[M1]]. Then stop."
    assert _segment_blocks(plan)[0].placeholders == (("[[M1]]", "https://example.org/a"),)


def test_url_mask_stops_at_non_ascii_text_and_full_width_punctuation():
    text = "详见https://example.com，该方法提出了新的证明。我们证明了这个定理。"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    [segment] = plan.segments
    assert segment.masked_text == "详见[[M1]]，该方法提出了新的证明。我们证明了这个定理。"
    assert _segment_blocks(plan)[0].placeholders == (("[[M1]]", "https://example.com"),)
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", "参见 https://example.org/a/b?x=1）以及更多内容。")])
    assert plan.segments[0].masked_text == "参见 [[M1]]）以及更多内容。"


@pytest.mark.parametrize(
    "text",
    [
        "The so-called ``rigidity'' property and the ``flexibility'' property differ.",
        "Compare the ``rigidity'' property with the ``(local) flexibility'' property.",
        "It`s clear that the model`s output is stable.",
    ],
)
def test_latex_quotes_and_backtick_apostrophes_are_not_code(text):
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    [segment] = plan.segments
    assert segment.masked_text == text
    assert segment.placeholder_tokens == ()


def test_inline_code_spans_still_mask_beside_spaces_punctuation_and_cjk():
    text = "Run `lake build` now, then `x`. 调用`foo()`函数。"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    [segment] = plan.segments
    assert segment.masked_text == "Run [[M1]] now, then [[M2]]. 调用[[M3]]函数。"
    assert [original for _token, original in _segment_blocks(plan)[0].placeholders] == [
        "`lake build`",
        "`x`",
        "`foo()`",
    ]


def test_tilde_fenced_code_is_protected_like_backtick_fences():
    text = (
        "Intro prose.\n"
        "~~~lean\ntheorem tilde_marker : True := by\n  trivial\n~~~\n"
        "Middle prose.\n"
        "~~~\nplain_tilde_marker = 1\n```\nstill_tilde_marker\n~~~~\n"
        "```\nbacktick_marker\n~~~\nstill_backtick_marker\n```\n"
        "Closing prose.\n"
    )
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["Intro prose.", "Middle prose.", "Closing prose."]
    _assert_absent_from_segments(plan, "tilde_marker", "backtick_marker", "~~~", "```")
    fields = _assert_identity(PROFILE_FIELDS, [("formal_sketch", "Sketch.\n~~~\nfield_tilde_marker\n~~~\n")])
    _assert_absent_from_segments(fields, "field_tilde_marker")


def test_unterminated_tilde_fence_protects_rest_of_part():
    text = "Intro prose.\n~~~python\nfence_marker = 1\n\nStill code text.\n"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["Intro prose."]


def test_math_environment_with_blank_line_stays_one_protected_span():
    environment = "\\begin{align}\na = 1\n\nb = 2\n\\end{align}"
    text = f"Intro text.\n\n{environment}\n\nOutro text."
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["Intro text.", "Outro text."]
    assert [block.kind for block in plan.parts[0][1]] == ["segment", "protected", "segment"]
    assert plan.parts[0][1][1].text == f"\n\n{environment}\n\n"
    inline = _assert_identity(PROFILE_DOCUMENT, [("content", f"Before {environment} after.")])
    [segment] = inline.segments
    assert segment.masked_text == "Before [[M1]] after."
    assert _segment_blocks(inline)[0].placeholders == (("[[M1]]", environment),)


def test_placeholders_renumber_from_one_per_segment():
    text = "First $a$ and $b$ here.\n---\nSecond $c$ there."
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["First [[M1]] and [[M2]] here.", "Second [[M1]] there."]
    blocks = _segment_blocks(plan)
    assert blocks[0].placeholders == (("[[M1]]", "$a$"), ("[[M2]]", "$b$"))
    assert blocks[1].placeholders == (("[[M1]]", "$c$"),)


def test_identical_prose_with_different_math_is_deduplicated():
    parts = [("one", "Let $a$ be prime."), ("two", "Let $b + c$ be prime."), ("three", "Let $a$ be prime.")]
    plan = _assert_identity(PROFILE_DOCUMENT, parts)
    [segment] = plan.segments
    assert segment.masked_text == "Let [[M1]] be prime."
    blocks = _segment_blocks(plan)
    assert len(blocks) == 3
    assert {block.source_hash for block in blocks} == {segment.source_hash}
    assert [block.placeholders for block in blocks] == [
        (("[[M1]]", "$a$"),),
        (("[[M1]]", "$b + c$"),),
        (("[[M1]]", "$a$"),),
    ]
    translated = assemble_parts(plan, {segment.source_hash: "Sea [[M1]] primo."})
    assert translated == {"one": "Sea $a$ primo.", "two": "Sea $b + c$ primo.", "three": "Sea $a$ primo."}


def test_placeholder_collision_switches_prefix_for_that_run_only():
    text = "Literal [[M1]] marker with $x$.\n---\nSee [[Main Page]] with $y$.\n---\nPlain $z$ run."
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == [
        "Literal [[MOTO1]] marker with [[MOTO2]].",
        "See [[Main Page]] with [[MOTO1]].",
        "Plain [[M1]] run.",
    ]
    blocks = _segment_blocks(plan)
    assert blocks[0].placeholders == (("[[MOTO1]]", "[[M1]]"), ("[[MOTO2]]", "$x$"))
    translated = assemble_parts(plan, {plan.segments[0].source_hash: "Wörtlich [[MOTO1]] mit [[MOTO2]].",
                                       plan.segments[1].source_hash: "Siehe [[Main Page]] mit [[MOTO1]].",
                                       plan.segments[2].source_hash: "Nur [[M1]] Lauf."})
    assert translated == {
        "content": "Wörtlich [[M1]] mit $x$.\n---\nSiehe [[Main Page]] mit $y$.\n---\nNur $z$ Lauf."
    }


def test_letter_free_passages_stay_protected():
    text = "Text here.\n---\n$$x = 1$$\n---\n12.5 (3)\n"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["Text here."]


def test_unterminated_lean_block_ends_before_appendix_marker():
    content = (
        f"{THEOREMS_APPENDIX_START}\n"
        "Theorem (proof_1) - One\n"
        "Lean 4 proof:\n"
        "theorem unterminated_marker : True := by trivial\n"
        f"{THEOREMS_APPENDIX_END}\n"
        "AI Self-Review and Limitations\n"
        "Review survives translation.\n"
    )
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", content)])
    assert _masked_texts(plan) == ["Review survives translation."]


def test_unterminated_fence_protects_rest_of_part():
    text = "Intro prose.\n```python\nfence_marker = 1\n\nStill code text.\n"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    assert _masked_texts(plan) == ["Intro prose."]


def _sentence(index: int) -> str:
    return f"Sentence {index} explains one more careful step of the argument about sums."


def _paragraph(start: int, sentences: int) -> str:
    return " ".join(_sentence(start + offset) for offset in range(sentences))


def test_every_paragraph_is_its_own_unit():
    paragraphs = [_paragraph(index * 100, 2) for index in range(12)]
    text = "\n\n".join(paragraphs)
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    _assert_segment_invariants(plan)
    assert _masked_texts(plan) == paragraphs
    blocks = plan.parts[0][1]
    assert [block.kind for block in blocks] == ["segment", "protected"] * (len(paragraphs) - 1) + ["segment"]
    assert all(block.text == "\n\n" for block in blocks if block.kind == "protected")


def test_heading_list_and_quote_paragraphs_are_separate_units():
    text = "## Results\n\nWe prove $p$ holds.\n\n- First item.\n- Second item.\n\n> A remark.\n"
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    _assert_segment_invariants(plan)
    assert _masked_texts(plan) == [
        "## Results",
        "We prove [[M1]] holds.",
        "- First item.\n- Second item.",
        "> A remark.",
    ]
    assert [segment.paragraph_markers for segment in plan.segments] == [("h2",), ("",), ("ul",), ("quote",)]


def test_paragraph_at_target_stays_whole_and_one_over_target_splits():
    sentences = 1
    while count_tokens(_paragraph(0, sentences + 1)) <= SEGMENT_TARGET_TOKENS:
        sentences += 1
    at_target = _paragraph(0, sentences)
    over_target = _paragraph(0, sentences + 1)
    assert count_tokens(at_target) <= SEGMENT_TARGET_TOKENS < count_tokens(over_target)

    plan = _assert_identity(PROFILE_DOCUMENT, [("content", f"Short opener.\n\n{at_target}\n\nShort closer.")])
    assert _masked_texts(plan) == ["Short opener.", at_target, "Short closer."]

    plan = _assert_identity(PROFILE_DOCUMENT, [("content", over_target)])
    _assert_segment_invariants(plan)
    pieces = _masked_texts(plan)
    assert len(pieces) == 2
    assert " ".join(pieces) == over_target
    assert all(segment.token_count <= SEGMENT_TARGET_TOKENS for segment in plan.segments)


def test_oversized_paragraph_splits_at_sentences():
    oversized = _paragraph(0, 120)
    assert count_tokens(oversized) > 2 * SEGMENT_TARGET_TOKENS
    text = f"Opening paragraph.\n\n{oversized}\n\nClosing paragraph."
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", text)])
    _assert_segment_invariants(plan)
    pieces = [segment for segment in plan.segments if segment.masked_text.startswith("Sentence")]
    assert len(pieces) >= 3
    for piece in pieces:
        assert piece.token_count <= SEGMENT_TARGET_TOKENS
        assert piece.paragraph_count == 1
        assert piece.masked_text.endswith("sums.")
    blocks = plan.parts[0][1]
    inner = [block.text for block in blocks[2:-2] if block.kind == "protected"]
    assert inner and all(text == " " for text in inner)


def test_oversized_sentence_splits_at_whitespace_never_inside_placeholders():
    words = " ".join(f"term $x_{{{index}}} + y$ grows" for index in range(450))
    assert count_tokens(re.sub(r"\$[^$]*\$", "[[M1]]", words)) > 2 * SEGMENT_TARGET_TOKENS
    plan = _assert_identity(PROFILE_DOCUMENT, [("content", words)])
    _assert_segment_invariants(plan)
    blocks = _segment_blocks(plan)
    assert len(blocks) >= 3
    assert all(segment.token_count <= SEGMENT_TARGET_TOKENS for segment in plan.segments)
    for block in blocks:
        assert block.placeholders
        assert all(re.fullmatch(r"\$x_\{\d+\} \+ y\$", original) for _, original in block.placeholders)
    for block in plan.parts[0][1]:
        if block.kind == "protected":
            assert block.text == " "


def test_plans_and_hashes_are_deterministic():
    paper = _realistic_paper()
    first = build_document_plan(PROFILE_DOCUMENT, [("content", paper)])
    second = build_document_plan(PROFILE_DOCUMENT, [("content", paper)])
    assert first == second
    expected_doc_hash = hashlib.sha256(
        json.dumps(
            {"version": SEGMENTATION_VERSION, "profile": PROFILE_DOCUMENT, "parts": [["content", paper]]},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert first.doc_hash == expected_doc_hash
    assert build_document_plan(PROFILE_FIELDS, [("content", paper)]).doc_hash != first.doc_hash
    assert build_document_plan(PROFILE_DOCUMENT, [("content", paper + " ")]).doc_hash != first.doc_hash
    assert build_document_plan(PROFILE_DOCUMENT, [("other", paper)]).doc_hash != first.doc_hash


@pytest.mark.parametrize(
    "profile, parts",
    [
        (PROFILE_DOCUMENT, [("content", _realistic_paper())]),
        (PROFILE_DOCUMENT, [("content", _brainstorm_database()), ("empty", "")]),
        (PROFILE_FIELDS, [("theorem_statement", "For all $n$, done."), ("formal_sketch", "```\nx\n```")]),
    ],
)
def test_plan_json_round_trip_is_lossless(profile, parts):
    plan = build_document_plan(profile, parts)
    text = plan_to_json(plan)
    restored = plan_from_json(text)
    assert restored == plan
    assert plan_to_json(restored) == text
    assert assemble_parts(restored, _identity(restored)) == dict(parts)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"segmentation_version": "moto-translate-seg-v0", "profile": "document",
                    "doc_hash": "x", "parts": [], "segments": []}),
        json.dumps({"segmentation_version": SEGMENTATION_VERSION, "profile": "document",
                    "doc_hash": "x", "parts": [["content", [{"kind": "mystery"}]]], "segments": []}),
        json.dumps({"segmentation_version": SEGMENTATION_VERSION, "profile": "document",
                    "doc_hash": "x", "parts": [], "segments": [{"source_hash": "h"}]}),
    ],
)
def test_plan_from_json_rejects_invalid_or_foreign_plans(text):
    with pytest.raises(ValueError):
        plan_from_json(text)


def test_assemble_parts_requires_every_segment():
    plan = build_document_plan(PROFILE_DOCUMENT, [("a", "First passage."), ("b", "Second passage.")])
    translations = _identity(plan)
    translations.pop(plan.segments[1].source_hash)
    assert assemble_parts(plan, translations) is None
    assert assemble_parts(plan, {}) is None


def test_translated_assembly_substitutes_and_unmasks_each_occurrence():
    text = "## Results\n\nWe prove $p$ holds.\n\n- Use \\cite{ref1}.\n"
    plan = build_document_plan(PROFILE_DOCUMENT, [("content", text)])
    heading, claim, item = plan.segments
    assert (heading.masked_text, claim.masked_text, item.masked_text) == (
        "## Results", "We prove [[M1]] holds.", "- Use [[M1]]."
    )
    translated = assemble_parts(
        plan,
        {
            heading.source_hash: "## Resultados",
            claim.source_hash: "Probamos que [[M1]] se cumple.",
            item.source_hash: "- Usa [[M1]].",
        },
    )
    assert translated == {
        "content": "## Resultados\n\nProbamos que $p$ se cumple.\n\n- Usa \\cite{ref1}.\n"
    }


def test_unmask_is_single_pass_and_ignores_unknown_tokens():
    placeholders = (("[[M1]]", "[[M2]]"), ("[[M2]]", "$x$"), ("[[M10]]", "$y$"))
    assert unmask("[[M1]] and [[M2]] and [[M10]] and [[M3]]", placeholders) == (
        "[[M2]] and $x$ and $y$ and [[M3]]"
    )
    assert unmask("no tokens", ()) == "no tokens"


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        build_document_plan("ui", [("content", "text")])
    with pytest.raises(ValueError):
        build_document_plan(PROFILE_DOCUMENT, [("content", "a"), ("content", "b")])
    with pytest.raises(ValueError):
        build_document_plan(PROFILE_DOCUMENT, [("content", None)])


_WORDS = (
    "alpha", "beta", "gamma", "sum", "bound", "prime", "Riemann", "zeta", "unique", "proof",
    "定理", "証明", "مبرهنة", "théorème", "Größe",
)
_INLINE = (
    "$x_1$", "\\(a+b\\)", "`code()`", "https://example.org/x", "\\cite{k}", "[[M3]]", "\\$5", "$ x",
)
_PROTECTED = (
    THEOREMS_APPENDIX_START,
    THEOREMS_APPENDIX_END,
    "=" * 80,
    "---",
    "Theorem (proof_r) [Novel] - Random title",
    "Status: verified by Lean 4",
    "Source: paper p1",
    "SUBMISSION #4 | Accepted: 2026-10-01",
    "# CHAPTER 3: Random",
    "MODEL CREDITS",
    "Proof ID: proof_r",
)


def _random_prose(rng: random.Random) -> str:
    tokens = []
    for _ in range(rng.randint(1, 14)):
        tokens.append(rng.choice(_INLINE) if rng.random() < 0.2 else rng.choice(_WORDS))
    return " ".join(tokens) + rng.choice(("", ".", "!", "?", "。"))


def _random_lines(rng: random.Random, serial: list) -> list:
    choice = rng.random()
    indent = rng.choice(("", "", "  ", "\t"))
    if choice < 0.35:
        return [indent + _random_prose(rng)]
    if choice < 0.5:
        return [rng.choice(("", "   ", "\t"))]
    if choice < 0.58:
        return [rng.choice(("## ", "- ", "1. ", "> ")) + _random_prose(rng)]
    if choice < 0.7:
        return [rng.choice(_PROTECTED)]
    if choice < 0.76:
        return [indent + "Statement: " + _random_prose(rng)]
    if choice < 0.84:
        serial[0] += 1
        lines = [rng.choice(("Lean 4 proof:", "Lean 4 Code:")), f"theorem leanmark{serial[0]} : True := by", "  trivial"]
        if rng.random() < 0.8:
            lines.append("---")
        return lines
    if choice < 0.87:
        serial[0] += 1
        lines = ["```python", f"codemark{serial[0]} = 1"]
        if rng.random() < 0.8:
            lines.append("```")
        return lines
    if choice < 0.9:
        serial[0] += 1
        lines = ["~~~lean", f"tildemark{serial[0]} = 1"]
        if rng.random() < 0.8:
            lines.append(rng.choice(("~~~", "~~~~")))
        return lines
    if choice < 0.95:
        return ["$$", "a = b + c", "$$"]
    return ["\\begin{equation}", "envmark = 1", "\\end{equation}"]


@pytest.mark.parametrize("seed", range(40))
def test_random_documents_round_trip_and_keep_protected_content(seed):
    rng = random.Random(seed)
    serial = [0]
    parts = []
    for part_index in range(rng.randint(1, 3)):
        lines = []
        for _ in range(rng.randint(0, 40)):
            lines.extend(_random_lines(rng, serial))
        newline = rng.choice(("\n", "\r\n"))
        text = newline.join(lines) + rng.choice(("", newline, newline * 2))
        parts.append((f"part{part_index}", text))
    for profile in (PROFILE_DOCUMENT, PROFILE_FIELDS):
        plan = _assert_identity(profile, parts)
        _assert_segment_invariants(plan)
        _assert_absent_from_segments(plan, "codemark", "tildemark", "envmark", "a = b + c", "$x_1$", "`code()`")
        if profile == PROFILE_DOCUMENT:
            _assert_absent_from_segments(plan, "leanmark", "Proof ID:", "SUBMISSION #", "Statement:", *_PROTECTED[:3])
        assert plan_from_json(plan_to_json(plan)) == plan
        _assert_unit_view_round_trip(plan, parts)


def _masked_by_hash(plan: DocumentPlan) -> dict:
    return {segment.source_hash: segment.masked_text for segment in plan.segments}


def _assert_unit_view_round_trip(plan: DocumentPlan, parts) -> list:
    """Every unit converts to global tokens and back, and the document reassembles exactly."""
    views = build_unit_views(plan)
    by_hash = _masked_by_hash(plan)
    assert [view.number for view in views] == list(range(1, len(views) + 1))
    assert [view.occurrence for view in views] == list(range(len(views)))
    assert [view.source_hash for view in views] == [block.source_hash for block in _segment_blocks(plan)]
    global_tokens = [token for view in views for token in view.global_tokens]
    prefix = "MOTO" if global_tokens and global_tokens[0].startswith("[[MOTO") else "M"
    assert global_tokens == [f"[[{prefix}{index}]]" for index in range(1, len(global_tokens) + 1)]
    round_trip = {}
    for view in views:
        masked = by_hash[view.source_hash]
        assert view.local_tokens == tuple(PLACEHOLDER_RE.findall(masked))
        assert len(view.global_tokens) == len(view.local_tokens)
        rewritten = to_global(masked, view)
        assert tuple(PLACEHOLDER_RE.findall(rewritten)) == view.global_tokens
        round_trip[view.occurrence] = to_local(rewritten, view)
        assert round_trip[view.occurrence] == masked
    assert assemble_occurrences(plan, round_trip) == dict(parts)
    return views


_TWO_PART_UNITS = [
    ("one", "Let $a$ and $b$ be primes.\n\nNo math here."),
    ("two", "Then $c$ follows.\n---\nLet $x$ and $y$ be primes."),
]


def test_unit_views_number_occurrences_across_parts_with_document_wide_tokens():
    plan = build_document_plan(PROFILE_DOCUMENT, _TWO_PART_UNITS)
    views = _assert_unit_view_round_trip(plan, _TWO_PART_UNITS)
    blocks = _segment_blocks(plan)
    assert views == [
        UnitView(1, "one", 0, blocks[0].source_hash, ("[[M1]]", "[[M2]]"), ("[[M1]]", "[[M2]]")),
        UnitView(2, "one", 1, blocks[1].source_hash, (), ()),
        UnitView(3, "two", 2, blocks[2].source_hash, ("[[M1]]",), ("[[M3]]",)),
        UnitView(4, "two", 3, blocks[0].source_hash, ("[[M1]]", "[[M2]]"), ("[[M4]]", "[[M5]]")),
    ]
    assert len(plan.segments) == 3
    assert build_unit_views(build_document_plan(PROFILE_DOCUMENT, [("empty", "  \n")])) == []


def test_fields_profile_units_number_across_fields():
    parts = [("theorem_statement", "For every $n$, it holds."), ("formal_sketch", "Induct on $n$.\n\nClose it.")]
    plan = build_document_plan(PROFILE_FIELDS, parts)
    views = _assert_unit_view_round_trip(plan, parts)
    assert [(view.number, view.part, view.global_tokens) for view in views] == [
        (1, "theorem_statement", ("[[M1]]",)),
        (2, "formal_sketch", ("[[M2]]",)),
        (3, "formal_sketch", ()),
    ]


def test_assemble_occurrences_translates_repeated_units_independently():
    plan = build_document_plan(PROFILE_DOCUMENT, _TWO_PART_UNITS)
    translations = {
        0: "Sean [[M1]] y [[M2]] primos.",
        1: "Sin matemáticas.",
        2: "Luego se sigue [[M1]].",
        3: "Sean [[M1]] y [[M2]] números primos.",
    }
    assert assemble_occurrences(plan, translations) == {
        "one": "Sean $a$ y $b$ primos.\n\nSin matemáticas.",
        "two": "Luego se sigue $c$.\n---\nSean $x$ y $y$ números primos.",
    }


def _masked_by_occurrence(plan: DocumentPlan) -> dict:
    by_hash = _masked_by_hash(plan)
    return {view.occurrence: by_hash[view.source_hash] for view in build_unit_views(plan)}


def test_assemble_occurrences_requires_every_occurrence():
    plan = build_document_plan(PROFILE_DOCUMENT, _TWO_PART_UNITS)
    complete = _masked_by_occurrence(plan)
    assert assemble_occurrences(plan, complete) == dict(_TWO_PART_UNITS)
    missing = dict(complete)
    missing.pop(3)
    assert assemble_occurrences(plan, missing) is None
    assert assemble_occurrences(plan, {**complete, 1: None}) is None
    assert assemble_occurrences(plan, {}) is None
    empty = build_document_plan(PROFILE_DOCUMENT, [("blank", "\n\n")])
    assert assemble_occurrences(empty, {}) == {"blank": "\n\n"}


def test_retokening_is_single_pass_when_local_and_global_numbers_overlap():
    parts = [("content", "First $a$ here.\n\nSecond $b$ and $c$ there.")]
    plan = build_document_plan(PROFILE_DOCUMENT, parts)
    first, second = _assert_unit_view_round_trip(plan, parts)
    assert first.global_tokens == ("[[M1]]",)
    assert second.local_tokens == ("[[M1]]", "[[M2]]")
    assert second.global_tokens == ("[[M2]]", "[[M3]]")
    assert to_global("Second [[M1]] and [[M2]] there.", second) == "Second [[M2]] and [[M3]] there."
    assert to_global("[[M2]] before [[M1]]", second) == "[[M3]] before [[M2]]"
    assert to_local("[[M3]] before [[M2]]", second) == "[[M2]] before [[M1]]"
    # Tokens a unit does not own pass through unchanged for the pre-checks to report.
    assert to_local("[[M1]] then [[M9]] and [[MOTO2]]", second) == "[[M1]] then [[M9]] and [[MOTO2]]"
    assert to_global("No tokens.", first) == "No tokens."


def test_token_shaped_original_switches_the_whole_document_to_moto_tokens():
    parts = [("content", "Plain $z$ run.\n---\nLiteral [[M1]] marker with $x$.")]
    plan = build_document_plan(PROFILE_DOCUMENT, parts)
    plain, literal = _assert_unit_view_round_trip(plan, parts)
    assert plain.local_tokens == ("[[M1]]",)
    assert plain.global_tokens == ("[[MOTO1]]",)
    assert literal.local_tokens == ("[[MOTO1]]", "[[MOTO2]]")
    assert literal.global_tokens == ("[[MOTO2]]", "[[MOTO3]]")
    by_hash = _masked_by_hash(plan)
    shown = to_global(by_hash[literal.source_hash], literal)
    assert shown == "Literal [[MOTO2]] marker with [[MOTO3]]."
    assert to_global(by_hash[plain.source_hash], plain) == "Plain [[MOTO1]] run."
    translated = assemble_occurrences(
        plan,
        {
            plain.occurrence: to_local("Simple [[MOTO1]].", plain),
            literal.occurrence: to_local("Wörtlich [[MOTO2]] mit [[MOTO3]].", literal),
        },
    )
    assert translated == {"content": "Simple $z$.\n---\nWörtlich [[M1]] mit $x$."}


def test_bracketed_text_beginning_with_m_switches_the_document_to_moto_tokens():
    parts = [("content", "Plain $z$ run.\n---\nSee [[Main Page]] with $y$.")]
    plan = build_document_plan(PROFILE_DOCUMENT, parts)
    plain, page = _assert_unit_view_round_trip(plan, parts)
    assert (plain.global_tokens, page.local_tokens, page.global_tokens) == (
        ("[[MOTO1]]",), ("[[MOTO1]]",), ("[[MOTO2]]",)
    )
    assert to_global(_masked_by_hash(plan)[page.source_hash], page) == "See [[Main Page]] with [[MOTO2]]."
    other = build_document_plan(PROFILE_DOCUMENT, [("content", "See $y$ only.")])
    assert build_unit_views(other)[0].global_tokens == ("[[M1]]",)


def test_unit_views_round_trip_crlf_text():
    parts = [("content", "Intro $a$ line.\r\nSecond line.\r\n\r\nNext $b$ and $c$.\r\n")]
    plan = build_document_plan(PROFILE_DOCUMENT, parts)
    first, second = _assert_unit_view_round_trip(plan, parts)
    by_hash = _masked_by_hash(plan)
    assert by_hash[first.source_hash] == "Intro [[M1]] line.\r\nSecond line."
    assert to_global(by_hash[first.source_hash], first) == "Intro [[M1]] line.\r\nSecond line."
    assert to_global(by_hash[second.source_hash], second) == "Next [[M2]] and [[M3]]."
    translated = assemble_occurrences(
        plan,
        {
            first.occurrence: to_local("Intro [[M1]] línea.\r\nSegunda línea.", first),
            second.occurrence: to_local("Luego [[M2]] y [[M3]].", second),
        },
    )
    assert translated == {"content": "Intro $a$ línea.\r\nSegunda línea.\r\n\r\nLuego $b$ y $c$.\r\n"}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_realistic_paper_unit_views_round_trip(newline):
    parts = [("content", _realistic_paper().replace("\n", newline)), ("volume", _volume("Closing words."))]
    plan = build_document_plan(PROFILE_DOCUMENT, parts)
    views = _assert_unit_view_round_trip(plan, parts)
    assert len(views) == len(_segment_blocks(plan)) >= len(plan.segments) > 10
    assert {view.part for view in views} == {"content", "volume"}


def test_module_constants_match_spec():
    assert SEGMENTATION_VERSION == "moto-translate-seg-v2"
    assert SEGMENT_TARGET_TOKENS == 600
    assert not hasattr(segmentation, "SEGMENT_MAX_TOKENS")
    assert segmentation.PROFILES == (PROFILE_DOCUMENT, PROFILE_FIELDS)
    assert PlanBlock(kind="protected").placeholders == ()
