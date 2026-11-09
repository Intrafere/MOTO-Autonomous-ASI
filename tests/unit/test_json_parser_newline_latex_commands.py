"""Opt-out of the ``\\n``-prefixed LaTeX pre-escapes in the shared JSON parser."""
from __future__ import annotations

import json

import pytest

from backend.shared.json_parser import parse_json, sanitize_json_response

# Model output with unescaped LaTeX commands that begin with "\n".
LATEX_RAW = r'{"submission": "Let \nu be a measure, \nabla f \neq 0, \neg p, \notin S, \not\in T, \newline end."}'
LATEX_DECODED = r"Let \nu be a measure, \nabla f \neq 0, \neg p, \notin S, \not\in T, \newline end."


def test_default_still_reads_newline_prefixed_names_as_latex_commands():
    assert parse_json(LATEX_RAW)["submission"] == LATEX_DECODED
    assert sanitize_json_response(LATEX_RAW) == sanitize_json_response(
        LATEX_RAW, escape_newline_latex_commands=True
    )
    # A JSON line break before "using" is still read as \nu by default.
    assert parse_json('{"text": "First line.\\nusing the bound."}')["text"] == "First line.\\nusing the bound."


@pytest.mark.parametrize(
    "text",
    [
        "First line.\nusing the bound.",
        "Primera parte.\n\notro párrafo.",
        "Első rész.\n\negy bekezdés.",
        "A\nequal split and\nablaut shift.",
        "Results:\notherwise final.",
    ],
)
def test_opt_out_keeps_json_line_breaks_before_such_words(text):
    raw = json.dumps({"text": text}, ensure_ascii=False)
    assert parse_json(raw, escape_newline_latex_commands=False)["text"] == text


def test_opt_out_still_pre_escapes_the_other_latex_commands():
    raw = r'{"text": "Use \textbf{bold}, \frac{a}{b}, \beta, \rightarrow, and \tau."}'
    expected = r"Use \textbf{bold}, \frac{a}{b}, \beta, \rightarrow, and \tau."
    assert parse_json(raw, escape_newline_latex_commands=False)["text"] == expected
    assert parse_json(raw)["text"] == expected
