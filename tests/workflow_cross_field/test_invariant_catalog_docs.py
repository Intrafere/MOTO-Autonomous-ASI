from __future__ import annotations

from pathlib import Path

from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID


CATALOG_PATH = (
    Path(__file__).parents[1] / "workflow_testing_plans" / "02_invariant_catalog.md"
)
START = "<!-- BEGIN GENERATED INVARIANT IDS -->"
END = "<!-- END GENERATED INVARIANT IDS -->"


def render_stable_id_projection() -> str:
    lines = [START, ""]
    lines.extend(f"- `{invariant_id}`" for invariant_id in sorted(INVARIANTS_BY_ID))
    lines.extend(["", END])
    return "\n".join(lines)


def test_invariant_catalog_document_enumerates_every_stable_id():
    text = CATALOG_PATH.read_text(encoding="utf-8")
    prefix, remainder = text.split(START, 1)
    _current, suffix = remainder.split(END, 1)
    expected = prefix + render_stable_id_projection() + suffix
    assert text == expected, (
        "Invariant catalog stable-ID projection is stale. Replace the generated "
        "marker block with render_stable_id_projection()."
    )
