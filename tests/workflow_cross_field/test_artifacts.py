from __future__ import annotations

import json

import pytest

from tests.workflow_cross_field.artifacts import (
    RESULTS_DIR,
    SCENARIOS_DIR,
    SCHEMA_VERSION,
    SEEDS,
    SUMMARY_PATH,
    TARGETS_DIR,
    TARGET_RISKS_JSON_PATH,
    TARGET_RISKS_MARKDOWN_PATH,
    render_artifacts,
)
from tests.workflow_cross_field.support_graph import (
    SUPPORT_GRAPH_JSON_PATH,
    SUPPORT_GRAPH_MARKDOWN_PATH,
)
from tests.workflow_harness.coverage_metadata import (
    InteractionCoverage,
    assert_coverage_metadata_valid,
)


def test_projection_is_deterministic_and_complete():
    rendered = render_artifacts()
    checked_in = {
        path: path.read_bytes()
        for path in (
            list(SCENARIOS_DIR.glob("*.json"))
            + list(TARGETS_DIR.glob("*.json"))
            + list(RESULTS_DIR.glob("*.json"))
            + [
                SUMMARY_PATH,
                TARGET_RISKS_JSON_PATH,
                TARGET_RISKS_MARKDOWN_PATH,
                SUPPORT_GRAPH_JSON_PATH,
                SUPPORT_GRAPH_MARKDOWN_PATH,
            ]
        )
    }
    assert rendered == checked_in
    assert list(SCENARIOS_DIR.glob("*.json"))
    assert "| browser_smoke |" in SUMMARY_PATH.read_text(encoding="utf-8")
    assert "uncovered" in SUMMARY_PATH.read_text(encoding="utf-8")


def test_generated_results_have_replay_diagnostics_and_evidence():
    for path in RESULTS_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["artifact_type"] == "result"
        if record["result"] == "passed":
            assert record["evidence"]
        else:
            assert record["diagnostics"]["reason"]
        if "target_id" in record:
            assert record["seed"] in SEEDS
            assert record["replay"]
            assert record["diagnostics"]["final_mode"]
            assert record["diagnostics"]["final_phase"]
            assert set(record["invariants"]) == set(record["exercised_invariants"])


def test_generated_targets_are_projected_and_linked_bidirectionally():
    target_paths = list(TARGETS_DIR.glob("*.json"))
    assert target_paths
    for path in target_paths:
        target = json.loads(path.read_text(encoding="utf-8"))
        assert target["schema_version"] == SCHEMA_VERSION
        assert target["artifact_type"] == "generation_target"
        assert target["target_id"] == path.stem
        assert target["fields"]
        assert target["invariants"]
        assert target["must_exercise"]
        assert target["result_ids"] == [
            f"{target['target_id']}_seed_{seed}" for seed in SEEDS
        ]
        for result_id in target["result_ids"]:
            result = json.loads(
                (RESULTS_DIR / f"{result_id}.json").read_text(encoding="utf-8")
            )
            assert result["target_id"] == target["target_id"]
            assert result["target_artifact"] == f"targets/{path.name}"
            assert result["scenario_id"] == result_id


def test_disk_artifacts_link_to_known_scenarios_and_catalog():
    from tests.workflow_browser_smoke.coverage_records import BROWSER_SMOKE_COVERAGE
    from tests.workflow_harness.cross_field_scenarios import CROSS_FIELD_SCENARIOS
    from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID
    from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE

    known = {item.scenario_id for item in CROSS_FIELD_SCENARIOS}
    known.update(item.scenario_id for item in REAL_ADAPTER_COVERAGE)
    known.update(item.scenario_id for item in BROWSER_SMOKE_COVERAGE)
    for path in SCENARIOS_DIR.glob("*.json"):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        assert scenario["schema_version"] == SCHEMA_VERSION
        assert scenario["artifact_type"] == "scenario"
        assert scenario["scenario_id"] in known
        assert set(scenario["invariants"]).issubset(INVARIANTS_BY_ID)
        assert (RESULTS_DIR / path.name).exists()
    for path in RESULTS_DIR.glob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        assert set(result["invariants"]).issubset(INVARIANTS_BY_ID)
        if "target_id" not in result:
            assert result["scenario_id"] in known


def test_target_risk_artifacts_are_versioned_ranked_and_linked():
    payload = json.loads(TARGET_RISKS_JSON_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "target_risk_analysis"
    assert len(payload["risks"]) == 12
    assert [risk["rank"] for risk in payload["risks"]] == list(range(1, 13))
    assert [(-risk["gap_score"], risk["risk_id"]) for risk in payload["risks"]] == sorted(
        (-risk["gap_score"], risk["risk_id"]) for risk in payload["risks"]
    )
    for risk in payload["risks"]:
        assert risk["required_fields"]
        assert risk["supporting_invariants"]
        assert risk["required_adapters"]
        assert {item["adapter"] for item in risk["support"]} == set(risk["required_adapters"])
        assert all(
            item["status"] in {"passed", "failed", "skipped", "blocked", "uncovered"}
            for item in risk["support"]
        )
    markdown = TARGET_RISKS_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert "Missing isolated artifacts" in markdown
    assert "real_coordinator: uncovered" in markdown


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seed": -1, "replay": ("start",), "evidence": ("proof",)}, "negative seed"),
        ({"seed": 1, "evidence": ("proof",)}, "seed but no replay"),
        ({"evidence": ()}, "passed without evidence"),
        ({"result": "blocked", "evidence": ()}, "without diagnostics"),
    ],
)
def test_extended_coverage_metadata_rejects_incomplete_records(changes, message):
    values = {
        "scenario_id": "validation_fixture",
        "fields": ("allowed_outputs",),
        "invariants": ("outputs.at_least_one_output_enabled",),
        "adapter": "model",
        "result": "passed",
        "test_file": "tests/workflow_cross_field/test_artifacts.py",
        "evidence": ("proof",),
    }
    values.update(changes)
    with pytest.raises(AssertionError, match=message):
        assert_coverage_metadata_valid(InteractionCoverage(**values))
