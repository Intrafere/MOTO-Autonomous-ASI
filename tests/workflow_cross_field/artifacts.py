from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tests.workflow_harness.coverage_metadata import AdapterType, InteractionCoverage
from tests.workflow_harness.cross_field_scenarios import CROSS_FIELD_SCENARIOS
from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID
from tests.workflow_harness.model import WorkflowModel
from tests.workflow_harness.recombinatory_generator import (
    DEFAULT_TARGETS,
    run_generated_scenario,
)
from tests.workflow_browser_smoke.coverage_records import BROWSER_SMOKE_COVERAGE
from tests.workflow_real_adapters.coverage_records import REAL_ADAPTER_COVERAGE
from tests.workflow_cross_field.target_risks import (
    TARGET_RISK_SCHEMA_VERSION,
    analyze_target_risks,
)
from tests.workflow_cross_field.support_graph import (
    SUPPORT_GRAPH_JSON_PATH,
    SUPPORT_GRAPH_MARKDOWN_PATH,
    render_support_graph_json,
    render_support_graph_markdown,
)


ROOT = Path(__file__).parent
SCENARIOS_DIR = ROOT / "scenarios"
TARGETS_DIR = ROOT / "targets"
RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = ROOT / "SUMMARY.md"
TARGET_RISKS_JSON_PATH = ROOT / "target_risks.json"
TARGET_RISKS_MARKDOWN_PATH = ROOT / "TARGET_RISKS.md"
SEEDS = (7, 29)
SCHEMA_VERSION = 1
ADAPTERS: tuple[AdapterType, ...] = (
    "model",
    "real_route",
    "real_coordinator",
    "browser_smoke",
)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def render_artifacts() -> dict[Path, bytes]:
    rendered: dict[Path, bytes] = {}
    records: list[InteractionCoverage] = [
        *REAL_ADAPTER_COVERAGE,
        *BROWSER_SMOKE_COVERAGE,
    ]

    for scenario in CROSS_FIELD_SCENARIOS:
        path = SCENARIOS_DIR / f"{scenario.scenario_id}.json"
        rendered[path] = _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "scenario",
                "scenario_id": scenario.scenario_id,
                "adapter": scenario.adapter,
                "fields": scenario.fields,
                "invariants": scenario.invariants,
                "actions": tuple(action.__name__ for action in scenario.actions),
                "must_exercise": scenario.must_exercise,
                "expected_result": scenario.expected_result,
            }
        )
        record = scenario.coverage_record("tests/workflow_scenarios/test_cross_field_analysis.py")
        records.append(record)
        result_path = RESULTS_DIR / f"{scenario.scenario_id}.json"
        rendered[result_path] = _json_bytes(_result_payload(record, source="curated"))

    for target in DEFAULT_TARGETS:
        result_ids = tuple(f"{target.target_id}_seed_{seed}" for seed in SEEDS)
        rendered[TARGETS_DIR / f"{target.target_id}.json"] = _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "generation_target",
                "target_id": target.target_id,
                "fields": target.fields,
                "invariants": target.invariants,
                "must_exercise": target.must_exercise,
                "max_steps": target.max_steps,
                "result_ids": result_ids,
            }
        )

    with tempfile.TemporaryDirectory(prefix="moto-workflow-artifacts-"):
        # The executable model performs only in-memory/path-containment checks here. A stable
        # logical root keeps planner signatures and generated artifacts process-independent.
        runtime_root = Path("/moto-workflow-artifacts")
        for target in DEFAULT_TARGETS:
            for seed in SEEDS:
                run = run_generated_scenario(
                    WorkflowModel(runtime_root=runtime_root / target.target_id / str(seed)),
                    target,
                    seed=seed,
                )
                scenario_id = f"{target.target_id}_seed_{seed}"
                path = RESULTS_DIR / f"{scenario_id}.json"
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "result",
                    "scenario_id": scenario_id,
                    "target_id": target.target_id,
                    "target_artifact": f"targets/{target.target_id}.json",
                    "adapter": "model",
                    "result": "passed",
                    "seed": seed,
                    "fields": run.fields,
                    "observed_fields": sorted(run.observed_fields),
                    "invariants": run.invariants,
                    "exercised_invariants": sorted(run.exercised_invariants),
                    "action_ids": run.action_ids,
                    "replay": run.replay,
                    "evidence": sorted(run.observed_evidence),
                    "diagnostics": {
                        "final_mode": run.final_mode.value,
                        "final_phase": run.final_phase.value,
                    },
                }
                rendered[path] = _json_bytes(payload)
                records.append(
                    InteractionCoverage(
                        scenario_id=scenario_id,
                        fields=run.fields,
                        invariants=run.invariants,
                        adapter="model",
                        result="passed",
                        test_file="tests/workflow_recombinatory/test_mode_aware_generator.py",
                        seed=seed,
                        replay=run.replay,
                        diagnostics=payload["diagnostics"],
                        evidence=tuple(sorted(run.observed_evidence)),
                    )
                )

    for record in REAL_ADAPTER_COVERAGE:
        rendered[RESULTS_DIR / f"{record.scenario_id}.json"] = _json_bytes(
            _result_payload(record, source="real_adapter")
        )
    for record in BROWSER_SMOKE_COVERAGE:
        rendered[RESULTS_DIR / f"{record.scenario_id}.json"] = _json_bytes(
            _result_payload(record, source="browser_smoke")
        )

    lines = [
        "# Cross-Field Coverage Summary",
        "",
        "Generated deterministically by `tests.workflow_cross_field.artifacts`.",
        "",
        f"Schema version: {SCHEMA_VERSION}.",
        "",
        "| Product law | model | real_route | real_coordinator | browser_smoke |",
        "|---|---|---|---|---|",
    ]
    for invariant in sorted(INVARIANTS_BY_ID):
        cells = [_status_cell(records, invariant, adapter) for adapter in ADAPTERS]
        lines.append(f"| `{invariant}` | {' | '.join(cells)} |")
    rendered[SUMMARY_PATH] = ("\n".join(lines) + "\n").encode()
    analyses = analyze_target_risks(records)
    rendered[TARGET_RISKS_JSON_PATH] = _json_bytes(
        {
            "schema_version": TARGET_RISK_SCHEMA_VERSION,
            "artifact_type": "target_risk_analysis",
            "risks": [_risk_payload(analysis) for analysis in analyses],
        }
    )
    risk_lines = [
        "# Inverse Target-Risk Analysis",
        "",
        "Generated deterministically by `tests.workflow_cross_field.artifacts` from Build A in-memory coverage records.",
        "",
        f"Schema version: {TARGET_RISK_SCHEMA_VERSION}.",
        "",
        "| Rank | Target risk | Priority | Gap score | Adapter support | Missing isolated artifacts |",
        "|---:|---|---|---:|---|---|",
    ]
    for analysis in analyses:
        support = "; ".join(f"{item.adapter}: {item.status}" for item in analysis.support)
        missing = ", ".join(f"`{item}`" for item in analysis.missing_isolated_artifacts) or "none"
        risk_lines.append(
            f"| {analysis.rank} | `{analysis.risk.risk_id}` — {analysis.risk.title} | "
            f"{analysis.risk.priority} | {analysis.gap_score} | {support} | {missing} |"
        )
    rendered[TARGET_RISKS_MARKDOWN_PATH] = ("\n".join(risk_lines) + "\n").encode()
    rendered[SUPPORT_GRAPH_JSON_PATH] = render_support_graph_json(records)
    rendered[SUPPORT_GRAPH_MARKDOWN_PATH] = render_support_graph_markdown(records)
    return rendered


def _risk_payload(analysis) -> dict[str, object]:
    return {
        "risk_id": analysis.risk.risk_id,
        "title": analysis.risk.title,
        "description": analysis.risk.description,
        "priority": analysis.risk.priority,
        "rank": analysis.rank,
        "gap_score": analysis.gap_score,
        "required_fields": analysis.risk.required_fields,
        "supporting_invariants": analysis.risk.supporting_invariants,
        "required_adapters": analysis.risk.required_adapters,
        "existing_scenarios": analysis.existing_scenarios,
        "missing_isolated_artifacts": analysis.missing_isolated_artifacts,
        "support": [
            {
                "adapter": item.adapter,
                "status": item.status,
                "result_statuses": item.result_statuses,
                "scenario_ids": item.scenario_ids,
                "missing_invariants": item.missing_invariants,
            }
            for item in analysis.support
        ],
    }


def _result_payload(record: InteractionCoverage, *, source: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "result",
        "source": source,
        "scenario_id": record.scenario_id,
        "adapter": record.adapter,
        "result": record.result,
        "fields": record.fields,
        "invariants": record.invariants,
        "test_file": record.test_file,
        "seed": record.seed,
        "replay": record.replay,
        "diagnostics": record.diagnostics or {},
        "evidence": record.evidence,
    }


def _status_cell(
    records: list[InteractionCoverage], invariant: str, adapter: AdapterType
) -> str:
    matches = [
        record for record in records
        if invariant in record.invariants and record.adapter == adapter
    ]
    if not matches:
        return "uncovered"
    statuses = sorted({record.result for record in matches})
    reasons = sorted({
        str(record.diagnostics.get("reason"))
        for record in matches
        if record.result in {"blocked", "skipped"}
        and record.diagnostics
        and record.diagnostics.get("reason")
    })
    return ", ".join(statuses) + (f" ({'; '.join(reasons)})" if reasons else "")


def write_artifacts(*, check: bool = False) -> int:
    rendered = render_artifacts()
    stale = [path for path, content in rendered.items() if not path.exists() or path.read_bytes() != content]
    expected = set(rendered)
    extras = (
        set(SCENARIOS_DIR.glob("*.json"))
        | set(TARGETS_DIR.glob("*.json"))
        | set(RESULTS_DIR.glob("*.json"))
    )
    stale.extend(sorted(extras - expected))
    if check:
        return 1 if stale else 0
    for path in extras - expected:
        path.unlink()
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raise SystemExit(write_artifacts(check=args.check))
