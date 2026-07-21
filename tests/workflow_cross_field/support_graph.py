from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from tests.workflow_cross_field.target_risks import TARGET_RISKS
from tests.workflow_harness.coverage_metadata import InteractionCoverage
from tests.workflow_harness.invariant_catalog import INVARIANTS_BY_ID


SUPPORT_GRAPH_SCHEMA_VERSION = 1
SUPPORT_GRAPH_JSON_PATH = Path(__file__).parent / "support_graph.json"
SUPPORT_GRAPH_MARKDOWN_PATH = Path(__file__).parent / "SUPPORT_GRAPH.md"

SupportStatus = Literal["passed", "failed", "skipped", "blocked", "uncovered"]
ADAPTERS = ("model", "real_route", "real_coordinator", "browser_smoke")
STATUS_ORDER = {"failed": 0, "blocked": 1, "skipped": 2, "passed": 3, "uncovered": 4}


@dataclass(frozen=True)
class SupportEdge:
    risk_id: str
    invariant_id: str
    field: str
    scenario_id: str | None
    adapter: str
    status: SupportStatus
    declared: bool
    observed: bool
    test_file: str | None = None
    reason: str | None = None


def build_support_graph(records: Iterable[InteractionCoverage]) -> dict[str, object]:
    ordered_records = tuple(sorted(records, key=_record_key))
    edges: list[SupportEdge] = []
    gaps: list[dict[str, object]] = []

    for risk in sorted(TARGET_RISKS, key=lambda item: item.risk_id):
        for invariant_id in sorted(risk.supporting_invariants):
            invariant = INVARIANTS_BY_ID[invariant_id]
            for adapter in ADAPTERS:
                matches = tuple(
                    record
                    for record in ordered_records
                    if record.adapter == adapter and invariant_id in record.invariants
                )
                declared = adapter in invariant.adapters
                if matches:
                    for record in matches:
                        reason = _reason(record)
                        edges.append(
                            SupportEdge(
                                risk_id=risk.risk_id,
                                invariant_id=invariant_id,
                                field=invariant.field,
                                scenario_id=record.scenario_id,
                                adapter=adapter,
                                status=record.result,
                                declared=declared,
                                observed=True,
                                test_file=record.test_file,
                                reason=reason,
                            )
                        )
                else:
                    edges.append(
                        SupportEdge(
                            risk_id=risk.risk_id,
                            invariant_id=invariant_id,
                            field=invariant.field,
                            scenario_id=None,
                            adapter=adapter,
                            status="uncovered",
                            declared=declared,
                            observed=False,
                        )
                    )
                    if adapter in risk.required_adapters:
                        gaps.append(
                            {
                                "gap_id": f"{risk.risk_id}:{invariant_id}:{adapter}",
                                "risk_id": risk.risk_id,
                                "invariant_id": invariant_id,
                                "field": invariant.field,
                                "adapter": adapter,
                                "status": "uncovered",
                                "declared": declared,
                                "reason": (
                                    "No observed coverage record exists for this required "
                                    "risk/invariant/adapter basis."
                                ),
                            }
                        )

    edge_payloads = [_edge_payload(edge) for edge in sorted(edges, key=_edge_key)]
    scenario_nodes = sorted(
        {
            record.scenario_id: {
                "scenario_id": record.scenario_id,
                "adapter": record.adapter,
                "result": record.result,
                "test_file": record.test_file,
            }
            for record in ordered_records
        }.values(),
        key=lambda item: item["scenario_id"],
    )
    return {
        "schema_version": SUPPORT_GRAPH_SCHEMA_VERSION,
        "artifact_type": "support_graph",
        "basis": {
            "declared": "Invariant catalog adapter declarations and target-risk requirements.",
            "observed": "InteractionCoverage records produced by executable tests or truthful gap records.",
        },
        "nodes": {
            "risks": [
                {"risk_id": risk.risk_id, "title": risk.title, "priority": risk.priority}
                for risk in sorted(TARGET_RISKS, key=lambda item: item.risk_id)
            ],
            "invariants": [
                {
                    "invariant_id": invariant_id,
                    "field": spec.field,
                    "declared_adapters": sorted(spec.adapters),
                }
                for invariant_id, spec in sorted(INVARIANTS_BY_ID.items())
            ],
            "scenarios": scenario_nodes,
            "adapters": list(ADAPTERS),
        },
        "edges": edge_payloads,
        "gaps": sorted(gaps, key=lambda item: item["gap_id"]),
    }


def validate_support_graph(graph: dict[str, object]) -> None:
    if graph.get("schema_version") != SUPPORT_GRAPH_SCHEMA_VERSION:
        raise AssertionError("Unexpected support graph schema version.")
    edges = graph.get("edges")
    gaps = graph.get("gaps")
    if not isinstance(edges, list) or not edges:
        raise AssertionError("Support graph must contain edges.")
    if not isinstance(gaps, list):
        raise AssertionError("Support graph gaps must be a list.")
    edge_keys: set[tuple[object, ...]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise AssertionError("Support graph edge must be an object.")
        required = {"risk_id", "invariant_id", "field", "adapter", "status", "declared", "observed"}
        if not required.issubset(edge):
            raise AssertionError(f"Support graph edge is missing fields: {required - set(edge)}")
        if edge["status"] not in STATUS_ORDER:
            raise AssertionError(f"Unknown support status {edge['status']!r}.")
        if edge["observed"] and not edge.get("scenario_id"):
            raise AssertionError("Observed support edge must identify its scenario.")
        if not edge["observed"] and edge["status"] != "uncovered":
            raise AssertionError("Unobserved support edge must be uncovered.")
        key = tuple(edge.get(name) for name in ("risk_id", "invariant_id", "adapter", "scenario_id"))
        if key in edge_keys:
            raise AssertionError(f"Duplicate support edge {key!r}.")
        edge_keys.add(key)
    gap_ids = [gap.get("gap_id") for gap in gaps if isinstance(gap, dict)]
    if len(gap_ids) != len(set(gap_ids)):
        raise AssertionError("Support graph gap ids must be unique.")


def render_support_graph_json(records: Iterable[InteractionCoverage]) -> bytes:
    graph = build_support_graph(records)
    validate_support_graph(graph)
    return (json.dumps(graph, indent=2, sort_keys=True) + "\n").encode()


def render_support_graph_markdown(records: Iterable[InteractionCoverage]) -> bytes:
    graph = build_support_graph(records)
    validate_support_graph(graph)
    lines = [
        "# Workflow Support Graph",
        "",
        "Generated deterministically from separately versioned declared and observed support bases.",
        "",
        f"Schema version: {SUPPORT_GRAPH_SCHEMA_VERSION}.",
        "",
        "| Risk | Invariant | Field | Adapter | Status | Basis | Scenario / reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for edge in graph["edges"]:
        basis = f"declared={'yes' if edge['declared'] else 'no'}, observed={'yes' if edge['observed'] else 'no'}"
        detail = edge.get("scenario_id") or edge.get("reason") or "no observed scenario"
        lines.append(
            f"| `{edge['risk_id']}` | `{edge['invariant_id']}` | `{edge['field']}` | "
            f"`{edge['adapter']}` | {edge['status']} | {basis} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Structured Gaps",
            "",
            "| Gap | Status | Declared | Reason |",
            "|---|---|---|---|",
        ]
    )
    for gap in graph["gaps"]:
        lines.append(
            f"| `{gap['gap_id']}` | {gap['status']} | "
            f"{'yes' if gap['declared'] else 'no'} | {gap['reason']} |"
        )
    if not graph["gaps"]:
        lines.append("| none | passed | n/a | All required bases are observed. |")
    return ("\n".join(lines) + "\n").encode()


def _reason(record: InteractionCoverage) -> str | None:
    if record.result not in {"blocked", "skipped", "failed"} or not record.diagnostics:
        return None
    return record.diagnostics.get("reason")


def _record_key(record: InteractionCoverage) -> tuple[object, ...]:
    return record.scenario_id, record.adapter, record.result, record.test_file


def _edge_key(edge: SupportEdge) -> tuple[object, ...]:
    return (
        edge.risk_id,
        edge.invariant_id,
        edge.field,
        edge.adapter,
        STATUS_ORDER[edge.status],
        edge.scenario_id or "",
    )


def _edge_payload(edge: SupportEdge) -> dict[str, object]:
    payload: dict[str, object] = {
        "risk_id": edge.risk_id,
        "invariant_id": edge.invariant_id,
        "field": edge.field,
        "scenario_id": edge.scenario_id,
        "adapter": edge.adapter,
        "status": edge.status,
        "declared": edge.declared,
        "observed": edge.observed,
        "test_file": edge.test_file,
    }
    if edge.reason:
        payload["reason"] = edge.reason
    return payload
