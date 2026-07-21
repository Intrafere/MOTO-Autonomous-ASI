"""Test-only adapters for driving real MOTO workflow boundaries.

Exports are lazy so metadata-only registry/deep-run collection does not import production
coordinators. Executing a real adapter still imports the same production boundary normally.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "EventCollector": "dependency_fakes",
    "FakeProofStage": "dependency_fakes",
    "FakeResearchMetadata": "dependency_fakes",
    "RoleConfigCapture": "dependency_fakes",
    "route_workflow_state": "dependency_fakes",
    "RealWorkflowObservation": "observed_state",
    "ManualAggregatorAdapter": "manual_aggregator_adapter",
    "minimal_aggregator_request": "manual_aggregator_adapter",
    "ManualCompilerAdapter": "manual_compiler_adapter",
    "minimal_compiler_request": "manual_compiler_adapter",
    "LeanOJAdapter": "leanoj_adapter",
    "minimal_leanoj_request": "leanoj_adapter",
    "StartOutcome": "cross_mode_adapter",
    "assert_single_race_winner": "cross_mode_adapter",
    "race_starts": "cross_mode_adapter",
    "FatalContextOverflowEvent": "event_assertions",
    "ProofContextOverflowEvent": "event_assertions",
    "assert_event_count": "event_assertions",
    "assert_fatal_context_overflow_event": "event_assertions",
    "assert_no_events": "event_assertions",
    "assert_proof_context_overflow_event": "event_assertions",
    "assert_route_identity": "event_assertions",
    "event_payloads": "event_assertions",
    "SourceTaggedDocument": "source_tagged_rag",
    "SourceTaggedRagIndex": "source_tagged_rag",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value

__all__ = [
    "EventCollector",
    "FakeProofStage",
    "FakeResearchMetadata",
    "RealWorkflowObservation",
    "RoleConfigCapture",
    "route_workflow_state",
    "ManualAggregatorAdapter",
    "minimal_aggregator_request",
    "ManualCompilerAdapter",
    "minimal_compiler_request",
    "LeanOJAdapter",
    "minimal_leanoj_request",
    "StartOutcome",
    "assert_single_race_winner",
    "race_starts",
    "FatalContextOverflowEvent",
    "ProofContextOverflowEvent",
    "assert_event_count",
    "assert_fatal_context_overflow_event",
    "assert_no_events",
    "assert_proof_context_overflow_event",
    "assert_route_identity",
    "event_payloads",
    "SourceTaggedDocument",
    "SourceTaggedRagIndex",
]
