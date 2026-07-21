# Inverse Target-Risk Analysis

Generated deterministically by `tests.workflow_cross_field.artifacts` from Build A in-memory coverage records.

Schema version: 1.

| Rank | Target risk | Priority | Gap score | Adapter support | Missing isolated artifacts |
|---:|---|---|---:|---|---|
| 1 | `risk.hosted_desktop_boundary` — Hosted mode invokes desktop-only behavior | critical | 15 | model: uncovered; real_route: uncovered | `model:api.hosted_desktop_only_routes_unavailable`, `model:proof_runtime.hosted_proof_settings_unavailable`, `real_route:api.hosted_desktop_only_routes_unavailable` |
| 2 | `risk.stale_child_phase_takeover` — Stale child output overrides parent phase | critical | 12 | model: uncovered; real_coordinator: blocked | `model:runtime.parent_action_fences_child_outputs`, `real_coordinator:runtime.parent_action_fences_child_outputs` |
| 3 | `risk.provider_pause_loses_checkpoint` — Provider pause loses resumable work | critical | 9 | model: passed; real_coordinator: blocked | `real_coordinator:provider.reset_wakes_without_corrupting_checkpoint`, `real_coordinator:provider.stop_resume_preserves_pause` |
| 4 | `risk.assistant_memory_crosses_boundary` — Assistant memory crosses role boundary | high | 8 | model: passed; real_coordinator: uncovered | `real_coordinator:assistant.disable_clears_live_pack`, `real_coordinator:assistant.no_validator_injection`, `real_coordinator:prompt.validator_excludes_assistant_memory` |
| 5 | `risk.generated_proofs_reenter_prompts` — Generated proofs reenter source prompts | high | 8 | model: uncovered; real_coordinator: uncovered | `model:proof_scope.generated_appendices_stripped_from_prompts`, `real_coordinator:prompt.proof_source_context_required` |
| 6 | `risk.pruned_paper_context_leak` — Pruned paper remains in model context | high | 8 | model: uncovered; real_coordinator: uncovered | `model:state.pruned_papers_excluded_from_context`, `real_coordinator:state.pruned_papers_excluded_from_context` |
| 7 | `risk.disabled_proof_runtime_executes` — Disabled proof runtime executes tools | critical | 6 | model: passed; real_coordinator: uncovered | `real_coordinator:proof_runtime.no_smt_when_disabled` |
| 8 | `risk.proofs_only_enters_paper` — Proofs-only run enters paper phases | critical | 6 | model: passed; real_coordinator: uncovered | `real_coordinator:outputs.at_least_one_output_enabled` |
| 9 | `risk.clear_destroys_history` — Clear destroys history or leaves active state | high | 0 | model: passed; real_coordinator: passed | none |
| 10 | `risk.context_overflow_activity_loses_identity` — Context overflow loses route identity or lifecycle semantics | critical | 0 | model: passed; real_coordinator: passed | none |
| 11 | `risk.direct_source_duplicated_by_rag` — Direct source is duplicated or displaced by RAG | high | 0 | model: passed; real_coordinator: passed | none |
| 12 | `risk.proof_scope_leak` — Proof records leak across scopes | critical | 0 | model: passed; real_coordinator: passed | none |
