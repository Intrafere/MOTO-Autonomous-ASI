# Inverse Target-Risk Analysis

Generated deterministically by `tests.workflow_cross_field.artifacts` from Build A in-memory coverage records.

Schema version: 2.

| Rank | Target risk | Priority | Gap score | Adapter support | Missing isolated artifacts |
|---:|---|---|---:|---|---|
| 1 | `risk.provider_retry_activity_incomplete` — Provider retry activity is missing, ambiguous, unsafe, or ephemeral | critical | 24 | model: uncovered; focused_production: blocked; browser_smoke: uncovered | `model:live_activity.all_retry_wait_sources_accounted`, `focused_production:live_activity.all_retry_wait_sources_accounted`, `browser_smoke:live_activity.all_retry_wait_sources_accounted`, `browser_smoke:live_activity.provider_retry_identity_stable`, `browser_smoke:live_activity.provider_retry_secret_safe` |
| 2 | `risk.hosted_desktop_boundary` — Hosted mode invokes desktop-only behavior | critical | 15 | model: uncovered; real_route: uncovered | `model:api.hosted_desktop_only_routes_unavailable`, `model:proof_runtime.hosted_proof_settings_unavailable`, `real_route:api.hosted_desktop_only_routes_unavailable` |
| 3 | `risk.stale_child_phase_takeover` — Stale child output overrides parent phase | critical | 12 | model: uncovered; real_coordinator: blocked | `model:runtime.parent_action_fences_child_outputs`, `real_coordinator:runtime.parent_action_fences_child_outputs` |
| 4 | `risk.context_overflow_activity_loses_identity` — Context overflow loses route identity or lifecycle semantics | critical | 9 | model: passed; real_coordinator: uncovered | `real_coordinator:events.autonomous_mandatory_overflow_checkpoint_preserved`, `real_coordinator:events.continuous_manual_overflow_terminal_after_drain` |
| 5 | `risk.provider_pause_loses_checkpoint` — Provider pause loses resumable work | critical | 9 | model: passed; real_coordinator: blocked | `real_coordinator:provider.reset_wakes_without_corrupting_checkpoint`, `real_coordinator:provider.stop_resume_preserves_pause` |
| 6 | `risk.assistant_memory_crosses_boundary` — Assistant memory crosses role boundary | high | 8 | model: passed; real_coordinator: uncovered | `real_coordinator:assistant.disable_clears_live_pack`, `real_coordinator:assistant.no_validator_injection`, `real_coordinator:prompt.validator_excludes_assistant_memory` |
| 7 | `risk.generated_proofs_reenter_prompts` — Generated proofs reenter source prompts | high | 8 | model: uncovered; real_coordinator: uncovered | `model:proof_scope.generated_appendices_stripped_from_prompts`, `real_coordinator:prompt.proof_source_context_required` |
| 8 | `risk.pruned_paper_context_leak` — Pruned paper remains in model context | high | 8 | model: uncovered; real_coordinator: uncovered | `model:state.pruned_papers_excluded_from_context`, `real_coordinator:state.pruned_papers_excluded_from_context` |
| 9 | `risk.disabled_proof_runtime_executes` — Disabled proof runtime executes tools | critical | 6 | model: passed; real_coordinator: uncovered | `real_coordinator:proof_runtime.no_smt_when_disabled` |
| 10 | `risk.proofs_only_enters_paper` — Proofs-only run enters paper phases | critical | 6 | model: passed; real_coordinator: uncovered | `real_coordinator:outputs.at_least_one_output_enabled` |
| 11 | `risk.invariant.assistant.non_blocking_retrieval` — Product law is violated: assistant.non_blocking_retrieval | medium | 2 | model: uncovered | `model:assistant.non_blocking_retrieval` |
| 12 | `risk.invariant.assistant.stagnant_pack_backoff_not_shutdown` — Product law is violated: assistant.stagnant_pack_backoff_not_shutdown | medium | 2 | model: uncovered | `model:assistant.stagnant_pack_backoff_not_shutdown` |
| 13 | `risk.invariant.proof_loop.continuous_explicit_ownership` — Product law is violated: proof_loop.continuous_explicit_ownership | medium | 2 | model: passed; real_route: passed; real_coordinator: uncovered | `real_coordinator:proof_loop.continuous_explicit_ownership` |
| 14 | `risk.invariant.proof_pruning.artifacts_and_future_memory_preserved` — Product law is violated: proof_pruning.artifacts_and_future_memory_preserved | medium | 2 | model: passed; focused_production: passed; real_coordinator: uncovered | `real_coordinator:proof_pruning.artifacts_and_future_memory_preserved` |
| 15 | `risk.invariant.proof_pruning.commit_lifecycle_fenced` — Product law is violated: proof_pruning.commit_lifecycle_fenced | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.commit_lifecycle_fenced` |
| 16 | `risk.invariant.proof_pruning.context_overflow_truthful` — Product law is violated: proof_pruning.context_overflow_truthful | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.context_overflow_truthful` |
| 17 | `risk.invariant.proof_pruning.no_prune_is_valid` — Product law is violated: proof_pruning.no_prune_is_valid | medium | 2 | model: passed; focused_production: passed; real_coordinator: uncovered | `real_coordinator:proof_pruning.no_prune_is_valid` |
| 18 | `risk.invariant.proof_pruning.occurrence_scope_isolated` — Product law is violated: proof_pruning.occurrence_scope_isolated | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.occurrence_scope_isolated` |
| 19 | `risk.invariant.proof_pruning.owning_run_context_excludes_pruned` — Product law is violated: proof_pruning.owning_run_context_excludes_pruned | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.owning_run_context_excludes_pruned` |
| 20 | `risk.invariant.proof_pruning.review_non_blocking` — Product law is violated: proof_pruning.review_non_blocking | medium | 2 | model: passed; focused_production: passed; real_coordinator: uncovered | `real_coordinator:proof_pruning.review_non_blocking` |
| 21 | `risk.invariant.proof_pruning.semantic_distinct_review_preserves_unique_routes` — Product law is violated: proof_pruning.semantic_distinct_review_preserves_unique_routes | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.semantic_distinct_review_preserves_unique_routes` |
| 22 | `risk.invariant.proof_pruning.validator_gates_automatic_mutation` — Product law is violated: proof_pruning.validator_gates_automatic_mutation | medium | 2 | model: passed; real_coordinator: blocked | `real_coordinator:proof_pruning.validator_gates_automatic_mutation` |
| 23 | `risk.invariant.proof_runtime.truncation_is_attempt_failure` — Product law is violated: proof_runtime.truncation_is_attempt_failure | medium | 2 | model: uncovered | `model:proof_runtime.truncation_is_attempt_failure` |
| 24 | `risk.invariant.runtime.child_tasks_count_as_active` — Product law is violated: runtime.child_tasks_count_as_active | medium | 2 | model: uncovered; real_route: passed | `model:runtime.child_tasks_count_as_active` |
| 25 | `risk.invariant.state.completed_brainstorm_no_replay_handoff` — Product law is violated: state.completed_brainstorm_no_replay_handoff | medium | 2 | model: uncovered | `model:state.completed_brainstorm_no_replay_handoff` |
| 26 | `risk.invariant.state.runtime_roots_are_active_roots` — Product law is violated: state.runtime_roots_are_active_roots | medium | 2 | model: uncovered; real_coordinator: passed | `model:state.runtime_roots_are_active_roots` |
| 27 | `risk.answer_wrapper_semantic_mix` — Answer wrappers alter routing semantics | high | 0 | model: passed | none |
| 28 | `risk.bulk_mutation_partial_publish` — Bulk mutation partially publishes | critical | 0 | model: passed | none |
| 29 | `risk.candidate_list_ownership_mismatch` — Candidate-list review crosses its ownership boundary | critical | 0 | focused_production: passed | none |
| 30 | `risk.clear_destroys_history` — Clear destroys history or leaves active state | high | 0 | model: passed; real_coordinator: passed | none |
| 31 | `risk.direct_source_duplicated_by_rag` — Direct source is duplicated or displaced by RAG | high | 0 | model: passed; real_coordinator: passed | none |
| 32 | `risk.feedback_projection_corruption` — Feedback projection truncates or drops newest | high | 0 | model: passed | none |
| 33 | `risk.group_novelty_partial_context` — Group novelty becomes partial or cross-run | critical | 0 | model: passed | none |
| 34 | `risk.invariant.events.frontend_events_include_scope_phase` — Product law is violated: events.frontend_events_include_scope_phase | medium | 0 | model: passed | none |
| 35 | `risk.invariant.events.proof_verified_after_registration` — Product law is violated: events.proof_verified_after_registration | medium | 0 | model: passed | none |
| 36 | `risk.invariant.outputs.papers_only_skips_proof_work` — Product law is violated: outputs.papers_only_skips_proof_work | medium | 0 | model: passed | none |
| 37 | `risk.invariant.prompt.user_prompt_direct_injected` — Product law is violated: prompt.user_prompt_direct_injected | medium | 0 | model: passed | none |
| 38 | `risk.invariant.proof_loop.automatic_round_policy_preserved` — Product law is violated: proof_loop.automatic_round_policy_preserved | medium | 0 | model: passed; real_coordinator: passed | none |
| 39 | `risk.invariant.runtime.single_active_workflow` — Product law is violated: runtime.single_active_workflow | medium | 0 | model: passed; real_route: passed; browser_smoke: passed | none |
| 40 | `risk.leanoj_parent_control_stale_commit` — LeanOJ child overrides parent control | critical | 0 | model: passed | none |
| 41 | `risk.proof_competition_wrong_order` — Proof competition violates ordered fallback | high | 0 | model: passed | none |
| 42 | `risk.proof_framing_output_mismatch` — Proof framing ignores output gate | high | 0 | model: passed | none |
| 43 | `risk.proof_run_identity_drift` — Proof run identity drifts across lifecycle | critical | 0 | model: passed | none |
| 44 | `risk.proof_scope_leak` — Proof records leak across scopes | critical | 0 | model: passed; real_coordinator: passed | none |
| 45 | `risk.provider_error_activity_source_unattributed` — Provider-error activity source publishes without route identity | critical | 0 | model: passed; real_coordinator: passed | none |
| 46 | `risk.provider_role_or_cooldown_alias` — Provider role or cooldown scope aliases | critical | 0 | model: passed | none |
| 47 | `risk.runtime_root_double_owner` — Runtime root gains competing owner | critical | 0 | model: passed | none |
| 48 | `risk.sleep_owner_leak` — Sleep inhibition owner leaks | medium | 0 | model: passed | none |
| 49 | `risk.solution_path_stale_commit` — Solution path accepts stale or unreviewed revision | critical | 0 | model: passed | none |
