# Cross-Field Coverage Summary

Generated deterministically by `tests.workflow_cross_field.artifacts`.

Schema version: 2.

| Product law | model | real_route | real_coordinator | browser_smoke |
|---|---|---|---|---|
| `api.hosted_desktop_only_routes_unavailable` | uncovered | uncovered | uncovered | passed |
| `assistant.disable_clears_live_pack` | passed | uncovered | uncovered | uncovered |
| `assistant.no_validator_injection` | passed | uncovered | uncovered | uncovered |
| `assistant.non_blocking_retrieval` | uncovered | uncovered | uncovered | uncovered |
| `assistant.stagnant_pack_backoff_not_shutdown` | uncovered | uncovered | uncovered | uncovered |
| `bulk_mutation.atomic_revision_fenced` | passed | uncovered | uncovered | uncovered |
| `events.autonomous_mandatory_overflow_checkpoint_preserved` | passed | uncovered | uncovered | uncovered |
| `events.context_overflow_persists_across_reload` | passed | uncovered | passed | passed |
| `events.context_overflow_route_identity` | passed | uncovered | passed | passed |
| `events.context_overflow_terminal_stop_once` | passed | uncovered | passed | passed |
| `events.continuous_manual_overflow_terminal_after_drain` | passed | uncovered | uncovered | uncovered |
| `events.frontend_events_include_scope_phase` | passed | uncovered | uncovered | uncovered |
| `events.proof_context_overflow_nonfatal` | passed | uncovered | passed | passed |
| `events.proof_verified_after_registration` | passed | uncovered | blocked (The production final loop combines unbounded model iteration, Lean execution, semantic review, persistence, and registration without a bounded route-level seam. Direct registration ordering is covered elsewhere; publishing a full lifecycle pass would require faking the transition owner rather than observing it.) | uncovered |
| `feedback_projection.newest_five_whole` | passed | uncovered | uncovered | uncovered |
| `leanoj.parent_control_generation_fences_children` | passed | uncovered | uncovered | uncovered |
| `live_activity.all_retry_wait_sources_accounted` | uncovered | uncovered | passed | uncovered |
| `live_activity.provider_error_route_identity` | passed | uncovered | uncovered | passed |
| `live_activity.provider_error_sources_route_attributed` | passed | uncovered | passed | uncovered |
| `live_activity.provider_retry_each_wait_visible` | passed | uncovered | uncovered | passed |
| `live_activity.provider_retry_identity_stable` | passed | uncovered | uncovered | uncovered |
| `live_activity.provider_retry_persists_reload` | passed | uncovered | uncovered | passed |
| `live_activity.provider_retry_secret_safe` | passed | uncovered | uncovered | uncovered |
| `outputs.at_least_one_output_enabled` | passed | passed | uncovered | passed |
| `outputs.papers_only_skips_proof_work` | passed | uncovered | uncovered | uncovered |
| `outputs.proofs_only_no_paper_phase` | passed | uncovered | passed | uncovered |
| `prompt.direct_sources_excluded_from_rag` | passed | uncovered | passed | uncovered |
| `prompt.generated_appendices_stripped` | passed | uncovered | passed | uncovered |
| `prompt.mandatory_source_overflow_fails_visible` | passed | uncovered | passed | uncovered |
| `prompt.proof_source_context_required` | passed | uncovered | uncovered | uncovered |
| `prompt.user_prompt_direct_injected` | passed | uncovered | uncovered | uncovered |
| `prompt.validator_excludes_assistant_memory` | passed | uncovered | uncovered | uncovered |
| `proof_competition.secondary_after_primary_exhaustion` | passed | uncovered | uncovered | uncovered |
| `proof_framing.allowed_outputs_gate` | passed | uncovered | uncovered | uncovered |
| `proof_loop.automatic_round_policy_preserved` | passed | uncovered | passed | uncovered |
| `proof_loop.continuous_explicit_ownership` | passed | passed | uncovered | uncovered |
| `proof_novelty.group_atomic_same_run` | passed | uncovered | uncovered | uncovered |
| `proof_pruning.artifacts_and_future_memory_preserved` | passed | uncovered | uncovered | uncovered |
| `proof_pruning.commit_lifecycle_fenced` | passed | uncovered | blocked (The focused snapshot test proves target/revision rechecks, but not the lifecycle-generation fence included in the invariant.) | uncovered |
| `proof_pruning.context_overflow_truthful` | passed | uncovered | blocked (Existing isolated tests observe nonfatal proof overflow and pruning triggers separately; no bounded seam observes an overflow arriving during a prune commit without synthesizing the transition owner.) | uncovered |
| `proof_pruning.no_prune_is_valid` | passed | uncovered | uncovered | uncovered |
| `proof_pruning.occurrence_scope_isolated` | passed | uncovered | blocked (The focused snapshot test proves same-run review exclusion only, not the full occurrence and owning-run isolation contract.) | uncovered |
| `proof_pruning.owning_run_context_excludes_pruned` | passed | uncovered | blocked (The linked focused test proves one cached-pack filter, not every owning-run direct-context and Assistant reuse surface claimed by the invariant.) | uncovered |
| `proof_pruning.review_non_blocking` | passed | uncovered | uncovered | uncovered |
| `proof_pruning.semantic_distinct_review_preserves_unique_routes` | passed | uncovered | blocked (Focused unit tests cover durable retained-support protection and section arbitration, but no bounded real-coordinator adapter yet observes canonically distinct semantic routes through proposal, validation, commit, and a later support-prune attempt.) | uncovered |
| `proof_pruning.validator_gates_automatic_mutation` | passed | uncovered | blocked (The focused contract test rejects Validator retargeting, but does not observe the complete proposal-to-validation-to-mutation gate.) | uncovered |
| `proof_run.identity_generation_fenced` | passed | uncovered | uncovered | uncovered |
| `proof_runtime.candidate_list_autonomous_restart_owned` | uncovered | uncovered | uncovered | uncovered |
| `proof_runtime.candidate_list_manual_process_local` | uncovered | uncovered | uncovered | uncovered |
| `proof_runtime.candidate_list_validator_gate` | uncovered | uncovered | uncovered | uncovered |
| `proof_runtime.hosted_proof_settings_unavailable` | uncovered | passed | uncovered | uncovered |
| `proof_runtime.no_lean_when_disabled` | passed | uncovered | passed | uncovered |
| `proof_runtime.no_smt_when_disabled` | passed | uncovered | uncovered | uncovered |
| `proof_runtime.truncation_is_attempt_failure` | uncovered | uncovered | uncovered | uncovered |
| `proof_scope.autonomous_events_not_manual` | passed | uncovered | passed | uncovered |
| `proof_scope.generated_appendices_stripped_from_prompts` | uncovered | uncovered | passed | uncovered |
| `proof_scope.manual_clear_archives_active` | passed | passed | passed | uncovered |
| `proof_scope.manual_not_in_autonomous_current` | passed | passed | passed | uncovered |
| `provider.exact_role_and_cooldown_scope` | passed | uncovered | uncovered | uncovered |
| `provider.pause_preserves_checkpoint` | passed | uncovered | passed | uncovered |
| `provider.reset_wakes_without_corrupting_checkpoint` | passed | uncovered | uncovered | uncovered |
| `provider.stop_resume_preserves_pause` | passed | uncovered | blocked (Existing coordinator coverage observes pause/reset resume, but no bounded test seam independently interleaves user Stop while the provider wait is active.) | uncovered |
| `runtime.child_tasks_count_as_active` | uncovered | passed | uncovered | uncovered |
| `runtime.parent_action_fences_child_outputs` | uncovered | uncovered | blocked (The invariant catalog declares model-only support and no existing isolated production seam exposes stale child output acceptance after parent takeover.) | uncovered |
| `runtime.single_active_workflow` | passed | passed | uncovered | passed |
| `runtime_root.single_owner_bound_paths` | passed | uncovered | uncovered | uncovered |
| `sleep.unique_owner_released` | passed | uncovered | uncovered | uncovered |
| `solution_path.activation_review_fenced` | passed | uncovered | uncovered | uncovered |
| `state.clear_removes_active_preserves_history` | passed | passed | passed | uncovered |
| `state.completed_brainstorm_no_replay_handoff` | uncovered | uncovered | uncovered | uncovered |
| `state.pruned_papers_excluded_from_context` | uncovered | passed | uncovered | uncovered |
| `state.runtime_roots_are_active_roots` | uncovered | uncovered | passed | uncovered |
| `supercharge_creativity.routing_separation` | passed | uncovered | uncovered | uncovered |
