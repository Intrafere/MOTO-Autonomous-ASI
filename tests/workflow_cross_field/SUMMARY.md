# Cross-Field Coverage Summary

Generated deterministically by `tests.workflow_cross_field.artifacts`.

Schema version: 1.

| Product law | model | real_route | real_coordinator | browser_smoke |
|---|---|---|---|---|
| `api.hosted_desktop_only_routes_unavailable` | uncovered | uncovered | uncovered | passed |
| `assistant.disable_clears_live_pack` | passed | uncovered | uncovered | uncovered |
| `assistant.no_validator_injection` | passed | uncovered | uncovered | uncovered |
| `assistant.non_blocking_retrieval` | uncovered | uncovered | uncovered | uncovered |
| `assistant.stagnant_pack_backoff_not_shutdown` | uncovered | uncovered | uncovered | uncovered |
| `events.context_overflow_persists_across_reload` | passed | uncovered | passed | passed |
| `events.context_overflow_route_identity` | passed | uncovered | passed | passed |
| `events.context_overflow_terminal_stop_once` | passed | uncovered | passed | passed |
| `events.frontend_events_include_scope_phase` | passed | uncovered | uncovered | uncovered |
| `events.proof_context_overflow_nonfatal` | passed | uncovered | passed | passed |
| `events.proof_verified_after_registration` | passed | uncovered | blocked (The production final loop combines unbounded model iteration, Lean execution, semantic review, persistence, and registration without a bounded route-level seam. Direct registration ordering is covered elsewhere; publishing a full lifecycle pass would require faking the transition owner rather than observing it.) | uncovered |
| `outputs.at_least_one_output_enabled` | passed | passed | uncovered | passed |
| `outputs.papers_only_skips_proof_work` | passed | uncovered | uncovered | uncovered |
| `outputs.proofs_only_no_paper_phase` | passed | uncovered | passed | uncovered |
| `prompt.direct_sources_excluded_from_rag` | passed | uncovered | passed | uncovered |
| `prompt.generated_appendices_stripped` | passed | uncovered | passed | uncovered |
| `prompt.mandatory_source_overflow_fails_visible` | passed | uncovered | passed | uncovered |
| `prompt.proof_source_context_required` | passed | uncovered | uncovered | uncovered |
| `prompt.user_prompt_direct_injected` | passed | uncovered | uncovered | uncovered |
| `prompt.validator_excludes_assistant_memory` | passed | uncovered | uncovered | uncovered |
| `proof_runtime.hosted_proof_settings_unavailable` | uncovered | passed | uncovered | uncovered |
| `proof_runtime.no_lean_when_disabled` | passed | uncovered | passed | uncovered |
| `proof_runtime.no_smt_when_disabled` | passed | uncovered | uncovered | uncovered |
| `proof_runtime.truncation_is_attempt_failure` | uncovered | uncovered | uncovered | uncovered |
| `proof_scope.autonomous_events_not_manual` | passed | uncovered | passed | uncovered |
| `proof_scope.generated_appendices_stripped_from_prompts` | uncovered | uncovered | passed | uncovered |
| `proof_scope.manual_clear_archives_active` | passed | passed | passed | uncovered |
| `proof_scope.manual_not_in_autonomous_current` | passed | passed | passed | uncovered |
| `provider.pause_preserves_checkpoint` | passed | uncovered | passed | uncovered |
| `provider.reset_wakes_without_corrupting_checkpoint` | passed | uncovered | uncovered | uncovered |
| `provider.stop_resume_preserves_pause` | passed | uncovered | blocked (Existing coordinator coverage observes pause/reset resume, but no bounded test seam independently interleaves user Stop while the provider wait is active.) | uncovered |
| `runtime.child_tasks_count_as_active` | uncovered | passed | uncovered | uncovered |
| `runtime.parent_action_fences_child_outputs` | uncovered | uncovered | blocked (The invariant catalog declares model-only support and no existing isolated production seam exposes stale child output acceptance after parent takeover.) | uncovered |
| `runtime.single_active_workflow` | passed | passed | uncovered | passed |
| `state.clear_removes_active_preserves_history` | passed | passed | passed | uncovered |
| `state.completed_brainstorm_no_replay_handoff` | uncovered | uncovered | uncovered | uncovered |
| `state.pruned_papers_excluded_from_context` | uncovered | passed | uncovered | uncovered |
| `state.runtime_roots_are_active_roots` | uncovered | uncovered | passed | uncovered |
