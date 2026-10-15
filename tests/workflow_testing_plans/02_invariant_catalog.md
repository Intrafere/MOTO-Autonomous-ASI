# Workflow Product-Law Invariant Catalog

This document mirrors the stable executable IDs in
`tests/workflow_harness/invariant_catalog.py`. The testing system is a removable
overlay: production code, launch paths, persistence, and runtime behavior must
not import or depend on this catalog, its scenarios, or generated artifacts.

Existing invariant IDs are append-only. Add IDs for new product laws; do not
rename or reuse an existing ID.

## Stable ID projection

The complete stable-ID inventory is generated from
`INVARIANTS_BY_ID`, sorted lexicographically, and checked by
`tests/workflow_cross_field/test_invariant_catalog_docs.py`. This avoids a
second hand-maintained registry drifting from executable coverage.

<!-- BEGIN GENERATED INVARIANT IDS -->

- `api.hosted_desktop_only_routes_unavailable`
- `assistant.disable_clears_live_pack`
- `assistant.no_validator_injection`
- `assistant.non_blocking_retrieval`
- `assistant.stagnant_pack_backoff_not_shutdown`
- `bulk_mutation.atomic_revision_fenced`
- `events.autonomous_mandatory_overflow_checkpoint_preserved`
- `events.context_overflow_persists_across_reload`
- `events.context_overflow_route_identity`
- `events.context_overflow_terminal_stop_once`
- `events.continuous_manual_overflow_terminal_after_drain`
- `events.frontend_events_include_scope_phase`
- `events.proof_context_overflow_nonfatal`
- `events.proof_verified_after_registration`
- `feedback_projection.newest_five_whole`
- `leanoj.parent_control_generation_fences_children`
- `live_activity.all_retry_wait_sources_accounted`
- `live_activity.provider_error_route_identity`
- `live_activity.provider_error_sources_route_attributed`
- `live_activity.provider_retry_each_wait_visible`
- `live_activity.provider_retry_identity_stable`
- `live_activity.provider_retry_persists_reload`
- `live_activity.provider_retry_secret_safe`
- `outputs.at_least_one_output_enabled`
- `outputs.papers_only_skips_proof_work`
- `outputs.proofs_only_no_paper_phase`
- `prompt.direct_sources_excluded_from_rag`
- `prompt.generated_appendices_stripped`
- `prompt.mandatory_source_overflow_fails_visible`
- `prompt.proof_source_context_required`
- `prompt.user_prompt_direct_injected`
- `prompt.validator_excludes_assistant_memory`
- `proof_competition.secondary_after_primary_exhaustion`
- `proof_framing.allowed_outputs_gate`
- `proof_loop.automatic_round_policy_preserved`
- `proof_loop.continuous_explicit_ownership`
- `proof_novelty.group_atomic_same_run`
- `proof_pruning.artifacts_and_future_memory_preserved`
- `proof_pruning.commit_lifecycle_fenced`
- `proof_pruning.context_overflow_truthful`
- `proof_pruning.no_prune_is_valid`
- `proof_pruning.occurrence_scope_isolated`
- `proof_pruning.owning_run_context_excludes_pruned`
- `proof_pruning.review_non_blocking`
- `proof_pruning.semantic_distinct_review_preserves_unique_routes`
- `proof_pruning.validator_gates_automatic_mutation`
- `proof_run.identity_generation_fenced`
- `proof_runtime.candidate_list_autonomous_restart_owned`
- `proof_runtime.candidate_list_manual_process_local`
- `proof_runtime.candidate_list_validator_gate`
- `proof_runtime.hosted_proof_settings_unavailable`
- `proof_runtime.no_lean_when_disabled`
- `proof_runtime.no_smt_when_disabled`
- `proof_runtime.truncation_is_attempt_failure`
- `proof_scope.autonomous_events_not_manual`
- `proof_scope.generated_appendices_stripped_from_prompts`
- `proof_scope.manual_clear_archives_active`
- `proof_scope.manual_not_in_autonomous_current`
- `provider.exact_role_and_cooldown_scope`
- `provider.pause_preserves_checkpoint`
- `provider.reset_wakes_without_corrupting_checkpoint`
- `provider.stop_resume_preserves_pause`
- `runtime.child_tasks_count_as_active`
- `runtime.parent_action_fences_child_outputs`
- `runtime.single_active_workflow`
- `runtime_root.single_owner_bound_paths`
- `sleep.unique_owner_released`
- `solution_path.activation_review_fenced`
- `state.clear_removes_active_preserves_history`
- `state.completed_brainstorm_no_replay_handoff`
- `state.pruned_papers_excluded_from_context`
- `state.runtime_roots_are_active_roots`
- `supercharge_creativity.routing_separation`

<!-- END GENERATED INVARIANT IDS -->

## Build 07 proof-pruning and continuous-loop laws

| Stable ID | Product law | Model scenario | Real-source observation |
|---|---|---|---|
| `proof_runtime.candidate_list_validator_gate` | Every proof round validates the complete candidate list before proof cost, accepts any list with at least one approved novel entry, and forwards only approved novel entries. | Direct invariant-catalog scenario | Candidate-list stage and contract coverage. |
| `proof_runtime.candidate_list_autonomous_restart_owned` | Autonomous candidate-list review persists only for the same run, source, trigger, and round across lifecycle restart. | Direct invariant-catalog scenario | Durable `ResearchMetadata` checkpoint coverage. |
| `proof_runtime.candidate_list_manual_process_local` | Manual candidate-list review is process-local, source-scoped, and cleared at round or terminal cleanup. | Direct invariant-catalog scenario | `ProofRunControl` lifecycle coverage. |
| `proof_pruning.artifacts_and_future_memory_preserved` | Pruning preserves canonical records, visibility, export/certificate and graph access, future-run memory, and SyntheticLib-facing eligibility. | `model_proof_pruning_continuous_contracts` | Proof database and proof-search regression coverage. |
| `proof_pruning.owning_run_context_excludes_pruned` | Owning-run direct/search/Assistant context excludes a pruned occurrence, including reused packs; future runs may retrieve it. | `model_proof_pruning_continuous_contracts` | Proof database and Assistant cache coverage. |
| `proof_pruning.validator_gates_automatic_mutation` | Automatic mutation requires a deterministically valid proposal and independent Validator acceptance. | `model_proof_pruning_continuous_contracts` | Pruning contract/agent coverage. |
| `proof_pruning.semantic_distinct_review_preserves_unique_routes` | Canonically distinct proofs may be suppressed only through bounded semantic evidence citing retained coverage; related but unique routes remain active. | `model_proof_pruning_continuous_contracts` | Semantic snapshot, agent-guard, and commit-fence coverage. |
| `proof_pruning.no_prune_is_valid` | A valid no-prune response is non-mutating and non-error. | `model_proof_pruning_continuous_contracts` | Held-review coordinator coverage. |
| `proof_pruning.review_non_blocking` | Proof solving and serialized registration continue while proposer/Validator review is pending. | `model_proof_pruning_continuous_contracts` | Async pruning coordinator coverage. |
| `proof_pruning.commit_lifecycle_fenced` | Stop, Clear, ownership, generation, revision, hash, dependency, and snapshot mismatches prevent mutation. | `model_proof_pruning_continuous_contracts` | Snapshot commit coverage. |
| `proof_pruning.context_overflow_truthful` | Candidate-local overflow stays nonfatal/deferred and cannot be relabeled as pruning success. | `model_proof_pruning_continuous_contracts` | The exact overflow-during-commit interleaving is `blocked`: no bounded production seam observes it without synthesizing the transition owner. |
| `proof_loop.continuous_explicit_ownership` | Continuous manual mode owns one reservation, exposes Stop and detailed per-round activity, completes normally after ten consecutive valid empty rounds with reset on discovery, has no restart-resume state, and cleans up ownership. | `model_proof_pruning_continuous_contracts` | Real route/manager lifecycle coverage. |
| `proof_loop.automatic_round_policy_preserved` | Autonomous proof rounds use proofs-only max 4, papers-plus-proofs brainstorm max 2, completed-paper max 1, and first-valid-zero exit; manual continuous mode remains separate. | `model_proof_pruning_continuous_contracts` | Real Autonomous coordinator policy matrix. |
| `proof_pruning.occurrence_scope_isolated` | Pruning one occurrence does not suppress canonical matches in another run or source. | `model_proof_pruning_continuous_contracts` | Revisioned proof snapshot coverage. |

The exact production files and executable selectors are maintained in
`tests/workflow_harness/source_mappings.py`. Any interaction that cannot be
observed safely through an existing bounded seam is recorded as `blocked`; it
must not be reported as a real-code pass.

## Existing catalog families

The executable catalog also retains the pre-Build-07 IDs for:

- top-level workflow exclusivity and parent/child fencing;
- Lean/SMT gates and Allowed Outputs;
- provider pause, Stop/Resume, and reset checkpoints;
- Assistant isolation and non-blocking retrieval;
- manual/autonomous proof-scope separation and clear/archive behavior;
- active/history filesystem boundaries and runtime-root containment;
- frontend event attribution and context-overflow lifecycle truth;
- direct prompt/source injection, RAG exclusion, and generated-appendix stripping;
- hosted unavailability for desktop-only routes.

## Deterministic projection

When catalog, scenario, mapping, coverage, or risk inputs change:

```text
python -m tests.workflow_cross_field.artifacts
python -m tests.workflow_cross_field.artifacts --check
```

Generated scenarios, results, support graph, risk analysis, and summary files
are projections only. Production remains fully functional if the entire
testing overlay is deleted.

Direct model scenarios are classified as deterministic model evidence. They
exercise the executable state model and invariant checkers, but do not claim
production-code coverage. Generated model targets are a separate recombinatory
coverage class and may overlap invariant IDs without being treated as duplicate
production evidence.
