# Workflow Support Graph

Generated deterministically from separately versioned declared and observed support bases.

Schema version: 1.

| Risk | Invariant | Field | Adapter | Status | Basis | Scenario / reason |
|---|---|---|---|---|---|---|
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_29 |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_7 |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | model_assistant_prompt_validator_exclusion |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | model_manual_proof_clear_scope_assistant |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `assistant.disable_clears_live_pack` | `assistant_memory` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_29 |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_7 |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `model` | passed | declared=yes, observed=yes | model_assistant_prompt_validator_exclusion |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `assistant.no_validator_injection` | `assistant_memory` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_29 |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_assistant_prompt_context_seed_7 |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `model` | passed | declared=yes, observed=yes | model_assistant_prompt_validator_exclusion |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.assistant_memory_crosses_boundary` | `prompt.validator_excludes_assistant_memory` | `prompt_context` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_29 |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_7 |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | model_manual_proof_clear_scope_assistant |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_archive_and_leanoj_clear_preserve_scope_contract |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_route` | passed | declared=yes, observed=yes | real_manual_aggregator_clear_archives_before_clear |
| `risk.clear_destroys_history` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_route` | passed | declared=yes, observed=yes | real_manual_compiler_papers_lifecycle_and_clear_archive |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_29 |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_7 |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `model` | passed | declared=yes, observed=yes | model_manual_proof_clear_scope_assistant |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_archive_and_leanoj_clear_preserve_scope_contract |
| `risk.clear_destroys_history` | `state.clear_removes_active_preserves_history` | `workflow_filesystem_state` | `real_route` | passed | declared=yes, observed=yes | real_manual_compiler_papers_lifecycle_and_clear_archive |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_persists_across_reload` | `websocket_api_contracts` | `browser_smoke` | passed | declared=yes, observed=yes | browser_overflow_activity_persists_with_attribution_and_terminal_deduplication |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_persists_across_reload` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_29 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_persists_across_reload` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_7 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_persists_across_reload` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_aggregator_overflow_identity_persists_across_reload |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_persists_across_reload` | `websocket_api_contracts` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `browser_smoke` | passed | declared=yes, observed=yes | browser_overflow_activity_persists_with_attribution_and_terminal_deduplication |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_29 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_7 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_autonomous_overflow_terminal_stop_once |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_aggregator_overflow_identity_persists_across_reload |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_proof_context_overflow_is_scoped_nonfatal |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_route_identity` | `websocket_api_contracts` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_terminal_stop_once` | `websocket_api_contracts` | `browser_smoke` | passed | declared=yes, observed=yes | browser_overflow_activity_persists_with_attribution_and_terminal_deduplication |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_terminal_stop_once` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_29 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_terminal_stop_once` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_7 |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_terminal_stop_once` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_autonomous_overflow_terminal_stop_once |
| `risk.context_overflow_activity_loses_identity` | `events.context_overflow_terminal_stop_once` | `websocket_api_contracts` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.context_overflow_activity_loses_identity` | `events.proof_context_overflow_nonfatal` | `websocket_api_contracts` | `browser_smoke` | passed | declared=yes, observed=yes | browser_overflow_activity_persists_with_attribution_and_terminal_deduplication |
| `risk.context_overflow_activity_loses_identity` | `events.proof_context_overflow_nonfatal` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_29 |
| `risk.context_overflow_activity_loses_identity` | `events.proof_context_overflow_nonfatal` | `websocket_api_contracts` | `model` | passed | declared=yes, observed=yes | generated_context_overflow_contract_seed_7 |
| `risk.context_overflow_activity_loses_identity` | `events.proof_context_overflow_nonfatal` | `websocket_api_contracts` | `real_coordinator` | passed | declared=yes, observed=yes | real_proof_context_overflow_is_scoped_nonfatal |
| `risk.context_overflow_activity_loses_identity` | `events.proof_context_overflow_nonfatal` | `websocket_api_contracts` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.direct_source_duplicated_by_rag` | `prompt.direct_sources_excluded_from_rag` | `prompt_context` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.direct_source_duplicated_by_rag` | `prompt.direct_sources_excluded_from_rag` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_29 |
| `risk.direct_source_duplicated_by_rag` | `prompt.direct_sources_excluded_from_rag` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_7 |
| `risk.direct_source_duplicated_by_rag` | `prompt.direct_sources_excluded_from_rag` | `prompt_context` | `real_coordinator` | passed | declared=yes, observed=yes | real_direct_source_rag_exclusion_matrix |
| `risk.direct_source_duplicated_by_rag` | `prompt.direct_sources_excluded_from_rag` | `prompt_context` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.direct_source_duplicated_by_rag` | `prompt.mandatory_source_overflow_fails_visible` | `prompt_context` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.direct_source_duplicated_by_rag` | `prompt.mandatory_source_overflow_fails_visible` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_29 |
| `risk.direct_source_duplicated_by_rag` | `prompt.mandatory_source_overflow_fails_visible` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_7 |
| `risk.direct_source_duplicated_by_rag` | `prompt.mandatory_source_overflow_fails_visible` | `prompt_context` | `real_coordinator` | passed | declared=yes, observed=yes | real_rigor_mandatory_source_overflow_fails_before_model_or_rag |
| `risk.direct_source_duplicated_by_rag` | `prompt.mandatory_source_overflow_fails_visible` | `prompt_context` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_29 |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_7 |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | model_proof_runtime_gating_allowed_outputs |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `real_coordinator` | passed | declared=yes, observed=yes | real_disabled_proof_runtime_skips_autonomous_checkpoint |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_lean_when_disabled` | `proof_runtime_gating` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_29 |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_7 |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `model` | passed | declared=yes, observed=yes | model_proof_runtime_gating_allowed_outputs |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.disabled_proof_runtime_executes` | `proof_runtime.no_smt_when_disabled` | `proof_runtime_gating` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `prompt.generated_appendices_stripped` | `prompt_context` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `prompt.generated_appendices_stripped` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_29 |
| `risk.generated_proofs_reenter_prompts` | `prompt.generated_appendices_stripped` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_7 |
| `risk.generated_proofs_reenter_prompts` | `prompt.generated_appendices_stripped` | `prompt_context` | `real_coordinator` | passed | declared=yes, observed=yes | real_model_visible_prompts_strip_generated_proof_appendices |
| `risk.generated_proofs_reenter_prompts` | `prompt.generated_appendices_stripped` | `prompt_context` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `prompt.proof_source_context_required` | `prompt_context` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `prompt.proof_source_context_required` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_29 |
| `risk.generated_proofs_reenter_prompts` | `prompt.proof_source_context_required` | `prompt_context` | `model` | passed | declared=yes, observed=yes | generated_rag_prompt_boundaries_seed_7 |
| `risk.generated_proofs_reenter_prompts` | `prompt.proof_source_context_required` | `prompt_context` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `prompt.proof_source_context_required` | `prompt_context` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `proof_scope.generated_appendices_stripped_from_prompts` | `proof_scope_isolation` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `proof_scope.generated_appendices_stripped_from_prompts` | `proof_scope_isolation` | `model` | uncovered | declared=yes, observed=no | no observed scenario |
| `risk.generated_proofs_reenter_prompts` | `proof_scope.generated_appendices_stripped_from_prompts` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_generated_proof_appendix_stripping |
| `risk.generated_proofs_reenter_prompts` | `proof_scope.generated_appendices_stripped_from_prompts` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_model_visible_prompts_strip_generated_proof_appendices |
| `risk.generated_proofs_reenter_prompts` | `proof_scope.generated_appendices_stripped_from_prompts` | `proof_scope_isolation` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `api.hosted_desktop_only_routes_unavailable` | `websocket_api_contracts` | `browser_smoke` | passed | declared=yes, observed=yes | browser_hosted_startup_respects_desktop_capability_boundaries |
| `risk.hosted_desktop_boundary` | `api.hosted_desktop_only_routes_unavailable` | `websocket_api_contracts` | `model` | uncovered | declared=yes, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `api.hosted_desktop_only_routes_unavailable` | `websocket_api_contracts` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `api.hosted_desktop_only_routes_unavailable` | `websocket_api_contracts` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `proof_runtime.hosted_proof_settings_unavailable` | `proof_runtime_gating` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `proof_runtime.hosted_proof_settings_unavailable` | `proof_runtime_gating` | `model` | uncovered | declared=yes, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `proof_runtime.hosted_proof_settings_unavailable` | `proof_runtime_gating` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.hosted_desktop_boundary` | `proof_runtime.hosted_proof_settings_unavailable` | `proof_runtime_gating` | `real_route` | passed | declared=yes, observed=yes | real_hosted_proof_settings_returns_desktop_unavailable |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_scoped_registered_proof_events_seed_29 |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_scoped_registered_proof_events_seed_7 |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | model_websocket_proof_scope_events |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_leanoj_master_proof_stop_resume_preserves_isolated_state |
| `risk.proof_scope_leak` | `proof_scope.autonomous_events_not_manual` | `proof_scope_isolation` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_29 |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_7 |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | model_manual_proof_clear_scope_assistant |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_archive_and_leanoj_clear_preserve_scope_contract |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_route` | passed | declared=yes, observed=yes | real_manual_aggregator_clear_archives_before_clear |
| `risk.proof_scope_leak` | `proof_scope.manual_clear_archives_active` | `proof_scope_isolation` | `real_route` | passed | declared=yes, observed=yes | real_manual_compiler_papers_lifecycle_and_clear_archive |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_29 |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | generated_manual_scope_clear_archive_seed_7 |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `model` | passed | declared=yes, observed=yes | model_manual_proof_clear_scope_assistant |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `real_coordinator` | passed | declared=yes, observed=yes | real_manual_compiler_proof_only_uses_rigor_settings_and_manual_scope |
| `risk.proof_scope_leak` | `proof_scope.manual_not_in_autonomous_current` | `proof_scope_isolation` | `real_route` | passed | declared=yes, observed=yes | real_proof_scope_matrix_isolated_under_temp_root |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `browser_smoke` | passed | declared=yes, observed=yes | browser_autonomous_start_stop_preserves_single_ui_owner |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_29 |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | generated_output_and_proof_runtime_gating_seed_7 |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | model_proof_runtime_gating_allowed_outputs |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proofs_only_enters_paper` | `outputs.at_least_one_output_enabled` | `allowed_outputs` | `real_route` | passed | declared=yes, observed=yes | real_manual_compiler_rejects_no_allowed_outputs |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_29 |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_7 |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `model` | passed | declared=yes, observed=yes | model_allowed_outputs_provider_pause_stop_resume |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `real_coordinator` | passed | declared=yes, observed=yes | real_autonomous_proofs_only_handoff_returns_to_topic_exploration |
| `risk.proofs_only_enters_paper` | `outputs.proofs_only_no_paper_phase` | `allowed_outputs` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_29 |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_7 |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | model_allowed_outputs_provider_pause_stop_resume |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `real_coordinator` | passed | declared=yes, observed=yes | real_autonomous_provider_credit_pause_preserves_checkpoint_and_resumes |
| `risk.provider_pause_loses_checkpoint` | `provider.pause_preserves_checkpoint` | `provider_pause_resume` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_29 |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_7 |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | model_allowed_outputs_provider_pause_stop_resume |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.reset_wakes_without_corrupting_checkpoint` | `provider_pause_resume` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_29 |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_allowed_outputs_provider_pause_resume_seed_7 |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_leanoj_durable_lifecycle_seed_29 |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | generated_leanoj_durable_lifecycle_seed_7 |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `model` | passed | declared=yes, observed=yes | model_allowed_outputs_provider_pause_stop_resume |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `real_coordinator` | blocked | declared=no, observed=yes | real_provider_stop_reset_checkpoint_unavailable_without_wait_seam |
| `risk.provider_pause_loses_checkpoint` | `provider.stop_resume_preserves_pause` | `provider_pause_resume` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.pruned_paper_context_leak` | `state.pruned_papers_excluded_from_context` | `workflow_filesystem_state` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.pruned_paper_context_leak` | `state.pruned_papers_excluded_from_context` | `workflow_filesystem_state` | `model` | uncovered | declared=yes, observed=no | no observed scenario |
| `risk.pruned_paper_context_leak` | `state.pruned_papers_excluded_from_context` | `workflow_filesystem_state` | `real_coordinator` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.pruned_paper_context_leak` | `state.pruned_papers_excluded_from_context` | `workflow_filesystem_state` | `real_route` | passed | declared=yes, observed=yes | real_pruned_paper_preserved_but_removed_from_active_context |
| `risk.stale_child_phase_takeover` | `runtime.parent_action_fences_child_outputs` | `runtime_exclusivity` | `browser_smoke` | uncovered | declared=no, observed=no | no observed scenario |
| `risk.stale_child_phase_takeover` | `runtime.parent_action_fences_child_outputs` | `runtime_exclusivity` | `model` | uncovered | declared=yes, observed=no | no observed scenario |
| `risk.stale_child_phase_takeover` | `runtime.parent_action_fences_child_outputs` | `runtime_exclusivity` | `real_coordinator` | blocked | declared=no, observed=yes | real_parent_action_fencing_unavailable_without_production_seam |
| `risk.stale_child_phase_takeover` | `runtime.parent_action_fences_child_outputs` | `runtime_exclusivity` | `real_route` | uncovered | declared=no, observed=no | no observed scenario |

## Structured Gaps

| Gap | Status | Declared | Reason |
|---|---|---|---|
| `risk.assistant_memory_crosses_boundary:assistant.disable_clears_live_pack:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.assistant_memory_crosses_boundary:assistant.no_validator_injection:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.assistant_memory_crosses_boundary:prompt.validator_excludes_assistant_memory:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.disabled_proof_runtime_executes:proof_runtime.no_smt_when_disabled:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.generated_proofs_reenter_prompts:prompt.proof_source_context_required:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.generated_proofs_reenter_prompts:proof_scope.generated_appendices_stripped_from_prompts:model` | uncovered | yes | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.hosted_desktop_boundary:api.hosted_desktop_only_routes_unavailable:model` | uncovered | yes | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.hosted_desktop_boundary:api.hosted_desktop_only_routes_unavailable:real_route` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.hosted_desktop_boundary:proof_runtime.hosted_proof_settings_unavailable:model` | uncovered | yes | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.proofs_only_enters_paper:outputs.at_least_one_output_enabled:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.provider_pause_loses_checkpoint:provider.reset_wakes_without_corrupting_checkpoint:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.pruned_paper_context_leak:state.pruned_papers_excluded_from_context:model` | uncovered | yes | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.pruned_paper_context_leak:state.pruned_papers_excluded_from_context:real_coordinator` | uncovered | no | No observed coverage record exists for this required risk/invariant/adapter basis. |
| `risk.stale_child_phase_takeover:runtime.parent_action_fences_child_outputs:model` | uncovered | yes | No observed coverage record exists for this required risk/invariant/adapter basis. |
