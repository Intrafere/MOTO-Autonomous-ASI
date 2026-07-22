from __future__ import annotations

from tests.workflow_harness.coverage_metadata import InteractionCoverage


BROWSER_SMOKE_COVERAGE = (
    InteractionCoverage(
        scenario_id="browser_hosted_startup_respects_desktop_capability_boundaries",
        fields=("websocket_api_contracts", "proof_runtime_gating", "runtime_exclusivity"),
        invariants=("api.hosted_desktop_only_routes_unavailable",),
        adapter="browser_smoke",
        result="passed",
        test_file="tests/workflow_browser_smoke/hosted-capabilities.spec.js",
        evidence=(
            "hosted_openrouter_startup",
            "desktop_provider_paths_not_requested",
        ),
        runner="playwright",
        asserted_invariants=("api.hosted_desktop_only_routes_unavailable",),
        test_selectors=(
            "tests/workflow_browser_smoke/hosted-capabilities.spec.js::hosted capabilities drive startup copy and hide desktop-only paths",
        ),
    ),
    InteractionCoverage(
        scenario_id="browser_autonomous_start_stop_preserves_single_ui_owner",
        fields=("runtime_exclusivity", "allowed_outputs", "websocket_api_contracts"),
        invariants=(
            "runtime.single_active_workflow",
            "outputs.at_least_one_output_enabled",
        ),
        adapter="browser_smoke",
        result="passed",
        test_file="tests/workflow_browser_smoke/autonomous-controls.spec.js",
        evidence=(
            "papers_only_start_payload",
            "running_owner_controls",
            "stop_restores_controls",
        ),
        runner="playwright",
        asserted_invariants=(
            "runtime.single_active_workflow",
            "outputs.at_least_one_output_enabled",
        ),
        test_selectors=(
            "tests/workflow_browser_smoke/autonomous-controls.spec.js::autonomous start sends hosted payload, locks controls, and stops cleanly",
        ),
    ),
    InteractionCoverage(
        scenario_id="browser_mode_switching_preserves_in_memory_default_and_developer_gate",
        fields=("runtime_exclusivity", "workflow_filesystem_state", "websocket_api_contracts"),
        invariants=("runtime.single_active_workflow",),
        adapter="browser_smoke",
        result="passed",
        test_file="tests/workflow_browser_smoke/mode-switching.spec.js",
        evidence=(
            "autonomous_default_mode",
            "manual_mode_switch",
            "leanoj_hidden_before_developer_gate",
            "leanoj_visible_after_developer_gate",
            "reload_restores_autonomous_mode",
        ),
        runner="playwright",
        asserted_invariants=("runtime.single_active_workflow",),
        test_selectors=(
            "tests/workflow_browser_smoke/mode-switching.spec.js::mode switching is in-memory and LeanOJ remains developer gated",
        ),
    ),
    InteractionCoverage(
        scenario_id="browser_overflow_activity_persists_with_attribution_and_terminal_deduplication",
        fields=("websocket_api_contracts", "workflow_filesystem_state", "provider_pause_resume"),
        invariants=(
            "events.context_overflow_route_identity",
            "events.context_overflow_persists_across_reload",
            "events.context_overflow_terminal_stop_once",
            "events.proof_context_overflow_nonfatal",
        ),
        adapter="browser_smoke",
        result="passed",
        test_file="tests/workflow_browser_smoke/overflow-activity.spec.js",
        evidence=(
            "configured_and_effective_route_visible",
            "proof_overflow_keeps_parent_active",
            "fatal_stop_not_duplicated",
            "activity_restored_after_reload",
        ),
        runner="playwright",
        asserted_invariants=(
            "events.context_overflow_route_identity",
            "events.context_overflow_persists_across_reload",
            "events.context_overflow_terminal_stop_once",
            "events.proof_context_overflow_nonfatal",
        ),
        test_selectors=(
            "tests/workflow_browser_smoke/overflow-activity.spec.js::overflow activity attributes routes, keeps proof overflow nonfatal, dedupes stop, and persists",
        ),
    ),
)
