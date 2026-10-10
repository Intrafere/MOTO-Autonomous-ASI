---
description: Developer mode gating for hidden frontend controls
paths:
  - "frontend/src/**/*.js"
  - "frontend/src/**/*.jsx"
---

# Developer Mode Gates

Developer mode is a hidden frontend switch toggled by pressing `Shift + Z + X` (`useDeveloperModeShortcut`); the chord is ignored, and the keystroke kept, while focus is in an editable text field. The state is stored in localStorage as `developerModeSettingsEnabled` and should stay centralized through `developerModeEnabled` props or the same storage key.

Known gated surfaces:
- `LeanOJ Proof Solver` appears in the mode selector only while developer mode is enabled. If developer mode is turned off while `appMode === 'leanoj'`, the app should return to autonomous mode.
- Supercharge checkboxes in Aggregator, Compiler, Autonomous, LeanOJ, and related role settings are developer-only. Start/request payloads must force `supercharge_enabled` false unless developer mode is currently enabled.
- Creativity Emphasis Boost checkboxes beside Aggregator, Autonomous, and LeanOJ start controls are developer-only for now. Start/request payloads must force `creativity_emphasis_boost_enabled=false` unless developer mode is currently enabled; keep this gate isolated so the feature can later move to standard mode cleanly.
- Raw JSON settings editors and developer-only model/settings toggles are developer-only and should close or hide when developer mode is disabled.
- Boost copy may mention Supercharge only when developer mode is enabled.
- Before launch, SyntheticLib uses the backend-owned lifecycle state
  (`coming_soon | developer_preview | launched`). Known prelaunch states show a
  private workspace only in developer mode; its local evidence audit, trade-ledger
  check, and local ledger controls (snapshot selection, archive/reset, and archive
  deletion require an inline confirmation) each run only on an explicit click, and closing clears selection
  and ignores late responses. Its authorization-readiness check is likewise
  explicit-click and read-only. Proof history's **Offer proofs for SyntheticLib access**
  panel follows the same presentation helper (`showTrade`), so it is
  developer-only before launch, never requests on mount, and offers checkboxes only for
  canonically eligible proofs behind an inline confirmation. Its **Prepare existing
  proofs** action runs a read-only dry run first and migrates released-build proofs
  only after an inline confirmation. Standard users retain
  the Coming soon surface and never see or request the SyntheticLib4 proof-search
  corpus (the backend corpus filter enforces this too); unknown values fail closed,
  and disabling developer mode closes the private workspace and stops any multi-step
  request before its follow-up call. The presentation helper's `showDeveloperWorkspace`
  keeps the workspace behind developer mode even after launch, `showTrade` keeps
  proof-trade desktop-only, and generic mode never shows the prelaunch integration.
  Browser developer mode never authorizes backend SyntheticLib execution, which
  requires separate explicit local preview authority.

When adding a new hidden or experimental frontend control, gate both the visible UI and any request/runtime payload field behind developer mode. Do not rely on hiding the checkbox alone.
