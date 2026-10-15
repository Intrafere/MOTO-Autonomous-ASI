---
description: MOTO cross-field workflow testing and deep verification requirements
---

# Cross-Field Workflow Testing

## #1 Rule: The Testing System Must Remain a Removable Overlay

The testing system is its own system that sits on top of the current production
program. Production code, launch paths, builds, and runtime behavior must never depend
on the testing folders, test artifacts, test-only imports, or test-only state. A user
must be able to delete the entire testing system (including `tests/` and generated test
artifacts) and still have a fully functioning MOTO program with unchanged production
behavior. This separability requirement takes priority over every other testing rule.

MOTO uses a project-specific testing overlay that combines executable product-law
invariants, deterministic state recombination, bounded real-code adapters with faked
dependencies, isolated scenario/result artifacts, inverse target-risk analysis, and a
support graph.

- Treat `tests/workflow_harness/invariant_catalog.py` and
  `tests/workflow_testing_plans/02_invariant_catalog.md` as the stable product-law catalog.
  Add invariant IDs instead of renaming existing IDs.
- Keep the overlay removable and behavior-neutral. Never add production dependencies on
  the testing system, runtime shortcuts, hidden stops/caps, impossible state transitions,
  or production-only test behavior.
- Keep external providers, Lean/SMT, browser services, credentials, and mutable runtime
  roots faked, blocked, or isolated unless a test explicitly owns that integration.
- `tests/conftest.py` binds every pytest session not already isolated by a harness to a
  temporary runtime root, fails the session on any write into the real `backend/data` or
  `backend/logs`, and ends the pytest process with every thread's stack when a test exceeds
  `MOTO_TEST_TIMEOUT_SECONDS` (default 600; `0` disables); tests must not depend
  on a developer's live data, runtime settings, or proof-search index.
- Record safely unobservable interactions as `blocked`; never report synthetic coverage
  as a real-code pass.
- Passed production coverage must name explicit executable selectors, asserted invariant
  IDs, observed transitions, and real/faked dependency boundaries. Use focused-production,
  route, and coordinator maturity labels truthfully; never infer evidence from test names.
- Frontend unit evidence is first-class only through exact executable Vitest selectors;
  keep it distinct from Playwright rendered/reload evidence.
- Every stable invariant must belong to an inverse target risk or have an explicit
  exemption, and generated artifacts must preserve executable provenance and report
  orphan or required-but-undeclared coverage gaps.
- Run focused tests while developing and `npm run test:workflows` after workflow-test or
  deterministic-artifact changes.
- Deep workflow suites are local/manual-only; never enable them in automatic GitHub
 Actions runs (push, pull request, or schedule). Normal CI must keep deep-test opt-ins disabled.
- Run `npm run test:deep` locally after every major build and after substantial orchestration,
  persistence, provider, proof, prompt/RAG, event-contract, runtime-root, or cross-mode
  change. It is additive: run `npm test` separately for normal backend/frontend coverage.
- Before any push, run `npm run test:github`: every `ci.yml` check, the release-evidence
  workflow (its focused tests in a pytest-only environment, as on its runner), and
  `npm run test:codeql` (GitHub's CodeQL actions/python/javascript analyses,
  security-extended suite), failing on any alert `origin/main` lacks. CodeQL default setup only
  scans pull requests into `main`; CI's `codeql-new-alerts` job repeats this on `dev/*` pushes.
  Never call a branch ready while a check failed or was skipped.
- No GitHub check may first run after a merge to `main`: every workflow also triggers on
  pull requests and `dev/**` pushes (enforced by a test). `tests/conftest.py` guards that need
  MOTO runtime modules stand down when those modules' dependencies are not installed.
- Regenerate deterministic cross-field artifacts with
  `python -m tests.workflow_cross_field.artifacts` when their inputs change, then verify
  that the checked-in projection is clean.
- Cross-workflow Live Activity laws must cover owning-run scope, distinct repeated-event
  identity, bounded legacy and normalized secret-redacted diagnostics, complete rendered
  validator accept/reject feedback, infrastructure-failure separation from semantic
  rejection, degraded persistence visibility, live/status exact-once terminal
  lifecycle presentation, and provider/API-error route identity (workflow mode, role,
  model, provider, OpenRouter configured/upstream host) in fields and the rendered line.
  Browser evidence must verify rendered and reload behavior;
  model-only evidence cannot stand in for production event emission or persistence.
- A production-wide AST retry-wait inventory must classify actual retry waits, explicit
 non-retry polling/synchronization waits, and blocked external/tool waits; only assured sources
  routed through the production emit-and-wait seam may count as passed coverage.
- A companion AST provider-error activity ledger classifies every provider/API-error
  activity source as syntactically route-attributed or explicitly exempt with a
  recorded reason (for example non-model loop/storage retries or pass-through forwarders);
  unclassified sources, stale policies, or mislabelled sites fail. Payload contents are
  proven by focused production tests, not by the ledger.
