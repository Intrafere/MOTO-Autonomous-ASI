# Remaining Cross-Field Work

The modernized overlay deliberately leaves the following interactions as `blocked`
instead of reporting synthetic or partial coverage as stronger evidence:

- Live external SyntheticLib proof contribution and acceptance. Local canonical
  preservation, search eligibility, and unchanged `syntheticlib_novel` accounting are
  covered, but the external service is intentionally unavailable to isolated tests.
- Real-provider semantic correctness for pruning proposer and Validator decisions.
  Deterministic contract, gating, fencing, and commit behavior are covered with model
  responses faked; provider/model quality is not a deterministic production seam.
- Overflow arriving inside the proof-prune commit critical section. Overflow-to-pressure
  and no-false-success behavior are covered outside that exact race, but production
  exposes no bounded hook inside the atomic commit section.
- Full owning-run exclusion across every Assistant cache/latest/stale/provider-cooldown
  reuse path. Focused cache-path tests exist, but no single bounded production-owned
  lifecycle currently observes every reuse path without replacing the transition owner.
- End-to-end semantic distinct-route pruning through proposer, Validator, commit, then a
  later attempt to prune a retained support. Snapshot/commit support protection and
  semantic contract tests exist separately; their combined model-driven lifecycle is not
  deterministically observable without faking semantic authority.

Acceptance criteria for removing a block:

1. A bounded production seam owns the real transition without adding production test hooks.
2. The test records explicit selectors, asserted invariant IDs, observed transitions, and
   all real, faked, and blocked dependencies.
3. The interaction runs under isolated runtime roots and cannot contact external providers,
   Lean/SMT, browsers, credentials, or user data unless the test explicitly owns them.
4. For external-provider blocks, a hermetic contract service or explicitly owned
   integration environment is available; no public network call is added to normal tests.
5. The deterministic result and support graph are regenerated and all workflow/deep suites pass.
