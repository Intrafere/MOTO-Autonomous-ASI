# Build 07 — Current Product Laws

This test-only build repairs overflow semantics and appends representative laws for
solution paths, proof competition, exact provider roles/cooldowns, whole-entry feedback
projection, atomic bulk mutation, runtime-root ownership, proof-run identity, proof
framing, sleep ownership, Supercharge/creativity separation, group novelty, and LeanOJ
parent control.

- `events.proof_context_overflow_nonfatal` now means candidate-local overflow only.
- Mandatory Autonomous overflow has a separate checkpoint-preserving terminal law.
- Continuous-manual mandatory overflow has a separate terminal-after-drain law.
- Every appended family has a stable catalog ID, positive model observation, adverse
  mutation fixture, and target risk.
- Unsafe production lifecycle interleavings remain intentionally model-only. Real
  adapter coverage must stay `blocked` until it can observe bounded production seams
  without synthesizing lifecycle ownership.

Focused verification:

`python -m pytest tests/workflow_scenarios/test_invariant_catalog.py tests/workflow_scenarios/test_cross_field_analysis.py tests/workflow_cross_field/test_target_risks.py -q`
