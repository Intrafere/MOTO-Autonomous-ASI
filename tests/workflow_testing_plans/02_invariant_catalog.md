# Workflow Product-Law Invariant Catalog

This document mirrors the stable executable IDs in
`tests/workflow_harness/invariant_catalog.py`. The testing system is a removable
overlay: production code, launch paths, persistence, and runtime behavior must
not import or depend on this catalog, its scenarios, or generated artifacts.

Existing invariant IDs are append-only. Add IDs for new product laws; do not
rename or reuse an existing ID.

## Build 07 proof-pruning and continuous-loop laws

| Stable ID | Product law | Model scenario | Real-source observation |
|---|---|---|---|
| `proof_runtime.candidate_list_validator_gate` | Every proof round validates the complete candidate list before proof cost, accepts exactly 75%, forwards only approved novel entries, and fences the latest five semantic rejections to the owning round. | Direct invariant-catalog scenario | Candidate-list stage and contract coverage. |
| `proof_pruning.artifacts_and_future_memory_preserved` | Pruning preserves canonical records, visibility, export/certificate and graph access, future-run memory, and SyntheticLib-facing eligibility. | `model_proof_pruning_continuous_contracts` | Proof database and proof-search regression coverage. |
| `proof_pruning.owning_run_context_excludes_pruned` | Owning-run direct/search/Assistant context excludes a pruned occurrence, including reused packs; future runs may retrieve it. | `model_proof_pruning_continuous_contracts` | Proof database and Assistant cache coverage. |
| `proof_pruning.validator_gates_automatic_mutation` | Automatic mutation requires a deterministically valid proposal and independent Validator acceptance. | `model_proof_pruning_continuous_contracts` | Pruning contract/agent coverage. |
| `proof_pruning.semantic_distinct_review_preserves_unique_routes` | Canonically distinct proofs may be suppressed only through bounded semantic evidence citing retained coverage; related but unique routes remain active. | `model_proof_pruning_continuous_contracts` | Semantic snapshot, agent-guard, and commit-fence coverage. |
| `proof_pruning.no_prune_is_valid` | A valid no-prune response is non-mutating and non-error. | `model_proof_pruning_continuous_contracts` | Held-review coordinator coverage. |
| `proof_pruning.review_non_blocking` | Proof solving and serialized registration continue while proposer/Validator review is pending. | `model_proof_pruning_continuous_contracts` | Async pruning coordinator coverage. |
| `proof_pruning.commit_lifecycle_fenced` | Stop, Clear, ownership, generation, revision, hash, dependency, and snapshot mismatches prevent mutation. | `model_proof_pruning_continuous_contracts` | Snapshot commit coverage. |
| `proof_pruning.context_overflow_truthful` | Candidate-local overflow stays nonfatal/deferred and cannot be relabeled as pruning success. | `model_proof_pruning_continuous_contracts` | The exact overflow-during-commit interleaving is `blocked`: no bounded production seam observes it without synthesizing the transition owner. |
| `proof_loop.continuous_explicit_ownership` | Continuous manual mode owns one reservation, runs without a no-candidate limit, exposes Stop and detailed per-round activity, has no restart-resume state, and cleans up ownership. | `model_proof_pruning_continuous_contracts` | Proof-run manager terminal-policy coverage. |
| `proof_loop.automatic_round_policy_preserved` | Pruning preserves the three Autonomous automatic callers' current up-to-four-round policy and first valid zero-candidate exit; manual continuous mode remains separate and has no Next Round control. | `model_proof_pruning_continuous_contracts` | Autonomous proof-round and manual terminal-policy coverage. |
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
