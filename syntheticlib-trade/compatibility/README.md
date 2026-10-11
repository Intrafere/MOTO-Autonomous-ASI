# MOTO Compatibility and Release Notes

This directory is reserved for a versioned compatibility matrix and release
notes shared by the MOTO and SyntheticLib teams.

## Current MOTO-owned preflight

MOTO publishes a deterministic, credential-free compatibility preflight here:

- `compatibility-matrix.json` binds the advertised MOTO package, build, and API
  contract to the current fail-closed SyntheticLib state.
- `moto-contract-test-result.json` records the passing
  `moto_owned_compatibility_preflight` result.
- `moto-compatibility-receipt.json` is the normalized receipt.
- `moto-compatibility-receipt.sha256` is its detached SHA-256 digest.
- `frozen-asset-manifest.json` records the fail-closed unpublished-asset state
  and the strict validation policy for future frozen assets.
- `moto-preflight-attestation.json` separates the package subject from
  execution provenance; the checked-in deterministic form is local-diagnostic
  evidence, while clean release attestations require an independently
  controlled CI/release identity.
- `compatibility-bundle-manifest.json` binds every machine-readable evidence
  member by byte length and SHA-256 and carries a digest over that member map.
- `MOTO_COMPATIBILITY_RECEIPT.md` is the human-readable projection.

Run `python -m scripts.check_syntheticlib_trade_contract` to verify the closed
eight-member checked-in evidence set without modifying it. Regeneration is
explicit through `--write`.
The passing result covers only MOTO's identity, matrix, normalization, digest,
and fail-closed behavior. It does not claim that an unpublished SyntheticLib
wire contract passed or that live proof trade is supported.

Canonical JSON v1 uses UTF-8, recursively sorted object keys, preserved array
order, compact separators, and finite JSON values. Verification treats this
directory as a closed evidence set, rejects unexpected files and symlinks, and
recomputes the result rather than trusting its `passed` field.
The verifier also requires the exact unpublished-state matrix and asset
manifest, so a coherently regenerated but policy-altered evidence set fails.
Publication restores the prior generation before propagating write failures or
interruptions, rejects unexpected directory entries, and the local release
simulator transactionally retains complete output without replacing nonempty
or symlinked destinations.

The three pass scopes are deliberately separate: the checked-in MOTO base
preflight, the independently executed clean-source release attestation, and the
future SyntheticLib frozen-wire contract suite. A pass in one scope does not
establish either of the others.

The separately triggered `SyntheticLib Compatibility Release Evidence`
workflow verifies the tracked diagnostic evidence from an immutable clean
checkout, stages a release manifest whose advertised `build_commit` is that
exact checkout `HEAD`, and generates release-specific base evidence. It then
publishes that base evidence plus a separate clean-source attestation and
closed release bundle as workflow artifacts. These release-stamped and
nondeterministic execution-provenance files are not written into this
checked-in deterministic directory. A workflow that fails the
identity/cleanliness gate produces no acceptable release attestation.

Maintainers can run the same release-generation sequence locally with
`python -m scripts.run_local_syntheticlib_compatibility_ci`. The command copies
only allowlisted contract implementation and release-orchestration inputs into a temporary Git
repository, creates a deterministic clean snapshot, regenerates release-stamped
base evidence, and verifies the clean-source bundle. It does not stage, commit,
or modify the real repository. Its output is diagnostic workflow-readiness
evidence only and cannot satisfy the independently controlled release gate.

Each published entry should bind:

- SyntheticLib contract and server release;
- supported MOTO application, package, and build identities;
- request/response schema and fixture versions;
- native-client profile and token-lifecycle version;
- proof-selection and novelty-mapping policy versions;
- supported Lean, Mathlib, and relevant proof provenance requirements;
- changed routes, fields, enums, limits, or errors;
- migration behavior for queued or pending requests; and
- deprecation and update-required dates.

Unknown or unsupported versions and enum values fail closed. Pending requests
retain their original contract identity and must not be silently reinterpreted
under a newer release. Compatibility entries are descriptive support
contracts; they do not authorize staging, production, legal activation, or
surface enablement.
