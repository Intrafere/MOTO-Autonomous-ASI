# MOTO SyntheticLib Proof-Trade Compatibility Receipt

## Scope

This is the MOTO-side compatibility receipt required by the MOTO and
SyntheticLib proof-trade integration contract. It records the supported
MOTO package/build identity, MOTO API contract version, compatibility decision,
scoped contract-preflight result, and normalized receipt digest.

This receipt does not activate proof trade, authorize a live endpoint, accept
legal terms, represent SyntheticLib server acceptance, or claim that an
unpublished SyntheticLib wire contract passed.
It is deterministic advertised-package evidence, not an independently executed
clean-source release attestation.

## Supported MOTO package/build identity

- Product: `MOTO Autonomous ASI`
- Package version: `1.1.08`
- Build commit: `40be14c63aee7027f02d36479416e1341385b6e3`
- Update channel: `main`
- Identity source: repository-root `moto-update-manifest.json`

The identity above is the advertised package identity. Runtime Git/archive
identity remains a separate diagnostic and is never substituted into this
release receipt.

## Contract versions

- MOTO API contract version: `build6-v120`
- Compatibility matrix: `moto-syntheticlib-compatibility-matrix.v1`
- Contract-test result: `moto-syntheticlib-contract-test-result.v1`
- Compatibility receipt: `moto-syntheticlib-compatibility-receipt.v1`
- Canonicalization: `moto-canonical-json-v1`
- MOTO-owned proof-selection and provenance policies:
- canonicalProofIdentity: `moto-proof-identity-v1`
- leanCompileReceipt: `moto-lean-compile-receipt-v1`
- legacyPortReceipt: `moto-legacy-port-receipt-v1`
- localNoveltyReceipt: `moto-local-novelty-receipt-v1`
- proofLocator: `moto-proof-locator-v1`
- sourceSelectionPolicy: `moto-trade-source-selection-v2`
- tradeDataBoundary: `moto-trade-data-boundary-v1`
- SyntheticLib proof-trade contract, native-client profile, canonical fixtures,
  and server release: **unpublished**

## Compatibility matrix

- Exact MOTO package/build/API-contract tuple: **supported by MOTO preflight**
- Frozen SyntheticLib proof-trade wire contract: **not available**
- Live proof-trade compatibility: **blocked**
- `supportedForLiveProofTrade`: `false`
- Reason code: `frozen_syntheticlib_contract_unpublished`
- Unknown versions and enum values: **update required; fail closed**

Unavailable frozen inputs:

- `client-profile/native-client-profile.json`
- `contracts/proof-trade.openapi.json`
- `contracts/schemas`
- `fixtures/v1`

The authoritative machine-readable matrix is `compatibility-matrix.json`.

## Contract-test result

- Scope: `moto_owned_compatibility_preflight`
- Status: `passed`
- Passed: `true`
- SyntheticLib wire contract tested: `false`
- Live compatibility supported: `false`

This passing result certifies only MOTO's deterministic, credential-free
compatibility preflight: package identity consistency, fail-closed matrix
behavior, canonical receipt normalization, and digest verification. It is not
a passing live or frozen-wire SyntheticLib contract result.

The authoritative result is `moto-contract-test-result.json`.

## Normalized compatibility receipt

Normalization rules:

1. UTF-8 encoding without a byte-order mark.
2. One compact JSON object with no insignificant whitespace.
3. Object keys are recursively sorted; array order is preserved.
4. JSON `null` represents an unpublished SyntheticLib identity.
5. SHA-256 covers the exact bytes of `moto-compatibility-receipt.json`, which
   has no trailing newline.

```json
{"canonicalization":"moto-canonical-json-v1","compatibility":{"reasonCode":"frozen_syntheticlib_contract_unpublished","status":"blocked","supportedForLiveProofTrade":false},"moto":{"apiContractVersion":"build6-v120","buildCommit":"40be14c63aee7027f02d36479416e1341385b6e3","manifestVersion":1,"updateChannel":"main","version":"1.1.08"},"receiptId":"moto-1.1.08-build6-v120-40be14c63aee7027f02d36479416e1341385b6e3","schemaVersion":"moto-syntheticlib-compatibility-receipt.v1","syntheticLib":{"fixtureVersion":null,"nativeClientProfileVersion":null,"proofTradeContractVersion":null,"serverRelease":null}}
```

Normalized receipt SHA-256:

`6de913d8c919d2e8cab4ce047063b306b7d7635b9829500fa2deea0e31853646`

The detached digest is `moto-compatibility-receipt.sha256`.

## Release acceptance gate

Live compatibility remains blocked until SyntheticLib publishes frozen,
versioned wire contracts, fixtures, and a registered native-client profile and
MOTO passes a separate credential-free wire-contract suite against those exact
artifacts. Staging and production remain separately authorized.
