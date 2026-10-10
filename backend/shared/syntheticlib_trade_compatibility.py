"""Deterministic, credential-free MOTO proof-trade compatibility preflight."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from backend.shared import build_info


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON_PATH = REPO_ROOT / "package.json"
BUILD_MANIFEST_PATH = REPO_ROOT / "moto-update-manifest.json"
PLAN_ROOT = REPO_ROOT / "PLANS" / "MOTO_SYNTHETICLIB_TRADE_INTEGRATION_PLAN"
COMPATIBILITY_DIR = PLAN_ROOT / "compatibility"

MATRIX_SCHEMA_VERSION = "moto-syntheticlib-compatibility-matrix.v1"
TEST_RESULT_SCHEMA_VERSION = "moto-syntheticlib-contract-test-result.v1"
RECEIPT_SCHEMA_VERSION = "moto-syntheticlib-compatibility-receipt.v1"
ASSET_MANIFEST_SCHEMA_VERSION = "moto-syntheticlib-frozen-assets.v1"
BUNDLE_SCHEMA_VERSION = "moto-syntheticlib-evidence-bundle.v1"
ATTESTATION_SCHEMA_VERSION = "moto-syntheticlib-execution-attestation.v1"
RELEASE_BUNDLE_SCHEMA_VERSION = "moto-syntheticlib-release-evidence.v1"
CANONICALIZATION_VERSION = "moto-canonical-json-v1"
VERIFIER_VERSION = "moto-syntheticlib-verifier-v3"
PREFLIGHT_SCOPE = "moto_owned_compatibility_preflight"
# MOTO-owned proof-selection and provenance policy versions bound into every
# matrix. Literals keep this module dependency-free for the isolated release
# runner; tests bind each value to its owning module constant.
MOTO_POLICY_VERSIONS = {
    "canonicalProofIdentity": "moto-proof-identity-v1",
    "proofLocator": "moto-proof-locator-v1",
    "leanCompileReceipt": "moto-lean-compile-receipt-v1",
    "localNoveltyReceipt": "moto-local-novelty-receipt-v1",
    "legacyPortReceipt": "moto-legacy-port-receipt-v1",
    "sourceSelectionPolicy": "moto-trade-source-selection-v2",
    "tradeDataBoundary": "moto-trade-data-boundary-v1",
}
BLOCKED_REASON = "frozen_syntheticlib_contract_unpublished"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BUILD_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

ARTIFACT_FILENAMES = {
    "matrix": "compatibility-matrix.json",
    "test_result": "moto-contract-test-result.json",
    "receipt": "moto-compatibility-receipt.json",
    "digest": "moto-compatibility-receipt.sha256",
    "assets": "frozen-asset-manifest.json",
    "bundle": "compatibility-bundle-manifest.json",
    "attestation": "moto-preflight-attestation.json",
    "markdown": "MOTO_COMPATIBILITY_RECEIPT.md",
}

RELEASE_ARTIFACT_FILENAMES = {
    "attestation": "moto-clean-source-attestation.json",
    "bundle": "moto-clean-source-attestation-bundle.json",
}

FROZEN_ASSET_REQUIREMENTS = (
    ("proofTradeOpenApi", "contracts/proof-trade.openapi.json", "file"),
    ("contractSchemas", "contracts/schemas", "directory_content"),
    ("nativeClientProfile", "client-profile/native-client-profile.json", "file"),
    ("canonicalFixtures", "fixtures/v1", "directory_content"),
)


class CompatibilityContractError(ValueError):
    """Raised when compatibility inputs violate the fail-closed contract."""


@dataclass(frozen=True)
class PackageIdentity:
    version: str
    build_commit: str
    update_channel: str
    api_contract_version: str
    manifest_version: int

    def as_camel_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "buildCommit": self.build_commit,
            "updateChannel": self.update_channel,
            "apiContractVersion": self.api_contract_version,
            "manifestVersion": self.manifest_version,
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompatibilityContractError(f"Required compatibility input is missing: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CompatibilityContractError(f"Required compatibility input is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise CompatibilityContractError(f"Required compatibility input is not an object: {path.name}")
    return value


def load_package_identity(
    *,
    manifest_path: Path = BUILD_MANIFEST_PATH,
    package_json_path: Path = PACKAGE_JSON_PATH,
) -> PackageIdentity:
    """Load the advertised package identity without substituting checkout HEAD."""
    manifest = _read_json_object(manifest_path)
    package = _read_json_object(package_json_path)

    version = str(manifest.get("version", "")).strip()
    package_version = str(package.get("version", "")).strip()
    build_commit = str(manifest.get("build_commit", "")).strip().lower()
    update_channel = str(manifest.get("update_channel", "")).strip()
    api_contract_version = str(manifest.get("api_contract_version", "")).strip()
    manifest_version = manifest.get("manifest_version")
    if type(manifest_version) is not int:
        raise CompatibilityContractError("Build manifest_version must be an integer")

    if not version or version != package_version:
        raise CompatibilityContractError(
            "Package version does not match moto-update-manifest.json"
        )
    if not BUILD_COMMIT_PATTERN.fullmatch(build_commit):
        raise CompatibilityContractError("Packaged build_commit must be a 40-character lowercase SHA")
    if not update_channel:
        raise CompatibilityContractError("Packaged update_channel is required")
    if not api_contract_version:
        raise CompatibilityContractError("Packaged api_contract_version is required")
    if manifest_version != 1:
        raise CompatibilityContractError(f"Unsupported build manifest version: {manifest_version}")
    if api_contract_version != build_info.API_CONTRACT_VERSION:
        raise CompatibilityContractError(
            "Packaged api_contract_version does not match the runtime build default"
        )

    return PackageIdentity(
        version=version,
        build_commit=build_commit,
        update_channel=update_channel,
        api_contract_version=api_contract_version,
        manifest_version=manifest_version,
    )


def _reject_case_colliding_frozen_asset_paths(plan_root: Path) -> None:
    """Reject aliases that differ from a reserved frozen path only by case."""
    reserved_paths = {
        Path(relative_path).as_posix().casefold(): Path(relative_path).as_posix()
        for _, relative_path, _ in FROZEN_ASSET_REQUIREMENTS
    }
    if not plan_root.exists():
        return
    try:
        children = tuple(plan_root.rglob("*"))
    except OSError as exc:
        raise CompatibilityContractError(
            "Frozen SyntheticLib asset tree could not be inspected"
        ) from exc
    for child in children:
        try:
            relative = child.relative_to(plan_root).as_posix()
        except ValueError as exc:  # pragma: no cover - Path.rglob containment guarantee
            raise CompatibilityContractError(
                "Frozen SyntheticLib asset path escaped the trusted plan root"
            ) from exc
        expected = reserved_paths.get(relative.casefold())
        if expected is not None and relative != expected:
            raise CompatibilityContractError(
                f"Case-colliding frozen asset path is not allowed: {relative} "
                f"(reserved path: {expected})"
            )


def inventory_frozen_assets(plan_root: Path = PLAN_ROOT) -> dict[str, dict[str, Any]]:
    """Classify frozen SyntheticLib-owned inputs without reading private/runtime data."""
    _reject_case_colliding_frozen_asset_paths(plan_root)
    inventory: dict[str, dict[str, Any]] = {}
    for asset_id, relative_path, kind in FROZEN_ASSET_REQUIREMENTS:
        path = plan_root / relative_path
        if kind == "file":
            available = path.is_file()
        elif kind == "directory_content":
            available = path.is_dir() and any(child.is_file() for child in path.rglob("*"))
        else:  # pragma: no cover - module-owned constant
            raise CompatibilityContractError(f"Unknown asset requirement kind: {kind}")
        inventory[asset_id] = {
            "path": relative_path,
            "kind": kind,
            "available": available,
        }
    return inventory


def _assert_expected_unpublished_state(inventory: Mapping[str, Mapping[str, Any]]) -> None:
    unknown = set(inventory) - {item[0] for item in FROZEN_ASSET_REQUIREMENTS}
    if unknown:
        raise CompatibilityContractError(
            f"Unknown frozen asset identifiers: {', '.join(sorted(unknown))}"
        )
    if any(bool(item.get("available")) for item in inventory.values()):
        raise CompatibilityContractError(
            "Frozen SyntheticLib assets are present but no supported contract version is registered"
        )


def build_compatibility_matrix(
    identity: PackageIdentity,
    asset_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _assert_expected_unpublished_state(asset_inventory)
    return {
        "schemaVersion": MATRIX_SCHEMA_VERSION,
        "matrixId": (
            f"moto-{identity.version}-{identity.api_contract_version}-"
            f"{identity.build_commit}"
        ),
        "moto": identity.as_camel_dict(),
        "motoPolicyVersions": dict(MOTO_POLICY_VERSIONS),
        "syntheticLib": {
            "proofTradeContractVersion": None,
            "nativeClientProfileVersion": None,
            "fixtureVersion": None,
            "serverRelease": None,
        },
        "assetInventory": dict(asset_inventory),
        "support": {
            "motoPreflightSupported": True,
            "supportedForLiveProofTrade": False,
            "status": "blocked",
            "reasonCode": BLOCKED_REASON,
            "unknownVersions": "update_required",
            "unknownEnums": "update_required",
        },
    }


def _expected_unpublished_inventory() -> dict[str, dict[str, Any]]:
    return {
        asset_id: {
            "path": relative_path,
            "kind": kind,
            "available": False,
        }
        for asset_id, relative_path, kind in FROZEN_ASSET_REQUIREMENTS
    }


def validate_matrix(
    identity: PackageIdentity,
    matrix: Mapping[str, Any],
) -> None:
    expected = build_compatibility_matrix(identity, _expected_unpublished_inventory())
    if matrix != expected:
        raise CompatibilityContractError(
            "Compatibility matrix does not exactly match the supported unpublished state"
        )


def build_receipt(
    identity: PackageIdentity,
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    validate_matrix(identity, matrix)
    support = matrix["support"]
    return {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "canonicalization": CANONICALIZATION_VERSION,
        "receiptId": (
            f"moto-{identity.version}-{identity.api_contract_version}-"
            f"{identity.build_commit}"
        ),
        "moto": identity.as_camel_dict(),
        "syntheticLib": dict(matrix["syntheticLib"]),
        "compatibility": {
            "status": support["status"],
            "supportedForLiveProofTrade": support["supportedForLiveProofTrade"],
            "reasonCode": support["reasonCode"],
        },
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return canonical v1 JSON (UTF-8, sorted keys, compact, finite values)."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_frozen_asset_manifest(
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schemaVersion": ASSET_MANIFEST_SCHEMA_VERSION,
        "status": "unpublished",
        "validationPolicy": {
            "exactMemberSetRequired": True,
            "contentHashesRequired": True,
            "semanticValidationRequired": True,
            "symlinksAllowed": False,
            "caseCollisionsAllowed": False,
            "pathsMustRemainUnderTrustedRoot": True,
        },
        "assets": dict(inventory),
    }


def _member_record(content: bytes) -> dict[str, Any]:
    return {"byteLength": len(content), "sha256": sha256_bytes(content)}


def build_independent_test_result(
    identity: PackageIdentity,
    matrix_bytes: bytes,
    receipt_bytes: bytes,
    digest_bytes: bytes,
    assets_bytes: bytes,
) -> dict[str, Any]:
    """Parse and verify candidate bytes before issuing a scoped result."""
    try:
        matrix = json.loads(matrix_bytes.decode("utf-8"))
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        assets = json.loads(assets_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityContractError("Compatibility evidence is not valid UTF-8 JSON") from exc
    if not all(isinstance(item, dict) for item in (matrix, receipt, assets)):
        raise CompatibilityContractError("Compatibility evidence roots must be JSON objects")

    validate_matrix(identity, matrix)
    if matrix_bytes != canonical_json_bytes(matrix):
        raise CompatibilityContractError("Compatibility matrix is noncanonical")
    expected_receipt = build_receipt(identity, matrix)
    if receipt != expected_receipt or receipt_bytes != canonical_json_bytes(expected_receipt):
        raise CompatibilityContractError("Normalized receipt is noncanonical or contradictory")
    if assets.get("schemaVersion") != ASSET_MANIFEST_SCHEMA_VERSION:
        raise CompatibilityContractError("Unsupported frozen-asset manifest schema")
    expected_assets = build_frozen_asset_manifest(matrix["assetInventory"])
    if assets != expected_assets or assets_bytes != canonical_json_bytes(expected_assets):
        raise CompatibilityContractError(
            "Frozen-asset manifest is noncanonical or contradicts the matrix"
        )

    digest = sha256_bytes(receipt_bytes)
    expected_digest_bytes = (
        f"{digest}  {ARTIFACT_FILENAMES['receipt']}\n".encode("ascii")
    )
    if digest_bytes != expected_digest_bytes:
        raise CompatibilityContractError("Detached receipt digest mismatch")

    return {
        "schemaVersion": TEST_RESULT_SCHEMA_VERSION,
        "scope": PREFLIGHT_SCOPE,
        "status": "passed",
        "passed": True,
        "verifierVersion": VERIFIER_VERSION,
        "packageIdentity": identity.as_camel_dict(),
        "verifiedInputs": {
            ARTIFACT_FILENAMES["matrix"]: _member_record(matrix_bytes),
            ARTIFACT_FILENAMES["receipt"]: _member_record(receipt_bytes),
            ARTIFACT_FILENAMES["digest"]: _member_record(digest_bytes),
            ARTIFACT_FILENAMES["assets"]: _member_record(assets_bytes),
        },
        "checks": {
            "packageAndManifestVersionsAgree": True,
            "apiContractMatchesPublicRuntimeContract": True,
            "matrixContainsExactPackageIdentity": True,
            "frozenAssetManifestMatchesMatrix": True,
            "unknownVersionsAndEnumsFailClosed": True,
            "canonicalReceiptDigestVerified": True,
        },
        "syntheticLibWireContract": {
            "tested": False,
            "status": "blocked",
            "supportedForLiveProofTrade": False,
            "reasonCode": BLOCKED_REASON,
        },
    }


def build_execution_attestation(
    identity: PackageIdentity,
    test_result_bytes: bytes,
    *,
    executor_commit: str | None = None,
    executor_tree: str | None = None,
    dirty: bool | None = None,
    observed_at: str | None = None,
    platform_name: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    verifier_hash = sha256_bytes(Path(__file__).read_bytes())
    return {
        "schemaVersion": ATTESTATION_SCHEMA_VERSION,
        "attestationMode": "local_diagnostic" if dirty is not False else "clean_source",
        "subject": identity.as_camel_dict(),
        "executor": {
            "sourceCommit": executor_commit,
            "sourceTree": executor_tree,
            "dirty": dirty,
            "verifierVersion": VERIFIER_VERSION,
            "verifierSha256": verifier_hash,
            "command": "python -m scripts.check_syntheticlib_trade_contract --check",
            "platform": platform_name,
            "pythonVersion": python_version,
        },
        "observedAt": observed_at,
        "result": _member_record(test_result_bytes),
    }


def _validate_utc_timestamp(value: str) -> None:
    if not UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise CompatibilityContractError(
            "Clean-source observedAt must be an RFC 3339 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CompatibilityContractError(
            "Clean-source observedAt is not a valid UTC timestamp"
        ) from exc
    if parsed.tzinfo != timezone.utc:
        raise CompatibilityContractError("Clean-source observedAt must use UTC")


def build_clean_source_attestation(
    identity: PackageIdentity,
    test_result_bytes: bytes,
    *,
    source_commit: str,
    source_tree: str,
    observed_at: str,
    platform_name: str,
    python_version: str,
) -> dict[str, Any]:
    """Bind an independently observed clean checkout to the advertised package."""
    source_commit = source_commit.strip().lower()
    source_tree = source_tree.strip().lower()
    if not BUILD_COMMIT_PATTERN.fullmatch(source_commit):
        raise CompatibilityContractError(
            "Clean-source commit must be a 40-character lowercase Git SHA"
        )
    if source_commit != identity.build_commit:
        raise CompatibilityContractError(
            "Clean-source commit does not match the advertised package build_commit"
        )
    if not BUILD_COMMIT_PATTERN.fullmatch(source_tree):
        raise CompatibilityContractError(
            "Clean-source tree must be a 40-character lowercase Git tree SHA"
        )
    _validate_utc_timestamp(observed_at)
    if not platform_name.strip():
        raise CompatibilityContractError("Clean-source platform is required")
    if not python_version.strip():
        raise CompatibilityContractError("Clean-source Python version is required")

    attestation = build_execution_attestation(
        identity,
        test_result_bytes,
        executor_commit=source_commit,
        executor_tree=source_tree,
        dirty=False,
        observed_at=observed_at,
        platform_name=platform_name.strip(),
        python_version=python_version.strip(),
    )
    if attestation["attestationMode"] != "clean_source":  # pragma: no cover
        raise CompatibilityContractError("Clean-source attestation mode was not selected")
    return attestation


def build_release_evidence(
    identity: PackageIdentity,
    test_result_bytes: bytes,
    *,
    source_commit: str,
    source_tree: str,
    observed_at: str,
    platform_name: str,
    python_version: str,
) -> dict[str, bytes]:
    """Create the separate, non-checked-in clean-source release evidence set."""
    attestation_bytes = canonical_json_bytes(
        build_clean_source_attestation(
            identity,
            test_result_bytes,
            source_commit=source_commit,
            source_tree=source_tree,
            observed_at=observed_at,
            platform_name=platform_name,
            python_version=python_version,
        )
    )
    bundle_payload = {
        "schemaVersion": RELEASE_BUNDLE_SCHEMA_VERSION,
        "canonicalization": CANONICALIZATION_VERSION,
        "subject": identity.as_camel_dict(),
        "members": {
            RELEASE_ARTIFACT_FILENAMES["attestation"]: _member_record(
                attestation_bytes
            )
        },
    }
    bundle = {
        **bundle_payload,
        "bundleDigest": sha256_bytes(canonical_json_bytes(bundle_payload)),
    }
    return {
        RELEASE_ARTIFACT_FILENAMES["attestation"]: attestation_bytes,
        RELEASE_ARTIFACT_FILENAMES["bundle"]: canonical_json_bytes(bundle),
    }


def build_bundle_manifest(members: Mapping[str, bytes]) -> dict[str, Any]:
    payload = {
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "canonicalization": CANONICALIZATION_VERSION,
        "members": {
            filename: _member_record(content)
            for filename, content in sorted(members.items())
        },
    }
    return {
        **payload,
        "bundleDigest": sha256_bytes(canonical_json_bytes(payload)),
    }


def render_markdown(
    identity: PackageIdentity,
    matrix: Mapping[str, Any],
    receipt: Mapping[str, Any],
    digest: str,
) -> str:
    unavailable = sorted(
        item["path"]
        for item in matrix["assetInventory"].values()
        if not item["available"]
    )
    missing_list = "\n".join(f"- `{path}`" for path in unavailable)
    policy_list = "\n".join(
        f"- {name}: `{version}`"
        for name, version in sorted(matrix["motoPolicyVersions"].items())
    )
    normalized = canonical_json_bytes(receipt).decode("utf-8")
    return f"""# MOTO SyntheticLib Proof-Trade Compatibility Receipt

## Scope

This is the MOTO-side compatibility receipt required by
`PLANS/MOTO_SYNTHETICLIB_TRADE_INTEGRATION_PLAN.md`. It records the supported
MOTO package/build identity, MOTO API contract version, compatibility decision,
scoped contract-preflight result, and normalized receipt digest.

This receipt does not activate proof trade, authorize a live endpoint, accept
legal terms, represent SyntheticLib server acceptance, or claim that an
unpublished SyntheticLib wire contract passed.
It is deterministic advertised-package evidence, not an independently executed
clean-source release attestation.

## Supported MOTO package/build identity

- Product: `MOTO Autonomous ASI`
- Package version: `{identity.version}`
- Build commit: `{identity.build_commit}`
- Update channel: `{identity.update_channel}`
- Identity source: repository-root `moto-update-manifest.json`

The identity above is the advertised package identity. Runtime Git/archive
identity remains a separate diagnostic and is never substituted into this
release receipt.

## Contract versions

- MOTO API contract version: `{identity.api_contract_version}`
- Compatibility matrix: `{matrix["schemaVersion"]}`
- Contract-test result: `{TEST_RESULT_SCHEMA_VERSION}`
- Compatibility receipt: `{receipt["schemaVersion"]}`
- Canonicalization: `{CANONICALIZATION_VERSION}`
- MOTO-owned proof-selection and provenance policies:
{policy_list}
- SyntheticLib proof-trade contract, native-client profile, canonical fixtures,
  and server release: **unpublished**

## Compatibility matrix

- Exact MOTO package/build/API-contract tuple: **supported by MOTO preflight**
- Frozen SyntheticLib proof-trade wire contract: **not available**
- Live proof-trade compatibility: **blocked**
- `supportedForLiveProofTrade`: `false`
- Reason code: `{BLOCKED_REASON}`
- Unknown versions and enum values: **update required; fail closed**

Unavailable frozen inputs:

{missing_list}

The authoritative machine-readable matrix is `compatibility-matrix.json`.

## Contract-test result

- Scope: `{PREFLIGHT_SCOPE}`
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
{normalized}
```

Normalized receipt SHA-256:

`{digest}`

The detached digest is `moto-compatibility-receipt.sha256`.

## Release acceptance gate

Live compatibility remains blocked until SyntheticLib publishes frozen,
versioned wire contracts, fixtures, and a registered native-client profile and
MOTO passes a separate credential-free wire-contract suite against those exact
artifacts. Staging and production remain separately authorized.
"""


def build_artifacts(
    *,
    manifest_path: Path = BUILD_MANIFEST_PATH,
    package_json_path: Path = PACKAGE_JSON_PATH,
    plan_root: Path = PLAN_ROOT,
) -> dict[str, bytes]:
    """Build candidates, independently verify them, then bind the evidence set."""
    identity = load_package_identity(
        manifest_path=manifest_path,
        package_json_path=package_json_path,
    )
    inventory = inventory_frozen_assets(plan_root)
    matrix = build_compatibility_matrix(identity, inventory)
    receipt = build_receipt(identity, matrix)
    digest = receipt_digest(receipt)
    if not SHA256_PATTERN.fullmatch(digest):  # pragma: no cover - hashlib guarantee
        raise CompatibilityContractError("Receipt digest is invalid")

    candidates = {
        ARTIFACT_FILENAMES["matrix"]: canonical_json_bytes(matrix),
        ARTIFACT_FILENAMES["receipt"]: canonical_json_bytes(receipt),
        ARTIFACT_FILENAMES["digest"]: f"{digest}  {ARTIFACT_FILENAMES['receipt']}\n".encode("ascii"),
        ARTIFACT_FILENAMES["assets"]: canonical_json_bytes(
            build_frozen_asset_manifest(inventory)
        ),
    }
    test_result = build_independent_test_result(
        identity,
        candidates[ARTIFACT_FILENAMES["matrix"]],
        candidates[ARTIFACT_FILENAMES["receipt"]],
        candidates[ARTIFACT_FILENAMES["digest"]],
        candidates[ARTIFACT_FILENAMES["assets"]],
    )
    test_result_bytes = canonical_json_bytes(test_result)
    attestation = build_execution_attestation(identity, test_result_bytes)
    attestation_bytes = canonical_json_bytes(attestation)
    bundle_members = {
        **candidates,
        ARTIFACT_FILENAMES["test_result"]: test_result_bytes,
        ARTIFACT_FILENAMES["attestation"]: attestation_bytes,
    }
    bundle = build_bundle_manifest(bundle_members)
    artifacts = {
        **bundle_members,
        ARTIFACT_FILENAMES["bundle"]: canonical_json_bytes(bundle),
        ARTIFACT_FILENAMES["markdown"]: render_markdown(
            identity,
            matrix,
            receipt,
            digest,
        ).encode("utf-8"),
    }
    return artifacts


def verify_evidence_set(
    *,
    manifest_path: Path = BUILD_MANIFEST_PATH,
    package_json_path: Path = PACKAGE_JSON_PATH,
    plan_root: Path = PLAN_ROOT,
    output_dir: Path = COMPATIBILITY_DIR,
) -> list[str]:
    """Independently parse and verify a closed checked-in evidence set."""
    identity = load_package_identity(
        manifest_path=manifest_path,
        package_json_path=package_json_path,
    )
    expected_names = set(ARTIFACT_FILENAMES.values())
    try:
        children = list(output_dir.iterdir())
    except OSError:
        return sorted(expected_names)
    if output_dir.is_symlink() or any(child.is_symlink() for child in children):
        return ["symlinked_evidence"]
    invalid_entries = sorted(
        child.name
        for child in children
        if not child.is_file() or (child.name == "README.md" and not child.is_file())
    )
    if invalid_entries:
        return [f"non_file_evidence:{name}" for name in invalid_entries]
    actual_names = {
        child.name
        for child in children
        if child.name != "README.md"
    }
    if actual_names != expected_names:
        return sorted(expected_names.symmetric_difference(actual_names))

    try:
        members = {name: (output_dir / name).read_bytes() for name in expected_names}
        matrix = json.loads(members[ARTIFACT_FILENAMES["matrix"]].decode("utf-8"))
        receipt = json.loads(members[ARTIFACT_FILENAMES["receipt"]].decode("utf-8"))
        assets = json.loads(members[ARTIFACT_FILENAMES["assets"]].decode("utf-8"))
        result = json.loads(members[ARTIFACT_FILENAMES["test_result"]].decode("utf-8"))
        bundle = json.loads(members[ARTIFACT_FILENAMES["bundle"]].decode("utf-8"))
        attestation = json.loads(members[ARTIFACT_FILENAMES["attestation"]].decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["malformed_evidence"]

    try:
        validate_matrix(identity, matrix)
        if members[ARTIFACT_FILENAMES["matrix"]] != canonical_json_bytes(matrix):
            raise CompatibilityContractError("Matrix is noncanonical")
        expected_receipt = build_receipt(identity, matrix)
        if receipt != expected_receipt:
            raise CompatibilityContractError("Receipt contradicts matrix")
        verified_result = build_independent_test_result(
            identity,
            members[ARTIFACT_FILENAMES["matrix"]],
            members[ARTIFACT_FILENAMES["receipt"]],
            members[ARTIFACT_FILENAMES["digest"]],
            members[ARTIFACT_FILENAMES["assets"]],
        )
        if result != verified_result or members[ARTIFACT_FILENAMES["test_result"]] != canonical_json_bytes(result):
            raise CompatibilityContractError("Test result is not independently reproducible")
        if assets != build_frozen_asset_manifest(inventory_frozen_assets(plan_root)):
            raise CompatibilityContractError("Frozen-asset inventory is stale")
        expected_attestation = build_execution_attestation(
            identity,
            members[ARTIFACT_FILENAMES["test_result"]],
        )
        if attestation != expected_attestation:
            raise CompatibilityContractError("Deterministic attestation is stale")
        bound_members = {
            name: content
            for name, content in members.items()
            if name not in {ARTIFACT_FILENAMES["bundle"], ARTIFACT_FILENAMES["markdown"]}
        }
        expected_bundle = build_bundle_manifest(bound_members)
        if bundle != expected_bundle or members[ARTIFACT_FILENAMES["bundle"]] != canonical_json_bytes(bundle):
            raise CompatibilityContractError("Evidence bundle manifest mismatch")
        expected_markdown = render_markdown(
            identity,
            matrix,
            receipt,
            receipt_digest(receipt),
        ).encode("utf-8")
        if members[ARTIFACT_FILENAMES["markdown"]] != expected_markdown:
            raise CompatibilityContractError("Markdown projection is stale")
    except (CompatibilityContractError, KeyError, TypeError) as exc:
        return [f"semantic_evidence_mismatch:{type(exc).__name__}:{exc}"]
    return []


def verify_release_evidence(
    *,
    source_commit: str,
    source_tree: str,
    observed_at: str,
    platform_name: str,
    python_version: str,
    manifest_path: Path = BUILD_MANIFEST_PATH,
    package_json_path: Path = PACKAGE_JSON_PATH,
    plan_root: Path = PLAN_ROOT,
    compatibility_dir: Path = COMPATIBILITY_DIR,
    output_dir: Path,
) -> list[str]:
    """Verify a closed clean-source release set against checked-in base evidence."""
    base_drift = verify_evidence_set(
        manifest_path=manifest_path,
        package_json_path=package_json_path,
        plan_root=plan_root,
        output_dir=compatibility_dir,
    )
    if base_drift:
        return [f"base_evidence:{item}" for item in base_drift]
    identity = load_package_identity(
        manifest_path=manifest_path,
        package_json_path=package_json_path,
    )
    try:
        test_result_bytes = (
            compatibility_dir / ARTIFACT_FILENAMES["test_result"]
        ).read_bytes()
        expected = build_release_evidence(
            identity,
            test_result_bytes,
            source_commit=source_commit,
            source_tree=source_tree,
            observed_at=observed_at,
            platform_name=platform_name,
            python_version=python_version,
        )
        children = list(output_dir.iterdir())
    except (OSError, CompatibilityContractError) as exc:
        return [f"release_evidence_error:{type(exc).__name__}:{exc}"]
    expected_names = set(RELEASE_ARTIFACT_FILENAMES.values())
    if output_dir.is_symlink() or any(child.is_symlink() for child in children):
        return ["symlinked_release_evidence"]
    invalid_entries = sorted(child.name for child in children if not child.is_file())
    if invalid_entries:
        return [f"non_file_release_evidence:{name}" for name in invalid_entries]
    actual_names = {child.name for child in children}
    if actual_names != expected_names:
        return sorted(expected_names.symmetric_difference(actual_names))
    return verify_artifacts(expected, output_dir=output_dir)


def verify_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    output_dir: Path = COMPATIBILITY_DIR,
) -> list[str]:
    drift: list[str] = []
    for filename, expected in artifacts.items():
        path = output_dir / filename
        try:
            actual = path.read_bytes()
        except OSError:
            drift.append(filename)
            continue
        if actual != expected:
            drift.append(filename)
    return drift


def write_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    output_dir: Path = COMPATIBILITY_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise CompatibilityContractError("Refusing to write through a symlinked evidence root")
    for filename in artifacts:
        if not filename or Path(filename).name != filename:
            raise CompatibilityContractError(
                f"Evidence filename must be one safe path component: {filename!r}"
            )
        target = output_dir / filename
        if target.is_symlink():
            raise CompatibilityContractError(f"Refusing to replace symlink: {filename}")
        if target.exists() and not target.is_file():
            raise CompatibilityContractError(
                f"Refusing to replace non-file evidence target: {filename}"
            )

    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temp_dir:
        transaction = Path(temp_dir)
        staging = transaction / "staging"
        backup = transaction / "backup"
        staging.mkdir()
        backup.mkdir()
        for filename, content in artifacts.items():
            (staging / filename).write_bytes(content)

        backed_up: list[str] = []
        published: list[str] = []
        try:
            for filename in artifacts:
                target = output_dir / filename
                if target.exists():
                    target.replace(backup / filename)
                    backed_up.append(filename)
            for filename in artifacts:
                (staging / filename).replace(output_dir / filename)
                published.append(filename)
        except BaseException as exc:
            rollback_errors: list[str] = []
            for filename in reversed(published):
                target = output_dir / filename
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    rollback_errors.append(filename)
            for filename in reversed(backed_up):
                saved = backup / filename
                try:
                    if saved.exists():
                        saved.replace(output_dir / filename)
                except OSError:
                    rollback_errors.append(filename)
            if rollback_errors:
                raise CompatibilityContractError(
                    "Evidence publication failed and rollback was incomplete for: "
                    + ", ".join(sorted(set(rollback_errors)))
                ) from exc
            if not isinstance(exc, OSError):
                raise
            raise CompatibilityContractError(
                "Evidence publication failed; the previous generation was restored"
            ) from exc
