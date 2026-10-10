import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from backend.shared.build_info import API_CONTRACT_VERSION
from backend.shared.syntheticlib_trade_compatibility import (
    BLOCKED_REASON,
    CompatibilityContractError,
    build_artifacts,
    build_clean_source_attestation,
    build_compatibility_matrix,
    build_independent_test_result,
    build_release_evidence,
    canonical_json_bytes,
    inventory_frozen_assets,
    load_package_identity,
    receipt_digest,
    validate_matrix,
    verify_evidence_set,
    verify_release_evidence,
    write_artifacts,
)


BUILD_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_identity(root: Path, *, version: str = "1.2.3") -> tuple[Path, Path]:
    manifest = root / "moto-update-manifest.json"
    package = root / "package.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "version": version,
                "build_commit": BUILD_COMMIT,
                "update_channel": "main",
                "api_contract_version": API_CONTRACT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    package.write_text(json.dumps({"version": version}), encoding="utf-8")
    return manifest, package


def test_package_identity_requires_manifest_and_package_version_agreement(
    tmp_path: Path,
) -> None:
    manifest, package = _write_identity(tmp_path)
    package.write_text(json.dumps({"version": "9.9.9"}), encoding="utf-8")

    with pytest.raises(CompatibilityContractError, match="Package version"):
        load_package_identity(manifest_path=manifest, package_json_path=package)


@pytest.mark.parametrize("manifest_version", [True, 1.5, "1"])
def test_package_identity_requires_a_real_integer_manifest_version(
    tmp_path: Path,
    manifest_version: object,
) -> None:
    manifest, package = _write_identity(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["manifest_version"] = manifest_version
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CompatibilityContractError, match="must be an integer"):
        load_package_identity(manifest_path=manifest, package_json_path=package)


def test_missing_frozen_assets_produce_scoped_passing_preflight(
    tmp_path: Path,
) -> None:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    plan_root.mkdir()

    artifacts = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )
    matrix = json.loads(artifacts["compatibility-matrix.json"])
    result = json.loads(artifacts["moto-contract-test-result.json"])
    receipt = json.loads(artifacts["moto-compatibility-receipt.json"])

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["scope"] == "moto_owned_compatibility_preflight"
    assert result["syntheticLibWireContract"]["tested"] is False
    assert matrix["support"]["supportedForLiveProofTrade"] is False
    assert matrix["support"]["reasonCode"] == BLOCKED_REASON
    assert receipt["moto"] == matrix["moto"] == result["packageIdentity"]
    assert json.loads(artifacts["compatibility-bundle-manifest.json"])[
        "schemaVersion"
    ] == "moto-syntheticlib-evidence-bundle.v1"


def test_unregistered_frozen_assets_fail_closed(tmp_path: Path) -> None:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    contract_path = plan_root / "contracts" / "proof-trade.openapi.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("{}", encoding="utf-8")
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )

    with pytest.raises(
        CompatibilityContractError,
        match="no supported contract version is registered",
    ):
        build_compatibility_matrix(identity, inventory_frozen_assets(plan_root))


def test_case_colliding_frozen_asset_path_fails_closed(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    colliding = plan_root / "Contracts" / "proof-trade.openapi.json"
    colliding.parent.mkdir(parents=True)
    colliding.write_text("{}", encoding="utf-8")

    with pytest.raises(CompatibilityContractError, match="Case-colliding"):
        inventory_frozen_assets(plan_root)


def test_unknown_asset_identifier_fails_closed(tmp_path: Path) -> None:
    manifest, package = _write_identity(tmp_path)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    inventory = inventory_frozen_assets(tmp_path / "plan")
    inventory["futureUnknownAsset"] = {
        "path": "unknown",
        "kind": "file",
        "available": False,
    }

    with pytest.raises(CompatibilityContractError, match="Unknown frozen asset"):
        build_compatibility_matrix(identity, inventory)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("matrixId",), "forged"),
        (("syntheticLib", "proofTradeContractVersion"), "v1"),
        (("support", "motoPreflightSupported"), False),
        (("support", "status"), "supported"),
        (("support", "unknownVersions"), "allow"),
        (("support", "unknownEnums"), "allow"),
        (("assetInventory", "proofTradeOpenApi", "available"), True),
    ],
)
def test_matrix_requires_exact_supported_unpublished_state(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    manifest, package = _write_identity(tmp_path)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    matrix = build_compatibility_matrix(
        identity,
        inventory_frozen_assets(tmp_path / "plan"),
    )
    target = matrix
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(CompatibilityContractError, match="exactly match"):
        validate_matrix(identity, matrix)


def test_matrix_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest, package = _write_identity(tmp_path)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    matrix = build_compatibility_matrix(
        identity,
        inventory_frozen_assets(tmp_path / "plan"),
    )
    matrix["futureField"] = True

    with pytest.raises(CompatibilityContractError, match="exactly match"):
        validate_matrix(identity, matrix)


def test_receipt_bytes_and_detached_digest_are_stable(tmp_path: Path) -> None:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    first = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )
    second = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )

    receipt_bytes = first["moto-compatibility-receipt.json"]
    receipt = json.loads(receipt_bytes)
    expected_digest = hashlib.sha256(receipt_bytes).hexdigest()

    assert first == second
    assert receipt_bytes == canonical_json_bytes(receipt)
    assert not receipt_bytes.endswith(b"\n")
    assert receipt_digest(receipt) == expected_digest
    assert first["moto-compatibility-receipt.sha256"] == (
        f"{expected_digest}  moto-compatibility-receipt.json\n".encode("ascii")
    )


def test_canonical_json_is_independent_of_mapping_insertion_order() -> None:
    assert canonical_json_bytes({"z": 1, "a": {"y": 2, "b": 3}}) == (
        b'{"a":{"b":3,"y":2},"z":1}'
    )


@pytest.mark.parametrize("suffix", [b"\r\n", b"\xef\xbb\xbf"])
def test_independent_result_rejects_noncanonical_matrix_bytes(
    tmp_path: Path,
    suffix: bytes,
) -> None:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    artifacts = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )

    with pytest.raises(CompatibilityContractError):
        build_independent_test_result(
            identity,
            artifacts["compatibility-matrix.json"] + suffix,
            artifacts["moto-compatibility-receipt.json"],
            artifacts["moto-compatibility-receipt.sha256"],
            artifacts["frozen-asset-manifest.json"],
        )


def test_artifact_publication_rolls_back_after_mid_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old = {"one.json": b"old-one", "two.json": b"old-two"}
    for filename, content in old.items():
        (output / filename).write_bytes(content)
    replacement = {"one.json": b"new-one", "two.json": b"new-two"}
    original_replace = Path.replace
    failed = False

    def fail_second_staged_publish(source: Path, target: Path) -> Path:
        nonlocal failed
        if source.parent.name == "staging" and source.name == "two.json" and not failed:
            failed = True
            raise OSError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_staged_publish)

    with pytest.raises(CompatibilityContractError, match="previous generation was restored"):
        write_artifacts(replacement, output_dir=output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == old


def test_artifact_publication_rolls_back_before_reraising_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old = {"one.json": b"old-one", "two.json": b"old-two"}
    for filename, content in old.items():
        (output / filename).write_bytes(content)
    replacement = {"one.json": b"new-one", "two.json": b"new-two"}
    original_replace = Path.replace
    interrupted = False

    def interrupt_second_staged_publish(source: Path, target: Path) -> Path:
        nonlocal interrupted
        if source.parent.name == "staging" and source.name == "two.json" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt_second_staged_publish)

    with pytest.raises(KeyboardInterrupt):
        write_artifacts(replacement, output_dir=output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == old


@pytest.mark.parametrize(
    "filename",
    [
        "compatibility-matrix.json",
        "moto-compatibility-receipt.json",
        "moto-compatibility-receipt.sha256",
        "frozen-asset-manifest.json",
        "moto-contract-test-result.json",
        "moto-preflight-attestation.json",
        "compatibility-bundle-manifest.json",
        "MOTO_COMPATIBILITY_RECEIPT.md",
    ],
)
def test_every_evidence_member_mutation_fails_verification(
    tmp_path: Path,
    filename: str,
) -> None:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    output = tmp_path / "output"
    plan_root.mkdir()
    artifacts = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )
    write_artifacts(artifacts, output_dir=output)
    path = output / filename
    path.write_bytes(path.read_bytes() + b" ")

    assert verify_evidence_set(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        output_dir=output,
    )


def _release_inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, bytes]]:
    manifest, package = _write_identity(tmp_path)
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    artifacts = build_artifacts(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
    )
    return manifest, package, plan_root, artifacts


def test_clean_source_attestation_binds_exact_release_subject(tmp_path: Path) -> None:
    manifest, package, _, artifacts = _release_inputs(tmp_path)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    attestation = build_clean_source_attestation(
        identity,
        artifacts["moto-contract-test-result.json"],
        source_commit=BUILD_COMMIT,
        source_tree="89abcdef0123456789abcdef0123456789abcdef",
        observed_at="2026-10-06T03:00:00Z",
        platform_name="Linux-test",
        python_version="3.11.9",
    )

    assert attestation["attestationMode"] == "clean_source"
    assert attestation["subject"] == identity.as_camel_dict()
    assert attestation["executor"]["sourceCommit"] == BUILD_COMMIT
    assert attestation["executor"]["dirty"] is False
    assert attestation["result"]["sha256"] == hashlib.sha256(
        artifacts["moto-contract-test-result.json"]
    ).hexdigest()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_commit": "f" * 40}, "advertised package build_commit"),
        ({"source_tree": ""}, "source tree"),
        ({"observed_at": "2026-10-06 03:00:00"}, "observedAt"),
        ({"platform_name": ""}, "platform"),
        ({"python_version": ""}, "Python version"),
    ],
)
def test_clean_source_attestation_rejects_incomplete_or_mismatched_provenance(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    manifest, package, _, artifacts = _release_inputs(tmp_path)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    values = {
        "source_commit": BUILD_COMMIT,
        "source_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "observed_at": "2026-10-06T03:00:00Z",
        "platform_name": "Linux-test",
        "python_version": "3.11.9",
        **overrides,
    }

    with pytest.raises(CompatibilityContractError, match=message):
        build_clean_source_attestation(
            identity,
            artifacts["moto-contract-test-result.json"],
            **values,
        )


def test_release_evidence_is_closed_and_mutation_detected(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    release_dir = tmp_path / "release"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    kwargs = {
        "source_commit": BUILD_COMMIT,
        "source_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "observed_at": "2026-10-06T03:00:00Z",
        "platform_name": "Linux-test",
        "python_version": "3.11.9",
    }
    release_artifacts = build_release_evidence(
        identity,
        artifacts["moto-contract-test-result.json"],
        **kwargs,
    )
    write_artifacts(release_artifacts, output_dir=release_dir)

    assert verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        **kwargs,
    ) == []

    attestation = release_dir / "moto-clean-source-attestation.json"
    attestation.write_bytes(attestation.read_bytes() + b" ")
    assert verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        **kwargs,
    )


def test_release_evidence_rejects_unexpected_member(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    release_dir = tmp_path / "release"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    kwargs = {
        "source_commit": BUILD_COMMIT,
        "source_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "observed_at": "2026-10-06T03:00:00Z",
        "platform_name": "Linux-test",
        "python_version": "3.11.9",
    }
    write_artifacts(
        build_release_evidence(
            identity,
            artifacts["moto-contract-test-result.json"],
            **kwargs,
        ),
        output_dir=release_dir,
    )
    (release_dir / "unexpected.json").write_text("{}", encoding="utf-8")

    assert "unexpected.json" in verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        **kwargs,
    )


def test_evidence_sets_reject_unexpected_directories(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    (compatibility_dir / "unexpected").mkdir()

    assert verify_evidence_set(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        output_dir=compatibility_dir,
    ) == ["non_file_evidence:unexpected"]


def test_release_evidence_rejects_unexpected_directory(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    release_dir = tmp_path / "release"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    kwargs = {
        "source_commit": BUILD_COMMIT,
        "source_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "observed_at": "2026-10-06T03:00:00Z",
        "platform_name": "Linux-test",
        "python_version": "3.11.9",
    }
    write_artifacts(
        build_release_evidence(
            identity,
            artifacts["moto-contract-test-result.json"],
            **kwargs,
        ),
        output_dir=release_dir,
    )
    (release_dir / "unexpected").mkdir()

    assert verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        **kwargs,
    ) == ["non_file_release_evidence:unexpected"]


def test_release_evidence_rejects_forged_executor_metadata(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    release_dir = tmp_path / "release"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    identity = load_package_identity(
        manifest_path=manifest,
        package_json_path=package,
    )
    kwargs = {
        "source_commit": BUILD_COMMIT,
        "source_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "observed_at": "2026-10-06T03:00:00Z",
        "platform_name": "Linux-test",
        "python_version": "3.11.9",
    }
    write_artifacts(
        build_release_evidence(
            identity,
            artifacts["moto-contract-test-result.json"],
            **kwargs,
        ),
        output_dir=release_dir,
    )

    drift = verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        **{**kwargs, "platform_name": "forged-platform"},
    )
    assert "moto-clean-source-attestation.json" in drift


def test_release_evidence_rejects_symlinked_member(tmp_path: Path) -> None:
    manifest, package, plan_root, artifacts = _release_inputs(tmp_path)
    compatibility_dir = tmp_path / "compatibility"
    release_dir = tmp_path / "release"
    write_artifacts(artifacts, output_dir=compatibility_dir)
    release_dir.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    try:
        (release_dir / "moto-clean-source-attestation.json").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform")
    (release_dir / "moto-clean-source-attestation-bundle.json").write_text(
        "{}",
        encoding="utf-8",
    )

    drift = verify_release_evidence(
        manifest_path=manifest,
        package_json_path=package,
        plan_root=plan_root,
        compatibility_dir=compatibility_dir,
        output_dir=release_dir,
        source_commit=BUILD_COMMIT,
        source_tree="89abcdef0123456789abcdef0123456789abcdef",
        observed_at="2026-10-06T03:00:00Z",
        platform_name="Linux-test",
        python_version="3.11.9",
    )
    assert drift == ["symlinked_release_evidence"]


def test_cli_release_mode_requires_explicit_output_directory() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_syntheticlib_trade_contract",
            "--write-release-evidence",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--release-output-dir is required" in result.stderr


def test_matrix_policy_versions_match_owning_module_constants() -> None:
    from backend.shared.lean_source_evidence import LEAN_COMPILE_RECEIPT_VERSION
    from backend.shared.legacy_port_receipt import LEGACY_PORT_RECEIPT_VERSION
    from backend.shared.local_novelty_receipt import LOCAL_NOVELTY_RECEIPT_VERSION
    from backend.shared.models import (
        LeanCompileReceipt,
        LegacyPortReceipt,
        LocalNoveltyAssessmentReceipt,
    )
    from backend.shared.proof_identity import CANONICAL_PROOF_IDENTITY_VERSION
    from backend.shared.proof_locator import PROOF_LOCATOR_VERSION
    from backend.shared.syntheticlib_source_selection import SOURCE_SELECTION_POLICY_VERSION
    from backend.shared.syntheticlib_trade_compatibility import MOTO_POLICY_VERSIONS
    from backend.shared.trade_data_boundary import TRADE_DATA_BOUNDARY_VERSION

    assert MOTO_POLICY_VERSIONS == {
        "canonicalProofIdentity": CANONICAL_PROOF_IDENTITY_VERSION,
        "proofLocator": PROOF_LOCATOR_VERSION,
        "leanCompileReceipt": LEAN_COMPILE_RECEIPT_VERSION,
        "localNoveltyReceipt": LOCAL_NOVELTY_RECEIPT_VERSION,
        "legacyPortReceipt": LEGACY_PORT_RECEIPT_VERSION,
        "sourceSelectionPolicy": SOURCE_SELECTION_POLICY_VERSION,
        "tradeDataBoundary": TRADE_DATA_BOUNDARY_VERSION,
    }
    assert LeanCompileReceipt.model_fields["schema_version"].default == LEAN_COMPILE_RECEIPT_VERSION
    assert (
        LocalNoveltyAssessmentReceipt.model_fields["schema_version"].default
        == LOCAL_NOVELTY_RECEIPT_VERSION
    )
    assert LegacyPortReceipt.model_fields["schema_version"].default == LEGACY_PORT_RECEIPT_VERSION


def test_matrix_with_altered_policy_versions_is_rejected() -> None:
    from backend.shared.syntheticlib_trade_compatibility import (
        CompatibilityContractError,
        _expected_unpublished_inventory,
        build_compatibility_matrix,
        load_package_identity,
        validate_matrix,
    )

    identity = load_package_identity()
    matrix = build_compatibility_matrix(identity, _expected_unpublished_inventory())
    for mutate in (
        lambda policies: policies.pop("sourceSelectionPolicy"),
        lambda policies: policies.__setitem__("extraPolicy", "moto-extra-v1"),
        lambda policies: policies.__setitem__("proofLocator", "moto-proof-locator-v2"),
    ):
        altered = {**matrix, "motoPolicyVersions": dict(matrix["motoPolicyVersions"])}
        mutate(altered["motoPolicyVersions"])
        with pytest.raises(CompatibilityContractError):
            validate_matrix(identity, altered)
