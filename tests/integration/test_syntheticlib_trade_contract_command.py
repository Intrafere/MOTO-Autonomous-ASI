import json
from pathlib import Path
import subprocess

from backend.shared.build_info import API_CONTRACT_VERSION
from scripts.check_syntheticlib_trade_contract import main


BUILD_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"


def _inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    manifest = root / "moto-update-manifest.json"
    package = root / "package.json"
    plan_root = root / "plan"
    output = root / "output"
    plan_root.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "version": "1.2.3",
                "build_commit": BUILD_COMMIT,
                "update_channel": "main",
                "api_contract_version": API_CONTRACT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    package.write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
    return manifest, package, plan_root, output


def _args(
    manifest: Path,
    package: Path,
    plan_root: Path,
    output: Path,
    mode: str,
) -> list[str]:
    return [
        mode,
        "--allow-custom-roots",
        "--manifest",
        str(manifest),
        "--package-json",
        str(package),
        "--plan-root",
        str(plan_root),
        "--output-dir",
        str(output),
    ]


def _initialize_clean_checkout(root: Path) -> tuple[str, str]:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Compatibility Test"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "compatibility-test@invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "--all"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "compatibility fixture"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tree


def _release_args(
    manifest: Path,
    package: Path,
    plan_root: Path,
    output: Path,
    release_output: Path,
    source_root: Path,
    mode: str,
) -> list[str]:
    return [
        *_args(manifest, package, plan_root, output, mode),
        "--release-output-dir",
        str(release_output),
        "--source-root",
        str(source_root),
    ]


def test_command_writes_then_checks_deterministic_artifacts(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)

    assert main(_args(manifest, package, plan_root, output, "--write")) == 0
    assert main(_args(manifest, package, plan_root, output, "--check")) == 0


def test_command_detects_artifact_drift(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    assert main(_args(manifest, package, plan_root, output, "--write")) == 0
    (output / "moto-compatibility-receipt.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert main(_args(manifest, package, plan_root, output, "--check")) == 1


def test_command_rejects_inconsistent_identity(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    package.write_text(json.dumps({"version": "different"}), encoding="utf-8")

    assert main(_args(manifest, package, plan_root, output, "--check")) == 2


def test_command_rejects_extra_evidence_file(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    assert main(_args(manifest, package, plan_root, output, "--write")) == 0
    (output / "stale-result.json").write_text("{}", encoding="utf-8")

    assert main(_args(manifest, package, plan_root, output, "--check")) == 1


def test_command_rejects_coherently_rewritten_semantic_matrix(
    tmp_path: Path,
) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    assert main(_args(manifest, package, plan_root, output, "--write")) == 0
    matrix_path = output / "compatibility-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["support"]["unknownEnums"] = "allow"
    matrix_path.write_text(
        json.dumps(matrix, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    assert main(_args(manifest, package, plan_root, output, "--check")) == 1


def test_command_rejects_receipt_with_utf8_bom(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    assert main(_args(manifest, package, plan_root, output, "--write")) == 0
    receipt_path = output / "moto-compatibility-receipt.json"
    receipt_path.write_bytes(b"\xef\xbb\xbf" + receipt_path.read_bytes())

    assert main(_args(manifest, package, plan_root, output, "--check")) == 1


def test_command_rejects_custom_roots_without_explicit_test_flag(
    tmp_path: Path,
) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    args = _args(manifest, package, plan_root, output, "--check")
    args.remove("--allow-custom-roots")

    assert main(args) == 2


def test_release_command_round_trip_uses_explicit_clean_source_root(
    tmp_path: Path,
) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    commit, _ = _initialize_clean_checkout(tmp_path)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["build_commit"] = commit
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    subprocess.run(["git", "add", "moto-update-manifest.json"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "bind advertised commit"],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_value["build_commit"] = commit
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    subprocess.run(["git", "add", "moto-update-manifest.json"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "--amend", "--no-edit"],
        cwd=tmp_path,
        check=True,
    )
    # A manifest cannot contain the hash of the commit that contains itself.
    # Release staging therefore lives outside the immutable source checkout.
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    release_root = tmp_path.parent / f"{tmp_path.name}-release-inputs"
    release_root.mkdir()
    release_manifest = release_root / "moto-update-manifest.json"
    manifest_value["build_commit"] = source_commit
    release_manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    release_package = release_root / "package.json"
    release_package.write_bytes(package.read_bytes())
    release_plan = release_root / "plan"
    release_plan.mkdir()
    release_output = release_root / "compatibility"
    release_evidence = release_root / "release-evidence"

    assert main(
        _args(
            release_manifest,
            release_package,
            release_plan,
            release_output,
            "--write",
        )
    ) == 0
    assert main(
        _release_args(
            release_manifest,
            release_package,
            release_plan,
            release_output,
            release_evidence,
            tmp_path,
            "--write-release-evidence",
        )
    ) == 0
    assert main(
        _release_args(
            release_manifest,
            release_package,
            release_plan,
            release_output,
            release_evidence,
            tmp_path,
            "--check-release-evidence",
        )
    ) == 0


def test_release_command_rejects_dirty_source_root(tmp_path: Path) -> None:
    manifest, package, plan_root, output = _inputs(tmp_path)
    commit, _ = _initialize_clean_checkout(tmp_path)
    release_root = tmp_path.parent / f"{tmp_path.name}-release-inputs"
    release_root.mkdir()
    release_manifest = release_root / "moto-update-manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["build_commit"] = commit
    release_manifest.write_text(json.dumps(value), encoding="utf-8")
    release_package = release_root / "package.json"
    release_package.write_bytes(package.read_bytes())
    release_plan = release_root / "plan"
    release_plan.mkdir()
    release_output = release_root / "compatibility"
    release_evidence = release_root / "release-evidence"
    assert main(
        _args(
            release_manifest,
            release_package,
            release_plan,
            release_output,
            "--write",
        )
    ) == 0
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")

    assert main(
        _release_args(
            release_manifest,
            release_package,
            release_plan,
            release_output,
            release_evidence,
            tmp_path,
            "--write-release-evidence",
        )
    ) == 2
