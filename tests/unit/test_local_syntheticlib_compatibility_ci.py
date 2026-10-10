import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.run_local_syntheticlib_compatibility_ci import (
    LocalCompatibilityCIError,
    SOURCE_FILES,
    _copy_snapshot,
    _initialize_snapshot,
    _retain_release_artifacts,
    _run,
    _write_release_inputs,
    run_local_compatibility_ci,
)


def _source_fixture(root: Path) -> None:
    for relative in SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "moto-update-manifest.json":
            path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.2.3",
                        "build_commit": "0" * 40,
                        "update_channel": "main",
                        "api_contract_version": "build-test",
                    }
                ),
                encoding="utf-8",
            )
        elif relative == "package.json":
            path.write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
        else:
            path.write_text(f"# {relative}\n", encoding="utf-8")


def test_snapshot_copies_only_allowlisted_contract_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    source.mkdir()
    snapshot.mkdir()
    _source_fixture(source)
    secret = source / "backend" / "data" / "secret.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("must not copy", encoding="utf-8")

    _copy_snapshot(source, snapshot)

    assert all((snapshot / relative).is_file() for relative in SOURCE_FILES)
    assert not (snapshot / "backend" / "data").exists()
    assert not (snapshot / ".env").exists()


def test_snapshot_commit_is_clean_and_stable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    source.mkdir()
    snapshot.mkdir()
    _source_fixture(source)
    _copy_snapshot(source, snapshot)

    commit, tree = _initialize_snapshot(snapshot)

    assert len(commit) == 40
    assert len(tree) == 40
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=snapshot,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""

    second_snapshot = tmp_path / "snapshot-second"
    second_snapshot.mkdir()
    _copy_snapshot(source, second_snapshot)
    second_commit, second_tree = _initialize_snapshot(second_snapshot)
    assert second_commit == commit
    assert second_tree == tree


def test_release_manifest_binds_temporary_snapshot_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    release = tmp_path / "release"
    source.mkdir()
    _source_fixture(source)
    commit = "0123456789abcdef0123456789abcdef01234567"

    manifest, package, plan, compatibility, evidence = _write_release_inputs(
        source,
        release,
        commit,
    )

    assert json.loads(manifest.read_text(encoding="utf-8"))["build_commit"] == commit
    assert json.loads(package.read_text(encoding="utf-8"))["version"] == "1.2.3"
    assert compatibility == plan / "compatibility"
    assert compatibility.is_dir()
    assert evidence == release / "release-evidence"


def test_missing_allowlisted_source_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    source.mkdir()
    snapshot.mkdir()
    _source_fixture(source)
    (source / SOURCE_FILES[-1]).unlink()

    with pytest.raises(LocalCompatibilityCIError, match="Required source file"):
        _copy_snapshot(source, snapshot)


def test_snapshot_identity_changes_with_runner_or_workflow_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    first = tmp_path / "first"
    first.mkdir()
    _copy_snapshot(source, first)
    first_commit, _ = _initialize_snapshot(first)

    for relative in (
        "scripts/run_local_syntheticlib_compatibility_ci.py",
        ".github/workflows/syntheticlib-compatibility-release.yml",
    ):
        changed = tmp_path / relative.replace("/", "-")
        changed.mkdir()
        (source / relative).write_text("changed orchestration\n", encoding="utf-8")
        _copy_snapshot(source, changed)
        changed_commit, _ = _initialize_snapshot(changed)
        assert changed_commit != first_commit
        _source_fixture(source)


def test_retain_release_artifacts_accepts_new_or_empty_destination(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "evidence.json").write_text("{}", encoding="utf-8")
    new_destination = tmp_path / "new-output"

    _retain_release_artifacts(release, new_destination)
    assert (new_destination / "evidence.json").read_text(encoding="utf-8") == "{}"

    empty_destination = tmp_path / "empty-output"
    empty_destination.mkdir()
    _retain_release_artifacts(release, empty_destination)
    assert (empty_destination / "evidence.json").read_text(encoding="utf-8") == "{}"


def test_retain_release_artifacts_preserves_nonempty_destination(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "evidence.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(LocalCompatibilityCIError, match="not empty"):
        _retain_release_artifacts(release, destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (destination / "evidence.json").exists()


def test_retain_release_artifacts_rejects_file_without_deleting_it(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    destination = tmp_path / "existing.txt"
    destination.write_text("keep", encoding="utf-8")

    with pytest.raises(LocalCompatibilityCIError, match="new directory"):
        _retain_release_artifacts(release, destination)

    assert destination.read_text(encoding="utf-8") == "keep"


def test_retain_release_artifacts_rejects_symlink(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "output-link"
    try:
        destination.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(LocalCompatibilityCIError, match="symlink"):
        _retain_release_artifacts(release, destination)


def test_retain_release_artifacts_rejects_filesystem_root(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    filesystem_root = Path(tmp_path.anchor)

    with pytest.raises(LocalCompatibilityCIError, match="filesystem root"):
        _retain_release_artifacts(release, filesystem_root)


def test_retain_release_artifacts_cleans_partial_copy_and_restores_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "evidence.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "empty-output"
    destination.mkdir()
    original_copytree = shutil.copytree

    def fail_after_partial_copy(source: Path, target: Path, *args: object, **kwargs: object) -> Path:
        target.mkdir(parents=True)
        (target / "partial.json").write_text("partial", encoding="utf-8")
        raise OSError("injected copy failure")

    monkeypatch.setattr(shutil, "copytree", fail_after_partial_copy)

    with pytest.raises(LocalCompatibilityCIError, match="Could not retain"):
        _retain_release_artifacts(release, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not any(
        child.name.startswith(".moto-syntheticlib-retained-")
        for child in tmp_path.iterdir()
    )
    monkeypatch.setattr(shutil, "copytree", original_copytree)


def test_gitignore_allowlists_only_public_compatibility_members() -> None:
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    compatibility_prefix = (
        "!PLANS/MOTO_SYNTHETICLIB_TRADE_INTEGRATION_PLAN/compatibility/"
    )
    allowlisted = {
        line.removeprefix(compatibility_prefix)
        for line in gitignore.splitlines()
        if line.startswith(compatibility_prefix) and line != compatibility_prefix
    }

    assert allowlisted == {
        "README.md",
        "compatibility-matrix.json",
        "moto-contract-test-result.json",
        "moto-compatibility-receipt.json",
        "moto-compatibility-receipt.sha256",
        "frozen-asset-manifest.json",
        "compatibility-bundle-manifest.json",
        "moto-preflight-attestation.json",
        "MOTO_COMPATIBILITY_RECEIPT.md",
    }
    assert (
        "!PLANS/MOTO_SYNTHETICLIB_TRADE_INTEGRATION_PLAN/compatibility/*"
        not in gitignore.splitlines()
    )


def test_run_reports_child_command_failure(tmp_path: Path) -> None:
    with pytest.raises(LocalCompatibilityCIError, match="Command failed"):
        _run(
            (sys.executable, "-c", "import sys; sys.exit(7)"),
            cwd=tmp_path,
        )


def test_complete_local_runner_preserves_source_and_retains_bound_evidence(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    status_before = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    retained = tmp_path / "retained"

    summary = run_local_compatibility_ci(
        source_root=source_root,
        artifact_output=retained,
    )

    assert summary["status"] == "passed"
    assert summary["scope"] == "local_workflow_equivalent_simulation"
    assert summary["independentlyControlledRelease"] is False
    assert len(str(summary["sourceSnapshotCommit"])) == 40
    assert len(str(summary["sourceSnapshotTree"])) == 40
    assert len(str(summary["releaseBundleDigest"])) == 64
    assert len(str(summary["receiptSha256"])) == 64
    assert summary["artifactOutput"] == str(retained.resolve())
    release_bundle = json.loads(
        (retained / "release-evidence" / "moto-clean-source-attestation-bundle.json")
        .read_text(encoding="utf-8")
    )
    assert release_bundle["bundleDigest"] == summary["releaseBundleDigest"]
    assert subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == status_before
