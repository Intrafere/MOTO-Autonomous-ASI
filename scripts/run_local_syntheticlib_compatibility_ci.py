"""Run the SyntheticLib compatibility release workflow in an isolated local snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    ".github/workflows/syntheticlib-compatibility-release.yml",
    "package.json",
    "moto-update-manifest.json",
    "backend/__init__.py",
    "backend/shared/__init__.py",
    "backend/shared/build_info.py",
    "backend/shared/syntheticlib_trade_compatibility.py",
    "scripts/__init__.py",
    "scripts/check_syntheticlib_trade_contract.py",
    "scripts/run_local_syntheticlib_compatibility_ci.py",
)
RELEASE_ATTESTATION = "moto-clean-source-attestation.json"
RELEASE_BUNDLE = "moto-clean-source-attestation-bundle.json"
LOCAL_OBSERVED_AT = "2000-01-01T00:00:00Z"


class LocalCompatibilityCIError(RuntimeError):
    """Raised when the isolated compatibility release simulation fails."""


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LocalCompatibilityCIError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def _copy_snapshot(source_root: Path, snapshot_root: Path) -> None:
    for relative in SOURCE_FILES:
        source = source_root / relative
        if not source.is_file():
            raise LocalCompatibilityCIError(f"Required source file is missing: {relative}")
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _initialize_snapshot(snapshot_root: Path) -> tuple[str, str]:
    _run(("git", "init", "--quiet"), cwd=snapshot_root)
    _run(("git", "config", "user.name", "MOTO Local Compatibility CI"), cwd=snapshot_root)
    _run(
        ("git", "config", "user.email", "local-compatibility-ci@invalid"),
        cwd=snapshot_root,
    )
    _run(("git", "add", "--all"), cwd=snapshot_root)
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": LOCAL_OBSERVED_AT,
        "GIT_COMMITTER_DATE": LOCAL_OBSERVED_AT,
    }
    _run(
        ("git", "commit", "--quiet", "-m", "local compatibility snapshot"),
        cwd=snapshot_root,
        env=commit_env,
    )
    commit = _run(("git", "rev-parse", "HEAD"), cwd=snapshot_root).stdout.strip()
    tree = _run(("git", "rev-parse", "HEAD^{tree}"), cwd=snapshot_root).stdout.strip()
    if _run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=snapshot_root,
    ).stdout.strip():
        raise LocalCompatibilityCIError("Temporary source snapshot is not clean")
    return commit, tree


def _write_release_inputs(
    source_root: Path,
    release_root: Path,
    snapshot_commit: str,
) -> tuple[Path, Path, Path, Path, Path]:
    release_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (source_root / "moto-update-manifest.json").read_text(encoding="utf-8")
    )
    manifest["build_commit"] = snapshot_commit
    manifest_path = release_root / "moto-update-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    package_path = release_root / "package.json"
    shutil.copy2(source_root / "package.json", package_path)
    plan_root = release_root / "plan"
    compatibility_dir = plan_root / "compatibility"
    release_evidence_dir = release_root / "release-evidence"
    compatibility_dir.mkdir(parents=True)
    return (
        manifest_path,
        package_path,
        plan_root,
        compatibility_dir,
        release_evidence_dir,
    )


def _retain_release_artifacts(release_root: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise LocalCompatibilityCIError("Artifact output must not be a symlink")
    if destination.anchor and destination == Path(destination.anchor):
        raise LocalCompatibilityCIError("Artifact output must not be a filesystem root")
    destination_was_empty = False
    if destination.exists():
        if not destination.is_dir():
            raise LocalCompatibilityCIError(
                "Artifact output must be a new directory or an existing empty directory"
            )
        try:
            if any(destination.iterdir()):
                raise LocalCompatibilityCIError(
                    "Artifact output directory already exists and is not empty"
                )
        except OSError as exc:
            raise LocalCompatibilityCIError(
                "Artifact output directory could not be inspected"
            ) from exc
        destination_was_empty = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".moto-syntheticlib-retained-",
        dir=destination.parent,
    ) as temp:
        staged = Path(temp) / "complete"
        try:
            shutil.copytree(release_root, staged)
            if destination_was_empty:
                destination.rmdir()
            staged.replace(destination)
        except BaseException as exc:
            if destination_was_empty and not destination.exists():
                try:
                    destination.mkdir()
                except OSError as restore_exc:
                    raise LocalCompatibilityCIError(
                        "Could not retain release artifacts or restore the empty destination"
                    ) from restore_exc
            if not isinstance(exc, OSError):
                raise
            raise LocalCompatibilityCIError("Could not retain release artifacts") from exc


def run_local_compatibility_ci(
    *,
    source_root: Path = REPO_ROOT,
    artifact_output: Path | None = None,
) -> dict[str, object]:
    """Execute release generation and verification without mutating the source tree."""
    source_root = source_root.resolve()
    source_status_before = _run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=source_root,
    ).stdout
    _run(
        (sys.executable, "-m", "scripts.check_syntheticlib_trade_contract", "--check"),
        cwd=source_root,
    )

    with tempfile.TemporaryDirectory(prefix="moto-syntheticlib-local-ci-") as temp:
        temp_root = Path(temp)
        snapshot_root = temp_root / "source"
        release_root = temp_root / "release"
        snapshot_root.mkdir()
        _copy_snapshot(source_root, snapshot_root)
        snapshot_commit, snapshot_tree = _initialize_snapshot(snapshot_root)
        (
            manifest_path,
            package_path,
            plan_root,
            compatibility_dir,
            release_evidence_dir,
        ) = _write_release_inputs(source_root, release_root, snapshot_commit)

        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(snapshot_root),
        }
        base = (
            sys.executable,
            "-m",
            "scripts.check_syntheticlib_trade_contract",
            "--allow-custom-roots",
            "--manifest",
            str(manifest_path),
            "--package-json",
            str(package_path),
            "--plan-root",
            str(plan_root),
            "--output-dir",
            str(compatibility_dir),
        )
        _run((*base, "--write"), cwd=snapshot_root, env=env)
        _run((*base, "--check"), cwd=snapshot_root, env=env)
        release_args = (
            *base,
            "--release-output-dir",
            str(release_evidence_dir),
        )
        _run(
            (
                *release_args,
                "--write-release-evidence",
                "--observed-at",
                LOCAL_OBSERVED_AT,
            ),
            cwd=snapshot_root,
            env=env,
        )
        _run((*release_args, "--check-release-evidence"), cwd=snapshot_root, env=env)

        if _run(
            ("git", "status", "--porcelain", "--untracked-files=normal"),
            cwd=snapshot_root,
        ).stdout.strip():
            raise LocalCompatibilityCIError(
                "Release simulation mutated the temporary source snapshot"
            )
        attestation = json.loads(
            (release_evidence_dir / RELEASE_ATTESTATION).read_text(encoding="utf-8")
        )
        bundle = json.loads(
            (release_evidence_dir / RELEASE_BUNDLE).read_text(encoding="utf-8")
        )
        if attestation.get("attestationMode") != "clean_source":
            raise LocalCompatibilityCIError("Release attestation is not clean_source")
        if attestation.get("executor", {}).get("sourceCommit") != snapshot_commit:
            raise LocalCompatibilityCIError(
                "Release attestation does not bind the temporary snapshot commit"
            )

        if artifact_output is not None:
            if artifact_output.is_symlink():
                raise LocalCompatibilityCIError("Artifact output must not be a symlink")
            destination = artifact_output.resolve()
            if destination == source_root or source_root in destination.parents:
                raise LocalCompatibilityCIError(
                    "Artifact output must be outside the source repository"
                )
            _retain_release_artifacts(release_root, destination)

        summary: dict[str, object] = {
            "status": "passed",
            "scope": "local_workflow_equivalent_simulation",
            "independentlyControlledRelease": False,
            "sourceSnapshotCommit": snapshot_commit,
            "sourceSnapshotTree": snapshot_tree,
            "releaseBundleDigest": bundle["bundleDigest"],
            "receiptSha256": hashlib.sha256(
                (compatibility_dir / "moto-compatibility-receipt.json").read_bytes()
            ).hexdigest(),
            "artifactOutput": str(artifact_output.resolve())
            if artifact_output is not None
            else None,
        }

    source_status_after = _run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=source_root,
    ).stdout
    if source_status_after != source_status_before:
        raise LocalCompatibilityCIError(
            "Local compatibility CI changed the real repository status"
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SyntheticLib compatibility release checks in a temporary Git snapshot."
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="Optional directory outside the repository in which to retain release evidence.",
    )
    args = parser.parse_args(argv)
    try:
        summary = run_local_compatibility_ci(artifact_output=args.artifact_output)
    except LocalCompatibilityCIError as exc:
        print(f"Local SyntheticLib compatibility CI failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
