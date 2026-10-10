"""Generate or check MOTO's credential-free SyntheticLib trade preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

from backend.shared.syntheticlib_trade_compatibility import (
    ARTIFACT_FILENAMES,
    BUILD_MANIFEST_PATH,
    COMPATIBILITY_DIR,
    PACKAGE_JSON_PATH,
    PLAN_ROOT,
    CompatibilityContractError,
    build_artifacts,
    build_release_evidence,
    load_package_identity,
    verify_evidence_set,
    verify_release_evidence,
    write_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check deterministic MOTO SyntheticLib compatibility artifacts."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in artifacts without modifying them (default).",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Explicitly regenerate the checked-in compatibility artifacts.",
    )
    mode.add_argument(
        "--write-release-evidence",
        action="store_true",
        help="Write separate clean-source release evidence from a clean Git checkout.",
    )
    mode.add_argument(
        "--check-release-evidence",
        action="store_true",
        help="Verify separate clean-source release evidence against this checkout.",
    )
    parser.add_argument("--manifest", type=Path, default=BUILD_MANIFEST_PATH)
    parser.add_argument("--package-json", type=Path, default=PACKAGE_JSON_PATH)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=COMPATIBILITY_DIR)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=BUILD_MANIFEST_PATH.parent,
        help="Git checkout used for clean-source provenance (custom roots require --allow-custom-roots).",
    )
    parser.add_argument(
        "--release-output-dir",
        type=Path,
        help="Required output directory for clean-source release evidence.",
    )
    parser.add_argument(
        "--observed-at",
        help="Explicit RFC 3339 UTC observation time for isolated deterministic tests.",
    )
    parser.add_argument(
        "--allow-custom-roots",
        action="store_true",
        help="Allow non-repository roots for isolated tests only.",
    )
    return parser


def _git_value(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompatibilityContractError(
            f"Unable to inspect release checkout with git {' '.join(args)}"
        ) from exc
    return result.stdout.strip()


def _clean_checkout_provenance(
    advertised_commit: str,
    *,
    source_root: Path = BUILD_MANIFEST_PATH.parent,
) -> tuple[str, str]:
    dirty = _git_value(source_root, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise CompatibilityContractError(
            "Clean-source release evidence requires a completely clean Git checkout"
        )
    resolved_commit = _git_value(source_root, "rev-parse", "HEAD")
    if resolved_commit != advertised_commit:
        raise CompatibilityContractError(
            "Clean checkout HEAD does not match the advertised package build_commit"
        )
    return resolved_commit, _git_value(source_root, "rev-parse", "HEAD^{tree}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_custom_roots and (
        args.plan_root.resolve() != PLAN_ROOT.resolve()
        or args.output_dir.resolve() != COMPATIBILITY_DIR.resolve()
        or args.manifest.resolve() != BUILD_MANIFEST_PATH.resolve()
        or args.package_json.resolve() != PACKAGE_JSON_PATH.resolve()
        or args.source_root.resolve() != BUILD_MANIFEST_PATH.parent.resolve()
    ):
        print("Custom compatibility roots require --allow-custom-roots.", file=sys.stderr)
        return 2
    if (args.write_release_evidence or args.check_release_evidence) and (
        args.release_output_dir is None
    ):
        print("--release-output-dir is required for release evidence.", file=sys.stderr)
        return 2
    try:
        artifacts = build_artifacts(
            manifest_path=args.manifest,
            package_json_path=args.package_json,
            plan_root=args.plan_root,
        )
    except CompatibilityContractError as exc:
        print(f"MOTO SyntheticLib compatibility preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.write_release_evidence or args.check_release_evidence:
        base_drift = verify_evidence_set(
            manifest_path=args.manifest,
            package_json_path=args.package_json,
            plan_root=args.plan_root,
            output_dir=args.output_dir,
        )
        if base_drift:
            print(
                "Checked-in compatibility evidence is missing or stale: "
                + ", ".join(base_drift),
                file=sys.stderr,
            )
            return 1
        try:
            identity = load_package_identity(
                manifest_path=args.manifest,
                package_json_path=args.package_json,
            )
            source_commit, source_tree = _clean_checkout_provenance(
                identity.build_commit,
                source_root=args.source_root,
            )
            test_result_bytes = (
                args.output_dir / ARTIFACT_FILENAMES["test_result"]
            ).read_bytes()
            if args.write_release_evidence:
                observed_at = args.observed_at or (
                    datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                release_artifacts = build_release_evidence(
                    identity,
                    test_result_bytes,
                    source_commit=source_commit,
                    source_tree=source_tree,
                    observed_at=observed_at,
                    platform_name=platform.platform(),
                    python_version=platform.python_version(),
                )
                write_artifacts(
                    release_artifacts,
                    output_dir=args.release_output_dir,
                )
            else:
                attestation_path = (
                    args.release_output_dir / "moto-clean-source-attestation.json"
                )
                attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
                observed_at = attestation["observedAt"]
                drift = verify_release_evidence(
                    source_commit=source_commit,
                    source_tree=source_tree,
                    observed_at=observed_at,
                    platform_name=platform.platform(),
                    python_version=platform.python_version(),
                    manifest_path=args.manifest,
                    package_json_path=args.package_json,
                    plan_root=args.plan_root,
                    compatibility_dir=args.output_dir,
                    output_dir=args.release_output_dir,
                )
                if drift:
                    print(
                        "Clean-source release evidence is invalid: "
                        + ", ".join(drift),
                        file=sys.stderr,
                    )
                    return 1
        except (
            CompatibilityContractError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"Clean-source release evidence failed: {exc}", file=sys.stderr)
            return 2
        action = "Wrote" if args.write_release_evidence else "Verified"
        print(f"{action} clean-source release evidence at {args.release_output_dir}")
        return 0

    if args.write:
        write_artifacts(artifacts, output_dir=args.output_dir)
        print(f"Wrote {len(artifacts)} compatibility artifacts to {args.output_dir}")
        return 0

    drift = verify_evidence_set(
        manifest_path=args.manifest,
        package_json_path=args.package_json,
        plan_root=args.plan_root,
        output_dir=args.output_dir,
    )
    if drift:
        print(
            "MOTO SyntheticLib compatibility artifacts are missing or stale: "
            + ", ".join(drift),
            file=sys.stderr,
        )
        print(
            "Regenerate intentionally with "
            "`python -m scripts.check_syntheticlib_trade_contract --write`.",
            file=sys.stderr,
        )
        return 1

    print(
        "MOTO SyntheticLib compatibility preflight passed "
        "(live proof-trade compatibility remains blocked)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
