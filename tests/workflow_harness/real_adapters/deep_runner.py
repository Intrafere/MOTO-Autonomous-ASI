"""Contained subprocess runner used by the opt-in Build F deep matrix."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from typing import Iterable

from .maturity_registry import RealAdapterDescriptor, RepeatContract


_PASSTHROUGH_ENV = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)
_WORKSPACE_MUTABLE_ROOTS = (
    "backend/data",
    "backend/logs",
    ".moto_instances",
    ".moto_launcher_state.json",
    ".moto_last_instance.json",
)
_IGNORED_NAMES = {"__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class DeepRunObservation:
    variant: str
    hash_seed: str
    returncode: int
    normalized_output: str
    runtime_state_digest: str
    stdout: str
    stderr: str
    workspace_changes: tuple[str, ...] = ()

    def repeat_value(self, contract: RepeatContract) -> tuple[object, ...]:
        base = (self.returncode, self.normalized_output)
        if contract is RepeatContract.NORMALIZED_PROCESS_AND_RUNTIME_STATE:
            return (*base, self.runtime_state_digest)
        return base


def minimal_deep_environment(
    base_environment: dict[str, str],
    *,
    workspace_root: Path,
    variant_root: Path,
    fake_root: Path,
    hash_seed: str,
) -> dict[str, str]:
    """Build an allowlisted environment with provider/proxy/keyring access disabled."""
    env = {
        key: base_environment[key]
        for key in _PASSTHROUGH_ENV
        if base_environment.get(key)
    }
    home = variant_root / "home"
    temp_root = variant_root / "tmp"
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "MOTO_REAL_ADAPTER_DEEP_TESTS": "0",
            "MOTO_WORKFLOW_DEEP_TESTS": "0",
            "MOTO_DATA_ROOT": str(variant_root / "data"),
            "MOTO_LOG_ROOT": str(variant_root / "logs"),
            "MOTO_GENERIC_MODE": "false",
            "MOTO_LM_STUDIO_BASE_URL": "http://127.0.0.1:9",
            "MOTO_LEAN4_ENABLED": "false",
            "MOTO_SMT_ENABLED": "false",
            "MOTO_SECRET_NAMESPACE": f"build-f-{hash_seed}",
            "PYTHONHASHSEED": hash_seed,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "PLAYWRIGHT_BROWSERS_PATH": str(variant_root / "no-browser-runtime"),
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
            "PYTHONPATH": os.pathsep.join((str(fake_root), str(workspace_root))),
        }
    )
    return env


def snapshot_paths(paths: Iterable[Path], *, max_depth: int | None = None) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            snapshot[str(root)] = "<absent>"
            continue
        candidates = (root,) if root.is_file() else root.rglob("*")
        for path in candidates:
            if max_depth is not None:
                try:
                    depth = len(path.relative_to(root).parts)
                except ValueError:
                    continue
                if depth > max_depth:
                    continue
            if any(part in _IGNORED_NAMES for part in path.parts) or not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            snapshot[str(path)] = digest.hexdigest()
    return snapshot


def workspace_mutable_snapshot(workspace_root: Path) -> dict[str, str]:
    # Runtime roots can contain very large historical corpora. Two levels catches accidental
    # fallback writes and root-state mutations without turning every deep case into a full data
    # integrity scan; individual scenario adapters still assert their concrete artifacts' roots.
    return snapshot_paths(
        (workspace_root / relative for relative in _WORKSPACE_MUTABLE_ROOTS),
        max_depth=2,
    )


def runtime_state_digest(variant_root: Path) -> str:
    state = snapshot_paths((variant_root / "data", variant_root / "logs"))
    normalized = "\n".join(
        f"{Path(path).relative_to(variant_root)}={digest}"
        for path, digest in sorted(state.items())
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def snapshot_diff(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    """Return a stable, content-based description of mutable workspace changes."""
    changes: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        if old is None:
            kind = "added"
        elif new is None:
            kind = "deleted"
        else:
            kind = "modified"
        changes.append(f"{kind}: {path}")
    return tuple(changes)


def normalize_pytest_output(text: str, roots: Iterable[Path]) -> str:
    normalized = text.replace("\\", "/")
    for root in roots:
        normalized = normalized.replace(str(root).replace("\\", "/"), "<ROOT>")
    normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<TIME>", normalized)
    normalized = re.sub(r"\bin \d+(?:\.\d+)? seconds?\b", "in <TIME>", normalized)
    return "\n".join(line.rstrip() for line in normalized.splitlines() if line.strip())


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=15,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_deep_variant(
    descriptor: RealAdapterDescriptor,
    *,
    workspace_root: Path,
    variant_root: Path,
    fake_root: Path,
    variant: str,
    hash_seed: str,
) -> DeepRunObservation:
    workspace_before = workspace_mutable_snapshot(workspace_root)
    env = minimal_deep_environment(
        os.environ,
        workspace_root=workspace_root,
        variant_root=variant_root,
        fake_root=fake_root,
        hash_seed=hash_seed,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *descriptor.pytest_args(),
        "-p",
        "no:cacheprovider",
        "-p",
        "tests.workflow_harness.real_adapters.deep_bootstrap",
        "--basetemp",
        str(variant_root / "pytest-temp"),
    ]
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=workspace_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=descriptor.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        stdout, stderr = process.communicate()
        raise AssertionError(
            f"{descriptor.scenario_id} ({variant}) exceeded "
            f"{descriptor.timeout_seconds}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ) from exc
    combined = f"{stdout}\n{stderr}"
    workspace_after = workspace_mutable_snapshot(workspace_root)
    return DeepRunObservation(
        variant=variant,
        hash_seed=hash_seed,
        returncode=process.returncode,
        normalized_output=normalize_pytest_output(
            combined, (workspace_root, variant_root)
        ),
        runtime_state_digest=runtime_state_digest(variant_root),
        stdout=stdout,
        stderr=stderr,
        workspace_changes=snapshot_diff(workspace_before, workspace_after),
    )


def remove_variant_root(variant_root: Path) -> None:
    if variant_root.exists():
        shutil.rmtree(variant_root)
