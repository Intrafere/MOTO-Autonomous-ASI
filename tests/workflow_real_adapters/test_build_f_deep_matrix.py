"""Environment-gated bounded repeated execution of mature real adapters."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.workflow_harness.real_adapters.deep_runner import (
    minimal_deep_environment,
    remove_variant_root,
    run_deep_variant,
    workspace_mutable_snapshot,
)
from tests.workflow_harness.real_adapters.maturity_registry import (
    AdapterMaturity,
    descriptors_for,
    validate_maturity_registry,
)


pytestmark = pytest.mark.skipif(
    os.getenv("MOTO_REAL_ADAPTER_DEEP_TESTS") != "1",
    reason="set MOTO_REAL_ADAPTER_DEEP_TESTS=1 to run the real-adapter deep matrix",
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEEP_DESCRIPTORS = descriptors_for(AdapterMaturity.DEEP)
PURPOSEFUL_VARIANTS = (
    ("cold", "17"),
    ("replay", "83"),
)
SENSITIVE_ENV_NAMES = {
    "OPENROUTER_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
}


def test_build_f_maturity_registry_preserves_deep_and_blocked_cases() -> None:
    validate_maturity_registry(workspace_root=WORKSPACE_ROOT)
    blocked = descriptors_for(AdapterMaturity.BLOCKED)
    assert len(DEEP_DESCRIPTORS) >= 6
    assert {descriptor.scenario_id for descriptor in blocked} == {
        "real_parent_action_fencing_unavailable_without_production_seam",
        "real_provider_stop_reset_checkpoint_unavailable_without_wait_seam",
        "real_leanoj_full_final_loop_not_safely_bounded",
    }


def _write_network_blocker(fake_root: Path) -> None:
    fake_root.mkdir(parents=True)
    (fake_root / "sitecustomize.py").write_text(
        "import socket\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise AssertionError('Build F deep tests forbid external network access')\n"
        "socket.create_connection = _blocked\n"
        "_original_connect = socket.socket.connect\n"
        "def _guarded_connect(self, address):\n"
        "    host = address[0] if isinstance(address, tuple) else str(address)\n"
        "    if host not in {'127.0.0.1', '::1', 'localhost'}:\n"
        "        raise AssertionError('Build F deep tests forbid external network access')\n"
        "    return _original_connect(self, address)\n"
        "socket.socket.connect = _guarded_connect\n",
        encoding="utf-8",
    )


def test_build_f_deep_environment_is_minimal_and_clears_sensitive_routes(
    tmp_path: Path,
) -> None:
    base = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        **{name: "must-not-pass-through" for name in SENSITIVE_ENV_NAMES},
    }
    env = minimal_deep_environment(
        base,
        workspace_root=WORKSPACE_ROOT,
        variant_root=tmp_path / "variant",
        fake_root=tmp_path / "fake",
        hash_seed="101",
    )

    assert SENSITIVE_ENV_NAMES.isdisjoint(env)
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    assert env["PYTHONHASHSEED"] == "101"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"


@pytest.mark.parametrize(
    "descriptor",
    DEEP_DESCRIPTORS,
    ids=lambda descriptor: descriptor.scenario_id,
)
def test_build_f_deep_real_adapter_matrix_is_bounded_repeatable_and_contained(
    tmp_path: Path,
    descriptor,
) -> None:
    before_workspace = workspace_mutable_snapshot(WORKSPACE_ROOT)
    observations = []
    variant_roots = []
    try:
        for variant, hash_seed in PURPOSEFUL_VARIANTS:
            variant_root = tmp_path / descriptor.scenario_id / variant
            fake_root = variant_root / "external-blocker"
            for root in (
                variant_root / "data",
                variant_root / "logs",
                variant_root / "tmp",
                variant_root / "home",
            ):
                root.mkdir(parents=True)
            _write_network_blocker(fake_root)
            variant_roots.append(variant_root)
            observations.append(
                run_deep_variant(
                    descriptor,
                    workspace_root=WORKSPACE_ROOT,
                    variant_root=variant_root,
                    fake_root=fake_root,
                    variant=variant,
                    hash_seed=hash_seed,
                )
            )

        assert observations[0].hash_seed != observations[1].hash_seed
        for observation in observations:
            assert observation.returncode == 0, (
                f"{descriptor.scenario_id} ({observation.variant}) failed\n"
                f"STDOUT:\n{observation.stdout}\nSTDERR:\n{observation.stderr}"
            )
            assert not observation.workspace_changes, (
                f"{descriptor.scenario_id} ({observation.variant}) changed repository runtime state:\n"
                + "\n".join(observation.workspace_changes)
            )
        assert descriptor.repeat_contract is not None
        assert observations[0].repeat_value(
            descriptor.repeat_contract
        ) == observations[1].repeat_value(descriptor.repeat_contract)
        assert workspace_mutable_snapshot(WORKSPACE_ROOT) == before_workspace
    finally:
        for variant_root in variant_roots:
            remove_variant_root(variant_root)
        assert all(not root.exists() for root in variant_roots)
