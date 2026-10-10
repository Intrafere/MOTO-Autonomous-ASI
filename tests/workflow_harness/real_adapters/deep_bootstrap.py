"""Early pytest bootstrap assertions for contained deep real-adapter children."""
from __future__ import annotations

import os
from pathlib import Path


def _within(path: str | Path, root: Path) -> bool:
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def pytest_sessionstart(session) -> None:  # pragma: no cover - executed in child pytest
    del session
    from backend.shared.config import system_config

    expected_data = Path(os.environ["MOTO_DATA_ROOT"]).resolve()
    expected_logs = Path(os.environ["MOTO_LOG_ROOT"]).resolve()
    assert Path(system_config.data_dir).resolve() == expected_data
    assert Path(system_config.logs_dir or "").resolve() == expected_logs

    data_paths = tuple(
        getattr(system_config, name)
        for name in type(system_config).model_fields
        if name.endswith(("_dir", "_file", "_path", "_root"))
        and getattr(system_config, name, None)
        and name not in {"logs_dir"}
    )
    escaped = [str(path) for path in data_paths if path and not _within(path, expected_data)]
    assert not escaped, f"Deep child has mutable paths outside MOTO_DATA_ROOT: {escaped}"
    assert os.environ.get("MOTO_DEEP_FAKE_BOUNDARIES"), (
        "Deep child must disclose its fake isolation boundaries."
    )
