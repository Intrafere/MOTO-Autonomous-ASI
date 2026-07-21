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

    data_paths = (
        system_config.user_uploads_dir,
        system_config.chroma_db_dir,
        system_config.shared_training_file,
        system_config.compiler_outline_file,
        system_config.compiler_paper_file,
        system_config.compiler_rejections_file,
        system_config.compiler_acceptances_file,
        system_config.compiler_declines_file,
        system_config.auto_brainstorms_dir,
        system_config.auto_papers_dir,
        system_config.auto_papers_archive_dir,
        system_config.auto_research_metadata_file,
        system_config.auto_research_stats_file,
        system_config.auto_workflow_state_file,
        system_config.auto_research_topic_rejections_file,
        system_config.auto_sessions_base_dir,
        system_config.lean4_workspace_dir,
    )
    escaped = [str(path) for path in data_paths if path and not _within(path, expected_data)]
    assert not escaped, f"Deep child has mutable paths outside MOTO_DATA_ROOT: {escaped}"
