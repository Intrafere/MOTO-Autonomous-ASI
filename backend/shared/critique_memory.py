"""
Paper Critique Memory Module.

Handles persistence of paper critiques from the validator model.
Supports three paper types: autonomous_paper, final_answer, compiler_paper.
Each paper can have up to 10 critiques stored (oldest removed when exceeded).

DUAL-PATH ARCHITECTURE
======================
The autonomous research system uses two storage modes:

1. LEGACY PATHS (backward compatibility):
   - Papers: backend/data/auto_papers/
   - Final answers: backend/data/auto_final_answer/
   - Used when existing legacy data is detected

2. SESSION-BASED PATHS (preferred for new sessions):
   - Papers: backend/data/auto_sessions/{session_id}/papers/
   - Final answers: backend/data/auto_sessions/{session_id}/final_answer/
   - Created for new research sessions

HOW TO USE base_dir PARAMETER:
- For session-based papers: Pass the trusted papers directory
- For final answers: Pass the trusted final answer directory
- If base_dir is None, falls back to legacy paths (for backward compatibility)

The compiler paper type always uses a single global file and ignores base_dir.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Literal
from datetime import datetime
import uuid

from backend.shared.config import system_config
from backend.shared.models import PaperCritique
from backend.shared.path_safety import (
    resolve_path_within_root,
    validate_single_path_component,
)

logger = logging.getLogger(__name__)

# Maximum number of critiques to store per paper
MAX_CRITIQUES_PER_PAPER = 10

# Paper type definitions
PaperType = Literal["autonomous_paper", "final_answer", "compiler_paper"]


def _get_legacy_data_dir() -> Path:
    """Return the shared legacy data directory for critique storage."""
    return Path(system_config.data_dir)


def _get_legacy_critiques_dir(paper_type: PaperType) -> Path:
    """Return the trusted legacy directory for a critique storage type."""
    data_dir = _get_legacy_data_dir()

    if paper_type == "autonomous_paper":
        return resolve_path_within_root(data_dir, "auto_papers")

    if paper_type == "final_answer":
        return resolve_path_within_root(data_dir, "auto_final_answer")

    if paper_type == "compiler_paper":
        return data_dir

    raise ValueError(f"Unknown paper_type: {paper_type}")


def _resolve_session_critiques_dir(base_dir: Path, paper_type: PaperType) -> Path:
    """
    Rebuild a session-aware critique directory from validated components.

    This prevents callers from passing arbitrary absolute paths into critique
    file operations. Only `<session_id>/papers` and `<session_id>/final_answer`
    directories under the trusted sessions root are allowed.
    """
    sessions_root = Path(system_config.auto_sessions_base_dir)
    candidate_dir = Path(base_dir)
    expected_leaf = "papers" if paper_type == "autonomous_paper" else "final_answer"

    try:
        relative_dir = candidate_dir.resolve(strict=False).relative_to(
            sessions_root.resolve(strict=False)
        )
    except ValueError as exc:
        raise ValueError(f"Untrusted critique storage directory: {base_dir}") from exc

    if len(relative_dir.parts) != 2 or relative_dir.parts[1] != expected_leaf:
        raise ValueError(f"Untrusted critique storage directory: {base_dir}")

    safe_session_id = validate_single_path_component(relative_dir.parts[0], "session ID")
    return resolve_path_within_root(sessions_root, safe_session_id, expected_leaf)


def _resolve_trusted_critiques_dir(
    paper_type: PaperType,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve critique storage to a trusted legacy or session-scoped directory.
    """
    if paper_type == "compiler_paper":
        return _get_legacy_critiques_dir(paper_type)

    legacy_dir = _get_legacy_critiques_dir(paper_type)
    if base_dir is None:
        return legacy_dir

    candidate_dir = Path(base_dir)
    if candidate_dir.resolve(strict=False) == legacy_dir.resolve(strict=False):
        return legacy_dir

    return _resolve_session_critiques_dir(candidate_dir, paper_type)


def _get_critiques_file_path(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Path:
    """
    Get the file path for storing critiques based on paper type.

    Args:
        paper_type: Type of paper ("autonomous_paper", "final_answer", "compiler_paper")
        paper_id: Required for autonomous_paper type (used in filename)
        base_dir: Optional trusted directory for session-aware storage.

    Returns:
        Path to the critiques JSON file
    """
    safe_paper_id = None
    if paper_id:
        safe_paper_id = validate_single_path_component(paper_id, "paper ID")

    if paper_type == "autonomous_paper":
        if not safe_paper_id:
            raise ValueError("paper_id is required for autonomous_paper type")
        critiques_dir = _resolve_trusted_critiques_dir(paper_type, base_dir)
        critiques_dir.mkdir(parents=True, exist_ok=True)
        return resolve_path_within_root(
            critiques_dir,
            f"paper_{safe_paper_id}_critiques.json",
        )

    if paper_type == "final_answer":
        critiques_dir = _resolve_trusted_critiques_dir(paper_type, base_dir)
        critiques_dir.mkdir(parents=True, exist_ok=True)
        return resolve_path_within_root(critiques_dir, "final_answer_critiques.json")

    if paper_type == "compiler_paper":
        critiques_dir = _resolve_trusted_critiques_dir(paper_type, base_dir)
        critiques_dir.mkdir(parents=True, exist_ok=True)
        return resolve_path_within_root(critiques_dir, "compiler_paper_critiques.json")

    raise ValueError(f"Unknown paper_type: {paper_type}")


async def save_critique(
    paper_type: PaperType,
    critique: PaperCritique,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> PaperCritique:
    """
    Save a critique to the paper's critique history.

    Maintains a maximum of MAX_CRITIQUES_PER_PAPER critiques per paper.
    Oldest critiques are removed when the limit is exceeded.
    """
    file_path = _get_critiques_file_path(paper_type, paper_id, base_dir)

    if not critique.critique_id:
        critique.critique_id = str(uuid.uuid4())[:8]

    critiques = await get_critiques(paper_type, paper_id, base_dir)
    critiques.insert(0, critique)

    while len(critiques) > MAX_CRITIQUES_PER_PAPER:
        removed = critiques.pop()
        logger.info(
            f"Removed oldest critique {removed.critique_id} "
            f"to maintain limit of {MAX_CRITIQUES_PER_PAPER}"
        )

    critiques_data = [c.model_dump() for c in critiques]
    for critique_dict in critiques_data:
        if isinstance(critique_dict.get("date"), datetime):
            critique_dict["date"] = critique_dict["date"].isoformat()

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(critiques_data, f, indent=2, default=str)
        logger.info(
            f"Saved critique {critique.critique_id} for {paper_type}"
            + (f" paper_id={paper_id}" if paper_id else "")
        )
    except Exception as e:
        logger.error(f"Failed to save critique for {paper_type}: {e}")
        raise

    return critique


async def get_critiques(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> List[PaperCritique]:
    """Get all critiques for a paper."""
    file_path = _get_critiques_file_path(paper_type, paper_id, base_dir)
    if not file_path.exists():
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            critiques_data = json.load(f)

        critiques = []
        for critique_dict in critiques_data:
            if isinstance(critique_dict.get("date"), str):
                try:
                    critique_dict["date"] = datetime.fromisoformat(critique_dict["date"])
                except ValueError:
                    critique_dict["date"] = datetime.now()
            critiques.append(PaperCritique(**critique_dict))

        return critiques
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse critiques for {paper_type}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load critiques for {paper_type}: {e}")
        return []


async def clear_critiques(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> bool:
    """Delete all critiques for a paper."""
    file_path = _get_critiques_file_path(paper_type, paper_id, base_dir)
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(
                f"Cleared critiques for {paper_type}"
                + (f" paper_id={paper_id}" if paper_id else "")
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete critiques for {paper_type}: {e}")
            raise

    return False


async def get_critique_by_id(
    paper_type: PaperType,
    critique_id: str,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Optional[PaperCritique]:
    """Get a specific critique by its ID."""
    critiques = await get_critiques(paper_type, paper_id, base_dir)

    for critique in critiques:
        if critique.critique_id == critique_id:
            return critique

    return None


async def get_latest_critique(
    paper_type: PaperType,
    paper_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Optional[PaperCritique]:
    """Get the most recent critique for a paper."""
    critiques = await get_critiques(paper_type, paper_id, base_dir)
    if critiques:
        return critiques[0]

    return None
