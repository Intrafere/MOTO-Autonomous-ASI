"""Durable prompt storage for manual Compiler mode."""

import asyncio
import logging
from pathlib import Path

import aiofiles

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

MANUAL_COMPILER_PROMPT_FILE = "manual_compiler_prompt.txt"


def get_manual_compiler_prompt_path() -> Path:
    return Path(system_config.data_dir) / MANUAL_COMPILER_PROMPT_FILE


async def save_manual_compiler_prompt(prompt: str) -> None:
    """Persist the latest manual Compiler prompt until explicit clear."""
    if not (prompt or "").strip():
        logger.warning("Refusing to overwrite manual Compiler prompt with an empty value")
        return
    path = get_manual_compiler_prompt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    async with aiofiles.open(temp_path, "w", encoding="utf-8") as handle:
        await handle.write(prompt or "")
    await asyncio.to_thread(temp_path.replace, path)


async def load_manual_compiler_prompt() -> str:
    """Load the latest manual Compiler prompt, if one has been persisted."""
    path = get_manual_compiler_prompt_path()
    if not path.exists():
        return ""
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
            return await handle.read()
    except Exception as exc:
        logger.debug("Unable to load manual Compiler prompt: %s", exc)
        return ""


async def clear_manual_compiler_prompt() -> None:
    """Clear manual Compiler prompt state after an explicit reset."""
    path = get_manual_compiler_prompt_path()
    if path.exists():
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.debug("Unable to clear manual Compiler prompt: %s", exc)
