"""
Optional Z3 / SMT client wrappers.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

from backend.shared.config import system_config
from backend.shared.models import SmtResult

logger = logging.getLogger(__name__)


class SmtClient:
    """Thin async wrapper around an external Z3 binary."""

    def __init__(self, z3_path: str, timeout: int) -> None:
        self.z3_path = str(z3_path or "").strip()
        self.timeout = max(int(timeout or 0), 1)

    def _resolve_executable(self) -> str:
        if self.z3_path:
            candidate = Path(self.z3_path).resolve()
            if candidate.exists():
                return str(candidate)

        for name in ("z3", "z3.exe"):
            resolved = shutil.which(name)
            if resolved:
                return resolved
        return self.z3_path or "z3"

    async def _run_process(
        self,
        args: list[str],
        *,
        stdin_text: str = "",
        timeout: Optional[int] = None,
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        effective_timeout = max(int(timeout or self.timeout), 1)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate((stdin_text or "").encode("utf-8")),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return (
                -1,
                "",
                f"Z3 process timed out after {effective_timeout} seconds.",
            )

        return (
            process.returncode,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def get_version(self) -> str:
        """Return the Z3 version string when available."""
        returncode, stdout, stderr = await self._run_process(
            [self._resolve_executable(), "-version"],
            timeout=15,
        )
        if returncode != 0:
            return (stderr or stdout).strip()
        return (stdout or stderr).strip()

    async def check_smt2(self, smtlib: str, timeout: Optional[int] = None) -> SmtResult:
        """Run an SMT-LIB payload through Z3 using stdin."""
        if not system_config.smt_enabled:
            return SmtResult(success=False, result="error", stderr="SMT support is disabled in system configuration.")

        payload = (smtlib or "").strip()
        if not payload:
            return SmtResult(success=False, result="error", stderr="No SMT-LIB payload was provided.")

        returncode, stdout, stderr = await self._run_process(
            [self._resolve_executable(), "-smt2", "-in"],
            stdin_text=payload + "\n",
            timeout=timeout,
        )
        first_line = next((line.strip().lower() for line in stdout.splitlines() if line.strip()), "")
        result = first_line if first_line in {"sat", "unsat", "unknown"} else "error"
        success = returncode == 0 and result in {"sat", "unsat", "unknown"}
        return SmtResult(
            success=success,
            result=result,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
        )


_smt_client: Optional[SmtClient] = None


def initialize_smt_client(z3_path: Optional[str] = None, timeout: Optional[int] = None) -> SmtClient:
    """Create or replace the singleton SMT client."""
    global _smt_client
    _smt_client = SmtClient(
        z3_path=z3_path or system_config.z3_path,
        timeout=timeout or system_config.smt_timeout,
    )
    return _smt_client


def get_smt_client() -> SmtClient:
    """Return the singleton SMT client, creating it from config if needed."""
    global _smt_client
    if _smt_client is None:
        _smt_client = SmtClient(
            z3_path=system_config.z3_path,
            timeout=system_config.smt_timeout,
        )
    return _smt_client


def clear_smt_client() -> None:
    """Reset the singleton SMT client."""
    global _smt_client
    _smt_client = None
