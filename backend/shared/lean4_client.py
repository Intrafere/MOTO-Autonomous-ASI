"""
Lean 4 clients for formal proof verification.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.shared.config import system_config

logger = logging.getLogger(__name__)


# Regexes used to reject vacuous proofs. A `sorry` or `admit` anywhere in the
# theorem body lets Lean 4 compile the file successfully (only emitting a
# warning), which historically made the verifier report `success=True` for
# proofs that were not proofs at all. These patterns match the tokens only
# when they are standalone identifiers so that legitimate substrings (like
# `sorryFree` or `admittedly`) are not flagged.
_FORBIDDEN_PROOF_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sorry", re.compile(r"(?<![A-Za-z0-9_'])sorry(?![A-Za-z0-9_'])")),
    ("admit", re.compile(r"(?<![A-Za-z0-9_'])admit(?![A-Za-z0-9_'])")),
)

# Lean 4 emits this warning whenever a declaration's body still contains
# `sorry`. We pick it up in the diagnostics output because it is the
# authoritative signal from Lean that the proof is incomplete.
_SORRY_WARNING_MARKERS: tuple[str, ...] = (
    "declaration uses 'sorry'",
    "uses 'sorry'",
    "contains 'sorry'",
    # Lean 4 also emits a warning about `sorryAx` when the axiom leaks into
    # the term elaborator. Treat it the same as a sorry warning.
    "uses sorry",
    "contains sorry",
)


@dataclass
class Lean4Result:
    """Result of one Lean 4 proof check."""
    success: bool
    error_output: str = ""
    goal_states: str = ""
    raw_stderr: str = ""
    tactic_error_slice: str = ""
    failing_tactic_index: int = -1


def _strip_lean_comments_and_strings(code: str) -> str:
    """Best-effort removal of Lean 4 comments and string literals.

    The forbidden-token scan runs on the stripped text so that a legitimate
    string literal containing the word ``sorry`` (for example in a
    documentation block) does not cause a false rejection. This is a
    deliberate approximation: we accept an occasional over-strip over
    misclassifying a real ``sorry`` in code as harmless.
    """
    if not code:
        return ""

    without_block_comments = re.sub(r"/-.*?-/", " ", code, flags=re.DOTALL)
    without_line_comments = re.sub(r"--[^\n]*", " ", without_block_comments)
    without_strings = re.sub(r'"(?:\\.|[^"\\])*"', ' "" ', without_line_comments)
    return without_strings


def _detect_forbidden_placeholder(code: str) -> Optional[str]:
    """Return the forbidden token name if ``code`` contains a placeholder proof.

    Checks for bare ``sorry`` / ``admit`` tokens after stripping comments and
    string literals. Returns ``None`` when the code is free of known
    placeholder markers.
    """
    stripped = _strip_lean_comments_and_strings(code)
    for token_name, pattern in _FORBIDDEN_PROOF_TOKEN_PATTERNS:
        if pattern.search(stripped):
            return token_name
    return None


def _output_contains_sorry_warning(output: str) -> bool:
    lowered = (output or "").lower()
    return any(marker in lowered for marker in _SORRY_WARNING_MARKERS)


_PLACEHOLDER_REJECTION_PREFIX = "PROOF REJECTED: PLACEHOLDER USED"
_MATHLIB_CACHE_ARCHIVE_RE = re.compile(r"\(([^()\r\n]+?\.ltar)\)")
_LEAN_WORKSPACE_ERROR_PREFIX = "LEAN 4 WORKSPACE ERROR"
_OLEAN_OBJECT_FILE_MISSING_RE = re.compile(
    r"object file ['\"].*?\.olean['\"] of module .*? does not exist",
    re.IGNORECASE,
)
_LEAN_WORKSPACE_ERROR_MARKERS: tuple[str, ...] = (
    "imports are out of date",
    "invalid or corrupt .olean",
    "invalid or corrupt olean",
)

# Markdown fence markers the LLM occasionally emits inside the `lean_code`
# JSON field even when instructed to return raw code. Strip them defensively so
# Lean 4 does not fail to parse the generated file on a stray ```lean line.
_LEAN_FENCE_OPEN_RE = re.compile(r"^\s*```(?:lean4?|lean)?\s*$", re.IGNORECASE)
_LEAN_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")

# Recognizes Lean's "unsolved goals" / "no goals to be solved" diagnostic text.
# We treat "no goals to be solved" specially because it almost always means the
# model appended one tactic too many after the proof was already closed.
_NO_GOALS_DIAGNOSTIC_RE = re.compile(
    r"(no goals to be solved|no goals|goals accomplished already)",
    re.IGNORECASE,
)
_NO_GOALS_HINT = (
    "HINT: Lean reported 'no goals to be solved'. This almost always means the "
    "previous tactic already closed the proof and the tactic at the reported "
    "line/column is extraneous. Remove that tactic (or any tactics after the "
    "goal-closing step) and resubmit. Do NOT add more tactics to try to fix it."
)


def _strip_markdown_fences(code: str) -> str:
    """Remove stray ``` / ```lean fences from LLM output."""
    if not code or "```" not in code:
        return code

    lines = code.splitlines()
    cleaned: list[str] = []
    in_fence = False
    for line in lines:
        if _LEAN_FENCE_OPEN_RE.match(line):
            in_fence = not in_fence
            continue
        if _LEAN_FENCE_CLOSE_RE.match(line):
            in_fence = not in_fence
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _deduplicate_leading_import(code: str) -> str:
    """Collapse repeated ``import Mathlib`` prefixes the model sometimes emits."""
    if not code:
        return code
    lines = code.splitlines()
    kept: list[str] = []
    seen_imports: set[str] = set()
    past_imports = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import "):
            if past_imports:
                kept.append(line)
                continue
            if stripped in seen_imports:
                continue
            seen_imports.add(stripped)
            kept.append(line)
        else:
            if stripped:
                past_imports = True
            kept.append(line)
    return "\n".join(kept)


def _format_placeholder_rejection(token_name: str, *, from_lean_diagnostic: bool) -> str:
    """Produce a structured, model-readable error for placeholder proofs."""
    if from_lean_diagnostic:
        reason = (
            "Lean 4 reported 'declaration uses sorry': the theorem body contains "
            "`sorry` or another placeholder. Lean only emits a warning for this, "
            "but MOTO rejects it because a proof with `sorry` is not a proof."
        )
    else:
        reason = (
            f"The submitted Lean 4 code contains a placeholder token `{token_name}`. "
            "MOTO rejects proofs that use `sorry` or `admit` anywhere in the "
            "theorem body because they do not actually prove the goal."
        )
    return (
        f"{_PLACEHOLDER_REJECTION_PREFIX}\n"
        f"{reason}\n"
        "Required fix: produce a Lean 4 proof that closes every goal without "
        "using `sorry`, `admit`, unresolved `axiom` stubs introduced solely to "
        "trivialize the target theorem, or any other placeholder. If the result "
        "cannot be proved yet, return a narrower lemma that you can fully "
        "prove instead."
    )


class Lean4Client:
    """Subprocess wrapper around the Lean 4 toolchain."""

    _lean_execution_lock: Optional[asyncio.Lock] = None

    def __init__(self, lean_path: str, workspace_dir: str) -> None:
        self.lean_path = str(lean_path or "").strip()
        self.workspace_dir = Path(workspace_dir).resolve()
        self._workspace_ready = False
        self._workspace_unhealthy_error = ""
        self._workspace_lock = asyncio.Lock()

    @classmethod
    def _get_lean_execution_lock(cls) -> asyncio.Lock:
        if Lean4Client._lean_execution_lock is None:
            Lean4Client._lean_execution_lock = asyncio.Lock()
        return Lean4Client._lean_execution_lock

    def _resolve_executable(self, name: str) -> str:
        if self.lean_path:
            lean_bin = Path(self.lean_path).resolve()
            sibling = lean_bin.parent / (f"{name}.exe" if lean_bin.suffix.lower() == ".exe" else name)
            if sibling.exists():
                return str(sibling)

        resolved = shutil.which(name)
        if resolved:
            return resolved
        return name

    @property
    def lake_path(self) -> str:
        return self._resolve_executable("lake")

    def uses_persistent_server(self) -> bool:
        """Return True when the client keeps a long-lived Lean process."""
        return False

    def is_server_active(self) -> bool:
        """Return True when the persistent server is currently alive."""
        return False

    async def warm_start(self) -> None:
        """Perform optional startup work during FastAPI lifespan."""
        return

    async def close(self) -> None:
        """Release client resources during backend shutdown."""
        return

    def get_mathlib_package_dir(self) -> Path:
        """Return the resolved Mathlib package directory inside the workspace."""
        return self.workspace_dir / ".lake" / "packages" / "mathlib"

    def get_mathlib_source_roots(self) -> list[Path]:
        """Return Mathlib source roots that can be indexed for lemma search."""
        package_dir = self.get_mathlib_package_dir()
        source_root = package_dir / "Mathlib"
        return [source_root] if source_root.exists() else []

    def get_mathlib_index_path(self) -> Path:
        """Return the cache path used for the local Mathlib declaration index."""
        return self.workspace_dir / ".moto_mathlib_index.json"

    def get_mathlib_commit(self) -> str:
        """Best-effort lookup of the pinned Mathlib revision for certificate export."""
        manifest_path = self.workspace_dir / "lake-manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for package in manifest.get("packages", []) or []:
                    if package.get("name") != "mathlib":
                        continue
                    for key in ("rev", "inputRev", "version"):
                        value = str(package.get(key, "")).strip()
                        if value:
                            return value
            except Exception as exc:
                logger.debug("Failed to read lake-manifest.json for Mathlib revision: %s", exc)

        package_dir = self.get_mathlib_package_dir()
        head_path = package_dir / ".git" / "HEAD"
        if head_path.exists():
            try:
                head_value = head_path.read_text(encoding="utf-8").strip()
                if head_value.startswith("ref:"):
                    ref_name = head_value.split(":", 1)[1].strip()
                    ref_path = package_dir / ".git" / ref_name
                    if ref_path.exists():
                        return ref_path.read_text(encoding="utf-8").strip()
                if head_value:
                    return head_value
            except Exception as exc:
                logger.debug("Failed to read Mathlib git HEAD: %s", exc)

        toolchain_path = package_dir / "lean-toolchain"
        if toolchain_path.exists():
            try:
                return toolchain_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.debug("Failed to read Mathlib lean-toolchain: %s", exc)

        return ""

    async def _run_process(
        self,
        args: list[str],
        *,
        cwd: Path,
        timeout: int,
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return (
                -1,
                "",
                f"Lean 4 process timed out after {timeout} seconds.",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return process.returncode, stdout, stderr

    async def _run_lean_file_once(
        self,
        *,
        temp_filename: str,
        prepared_code: str,
        timeout: int,
    ) -> tuple[int, str, str]:
        temp_path = self.workspace_dir / temp_filename
        try:
            temp_path.write_text(prepared_code, encoding="utf-8")
            return await self._run_process(
                [self.lake_path, "env", self.lean_path or self._resolve_executable("lean"), temp_filename],
                cwd=self.workspace_dir,
                timeout=timeout,
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                logger.debug("Could not remove temporary Lean file %s", temp_path)

    @staticmethod
    def _combined_process_output(stdout: str, stderr: str) -> str:
        return "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()

    @staticmethod
    def _is_workspace_infrastructure_error(output: str) -> bool:
        text = output or ""
        lowered = text.lower()
        return bool(_OLEAN_OBJECT_FILE_MISSING_RE.search(text)) or any(
            marker in lowered for marker in _LEAN_WORKSPACE_ERROR_MARKERS
        )

    @staticmethod
    def _format_workspace_infrastructure_error(output: str) -> str:
        detail = " ".join((output or "").split())
        if len(detail) > 2000:
            detail = detail[:2000] + "..."
        return (
            f"{_LEAN_WORKSPACE_ERROR_PREFIX}: Mathlib cache/workspace repair failed. "
            "This is an infrastructure problem, not a proof error. "
            "Lean reported missing or invalid compiled Mathlib artifacts. "
            f"Original diagnostic: {detail or '[none]'}"
        )

    def _mark_workspace_unhealthy(self, output: str) -> None:
        self._workspace_ready = False
        self._workspace_unhealthy_error = self._format_workspace_infrastructure_error(output)

    def _workspace_unavailable_result(self, *, tactic_script: bool = False) -> Lean4Result:
        error_output = self._workspace_unhealthy_error or (
            f"{_LEAN_WORKSPACE_ERROR_PREFIX}: Lean 4 workspace is not ready. "
            "This is an infrastructure problem, not a proof error."
        )
        if tactic_script:
            return Lean4Result(
                success=False,
                error_output=error_output,
                tactic_error_slice=error_output,
                failing_tactic_index=-1,
            )
        return Lean4Result(success=False, error_output=error_output)

    async def _repair_workspace_after_infrastructure_error(self, output: str) -> bool:
        logger.warning(
            "Lean 4 workspace infrastructure error detected; invalidating workspace cache and refetching Mathlib artifacts. Diagnostic: %s",
            self._format_workspace_infrastructure_error(output),
        )
        async with self._workspace_lock:
            self._workspace_unhealthy_error = ""
            self._workspace_ready = False
            repaired = await self._ensure_workspace_locked()
            if not repaired:
                self._mark_workspace_unhealthy(output)
            return repaired

    async def get_version(self) -> str:
        """Return the Lean 4 version string when available."""
        lean_cmd = self.lean_path or self._resolve_executable("lean")
        returncode, stdout, stderr = await self._run_process(
            [lean_cmd, "--version"],
            cwd=self.workspace_dir if self.workspace_dir.exists() else Path.cwd(),
            timeout=15,
        )
        if returncode != 0:
            return (stderr or stdout).strip()
        return (stdout or stderr).strip()

    async def ensure_workspace(self) -> bool:
        """Create a reusable Mathlib-enabled workspace if missing."""
        async with self._workspace_lock:
            return await self._ensure_workspace_locked()

    async def _ensure_workspace_locked(self) -> bool:
        """Create a reusable Mathlib-enabled workspace while holding the workspace lock."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        lean_toolchain_path = self.workspace_dir / "lean-toolchain"
        lakefile_path = self.workspace_dir / "lakefile.lean"
        root_file_path = self.workspace_dir / "MOTOProofWorkspace.lean"

        if not lean_toolchain_path.exists():
            lean_toolchain_path.write_text("leanprover/lean4:stable\n", encoding="utf-8")

        if not lakefile_path.exists():
            lakefile_path.write_text(
                "\n".join(
                    [
                        "import Lake",
                        "open Lake DSL",
                        "",
                        "package «moto_proof_workspace» where",
                        "",
                        "require mathlib from git",
                        '  "https://github.com/leanprover-community/mathlib4.git"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        if not root_file_path.exists():
            root_file_path.write_text("import Mathlib\n", encoding="utf-8")

        lake_cmd = self.lake_path
        if not shutil.which(Path(lake_cmd).name) and not Path(lake_cmd).exists():
            logger.warning("Lean 4 workspace not ready: 'lake' executable was not found.")
            return False

        mathlib_pkg_dir = self.workspace_dir / ".lake" / "packages" / "mathlib"
        needs_bootstrap = not mathlib_pkg_dir.exists()

        if needs_bootstrap or not self._workspace_ready:
            logger.info("Bootstrapping Lean 4 workspace at %s", self.workspace_dir)

            update_rc, update_stdout, update_stderr = await self._run_process(
                [lake_cmd, "update"],
                cwd=self.workspace_dir,
                timeout=max(system_config.lean4_proof_timeout, 120),
            )
            if update_rc != 0:
                self._mark_workspace_unhealthy(update_stderr or update_stdout)
                logger.warning(
                    "Lean 4 workspace update failed: %s",
                    (update_stderr or update_stdout).strip(),
                )
                return False

            # The project's lean-toolchain MUST match Mathlib's pinned toolchain,
            # otherwise `lake exe cache get` refuses to download the prebuilt
            # `.olean` files. When that happens, every later `import Mathlib`
            # tries to read object files compiled with a different Lean version
            # (or missing entirely) and aborts with:
            #     error: object file '.../Mathlib/....olean' ...
            # Align the toolchains automatically and re-run `lake update` so the
            # workspace is actually usable for proof checking.
            if self._align_toolchain_with_mathlib(lean_toolchain_path, mathlib_pkg_dir):
                logger.info(
                    "Aligned workspace lean-toolchain with Mathlib; re-running lake update."
                )
                update_rc, update_stdout, update_stderr = await self._run_process(
                    [lake_cmd, "update"],
                    cwd=self.workspace_dir,
                    timeout=max(system_config.lean4_proof_timeout, 120),
                )
                if update_rc != 0:
                    self._mark_workspace_unhealthy(update_stderr or update_stdout)
                    logger.warning(
                        "Lean 4 workspace update after toolchain alignment failed: %s",
                        (update_stderr or update_stdout).strip(),
                    )
                    return False

            cache_rc, cache_stdout, cache_stderr = await self._fetch_mathlib_cache(
                lake_cmd=lake_cmd,
                cwd=self.workspace_dir,
            )
            if cache_rc != 0:
                self._mark_workspace_unhealthy(cache_stderr or cache_stdout)
                logger.error(
                    "Lean 4 Mathlib cache fetch failed; proof checking would hit "
                    "'object file' errors. Details: %s",
                    (cache_stderr or cache_stdout).strip(),
                )
                return False

        self._workspace_ready = True
        self._workspace_unhealthy_error = ""
        return True

    async def _fetch_mathlib_cache(
        self,
        *,
        lake_cmd: str,
        cwd: Path,
    ) -> tuple[int, str, str]:
        """Fetch Mathlib's cache, retrying once after pruning corrupt downloads."""
        timeout = max(system_config.lean4_proof_timeout, 600)
        cache_args = [lake_cmd, "exe", "cache", "get"]
        cache_rc, cache_stdout, cache_stderr = await self._run_process(
            cache_args,
            cwd=cwd,
            timeout=timeout,
        )
        if cache_rc == 0:
            return cache_rc, cache_stdout, cache_stderr

        failed_output = "\n".join(part for part in (cache_stdout, cache_stderr) if part).strip()
        removed_archives = self._remove_failed_mathlib_cache_archives(failed_output)
        if removed_archives:
            logger.warning(
                "Lean 4 Mathlib cache fetch failed after corrupt archive download; "
                "removed %d failed .ltar archive(s) and retrying once.",
                removed_archives,
            )
        else:
            logger.warning(
                "Lean 4 Mathlib cache fetch failed; retrying once in case another "
                "cache process left transient state behind. Details: %s",
                failed_output,
            )

        return await self._run_process(
            cache_args,
            cwd=cwd,
            timeout=timeout,
        )

    @staticmethod
    def _remove_failed_mathlib_cache_archives(output: str) -> int:
        """Delete only the failed `.ltar` archives named by `lake exe cache get`."""
        removed = 0
        seen: set[Path] = set()
        for match in _MATHLIB_CACHE_ARCHIVE_RE.finditer(output or ""):
            archive_path = Path(match.group(1).strip())
            if archive_path in seen:
                continue
            seen.add(archive_path)

            if archive_path.suffix.lower() != ".ltar":
                continue
            if archive_path.parent.name.lower() != "mathlib":
                continue

            try:
                if archive_path.exists():
                    archive_path.unlink()
                    removed += 1
            except OSError as exc:
                logger.debug("Failed to remove corrupt Mathlib cache archive %s: %s", archive_path, exc)
        return removed

    @staticmethod
    def _align_toolchain_with_mathlib(
        workspace_toolchain_path: Path,
        mathlib_pkg_dir: Path,
    ) -> bool:
        """Copy Mathlib's lean-toolchain into the workspace if they differ.

        Returns True when a change was made, False otherwise.
        """
        mathlib_toolchain_path = mathlib_pkg_dir / "lean-toolchain"
        if not mathlib_toolchain_path.exists():
            return False

        try:
            mathlib_toolchain = mathlib_toolchain_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.debug("Failed to read Mathlib lean-toolchain: %s", exc)
            return False

        if not mathlib_toolchain:
            return False

        try:
            current_toolchain = (
                workspace_toolchain_path.read_text(encoding="utf-8").strip()
                if workspace_toolchain_path.exists()
                else ""
            )
        except OSError as exc:
            logger.debug("Failed to read workspace lean-toolchain: %s", exc)
            current_toolchain = ""

        if current_toolchain == mathlib_toolchain:
            return False

        try:
            workspace_toolchain_path.write_text(
                mathlib_toolchain + "\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Failed to write workspace lean-toolchain: %s", exc)
            return False

        logger.info(
            "Updated workspace lean-toolchain: '%s' -> '%s'",
            current_toolchain or "<missing>",
            mathlib_toolchain,
        )
        return True

    def _prepare_lean_code(self, lean_code: str) -> str:
        stripped = (lean_code or "").strip()
        if not stripped:
            return ""

        stripped = _strip_markdown_fences(stripped).strip()
        stripped = _deduplicate_leading_import(stripped).strip()
        if not stripped:
            return ""

        first_lines = stripped.splitlines()[:5]
        if not any(line.strip().startswith("import ") for line in first_lines):
            stripped = f"import Mathlib\n\n{stripped}"
        return stripped + "\n"

    @staticmethod
    def _prioritize_errors_in_output(output: str) -> str:
        """Reorder Lean 4 diagnostics so ``error:`` lines come first.

        Lean 4 emits warnings (e.g., deprecation notices) alongside real
        errors. The displayed error preview and the retry prompt excerpt can
        get truncated on the warning line, hiding the actual failure from the
        model. This helper surfaces error lines (and a few lines of trailing
        context) before the remaining diagnostics without dropping any
        information.
        """
        raw = output or ""
        if not raw.strip():
            return raw

        lines = raw.splitlines()
        error_pattern = re.compile(r":\s*error\s*:", re.IGNORECASE)
        error_indices = [idx for idx, line in enumerate(lines) if error_pattern.search(line)]
        if not error_indices:
            return raw

        ordered: list[str] = []
        seen: set[int] = set()
        for idx in error_indices:
            for offset in range(idx, min(len(lines), idx + 4)):
                if offset in seen:
                    continue
                seen.add(offset)
                ordered.append(lines[offset])
        for idx, line in enumerate(lines):
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(line)
        return "\n".join(ordered)

    @staticmethod
    def _extract_goal_states(output: str) -> str:
        goal_lines = [line for line in output.splitlines() if "⊢" in line or "goals" in line.lower()]
        return "\n".join(goal_lines).strip()

    @staticmethod
    def _extract_error_line_number(output: str, filename: str) -> int:
        pattern = re.compile(rf"{re.escape(filename)}:(\d+):\d+:\s+error:")
        match = pattern.search(output or "")
        if not match:
            return -1
        try:
            return int(match.group(1))
        except ValueError:
            return -1

    @staticmethod
    def _has_no_goals_diagnostic(output: str) -> bool:
        """Return True when Lean's output contains a 'no goals to be solved' error."""
        return bool(_NO_GOALS_DIAGNOSTIC_RE.search(output or ""))

    @staticmethod
    def _annotate_no_goals_hint(error_output: str) -> str:
        """Prepend the targeted 'no goals' hint if Lean reported that diagnostic.

        Appending a concrete repair instruction to the feedback dramatically
        improves the next LLM attempt, because the raw diagnostic by itself
        tends to make models try to *add* more tactics rather than removing
        the extraneous one.
        """
        if not error_output:
            return error_output
        if not Lean4Client._has_no_goals_diagnostic(error_output):
            return error_output
        if _NO_GOALS_HINT in error_output:
            return error_output
        return f"{_NO_GOALS_HINT}\n\n{error_output}"

    @staticmethod
    def _format_tactic_lines(tactic_list: list[str]) -> list[str]:
        lines: list[str] = []
        for tactic in tactic_list:
            stripped = str(tactic or "").rstrip()
            if not stripped:
                continue
            for line in stripped.splitlines():
                lines.append(f"  {line.rstrip()}")
        return lines

    def _build_tactic_script(
        self,
        theorem_header: str,
        tactic_list: list[str],
    ) -> tuple[str, list[tuple[int, int, str]]]:
        header = (theorem_header or "").strip()
        if not header:
            return "", []

        if ":= by" not in header and not header.rstrip().endswith("by"):
            header = f"{header} := by"

        user_lines = header.splitlines()
        formatted_tactic_lines = self._format_tactic_lines(tactic_list)
        if not formatted_tactic_lines:
            return "", []

        tactic_ranges: list[tuple[int, int, str]] = []
        current_line = len(user_lines) + 1
        for tactic in tactic_list:
            stripped = str(tactic or "").rstrip()
            if not stripped:
                continue
            tactic_lines = [f"  {line.rstrip()}" for line in stripped.splitlines()]
            start_line = current_line
            end_line = current_line + len(tactic_lines) - 1
            tactic_ranges.append((start_line, end_line, stripped))
            user_lines.extend(tactic_lines)
            current_line = end_line + 1

        body = "\n".join(user_lines)
        needs_import = not any(line.strip().startswith("import ") for line in user_lines[:5])
        import_offset = 2 if needs_import else 0
        code = self._prepare_lean_code(body)
        adjusted_ranges = [
            (start + import_offset, end + import_offset, tactic)
            for start, end, tactic in tactic_ranges
        ]
        return code, adjusted_ranges

    @staticmethod
    def _extract_tactic_error_slice(
        combined_output: str,
        temp_filename: str,
        tactic_ranges: list[tuple[int, int, str]],
    ) -> tuple[str, int]:
        error_line = Lean4Client._extract_error_line_number(combined_output, temp_filename)
        failing_tactic_index = -1
        failing_tactic = ""
        if error_line >= 0:
            for index, (start_line, end_line, tactic) in enumerate(tactic_ranges):
                if start_line <= error_line <= end_line:
                    failing_tactic_index = index
                    failing_tactic = tactic
                    break

        output_lines = (combined_output or "").splitlines()
        matching_index = next(
            (
                idx
                for idx, line in enumerate(output_lines)
                if temp_filename in line and "error:" in line
            ),
            -1,
        )
        if matching_index >= 0:
            error_slice = "\n".join(output_lines[matching_index : matching_index + 6]).strip()
        else:
            error_slice = (combined_output or "").strip()

        if failing_tactic:
            error_slice = (
                f"Likely failing tactic {failing_tactic_index + 1}: {failing_tactic}\n"
                f"{error_slice}"
            ).strip()
        return error_slice, failing_tactic_index

    async def check_proof(self, lean_code: str, timeout: int = 120) -> Lean4Result:
        """Write a temp Lean file, run Lean 4, and return structured feedback."""
        if not system_config.lean4_enabled:
            return Lean4Result(success=False, error_output="Lean 4 is disabled in system configuration.")

        prepared_code = self._prepare_lean_code(lean_code)
        if not prepared_code:
            return Lean4Result(success=False, error_output="No Lean 4 code was provided.")

        # Fast pre-check: reject placeholder proofs before invoking Lean so
        # the model learns the rejection reason even when Lean would have
        # compiled the file with only a warning.
        placeholder = _detect_forbidden_placeholder(prepared_code)
        if placeholder:
            return Lean4Result(
                success=False,
                error_output=_format_placeholder_rejection(placeholder, from_lean_diagnostic=False),
            )

        workspace_ready = await self.ensure_workspace()
        if not workspace_ready:
            return self._workspace_unavailable_result()

        temp_filename = f"MOTOProofCheck_{uuid.uuid4().hex}.lean"
        async with self._get_lean_execution_lock():
            workspace_ready = await self.ensure_workspace()
            if not workspace_ready:
                return self._workspace_unavailable_result()
            returncode, stdout, stderr = await self._run_lean_file_once(
                temp_filename=temp_filename,
                prepared_code=prepared_code,
                timeout=timeout,
            )
            combined_output = self._combined_process_output(stdout, stderr)
            if self._is_workspace_infrastructure_error(combined_output):
                repaired = await self._repair_workspace_after_infrastructure_error(combined_output)
                if repaired:
                    returncode, stdout, stderr = await self._run_lean_file_once(
                        temp_filename=temp_filename,
                        prepared_code=prepared_code,
                        timeout=timeout,
                    )
                    combined_output = self._combined_process_output(stdout, stderr)
                if self._is_workspace_infrastructure_error(combined_output):
                    self._mark_workspace_unhealthy(combined_output)
                    return Lean4Result(
                        success=False,
                        error_output=self._workspace_unhealthy_error,
                        goal_states=self._extract_goal_states(combined_output),
                        raw_stderr=stderr.strip(),
                    )

        goal_states = self._extract_goal_states(combined_output)

        # Positive pass: Lean must exit cleanly AND the diagnostics must
        # not contain an `error:` line AND must not contain Lean's own
        # "declaration uses 'sorry'" warning. We treat the sorry warning
        # as a proof-level failure so vacuous proofs cannot slip through.
        lowered = combined_output.lower()
        has_error_diagnostic = "error:" in lowered
        has_sorry_warning = _output_contains_sorry_warning(combined_output)
        lean_exited_cleanly = returncode == 0
        positive_pass = (
            lean_exited_cleanly
            and not has_error_diagnostic
            and not has_sorry_warning
        )

        if positive_pass:
            return Lean4Result(
                success=True,
                error_output="",
                goal_states=goal_states,
                raw_stderr=stderr.strip(),
            )

        if has_sorry_warning and not has_error_diagnostic and lean_exited_cleanly:
            rejection = _format_placeholder_rejection("sorry", from_lean_diagnostic=True)
            detail = f"{rejection}\n\nOriginal Lean 4 diagnostics:\n{combined_output}".strip()
            return Lean4Result(
                success=False,
                error_output=detail,
                goal_states=goal_states,
                raw_stderr=stderr.strip(),
            )

        error_output = combined_output or "Lean 4 rejected the proof without additional diagnostics."
        return Lean4Result(
            success=False,
            error_output=self._annotate_no_goals_hint(self._prioritize_errors_in_output(error_output)),
            goal_states=goal_states,
            raw_stderr=stderr.strip(),
        )

    async def check_tactic_script(
        self,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int = 120,
    ) -> Lean4Result:
        """Compile-check a theorem header plus tactic list using the standard Lean subprocess path.

        When Lean reports ``no goals to be solved`` at an identifiable tactic
        index, this method performs a single deterministic auto-repair pass by
        dropping the extraneous tactic(s) and re-checking before returning the
        result. This avoids burning an LLM retry on a purely mechanical fix.
        """
        if not system_config.lean4_enabled:
            return Lean4Result(success=False, error_output="Lean 4 is disabled in system configuration.")

        return await self._check_tactic_script_with_auto_repair(
            theorem_header=theorem_header,
            tactic_list=list(tactic_list or []),
            timeout=timeout,
            auto_repair_attempts_remaining=1,
        )

    async def _check_tactic_script_with_auto_repair(
        self,
        *,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int,
        auto_repair_attempts_remaining: int,
    ) -> Lean4Result:
        result = await self._run_tactic_script_once(
            theorem_header=theorem_header,
            tactic_list=tactic_list,
            timeout=timeout,
        )

        if (
            not result.success
            and auto_repair_attempts_remaining > 0
            and result.failing_tactic_index >= 0
            and self._has_no_goals_diagnostic(result.error_output)
            and result.failing_tactic_index < len(tactic_list)
        ):
            trimmed = tactic_list[: result.failing_tactic_index]
            if trimmed and trimmed != tactic_list:
                logger.info(
                    "Lean 4 reported 'no goals' at tactic index %s; "
                    "auto-trimming %s trailing tactic(s) and retrying once.",
                    result.failing_tactic_index,
                    len(tactic_list) - len(trimmed),
                )
                repaired = await self._check_tactic_script_with_auto_repair(
                    theorem_header=theorem_header,
                    tactic_list=trimmed,
                    timeout=timeout,
                    auto_repair_attempts_remaining=auto_repair_attempts_remaining - 1,
                )
                if repaired.success:
                    return repaired

        return result

    async def _run_tactic_script_once(
        self,
        *,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int,
    ) -> Lean4Result:
        prepared_code, tactic_ranges = self._build_tactic_script(theorem_header, tactic_list)
        if not prepared_code:
            return Lean4Result(success=False, error_output="No tactic script could be constructed from the provided theorem header and tactics.")

        placeholder = _detect_forbidden_placeholder(prepared_code)
        if placeholder:
            return Lean4Result(
                success=False,
                error_output=_format_placeholder_rejection(placeholder, from_lean_diagnostic=False),
            )

        workspace_ready = await self.ensure_workspace()
        if not workspace_ready:
            return self._workspace_unavailable_result(tactic_script=True)

        temp_filename = f"MOTOProofTacticCheck_{uuid.uuid4().hex}.lean"
        async with self._get_lean_execution_lock():
            workspace_ready = await self.ensure_workspace()
            if not workspace_ready:
                return self._workspace_unavailable_result(tactic_script=True)
            returncode, stdout, stderr = await self._run_lean_file_once(
                temp_filename=temp_filename,
                prepared_code=prepared_code,
                timeout=timeout,
            )
            combined_output = self._combined_process_output(stdout, stderr)
            if self._is_workspace_infrastructure_error(combined_output):
                repaired = await self._repair_workspace_after_infrastructure_error(combined_output)
                if repaired:
                    returncode, stdout, stderr = await self._run_lean_file_once(
                        temp_filename=temp_filename,
                        prepared_code=prepared_code,
                        timeout=timeout,
                    )
                    combined_output = self._combined_process_output(stdout, stderr)
                if self._is_workspace_infrastructure_error(combined_output):
                    self._mark_workspace_unhealthy(combined_output)
                    error_output = self._workspace_unhealthy_error
                    return Lean4Result(
                        success=False,
                        error_output=error_output,
                        goal_states=self._extract_goal_states(combined_output),
                        raw_stderr=stderr.strip(),
                        tactic_error_slice=error_output,
                        failing_tactic_index=-1,
                    )

        goal_states = self._extract_goal_states(combined_output)
        lowered = combined_output.lower()
        has_error_diagnostic = "error:" in lowered
        has_sorry_warning = _output_contains_sorry_warning(combined_output)
        lean_exited_cleanly = returncode == 0
        positive_pass = (
            lean_exited_cleanly
            and not has_error_diagnostic
            and not has_sorry_warning
        )
        tactic_error_slice, failing_tactic_index = self._extract_tactic_error_slice(
            combined_output,
            temp_filename,
            tactic_ranges,
        )

        if positive_pass:
            return Lean4Result(
                success=True,
                error_output="",
                goal_states=goal_states,
                raw_stderr=stderr.strip(),
                tactic_error_slice="",
                failing_tactic_index=-1,
            )

        if has_sorry_warning and not has_error_diagnostic and lean_exited_cleanly:
            rejection = _format_placeholder_rejection("sorry", from_lean_diagnostic=True)
            detail = f"{rejection}\n\nOriginal Lean 4 diagnostics:\n{combined_output}".strip()
            return Lean4Result(
                success=False,
                error_output=detail,
                goal_states=goal_states,
                raw_stderr=stderr.strip(),
                tactic_error_slice=rejection,
                failing_tactic_index=failing_tactic_index,
            )

        error_output = tactic_error_slice or combined_output or "Lean 4 rejected the tactic script without additional diagnostics."
        return Lean4Result(
            success=False,
            error_output=self._annotate_no_goals_hint(self._prioritize_errors_in_output(error_output)),
            goal_states=goal_states,
            raw_stderr=stderr.strip(),
            tactic_error_slice=self._annotate_no_goals_hint(tactic_error_slice),
            failing_tactic_index=failing_tactic_index,
        )


class Lean4LspClient(Lean4Client):
    """Persistent Lean LSP client with subprocess fallback."""

    def __init__(self, lean_path: str, workspace_dir: str, *, idle_timeout: int = 600) -> None:
        super().__init__(lean_path=lean_path, workspace_dir=workspace_dir)
        self._idle_timeout = max(int(idle_timeout or 0), 0)
        self._subprocess_fallback = Lean4Client(lean_path=lean_path, workspace_dir=workspace_dir)
        self._server_process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._startup_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._pending_responses: dict[int, asyncio.Future[Any]] = {}
        self._diagnostics_by_uri: dict[str, dict[str, Any]] = {}
        self._file_progress_by_uri: dict[str, dict[str, Any]] = {}
        self._open_document_versions: dict[str, int] = {}
        self._stderr_buffer: list[str] = []
        self._next_request_id = 0
        self._next_document_version = 0
        self._initialized = False
        self._lsp_healthy = True
        self._idle_handle: Optional[asyncio.TimerHandle] = None
        self._expected_shutdown = False

    def uses_persistent_server(self) -> bool:
        return True

    def is_server_active(self) -> bool:
        return bool(
            self._server_process
            and self._server_process.returncode is None
            and self._initialized
        )

    async def warm_start(self) -> None:
        """Best-effort startup of the persistent Lean server."""
        if not system_config.lean4_enabled or not system_config.lean4_lsp_enabled:
            return
        if not self._lsp_healthy:
            return
        workspace_ready = await self.ensure_workspace()
        if not workspace_ready:
            logger.warning("Lean 4 LSP warm start skipped because the workspace is not ready.")
            return
        try:
            await self._ensure_server_started()
        except Exception as exc:
            await self._mark_unhealthy(f"warm start failed: {exc}")

    async def close(self) -> None:
        await self._shutdown_server(mark_unhealthy=False)

    def _cancel_idle_shutdown(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None

    def _schedule_idle_shutdown(self) -> None:
        self._cancel_idle_shutdown()
        if self._idle_timeout <= 0 or not self.is_server_active():
            return
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(
            self._idle_timeout,
            lambda: asyncio.create_task(self._shutdown_if_idle()),
        )

    async def _shutdown_if_idle(self) -> None:
        if self._operation_lock.locked():
            self._schedule_idle_shutdown()
            return
        await self._shutdown_server(mark_unhealthy=False)

    async def _shutdown_server(self, *, mark_unhealthy: bool) -> None:
        self._cancel_idle_shutdown()

        if mark_unhealthy:
            self._lsp_healthy = False

        # Signal the reader loop that any upcoming EOF is intentional so it
        # does not log the shutdown as an unexpected failure.
        self._expected_shutdown = True

        process = self._server_process
        self._server_process = None
        self._initialized = False

        pending = list(self._pending_responses.values())
        self._pending_responses.clear()
        for future in pending:
            if not future.done():
                future.set_exception(RuntimeError("Lean 4 LSP server stopped before replying."))

        if process is not None:
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
                with suppress(Exception):
                    await process.stdin.wait_closed()
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    with suppress(Exception):
                        await process.wait()

        for task_name in ("_reader_task", "_stderr_task"):
            task = getattr(self, task_name)
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
                setattr(self, task_name, None)

        self._diagnostics_by_uri.clear()
        self._file_progress_by_uri.clear()
        self._open_document_versions.clear()

    def _stderr_tail(self) -> str:
        return "\n".join(self._stderr_buffer[-200:]).strip()

    def _next_id(self) -> int:
        self._next_request_id += 1
        return self._next_request_id

    def _next_version(self) -> int:
        self._next_document_version += 1
        return self._next_document_version

    async def _read_message(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line:
                raise EOFError("Lean 4 LSP stream closed.")
            if line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length_raw = headers.get("content-length", "")
        if not content_length_raw:
            raise ValueError("Lean 4 LSP message did not include Content-Length.")

        content_length = int(content_length_raw)
        payload = await reader.readexactly(content_length)
        return json.loads(payload.decode("utf-8", errors="replace"))

    async def _write_message(self, payload: dict[str, Any]) -> None:
        if self._server_process is None or self._server_process.stdin is None or self._server_process.returncode is not None:
            raise RuntimeError("Lean 4 LSP server is not running.")

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._server_process.stdin.write(header + body)
        await self._server_process.stdin.drain()

    async def _send_request(self, method: str, params: dict[str, Any], *, timeout: int) -> Any:
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_responses[request_id] = future
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_responses.pop(request_id, None)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def _store_diagnostics(self, params: dict[str, Any]) -> None:
        uri = str(params.get("uri", "") or "").strip()
        if not uri:
            return
        version = params.get("version")
        if version is None:
            version = self._open_document_versions.get(uri, -1)
        self._diagnostics_by_uri[uri] = {
            "version": int(version) if isinstance(version, int) else self._open_document_versions.get(uri, -1),
            "diagnostics": list(params.get("diagnostics") or []),
        }

    def _store_file_progress(self, params: dict[str, Any]) -> None:
        text_document = params.get("textDocument") or {}
        uri = str(text_document.get("uri", "") or "").strip()
        if not uri:
            return
        processing = list(params.get("processing") or [])
        self._file_progress_by_uri[uri] = {
            "version": text_document.get("version", self._open_document_versions.get(uri, -1)),
            "fatal_error": any(item.get("kind") == "fatalError" for item in processing if isinstance(item, dict)),
        }

    async def _reader_loop(self) -> None:
        assert self._server_process is not None and self._server_process.stdout is not None
        reader = self._server_process.stdout

        try:
            while True:
                payload = await self._read_message(reader)
                if "id" in payload and ("result" in payload or "error" in payload):
                    response_id = payload.get("id")
                    future = self._pending_responses.get(response_id)
                    if future is None or future.done():
                        continue
                    if "error" in payload and payload["error"]:
                        error = payload["error"]
                        future.set_exception(RuntimeError(f"Lean 4 LSP error: {error}"))
                    else:
                        future.set_result(payload.get("result"))
                    continue

                method = str(payload.get("method", "") or "")
                params = payload.get("params") or {}
                if method == "textDocument/publishDiagnostics" and isinstance(params, dict):
                    self._store_diagnostics(params)
                elif method == "$/lean/fileProgress" and isinstance(params, dict):
                    self._store_file_progress(params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._expected_shutdown:
                logger.info("Lean 4 LSP reader exited after idle shutdown")
            else:
                logger.warning("Lean 4 LSP reader stopped: %s", exc)
            pending = list(self._pending_responses.values())
            self._pending_responses.clear()
            for future in pending:
                if not future.done():
                    future.set_exception(RuntimeError(f"Lean 4 LSP reader stopped: {exc}"))

    async def _stderr_loop(self) -> None:
        assert self._server_process is not None and self._server_process.stderr is not None
        reader = self._server_process.stderr
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self._stderr_buffer.append(decoded)
                    if len(self._stderr_buffer) > 200:
                        self._stderr_buffer = self._stderr_buffer[-200:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Lean 4 LSP stderr reader stopped: %s", exc)

    async def _ensure_server_started(self) -> bool:
        if not self._lsp_healthy:
            return False

        async with self._startup_lock:
            if self.is_server_active():
                return True

            if self._server_process is not None and self._server_process.returncode is not None:
                await self._shutdown_server(mark_unhealthy=False)

            try:
                process = await asyncio.create_subprocess_exec(
                    self.lake_path,
                    "serve",
                    cwd=str(self.workspace_dir),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._server_process = process
                self._expected_shutdown = False
                self._reader_task = asyncio.create_task(self._reader_loop())
                self._stderr_task = asyncio.create_task(self._stderr_loop())

                root_uri = self.workspace_dir.resolve().as_uri()
                await self._send_request(
                    "initialize",
                    {
                        "processId": None,
                        "rootPath": str(self.workspace_dir),
                        "rootUri": root_uri,
                        "capabilities": {},
                        "clientInfo": {"name": "MOTO", "version": "build4"},
                        "workspaceFolders": [
                            {
                                "uri": root_uri,
                                "name": self.workspace_dir.name or "workspace",
                            }
                        ],
                    },
                    timeout=30,
                )
                await self._send_notification("initialized", {})
                self._initialized = True
                self._schedule_idle_shutdown()
                logger.info("Lean 4 LSP server started for workspace %s", self.workspace_dir)
                return True
            except Exception as exc:
                await self._mark_unhealthy(f"startup failed: {exc}")
                return False

    async def _mark_unhealthy(self, reason: str) -> None:
        logger.warning("Lean 4 LSP unhealthy; falling back to subprocess mode: %s", reason)
        await self._shutdown_server(mark_unhealthy=True)

    @staticmethod
    def _diagnostic_is_error(diagnostic: dict[str, Any]) -> bool:
        severity = diagnostic.get("severity")
        if severity is not None:
            return int(severity) == 1
        message = str(diagnostic.get("message", "") or "").strip()
        return bool(message)

    @staticmethod
    def _severity_label(diagnostic: dict[str, Any]) -> str:
        severity = diagnostic.get("severity")
        try:
            severity_num = int(severity)
        except (TypeError, ValueError):
            severity_num = 1
        return {
            1: "error",
            2: "warning",
            3: "information",
            4: "hint",
        }.get(severity_num, "error")

    def _format_diagnostic_output(self, temp_filename: str, diagnostics: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for diagnostic in diagnostics:
            range_info = diagnostic.get("range") or {}
            start = range_info.get("start") or {}
            line_number = int(start.get("line", 0) or 0) + 1
            column_number = int(start.get("character", 0) or 0) + 1
            severity = self._severity_label(diagnostic)
            message = str(diagnostic.get("message", "") or "").strip() or "Lean 4 reported a diagnostic without a message."
            message_lines = message.splitlines() or [message]
            lines.append(f"{temp_filename}:{line_number}:{column_number}: {severity}: {message_lines[0]}")
            lines.extend(message_lines[1:])
        return "\n".join(lines).strip()

    def _result_from_diagnostics(
        self,
        temp_filename: str,
        diagnostics: list[dict[str, Any]],
        *,
        tactic_ranges: Optional[list[tuple[int, int, str]]] = None,
        fatal_error: bool = False,
    ) -> Lean4Result:
        combined_output = self._format_diagnostic_output(temp_filename, diagnostics)
        goal_states = self._extract_goal_states(combined_output)
        raw_stderr = self._stderr_tail()
        has_errors = fatal_error or any(self._diagnostic_is_error(diagnostic) for diagnostic in diagnostics)
        has_sorry_warning = _output_contains_sorry_warning(combined_output) or any(
            _output_contains_sorry_warning(str(diagnostic.get("message", "") or ""))
            for diagnostic in diagnostics
        )

        if not has_errors and not has_sorry_warning:
            return Lean4Result(
                success=True,
                error_output="",
                goal_states=goal_states,
                raw_stderr=raw_stderr,
            )

        if has_sorry_warning and not has_errors:
            rejection = _format_placeholder_rejection("sorry", from_lean_diagnostic=True)
            detail = f"{rejection}\n\nOriginal Lean 4 diagnostics:\n{combined_output}".strip()
            return Lean4Result(
                success=False,
                error_output=detail,
                goal_states=goal_states,
                raw_stderr=raw_stderr,
                tactic_error_slice=rejection,
                failing_tactic_index=-1,
            )

        tactic_error_slice = ""
        failing_tactic_index = -1
        if tactic_ranges:
            tactic_error_slice, failing_tactic_index = self._extract_tactic_error_slice(
                combined_output,
                temp_filename,
                tactic_ranges,
            )

        error_output = tactic_error_slice or combined_output
        if not error_output and fatal_error:
            error_output = "Lean 4 LSP reported a fatal error while processing the proof."
        if not error_output:
            error_output = "Lean 4 rejected the proof without additional diagnostics."

        return Lean4Result(
            success=False,
            error_output=self._annotate_no_goals_hint(self._prioritize_errors_in_output(error_output)),
            goal_states=goal_states,
            raw_stderr=raw_stderr,
            tactic_error_slice=self._annotate_no_goals_hint(tactic_error_slice),
            failing_tactic_index=failing_tactic_index,
        )

    async def _check_via_lsp(
        self,
        prepared_code: str,
        *,
        temp_filename: str,
        timeout: int,
        tactic_ranges: Optional[list[tuple[int, int, str]]] = None,
    ) -> Lean4Result:
        if not await self._ensure_server_started():
            raise RuntimeError("Lean 4 LSP server is unavailable.")

        temp_path = self.workspace_dir / temp_filename
        uri = temp_path.resolve().as_uri()
        version = self._next_version()
        self._diagnostics_by_uri.pop(uri, None)
        self._file_progress_by_uri.pop(uri, None)
        self._open_document_versions[uri] = version

        try:
            temp_path.write_text(prepared_code, encoding="utf-8")
            await self._send_notification(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "lean4",
                        "version": version,
                        "text": prepared_code,
                    },
                    "dependencyBuildMode": "once",
                },
            )
            await self._send_request(
                "textDocument/waitForDiagnostics",
                {
                    "uri": uri,
                    "version": version,
                },
                timeout=timeout,
            )

            diagnostic_bundle = self._diagnostics_by_uri.get(uri) or {}
            progress_bundle = self._file_progress_by_uri.get(uri) or {}
            diagnostics = list(diagnostic_bundle.get("diagnostics") or [])
            fatal_error = bool(progress_bundle.get("fatal_error"))
            return self._result_from_diagnostics(
                temp_filename,
                diagnostics,
                tactic_ranges=tactic_ranges,
                fatal_error=fatal_error,
            )
        finally:
            if self.is_server_active():
                with suppress(Exception):
                    await self._send_notification(
                        "textDocument/didClose",
                        {
                            "textDocument": {
                                "uri": uri,
                            }
                        },
                    )
            self._diagnostics_by_uri.pop(uri, None)
            self._file_progress_by_uri.pop(uri, None)
            self._open_document_versions.pop(uri, None)
            with suppress(OSError):
                if temp_path.exists():
                    temp_path.unlink()

    async def check_proof(self, lean_code: str, timeout: int = 120) -> Lean4Result:
        """Check a proof through the persistent Lean LSP when healthy, otherwise fall back."""
        if not system_config.lean4_enabled:
            return Lean4Result(success=False, error_output="Lean 4 is disabled in system configuration.")

        prepared_code = self._prepare_lean_code(lean_code)
        if not prepared_code:
            return Lean4Result(success=False, error_output="No Lean 4 code was provided.")

        placeholder = _detect_forbidden_placeholder(prepared_code)
        if placeholder:
            return Lean4Result(
                success=False,
                error_output=_format_placeholder_rejection(placeholder, from_lean_diagnostic=False),
            )

        workspace_ready = await self.ensure_workspace()
        if not workspace_ready:
            return self._workspace_unavailable_result()

        if not self._lsp_healthy:
            return await self._subprocess_fallback.check_proof(lean_code, timeout=timeout)

        async with self._operation_lock:
            self._cancel_idle_shutdown()
            try:
                result = await self._check_via_lsp(
                    prepared_code,
                    temp_filename=f"MOTOProofCheck_{uuid.uuid4().hex}.lean",
                    timeout=timeout,
                )
                if self._is_workspace_infrastructure_error(result.error_output):
                    await self._mark_unhealthy(result.error_output)
                    return await self._subprocess_fallback.check_proof(lean_code, timeout=timeout)
                return result
            except Exception as exc:
                await self._mark_unhealthy(str(exc))
                return await self._subprocess_fallback.check_proof(lean_code, timeout=timeout)
            finally:
                if self._lsp_healthy:
                    self._schedule_idle_shutdown()

    async def check_tactic_script(
        self,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int = 120,
    ) -> Lean4Result:
        """Check a tactic script through the persistent Lean LSP when healthy, otherwise fall back.

        Mirrors the subprocess path by performing one deterministic auto-repair
        attempt when Lean reports ``no goals to be solved`` at a known tactic
        index.
        """
        if not system_config.lean4_enabled:
            return Lean4Result(success=False, error_output="Lean 4 is disabled in system configuration.")

        return await self._check_tactic_script_via_lsp_with_auto_repair(
            theorem_header=theorem_header,
            tactic_list=list(tactic_list or []),
            timeout=timeout,
            auto_repair_attempts_remaining=1,
        )

    async def _check_tactic_script_via_lsp_with_auto_repair(
        self,
        *,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int,
        auto_repair_attempts_remaining: int,
    ) -> Lean4Result:
        result = await self._run_tactic_script_via_lsp_once(
            theorem_header=theorem_header,
            tactic_list=tactic_list,
            timeout=timeout,
        )

        if (
            not result.success
            and auto_repair_attempts_remaining > 0
            and result.failing_tactic_index >= 0
            and self._has_no_goals_diagnostic(result.error_output)
            and result.failing_tactic_index < len(tactic_list)
        ):
            trimmed = tactic_list[: result.failing_tactic_index]
            if trimmed and trimmed != tactic_list:
                logger.info(
                    "Lean 4 LSP reported 'no goals' at tactic index %s; "
                    "auto-trimming %s trailing tactic(s) and retrying once.",
                    result.failing_tactic_index,
                    len(tactic_list) - len(trimmed),
                )
                repaired = await self._check_tactic_script_via_lsp_with_auto_repair(
                    theorem_header=theorem_header,
                    tactic_list=trimmed,
                    timeout=timeout,
                    auto_repair_attempts_remaining=auto_repair_attempts_remaining - 1,
                )
                if repaired.success:
                    return repaired

        return result

    async def _run_tactic_script_via_lsp_once(
        self,
        *,
        theorem_header: str,
        tactic_list: list[str],
        timeout: int,
    ) -> Lean4Result:
        prepared_code, tactic_ranges = self._build_tactic_script(theorem_header, tactic_list)
        if not prepared_code:
            return Lean4Result(success=False, error_output="No tactic script could be constructed from the provided theorem header and tactics.")

        placeholder = _detect_forbidden_placeholder(prepared_code)
        if placeholder:
            return Lean4Result(
                success=False,
                error_output=_format_placeholder_rejection(placeholder, from_lean_diagnostic=False),
            )

        workspace_ready = await self.ensure_workspace()
        if not workspace_ready:
            return self._workspace_unavailable_result(tactic_script=True)

        if not self._lsp_healthy:
            return await self._subprocess_fallback._run_tactic_script_once(
                theorem_header=theorem_header,
                tactic_list=tactic_list,
                timeout=timeout,
            )

        async with self._operation_lock:
            self._cancel_idle_shutdown()
            try:
                result = await self._check_via_lsp(
                    prepared_code,
                    temp_filename=f"MOTOProofTacticCheck_{uuid.uuid4().hex}.lean",
                    timeout=timeout,
                    tactic_ranges=tactic_ranges,
                )
                if self._is_workspace_infrastructure_error(result.error_output):
                    await self._mark_unhealthy(result.error_output)
                    return await self._subprocess_fallback._run_tactic_script_once(
                        theorem_header=theorem_header,
                        tactic_list=tactic_list,
                        timeout=timeout,
                    )
                return result
            except Exception as exc:
                await self._mark_unhealthy(str(exc))
                return await self._subprocess_fallback._run_tactic_script_once(
                    theorem_header=theorem_header,
                    tactic_list=tactic_list,
                    timeout=timeout,
                )
            finally:
                if self._lsp_healthy:
                    self._schedule_idle_shutdown()


_lean4_client: Optional[Lean4Client] = None


def _build_client(lean_path: str, workspace_dir: str) -> Lean4Client:
    if system_config.lean4_lsp_enabled:
        return Lean4LspClient(
            lean_path=lean_path,
            workspace_dir=workspace_dir,
            idle_timeout=system_config.lean4_lsp_idle_timeout,
        )
    return Lean4Client(
        lean_path=lean_path,
        workspace_dir=workspace_dir,
    )


def initialize_lean4_client(lean_path: Optional[str] = None, workspace_dir: Optional[str] = None) -> Lean4Client:
    """Create or replace the singleton Lean 4 client."""
    global _lean4_client
    _lean4_client = _build_client(
        lean_path=lean_path or system_config.lean4_path,
        workspace_dir=workspace_dir or system_config.lean4_workspace_dir,
    )
    return _lean4_client


def get_lean4_client() -> Lean4Client:
    """Return the singleton Lean 4 client, creating it from config if needed."""
    global _lean4_client
    if _lean4_client is None:
        _lean4_client = _build_client(
            lean_path=system_config.lean4_path,
            workspace_dir=system_config.lean4_workspace_dir,
        )
    return _lean4_client


async def close_lean4_client() -> None:
    """Close the singleton Lean 4 client if it owns a persistent server."""
    client = _lean4_client
    if client is not None:
        await client.close()


def clear_lean4_client() -> None:
    """Reset the singleton Lean 4 client."""
    global _lean4_client
    _lean4_client = None
