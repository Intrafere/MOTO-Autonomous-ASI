"""Shared integrity checks for Lean 4 proof outputs."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.autonomous.prompts.proof_prompts import build_proof_statement_alignment_prompt
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.model_error_utils import is_non_retryable_model_error
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

_LEAN_DECL_NAME = r"(?:[A-Za-z_][A-Za-z0-9_'.]*|«[^»]+»)"

_DECLARATION_DEVICE_COMMAND_RE = re.compile(
    r"^\s*(?:@\[[^\]]+\]\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+)*"
    r"(axiom|constant|opaque)\b(?P<body>.*?)"
    r"(?=^\s*(?:@\[[^\]]+\]\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+)*"
    r"(?:axiom|constant|opaque|theorem|lemma|def|example|import|namespace|section|end|open|"
    r"variable|variables|structure|class|inductive|instance|abbrev)\b|\Z)",
    re.MULTILINE | re.DOTALL,
)
_DECLARATION_NAME_RE = re.compile(_LEAN_DECL_NAME)
_DECLARATION_BINDER_RE = re.compile(rf"\(\s*({_LEAN_DECL_NAME}(?:\s+{_LEAN_DECL_NAME})*)\s*:")
_DECLARATION_LEADING_NAMES_RE = re.compile(rf"^\s*({_LEAN_DECL_NAME}(?:\s+{_LEAN_DECL_NAME})*)\s*(?::|:=|where\b|$)")


@dataclass
class LeanProofIntegrityResult:
    """Result of non-Lean integrity checks applied after Lean accepts code."""
    valid: bool
    reason: str = ""
    category: str = "ok"
    introduced_devices: list[str] = field(default_factory=list)


def strip_lean_comments_and_strings(code: str) -> str:
    """Best-effort removal of comments and string literals before source scanning."""
    without_block_comments = re.sub(r"/-.*?-/", " ", code or "", flags=re.DOTALL)
    without_line_comments = re.sub(r"--[^\n]*", " ", without_block_comments)
    return re.sub(r'"(?:\\.|[^"\\])*"', ' "" ', without_line_comments)


def find_declaration_devices(code: str) -> set[tuple[str, str]]:
    """Return axiom/constant/opaque declarations found in Lean source."""
    devices: set[tuple[str, str]] = set()
    for match in _DECLARATION_DEVICE_COMMAND_RE.finditer(strip_lean_comments_and_strings(code)):
        kind = match.group(1)
        body = match.group("body") or ""
        names: list[str] = []

        for binder_match in _DECLARATION_BINDER_RE.finditer(body):
            names.extend(name.group(0) for name in _DECLARATION_NAME_RE.finditer(binder_match.group(1)))

        if not names:
            leading_match = _DECLARATION_LEADING_NAMES_RE.match(body)
            if leading_match:
                names.extend(name.group(0) for name in _DECLARATION_NAME_RE.finditer(leading_match.group(1)))

        for name in names:
            devices.add((kind, name))
    return devices


def find_introduced_declaration_devices(lean_code: str, allowed_baseline: str = "") -> list[str]:
    """Return declaration devices present in ``lean_code`` but absent from baseline."""
    allowed = find_declaration_devices(allowed_baseline)
    introduced: list[str] = []
    for kind, name in sorted(find_declaration_devices(lean_code)):
        if (kind, name) not in allowed:
            introduced.append(f"{kind} {name}")
    return introduced


def validate_lean_proof_integrity(
    *,
    lean_code: str,
    allowed_baseline: str = "",
) -> LeanProofIntegrityResult:
    """Reject fake declaration devices that Lean accepts but MOTO does not."""
    introduced = find_introduced_declaration_devices(
        lean_code=lean_code,
        allowed_baseline=allowed_baseline,
    )
    if introduced:
        return LeanProofIntegrityResult(
            valid=False,
            category="forbidden_declaration_device",
            introduced_devices=introduced,
            reason=(
                "LEAN PROOF INTEGRITY REJECTED: the submitted Lean code introduces new "
                "axiom/constant/opaque declarations not present in the allowed baseline: "
                f"{', '.join(introduced[:8])}. Do not prove results by adding fake assumptions; "
                "use constructive Lean/Mathlib proof terms or tactics."
            ),
        )
    return LeanProofIntegrityResult(valid=True)


async def validate_lean_statement_alignment(
    *,
    user_prompt: str,
    theorem_statement: str,
    formal_sketch: str,
    lean_code: str,
    source_excerpt: str,
    validator_model: str,
    validator_context: int,
    validator_max_tokens: int,
    task_id: str,
    role_id: str,
) -> LeanProofIntegrityResult:
    """Use an LLM validator to ensure accepted Lean code matches the intended claim."""
    prompt = build_proof_statement_alignment_prompt(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        formal_sketch=formal_sketch,
        lean_code=lean_code,
        source_excerpt=source_excerpt,
    )
    max_input_tokens = validator_context - validator_max_tokens
    trimmed_excerpt = source_excerpt or ""
    while count_tokens(prompt) > max_input_tokens and len(trimmed_excerpt) > 1500:
        trimmed_excerpt = trimmed_excerpt[: max(len(trimmed_excerpt) // 2, 1500)]
        prompt = build_proof_statement_alignment_prompt(
            user_prompt=user_prompt,
            theorem_statement=theorem_statement,
            formal_sketch=formal_sketch,
            lean_code=lean_code,
            source_excerpt=trimmed_excerpt,
        )

    try:
        response = await api_client_manager.generate_completion(
            task_id=task_id,
            role_id=role_id,
            model=validator_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=validator_max_tokens,
            temperature=0.0,
        )
        if not response or not response.get("choices"):
            return LeanProofIntegrityResult(
                valid=False,
                category="statement_alignment_unavailable",
                reason="LEAN PROOF INTEGRITY REJECTED: statement-alignment validator returned no response.",
            )
        message = response["choices"][0].get("message", {})
        content = message.get("content") or message.get("reasoning") or ""
        if not content:
            return LeanProofIntegrityResult(
                valid=False,
                category="statement_alignment_unavailable",
                reason="LEAN PROOF INTEGRITY REJECTED: statement-alignment validator returned empty content.",
            )
        data = parse_json(content)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        if is_non_retryable_model_error(exc):
            raise
        logger.warning("Lean statement alignment validation failed: %s", exc)
        return LeanProofIntegrityResult(
            valid=False,
            category="statement_alignment_unavailable",
            reason=(
                "LEAN PROOF INTEGRITY REJECTED: statement-alignment validation failed before "
                f"a usable decision was produced: {type(exc).__name__}: {exc}"
            ),
        )

    decision = str(data.get("decision") or "").strip().lower()
    reasoning = str(data.get("reasoning") or data.get("summary") or "").strip()
    if decision != "accept":
        return LeanProofIntegrityResult(
            valid=False,
            category="statement_alignment_rejected",
            reason=(
                "LEAN PROOF INTEGRITY REJECTED: Lean accepted the code, but the statement-alignment "
                f"validator rejected it as unrelated or insufficient. {reasoning}"
            ).strip(),
        )
    return LeanProofIntegrityResult(valid=True, reason=reasoning, category="statement_alignment")


async def validate_full_lean_proof_integrity(
    *,
    user_prompt: str,
    theorem_statement: str,
    formal_sketch: str,
    lean_code: str,
    source_excerpt: str,
    allowed_baseline: str,
    validator_model: Optional[str] = None,
    validator_context: int = 131072,
    validator_max_tokens: int = 25000,
    task_id: str = "proof_integrity_000",
    role_id: str = "proof_integrity_validator",
    require_statement_alignment: bool = True,
) -> LeanProofIntegrityResult:
    """Run all post-Lean integrity checks used by proof-producing systems."""
    structural = validate_lean_proof_integrity(
        lean_code=lean_code,
        allowed_baseline=allowed_baseline,
    )
    if not structural.valid:
        return structural
    if not require_statement_alignment:
        return structural
    if not validator_model:
        return LeanProofIntegrityResult(
            valid=False,
            category="statement_alignment_unavailable",
            reason="LEAN PROOF INTEGRITY REJECTED: no validator model was configured for statement alignment.",
        )
    return await validate_lean_statement_alignment(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        formal_sketch=formal_sketch,
        lean_code=lean_code,
        source_excerpt=source_excerpt,
        validator_model=validator_model,
        validator_context=validator_context,
        validator_max_tokens=validator_max_tokens,
        task_id=task_id,
        role_id=role_id,
    )
