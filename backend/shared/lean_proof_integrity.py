"""Shared integrity checks for Lean 4 proof outputs."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.autonomous.prompts.proof_prompts import build_proof_statement_alignment_prompt
from backend.shared.api_client_manager import RetryableProviderError, api_client_manager
from backend.shared.config import rag_config
from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text
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
    matches_intended: Optional[bool] = None
    actual_theorem_statement: str = ""
    actual_theorem_name: str = ""
    relationship_to_candidate: str = ""
    downshift_reason: str = ""


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


def extract_primary_lean_theorem(lean_code: str) -> tuple[str, str]:
    """Best-effort extraction of the main theorem/lemma header from Lean code."""
    cleaned = strip_lean_comments_and_strings(lean_code)
    headers: list[tuple[str, str]] = []
    collecting = False
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        header = " ".join(part.strip() for part in current if part.strip())
        header = re.sub(r"\s*:=\s*by\b.*$", "", header).strip()
        header = re.sub(r"\s*:=\s*.*$", "", header).strip()
        if header:
            parts = header.split()
            name = ""
            if len(parts) >= 2 and parts[0] in {"theorem", "lemma"}:
                name = parts[1]
            headers.append((name, header))
        current = []

    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if re.match(r"^(theorem|lemma|example)\b", stripped):
            if collecting:
                flush_current()
            collecting = True
            current = [stripped]
            if ":=" in stripped:
                flush_current()
                collecting = False
            continue
        if collecting:
            if re.match(
                r"^(def|structure|class|inductive|instance|abbrev|namespace|section|end|open|variable|variables)\b",
                stripped,
            ):
                flush_current()
                collecting = False
                continue
            current.append(stripped)
            if ":=" in stripped:
                flush_current()
                collecting = False

    if collecting:
        flush_current()

    return headers[-1] if headers else ("", "")


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
    """Classify whether accepted Lean code matches the intended claim without rejecting it."""
    fallback_name, fallback_statement = extract_primary_lean_theorem(lean_code)
    prompt = build_proof_statement_alignment_prompt(
        user_prompt=user_prompt,
        theorem_statement=theorem_statement,
        formal_sketch=formal_sketch,
        lean_code=lean_code,
        source_excerpt=source_excerpt,
    )
    max_input_tokens = rag_config.get_available_input_tokens(validator_context, validator_max_tokens)
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
    if count_tokens(prompt) > max_input_tokens:
        return LeanProofIntegrityResult(
            valid=True,
            category="statement_alignment_unavailable",
            reason="Statement-alignment classifier prompt exceeded the configured context window; preserving Lean-accepted proof.",
            matches_intended=None,
            actual_theorem_statement=fallback_statement or theorem_statement,
            actual_theorem_name=fallback_name,
            relationship_to_candidate="alignment_unavailable",
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
                valid=True,
                category="statement_alignment_unavailable",
                reason="Statement-alignment classifier returned no response; preserving Lean-accepted proof.",
                matches_intended=None,
                actual_theorem_statement=fallback_statement or theorem_statement,
                actual_theorem_name=fallback_name,
                relationship_to_candidate="alignment_unavailable",
            )
        message = response["choices"][0].get("message", {})
        content = extract_message_text(message)
        if not content:
            return LeanProofIntegrityResult(
                valid=True,
                category="statement_alignment_unavailable",
                reason="Statement-alignment classifier returned empty content; preserving Lean-accepted proof.",
                matches_intended=None,
                actual_theorem_statement=fallback_statement or theorem_statement,
                actual_theorem_name=fallback_name,
                relationship_to_candidate="alignment_unavailable",
            )
        data = parse_json(content)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        if isinstance(exc, RetryableProviderError) or is_non_retryable_model_error(exc):
            raise
        logger.warning("Lean statement alignment validation failed: %s", exc)
        return LeanProofIntegrityResult(
            valid=True,
            category="statement_alignment_unavailable",
            reason=(
                "Statement-alignment classification failed before a usable decision was produced; "
                f"preserving Lean-accepted proof. {type(exc).__name__}: {exc}"
            ),
            matches_intended=None,
            actual_theorem_statement=fallback_statement or theorem_statement,
            actual_theorem_name=fallback_name,
            relationship_to_candidate="alignment_unavailable",
        )

    raw_matches = data.get("matches_intended")
    if isinstance(raw_matches, bool):
        matches_intended = raw_matches
    else:
        decision = str(data.get("decision") or "").strip().lower()
        matches_intended = decision == "accept" if decision else None

    actual_statement = str(
        data.get("actual_theorem_statement")
        or data.get("proved_theorem_statement")
        or data.get("verified_theorem_statement")
        or ""
    ).strip()
    if not actual_statement:
        actual_statement = theorem_statement if matches_intended is True else (fallback_statement or theorem_statement)
    actual_name = str(data.get("actual_theorem_name") or data.get("theorem_name") or fallback_name).strip()
    relationship = str(data.get("relationship_to_candidate") or data.get("relationship") or "").strip()
    downshift_reason = str(data.get("downshift_reason") or data.get("summary") or "").strip()
    reasoning = str(data.get("reasoning") or data.get("summary") or "").strip()

    category = "statement_alignment" if matches_intended is True else "statement_downshifted"
    if matches_intended is None:
        category = "statement_alignment_uncertain"

    return LeanProofIntegrityResult(
        valid=True,
        reason=reasoning or downshift_reason,
        category=category,
        matches_intended=matches_intended,
        actual_theorem_statement=actual_statement,
        actual_theorem_name=actual_name,
        relationship_to_candidate=relationship,
        downshift_reason=downshift_reason,
    )


async def validate_full_lean_proof_integrity(
    *,
    user_prompt: str,
    theorem_statement: str,
    formal_sketch: str,
    lean_code: str,
    source_excerpt: str,
    allowed_baseline: str,
    validator_model: Optional[str] = None,
    validator_context: int = 0,
    validator_max_tokens: int = 0,
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
        fallback_name, fallback_statement = extract_primary_lean_theorem(lean_code)
        return LeanProofIntegrityResult(
            valid=True,
            category="statement_alignment_unavailable",
            reason="No validator model configured for statement alignment; preserving Lean-accepted proof.",
            matches_intended=None,
            actual_theorem_statement=fallback_statement or theorem_statement,
            actual_theorem_name=fallback_name,
            relationship_to_candidate="alignment_unavailable",
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
