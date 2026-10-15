"""Deterministic, path-free evidence helpers for Lean verification."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


LEAN_SOURCE_TRANSFORMATION_VERSION = "moto-lean-source-v1"
LEAN_IMPORT_PARSER_VERSION = "moto-lean-imports-v1"
LEAN_VERIFICATION_PROFILE_VERSION = "moto-lean-verification-v1"
LEAN_COMPILE_RECEIPT_VERSION = "moto-lean-compile-receipt-v1"
PORTABLE_DEPENDENCY_VERSION = "moto-portable-dependency-v1"


def source_bytes(source: str) -> bytes:
    """Return the exact UTF-8 representation used by MOTO Lean checks."""
    return (source or "").encode("utf-8")


def source_sha256(source: str) -> str:
    return hashlib.sha256(source_bytes(source)).hexdigest()


def _strip_comments_and_strings_preserving_newlines(source: str) -> str:
    """Mask comments/strings while preserving line boundaries for import parsing."""
    text = source or ""
    output: list[str] = []
    index = 0
    block_depth = 0
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]
        pair = text[index : index + 2]
        if block_depth:
            if pair == "/-":
                block_depth += 1
                output.extend("  ")
                index += 2
                continue
            if pair == "-/":
                block_depth -= 1
                output.extend("  ")
                index += 2
                continue
            output.append("\n" if char == "\n" else " ")
            index += 1
            continue
        if in_string:
            output.append("\n" if char == "\n" else " ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if pair == "/-":
            block_depth = 1
            output.extend("  ")
            index += 2
            continue
        if pair == "--":
            newline = text.find("\n", index)
            if newline < 0:
                output.extend(" " * (len(text) - index))
                break
            output.extend(" " * (newline - index))
            index = newline
            continue
        if char == '"':
            in_string = True
            output.append(" ")
            index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


_IMPORT_LINE_RE = re.compile(r"(?m)^[ \t]*import[ \t]+([^\n]+)$")
_LEAN_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*$")


def parse_explicit_import_modules(source: str) -> list[str]:
    """Parse ordered, unique top-level module names from Lean import commands."""
    masked = _strip_comments_and_strings_preserving_newlines(source)
    modules: list[str] = []
    seen: set[str] = set()
    for match in _IMPORT_LINE_RE.finditer(masked):
        for token in match.group(1).split():
            module = token.strip()
            if not _LEAN_MODULE_RE.fullmatch(module) or module in seen:
                continue
            seen.add(module)
            modules.append(module)
    return modules


def canonical_receipt_payload(payload: dict[str, Any]) -> bytes:
    """Canonicalize a receipt payload without its self-digest."""
    normalized = dict(payload)
    normalized.pop("receipt_digest", None)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def receipt_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_receipt_payload(payload)).hexdigest()


def seal_receipt(receipt: Any) -> Any:
    """Return a receipt model whose digest covers its own JSON serialization."""
    payload = receipt.model_dump(mode="json")
    payload["receipt_digest"] = receipt_digest(payload)
    return type(receipt).model_validate(payload)


def has_v1_1_08_receipt_digest(receipt: Any) -> bool:
    """Match receipts sealed by v1.1.08, which digested ``str(datetime)`` values."""
    return receipt.receipt_digest == receipt_digest(receipt.model_dump(mode="python"))


def transformations_between(original_source: str, checked_source: str) -> list[str]:
    """Return stable coarse transformation labels without leaking source text."""
    if original_source == checked_source:
        return []
    transformations: list[str] = []
    original = original_source or ""
    checked = checked_source or ""
    if original.strip() != original:
        transformations.append("trim_outer_whitespace")
    if "```" in original and "```" not in checked:
        transformations.append("strip_markdown_fence")
    original_imports = parse_explicit_import_modules(original)
    checked_imports = parse_explicit_import_modules(checked)
    if not original_imports and checked_imports:
        transformations.append("insert_default_import")
    if len(original_imports) != len(set(original_imports)):
        transformations.append("deduplicate_leading_import")
    if not checked.endswith("\n") or not original.endswith("\n"):
        transformations.append("normalize_final_newline")
    if not transformations:
        transformations.append("normalize_lean_source")
    return transformations


def contains_private_path(value: str) -> bool:
    """Conservative guard for dependency values that look host-local."""
    text = (value or "").strip()
    return bool(
        re.search(r"(?i)(?:^[A-Z]:[\\/]|(?:^|[\\/])users[\\/]|^/home/|^/tmp/|^/var/)", text)
    )


def stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
