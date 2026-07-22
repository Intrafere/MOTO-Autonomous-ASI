"""Versioned canonical identity for internally managed Lean proof artifacts."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


CANONICAL_PROOF_IDENTITY_VERSION = "moto-proof-identity-v1"


def canonicalize_theorem_statement(value: str) -> str:
    """Collapse theorem-statement whitespace for exact identity matching."""
    return " ".join((value or "").split())


def canonicalize_lean_code(value: str) -> str:
    """Normalize Lean newlines and outer whitespace without changing its body."""
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalProofIdentity:
    version: str
    theorem_statement_hash: str
    lean_code_hash: str

    @property
    def key(self) -> str:
        return f"{self.version}:{self.theorem_statement_hash}:{self.lean_code_hash}"


def canonical_proof_identity(
    theorem_statement: str,
    lean_code: str,
) -> CanonicalProofIdentity:
    """Build the canonical identity used by MOTO/manual/LeanOJ proof records."""
    return CanonicalProofIdentity(
        version=CANONICAL_PROOF_IDENTITY_VERSION,
        theorem_statement_hash=_sha256(canonicalize_theorem_statement(theorem_statement)),
        lean_code_hash=_sha256(canonicalize_lean_code(lean_code)),
    )
