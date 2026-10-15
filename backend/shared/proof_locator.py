"""Canonical local proof-occurrence locators for future proof trade."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Optional

from backend.shared.installation_identity import InstallationBinding
from backend.shared.models import CanonicalProofLocator

PROOF_LOCATOR_VERSION = "moto-proof-locator-v1"
CLIENT_PROOF_ID_VERSION = "moto-client-proof-id-v1"
CLIENT_BATCH_ID_VERSION = "moto-client-batch-id-v1"
PROOF_STORE_ID_VERSION = "moto-proof-store-id-v1"
_CLIENT_PROOF_ID_RE = re.compile(r"^moto_proof_[0-9a-f]{32}$")
_CLIENT_BATCH_ID_RE = re.compile(r"^moto_batch_[0-9a-f]{32}$")
_SELECTION_SNAPSHOT_ID_RE = re.compile(r"^moto_selection_[0-9a-f]{32}$")


def new_stable_store_id() -> str:
    return f"moto_store_{uuid.uuid4().hex}"


def new_client_proof_id() -> str:
    return f"moto_proof_{uuid.uuid4().hex}"


def is_valid_client_proof_id(value: str) -> bool:
    """Return whether value is a canonical opaque client-proof identifier."""
    return bool(_CLIENT_PROOF_ID_RE.fullmatch(str(value or "")))


def new_client_batch_id() -> str:
    """Return an opaque future trade-batch ID without creating batch state."""
    return f"moto_batch_{uuid.uuid4().hex}"


def is_valid_client_batch_id(value: str) -> bool:
    """Return whether value is a canonical opaque client-batch identifier."""
    return bool(_CLIENT_BATCH_ID_RE.fullmatch(str(value or "")))


def new_selection_snapshot_id() -> str:
    """Return an opaque local proof-trade selection-snapshot identifier."""
    return f"moto_selection_{uuid.uuid4().hex}"


def is_valid_selection_snapshot_id(value: str) -> bool:
    """Return whether value is a canonical opaque selection-snapshot identifier."""
    return bool(_SELECTION_SNAPSHOT_ID_RE.fullmatch(str(value or "")))


def build_proof_locator(
    *,
    binding: InstallationBinding,
    store_kind: str,
    stable_store_id: str,
    run_id: str,
    proof_id: str,
    canonical_identity_version: str,
    theorem_hash: str,
    lean_code_hash: str,
    session_id: str = "",
    archive_id: str = "",
) -> CanonicalProofLocator:
    return CanonicalProofLocator(
        locator_version=PROOF_LOCATOR_VERSION,
        installation_id=binding.installation_id,
        root_binding_id=binding.root_binding_id,
        store_kind=store_kind,
        stable_store_id=stable_store_id,
        session_id=session_id,
        archive_id=archive_id,
        run_id=run_id,
        proof_id=proof_id,
        canonical_identity_version=canonical_identity_version,
        canonical_theorem_statement_hash=theorem_hash,
        canonical_lean_code_hash=lean_code_hash,
    )


def canonical_locator_digest(locator: CanonicalProofLocator) -> str:
    payload = json.dumps(
        locator.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def qualified_locator_digest(
    locator: Optional[CanonicalProofLocator],
) -> str:
    return canonical_locator_digest(locator) if locator is not None else ""
