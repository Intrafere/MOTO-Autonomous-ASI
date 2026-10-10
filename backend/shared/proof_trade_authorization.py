"""Contract-independent proof-trade desktop authorization scaffolding.

This module freezes only what the canonical cross-team plan already fixes:

* public native client, OAuth Authorization Code with PKCE ``S256`` only;
* the exact proof-trade scope set (no aliases, no extra corpus scopes);
* a distinct OS-keyring credential record bound to the registered client and
  the stable MOTO installation identity, never ``syntheticlib4_api_key`` or a
  provider OAuth record; and
* fail-closed readiness in generic/hosted mode, without a usable keyring, or
  without a verified installation binding.

It performs no network access, login, browser return handling, token
exchange, or credential write. The redirect form, endpoints, client ID, token
shape, and lifetimes belong to the unpublished registered native-client
profile, so authorization stays disabled until Build 9 ingests that profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Iterable, Tuple

from backend.shared import secret_store
from backend.shared.config import system_config
from backend.shared.oauth_pkce import PKCE_METHOD

logger = logging.getLogger(__name__)

AUTHORIZATION_READINESS_VERSION = "moto-proof-trade-authorization-readiness-v1"
PROOF_TRADE_SCOPES: Tuple[str, ...] = ("proof_trade:create", "proof_trade:read_own")
PROOF_TRADE_CREDENTIAL_PREFIX = "proof_trade_credential"
REDIRECT_POLICY = "profile_defined"

REASON_GENERIC_MODE = "generic_mode_unavailable"
REASON_PROFILE_UNPUBLISHED = "native_client_profile_unpublished"
REASON_CONTRACT_UNPUBLISHED = "frozen_syntheticlib_contract_unpublished"
REASON_ASSET_INVENTORY_INVALID = "frozen_asset_inventory_invalid"
REASON_CONTRACT_NOT_REGISTERED = "frozen_contract_not_registered"
REASON_KEYRING_UNAVAILABLE = "keyring_unavailable"
REASON_INSTALLATION_UNAVAILABLE = "installation_identity_unavailable"
REASON_INSTALLATION_UNBOUND = "installation_identity_unbound"

SCOPE_MISSING = "scope_missing"
SCOPE_UNEXPECTED = "scope_unexpected"
SCOPE_MALFORMED = "scope_malformed"


def validate_exact_proof_trade_scopes(granted: Iterable[object]) -> Tuple[str, ...]:
    """Return reason codes; an empty tuple means exactly the required scopes.

    Aliases such as ``submissions:write`` and the separate ``moto:retrieve`` /
    ``moto:attest`` corpus scopes are never equivalent to proof-trade scopes.
    """
    reasons = set()
    received = set()
    for scope in granted:
        if not isinstance(scope, str) or not scope or scope != scope.strip():
            reasons.add(SCOPE_MALFORMED)
            continue
        received.add(scope)
    required = set(PROOF_TRADE_SCOPES)
    if required - received:
        reasons.add(SCOPE_MISSING)
    if received - required:
        reasons.add(SCOPE_UNEXPECTED)
    return tuple(sorted(reasons))


def proof_trade_credential_record_name(client_id: str, installation_id: str) -> str:
    """Return the distinct keyring record name for a client + installation pair."""
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("A registered client ID is required.")
    if not isinstance(installation_id, str) or not installation_id.strip():
        raise ValueError("A stable installation ID is required.")
    client_digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]
    return f"{PROOF_TRADE_CREDENTIAL_PREFIX}::{client_digest}::{installation_id}"


def _frozen_asset_reasons() -> Tuple[bool, Tuple[str, ...]]:
    """Return (profile_published, reasons) from the presence-only asset inventory."""
    from backend.shared import syntheticlib_trade_compatibility as compatibility

    try:
        inventory = compatibility.inventory_frozen_assets()
    except compatibility.CompatibilityContractError:
        return False, (REASON_ASSET_INVENTORY_INVALID,)
    except OSError:
        return False, (REASON_ASSET_INVENTORY_INVALID,)
    profile_published = bool(inventory.get("nativeClientProfile", {}).get("available"))
    reasons = []
    if not profile_published:
        reasons.append(REASON_PROFILE_UNPUBLISHED)
    if all(bool(item.get("available")) for item in inventory.values()):
        # Assets alone never enable authorization; Build 8 must register them.
        reasons.append(REASON_CONTRACT_NOT_REGISTERED)
    else:
        reasons.append(REASON_CONTRACT_UNPUBLISHED)
    return profile_published, tuple(reasons)


def _installation_state() -> Tuple[bool, Tuple[str, ...]]:
    from backend.shared import installation_identity

    try:
        binding = installation_identity.read_installation_binding()
    except Exception as exc:  # copied/moved/foreign roots or keyring failure
        logger.info(
            "Proof-trade installation identity unavailable (%s)", type(exc).__name__
        )
        return False, (REASON_INSTALLATION_UNAVAILABLE,)
    if binding is None:
        return False, (REASON_INSTALLATION_UNBOUND,)
    return True, ()


@dataclass(frozen=True)
class AuthorizationReadiness:
    readiness_version: str = AUTHORIZATION_READINESS_VERSION
    pkce_method: str = PKCE_METHOD
    redirect_policy: str = REDIRECT_POLICY
    required_scopes: Tuple[str, ...] = PROOF_TRADE_SCOPES
    profile_published: bool = False
    keyring_available: bool = False
    installation_bound: bool = False
    authorization_enabled: bool = False
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)


def get_authorization_readiness() -> AuthorizationReadiness:
    """Return metadata-only readiness; never creates identity, files, or credentials."""
    if system_config.generic_mode:
        return AuthorizationReadiness(reason_codes=(REASON_GENERIC_MODE,))
    reasons = []
    profile_published, asset_reasons = _frozen_asset_reasons()
    reasons.extend(asset_reasons)
    keyring_available = secret_store.keyring_backend_available()
    if not keyring_available:
        reasons.append(REASON_KEYRING_UNAVAILABLE)
    installation_bound = False
    if keyring_available:
        installation_bound, installation_reasons = _installation_state()
        reasons.extend(installation_reasons)
    else:
        reasons.append(REASON_INSTALLATION_UNAVAILABLE)
    return AuthorizationReadiness(
        profile_published=profile_published,
        keyring_available=keyring_available,
        installation_bound=installation_bound,
        reason_codes=tuple(sorted(set(reasons))),
    )
