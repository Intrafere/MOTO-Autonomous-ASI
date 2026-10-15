import re

import pytest

from backend.shared import oauth_pkce
from backend.shared.openai_codex_client import OpenAICodexClient
from backend.shared.xai_grok_client import XAIGrokClient


def test_s256_challenge_matches_rfc7636_appendix_b_vector() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert oauth_pkce.pkce_s256_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_generated_pair_is_unpadded_url_safe_and_consistent() -> None:
    verifier, challenge = oauth_pkce.generate_pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert re.fullmatch(r"[A-Za-z0-9_-]+", verifier)
    assert "=" not in challenge
    assert challenge == oauth_pkce.pkce_s256_challenge(verifier)
    assert oauth_pkce.generate_pkce_pair()[0] != verifier


@pytest.mark.parametrize("method", ["plain", "s256", "", "S512"])
def test_only_s256_is_accepted(method) -> None:
    with pytest.raises(oauth_pkce.UnsupportedPkceMethodError):
        oauth_pkce.require_s256(method)
    assert oauth_pkce.require_s256("S256") == "S256"


def test_state_generation_and_constant_time_matching() -> None:
    state = oauth_pkce.generate_oauth_state()
    assert len(state) >= 43
    assert oauth_pkce.oauth_state_matches(state, state)
    assert not oauth_pkce.oauth_state_matches(state, state + "x")
    assert not oauth_pkce.oauth_state_matches(state, "")
    assert not oauth_pkce.oauth_state_matches("", "")
    assert not oauth_pkce.oauth_state_matches(state, None)


def test_provider_clients_share_the_single_pkce_implementation() -> None:
    assert OpenAICodexClient.generate_pkce_pair is oauth_pkce.generate_pkce_pair
    assert XAIGrokClient.generate_pkce_pair is oauth_pkce.generate_pkce_pair
