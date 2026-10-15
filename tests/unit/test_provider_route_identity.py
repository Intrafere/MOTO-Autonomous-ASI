"""Pure helper tests for provider/API-error route identity.

Route identity (workflow mode, role, model, provider and host) is display and
structured-field data only. These tests pin the labels and host segments, and
prove that attaching identity to typed provider errors never changes their
exception text or the text classifiers that consume it.
"""
from __future__ import annotations

import httpx
import pytest

from backend.compiler.core.compiler_coordinator import _classify_submitter_error
from backend.shared.api_client_manager import ProviderCooldownError, RetryableProviderError
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_provider_context_length_error,
    is_transient_model_call_error,
)
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)
from backend.shared.provider_route_identity import (
    NON_MODEL_PROVIDERS,
    ROUTE_IDENTITY_FIELD_NAMES,
    ROUTE_IDENTITY_MAX_CHARS,
    format_host_segment,
    host_fields,
    merge_route_identity,
    role_label,
    route_identity_fields,
    route_identity_from_error,
    with_route_identity,
    workflow_label,
)

MODEL = "vendor/test-free:free"
SUBMITTER_2_AUTO = "[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter"


def _fields(**overrides):
    values = {
        "workflow_mode": "autonomous",
        "role_id": "aggregator_submitter_2",
        "model": MODEL,
        "provider": "openrouter",
    }
    values.update(overrides)
    return route_identity_fields(**values)


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        ("autonomous", "Autonomous"),
        ("aggregator", "Aggregator"),
        ("compiler", "Compiler"),
        ("leanoj", "LeanOJ"),
        ("manual_proof_check", "Manual proof check"),
        ("AUTONOMOUS", "Autonomous"),
        ("", ""),
        (None, ""),
    ],
)
def test_workflow_label(mode, label):
    assert workflow_label(mode) == label


def test_route_identity_fields_drop_invalid_workflow_mode():
    fields = _fields(workflow_mode="manual_compiler", configured_host_provider="")

    assert "workflow_mode" not in fields
    assert fields["route_identity"].startswith("[Submitter 2 · ")


@pytest.mark.parametrize(
    ("role_id", "label"),
    [
        ("aggregator_submitter_1", "Submitter 1"),
        ("aggregator_submitter_2", "Submitter 2"),
        ("aggregator_submitter_10", "Submitter 10"),
        ("agg_sub3", "Submitter 3"),
        ("aggregator_validator", "Validator"),
        ("compiler_validator", "Validator"),
        ("compiler_writer", "Writing Submitter"),
        ("compiler_high_param", "Rigor & Proofs"),
        ("critique_validator", "Critique Validator"),
        ("paper_critic", "Paper Critic"),
        ("aggregator_single_model", "Single-Model Submitter"),
        ("rag_vector_search", "RAG Vector Search"),
        ("embeddings", "Embeddings"),
        ("aggregator_assistant", "Assistant (aggregator)"),
        ("autonomous_assistant", "Assistant (autonomous)"),
        ("compiler_assistant", "Assistant (compiler)"),
        ("leanoj_assistant", "Assistant (LeanOJ)"),
        ("manual_proof_assistant", "Assistant (manual proof)"),
        ("autonomous_proof_formalization_brainstorm", "Proof Formalization (brainstorm)"),
        ("autonomous_proof_novelty_paper_0123456789", "Proof Novelty (paper)"),
        (
            "autonomous_proof_candidate_list_validator_brainstorm_abcdef0123",
            "Proof Candidate List Validator (brainstorm)",
        ),
        ("autonomous_proof_prune_proposer_brainstorm_run_9", "Proof Prune Proposer (brainstorm)"),
        ("autonomous_proof_lemma_search_paper_dep", "Proof Lemma Search (paper)"),
        (
            "autonomous_proof_formalization_compiler_manual_paper_run_1",
            "Proof Formalization (manual paper)",
        ),
        ("autonomous_proof_identification_manual_brainstorm", "Proof Identification (manual brainstorm)"),
        ("autonomous_proof_framing_gate", "Proof Framing Gate"),
        (
            "autonomous_proof_formalization_brainstorm_competition_2_0123456789abcdef",
            "Proof Formalization (brainstorm, competitor 2)",
        ),
        ("compiler_rigor_secondary_0123456789ab_1", "Rigor Secondary Formalizer 1"),
        ("openrouter_direct", "direct call"),
        ("lm_studio_direct", "direct call"),
        ("openai_codex_direct", "direct call"),
        ("autonomous_topic_validator", "Topic Validator"),
        ("autonomous_completion_reviewer_0123abcdef", "Completion Reviewer"),
        ("leanoj_brainstorm_submitter_2", "Brainstorm Submitter 2"),
        ("leanoj_final_solver", "Final Solver"),
        ("compiler_critique", "Critique"),
        ("compiler", "Compiler"),
        ("", ""),
        (None, ""),
    ],
)
def test_role_label_for_primary_route(role_id, label):
    assert role_label(role_id) == label
    assert role_label(role_id, "primary") == label


@pytest.mark.parametrize(
    ("route_kind", "submitter_label", "proof_label"),
    [
        ("", "Submitter 2", "Proof Formalization (brainstorm)"),
        ("primary", "Submitter 2", "Proof Formalization (brainstorm)"),
        ("boost", "Submitter 2 (boost)", "Proof Formalization (brainstorm, boost)"),
        (
            "fallback",
            "Submitter 2 (LM Studio fallback)",
            "Proof Formalization (brainstorm, LM Studio fallback)",
        ),
        (
            "lm_studio_fallback",
            "Submitter 2 (LM Studio fallback)",
            "Proof Formalization (brainstorm, LM Studio fallback)",
        ),
        (
            "free_rotation",
            "Submitter 2 (free rotation)",
            "Proof Formalization (brainstorm, free rotation)",
        ),
        (
            "auto_selector",
            "Submitter 2 (auto-selector)",
            "Proof Formalization (brainstorm, auto-selector)",
        ),
        ("embeddings", "Submitter 2 (embeddings)", "Proof Formalization (brainstorm, embeddings)"),
        ("not_a_route_kind", "Submitter 2", "Proof Formalization (brainstorm)"),
    ],
)
def test_role_label_names_non_primary_route_kinds(route_kind, submitter_label, proof_label):
    assert role_label("aggregator_submitter_2", route_kind) == submitter_label
    assert role_label("autonomous_proof_formalization_brainstorm", route_kind) == proof_label


def test_role_label_does_not_repeat_a_route_kind_matching_the_role():
    assert role_label("embeddings", "embeddings") == "Embeddings"


def test_aggregator_and_autonomous_assistants_are_distinguishable():
    shared = {
        "workflow_mode": "autonomous",
        "model": "vendor/test-ultra:free",
        "provider": "openrouter",
        "configured_host_provider": "",
    }
    aggregator = route_identity_fields(role_id="aggregator_assistant", **shared)
    autonomous = route_identity_fields(role_id="autonomous_assistant", **shared)

    assert aggregator["role_label"] == "Assistant (aggregator)"
    assert autonomous["role_label"] == "Assistant (autonomous)"
    assert aggregator["route_identity"] != autonomous["route_identity"]
    assert role_label("autonomous_assistant", "boost") == "Assistant (autonomous, boost)"


def test_model_role_label_replaces_display_pseudo_role():
    fields = route_identity_fields(
        workflow_mode="aggregator",
        role_id="aggregator_single_model",
        model_role_id="aggregator_submitter_2",
        model=MODEL,
        provider="openrouter",
        configured_host_provider="",
        upstream_host_provider="TestHost",
    )

    assert fields["role_id"] == "aggregator_single_model"
    assert fields["model_role_id"] == "aggregator_submitter_2"
    assert fields["role_label"] == "Submitter 2"
    assert fields["route_identity"].startswith("[Aggregator · Submitter 2 · ")
    same = route_identity_fields(role_id="compiler_writer", model_role_id="compiler_writer")
    assert "model_role_id" not in same


def test_pinned_host():
    fields = _fields(configured_host_provider="TestHost")

    assert fields["host_routing"] == "pinned"
    assert fields["configured_host_provider"] == "TestHost"
    assert fields["effective_host_provider"] == "TestHost"
    assert fields["route_identity"] == f"{SUBMITTER_2_AUTO} · host TestHost (pinned)]"


def test_pinned_host_with_different_upstream_report():
    fields = _fields(configured_host_provider="TestHost", upstream_host_provider="OtherHost")

    assert fields["host_routing"] == "pinned"
    assert fields["effective_host_provider"] == "OtherHost"
    assert fields["route_identity"].endswith("host TestHost (pinned) → OtherHost]")


def test_auto_host_with_reported_upstream():
    fields = _fields(configured_host_provider="", upstream_host_provider="TestHost")

    assert fields["host_routing"] == "auto"
    assert fields["configured_host_provider"] == ""
    assert fields["effective_host_provider"] == "TestHost"
    assert fields["route_identity"] == f"{SUBMITTER_2_AUTO} · host auto → TestHost]"


def test_auto_host_without_upstream_names_request_id():
    fields = _fields(configured_host_provider="", provider_request_id="gen-test")

    assert fields["host_routing"] == "auto"
    assert "effective_host_provider" not in fields
    assert fields["provider_request_id"] == "gen-test"
    assert fields["route_identity"] == (
        f"{SUBMITTER_2_AUTO} · host auto (upstream not reported; request gen-test)]"
    )


def test_auto_host_without_upstream_or_request_id():
    fields = _fields(configured_host_provider="")

    assert fields["route_identity"] == f"{SUBMITTER_2_AUTO} · host auto (upstream not reported)]"


def test_auto_fallback_host_with_and_without_upstream():
    reported = _fields(
        configured_host_provider="PinnedHost",
        auto_fallback_from_host="PinnedHost",
        upstream_host_provider="TestHost",
    )
    unreported = _fields(
        configured_host_provider="PinnedHost",
        auto_fallback_from_host="PinnedHost",
        provider_request_id="gen-test",
    )

    assert reported["host_routing"] == "auto_fallback"
    assert reported["configured_host_provider"] == "PinnedHost"
    assert reported["effective_host_provider"] == "TestHost"
    assert reported["route_identity"].endswith(
        "host auto (pinned PinnedHost unavailable) → TestHost]"
    )
    assert unreported["host_routing"] == "auto_fallback"
    assert unreported["route_identity"].endswith(
        "host auto (pinned PinnedHost unavailable; upstream not reported; request gen-test)]"
    )


def test_legacy_unknown_host_reports_no_host_segment():
    fields = _fields()

    assert "host_routing" not in fields
    assert "configured_host_provider" not in fields
    assert "effective_host_provider" not in fields
    assert fields["route_identity"] == f"{SUBMITTER_2_AUTO}]"
    assert format_host_segment({"configured_host_provider": "TestHost"}) == ""
    assert format_host_segment({}) == ""


def test_upstream_report_without_configured_host_is_auto():
    fields = _fields(upstream_host_provider="TestHost")

    assert fields["host_routing"] == "auto"
    assert fields["route_identity"].endswith("host auto → TestHost]")


@pytest.mark.parametrize(
    ("provider", "label"),
    [
        ("openai_codex_oauth", "OpenAI Codex"),
        ("xai_grok_oauth", "xAI Grok"),
        ("sakana_fugu", "Sakana Fugu"),
        ("lm_studio", "LM Studio"),
    ],
)
def test_non_openrouter_routes_are_direct_without_host_segment(provider, label):
    fields = route_identity_fields(
        workflow_mode="compiler",
        role_id="compiler_writer",
        model="gpt-test-codex",
        provider=provider,
        configured_host_provider="TestHost",
        upstream_host_provider="Groq",
        auto_fallback_from_host="PinnedHost",
    )

    assert fields["host_routing"] == "direct"
    assert "configured_host_provider" not in fields
    assert "effective_host_provider" not in fields
    assert fields["provider_label"] == label
    assert fields["route_identity"] == (
        f"[Compiler · Writing Submitter · gpt-test-codex via {label}]"
    )
    assert host_fields(provider, configured_host="TestHost") == {"host_routing": "direct"}


@pytest.mark.parametrize("provider", sorted(NON_MODEL_PROVIDERS))
def test_non_model_providers_produce_mode_and_role_only(provider):
    fields = route_identity_fields(
        workflow_mode="aggregator",
        role_id="aggregator_submitter_1",
        provider=provider,
        provider_label_text="Aggregator submitter 1",
        configured_host_provider="",
    )

    assert fields["route_identity"] == "[Aggregator · Submitter 1]"
    assert "model" not in fields
    assert "configured_host_provider" not in fields
    assert "effective_host_provider" not in fields
    assert fields.get("host_routing") in (None, "direct")


def test_placeholder_providers_never_report_host_fields():
    for provider in ("", "unknown", "configured_role", "provider"):
        assert host_fields(provider, configured_host="", upstream_host="TestHost") == {}


def test_route_identity_fields_only_emit_known_field_names():
    fields = route_identity_fields(
        workflow_mode="autonomous",
        role_id="aggregator_single_model",
        model_role_id="aggregator_submitter_2",
        model=MODEL,
        provider="openrouter",
        provider_label_text="OpenRouter",
        route_kind="free_rotation",
        task_id="agg_sub2_001",
        configured_model=MODEL,
        configured_provider="openrouter",
        configured_host_provider="",
        upstream_host_provider="TestHost",
        provider_request_id="gen-test",
    )

    assert set(fields) <= set(ROUTE_IDENTITY_FIELD_NAMES)
    assert fields["role_label"] == "Submitter 2 (free rotation)"
    assert route_identity_fields() == {}


def test_route_identity_is_redacted_and_capped():
    secret = "sk-or-v1-" + "a" * 48
    redacted = _fields(model=f"vendor/model {secret}", configured_host_provider="")
    capped = _fields(
        role_id="aggregator_submitter_2" * 20,
        model="vendor/" + "x" * 600,
        configured_host_provider="",
        upstream_host_provider="H" * 300,
    )

    assert secret not in redacted["route_identity"]
    assert secret not in redacted["model"]
    assert len(capped["route_identity"]) <= ROUTE_IDENTITY_MAX_CHARS
    assert capped["route_identity"].startswith("[Autonomous")
    assert capped["route_identity"].endswith("...]")


def test_with_route_identity_prefixes_once():
    fields = _fields(configured_host_provider="", upstream_host_provider="TestHost")
    identity = fields["route_identity"]
    message = with_route_identity("OpenRouter API error.", fields)

    assert message == f"{identity} OpenRouter API error."
    assert with_route_identity(message, fields) == message
    assert with_route_identity(message, identity) == message
    embedded = f"Retry notice: {identity} OpenRouter API error."
    assert with_route_identity(embedded, fields) == embedded
    assert with_route_identity("  plain  ", {}) == "plain"
    assert with_route_identity("plain", None) == "plain"
    assert with_route_identity(None, fields) == identity


def test_with_route_identity_is_idempotent_after_truncation_and_redaction():
    secret = "sk-or-v1-" + "b" * 48
    fields = _fields(
        model=f"vendor/{'y' * 600} {secret}",
        configured_host_provider="",
        upstream_host_provider="TestHost",
    )
    once = with_route_identity("OpenRouter API error.", fields)
    twice = with_route_identity(once, fields)

    assert once == twice
    assert once.count(fields["route_identity"]) == 1
    assert secret not in once


def test_merge_route_identity_never_overrides_producer_keys():
    fields = _fields(configured_host_provider="", upstream_host_provider="TestHost")
    merged = merge_route_identity(
        {
            "workflow_mode": "compiler",
            "role_id": "compiler_critique",
            "model": "",
            "provider": "configured_role",
            "message": "producer message",
        },
        fields,
    )

    assert merged["workflow_mode"] == "compiler"
    assert merged["role_id"] == "compiler_critique"
    assert merged["model"] == MODEL
    assert merged["provider"] == "openrouter"
    assert merged["message"] == "producer message"
    assert merged["route_identity"] == fields["route_identity"]
    assert "configured_host_provider" not in merged
    assert merge_route_identity({"provider": "chromadb"}, {"provider": "openrouter"}) == {
        "provider": "chromadb",
    }


def _no_choices_route_error(**route_overrides) -> ProviderRouteError:
    route = {
        "provider": "openrouter",
        "model": MODEL,
        "role_id": "aggregator_submitter_2",
        "task_id": "agg_sub2_001",
    }
    route.update(route_overrides)
    return ProviderRouteError(
        "OpenRouter returned no completion choices after 3 attempts.",
        route=ProviderRouteIdentity(**route),
        cause=httpx.ReadError("OpenRouter upstream response omitted completion choices"),
    )


def test_route_identity_from_provider_route_error():
    error = _no_choices_route_error(
        route_kind="free_rotation",
        configured_provider="openrouter",
        configured_model="vendor/configured:free",
        upstream_host_provider="TestHost",
        provider_request_id="gen-test",
    )

    found = route_identity_from_error(error)

    assert found == {
        "configured_host_provider": "",
        "provider": "openrouter",
        "model": MODEL,
        "role_id": "aggregator_submitter_2",
        "task_id": "agg_sub2_001",
        "route_kind": "free_rotation",
        "configured_provider": "openrouter",
        "configured_model": "vendor/configured:free",
        "upstream_host_provider": "TestHost",
        "provider_request_id": "gen-test",
    }
    composed = route_identity_fields(workflow_mode="autonomous", **found)
    assert composed["route_identity"] == (
        "[Autonomous · Submitter 2 (free rotation) · vendor/test-free:free via OpenRouter"
        " · host auto → TestHost]"
    )


def test_route_identity_from_auto_fallback_route_error():
    error = _no_choices_route_error(
        auto_fallback_from_host="PinnedHost",
        upstream_host_provider="TestHost",
    )

    found = route_identity_from_error(error)
    composed = route_identity_fields(workflow_mode="autonomous", **found)

    assert found["auto_fallback_from_host"] == "PinnedHost"
    assert composed["host_routing"] == "auto_fallback"
    assert composed["configured_host_provider"] == "PinnedHost"


def test_route_identity_from_context_length_error():
    error = ProviderContextLengthError(
        "LM Studio rejected the request because its input exceeded the provider context limit.",
        route=ProviderRouteIdentity(
            provider="lm_studio",
            model="local-test-model",
            role_id="compiler_writer",
            route_kind="fallback",
        ),
    )

    composed = route_identity_fields(workflow_mode="compiler", **route_identity_from_error(error))

    assert composed["route_identity"] == (
        "[Compiler · Writing Submitter (LM Studio fallback) · local-test-model via LM Studio]"
    )
    assert composed["host_routing"] == "direct"


def test_route_identity_from_repair_required_error():
    unknown_host = ProviderRepairRequiredError(
        provider="xai_grok_oauth",
        provider_label="xAI Grok",
        role_id="autonomous_assistant",
        model="grok-test",
        reason="spending_limit_reached",
        message="xAI Grok requires repair.",
        task_id="assistant_001",
    )
    auto_host = ProviderRepairRequiredError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="compiler_writer",
        model=MODEL,
        reason="credit_exhaustion",
        message="OpenRouter requires repair.",
        configured_host_provider="",
        upstream_host_provider="Groq",
    )

    unknown_found = route_identity_from_error(unknown_host)
    auto_found = route_identity_from_error(auto_host)

    assert "configured_host_provider" not in unknown_found
    assert unknown_found["provider_label_text"] == "xAI Grok"
    assert unknown_found["task_id"] == "assistant_001"
    assert route_identity_fields(**unknown_found)["host_routing"] == "direct"
    assert auto_found["configured_host_provider"] == ""
    assert route_identity_fields(**auto_found)["route_identity"].endswith("host auto → Groq]")


def test_route_identity_from_retryable_provider_error():
    error = RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="aggregator_submitter_2",
        model=MODEL,
        reason="transient_provider_error",
        message="OpenRouter transient provider failure",
        route_kind="boost",
        task_id="agg_sub2_001",
        configured_provider="openrouter",
        configured_model="vendor/configured:free",
        configured_host_provider="",
        upstream_host_provider="TestHost",
        provider_request_id="gen-test",
    )
    workflow_loop = RetryableProviderError(
        provider="unknown",
        provider_label="Aggregator submitter 1",
        role_id="aggregator_submitter_1",
        model="",
        reason="submitter_loop_retry",
        message="loop retry",
    )

    composed = route_identity_fields(workflow_mode="aggregator", **route_identity_from_error(error))
    loop_found = route_identity_from_error(workflow_loop)

    assert composed["role_label"] == "Submitter 2 (boost)"
    assert composed["route_identity"].endswith("host auto → TestHost]")
    assert "provider" not in loop_found
    assert route_identity_fields(workflow_mode="aggregator", **loop_found)["route_identity"] == (
        "[Aggregator · Submitter 1]"
    )


def test_route_identity_from_cause_chain_prefers_outermost_and_fills_gaps():
    inner = _no_choices_route_error(upstream_host_provider="TestHost", provider_request_id="gen-test")
    outer = RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="aggregator_single_model",
        model=MODEL,
        reason="transient_provider_error",
        message="wrapped",
        route_kind="primary",
    )
    outer.__cause__ = inner
    wrapper = RuntimeError("generic wrapper")
    wrapper.__cause__ = inner

    outer_found = route_identity_from_error(outer)
    wrapper_found = route_identity_from_error(wrapper)

    assert outer_found["role_id"] == "aggregator_single_model"
    assert outer_found["upstream_host_provider"] == "TestHost"
    assert outer_found["provider_request_id"] == "gen-test"
    assert outer_found["task_id"] == "agg_sub2_001"
    assert wrapper_found["role_id"] == "aggregator_submitter_2"
    assert wrapper_found["model"] == MODEL


def test_route_identity_from_error_is_bounded_and_cycle_safe():
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    deep: BaseException = _no_choices_route_error()
    for index in range(6):
        wrapper = RuntimeError(f"wrapper {index}")
        wrapper.__cause__ = deep
        deep = wrapper

    assert route_identity_from_error(first) == {}
    assert route_identity_from_error(None) == {}
    assert route_identity_from_error(deep) == {}
    assert route_identity_from_error(deep, max_depth=10)["provider"] == "openrouter"


# Upstream hosts and configured models whose names look like provider markers
# (xAI, Groq, grok, codex) must stay out of exception text and classifiers.
_IDENTITY_ROUTE = {
    "upstream_host_provider": "xAI",
    "auto_fallback_from_host": "Groq",
    "provider_request_id": "gen-grok-codex",
    "configured_provider": "xai_grok_oauth",
    "configured_model": "x-ai/grok-codex-test",
}
_IDENTITY_RETRYABLE = {
    "route_kind": "free_rotation",
    "task_id": "agg_sub2_grok_001",
    "configured_provider": "openai_codex_oauth",
    "configured_model": "openai/gpt-test-codex",
    "configured_host_provider": "Groq",
    "upstream_host_provider": "xAI",
    "auto_fallback_from_host": "Groq",
    "provider_request_id": "gen-grok-codex",
}
_IDENTITY_REPAIR = {
    "configured_host_provider": "Groq",
    "upstream_host_provider": "xAI",
    "auto_fallback_from_host": "Groq",
    "task_id": "agg_sub2_grok_001",
}


def _route(model: str, **identity) -> ProviderRouteIdentity:
    return ProviderRouteIdentity(
        provider="openrouter",
        model=model,
        role_id="aggregator_submitter_2",
        **identity,
    )


def _exception_pairs():
    """Return (name, plain, with_identity) pairs for every typed provider error."""
    return [
        (
            "route_error_api_text",
            ProviderRouteError(
                "OpenRouter API error: HTTP 502 bad gateway",
                route=_route(MODEL),
                cause=ValueError("upstream refused"),
            ),
            ProviderRouteError(
                "OpenRouter API error: HTTP 502 bad gateway",
                route=_route(MODEL, **_IDENTITY_ROUTE),
                cause=ValueError("upstream refused"),
            ),
        ),
        (
            "route_error_transport_cause",
            ProviderRouteError(
                "OpenRouter returned no completion choices after 3 attempts.",
                route=_route(MODEL),
                cause=httpx.ReadError("omitted choices"),
            ),
            ProviderRouteError(
                "OpenRouter returned no completion choices after 3 attempts.",
                route=_route(MODEL, **_IDENTITY_ROUTE),
                cause=httpx.ReadError("omitted choices"),
            ),
        ),
        (
            "route_error_neutral_cause",
            ProviderRouteError(
                "Upstream service unavailable",
                route=_route(MODEL),
                cause=ValueError("Upstream service unavailable"),
            ),
            ProviderRouteError(
                "Upstream service unavailable",
                route=_route(MODEL, **_IDENTITY_ROUTE),
                cause=ValueError("Upstream service unavailable"),
            ),
        ),
        (
            "context_length_error",
            ProviderContextLengthError(
                "OpenRouter rejected the request because its input exceeded the provider context limit.",
                route=_route(MODEL),
            ),
            ProviderContextLengthError(
                "OpenRouter rejected the request because its input exceeded the provider context limit.",
                route=_route(MODEL, **_IDENTITY_ROUTE),
            ),
        ),
        (
            "repair_required_error",
            ProviderRepairRequiredError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="credit_exhaustion",
                message="Upstream service unavailable for this route.",
            ),
            ProviderRepairRequiredError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="credit_exhaustion",
                message="Upstream service unavailable for this route.",
                **_IDENTITY_REPAIR,
            ),
        ),
        (
            "retryable_neutral_text",
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="transient_provider_error",
                message="Upstream service unavailable",
            ),
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="transient_provider_error",
                message="Upstream service unavailable",
                **_IDENTITY_RETRYABLE,
            ),
        ),
        (
            "retryable_api_text",
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="transient_provider_error",
                message="OpenRouter API error: HTTP 502 bad gateway",
            ),
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model=MODEL,
                reason="transient_provider_error",
                message="OpenRouter API error: HTTP 502 bad gateway",
                **_IDENTITY_RETRYABLE,
            ),
        ),
        (
            "retryable_grok_model",
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model="x-ai/grok-test",
                reason="transient_provider_error",
                message="Request failed: model is not loaded",
            ),
            RetryableProviderError(
                provider="openrouter",
                provider_label="OpenRouter",
                role_id="aggregator_submitter_2",
                model="x-ai/grok-test",
                reason="transient_provider_error",
                message="Request failed: model is not loaded",
                **_IDENTITY_RETRYABLE,
            ),
        ),
        (
            "cooldown_error",
            ProviderCooldownError(
                provider="openai_codex_oauth",
                provider_label="OpenAI Codex",
                role_id="compiler_writer",
                model="gpt-test-codex",
                resets_at=None,
                resets_in_seconds=60,
            ),
            ProviderCooldownError(
                provider="openai_codex_oauth",
                provider_label="OpenAI Codex",
                role_id="compiler_writer",
                model="gpt-test-codex",
                resets_at=None,
                resets_in_seconds=60,
                **_IDENTITY_RETRYABLE,
            ),
        ),
    ]


_CLASSIFIERS = (
    is_transient_model_call_error,
    is_non_retryable_model_error,
    is_provider_context_length_error,
    _classify_submitter_error,
)


@pytest.mark.parametrize(
    ("name", "plain", "with_identity"),
    _exception_pairs(),
    ids=[pair[0] for pair in _exception_pairs()],
)
def test_attached_route_identity_never_changes_text_or_classification(name, plain, with_identity):
    assert str(with_identity) == str(plain)
    for classifier in _CLASSIFIERS:
        assert classifier(with_identity) == classifier(plain), classifier.__name__
    for marker in ("xAI", "Groq", "gen-grok-codex", "grok-codex-test", "gpt-test-codex"):
        if marker not in str(plain):
            assert marker not in str(with_identity)
    if isinstance(plain, RetryableProviderError):
        assert with_identity.provider == plain.provider
        assert with_identity.model == plain.model
        assert with_identity.error_summary == plain.error_summary


def test_classification_probes_are_sensitive_to_identity_text():
    """Guard the probes: leaked identity text would flip these classifications."""
    neutral = RetryableProviderError(
        provider="openrouter",
        provider_label="OpenRouter",
        role_id="aggregator_submitter_2",
        model=MODEL,
        reason="transient_provider_error",
        message="Upstream service unavailable",
    )
    api_text = RuntimeError("OpenRouter API error: HTTP 502 bad gateway")

    assert is_transient_model_call_error(neutral) is False
    assert is_transient_model_call_error(RuntimeError("[xAI Grok] Upstream service unavailable"))
    assert _classify_submitter_error(api_text)[0] == "API transport error"
    assert _classify_submitter_error(
        RuntimeError(f"{SUBMITTER_2_AUTO}] OpenRouter API error: HTTP 502 bad gateway")
    )[0] == "Submitter error"


def test_route_identity_does_not_change_safe_summary():
    plain = _route(MODEL, host_provider="TestHost", route_kind="primary")
    with_identity = _route(
        MODEL,
        host_provider="TestHost",
        route_kind="primary",
        **_IDENTITY_ROUTE,
    )

    assert with_identity.safe_summary() == plain.safe_summary()
    assert with_identity.safe_summary() == (
        "provider=openrouter, model=vendor/test-free:free, route=primary, host=TestHost, "
        "role=aggregator_submitter_2"
    )
