from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backend.shared import provider_route_identity
from tests.workflow_harness.provider_error_activity_inventory import (
    ACTIVITY_POLICIES,
    PROVIDER_ERROR_ACTIVITY_EVENTS,
    ActivityClassification,
    SiteKind,
    build_provider_error_activity_inventory,
    format_activity_site,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_provider_error_activity_source_is_route_attributed_or_exempt() -> None:
    inventory = build_provider_error_activity_inventory(REPO_ROOT)
    problems = [
        *(f"unclassified: {format_activity_site(site)}" for site in inventory.unclassified),
        *(f"mismatch: {format_activity_site(site)}" for site in inventory.mismatched),
        *(f"stale: {policy.selector!r}" for policy in inventory.stale_policies),
    ]
    assert not problems, "Provider-error activity inventory problems:\n" + "\n".join(problems)
    assert inventory.route_attributed
    assert inventory.attributed, "The inventory must not pass without attributed sources."
    for kind in SiteKind:
        assert any(site.kind is kind for site in inventory.sites), f"No {kind.value} site was discovered."
    for site in inventory.exempt:
        assert site.rationale.strip(), format_activity_site(site)
    for site in inventory.attributed:
        assert site.evidence, format_activity_site(site)


def test_provider_error_activity_event_set_matches_production() -> None:
    assert PROVIDER_ERROR_ACTIVITY_EVENTS == provider_route_identity.PROVIDER_ERROR_ACTIVITY_EVENTS


def test_stale_and_mislabelled_provider_error_policies_fail_closed() -> None:
    stale = replace(
        ACTIVITY_POLICIES[0],
        selector=replace(ACTIVITY_POLICIES[0].selector, occurrence=999),
    )
    inventory = build_provider_error_activity_inventory(REPO_ROOT, policies=(*ACTIVITY_POLICIES, stale))
    assert stale in inventory.stale_policies
    assert not inventory.route_attributed

    dropped = build_provider_error_activity_inventory(REPO_ROOT, policies=ACTIVITY_POLICIES[1:])
    assert ACTIVITY_POLICIES[0].selector in {site.selector for site in dropped.unclassified}
    assert not dropped.complete

    exempt_index, exempt_policy = next(
        (index, policy)
        for index, policy in enumerate(ACTIVITY_POLICIES)
        if policy.classification is ActivityClassification.EXEMPT
        and policy.selector.path.endswith("rag_manager.py")
    )
    relabelled = (
        *ACTIVITY_POLICIES[:exempt_index],
        replace(exempt_policy, classification=ActivityClassification.ATTRIBUTED),
        *ACTIVITY_POLICIES[exempt_index + 1:],
    )
    mislabelled = build_provider_error_activity_inventory(REPO_ROOT, policies=relabelled)
    assert exempt_policy.selector in {site.selector for site in mislabelled.mismatched}
    assert not mislabelled.route_attributed
