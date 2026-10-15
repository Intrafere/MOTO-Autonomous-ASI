from __future__ import annotations

from pathlib import Path

from tests.workflow_harness.retry_wait_inventory import (
    WaitClassification,
    build_retry_wait_inventory,
    format_wait_site,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_production_retry_wait_inventory_classifies_every_wait() -> None:
    inventory = build_retry_wait_inventory(REPO_ROOT)
    assert not inventory.unclassified and not inventory.stale_policies, (
        "Retry-wait inventory has unclassified or stale entries:\n"
        + "\n".join(format_wait_site(site) for site in inventory.unclassified)
        + "\nStale selectors:\n"
        + "\n".join(repr(policy.selector) for policy in inventory.stale_policies)
    )
    assert inventory.assured_retries, "The inventory must not pass without an assured retry source."
    assert any(
        site.classification is WaitClassification.NON_RETRY
        for site in inventory.sites
    ), "The inventory must explicitly distinguish non-retry waits."


def test_all_retry_wait_sources_accounted_are_assured() -> None:
    inventory = build_retry_wait_inventory(REPO_ROOT)
    assert inventory.complete_and_assured, "\n".join(
        format_wait_site(site) for site in inventory.unassured_retries
    )


def test_assured_retry_waits_name_the_activity_seam() -> None:
    inventory = build_retry_wait_inventory(REPO_ROOT)
    for site in inventory.assured_retries:
        assert site.assurance
        assert "provider_retry" in site.assurance or "emit_and_wait_for_retry" in site.assurance
