from pathlib import Path

from tests.workflow_browser_smoke.coverage_records import BROWSER_SMOKE_COVERAGE
from tests.workflow_harness.coverage_metadata import assert_coverage_records_valid


def test_browser_smoke_coverage_metadata_is_valid():
    assert_coverage_records_valid(BROWSER_SMOKE_COVERAGE)


def test_passed_browser_coverage_links_exact_playwright_tests_and_invariants():
    for record in BROWSER_SMOKE_COVERAGE:
        assert record.runner == "playwright"
        assert len(record.test_selectors) == 1
        assert record.test_selectors[0].startswith(f"{record.test_file}::")
        assert set(record.asserted_invariants) == set(record.invariants)


def test_every_browser_smoke_spec_has_exactly_one_coverage_record():
    smoke_dir = Path(__file__).resolve().parent
    repository_root = smoke_dir.parents[1]
    expected_test_files = {
        path.relative_to(repository_root).as_posix()
        for path in smoke_dir.glob("*.spec.js")
    }
    represented_test_files = [record.test_file for record in BROWSER_SMOKE_COVERAGE]

    assert len(represented_test_files) == len(set(represented_test_files))
    assert set(represented_test_files) == expected_test_files
    assert all((repository_root / test_file).is_file() for test_file in represented_test_files)
