import json

import pytest

from backend.autonomous.memory.paper_library import PaperLibrary
from backend.shared.models import PaperMetadata
from backend.shared.paper_proofs import analyze_paper_content, strip_paper_proofs


APPENDIX_START = "[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]"
APPENDIX_END = "[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]"


def test_parser_preserves_order_titles_repeated_volume_sections_and_self_review():
    content = (
        "# CHAPTER 1: First\nPaper one words.\n"
        f"{APPENDIX_START}\n"
        "Theorem (proof_2) [Novel] - Ordered title\n"
        "Statement: True\nLean 4 proof:\ntheorem p2 : True := by trivial\n---\n"
        f"{APPENDIX_END}\n"
        "AI Self-Review and Limitations\nKeep this review.\n"
        f"{'#' * 80}\n# CHAPTER 2: Second\nPaper two words.\n"
        "=== PROOFS GENERATED FROM THIS PAPER (Lean 4 Verified) ===\n"
        "Theorem (proof_9) [Known] - Legacy title\n"
        "Statement: True\nLean 4 proof:\ntheorem p9 : True := by trivial\n---\n"
        f"{'#' * 80}\n# CHAPTER 3: Third\nFinal prose.\n"
        "=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===\n"
        "Theorem (proof_10) [Known] - Attached title\n"
        "Statement: True\nLean 4 proof:\ntheorem p10 : True := by trivial\n---\n"
    )

    result = analyze_paper_content(content)

    assert result["proof_count"] == 3
    assert [proof["order"] for proof in result["proofs"]] == [1, 2, 3]
    assert [proof["title"] for proof in result["proofs"]] == [
        "Ordered title",
        "Legacy title",
        "Attached title",
    ]
    assert "Keep this review." in result["paper_content"]
    assert "# CHAPTER 3: Third" in result["paper_content"]
    assert "theorem p2" not in result["paper_content"]
    assert result["total_word_count"] == (
        result["paper_word_count"] + result["proof_word_count"]
    )
    assert result["total_character_count"] == len(content)


def test_strip_keeps_compiler_markers_and_nonproof_content():
    content = (
        "Body.\n"
        f"{APPENDIX_START}\n"
        "Theorem (proof_1) - One\nLean 4 proof:\ntheorem one : True := by trivial\n"
        f"{APPENDIX_END}\n"
        "AI Self-Review and Limitations\nReview survives.\n"
    )

    stripped = strip_paper_proofs(content)

    assert APPENDIX_START in stripped
    assert APPENDIX_END in stripped
    assert "theorem one" not in stripped
    assert "Review survives." in stripped


@pytest.mark.asyncio
async def test_append_refreshes_metadata_metrics_atomically(tmp_path):
    library = PaperLibrary._build_scoped_library(tmp_path)
    await library.initialize()
    paper_id = "metrics"
    paper_path = library._get_paper_path(paper_id)
    metadata_path = library._get_metadata_path(paper_id)
    paper_path.write_text(
        f"Body words.\n{APPENDIX_START}\nplaceholder\n{APPENDIX_END}\n",
        encoding="utf-8",
    )
    await library._save_metadata(PaperMetadata(paper_id=paper_id, title="Metrics"))

    assert await library.append_proofs_section(
        paper_id,
        {
            "proof_id": "proof_1",
            "theorem_name": "Metric theorem",
            "theorem_statement": "True",
            "lean_code": "theorem metric : True := by trivial",
        },
    )

    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    current = paper_path.read_text(encoding="utf-8")
    expected = analyze_paper_content(current)
    assert saved["proof_count"] == 1
    assert saved["total_word_count"] == expected["total_word_count"]
    assert saved["proof_character_count"] == expected["proof_character_count"]
    assert not list(tmp_path.glob("*.tmp"))
