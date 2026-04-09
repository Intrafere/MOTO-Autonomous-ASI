"""
Paper Title Exploration Prompts - Builds the aggregator-compatible user prompt for
the paper title exploration phase that collects 5 validated candidate titles
before final title selection.

The exploration phase reuses the full Part 1 aggregator infrastructure (parallel
submitters, batch validation up to 3, queue management) by framing the task as
a standard aggregation with a specially crafted user prompt.
"""
from typing import List, Dict, Any, Optional


def build_title_exploration_user_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    existing_papers_from_brainstorm: List[Dict[str, Any]],
    reference_papers: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build the user prompt passed to the aggregator for paper title exploration.

    This prompt frames the aggregation task so that submitters generate candidate
    paper titles and the validator checks quality plus diversity. The standard
    aggregator submitter and validator system prompts handle the rest.

    Args:
        user_research_prompt: User's high-level research goal
        topic_prompt: Brainstorm topic, Tier 3 certainty context, or chapter brief
        brainstorm_summary: Summary of the source material the paper will be built from
        existing_papers_from_brainstorm: Related completed papers that titles must not duplicate
        reference_papers: Optional reference papers informing this paper or chapter
    """
    parts = []

    parts.append("=== PAPER TITLE EXPLORATION PHASE ===\n")
    parts.append("You are in a PAPER TITLE EXPLORATION phase. You are NOT writing the paper itself.\n")
    parts.append("Instead, your task is to propose ONE CANDIDATE PAPER TITLE per submission.")
    parts.append("The system will collect 5 validated candidate titles before a later final")
    parts.append("selection chooses the actual title.\n")
    parts.append("Each submission should contain:")
    parts.append("- One candidate paper title")
    parts.append("- Brief reasoning for why the title is strong, accurate, and distinct\n")
    parts.append("The validator will check QUALITY and DIVERSITY:")
    parts.append("- Weak, vague, or generic titles will be rejected")
    parts.append("- Titles too similar to already-accepted candidates will be rejected")
    parts.append("- Titles too similar to already-completed related papers should be rejected")
    parts.append("- The goal is to map multiple plausible title directions before committing\n")
    parts.append("WHAT MAKES A GOOD CANDIDATE TITLE:")
    parts.append("- Accurately captures the paper's likely mathematical content")
    parts.append("- Specific enough to communicate the core focus")
    parts.append("- Professional and suitable for a mathematical research paper")
    parts.append("- Distinct from already-accepted candidate titles")
    parts.append("- Distinct from related completed papers listed below")
    parts.append("- If this is a final-answer or chapter paper, the title should match that role directly\n")
    parts.append("DIVERSITY IS PARAMOUNT:")
    parts.append("Do not submit near-duplicates, minor rephrasings, or cosmetic variants.")
    parts.append("Propose genuinely different title framings, emphases, or structural approaches.\n")
    parts.append("FORMAT YOUR SUBMISSION AS:")
    parts.append("State the candidate title clearly, then explain why it is valuable and")
    parts.append("how it differs from existing accepted candidates or related papers.\n")

    parts.append(f"USER RESEARCH GOAL:\n{user_research_prompt}\n")
    parts.append(f"PAPER CONTEXT / SOURCE TOPIC:\n{topic_prompt}\n")
    parts.append(f"SOURCE MATERIAL SUMMARY:\n{brainstorm_summary}\n")

    if existing_papers_from_brainstorm:
        parts.append("\nEXISTING RELATED PAPERS (do not duplicate these title directions):")
        for paper in existing_papers_from_brainstorm:
            abstract = paper.get("abstract", "N/A")
            if isinstance(abstract, str) and len(abstract) > 300:
                abstract = abstract[:300] + "..."
            parts.append(f"  - {paper.get('paper_id', 'N/A')}: \"{paper.get('title', 'N/A')}\"")
            parts.append(f"    Abstract: {abstract}")
    else:
        parts.append("\nEXISTING RELATED PAPERS: None")

    if reference_papers:
        parts.append("\nREFERENCE PAPERS INFORMING THIS TITLE:")
        for paper in reference_papers:
            abstract = paper.get("abstract", "N/A")
            if isinstance(abstract, str) and len(abstract) > 220:
                abstract = abstract[:220] + "..."
            parts.append(f"  - {paper.get('paper_id', 'N/A')}: \"{paper.get('title', 'N/A')}\"")
            parts.append(f"    Abstract: {abstract}")

    return "\n".join(parts)
