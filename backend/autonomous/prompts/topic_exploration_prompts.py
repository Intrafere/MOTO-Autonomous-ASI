"""
Topic Exploration Prompts - Builds the aggregator-compatible user prompt for the
topic exploration phase that collects 5 validated candidate brainstorm questions
before topic selection.

The exploration phase reuses the full Part 1 aggregator infrastructure (parallel
submitters, batch validation up to 3, queue management) by framing the task as
a standard aggregation with a specially crafted user prompt.
"""
from typing import List, Dict, Any


def build_exploration_user_prompt(
    user_research_prompt: str,
    brainstorms_summary: List[Dict[str, Any]],
    papers_summary: List[Dict[str, Any]]
) -> str:
    """
    Build the user prompt passed to the aggregator for topic exploration.
    
    This prompt frames the aggregation task so that submitters generate candidate
    brainstorm questions and the validator checks quality + diversity. The standard
    aggregator submitter/validator system prompts handle the rest.
    
    Args:
        user_research_prompt: User's high-level research goal
        brainstorms_summary: All existing brainstorms with metadata
        papers_summary: All completed papers with title/abstract
    """
    parts = []
    
    parts.append("=== TOPIC EXPLORATION PHASE ===\n")
    parts.append("You are in a TOPIC EXPLORATION phase. Your task is to propose CANDIDATE BRAINSTORM QUESTIONS")
    parts.append("that maximize the chance of the strongest credible, genuinely novel, and rigorous DIRECT solution to the research goal below.\n")
    parts.append("First prefer candidate questions that aggressively attack the user's WHOLE question as stated,")
    parts.append("no partial solutions. If the whole question cannot be attacked in one shot,")
    parts.append("propose the next best necessary piece whose answer")
    parts.append("would visibly advance the original question. Use indirect/support avenues only when they are")
    parts.append("clearly required for that full-question route.\n")
    parts.append("Each submission should contain ONE candidate brainstorm question and reasoning for why")
    parts.append("it is a valuable, distinct direction. The validator will check quality and DIVERSITY —")
    parts.append("candidates that overlap with already-accepted ones will be REJECTED.\n")
    parts.append("WHAT MAKES A GOOD CANDIDATE QUESTION:")
    parts.append("- Most directly targets answering the user's whole problem")
    parts.append("- If piecewise, targets a clearly necessary piece of the full problem")
    parts.append("- Specific enough to guide focused solution seeking (not vague)")
    parts.append("- Novel relative to already-accepted candidates and existing brainstorms")
    parts.append("- Relevant to the research goal below")
    parts.append("- Opens a DISTINCT high-impact solution, invention, or research direction not already represented")
    parts.append("- Does not retreat to an easier adjacent/practical/background route when a direct whole-question route is available")
    parts.append("- Identifies a concrete mechanism, decisive question, obstruction, design route, algorithm, theorem, experiment, or other problem-appropriate path")
    parts.append("- Uses domain- and claim-appropriate rigor: mathematics, theorem discovery, proof, and formalization remain first-class whenever relevant")
    parts.append("- For empirical, artifact, or literature-dependent routes, requires honest evidence/provenance or explicit hypothesis, proposed-test, or proposed-work language")
    parts.append("- Actionable — a brainstorm session could produce meaningful progress from it\n")
    parts.append("DIVERSITY IS PARAMOUNT:")
    parts.append("Your candidate MUST be SUBSTANTIVELY DIFFERENT from already-accepted candidates.")
    parts.append("The goal is to compare the BEST direct-answer paths before committing to one.")
    parts.append("Do not propose shallow variations of existing candidates — propose genuinely different,")
    parts.append("high-value avenues with a preference for the most direct rigorous routes.\n")
    parts.append("FORMAT YOUR SUBMISSION AS:")
    parts.append("State the candidate brainstorm question clearly, then explain why it is valuable and")
    parts.append("distinct from any existing candidates.\n")
    
    parts.append(f"RESEARCH GOAL:\n{user_research_prompt}\n")
    
    # Existing brainstorms
    if brainstorms_summary:
        parts.append("\nEXISTING BRAINSTORM TOPICS (already explored or in progress):")
        for b in brainstorms_summary:
            parts.append(f"  - {b.get('topic_id', 'N/A')}: {b.get('topic_prompt', 'N/A')} "
                        f"(status: {b.get('status', 'N/A')}, submissions: {b.get('submission_count', 0)}, "
                        f"papers: {b.get('papers_generated', 0)})")
    else:
        parts.append("\nEXISTING BRAINSTORM TOPICS: None yet")
    
    # Existing papers
    if papers_summary:
        parts.append("\nCOMPLETED PAPERS:")
        for p in papers_summary:
            abstract = p.get('abstract', 'N/A')
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            parts.append(f"  - {p.get('paper_id', 'N/A')}: \"{p.get('title', 'N/A')}\"")
            parts.append(f"    Abstract: {abstract}")
    else:
        parts.append("\nCOMPLETED PAPERS: None yet")
    
    return "\n".join(parts)
