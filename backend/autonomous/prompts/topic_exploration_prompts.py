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
    parts.append("that maximize the chance of the strongest credible, genuinely novel, and rigorous DIRECT solution to the research goal below")
    parts.append("by aiming the brainstorm at the right territory, NOT by deciding its answer in advance.\n")
    parts.append("First prefer candidate questions that aggressively attack the user's WHOLE question as stated.")
    parts.append("If the whole question cannot be attacked in one shot,")
    parts.append("propose the next best necessary piece, or another genuinely novel incremental or indirect route, whose answer")
    parts.append("would plausibly advance the original question. Indirect/support avenues must be genuinely novel and")
    parts.append("plausibly advance that full-question route.\n")
    parts.append("Each submission should contain ONE candidate brainstorm question and reasoning for why")
    parts.append("it is a valuable, distinct direction. The validator will check quality, DIVERSITY, and OPENNESS —")
    parts.append("candidates that overlap with already-accepted ones, or that lock the brainstorm into a presumed answer")
    parts.append("or a single predetermined solution path, will be REJECTED.\n")
    parts.append("THE BRAINSTORM IS THE SMARTEST PHASE — DO NOT PRE-SOLVE IT:")
    parts.append("The brainstorm that follows runs many parallel submitters, a validator, and repeated review rounds.")
    parts.append("That is where the real solving happens: approaches are proposed, compared, combined, and discarded there.")
    parts.append("A candidate question only aims that brainstorm; it must not decide the solution for it.")
    parts.append("- Be clear about WHAT to attack (the whole question, a decisive sub-question, or a solution arena), and leave")
    parts.append("  HOW to solve it and WHAT the answer turns out to be open")
    parts.append("- Do not embed a presumed final answer, a single predetermined construction or method, or a guaranteed")
    parts.append("  solution path in the question")
    parts.append("- Frame it so that if the most obvious approach inside it fails, the brainstorm can still pivot to other")
    parts.append("  approaches, refute the premise, or reach a different conclusion within the same topic")
    parts.append("- Prefer open forms such as \"determine whether...\", \"find or characterize...\", or \"what mechanisms or")
    parts.append("  designs could...\" over committed forms such as \"prove X via method Y\" or \"build design Z\", unless that")
    parts.append("  exact statement is the user's own question")
    parts.append("- Broad enough to hold several competing approaches, but still focused on the research goal (broad does not mean vague)\n")
    parts.append("WHAT MAKES A GOOD CANDIDATE QUESTION:")
    parts.append("- Most directly targets answering the user's whole problem")
    parts.append("- If piecewise, incremental, or indirect, targets a genuinely novel piece or route with a plausible path toward the full problem")
    parts.append("- Clearly scoped to a problem, sub-question, or solution arena, while staying open about method and answer")
    parts.append("- Novel relative to already-accepted candidates and existing brainstorms")
    parts.append("- Relevant to the research goal below")
    parts.append("- Opens a DISTINCT high-impact solution, invention, or research direction not already represented")
    parts.append("- Does not retreat to an easier adjacent/practical/background route that adds no genuine novelty toward the research goal")
    parts.append("- Points at a problem-appropriate arena — a decisive question, obstruction, family of mechanisms, design route,")
    parts.append("  class of algorithms, theorem area, experiment, or similar — without dictating which single mechanism, design,")
    parts.append("  or construction must succeed")
    parts.append("- Uses domain- and claim-appropriate rigor: mathematics, theorem discovery, proof, and formalization remain first-class whenever relevant")
    parts.append("- For empirical, artifact, or literature-dependent routes, requires honest evidence/provenance or explicit hypothesis, proposed-test, or proposed-work language")
    parts.append("- Actionable — a brainstorm session could produce meaningful progress from it, even if its first-guess approach fails\n")
    parts.append("DIVERSITY IS PARAMOUNT:")
    parts.append("Your candidate MUST be SUBSTANTIVELY DIFFERENT from already-accepted candidates.")
    parts.append("The goal is to compare the BEST distinct territories for attacking the research goal before committing to one.")
    parts.append("Candidates should differ in WHAT they attack or the angle of attack, not merely in which answer they presume.")
    parts.append("Do not propose shallow variations of existing candidates — propose genuinely different,")
    parts.append("high-value avenues with a preference for the most direct rigorous routes.\n")
    parts.append("FORMAT YOUR SUBMISSION AS:")
    parts.append("State the candidate brainstorm question clearly, then explain why it is valuable, why it is")
    parts.append("distinct from any existing candidates, and how it leaves the brainstorm room to find, compare,")
    parts.append("or reject approaches.\n")
    
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
