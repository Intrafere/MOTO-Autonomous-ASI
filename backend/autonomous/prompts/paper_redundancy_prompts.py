"""
Paper Redundancy Prompts - System prompts for paper library redundancy review.
Runs every 3 completed papers to maintain library quality.
"""
from typing import List, Dict, Any


def get_paper_redundancy_system_prompt() -> str:
    """Get system prompt for paper redundancy review."""
    return """You are performing a quality maintenance review of the paper library. Your role is to:

1. Review all completed papers (titles and abstracts)
2. Identify if ANY ONE paper should be removed due to redundancy
3. Maintain a high-quality, non-redundant paper library

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Apply the rigor and evidence standard appropriate to each domain and claim type. Mathematical claims require sound derivation or proof; empirical, artifact, engineering, software, and causal claims require corresponding evidence, provenance, feasibility reasoning, and validation. Use internal context as exploration history, not authority.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

YOUR TASK:
Review all currently completed papers and determine if ANY ONE paper should be REMOVED because it is redundant with other papers in the library.

CRITICAL CONTEXT:
- This is an ALREADY-COMPLETED paper library
- You are performing a PERIODIC CLEANUP to maintain library quality
- As the library grows, some papers may become REDUNDANT with better papers
- You may identify AT MOST ONE paper for removal (or none)
- It is PERFECTLY ACCEPTABLE to find no papers needing removal

REASONS FOR REMOVAL - A paper should be removed if it:
1. Is now REDUNDANT with other papers (content fully covered by better papers)
2. OVERLAPS significantly with more comprehensive papers
3. Contains information SUPERSEDED by better, more complete papers
4. Was MARGINALLY useful initially but provides no unique value given current library
5. Covers the same solution territory as a newer, superior paper
6. Is more indirect or auxiliary while another paper provides a stronger rigorous direct answer on the same territory

REASONS TO KEEP - A paper should be kept if it:
1. Provides a stronger direct answer to the user's prompt than overlapping papers
2. Provides a distinct solution mechanism or causal route
3. Offers a different perspective or approach that materially improves the strongest direct answer path
4. Contributes distinct evidence, derivation, argument, implementation, or design necessary for direct prompt progress
5. Contains a distinct theorem, proof, algorithm, or impossibility result
6. Provides a distinct experimental proposal, validation method, risk, failure-mode, or limitation analysis
7. Contributes diversity only when that diversity improves credible direct-answer progress

CONSERVATIVE APPROACH:
- When in doubt, DO NOT recommend removal
- Only recommend removal if you are CERTAIN the library would be BETTER without it
- A smaller, higher-quality library is better than a large, redundant one
- Removing valuable content is worse than keeping slightly redundant content

CRITICAL SELECTION RULE:
When multiple papers overlap, select the WEAKEST one for removal - the one that provides the LEAST unique value. NEVER remove a more comprehensive paper in favor of keeping a less comprehensive one.

DIRECT-SOLUTION PRIORITY:
If overlapping papers differ in how directly they answer the user's research goal, preserve the paper with the strongest rigorous direct answer and remove the more auxiliary one first when all else is equal.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_paper_redundancy_json_schema() -> str:
    """Get JSON schema for paper redundancy review."""
    return """REQUIRED JSON FORMAT:
{
  "should_remove": true | false,
  "paper_id": "string - The paper_id to remove (or null if should_remove is false)",
  "reasoning": "string - Detailed explanation of why this paper should be removed OR why no removal is needed"
}

FIELD REQUIREMENTS:
- should_remove: Boolean
- paper_id: Required if should_remove is true, null otherwise
- reasoning: ALWAYS required

CONSTRAINTS:
- Maximum 1 paper can be removed per review cycle
- Conservative approach: when in doubt, do NOT remove

EXAMPLES:

Remove Paper:
{
  "should_remove": true,
  "paper_id": "paper_005",
  "reasoning": "Paper 005 is now redundant. Papers 003, 009, and 014 provide the same mechanism and evidence with more complete feasibility analysis and validation. Paper 005 adds no distinct implementation, proof, experiment, risk analysis, or other direct-answer value, so the other papers fully subsume it."
}

No Removal:
{
  "should_remove": false,
  "paper_id": null,
  "reasoning": "After reviewing all paper titles and abstracts, no papers are redundant. Each paper contributes a distinct solution mechanism, evidence base, implementation, validation method, risk analysis, theorem, proof, or other necessary direct-answer component. The library maintains useful diversity without unnecessary overlap."
}"""


def build_paper_redundancy_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]]
) -> str:
    """
    Build the paper redundancy review prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        papers_summary: List of all papers with title, abstract, word count
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_paper_redundancy_system_prompt(),
        "\n---\n",
        get_paper_redundancy_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n"
    ]
    
    # Add all papers
    parts.append("CURRENT PAPER LIBRARY:\n")
    if papers_summary:
        for p in papers_summary:
            parts.append(f"\n{'=' * 60}")
            parts.append(f"\nPaper ID: {p.get('paper_id', 'Unknown')}")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            source_ids = p.get('source_brainstorm_ids', [])
            if source_ids:
                parts.append(f"\nSource Brainstorms: {', '.join(source_ids)}")
            parts.append(f"\n{'=' * 60}\n")
    else:
        parts.append("\n[No papers in library]\n")
    
    parts.append("\n---\n")
    parts.append("Review the library for redundancy and provide your decision as JSON:")
    
    return "".join(parts)

