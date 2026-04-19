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

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
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
5. Covers the same mathematical territory as a newer, superior paper

REASONS TO KEEP - A paper should be kept if it:
1. Provides ANY unique mathematical content not covered elsewhere
2. Offers a different perspective or approach even if related to other papers
3. Contains specific proofs, theorems, or techniques not present elsewhere
4. Contributes to research diversity in any meaningful way
5. Covers distinct mathematical subtopics within a broader area

CONSERVATIVE APPROACH:
- When in doubt, DO NOT recommend removal
- Only recommend removal if you are CERTAIN the library would be BETTER without it
- A smaller, higher-quality library is better than a large, redundant one
- Removing valuable content is worse than keeping slightly redundant content

CRITICAL SELECTION RULE:
When multiple papers overlap, select the WEAKEST one for removal - the one that provides the LEAST unique value. NEVER remove a more comprehensive paper in favor of keeping a less comprehensive one.

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
  "reasoning": "Paper 005 on 'Basic Principles of Class Field Theory' is now redundant. Papers 003, 009, and 014 provide more comprehensive coverage of class field theory with deeper mathematical rigor and broader applications. Paper 005's unique contributions (elementary introduction) are minimal and the library would be stronger without this redundant entry. The other papers fully subsume its content."
}

No Removal:
{
  "should_remove": false,
  "paper_id": null,
  "reasoning": "After reviewing all paper titles and abstracts, no papers are redundant. Each paper provides unique perspectives, covers distinct mathematical areas, or approaches common topics from different angles. The library maintains good diversity without unnecessary overlap."
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

