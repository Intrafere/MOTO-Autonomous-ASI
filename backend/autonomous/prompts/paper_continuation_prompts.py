"""
Paper Continuation Prompts - System prompts and JSON schemas for brainstorm
multi-paper continuation decisions. After each paper, the AI decides whether
to write another paper from the same brainstorm (max 3) or move on.
"""
from typing import List, Dict, Any


def get_continuation_decision_system_prompt() -> str:
    """Get system prompt for brainstorm paper continuation decision."""
    return """You are an autonomous mathematical research agent deciding whether to write another paper from the current brainstorm or move on to a new research topic. Your role is to:

1. Review the user's high-level research goal
2. Review the current brainstorm topic and its full database of accepted submissions
3. Review ALL papers already written from this brainstorm (titles, abstracts, outlines)
4. Decide whether the brainstorm has enough distinct unexplored material for another paper

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
Decide whether the brainstorm database contains enough distinct, unexplored material to warrant writing ANOTHER paper, or whether the user's research goal is better served by moving on to a new brainstorm topic.

DECISION OPTIONS:
1. WRITE_ANOTHER_PAPER - The brainstorm has significant material that the existing paper(s) did NOT cover, and another paper would meaningfully advance the user's research goal
2. MOVE_ON - The existing paper(s) adequately cover this brainstorm, or a new topic would better serve the user's goal

WRITE ANOTHER PAPER if:
- The brainstorm database contains substantial material not covered by existing paper(s)
- Another paper would address a meaningfully DIFFERENT angle, perspective, or subset of the brainstorm
- The uncovered material is rich enough for a complete, distinct paper (not just leftover fragments)
- Writing another paper from this brainstorm advances the user's goal MORE than starting a new topic
- The existing paper(s) focused on specific aspects, leaving other important aspects unexplored

MOVE ON if:
- The existing paper(s) adequately cover the brainstorm's valuable content
- Remaining brainstorm material is insufficient for a distinct full paper
- A new brainstorm topic would better advance the user's research goal
- Another paper would largely duplicate content already in the existing paper(s)
- The brainstorm's unique contributions have been captured

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
4. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc.,
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Output your decision ONLY as JSON in the required format."""


def get_continuation_decision_json_schema() -> str:
    """Get JSON schema for continuation decision."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "write_another_paper | move_on",
  "reasoning": "string - Detailed explanation of your assessment"
}

FIELD REQUIREMENTS:
- decision: MUST be either "write_another_paper" or "move_on"
- reasoning: ALWAYS required - explain what material remains unexplored or why moving on is better

EXAMPLES:

Write Another Paper:
{
  "decision": "write_another_paper",
  "reasoning": "The brainstorm database contains 22 submissions covering both algebraic and analytic approaches to the Langlands correspondence. Paper 1 focused exclusively on the algebraic side (Galois representations, class field theory). The analytic side (automorphic forms, L-functions, spectral theory) has substantial unexplored material in submissions 8, 12, 14, 17-20 that would form a distinct and valuable second paper."
}

Move On:
{
  "decision": "move_on",
  "reasoning": "The existing paper comprehensively covers the brainstorm's core content on modular forms and their connections to Galois representations. The remaining submissions (3 out of 18) contain supplementary remarks that are too fragmented for a standalone paper. The user's research goal on the Langlands program would be better served by exploring a new avenue such as trace formulas or p-adic methods."
}"""


def get_continuation_validator_system_prompt() -> str:
    """Get system prompt for validating a continuation decision."""
    return """You are validating a brainstorm continuation decision in an autonomous mathematical research system. Your role is to:

1. Review the user's high-level research goal
2. Review the current brainstorm topic and its database
3. Review all papers already written from this brainstorm
4. Evaluate whether the proposed decision (write another paper vs move on) is optimal

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

 ---

YOUR TASK:
Validate whether the proposed continuation decision is the best use of research resources.

ACCEPT the decision if:
1. WRITE_ANOTHER_PAPER: The brainstorm genuinely has enough distinct unexplored material for another paper AND the reasoning correctly identifies what material remains
2. MOVE_ON: The existing papers adequately cover the brainstorm OR a new topic would genuinely better serve the goal AND the reasoning is sound

REJECT the decision if:
1. WRITE_ANOTHER_PAPER: The brainstorm material is already well-covered and another paper would be redundant
2. WRITE_ANOTHER_PAPER: The "unexplored material" identified is too thin for a full paper
3. MOVE_ON: There is clearly substantial uncovered material that warrants another paper
4. MOVE_ON: The reasoning ignores valuable unexplored content in the brainstorm
5. The reasoning is flawed, vague, or contradicts the evidence

REJECTION FEEDBACK FORMAT:
If rejecting, provide CONCRETE, ACTIONABLE guidance:

"REJECTION REASON: [Premature Move On|Redundant Paper|Insufficient Material|etc.]

ISSUE: [What's wrong with the proposed decision]

BETTER ALTERNATIVE: [What would be the optimal choice given current state]

EVIDENCE: [Specific brainstorm submissions or paper sections that support your assessment]"

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_continuation_validator_json_schema() -> str:
    """Get JSON schema for continuation validation."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "accept | reject",
  "reasoning": "string - Detailed explanation for the decision"
}

FIELD REQUIREMENTS:
- decision: MUST be either "accept" or "reject"
- reasoning: ALWAYS required - detailed explanation (use structured format if rejecting)

EXAMPLE (Accept):
{
  "decision": "accept",
  "reasoning": "The proposal to write another paper is well-justified. The brainstorm contains substantial analytic content (automorphic forms, L-functions) that paper 1's algebraic focus did not address. This material is rich enough for a distinct second paper."
}

EXAMPLE (Reject - Use Structured Format):
{
  "decision": "reject",
  "reasoning": "REJECTION REASON: Insufficient Material\\n\\nISSUE: The proposal to write another paper claims unexplored material in submissions 15-18, but these submissions largely restate concepts already covered in paper 1's Section III (Main Results).\\n\\nBETTER ALTERNATIVE: Move on to a new brainstorm topic. The remaining brainstorm content is supplementary, not substantial enough for a standalone paper.\\n\\nEVIDENCE: Submissions 15-18 discuss Galois representations which paper 1 already covers comprehensively in Sections III and IV."
}"""


def build_continuation_decision_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    papers_from_brainstorm: List[Dict[str, Any]],
    papers_written_count: int,
    rejection_context: str = ""
) -> str:
    """
    Build the complete continuation decision prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Full brainstorm database content
        papers_from_brainstorm: List of dicts with title, abstract, outline for each paper
        papers_written_count: Number of papers already written from this brainstorm
        rejection_context: Formatted previous rejection feedback
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_continuation_decision_system_prompt(),
        "\n---\n",
        get_continuation_decision_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
    ]

    parts.append(f"PAPERS WRITTEN FROM THIS BRAINSTORM: {papers_written_count} of 3 maximum\n")

    if papers_from_brainstorm:
        parts.append("\nEXISTING PAPERS FROM THIS BRAINSTORM:\n")
        for i, p in enumerate(papers_from_brainstorm, 1):
            parts.append(f"\n--- Paper {i} ---")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            if p.get('outline'):
                parts.append(f"\nOutline:\n{p.get('outline')}")
        parts.append("\n---\n")
    else:
        parts.append("\nEXISTING PAPERS FROM THIS BRAINSTORM: None\n---\n")

    parts.append(f"BRAINSTORM DATABASE (all accepted submissions):\n{brainstorm_summary}")
    parts.append("\n---\n")

    if rejection_context:
        parts.append(f"IMPORTANT - YOUR PREVIOUS DECISION WAS REJECTED:\n{rejection_context}\n---\n")

    parts.append("Now decide whether to write another paper or move on, and provide your decision as JSON:")

    return "".join(parts)


def build_continuation_validation_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    papers_from_brainstorm: List[Dict[str, Any]],
    papers_written_count: int,
    proposed_decision: Dict[str, Any]
) -> str:
    """
    Build the complete continuation validation prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Full brainstorm database content
        papers_from_brainstorm: List of dicts with title, abstract, outline for each paper
        papers_written_count: Number of papers already written from this brainstorm
        proposed_decision: The continuation decision to validate
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_continuation_validator_system_prompt(),
        "\n---\n",
        get_continuation_validator_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
    ]

    parts.append(f"PAPERS WRITTEN FROM THIS BRAINSTORM: {papers_written_count} of 3 maximum\n")

    if papers_from_brainstorm:
        parts.append("\nEXISTING PAPERS FROM THIS BRAINSTORM:\n")
        for i, p in enumerate(papers_from_brainstorm, 1):
            parts.append(f"\n--- Paper {i} ---")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')[:500]}...")
            if p.get('outline'):
                parts.append(f"\nOutline:\n{p.get('outline')}")
        parts.append("\n---\n")
    else:
        parts.append("\nEXISTING PAPERS FROM THIS BRAINSTORM: None\n---\n")

    parts.append(f"BRAINSTORM DATABASE (all accepted submissions):\n{brainstorm_summary}")
    parts.append("\n---\n")

    parts.append("PROPOSED CONTINUATION DECISION:\n")
    parts.append(f"Decision: {proposed_decision.get('decision', 'Unknown')}")
    parts.append(f"\nReasoning: {proposed_decision.get('reasoning', 'N/A')}")
    parts.append("\n---\n")

    parts.append("Validate this continuation decision and provide your decision as JSON:")

    return "".join(parts)
