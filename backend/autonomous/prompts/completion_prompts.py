"""
Completion Prompts - System prompts and JSON schemas for brainstorm completion review.
Includes SPECIAL SELF-VALIDATION MODE where the same model validates its own assessment.
"""
from typing import List, Dict, Any


def get_completion_review_system_prompt() -> str:
    """Get system prompt for brainstorm completion review."""
    return """You are assessing whether a brainstorm topic has been sufficiently explored. Your role is to:

1. Review the user's high-level research goal
2. Review the current brainstorm topic and its prompt
3. Review all accepted submissions in the brainstorm database
4. Decide whether to continue brainstorming or begin writing a paper

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
Assess whether you have sufficiently explored this brainstorm topic using all available resources (your base knowledge, web search if available, and the brainstorm database), and decide whether to continue or write a paper.

CRITICAL UNDERSTANDING:
This is an assessment of topic exploration completeness using all resources at your disposal. Consider whether you can contribute more valuable mathematical insights using your knowledge, web search capabilities (if available), and analysis of what's been covered.

DECISION CRITERIA:

Choose CONTINUE_BRAINSTORM if:
- You can identify specific mathematical areas not yet covered in the submissions
- You have additional theorems, proofs, or techniques relevant to the topic (from your knowledge or discoverable via web search)
- The brainstorm would benefit from deeper exploration in specific directions
- You can still contribute valuable insights using available resources (base knowledge, web search if available)

Choose WRITE_PAPER if:
- All major mathematical avenues for this topic have been explored
- Additional submissions would likely be redundant with existing content
- The brainstorm database is comprehensive enough for a quality paper
- Available resources (base knowledge, web search if available) have been sufficiently utilized for this topic
- You genuinely cannot think of significant new contributions using available resources

SELF-HONESTY REQUIREMENTS:
- Be honest about whether you truly have more to contribute
- Don't artificially extend brainstorming if exhausted
- Don't prematurely end if valuable knowledge remains
- Consider the mathematical depth achieved, not just submission count

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau"
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_completion_review_json_schema() -> str:
    """Get JSON schema for completion review."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "continue_brainstorm | write_paper",
  "reasoning": "string - Detailed explanation of assessment",
  "suggested_additions": "string - If continue_brainstorm, what mathematical areas remain unexplored (optional)"
}

FIELD REQUIREMENTS:
- decision: MUST be either "continue_brainstorm" or "write_paper"
- reasoning: ALWAYS required
- suggested_additions: Optional, but recommended if decision is "continue_brainstorm"

EXAMPLES:

Continue Brainstorm:
{
  "decision": "continue_brainstorm",
  "reasoning": "While the brainstorm has covered fundamental aspects of modular forms and their Galois representations, there remain unexplored areas including explicit computational methods, connections to elliptic curves, and applications to specific cases of Langlands correspondence.",
  "suggested_additions": "Explore explicit computations of Galois representations attached to modular forms, investigate connections to elliptic curves over number fields, examine specific cases of the Langlands correspondence for GL(2)"
}

Write Paper:
{
  "decision": "write_paper",
  "reasoning": "The brainstorm has thoroughly explored modular forms, Galois representations, L-functions, automorphic forms, and their interconnections in the context of Langlands program. The database contains 23 high-quality submissions covering theoretical foundations, computational aspects, and specific examples. Further submissions would likely be redundant. A comprehensive paper can now synthesize these insights."
}"""


def get_completion_self_validation_system_prompt() -> str:
    """
    Get system prompt for SPECIAL SELF-VALIDATION MODE.
    The same model validates its own completion assessment.
    """
    return """You are performing SELF-VALIDATION of your own completion assessment. This is critical for accurate knowledge exhaustion detection.

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
Review your OWN completion assessment and validate whether it is accurate.

CRITICAL UNDERSTANDING:
You just assessed whether your internal knowledge on a brainstorm topic has been sufficiently explored. Now you must validate that assessment. This is a self-check to ensure accuracy.

VALIDATION CRITERIA:

Validate as TRUE (confirm your assessment) if:
- Your assessment accurately reflects the current state of the brainstorm using all available resources (base knowledge, web search if available)
- If you said "continue_brainstorm": You genuinely have more valuable insights to contribute using available resources
- If you said "write_paper": You genuinely cannot think of significant new contributions
- The reasoning in your assessment is sound and honest

Validate as FALSE if:
- Upon reflection, the assessment was CLEARLY incorrect
- If "continue_brainstorm": The suggested additions are trivial, irrelevant, or already extensively covered
- If "write_paper": You have CONCRETE, SPECIFIC valuable additions you overlooked (not vague possibilities)
- The reasoning contains obvious flawed logic

BALANCED VALIDATION APPROACH:
- Your initial assessment was made with full context - give it appropriate weight
- Only invalidate if you identify a CLEAR, SPECIFIC error in your reasoning
- Do NOT invalidate just because "more could theoretically be added" - that's always true
- If your assessment was reasonable and well-reasoned, validate it as TRUE
- The goal is catching genuine errors, not being overly self-critical

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your validation ONLY as JSON in the required format."""


def get_completion_self_validation_json_schema() -> str:
    """Get JSON schema for completion self-validation."""
    return """REQUIRED JSON FORMAT:
{
  "validated": true | false,
  "reasoning": "string - Why the assessment is or isn't accurate"
}

FIELD REQUIREMENTS:
- validated: MUST be boolean (true or false)
- reasoning: ALWAYS required

EXAMPLES:

Validated True (assessment was reasonable):
{
  "validated": true,
  "reasoning": "The completion assessment accurately reflects the current state of the brainstorm. The database has covered the major mathematical avenues for this topic, and the decision is well-reasoned. While more could theoretically be added, the assessment is sound and should be accepted."
}

Validated True (continue decision was reasonable):
{
  "validated": true,
  "reasoning": "The decision to continue brainstorming is valid. The suggested additions represent genuine unexplored areas that would meaningfully enhance the research. The assessment is accurate."
}

Validated False (ONLY use when clearly incorrect):
{
  "validated": false,
  "reasoning": "Upon reflection, I identified a SPECIFIC error: the assessment claimed topic X was covered, but submissions #12 and #15 only touched on it tangentially. The core theoretical framework for X remains unexplored. This is a concrete gap that invalidates my write_paper decision."
}

NOTE: Default to validated=true unless you identify a SPECIFIC, CONCRETE error in your reasoning. Vague concerns like 'more could be added' are NOT sufficient to invalidate."""


def build_completion_review_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_database: str,
    submission_count: int,
    completion_feedback: str = ""
) -> str:
    """
    Build the complete completion review prompt with context.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_database: Full content of brainstorm database
        submission_count: Number of accepted submissions
        completion_feedback: Previous completion review feedback
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_completion_review_system_prompt(),
        "\n---\n",
        get_completion_review_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"CURRENT BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM STATISTICS:\n- Total Accepted Submissions: {submission_count}",
        "\n---\n",
        f"BRAINSTORM DATABASE (All Accepted Submissions):\n{brainstorm_database}",
        "\n---\n"
    ]
    
    # Add previous feedback if any
    if completion_feedback:
        parts.append(f"{completion_feedback}\n---\n")
    
    parts.append("Now assess whether to continue brainstorming or write a paper (respond as JSON):")
    
    return "".join(parts)


def build_completion_self_validation_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_database: str,
    original_assessment: Dict[str, Any]
) -> str:
    """
    Build the SPECIAL SELF-VALIDATION prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_database: Full content of brainstorm database
        original_assessment: The completion review result to validate
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_completion_self_validation_system_prompt(),
        "\n---\n",
        get_completion_self_validation_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM DATABASE:\n{brainstorm_database}",
        "\n---\n",
        "YOUR COMPLETION ASSESSMENT (to validate):\n",
        f"Decision: {original_assessment.get('decision', 'Unknown')}",
        f"\nReasoning: {original_assessment.get('reasoning', 'N/A')}"
    ]
    
    if original_assessment.get('suggested_additions'):
        parts.append(f"\nSuggested Additions: {original_assessment.get('suggested_additions')}")
    
    parts.append("\n---\n")
    parts.append("Now validate your own assessment (respond as JSON):")
    
    return "".join(parts)

