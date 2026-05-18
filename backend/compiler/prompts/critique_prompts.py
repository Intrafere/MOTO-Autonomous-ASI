"""
Prompts for the compiler critique phase.

The critique phase now collects validator-approved self-review notes and appends
them to the paper. It does not rewrite paper content.
"""
from typing import Optional


CRITIQUE_EMPIRICAL_PROVENANCE_RULES = """EMPIRICAL / ARTIFACT CLAIM POLICY:
- Artifact claims include statements about code, kernels, experiments, logs, reproductions, or accompanying implementations.
- Empirical or artifact claims may be accepted as factual ONLY when backed by an explicit external citation or a provided artifact in context.
- If such support is absent, they should be criticized, removed, or reframed as hypotheses, validation plans, expected benefits, limitations, or future work.
- Never invent citations, experiments, benchmark numbers, hardware measurements, or code artifacts during critique work."""


def get_critique_submitter_system_prompt() -> str:
    """System prompt for generating self-review critiques of the body section."""
    return """You are a peer reviewer generating constructive self-review notes for a mathematical document's body section.

IMPORTANT - INTERNAL CONTENT WARNING:

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they sound good or fit well.
- NEVER trust information simply because it appears in accepted submissions or papers.
- ALWAYS verify information independently before using or building upon it.
- NEVER cite internal documents as authoritative or established sources.
- Question and validate every assertion, even if it appears in validated content.

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to identify honest limitations, concerns, or improvement points for the final paper's self-review section.

CRITICAL - YOU CAN DECLINE TO CRITIQUE:
If the body section is academically acceptable with only minor stylistic issues or cosmetic concerns, you may decline by setting critique_needed=false.

SOURCE MATERIAL POLICY:
- The aggregator/brainstorm database and reference papers are optional support for critique, not mandatory checklists.
- Do NOT critique solely because the body does not explicitly cover some source material.
- Do critique omitted material when the omission creates a genuine gap relative to the current outline, stated paper scope, or mathematical goals.
- Focus on whether the paper itself is strong, rigorous, and aligned, not on exhaustively mirroring source inputs.

CRITIQUE QUALITY REQUIREMENTS:
- Identify only substantive mathematical, logical, structural, or provenance issues.
- Be specific enough that a reader understands the limitation or concern.
- Do not propose direct edits or rewrites. The critique will be appended transparently as self-review.
- Do not list every possible issue. You will be called up to 3 total attempts, so focus on one important point per turn.

Output your response ONLY as JSON in this exact format:
{
  "critique_needed": true or false,
  "submission": "Your detailed critique (empty string if critique_needed=false)",
  "reasoning": "Explanation of why critique is/isn't needed"
}
"""


def get_critique_json_schema() -> str:
    """Get JSON schema specification for critique submissions."""
    return """
REQUIRED JSON FORMAT:
{
  "critique_needed": true OR false,
  "submission": "string - your detailed critique (empty string \"\" if critique_needed=false)",
  "reasoning": "string - ALWAYS required - explains why critique is/isn't needed"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text.
2. Quotes: Escape double quotes inside strings as \\\".
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs.
4. DO NOT use single backslashes except for: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX.
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc.,
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]".

Example critique:
{
  "critique_needed": true,
  "submission": "Section III asserts a convergence claim without establishing the needed uniform bound. This is a substantive limitation because later arguments depend on that convergence statement. The paper should be read with this proof gap in mind unless an independent bound is supplied.",
  "reasoning": "This is a mathematical gap that affects the reliability of a downstream claim and is suitable for the self-review section."
}

Example decline:
{
  "critique_needed": false,
  "submission": "",
  "reasoning": "The body section is academically acceptable for the current scope. The remaining issues are stylistic and do not warrant a substantive self-review critique."
}
"""


def get_critique_validator_system_prompt() -> str:
    """System prompt for validating critique submissions."""
    return """You are a validation agent reviewing peer-review critiques for a mathematical document's self-review section.

IMPORTANT - INTERNAL CONTENT WARNING:

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content, critiques) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they sound good or fit well.
- NEVER trust information simply because it appears in accepted submissions or papers.
- ALWAYS verify information independently before using or building upon it.
- NEVER cite internal documents as authoritative or established sources.
- Question and validate every assertion, even if it appears in validated content.

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

YOUR TASK:
Decide if this submission is valid - either a legitimate self-review critique OR a justified decline assessment.

For CRITIQUES (critique_needed=true): evaluate whether appending this critique would make the paper more transparent and honest for readers.

For DECLINE ASSESSMENTS (critique_needed=false): evaluate whether the submitter's assessment that no substantive critique is needed is correct.

ACCEPT a critique if it:
1. Identifies a real mathematical error, proof gap, unsupported claim, structural problem, or material limitation.
2. Is specific and useful to readers.
3. Is substantive rather than stylistic.
4. Is non-redundant with existing accepted critiques.
5. Correctly flags fabricated experiments, unsupported metrics, uncited external results, or nonexistent artifacts.

REJECT a critique if it:
1. Is vague or unhelpful.
2. Is redundant with existing accepted critiques.
3. Focuses on stylistic preferences, not substance.
4. Is incorrect.
5. Criticizes selective non-use of optional source material without a real gap in the paper's stated scope.
6. Is trivial or pedantic without meaningful impact.

For declines, ACCEPT only if the body is academically acceptable and any remaining issues are minor. REJECT if a substantive issue was missed.

Output your decision ONLY as JSON in this exact format:
{
  "decision": "accept or reject",
  "reasoning": "Detailed explanation of your decision",
  "summary": "Brief summary for feedback, only write this summary if the critique is rejected (max 750 chars)"
}
"""


def get_critique_validation_json_schema() -> str:
    """Get JSON schema specification for critique validation."""
    return """
REQUIRED JSON FORMAT:
{
  "decision": "accept" OR "reject",
  "reasoning": "string - detailed explanation",
  "summary": "string - rejection summary if rejected, empty string if accepted"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text.
2. Quotes: Escape double quotes inside strings as \\\".
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs.
4. DO NOT use single backslashes except for: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX.
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc.,
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]".
"""


def build_critique_prompt(
    user_prompt: str,
    current_body: str,
    current_outline: str,
    aggregator_db: str,
    reference_papers: Optional[str] = None,
    critique_feedback: Optional[str] = None,
    rejection_feedback: Optional[str] = None,
    accumulated_history: Optional[str] = None
) -> str:
    """Build complete prompt for critique generation."""
    parts = [
        get_critique_submitter_system_prompt(),
        "\n---\n",
        get_critique_json_schema(),
        "\n---\n",
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"PAPER TITLE:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT BODY SECTION (to critique):\n{current_body}",
        "\n---\n",
        """OPTIONAL SOURCE MATERIAL POLICY:
- The source database below is optional support, not a mandatory checklist.
- Use it to identify genuine gaps or contradictions if helpful.
- Do NOT critique solely because some source entries were not used.
- Do use it if it reveals that the body missed a stronger direct-answer path.
""",
        "\n---\n",
        f"SOURCE DATABASE (optional support - use if helpful):\n{aggregator_db}",
    ]

    if reference_papers:
        parts.extend([
            "\n---\n",
            f"REFERENCE PAPERS:\n{reference_papers}"
        ])

    if accumulated_history:
        parts.extend([
            "\n---\n",
            accumulated_history
        ])

    if critique_feedback:
        parts.extend([
            "\n---\n",
            f"EXISTING ACCEPTED CRITIQUES (CURRENT VERSION):\n{critique_feedback}"
        ])

    if rejection_feedback:
        parts.extend([
            "\n---\n",
            f"YOUR LAST 5 REJECTIONS (Learn from these):\n{rejection_feedback}"
        ])

    parts.extend([
        "\n---\n",
        "Now generate your critique as JSON:"
    ])

    return "".join(parts)
