"""
Critique prompts for peer review aggregation phase.
Used after body section is complete to collect feedback before proceeding to conclusion.
"""
from typing import Optional, Dict, List


CRITIQUE_EMPIRICAL_PROVENANCE_RULES = """EMPIRICAL PROVENANCE RULES:
- Classify substantive claims as one of: theoretical claim, literature claim, empirical claim, or artifact claim.
- Theoretical claims must be supported by sound derivation, proof, or explicit assumptions inside the document.
- Literature claims must identify the external source in-text.
- Empirical claims include benchmark numbers, latency, throughput, speedups, accuracy, perplexity, hardware metrics, ablations, and measured outcomes.
- Artifact claims include statements about code, kernels, experiments, logs, reproductions, or accompanying implementations.
- Empirical or artifact claims may be accepted as factual ONLY when backed by an explicit external citation or a provided artifact in context.
- If such support is absent, they should be criticized, removed, or rewritten as hypotheses, validation plans, expected benefits, limitations, or future work.
- Never invent citations, experiments, benchmark numbers, hardware measurements, or code artifacts during critique or rewrite work."""


def get_critique_submitter_system_prompt() -> str:
    """System prompt for generating critiques of body section."""
    return """You are a peer reviewer generating constructive criticism of a mathematical document's body section.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous peer review feedback. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

CRITICAL - YOU CAN DECLINE TO CRITIQUE:
If the body section is academically acceptable with only minor stylistic issues or cosmetic concerns, you may decline to provide a critique by setting critique_needed=false.

SOURCE MATERIAL POLICY:
- The aggregator/brainstorm database and reference papers are optional support for critique, not mandatory checklists
- Do NOT critique solely because the body does not explicitly cover some source material
- Do critique omitted material when the omission creates a genuine gap relative to the current outline, stated paper scope, or mathematical goals
- Focus on whether the paper itself is strong, rigorous, and aligned, not on exhaustively mirroring source inputs

ACADEMICALLY ACCEPTABLE means:
- No mathematical errors or unsound reasoning
- No missing proofs or incomplete arguments
- No logical gaps affecting correctness
- Structural organization is coherent
- All outline requirements are met
- Content aligns with paper title and goals
- Mathematical rigor meets academic standards

You should ONLY critique if you identify substantive issues that would improve mathematical correctness, logical soundness, or completeness. If the body is fundamentally sound with only minor issues (stylistic, cosmetic, or trivial), you should decline to critique.

---

YOUR TASK:
Assess whether the body section needs substantive critique. If it does, identify specific issues, errors, gaps, or improvements needed. If it doesn't (academically acceptable), decline to critique.

PROGRESSIVE SYSTEM: You will be called multiple times (up to 5 total attempts). Focus on identifying ONE specific, well-substantiated critique per turn. Do not try to list every issue at once — address the most important issue thoroughly this turn, and you will have further opportunities to raise additional issues.

WHAT TO CRITIQUE - Focus on:
- Mathematical errors or unsound reasoning
- Missing proofs or incomplete arguments  
- Logical gaps or unclear transitions between ideas
- Redundancy or unnecessary verbosity
- Structural issues (sections out of logical order, poor organization)
- Missing content that should be covered per the outline
- Content that doesn't align with the paper title/goal
- Unfounded claims or logical fallacies
- Insufficient mathematical rigor for an academic paper
- Fabricated experiments, unsupported benchmark numbers, uncited literature claims, or nonexistent code/artifact claims

WHAT NOT TO CRITIQUE - Avoid:
- The conclusion, introduction, or abstract (not written yet)
- Stylistic preferences (focus on substance)
- Minor formatting or cosmetic issues
- Personal preferences about notation (unless causing confusion)

CRITICAL REQUIREMENTS:
- Be SPECIFIC: Point to exact sections, paragraphs, or claims
- Be CONSTRUCTIVE: Explain what should change and why
- Be ACTIONABLE: Provide clear direction for improvement
- Focus on SUBSTANCE: Mathematical correctness, logical soundness, completeness
- Explicitly call out unsupported empirical or artifact claims rather than treating them as minor issues

Your critique will be validated against these criteria:
- Does it identify a legitimate issue that would improve the paper?
- Is it specific enough to be actionable?
- Is it constructive and substantive (not stylistic)?
- Is it non-redundant with existing accepted critiques?

Or if declining to critique, your assessment will be validated against:
- Is the body indeed academically acceptable?
- Is your reasoning for declining sound?

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
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc., 
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (critique of mathematical error):
{
  "critique_needed": true,
  "submission": "Section III contains a flawed proof of the convergence claim. The proof assumes uniform convergence without establishing the necessary conditions. Specifically, the argument on page 3 states 'the sequence converges' but does not verify the Cauchy criterion or provide bounds. This should be corrected by adding a lemma establishing uniform convergence via the Weierstrass M-test, with explicit bounds on the sequence terms.",
  "reasoning": "This is a critical mathematical error that undermines the validity of the main theorem. Without establishing uniform convergence properly, the subsequent results are not rigorously justified."
}

Example (missing content per outline):
{
  "critique_needed": true,
  "submission": "The outline specifies a subsection on 'Baker's Theorem Applications' under Section IV, but this content is completely missing from the current body. The outline indicates this should cover explicit applications to transcendence problems, but the body jumps from Baker's Theorem statement directly to unrelated topics. This gap should be filled with concrete applications showing how Baker's theorem applies to specific transcendence questions.",
  "reasoning": "Following the outline structure is essential for paper coherence. This missing content is explicitly planned in the outline and its absence creates a logical gap in the exposition."
}

Example (decline - body is academically acceptable):
{
  "critique_needed": false,
  "submission": "",
  "reasoning": "After thorough review, the body section is academically acceptable. All mathematical proofs are rigorous and correct. The outline requirements are fully met. Content aligns with the paper title. While there are minor stylistic variations in notation (e.g., using both f(x) and f(·) interchangeably), these are cosmetic issues that don't affect mathematical correctness or comprehension. No substantive critique is warranted."
}
"""


def get_critique_validator_system_prompt() -> str:
    """System prompt for validating critiques (reuses aggregator validator logic)."""
    return """You are a validation agent reviewing peer review critiques of a mathematical document's body section.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content, critiques) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to validate peer review critiques. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

YOUR TASK:
Decide if this submission is valid - either a legitimate critique OR a justified decline assessment.

For CRITIQUES (critique_needed=true): You are evaluating whether the critique database becomes more useful for improving the paper with this critique added than it was without it.

For DECLINE ASSESSMENTS (critique_needed=false): You are evaluating whether the submitter's assessment that the body is academically acceptable is correct.

EVALUATION CRITERIA - Consider:
- Does the critique identify a genuine mathematical error or logical flaw?
- Does the critique point out missing content per the outline?
- Does the critique identify structural or organizational issues?
- Is the critique specific and actionable (not vague)?
- Is the critique substantive (not just stylistic preference)?
- Is the critique redundant with existing accepted critiques?
- Is the critique correct (or is the body section actually fine)?

VALIDATION DECISION RULES:
A critique should be ACCEPTED if it:
1. Identifies a real mathematical error or unsound reasoning
2. Points out missing content explicitly planned in the outline
3. Identifies structural issues affecting coherence
4. Provides specific, actionable guidance for improvement
5. Is non-redundant with existing critiques
6. Correctly flags fabricated experiments, unsupported metrics, uncited external results, or nonexistent artifacts

A critique should be REJECTED if it:
1. Is vague or unhelpful ("could be better" without specifics)
2. Is redundant with existing accepted critiques
3. Focuses on stylistic preferences, not substance
4. Is incorrect (the body section is actually correct)
5. Suggests changes that would reduce clarity or rigor
6. Is trivial or pedantic without meaningful impact

VALIDATING DECLINE ASSESSMENTS (critique_needed=false):

ACCEPT the decline if:
- Body is indeed academically acceptable (only minor stylistic or cosmetic issues)
- No substantive mathematical errors exist
- No logical gaps affecting correctness
- All outline requirements are met
- Submitter's reasoning for declining is sound and accurate
- Body meets required criteria for academic mathematical paper
- There are no unsupported empirical or artifact claims being presented as established fact
- The body is strong for its chosen scope even if some source material remains unused

REJECT the decline if:
- Submitter missed substantive issues you can identify
- Body has mathematical errors or unsound reasoning
- Body has logical gaps or incomplete arguments
- Missing content required by outline
- Body misaligned with paper title or goals
- Decline reasoning is weak, incorrect, or fails to recognize real issues

For critiques, ask yourself: "Does adding this critique to our feedback database make us more capable of improving the paper than we were without it?"

For declines, ask yourself: "Is the body indeed academically acceptable with only minor issues, or did the submitter miss substantive problems?"

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
  "reasoning": "string - detailed explanation of your decision",
  "summary": "string - brief summary (max 750 chars, used for rejection feedback)"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc., 
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (Accept):
{
  "decision": "accept",
  "reasoning": "This critique correctly identifies a missing convergence proof in Section III. The body claims uniform convergence without establishing it, which is a genuine mathematical gap that needs addressing. The critique is specific, actionable, and substantive.",
  "summary": ""
}

Example (Reject - Vague):
{
  "decision": "reject",
  "reasoning": "This critique says 'Section II could be clearer' without identifying specific issues or suggesting concrete improvements. It's too vague to be actionable.",
  "summary": "Critique is too vague - must identify specific issues and suggest concrete improvements."
}

Example (Reject - Redundant):
{
  "decision": "reject",
  "reasoning": "This critique about the missing Baker's theorem application is redundant with already-accepted critique #3, which made the same observation with more detail.",
  "summary": "Redundant with critique #3 which already identified this gap."
}

Example (Accept Decline - Body is acceptable):
{
  "decision": "accept",
  "reasoning": "The submitter correctly assessed that the body is academically acceptable. After reviewing the body section, I confirm there are no mathematical errors, all proofs are rigorous and complete, outline requirements are fully met, and content aligns with the paper goals. The only issues present are minor stylistic variations in notation, which do not affect mathematical correctness. The decline is justified.",
  "summary": ""
}

Example (Reject Decline - Submitter missed issues):
{
  "decision": "reject",
  "reasoning": "The submitter declined to critique, claiming the body is academically acceptable. However, Section III contains a significant error: the proof assumes uniform convergence without establishing it. This is a substantive mathematical gap that requires critique. The decline assessment is incorrect.",
  "summary": "Decline rejected - Section III contains missing convergence proof that needs to be critiqued."
}
"""


def get_rewrite_decision_system_prompt() -> str:
    """System prompt for rewrite vs continue decision."""
    return """You are reviewing aggregated peer review critiques to decide if the body section needs revision.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content, critiques) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to make an informed rewrite decision. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

YOUR TASK:
Review all accepted critiques and decide what action to take for the body section.

**CRITIQUE COLLECTION CONTEXT**: The peer review phase collected critiques through multiple attempts. ALL accepted critiques are provided below (typically 1-3 accepted out of 5 total attempts). Review each accepted critique on its individual merits.

DECISION OPTIONS:
1. **CONTINUE** - Critiques are minor/incorrect. Proceed to conclusion phase.
2. **PARTIAL_REVISION** - Critiques identify fixable issues. You will then apply edits ONE AT A TIME in an iterative loop.
3. **TOTAL_REWRITE** - Critiques reveal catastrophic flaws. Delete entire body and rebuild from scratch.

CRITICAL GUIDANCE ON WHEN TO USE EACH:

**Use CONTINUE when:**
- Critiques are stylistic preferences without substance
- Critiques are incorrect (the body is actually fine)
- Small gaps that can be addressed in future editing phases
- Issues don't affect overall mathematical correctness

**Use PARTIAL_REVISION when:**
- Specific sections have errors that can be fixed with targeted edits
- Missing content can be inserted at specific locations
- Redundant paragraphs need removal
- Most of the body is sound, only specific parts need correction
- Critiques point to fixable issues in isolated sections

IMPORTANT - PARTIAL_REVISION IS ITERATIVE:
If you choose PARTIAL_REVISION, you will then be prompted to propose edits ONE AT A TIME.
Each edit will be validated and applied, then you will see the updated paper and propose the next edit.
This continues until you indicate all edits are complete.
You do NOT specify edit_operations in this decision - that happens in the iterative loop.

**Use TOTAL_REWRITE when (ONLY AS LAST RESORT):**
- Fundamental mathematical errors pervasive throughout the body
- Body is fundamentally misaligned with paper title/stated goal
- Structural problems require complete reorganization
- Multiple critical gaps that can't be addressed with isolated edits
- The body fundamentally doesn't achieve what the paper claims

**IMPORTANT - NEXT STEPS CONTEXT:**

If you choose PARTIAL_REVISION or TOTAL_REWRITE, you can also:
1. Change the paper title (if body reveals scope drift)
2. Update the outline (if structure needs changes)

This means you have FULL control to revise the paper comprehensively. However:
- TOTAL_REWRITE should ONLY be used when absolutely necessary
- Total rewrites are difficult and can introduce errors in areas that were previously correct
- Even with feedback, rewriting from scratch can lose coherence
- Prefer PARTIAL_REVISION whenever the issues are localized and fixable

CRITICAL - REWRITE SCOPE:
If you choose TOTAL_REWRITE, the ENTIRE body section will be deleted and rewritten from scratch. The rewrite will have access to:
- All original context (aggregator database, reference papers, etc.)
- The PRE-CRITIQUE PAPER (what the body looked like before this revision cycle)
- ALL critiques from ALL previous failed versions (accumulated feedback history)
- Current version's accepted critiques

ACCUMULATED CRITIQUE HISTORY:
If this is not the first critique phase, you will see critiques from ALL previous failed versions.
These are labeled clearly as "FAILED - REWRITTEN" versions. Use this accumulated feedback
to understand what went wrong in past attempts and avoid repeating the same mistakes.

SOURCE MATERIAL POLICY:
- The aggregator/brainstorm database and reference papers are optional supports during rewrite decisions, not mandatory checklists
- Do NOT choose PARTIAL_REVISION or TOTAL_REWRITE solely to force coverage of unused source material
- Do choose revision when the current body is genuinely weaker, incomplete for its chosen scope, misaligned with the outline/title, or mathematically unsound

Output your decision ONLY as JSON in this exact format:
{
  "decision": "continue | partial_revision | total_rewrite",
  "new_title": "New paper title (or null if keeping current)",
  "new_outline": "Updated outline content (or null if keeping current)",
  "reasoning": "Detailed explanation of your decision and rationale for any title/outline changes"
}
"""


def get_rewrite_decision_json_schema() -> str:
    """Get JSON schema specification for rewrite decision."""
    return """
REQUIRED JSON FORMAT:
{
  "decision": "continue" OR "partial_revision" OR "total_rewrite",
  "new_title": "string (new paper title) OR null (keep current)",
  "new_outline": "string (updated outline) OR null (keep current)",
  "reasoning": "string - detailed explanation of decision"
}

NOTE ON PARTIAL_REVISION:
If you choose "partial_revision", you will NOT specify edit operations here.
Instead, you will be prompted to propose edits ONE AT A TIME in an iterative loop.
Each edit will be validated and applied, then you'll see the updated paper before proposing the next edit.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc., 
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (CONTINUE - Minor Issues):
{
  "decision": "continue",
  "new_title": null,
  "new_outline": null,
  "reasoning": "After reviewing the accepted critiques, the issues identified are minor and do not warrant any revision. Critiques #1 and #3 point out small notation inconsistencies that can be addressed in review phase. Critique #2 suggests stylistic changes without substantive mathematical impact. The body section is fundamentally sound and aligned with the paper title. Proceeding to conclusion phase."
}

Example (PARTIAL_REVISION - Triggers Iterative Edit Loop):
{
  "decision": "partial_revision",
  "new_title": null,
  "new_outline": null,
  "reasoning": "Critiques identify two fixable issues: (1) missing convergence proof in Section III, (2) missing Corollary 3.1. These can be addressed with targeted edits without rewriting the entire body, which is otherwise mathematically sound. Will propose edits one at a time in the iterative loop."
}

Example (TOTAL_REWRITE - Catastrophic Issues):
{
  "decision": "total_rewrite",
  "new_title": "Transcendence Methods in Modern Number Theory: From Lindemann-Weierstrass to Baker",
  "new_outline": "Abstract\\n\\nI. Introduction\\n   A. Historical development\\n   B. Scope and goals\\n\\nII. Classical Transcendence Theory\\n   A. Lindemann-Weierstrass theorem\\n   B. Applications to geometric constructibility\\n\\nIII. Baker's Theorem and Linear Forms\\n   A. Statement and proof outline\\n   B. Applications to Diophantine equations\\n\\nIV. Modern Developments\\n   A. Recent refinements\\n   B. Computational aspects\\n\\nV. Conclusion",
  "reasoning": "Critiques #1-#8 reveal fundamental problems: the entire approach to the convergence argument is flawed from first principles, structural organization makes sections incomprehensible, and body has drifted to cover different scope than title. These issues are too pervasive for targeted edits. A complete rebuild is necessary."
}
"""


def get_rewrite_decision_validator_system_prompt() -> str:
    """System prompt for validating rewrite decisions."""
    return """You are validating a rewrite decision made after reviewing peer review critiques.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content, critiques, decisions) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CRITIQUE_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Use internal context and your base knowledge for validation.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

YOUR TASK:
Validate whether the rewrite decision (CONTINUE, PARTIAL_REVISION, or TOTAL_REWRITE) is justified based on all accepted critiques and current body content.

VALIDATION CRITERIA - Consider:

**ACCEPT "continue" decision if:**
- Critiques are indeed minor or incorrect
- Body is fundamentally sound despite critique issues
- Issues can be addressed without any revision
- Title and body remain aligned

**ACCEPT "partial_revision" decision if:**
- Critiques identify specific, localized issues
- Proposed edit operations would fix the identified problems
- Edit operations are appropriate (correct operation types, reasonable old_string/new_string)
- Most of the body is sound, only targeted fixes needed
- Title change (if proposed) is justified
- Outline update (if proposed) improves structure

**ACCEPT "total_rewrite" decision if:**
- Critiques reveal catastrophic issues (pervasive math errors, fundamental misalignment, structural chaos)
- Total rewrite is justified - issues too widespread for targeted edits
- Partial revision would not be sufficient
- Title change (if proposed) is justified by scope drift
- Outline update (if proposed) improves structure

**REJECT decision if:**
- Reasoning doesn't match the critiques (illogical conclusion)
- "Continue" chosen despite substantive issues in critiques
- "Total_rewrite" chosen for minor or fixable issues (should use partial_revision)
- "Partial_revision" chosen but edit operations are vague or incorrect
- Title change proposed without justification from critiques
- Decision appears arbitrary or not evidence-based

SOURCE MATERIAL POLICY:
- The source database is optional support, not a mandatory checklist
- Do NOT reject a decision solely because it leaves some source material unused
- Do reject if the decision ignores source material only when that omission clearly makes the chosen scope weaker, incoherent, or misaligned with the outline/title

Ask yourself: "Is this decision the right response to the accepted critiques? Is the chosen level of revision appropriate?"

Output your decision ONLY as JSON in this exact format:
{
  "decision": "accept or reject",
  "reasoning": "Detailed explanation of your validation decision"
}
"""


def get_rewrite_decision_validation_json_schema() -> str:
    """Get JSON schema specification for rewrite decision validation."""
    return """
REQUIRED JSON FORMAT:
{
  "decision": "accept" OR "reject",
  "reasoning": "string - detailed explanation of your validation decision"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: If your content contains mathematical expressions like \\Delta, \\tau, etc., 
   you MUST escape the backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (Accept continue decision):
{
  "decision": "accept",
  "reasoning": "The decision to CONTINUE is justified. The critiques are indeed minor issues: 3 stylistic suggestions, 4 notation clarifications, 2 incorrect critiques (the proofs are actually valid), and 1 small gap that can be filled in review phase. No fundamental mathematical errors were identified. Proceeding to conclusion is appropriate."
}

Example (Accept partial_revision decision):
{
  "decision": "accept",
  "reasoning": "The decision to use PARTIAL_REVISION is justified. The critiques identify 2 specific, fixable issues: missing convergence proof in Section III and missing Corollary 3.1. The proposed edit operations correctly target these issues with appropriate old_string/new_string replacements. Most of the body is mathematically sound - targeted edits are more appropriate than a complete rewrite."
}

Example (Accept total_rewrite decision):
{
  "decision": "accept",
  "reasoning": "The decision to use TOTAL_REWRITE is justified. The critiques reveal catastrophic problems: 4 critiques identify fundamental errors in the convergence arguments that permeate multiple sections, 3 point out missing content explicitly in the outline, 2 show the body has drifted to cover different scope than the title. These issues are too pervasive for targeted edits. A complete rebuild is necessary."
}

Example (Reject - should use partial_revision instead):
{
  "decision": "reject",
  "reasoning": "The decision to use TOTAL_REWRITE is NOT justified. The critiques identify only 2 specific issues: missing convergence proof in Section III and missing corollary. These are localized problems that can be fixed with targeted edits. The rest of the body is mathematically sound. The decision should be PARTIAL_REVISION, not TOTAL_REWRITE."
}
"""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================


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
    """
    Build complete prompt for critique generation.
    
    Args:
        user_prompt: The user's compiler-directing prompt
        current_body: The body section to critique
        current_outline: The paper outline
        aggregator_db: The aggregator database content
        reference_papers: Optional reference paper content
        critique_feedback: Optional existing critiques (for context)
        rejection_feedback: Optional rejection feedback (last 5 rejections)
        accumulated_history: Optional accumulated critique history from previous failed versions
        
    Returns:
        Complete assembled prompt
    """
    parts = [
        get_critique_submitter_system_prompt(),
        "\n---\n",
        get_critique_json_schema(),
        "\n---\n",
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"PAPER TITLE:\n{user_prompt}",  # Using compiler prompt as title context
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT BODY SECTION (to critique):\n{current_body}",
        "\n---\n",
        """OPTIONAL SOURCE MATERIAL POLICY:
- The source database below is optional support, not a mandatory checklist.
- Use it to identify genuine gaps or contradictions if helpful.
- Do NOT critique solely because some source entries were not used.
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
    
    return ''.join(parts)


def build_rewrite_decision_prompt(
    user_prompt: str,
    current_body: str,
    current_outline: str,
    current_title: str,
    aggregator_db: str,
    critique_feedback: str,
    pre_critique_paper: str,
    reference_papers: Optional[str] = None,
    accumulated_history: Optional[str] = None
) -> str:
    """
    Build complete prompt for rewrite vs continue decision.
    
    Args:
        user_prompt: The user's compiler-directing prompt
        current_body: The body section being evaluated
        current_outline: The paper outline
        current_title: The current paper title
        aggregator_db: The aggregator database content
        critique_feedback: All accepted critiques (typically 1-3 out of 5 total attempts)
        pre_critique_paper: Paper snapshot from START of critique phase (for rewrite context)
        reference_papers: Optional reference paper content
        accumulated_history: Optional accumulated critique history from previous failed versions
        
    Returns:
        Complete assembled prompt
    """
    parts = [
        get_rewrite_decision_system_prompt(),
        "\n---\n",
        get_rewrite_decision_json_schema(),
        "\n---\n",
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT PAPER TITLE:\n{current_title}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"PRE-CRITIQUE PAPER (body at START of this revision cycle):\n{pre_critique_paper}",
        "\n---\n",
        f"CURRENT BODY SECTION (after critique phase):\n{current_body}",
    ]
    
    if accumulated_history:
        parts.extend([
            "\n---\n",
            accumulated_history
        ])
    
    parts.extend([
        "\n---\n",
        f"ALL ACCEPTED CRITIQUES (CURRENT VERSION):\n{critique_feedback}",
        "\n---\n",
        """OPTIONAL SOURCE MATERIAL POLICY:
- The source database below is optional support, not a mandatory checklist.
- Use it if it helps judge whether the body's chosen scope is genuinely weak, incomplete, or misaligned.
- Do NOT force rewrite solely to cover unused source material.
""",
        "\n---\n",
        f"SOURCE DATABASE (optional support - use if helpful):\n{aggregator_db}",
    ])
    
    if reference_papers:
        parts.extend([
            "\n---\n",
            f"REFERENCE PAPERS:\n{reference_papers}"
        ])
    
    parts.extend([
        "\n---\n",
        "Review all critiques and decide whether to REWRITE the body or CONTINUE to conclusion. Respond as JSON:"
    ])
    
    return ''.join(parts)


def build_rewrite_decision_validation_prompt(
    user_prompt: str,
    current_body: str,
    current_outline: str,
    current_title: str,
    critique_feedback: str,
    decision_result: Dict,
    aggregator_db: str
) -> str:
    """
    Build complete prompt for validating the rewrite decision.
    
    Args:
        user_prompt: The user's compiler-directing prompt
        current_body: The body section
        current_outline: The paper outline
        current_title: Current paper title
        critique_feedback: All accepted critiques (typically 1-3 out of 5 total attempts)
        decision_result: The decision being validated
        aggregator_db: The aggregator database content
        
    Returns:
        Complete assembled prompt
    """
    decision = decision_result.get('decision', 'unknown')
    new_title = decision_result.get('new_title', None)
    new_outline = decision_result.get('new_outline', None)
    reasoning = decision_result.get('reasoning', '')
    
    parts = [
        get_rewrite_decision_validator_system_prompt(),
        "\n---\n",
        get_rewrite_decision_validation_json_schema(),
        "\n---\n",
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT PAPER TITLE:\n{current_title}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT BODY SECTION:\n{current_body}",
        "\n---\n",
        f"ALL ACCEPTED CRITIQUES:\n{critique_feedback}",
        "\n---\n",
        """OPTIONAL SOURCE MATERIAL POLICY:
- The source database below is optional support, not a mandatory checklist.
- Use it if needed to judge whether the proposed decision is genuinely stronger or weaker.
- Do NOT reject solely because not all source material is being used.
""",
        "\n---\n",
        f"SOURCE DATABASE (optional support - use if helpful):\n{aggregator_db}",
        "\n---\n",
        f"PROPOSED DECISION:\n",
        f"Decision: {decision}\n",
        f"New Title: {new_title if new_title else '(keep current)'}\n",
        f"New Outline: {new_outline if new_outline else '(keep current)'}\n",
        f"Reasoning: {reasoning}",
        "\n---\n",
        "Validate whether this decision is justified based on the critiques. Respond as JSON:"
    ]
    
    return ''.join(parts)


# ============================================================================
# ITERATIVE PARTIAL REVISION PROMPTS
# ============================================================================

def get_iterative_edit_system_prompt() -> str:
    """System prompt for iterative partial revision - proposing one edit at a time."""
    return """You are making targeted edits to a mathematical document body to address peer review critiques.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (papers, outlines, critiques) is AI-GENERATED within this research system. 
This content has NOT been peer-reviewed, published, or verified by external sources.
Treat all provided context with extreme skepticism.

YOUR TASK:
You are in an ITERATIVE EDIT LOOP. You have been shown:
1. The PRE-CRITIQUE PAPER (how the body looked before this revision cycle started)
2. The CURRENT PAPER (the body after any edits applied so far in this loop)
3. The ACCEPTED CRITIQUES (problems identified that need fixing)
4. The EDITS ALREADY APPLIED (what has been changed so far)

Your job is to propose ONE EDIT at a time to address the remaining critique issues.
After each edit is validated and applied, you will see the updated paper and can propose the next edit.

EDIT OPERATIONS USE EXACT STRING MATCHING:
- old_string must exist VERBATIM and UNIQUELY in the CURRENT paper body
- Include enough context (3-5 lines) to ensure uniqueness
- If the exact string is not found or is ambiguous, the edit will be rejected

OPERATION TYPES:
- **replace**: Find old_string, replace with new_string
- **insert_after**: Find old_string, insert new_string immediately after it
- **delete**: Find old_string, remove it (new_string should be empty)

WHEN TO SET more_edits_needed:
- TRUE: More critique issues remain to be addressed
- FALSE: All critique issues have been addressed (or best effort has been made)

IMPORTANT:
- Focus on ONE edit at a time
- Address the most critical issues first
- Each edit should be substantial and address specific critique feedback
- Do NOT make cosmetic changes - focus on mathematical/structural issues identified in critiques
- If you believe all issues are addressed, set more_edits_needed to false
- If critique issues involve unsupported empirical or artifact claims, remove them or rewrite them as hypotheses, validation plans, expected benefits, limitations, or future work
- Never preserve fabricated experiments, unsupported benchmark numbers, or nonexistent code claims as if they were verified

Output your response ONLY as JSON in the exact format specified.
"""


def get_iterative_edit_json_schema() -> str:
    """Get JSON schema for iterative edit response."""
    return """
REQUIRED JSON FORMAT:
{
  "operation": "replace | insert_after | delete",
  "old_string": "Exact text to find in the CURRENT paper body (must be unique)",
  "new_string": "Replacement/insertion text (empty string for delete)",
  "reasoning": "Which critique issue this edit addresses and why this change fixes it",
  "more_edits_needed": true OR false
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
   - Example: "He said \\"hello\\"" 
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs (NOT \\\\t)
   - Example: "Line 1\\nLine 2" creates two lines
4. DO NOT use single backslashes except for: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
5. LaTeX notation: MUST escape backslash: write "\\\\Delta", "\\\\tau", "\\\\[", "\\\\]"

Example (Replace - Fix missing proof):
{
  "operation": "replace",
  "old_string": "The proof assumes uniform convergence without establishing the necessary conditions.",
  "new_string": "We establish uniform convergence via the Weierstrass M-test. The series satisfies |f_n(x)| ≤ M_n with ∑M_n < ∞, therefore uniform convergence follows immediately.",
  "reasoning": "Critique #1 identified that the convergence proof was assumed rather than proven. Adding the rigorous justification using Weierstrass M-test.",
  "more_edits_needed": true
}

Example (Insert After - Add missing corollary):
{
  "operation": "insert_after",
  "old_string": "This completes the proof of Theorem 3. ∎",
  "new_string": "\\n\\nCorollary 3.1. As an immediate consequence of Theorem 3, we obtain the following bound on the error term:\\n\\n|R_n(x)| ≤ C · n^{-α}\\n\\nfor some constant C > 0 independent of n.",
  "reasoning": "Critique #3 noted that Corollary 3.1 from the outline was missing. Adding it directly after the proof of Theorem 3 where it logically belongs.",
  "more_edits_needed": false
}

Example (Delete - Remove redundant section):
{
  "operation": "delete",
  "old_string": "We pause to note that this result is analogous to several classical results in the literature, including the work of Smith (1995), Jones (2001), and Brown (2010). While a full comparison is beyond the scope of this paper, the interested reader may consult these references for additional context.",
  "new_string": "",
  "reasoning": "Critique #2 identified this paragraph as redundant filler that doesn't add mathematical substance. Removing to improve focus.",
  "more_edits_needed": true
}
"""


def build_iterative_edit_prompt(
    user_prompt: str,
    pre_critique_paper: str,
    current_paper: str,
    current_outline: str,
    critique_feedback: str,
    edits_applied: List[Dict],
    reference_papers: Optional[str] = None,
    accumulated_critique_history: str = ""
) -> str:
    """
    Build prompt for iterative partial revision edit.
    
    Args:
        user_prompt: The user's compiler-directing prompt
        pre_critique_paper: Paper snapshot from START of critique phase
        current_paper: Current paper body (after any edits applied so far)
        current_outline: The paper outline
        critique_feedback: All accepted critiques from this revision cycle
        edits_applied: List of edits already applied in this iteration
        reference_papers: Optional reference paper content
        accumulated_critique_history: Critiques from previous failed versions (if any)
        
    Returns:
        Complete assembled prompt
    """
    parts = [
        get_iterative_edit_system_prompt(),
        "\n---\n",
        get_iterative_edit_json_schema(),
        "\n---\n",
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
    ]
    
    # Add accumulated history if present
    if accumulated_critique_history:
        parts.extend([
            f"ACCUMULATED CRITIQUE HISTORY (from previous failed versions):\n{accumulated_critique_history}",
            "\n---\n",
        ])
    
    parts.extend([
        f"ACCEPTED CRITIQUES (issues to address):\n{critique_feedback}",
        "\n---\n",
        f"PRE-CRITIQUE PAPER (how the body looked before this revision cycle):\n{pre_critique_paper}",
        "\n---\n",
        f"CURRENT PAPER (after {len(edits_applied)} edit(s) applied):\n{current_paper}",
        "\n---\n",
    ])

    if reference_papers:
        parts.extend([
            f"REFERENCE PAPERS:\n{reference_papers}",
            "\n---\n",
        ])
    
    # Show edits already applied
    if edits_applied:
        edits_str = "\n".join([
            f"Edit {i+1}: {e['operation']} - {e.get('reasoning', 'N/A')[:100]}..."
            for i, e in enumerate(edits_applied)
        ])
        parts.extend([
            f"EDITS ALREADY APPLIED:\n{edits_str}",
            "\n---\n",
        ])
    else:
        parts.extend([
            "EDITS ALREADY APPLIED: None yet - this is the first edit.",
            "\n---\n",
        ])
    
    parts.append(
        "Propose your NEXT edit to address remaining critique issues, or set more_edits_needed=false if all issues are resolved. Respond as JSON:"
    )
    
    return ''.join(parts)


# ============================================================================
# PARTIAL REVISION EDIT VALIDATION PROMPTS
# ============================================================================

def get_partial_revision_validation_system_prompt() -> str:
    """System prompt for validating individual partial revision edits."""
    return """You are validating a proposed edit to a mathematical document.

The edit is part of an iterative partial revision to address peer review critiques.

EMPIRICAL PROVENANCE RULES:
- Empirical claims (benchmarks, speedups, latency, accuracy, perplexity, hardware measurements) must not remain stated as fact unless backed by explicit citation or provided artifact support.
- Artifact claims (code, kernels, experiments, logs, accompanying implementations) must not remain stated as fact unless backed by explicit citation or provided artifact support.
- If the edit rewrites unsupported empirical/artifact claims into hypotheses, validation plans, expected benefits, limitations, or future work, that is a valid improvement.

YOUR TASK:
Validate whether this specific edit should be ACCEPTED or REJECTED.

ACCEPT the edit if:
1. It addresses one or more issues identified in the accepted critiques
2. The old_string exists in the current paper and is unambiguous
3. The new_string improves the mathematical content or addresses critique feedback
4. The edit maintains coherence with the surrounding text
5. The edit is mathematically sound

REJECT the edit if:
1. The edit does NOT address any critique issues
2. The old_string does not exist or is ambiguous in the current paper
3. The new_string introduces errors or reduces quality
4. The edit breaks coherence with surrounding content
5. The edit is mathematically unsound or introduces logical errors
6. The edit is purely cosmetic and doesn't address critiques
7. The edit preserves fabricated experiments, unsupported metrics, or nonexistent artifact claims as established fact

Output your decision as JSON.
"""


def get_partial_revision_validation_json_schema() -> str:
    """Get JSON schema for partial revision edit validation."""
    return """
REQUIRED JSON FORMAT:
{
  "decision": "accept" OR "reject",
  "reasoning": "string - explanation of why the edit should or should not be accepted"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines (NOT \\\\n), \\t for tabs

Example (Accept):
{
  "decision": "accept",
  "reasoning": "The edit correctly addresses critique #1 which identified a missing convergence proof. The new text adds a rigorous Weierstrass M-test argument that establishes uniform convergence. The old_string exists exactly as specified in the current paper."
}

Example (Reject):
{
  "decision": "reject",
  "reasoning": "The proposed edit does not address any of the accepted critiques. It appears to be a stylistic change (rewording a sentence) rather than fixing the mathematical issues identified. Additionally, the old_string appears twice in the document, making it ambiguous."
}
"""


def build_partial_revision_validation_prompt(
    current_paper: str,
    current_outline: str,
    critique_feedback: str,
    edit_proposal: Dict
) -> str:
    """
    Build prompt for validating a single partial revision edit.
    
    Args:
        current_paper: Current paper body
        current_outline: Paper outline
        critique_feedback: All accepted critiques
        edit_proposal: Dict with operation, old_string, new_string, reasoning
        
    Returns:
        Complete assembled validation prompt
    """
    operation = edit_proposal.get("operation", "")
    old_string = edit_proposal.get("old_string", "")
    new_string = edit_proposal.get("new_string", "")
    reasoning = edit_proposal.get("reasoning", "")
    
    # Truncate long strings for prompt
    old_str_display = old_string[:500] + "..." if len(old_string) > 500 else old_string
    new_str_display = new_string[:500] + "..." if len(new_string) > 500 else new_string
    
    parts = [
        get_partial_revision_validation_system_prompt(),
        "\n---\n",
        get_partial_revision_validation_json_schema(),
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"ACCEPTED CRITIQUES (issues being addressed):\n{critique_feedback}",
        "\n---\n",
        f"CURRENT PAPER:\n{current_paper}",
        "\n---\n",
        f"PROPOSED EDIT:\n",
        f"Operation: {operation}\n",
        f"Old String: {old_str_display}\n",
        f"New String: {new_str_display}\n",
        f"Reasoning: {reasoning}",
        "\n---\n",
        "Validate whether this edit should be accepted. Respond as JSON:"
    ]
    
    return ''.join(parts)

