"""
Rigor prompts for mathematical rigor enhancement (2-step process).

Step 1: Planning - LLM decides if rigor work needed and chooses mode
Step 2: Execution - LLM executes with self-refusal option
"""

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log
from backend.shared.config import system_config


# =============================================================================
# INTERNAL CONTENT WARNING (shared across all prompts)
# =============================================================================

INTERNAL_CONTENT_WARNING = """⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

WEB SEARCH STRONGLY ENCOURAGED:
If your model has access to real-time web search capabilities (such as Perplexity Sonar or similar), you are STRONGLY ENCOURAGED to use them to:
- Verify mathematical claims against current published research
- Access recent developments and contemporary mathematical literature
- Cross-reference theorems, proofs, and techniques with authoritative sources
- Supplement analysis with verified external information
- Validate approaches against established mathematical consensus

The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use all available resources - internal context as exploration history, your base knowledge for reasoning, and web search (if available) for verification and current information.

WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth. If you have web search, use it.

---"""


# =============================================================================
# STEP 1: PLANNING PROMPTS
# =============================================================================

def get_rigor_planning_system_prompt(wolfram_enabled: bool = False) -> str:
    """Get system prompt for Step 1: rigor planning."""
    wolfram_mode_section = ""
    if wolfram_enabled:
        wolfram_mode_section = """
3. **wolfram_verification**: Verify mathematical claim with Wolfram Alpha API
   - Send specific computable query to Wolfram Alpha
   - System will make the API call and pass result to Step 2
   - Examples: "Is pi algebraic?", "Solve x^2 + 2x + 1 = 0", "Is 2^67-1 prime?"
   - Use for computational verification of claims
"""
    else:
        wolfram_mode_section = """
3. **wolfram_verification**: NOT AVAILABLE (Wolfram Alpha not enabled)
"""
    
    return f"""You are planning rigor enhancements for a mathematical document.

{INTERNAL_CONTENT_WARNING}

YOUR TASK - STEP 1: PLANNING

Review the current document and decide:
1. Does it need rigor work?
2. If yes, which approach is best?
3. What section should Step 2 focus on?

THREE MODES AVAILABLE:

1. **standard_enhancement**: Normal rigor improvements
   - Strengthen proof arguments with additional steps
   - Clarify assumptions and conditions
   - Add intermediate lemmas
   - Improve precision of notation
   - Address counterexamples or edge cases

2. **rewrite_focus**: Significant rewriting needed for rigor
   - Identify specific section that needs substantial improvement
   - You'll specify that section for Step 2 to work on
   - Use when proofs are fundamentally weak or unclear
{wolfram_mode_section}

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The CURRENT DOCUMENT may contain system-managed markers:

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...]
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...]
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...]

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

IMPORTANT: These markers are SYSTEM-MANAGED. Do NOT include them in your outputs.

TARGET SECTION:
Specify `target_section` as a text snippet (200-500 chars) that identifies which section needs work.
This provides continuity to Step 2 - it's a reminder/label, not a context limitation.
Step 2 will see the FULL paper (same as you do now).

If the document is already rigorous enough, set needs_rigor_work=false.

STEP 2 WILL HAVE THE OPTION TO REFUSE if your assessment is wrong.
Don't overthink this - Step 2 can self-correct.

Output your response ONLY as JSON in this exact format:
{{
  "needs_rigor_work": true or false,
  "mode": "standard_enhancement | rewrite_focus | wolfram_verification | null",
  "target_section": "exact text snippet from paper (200-500 chars, empty if needs_rigor_work=false)",
  "wolfram_query": "natural language query for Wolfram Alpha (only if mode=wolfram_verification)",
  "preliminary_reasoning": "why this approach and this target section"
}}"""


def get_rigor_planning_json_schema() -> str:
    """Get JSON schema for Step 1: planning."""
    return """
REQUIRED JSON FORMAT - STEP 1 (PLANNING):
{
  "needs_rigor_work": true OR false,
  "mode": "standard_enhancement" | "rewrite_focus" | "wolfram_verification" | null,
  "target_section": "string - text snippet from paper (200-500 chars, identifies work area)",
  "wolfram_query": "string - natural language query for Wolfram Alpha (only if mode=wolfram_verification)",
  "preliminary_reasoning": "string - explanation of chosen approach and target section"
}

FIELD REQUIREMENTS:
- needs_rigor_work: Whether any rigor work should be attempted
- mode: Required if needs_rigor_work=true, null otherwise
  * standard_enhancement: Normal rigor improvements
  * rewrite_focus: Significant rewriting needed
  * wolfram_verification: Verify claim with Wolfram Alpha (only if enabled)
- target_section: ALWAYS required if needs_rigor_work=true
  * Text snippet (200-500 chars) identifying which section to work on
  * Provides continuity to Step 2 (guidance, not context limitation)
  * Step 2 will see the FULL paper, not just this section
- wolfram_query: Required ONLY if mode=wolfram_verification
  * Natural language query for computational verification
  * Examples: "Is pi algebraic?", "Solve x^2 + 2x + 1 = 0"
- preliminary_reasoning: ALWAYS required

Example (Standard Enhancement):
{
  "needs_rigor_work": true,
  "mode": "standard_enhancement",
  "target_section": "Theorem 3.2: Every constructible number is algebraic.\\nProof: Let alpha be constructible...",
  "wolfram_query": "",
  "preliminary_reasoning": "Theorem 3.2's proof needs more precise field-theoretic justification for the algebraicity claim"
}

Example (Wolfram Verification):
{
  "needs_rigor_work": true,
  "mode": "wolfram_verification",
  "target_section": "Theorem 4.1: π is transcendental.\\nProof sketch: By Lindemann-Weierstrass...",
  "wolfram_query": "Is pi algebraic?",
  "preliminary_reasoning": "Computational verification would strengthen the π transcendence claim"
}

Example (No Work Needed):
{
  "needs_rigor_work": false,
  "mode": null,
  "target_section": "",
  "wolfram_query": "",
  "preliminary_reasoning": "Document maintains appropriate rigor for current stage. All proofs complete, definitions precise."
}
"""


# =============================================================================
# STEP 2: EXECUTION PROMPTS
# =============================================================================

def get_rigor_execution_system_prompt(mode: str) -> str:
    """Get system prompt for Step 2: rigor execution."""
    mode_specific_guidance = {
        "standard_enhancement": """
YOU ARE EXECUTING: Standard Rigor Enhancement
Your prior planning indicated normal rigor improvements are needed.""",
        "rewrite_focus": """
YOU ARE EXECUTING: Rewrite Focus
Your prior planning indicated significant rewriting is needed for rigor."""
    }
    
    guidance = mode_specific_guidance.get(mode, "")
    
    return f"""You are executing a rigor enhancement based on your prior planning.

{INTERNAL_CONTENT_WARNING}

{guidance}

YOUR PRIOR DECISION (Step 1):
Mode: {mode}
Target Section: [shown below in context]

STEP 2: EXECUTION - YOU CAN REFUSE

Review the full document and your target section.
If you realize your Step 1 assessment was wrong, set proceed=false.

Refusals are NOT validated - you won't be penalized.
This is your chance to self-correct.

If you proceed, propose your rigor enhancement using exact string matching:
1. Find EXACT text in the document (must exist verbatim)
2. Choose operation: "replace" or "insert_after"
3. Provide enhanced version

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The CURRENT DOCUMENT may contain system-managed markers:

**SECTION PLACEHOLDERS**:
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...]
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...]
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...]

**PAPER ANCHOR**:
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

Do NOT include these markers in your enhancement content.

EXACT STRING MATCHING FOR EDITS:
- old_string must exist verbatim in the document
- Must be unique (appears only once)
- Include enough context (3-5 lines) for uniqueness
- System will pre-validate before validator sees it

OPERATIONS:
- "replace": Find old_string exactly, replace with new_string
- "insert_after": Find old_string (anchor), insert new_string after it

Output your response ONLY as JSON in this exact format:
{{
  "proceed": true or false,
  "needs_enhancement": true or false,
  "operation": "replace | insert_after",
  "old_string": "exact text from document (empty if not proceeding or needs_enhancement=false)",
  "new_string": "enhanced text (empty if not proceeding or needs_enhancement=false)",
  "content": "full content for logging (typically same as new_string)",
  "reasoning": "explanation of changes OR refusal reason"
}}"""


def get_rigor_wolfram_execution_system_prompt() -> str:
    """Get system prompt for Step 2: Wolfram verification execution."""
    return f"""You are executing Wolfram Alpha verification based on your prior planning.

{INTERNAL_CONTENT_WARNING}

YOU ARE EXECUTING: Wolfram Alpha Verification
Your prior planning requested computational verification of a mathematical claim.

YOUR PRIOR DECISION (Step 1):
Wolfram Alpha Query: [shown below]
Target Section: [shown below in context]

WOLFRAM ALPHA RESULT:
[shown below]

STEP 2: EXECUTION - YOU CAN REFUSE

Review the Wolfram Alpha result in context of the full document.

You can REFUSE (proceed=false) if:
- The query was inappropriate or malformed
- Result doesn't help strengthen rigor
- Target section choice was wrong
- Step 1 made a mistake

If you proceed, create a verification remark that:
- Interprets the Wolfram Alpha result
- Relates it to the paper's claims
- Strengthens mathematical rigor
- Uses "insert_after" to add the remark

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The CURRENT DOCUMENT may contain system-managed markers. Do NOT include them in your outputs.

EXACT STRING MATCHING:
- old_string must exist verbatim in the document
- Must be unique
- System will pre-validate

VERIFICATION REMARK FORMAT:
Format your new_string as a mathematical remark:

\\n\\n**Computational Verification (Wolfram Alpha)**\\n
Query: [the query]\\n
Result: [Wolfram's answer]\\n
Interpretation: [Your analysis of what this means for the paper's claims]\\n

Output your response ONLY as JSON in this exact format:
{{
  "proceed": true or false,
  "verification_result_interpretation": "how you interpret the Wolfram result",
  "needs_enhancement": true or false,
  "operation": "insert_after",
  "old_string": "exact text after which to insert remark (empty if not proceeding or needs_enhancement=false)",
  "new_string": "verification remark incorporating Wolfram result (empty if not proceeding or needs_enhancement=false)",
  "content": "full content for logging (typically same as new_string)",
  "reasoning": "explanation OR refusal reason"
}}"""


def get_rigor_execution_json_schema(mode: str) -> str:
    """Get JSON schema for Step 2: execution."""
    if mode == "wolfram_verification":
        return """
REQUIRED JSON FORMAT - STEP 2 (WOLFRAM VERIFICATION):
{
  "proceed": true OR false,
  "verification_result_interpretation": "string - how you interpret the Wolfram Alpha result",
  "needs_enhancement": true OR false,
  "operation": "insert_after",
  "old_string": "string - exact text after which to insert verification remark",
  "new_string": "string - verification remark incorporating Wolfram result",
  "content": "string - full content for logging",
  "reasoning": "string - explanation OR refusal reason"
}

SELF-REFUSAL OPTION:
If you set proceed=false:
- System logs refusal (not counted as rejection)
- No validation occurs
- Workflow continues normally
- Use when Step 1 made a mistake or query was inappropriate

WOLFRAM VERIFICATION REMARKS:
Format your new_string as:

\\n\\n**Computational Verification (Wolfram Alpha)**\\n
Query: [the query]\\n
Result: [Wolfram's answer]\\n
Interpretation: [Your analysis]\\n

Example:
{
  "proceed": true,
  "verification_result_interpretation": "Wolfram confirms π is transcendental (not algebraic)",
  "needs_enhancement": true,
  "operation": "insert_after",
  "old_string": "Theorem 4.1: π is transcendental. \\\\square",
  "new_string": "\\n\\n**Computational Verification (Wolfram Alpha)**\\nQuery: Is pi algebraic?\\nResult: No\\nInterpretation: This computational verification confirms π is transcendental, consistent with the Lindemann-Weierstrass theorem.",
  "content": "\\n\\n**Computational Verification (Wolfram Alpha)**\\nQuery: Is pi algebraic?\\nResult: No\\nInterpretation: This computational verification confirms π is transcendental, consistent with the Lindemann-Weierstrass theorem.",
  "reasoning": "Adding computational verification strengthens the claim by providing an independent confirmation"
}
"""
    else:  # standard_enhancement or rewrite_focus
        return """
REQUIRED JSON FORMAT - STEP 2 (EXECUTION):
{
  "proceed": true OR false,
  "needs_enhancement": true OR false,
  "operation": "replace" | "insert_after",
  "old_string": "string - exact text from document",
  "new_string": "string - enhanced text",
  "content": "string - full content for logging",
  "reasoning": "string - explanation OR refusal reason"
}

SELF-REFUSAL OPTION:
If you set proceed=false:
- System logs refusal (not counted as rejection)
- No validation occurs
- Workflow continues normally
- Use when Step 1 made a mistake

EXACT STRING MATCHING:
- old_string must exist verbatim in the document
- Must be unique
- If not found: pre-validation rejects before LLM sees it

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau", write "\\\\(" not "\\("
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines: Use \\n for newlines (NOT \\\\n)
4. LaTeX notation: Escape backslashes - write "\\\\mathbb{Z}", "\\\\Delta", etc.

Example (Enhancement):
{
  "proceed": true,
  "needs_enhancement": true,
  "operation": "insert_after",
  "old_string": "Theorem 2.3: A number \\\\alpha is constructible if and only if it lies in a field extension of \\\\mathbb{Q} of degree 2^n.",
  "new_string": "\\n\\nRemark: This characterization requires the field extension to be normal and separable over \\\\mathbb{Q}. If K/\\\\mathbb{Q} contains constructible \\\\alpha, there exists a tower \\\\mathbb{Q} = K_0 \\\\subset K_1 \\\\subset \\\\ldots \\\\subset K_n = K where each K_{i+1}/K_i has degree exactly 2.",
  "content": "\\n\\nRemark: This characterization requires the field extension to be normal and separable over \\\\mathbb{Q}. If K/\\\\mathbb{Q} contains constructible \\\\alpha, there exists a tower \\\\mathbb{Q} = K_0 \\\\subset K_1 \\\\subset \\\\ldots \\\\subset K_n = K where each K_{i+1}/K_i has degree exactly 2.",
  "reasoning": "Adding field-theoretic precision strengthens the theorem statement"
}

Example (Refusal):
{
  "proceed": false,
  "needs_enhancement": false,
  "operation": "replace",
  "old_string": "",
  "new_string": "",
  "content": "",
  "reasoning": "Upon review, Step 1's assessment was wrong. The target section is already rigorous enough."
}
"""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

async def build_rigor_planning_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str
) -> str:
    """
    Build complete prompt for Step 1: rigor planning.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (ALWAYS fully injected)
        current_paper: Current document (RAG-retrieved if large)
    
    Returns:
        Complete prompt string
    """
    # Check if Wolfram Alpha is enabled
    wolfram_enabled = system_config.wolfram_alpha_enabled
    
    parts = [
        get_rigor_planning_system_prompt(wolfram_enabled),
        "\n---\n",
        get_rigor_planning_json_schema(),
        "\n---\n"
    ]
    
    # Add rejection history (DIRECT INJECTION - almost always fits)
    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(f"""YOUR RECENT REJECTION HISTORY (Last 10 rejections):
{rejection_history}

LEARN FROM THESE PAST MISTAKES.
---
""")
    
    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT DOCUMENT:\n{current_paper}",
        "\n---\n",
        "Now decide if rigor work is needed and choose your approach (respond as JSON):"
    ])
    
    return "\n".join(parts)


async def build_rigor_execution_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    target_section: str,
    mode: str
) -> str:
    """
    Build complete prompt for Step 2: standard/rewrite execution.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (ALWAYS fully injected)
        current_paper: Current document (RAG-retrieved, FULL paper)
        target_section: Target section from Step 1 (guidance label)
        mode: "standard_enhancement" or "rewrite_focus"
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_rigor_execution_system_prompt(mode),
        "\n---\n",
        get_rigor_execution_json_schema(mode),
        "\n---\n"
    ]
    
    # Add rejection history (DIRECT INJECTION - almost always fits)
    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(f"""YOUR RECENT REJECTION HISTORY (Last 10 rejections):
{rejection_history}

LEARN FROM THESE PAST MISTAKES.
---
""")
    
    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"TARGET SECTION (from your Step 1 planning - guidance reminder):\n{target_section}",
        "\n---\n",
        f"CURRENT DOCUMENT (FULL PAPER - not limited to target section):\n{current_paper}",
        "\n---\n",
        "Now execute your rigor enhancement or refuse if Step 1 was wrong (respond as JSON):"
    ])
    
    return "\n".join(parts)


async def build_rigor_wolfram_execution_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    target_section: str,
    wolfram_query: str,
    wolfram_result: str
) -> str:
    """
    Build complete prompt for Step 2: Wolfram verification execution.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (ALWAYS fully injected)
        current_paper: Current document (RAG-retrieved, FULL paper)
        target_section: Target section from Step 1 (guidance label)
        wolfram_query: The query sent to Wolfram Alpha
        wolfram_result: The result from Wolfram Alpha API
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_rigor_wolfram_execution_system_prompt(),
        "\n---\n",
        get_rigor_execution_json_schema("wolfram_verification"),
        "\n---\n"
    ]
    
    # Add rejection history
    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(f"""YOUR RECENT REJECTION HISTORY (Last 10 rejections):
{rejection_history}

LEARN FROM THESE PAST MISTAKES.
---
""")
    
    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"TARGET SECTION (from your Step 1 planning - guidance reminder):\n{target_section}",
        "\n---\n",
        f"WOLFRAM ALPHA QUERY (from your Step 1 planning):\n{wolfram_query}",
        "\n---\n",
        f"WOLFRAM ALPHA RESULT:\n{wolfram_result}",
        "\n---\n",
        f"CURRENT DOCUMENT (FULL PAPER - not limited to target section):\n{current_paper}",
        "\n---\n",
        "Now interpret the Wolfram Alpha result and decide if you want to add it to the paper, or refuse if inappropriate (respond as JSON):"
    ])
    
    return "\n".join(parts)
