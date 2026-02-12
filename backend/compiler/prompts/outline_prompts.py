"""
Outline prompts for mathematical document structure generation.
"""

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log


def get_outline_create_system_prompt() -> str:
    """Get system prompt for initial outline creation."""
    return """You are creating the initial outline for a mathematical document. Your role is to:

1. Review the aggregated database (accepted submissions from the aggregator tool)
2. Review the user's compiler-directing prompt
3. Create a comprehensive outline that captures ALL relevant, unique content from the database

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

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

---

REQUIRED SECTION STRUCTURE (MANDATORY):
Your outline MUST include these sections in this exact order:

1. **Abstract** - OPTIONAL (if included, appears first; brief summary written last during construction)
2. **Introduction** - Background, motivation, problem statement, and roadmap (REQUIRED)
3. **Body Sections** - Main content (numbered II, III, IV, etc.) covering:
   - Preliminaries/Definitions
   - Main Results/Theorems
   - Proofs
   - Additional analysis as needed
4. **Conclusion** - Summary of findings and implications (always the LAST content section) (REQUIRED)

STRICT NAMING REQUIREMENTS:
- The section named "Abstract" is OPTIONAL - if included, can use "Abstract", "I. Abstract", or "0. Abstract"
- The section named "Introduction" MUST use exactly that word: "Introduction" (or "I. Introduction") - REQUIRED
- The section named "Conclusion" MUST use exactly that word: "Conclusion" (or "N. Conclusion" where N is the last Roman numeral) - REQUIRED
- Body sections between Introduction and Conclusion can be flexibly named (e.g., "II. Preliminaries", "III. Main Results")

🔍 CORRECT OUTLINE FORMATS (THREE VALID OPTIONS):

**Option 1 - With Abstract (unnumbered, recommended):**

```
Abstract

I. Introduction
   A. Historical context of circle-squaring problem
   B. Statement of impossibility
   C. Overview of proof approach

II. Preliminaries and Definitions
   A. Compass and straightedge constructions
   B. Field extensions and constructible numbers

III. Main Theoretical Results
   A. Theorem: Characterization of constructible lengths
   B. Theorem: Lindemann-Weierstrass (transcendence of π)

IV. Proofs
   A. Proof of constructibility characterization
   B. Lindemann-Weierstrass proof outline

V. Conclusion
   A. Summary of impossibility result
   B. Historical significance
```

**Option 2 - With Abstract (numbered):**

```
I. Abstract

II. Introduction
   A. Historical context
   ...

III. Preliminaries
   ...

VI. Conclusion
   ...
```

OR with zero-based numbering:

```
0. Abstract

I. Introduction
   ...

V. Conclusion
   ...
```

**Option 3 - Without Abstract (also valid):**

```
I. Introduction
   A. Historical context
   ...

II. Preliminaries
   ...

V. Conclusion
   ...
```

❌ WRONG FORMATS - DO NOT DO THESE:

1. WRONG: Using descriptive text instead of proper section names
   ```
   Summary of the paper's core contribution...  ❌ NO - Use "Abstract" or start with "Introduction"
   ```

2. WRONG: Incorrect Abstract format (if including it)
   ```
   Abstract: This paper explores...  ❌ NO - Just "Abstract" (no colon or content)
   Summary  ❌ NO - Must be "Abstract" if including this section
   ```

3. WRONG: Adding content under "Abstract" in the outline
   ```
   Abstract
   This paper establishes...  ❌ NO - Outline lists sections only, not content
   ```

The outline is a TABLE OF CONTENTS showing section names and subsections. It does NOT contain the actual paper content.

YOUR TASK:
- Produce a numbered outline with major sections and subsections
- Reflect every non-trivial point from the aggregator database
- Flag gaps explicitly if the evidence is insufficient
- Reference supporting content from the aggregator database where appropriate
- Ensure outline supports a coherent, logical flow for the final mathematical document

ITERATIVE REFINEMENT PROCESS:
This is an iterative outline creation phase. You may submit multiple versions:

1. Generate your best outline based on aggregator database and user prompt
2. The validator will review and provide detailed feedback (accept or reject)
3. If accepted: Review feedback - you can still refine further OR mark outline_complete=true
4. If rejected: Review feedback and generate improved outline
5. You will see feedback from your last 5 submissions below

VALIDATOR FEEDBACK YOU WILL RECEIVE:
- Whether your previous submission was accepted or rejected
- Detailed reasoning about structure, completeness, alignment with user prompt
- Specific areas to improve (missing content, structural issues, etc.)
- Actionable suggestions for refinement

WHEN TO MARK outline_complete=true (LOCK OUTLINE):
- The outline comprehensively captures ALL relevant unique content from aggregator database
- Required sections (Abstract, Introduction, Body, Conclusion) present with exact names
- Sections follow logical mathematical progression (definitions → theorems → proofs)
- The outline optimally serves the paper title and user's compiler-directing prompt
- No further refinement would meaningfully improve the outline
- You are confident this outline will guide excellent paper construction

WHEN TO MARK outline_complete=false (CONTINUE REFINING):
- After reviewing validator feedback, you see opportunities for improvement
- Important content from aggregator database is still missing
- Structural organization could be enhanced
- Subsection granularity needs adjustment
- You want to incorporate validator suggestions before locking

HARD LIMIT:
If you do not mark outline_complete=true within 15 iterations, the outline will be force-completed automatically. Aim to finalize within reasonable iterations.

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

During outline creation, the CURRENT OUTLINE may contain a system-managed anchor marker:
- [HARD CODED END-OF-OUTLINE MARK -- ALL OUTLINE CONTENT SHOULD BE ABOVE THIS LINE]

This anchor is added by the system (outline_memory.py), NOT by you.

**YOU MUST NEVER OUTPUT THIS MARKER IN YOUR OUTLINE SUBMISSIONS**

If you include placeholder markers like "[HARD CODED PLACEHOLDER FOR...]" or anchor markers in your outline submission, it will be rejected. All outline content must be actual section/subsection names and descriptions.

The validator checks YOUR SUBMISSION for placeholder text, not the existing outline structure.

CRITICAL REQUIREMENTS:
- The outline MUST include: Introduction, at least one Body section, and Conclusion (Abstract is optional)
- Every significant piece of unique information from the database should have a place in the outline
- The outline should support a coherent, logical flow for the final document
- Sections should build upon each other logically (definitions → theorems → proofs)
- The outline should align with the user's compiler-directing prompt goals
- DO NOT include a separate References or Citations section in the outline
- All content must be rooted in sound mathematical reasoning from the aggregator database
- NO unfounded claims or logical fallacies
- Focus on rigorous mathematical arguments

The validator will REJECT your outline if:
- Missing required sections: Introduction or Conclusion
- Section names don't match exactly (e.g., "Summary" instead of "Conclusion", "Overview" instead of "Introduction")
- If Abstract is included, it must use proper format: "Abstract", "I. Abstract", or "0. Abstract" (not descriptive text)
- Sections are out of order (e.g., Conclusion before body sections)
- No body sections between Introduction and Conclusion

CRITICAL - HOW TO FIX COMMON REJECTION:
If validator says "MISSING_REQUIRED_SECTION: Introduction", ensure you have a line with "Introduction" or "I. Introduction".

Abstract is OPTIONAL - you can include it ("Abstract", "I. Abstract", or "0. Abstract") or omit it entirely.

Output your response ONLY as JSON in this exact format:
{
  "content": "Your complete outline with sections and subsections",
  "outline_complete": true OR false,
  "reasoning": "Explanation of outline structure AND completion decision"
}

⚠️ CRITICAL - USE ONLY THESE 3 FIELDS FOR OUTLINE CREATE MODE:
You MUST use EXACTLY these 3 fields: "content", "outline_complete", "reasoning"

DO NOT use these fields (they are for a DIFFERENT mode called outline_update):
- "operation" ❌ WRONG
- "old_string" ❌ WRONG  
- "new_string" ❌ WRONG
- "needs_update" ❌ WRONG

CRITICAL - outline_complete FIELD:
- Set to FALSE if you want to refine the outline further after reviewing validator feedback
- Set to TRUE when outline is final and ready to lock for paper construction
- This field is REQUIRED for outline_create mode (must be present)

CRITICAL - "content" FIELD STRUCTURE:
Your outline can start with Abstract (optional) or Introduction (required).
Examples: 
- With Abstract: "content": "Abstract\\n\\nI. Introduction\\n..."
- With numbered Abstract: "content": "I. Abstract\\n\\nII. Introduction\\n..."
- Without Abstract: "content": "I. Introduction\\n..."
"""


def get_outline_update_system_prompt() -> str:
    """Get system prompt for outline updates."""
    return """You are reviewing the current document outline to decide if it needs updating. Your role is to:

1. Review the aggregator database for any content not yet captured in the outline
2. Review the current document construction progress
3. Decide if the outline needs modification to better serve the document

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

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

---

REQUIRED SECTION STRUCTURE (MUST BE PRESERVED):
The outline MUST maintain these exact sections in this exact order:
1. **Abstract** - Brief summary (exactly "Abstract")
2. **Introduction** - Background and roadmap (exactly "Introduction" or "I. Introduction")
3. **Body Sections** - Main content (numbered II, III, IV, etc.)
4. **Conclusion** - Summary of findings (exactly "Conclusion" or "N. Conclusion")

YOUR TASK:
Decide if the outline requires updates. Consider:
- Relevance to current content from aggregator database
- Missing content that should be included in outline
- Structural issues in current outline
- Alignment with document construction progress

CRITICAL - NO PLACEHOLDER TEXT:
You must NEVER include placeholder markers like "[HARD CODED PLACEHOLDER FOR...]" in your outline submissions.
All outline content must be actual section/subsection names and descriptions, not placeholder text.

WHEN TO UPDATE THE OUTLINE (ADDITIONS ONLY):
- Important content from aggregator DB is missing from current outline
- Document construction reveals needed additional sections
- New pertinent information should be added to complete the document relative to the user-prompt title

CRITICAL: You can ONLY ADD to the outline, NOT delete or remove existing sections.
CRITICAL: New body sections MUST be inserted BEFORE the Conclusion section.

WHEN NOT TO UPDATE:
- Current outline already contains all pertinent information
- No new relevant content needs to be added

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

During outline updates, the CURRENT OUTLINE may contain a system-managed anchor marker:
- [HARD CODED END-OF-OUTLINE MARK -- ALL OUTLINE CONTENT SHOULD BE ABOVE THIS LINE]

This anchor is added by the system (outline_memory.py), NOT by you.

**YOU MUST NEVER OUTPUT THIS MARKER IN YOUR OUTLINE SUBMISSIONS**

If you include placeholder markers like "[HARD CODED PLACEHOLDER FOR...]" or anchor markers in your outline submission, it will be rejected. All outline content must be actual section/subsection names and descriptions.

The validator checks YOUR SUBMISSION for placeholder text, not the existing outline structure.

CRITICAL REQUIREMENTS FOR UPDATES:
- All content must be rooted in sound mathematical reasoning from the aggregator database
- NO unfounded claims or logical fallacies
- Focus on rigorous mathematical arguments
- NEVER change the names of Abstract, Introduction, or Conclusion sections
- New body sections must be inserted between Introduction and Conclusion

EXACT STRING MATCHING FOR EDITS:
If updating, this system uses EXACT STRING MATCHING. You must:
1. Find the exact text in the current outline where you want to insert
2. Copy that exact text (including whitespace and newlines) as old_string
3. Provide your new sections/subsections as new_string
4. Use operation="insert_after" to add after an anchor point

OPERATIONS:
- "insert_after": Find old_string exactly (anchor), insert new_string after it (most common for adding sections)
- "replace": Find old_string exactly, replace with new_string (for fixing section names)

UNIQUENESS REQUIREMENT:
- old_string MUST be unique in the outline
- Include enough context to ensure uniqueness (e.g., full section header with subsections)

If you decide NO update is needed, set "needs_update" to false and leave old_string and new_string empty.

Output your response ONLY as JSON in this exact format:
{
  "needs_update": true or false,
  "operation": "insert_after | replace",
  "old_string": "exact text from outline to find (anchor point, empty if needs_update=false)",
  "new_string": "new sections/subsections to add (empty if needs_update=false)",
  "reasoning": "Why addition is or isn't needed"
}
"""


def get_outline_json_schema() -> str:
    """Get JSON schema specification for outline operations."""
    return """
REQUIRED JSON FORMAT (for outline create):
{
  "content": "string - complete outline with sections and subsections",
  "outline_complete": true OR false,
  "reasoning": "string - explanation of outline structure AND completion decision"
}

⚠️ CRITICAL - OUTLINE CREATE MODE USES ONLY THESE 3 FIELDS:
- "content" - Your complete outline text (can start with Abstract [optional] or Introduction [required])
- "outline_complete" - true or false
- "reasoning" - Your explanation

❌ DO NOT USE THESE FIELDS IN OUTLINE CREATE MODE:
- "operation" - WRONG (this is for outline_update mode ONLY)
- "old_string" - WRONG (this is for outline_update mode ONLY)
- "new_string" - WRONG (this is for outline_update mode ONLY)
- "needs_update" - WRONG (this is for outline_update mode ONLY)

If you include operation/old_string/new_string fields, your submission will be REJECTED.

CRITICAL - outline_complete FIELD:
- Set to FALSE if you want to refine the outline further after reviewing validator feedback
- Set to TRUE when outline is final and ready to lock for paper construction
- This field is REQUIRED for outline_create mode (must be present)

REQUIRED JSON FORMAT (for outline update):
{
  "needs_update": true OR false,
  "operation": "insert_after | replace",
  "old_string": "exact text from outline (anchor point, empty if needs_update=false)",
  "new_string": "new sections to add (empty if needs_update=false)",
  "reasoning": "string - explanation of decision"
}

MANDATORY SECTION STRUCTURE:
Every outline MUST contain these sections with EXACT names in this order:
1. Abstract (exactly "Abstract")
2. Introduction (exactly "Introduction" or "I. Introduction")
3. Body sections (II, III, IV, etc. - flexible naming)
4. Conclusion (exactly "Conclusion" or "N. Conclusion")

✅ CORRECT Example (Iteration 2 - Continue Refining):
{
  "content": "Abstract\\n\\nI. Introduction\\n   A. Historical context\\n   B. Problem statement\\n\\nII. Preliminaries\\n   A. Basic definitions\\n   B. Constructible numbers\\n\\nIII. Main Results\\n   A. Theorem 1: Characterization\\n   B. Theorem 2: Transcendence of pi\\n\\nIV. Conclusion",
  "outline_complete": false,
  "reasoning": "Outline structure improved based on validator feedback. Added Problem statement subsection under Introduction. However, validator noted missing coverage of Galois theory connections (submission 11) and Baker's theorem (submissions 8, 12). Will add these in next iteration before marking complete."
}

Note: The "content" field can start with "Abstract" (optional) or "Introduction" (required).

❌ WRONG - DO NOT DO THIS:
{
  "operation": "full_content",
  "new_string": "Abstract\\n\\nI. Introduction\\n...",
  "content": "I. Introduction\\n...",
  "outline_complete": false
}

This is WRONG because:
1. Uses "operation" field (only for outline_update mode, NOT outline_create)
2. Uses "new_string" field (only for outline_update mode, NOT outline_create)
3. "content" field has incorrect structure (must start with Abstract [optional] or Introduction [required])

Example (Iteration 6 - Ready to Lock):
{
  "content": "Abstract\\n\\nI. Introduction\\n   A. Historical context of circle-squaring\\n   B. Problem statement and impossibility\\n   C. Overview of proof approach\\n\\nII. Preliminaries and Definitions\\n   A. Compass and straightedge constructions\\n   B. Field extensions and constructible numbers\\n   C. Algebraic vs. transcendental numbers\\n   D. Galois theory connections\\n\\nIII. Main Theoretical Results\\n   A. Theorem: Characterization of constructible lengths\\n   B. Theorem: Lindemann-Weierstrass (transcendence of pi)\\n   C. Theorem: Baker's theorem and applications\\n   D. Corollary: Impossibility of squaring the circle\\n\\nIV. Proofs and Derivations\\n   A. Proof of constructibility characterization\\n   B. Outline of Lindemann-Weierstrass proof\\n   C. Derivation of main impossibility result\\n\\nV. Conclusion\\n   A. Summary of impossibility result\\n   B. Related problems and historical significance",
  "outline_complete": true,
  "reasoning": "Outline now comprehensively captures ALL content from aggregator database. Added Galois theory subsection (addressing feedback from iteration 2). Added Baker's theorem coverage (addressing feedback from iterations 2-4). Structure follows logical progression from basic definitions through theorems to proofs. All required sections present with correct names. Ready to lock and begin paper construction."
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

Example (Outline Update - No update needed):
{
  "needs_update": false,
  "operation": "insert_after",
  "old_string": "",
  "new_string": "",
  "reasoning": "The current outline already captures all relevant content from the aggregator database. All theorems and proofs mentioned in accepted submissions have corresponding outline sections."
}

Example (Outline Update - Adding subsection under Section II):
{
  "needs_update": true,
  "operation": "insert_after",
  "old_string": "   C. Algebraic vs. transcendental numbers",
  "new_string": "\\n   D. Connection to Galois theory\\n      1. Field automorphisms and constructibility\\n      2. Relationship to solvability by radicals",
  "reasoning": "The aggregator database contains important Galois theory connections not yet in the outline. Inserting after subsection C to add new subsection D under Section II."
}

Example (Outline Update - Adding new section before Conclusion):
{
  "needs_update": true,
  "operation": "insert_after",
  "old_string": "IV. Proofs and Derivations\\n   A. Proof of constructibility characterization\\n   B. Outline of Lindemann-Weierstrass proof",
  "new_string": "\\n\\nV. Applications and Extensions\\n   A. Related impossibility results\\n   B. Constructible regular polygons",
  "reasoning": "The aggregator database contains applications content not yet reflected. Adding new Section V after Section IV, before Conclusion."
}
"""


async def build_outline_create_prompt(
    user_prompt: str,
    rag_evidence: str
) -> str:
    """
    Build complete prompt for outline creation.
    
    Args:
        user_prompt: User's compiler-directing prompt
        rag_evidence: RAG-retrieved evidence from aggregator database
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_outline_create_system_prompt(),
        "\n---\n",
        get_outline_json_schema(),
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
    
    # Add creation feedback (last 5 validator reviews)
    # CRITICAL: This includes the last ACCEPTED outline so model can see its own work
    from backend.compiler.memory.outline_memory import outline_memory
    creation_feedback = await outline_memory.get_creation_feedback()
    if creation_feedback:
        parts.append(f"""YOUR OUTLINE CREATION FEEDBACK (Last 5 validator reviews):

{creation_feedback}

IMPORTANT: If you see "YOUR LAST ACCEPTED OUTLINE" above, that outline was already ACCEPTED.
You should set outline_complete=true to lock it UNLESS you have specific improvements to make.
Do NOT keep generating similar outlines indefinitely - if accepted, decide to lock or improve.
---
""")
    
    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"AGGREGATOR DATABASE EVIDENCE:\n{rag_evidence}",
        "\n---\n",
        "Now generate your outline as JSON:"
    ])
    
    return "\n".join(parts)


async def build_outline_update_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str = ""
) -> str:
    """
    Build complete prompt for outline update.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (always fully injected)
        current_paper: Current document progress
        rag_evidence: RAG-retrieved evidence from aggregator database
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_outline_update_system_prompt(),
        "\n---\n",
        get_outline_json_schema(),
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
        f"CURRENT DOCUMENT PROGRESS:\n{current_paper}"
    ])
    
    if rag_evidence:
        parts.append("\n---\n")
        parts.append(f"AGGREGATOR DATABASE EVIDENCE:\n{rag_evidence}")
    
    parts.append("\n---\n")
    parts.append("Now decide if outline update is needed (respond as JSON):")
    
    return "\n".join(parts)
