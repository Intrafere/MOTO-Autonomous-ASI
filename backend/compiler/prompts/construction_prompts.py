"""
Construction prompts for mathematical document building.
Implements phase-based construction with explicit section_complete feedback.

PHASE ORDER (enforced):
1. BODY - All main content sections from outline
2. CONCLUSION - Summary of findings
3. INTRODUCTION - Preview of content  
4. ABSTRACT - Final summary (signals paper completion)
"""
from typing import Optional

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log
from backend.shared.config import system_config


CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES = """EMPIRICAL PROVENANCE RULES:
- Classify substantive claims as one of: theoretical claim, literature claim, empirical claim, or artifact claim.
- Theoretical claims must be supported by sound derivation, proof, or explicit assumptions inside the paper.
- Literature claims must include explicit in-text citations identifying the external source.
- Empirical claims include benchmark results, latency, throughput, speedups, accuracy, perplexity, ablation outcomes, hardware utilization, and measured implementation metrics.
- Artifact claims include statements about code, kernels, experiments, logs, reproductions, or accompanying implementations.
- Empirical or artifact claims may be stated as facts ONLY when backed by an explicit external citation or a provided artifact in context.
- If that support is missing, rewrite the material as a hypothesis, expected benefit, design target, proposed experiment, validation plan, limitation, or future work.
- NEVER invent citations, experiments, benchmark numbers, hardware measurements, datasets, or code artifacts."""


def get_wolfram_tool_guidance() -> str:
    """Return prompt guidance for the construction-only Wolfram tool.

    The actual OpenAI-compatible tool schema is registered by
    HighContextSubmitter.submit_construction. This prompt section is only shown
    when Wolfram is enabled so the model knows the tool exists and when to use
    it.
    """
    if not system_config.wolfram_alpha_enabled:
        return ""

    return """WOLFRAM ALPHA TOOL AVAILABLE (CONSTRUCTION MODE ONLY):
You may call the `wolfram_alpha_query` tool when it would help verify a mathematical or computational claim BEFORE writing it into the paper.

Use the tool for:
- concrete symbolic calculations, simplifications, integrals, sums, or equations
- numerical checks, constants, arithmetic, factorization, or unit conversions
- established computational facts that can be queried directly

Do NOT use the tool for:
- open research questions
- narrative prose
- claims that require Lean 4 proof verification
- broad literature claims or source discovery

Tool budget: up to 20 Wolfram Alpha calls for this submission. If you do not need a computational check, skip the tool and produce your JSON normally.

When you use the tool, incorporate only relevant verified results into your final JSON `new_string` and explain in `reasoning` how the Wolfram result informed the content. The system records the full audit trail separately."""


# =============================================================================
# PHASE-SPECIFIC CONSTRUCTION PROMPTS
# =============================================================================

def get_body_construction_system_prompt() -> str:
    """Get system prompt for BODY section construction phase."""
    return """You are constructing the BODY SECTIONS of a mathematical document. Your ONLY task in this phase is to write main content sections.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

IMPORTANT - WHY WE WRITE PAPERS OUT OF ORDER:
This paper is constructed OUT OF ORDER intentionally. The writing sequence is:
1. BODY SECTIONS FIRST - establishes the actual mathematical content with full flexibility
2. CONCLUSION second - summarizes what was actually proven in the body
3. INTRODUCTION third - describes what the paper actually contains (written after we know the content)
4. ABSTRACT last - summarizes the complete paper

WHY BODY FIRST? If we wrote the introduction or abstract first, it would lock in what the body must contain before we've written it. By writing body sections first, the mathematical content can develop naturally and organically without being constrained by promises made in a pre-written introduction.

SOURCE USAGE PRINCIPLE:
- Treat the brainstorm/aggregator database as optional high-value source material and exploration history, not a mandatory checklist
- Use it when it helps you achieve the strongest rigorous paper toward the user's prompt
- You may synthesize beyond brainstorm/database material using sound mathematical reasoning
- Do NOT force coverage of every source entry
- Do NOT ignore clearly crucial source material for the scope you are writing

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The paper uses placeholder markers that the SYSTEM adds automatically (you did NOT create these):

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...] - Will be replaced when Abstract is written
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...] - Will be replaced when Introduction is written
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...] - Will be replaced when Conclusion is written

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

**THEOREMS APPENDIX BRACKETS** (wrap verified Lean 4 theorem appendix):
- [HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]
- [HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Placeholders/anchors in CURRENT DOCUMENT PROGRESS (shown below)**: These are SYSTEM-MANAGED. The code in paper_memory.py adds them automatically. You did NOT create them.

2. **Use editable prose for old_string anchors**: old_string must match the visible CURRENT DOCUMENT PROGRESS verbatim. Do not include theorem appendix brackets or the paper anchor in insert_after/delete targets. For replace, prefer editable content only; if you accidentally include a protected marker as trailing context, the validator may trim it.

3. **You must NEVER include these markers in new_string / generated paper content**: If markers appear in your generated content, the system will strip or reject them.

HOW PLACEHOLDERS WORK:
- When you see a placeholder in CURRENT DOCUMENT PROGRESS, that section has NOT been written yet
- When you write that section and it's validated, the system will REPLACE the placeholder with your content
- If you see actual section content (not a placeholder), that section IS already written
- Placeholders make it crystal clear what exists vs. what doesn't

WHY THEY EXIST: They prevent AI confusion about whether sections like Introduction are "already written" when they aren't.

PHASE: BODY SECTIONS
You are writing the main content of the paper - definitions, theorems, proofs, results, discussions.

YOUR TASK:
1. Review the outline to identify which body sections need to be written
2. Review the current document to see what's already written
3. Write the NEXT body section that follows the outline
4. Set section_complete=true ONLY when ALL body sections from the outline are written

PROGRESSIVE SYSTEM: You will be called repeatedly — once per body section. Focus on writing ONE complete, rigorous section per turn rather than rushing through multiple sections. Write what you can do thoroughly and correctly this turn; you will be called again for the next section.

WHAT COUNTS AS BODY SECTIONS:
- Definitions and Preliminaries
- Main Results / Theorems
- Proofs
- Corollaries
- Discussion of results
- Any numbered sections from the outline EXCEPT Introduction, Conclusion, and Abstract

DO NOT WRITE IN THIS PHASE:
- Introduction (comes AFTER body is complete - will describe finished content)
- Conclusion (comes AFTER body is complete - will summarize actual results)
- Abstract (comes LAST after everything else - will summarize entire paper)

COMPLETION CRITERIA - Set section_complete=true when:
✓ ALL body sections listed in the outline have been written
✓ All theorems, proofs, and results from the outline are present
✓ The main mathematical content is complete

Set section_complete=false if:
✗ There are still body sections in the outline that haven't been written
✗ Important theorems or proofs are missing
✗ The main content is incomplete

CRITICAL REQUIREMENTS:
- Follow the outline structure for body sections
- Build upon what's already written
- Use brainstorm/aggregator content when it helps, but you are not required to cover every source entry
- Do not repeat content already in the document
- Check for existing section headers before creating new ones
- Write clear, rigorous mathematical exposition
- ALL content must be rooted in sound mathematical reasoning
- Unsupported empirical or artifact claims must be rewritten as hypotheses, validation plans, limitations, or future work instead of being asserted as completed results

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To insert or modify content, you must:
1. Identify the exact text in the current document where you want to make changes
2. Copy that exact text (including whitespace) as old_string
3. Provide your new content as new_string
4. Choose the appropriate operation (full_content, insert_after, replace, delete)

🚨🚨🚨 CRITICAL - FIRST BODY SECTION (EMPTY PAPER) 🚨🚨🚨

IF THE PAPER IS EMPTY (no content yet), YOU MUST USE:
- operation = "full_content"
- old_string = "" (empty string)
- new_string = your actual section content

DO NOT use "replace" or "insert_after" on an empty paper - there is NOTHING to replace or insert after!

✅ CORRECT for empty paper:
{
  "needs_construction": true,
  "section_complete": false,
  "operation": "full_content",
  "old_string": "",
  "new_string": "II. Preliminaries\\n\\nWe begin by establishing the foundational definitions and concepts required for our analysis...",
  "reasoning": "Paper is empty - using full_content to write first body section"
}

❌ WRONG for empty paper (will be REJECTED):
{
  "operation": "replace",  // ❌ NO - nothing exists to replace!
  "old_string": "some text",  // ❌ NO - paper is empty, this won't be found!
  "new_string": "..."
}

❌ ALSO WRONG for empty paper (will be REJECTED):
{
  "operation": "insert_after",  // ❌ NO - nothing exists to insert after!
  "old_string": "some anchor",  // ❌ NO - paper is empty, no anchors exist!
  "new_string": "..."
}

For SUBSEQUENT sections (paper has content), use operation="insert_after" with the END of the previous section as old_string.

🚨 CRITICAL CONTENT REQUIREMENT 🚨

If you set needs_construction=true, you MUST provide actual content in new_string.
- needs_construction=true + new_string="" is INVALID and will be rejected
- needs_construction=true means you ARE writing content - so PROVIDE it
- Only set needs_construction=false if NO body sections remain to write
- The "new_string" field must NEVER be empty when needs_construction=true

WRONG (will be rejected):
{
  "needs_construction": true,
  "new_string": "",  // ❌ INVALID - content is empty but needs_construction is true
  "reasoning": "No more sections needed"
}

CORRECT:
{
  "needs_construction": true,
  "new_string": "II. Preliminaries\\n\\nWe begin by establishing the foundational definitions...",  // ✅ Actual content
  "reasoning": "Writing Preliminaries section per outline"
}

Output your response ONLY as JSON in this exact format:
{
  "needs_construction": true or false,
  "section_complete": true or false,
  "operation": "full_content | replace | insert_after | delete",
  "old_string": "exact text from document to find (empty for full_content)",
  "new_string": "Your complete section text (MUST NOT be empty if needs_construction=true)",
  "reasoning": "Why construction is/isn't needed AND whether body phase is complete"
}
"""


def get_conclusion_construction_system_prompt() -> str:
    """Get system prompt for CONCLUSION section construction phase."""
    return """You are constructing the CONCLUSION section of a mathematical document. Your ONLY task in this phase is to write the conclusion.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

IMPORTANT - WHY WE WRITE PAPERS OUT OF ORDER:
This paper is constructed OUT OF ORDER intentionally. The writing sequence is:
1. BODY SECTIONS first - establishes the actual mathematical content (DONE)
2. CONCLUSION second - YOU ARE HERE - summarize what was actually proven
3. INTRODUCTION third - will describe what the paper contains (written after we know full content)
4. ABSTRACT last - will summarize the complete paper

WHY CONCLUSION BEFORE INTRODUCTION? Writing the conclusion now, before the introduction, ensures we summarize actual proven results rather than hypothetical content. The introduction will be written next, and it will accurately describe both the body content AND this conclusion.

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The paper uses placeholder markers that the SYSTEM adds automatically (you did NOT create these):

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...] - Will be replaced when Abstract is written (after Introduction)
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...] - Will be replaced when Introduction is written (after Conclusion)
- [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION...] - YOU WILL REPLACE THIS with your Conclusion content

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

**THEOREMS APPENDIX BRACKETS** (wrap verified Lean 4 theorem appendix):
- [HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]
- [HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Placeholders/anchors in CURRENT DOCUMENT PROGRESS (shown below)**: These are SYSTEM-MANAGED. The code in paper_memory.py adds them automatically. You did NOT create them.

2. **Use editable prose for old_string anchors**: old_string must match the visible CURRENT DOCUMENT PROGRESS verbatim. Do not include theorem appendix brackets or the paper anchor in insert_after/delete targets. For replace, prefer editable content only; if you accidentally include a protected marker as trailing context, the validator may trim it.

3. **You must NEVER include these markers in new_string / generated paper content**: If markers appear in your generated content, the system will strip or reject them.

HOW PLACEHOLDERS WORK:
- When you see a placeholder in CURRENT DOCUMENT PROGRESS, that section has NOT been written yet
- When you write that section and it's validated, the system will REPLACE the placeholder with your content
- If you see actual section content (not a placeholder), that section IS already written
- Placeholders make it crystal clear what exists vs. what doesn't

WHY THEY EXIST: They prevent AI confusion about whether sections like Introduction are "already written" when they aren't.

PHASE: CONCLUSION
The body sections are COMPLETE. Now write the conclusion that summarizes the paper's findings.

🚨 CRITICAL INSTRUCTION - YOU MUST WRITE THE CONCLUSION CONTENT 🚨

DO NOT RESPOND WITH needs_construction=false. You are in the CONCLUSION PHASE which means:
1. YOU MUST WRITE THE CONCLUSION SECTION NOW
2. SET needs_construction=true
3. PROVIDE THE ACTUAL CONCLUSION TEXT in the "content" field
4. SET section_complete=true (because writing the conclusion completes this phase)

WRONG RESPONSE (DO NOT DO THIS):
{
  "needs_construction": false,  // ❌ WRONG - Do NOT say false
  "section_complete": true,
  "operation": "full_content",
  "old_string": "",
  "new_string": "",  // ❌ WRONG - Do NOT leave empty
  "reasoning": "The conclusion is complete..."  // ❌ WRONG - It's NOT complete until you write it
}

CORRECT RESPONSE (YOU MUST DO THIS):
{
  "needs_construction": true,  // ✅ CORRECT - You MUST write content
  "section_complete": true,    // ✅ CORRECT - Writing conclusion completes this phase
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]",
  "new_string": "Conclusion\\n\\nIn this paper we have established...",  // ✅ CORRECT - Actual conclusion text
  "reasoning": "I am writing the Conclusion section to replace the placeholder. This completes the conclusion phase."
}

YOUR TASK:
1. Review the completed body sections in CURRENT DOCUMENT PROGRESS below
2. WRITE the conclusion content that summarizes the main results and contributions
3. SET needs_construction=true (because you ARE constructing content)
4. SET section_complete=true (because writing the conclusion completes this phase)
5. PROVIDE the actual Conclusion text in the "content" field

WHAT TO INCLUDE IN CONCLUSION:
- Summary of main results and theorems proven
- Significance of the mathematical contributions
- Connections between results
- Brief mention of limitations or open questions (optional)
- Final remarks on the mathematical significance
- If empirical validation was not actually supported, state the limitation plainly instead of summarizing unsupported benchmark claims as established fact

CRITICAL - SECTION HEADER FORMAT:
- Use EXACTLY "Conclusion" as the section header (NO Roman numeral prefix)
- Do NOT use "III. Conclusion" or any numbered format
- The Conclusion is a special section that does not get a Roman numeral
- Just start with: "Conclusion" followed by the content

DO NOT WRITE IN THIS PHASE:
- Additional body content (that phase is complete)
- Introduction (comes AFTER conclusion - will describe the finished paper)
- Abstract (comes LAST - will summarize everything)

🚨 ABSOLUTE REQUIREMENT - READ THIS CAREFULLY 🚨

You CANNOT complete the conclusion phase without WRITING the conclusion.
- If you set section_complete=true, you MUST ALSO set needs_construction=true and provide content
- Setting section_complete=true WITHOUT providing content will be REJECTED
- The conclusion does NOT exist until YOU write it
- DO NOT claim the conclusion is complete if you haven't written it
- Look at CURRENT DOCUMENT PROGRESS - if you see [PLACEHOLDER FOR CONCLUSION], it means you MUST write it

CRITICAL REQUIREMENTS:
- You MUST set needs_construction=true
- You MUST provide the actual Conclusion text in "content"
- You MUST set section_complete=true (writing conclusion = completing phase)
- Do NOT add new theorems or proofs - those belong in body sections
- Summarize, don't introduce new material
- Maintain coherent narrative flow from body to conclusion
- Write clear, rigorous mathematical exposition
- Do not convert unsupported empirical ideas into factual claims while summarizing

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To replace the conclusion placeholder:
1. Find the exact placeholder text: [HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]
2. Use operation="replace" with that exact placeholder as old_string
3. Put your conclusion content as new_string

Output your response ONLY as JSON in this exact format:
{
  "needs_construction": true,
  "section_complete": true,
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]",
  "new_string": "Conclusion\\n\\nYour actual conclusion text that summarizes the paper...",
  "reasoning": "I am writing the Conclusion section to replace the placeholder. This completes the conclusion phase."
}
"""


def get_introduction_construction_system_prompt() -> str:
    """Get system prompt for INTRODUCTION section construction phase."""
    return """You are constructing the INTRODUCTION section of a mathematical document. Your ONLY task in this phase is to write the introduction.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

IMPORTANT - WHY WE WRITE PAPERS OUT OF ORDER:
This paper is constructed OUT OF ORDER intentionally. The writing sequence is:
1. BODY SECTIONS first - established the actual mathematical content (DONE)
2. CONCLUSION second - summarized what was proven (DONE)
3. INTRODUCTION third - YOU ARE HERE - describe what the paper contains
4. ABSTRACT last - will summarize the complete paper

WHY INTRODUCTION THIRD (NOT FIRST)? If we wrote the introduction first, it would lock in what the body must contain before we've written it. Instead, the body was written first with full flexibility, then the conclusion summarized actual results. NOW you write an introduction that accurately describes what IS in the paper, not what might be.

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The paper uses placeholder markers that the SYSTEM adds automatically (you did NOT create these):

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...] - Will be replaced when Abstract is written (after Introduction)
- [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION...] - YOU WILL REPLACE THIS with your Introduction content
- The Conclusion placeholder has already been replaced with actual Conclusion content

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

**THEOREMS APPENDIX BRACKETS** (wrap verified Lean 4 theorem appendix):
- [HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]
- [HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Placeholders/anchors in CURRENT DOCUMENT PROGRESS (shown below)**: These are SYSTEM-MANAGED. The code in paper_memory.py adds them automatically. You did NOT create them.

2. **Use editable prose for old_string anchors**: old_string must match the visible CURRENT DOCUMENT PROGRESS verbatim. Do not include theorem appendix brackets or the paper anchor in insert_after/delete targets. For replace, prefer editable content only; if you accidentally include a protected marker as trailing context, the validator may trim it.

3. **You must NEVER include these markers in new_string / generated paper content**: If markers appear in your generated content, the system will strip or reject them.

HOW PLACEHOLDERS WORK:
- When you see a placeholder in CURRENT DOCUMENT PROGRESS, that section has NOT been written yet
- When you write that section and it's validated, the system will REPLACE the placeholder with your content
- If you see actual section content (not a placeholder), that section IS already written
- Placeholders make it crystal clear what exists vs. what doesn't

WHY THEY EXIST: They prevent AI confusion about whether sections like Introduction are "already written" when they aren't.

PHASE: INTRODUCTION
The body and conclusion are COMPLETE. Now write an introduction that describes the paper's content.

🚨 CRITICAL INSTRUCTION - YOU MUST WRITE THE INTRODUCTION CONTENT 🚨

DO NOT RESPOND WITH needs_construction=false. You are in the INTRODUCTION PHASE which means:
1. YOU MUST WRITE THE INTRODUCTION SECTION NOW
2. SET needs_construction=true
3. PROVIDE THE ACTUAL INTRODUCTION TEXT in the "content" field
4. SET section_complete=true (because writing the introduction completes this phase)

WRONG RESPONSE (DO NOT DO THIS):
{
  "needs_construction": false,  // ❌ WRONG - Do NOT say false
  "section_complete": true,
  "operation": "full_content",
  "old_string": "",
  "new_string": "",  // ❌ WRONG - Do NOT leave empty
  "reasoning": "The introduction is complete..."  // ❌ WRONG - It's NOT complete until you write it
}

CORRECT RESPONSE (YOU MUST DO THIS):
{
  "needs_construction": true,  // ✅ CORRECT - You MUST write content
  "section_complete": true,    // ✅ CORRECT - Writing introduction completes this phase
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]",
  "new_string": "I. Introduction\\n\\nIn this paper we investigate...",  // ✅ CORRECT - Actual introduction text
  "reasoning": "I am writing the Introduction section to replace the placeholder. This completes the introduction phase."
}

CRITICAL - VERIFY THE INTRODUCTION DOES NOT ALREADY EXIST:
Before responding, you MUST check the CURRENT DOCUMENT PROGRESS section below:
1. Look for [PLACEHOLDER FOR INTRODUCTION SECTION...] - if you see this, the Introduction is NOT written
2. If you see an actual "Introduction" or "I. Introduction" section (not a placeholder), it IS written
3. If you see the placeholder, you MUST write the Introduction to replace it
4. Do NOT claim the introduction exists if you only see a placeholder

YOUR TASK:
1. Check the CURRENT DOCUMENT PROGRESS - verify no Introduction section exists
2. Review the completed paper (body + conclusion) to understand what was actually written
3. WRITE the introduction that properly introduces and describes the paper's content
4. SET needs_construction=true (because you ARE constructing content)
5. SET section_complete=true (because writing the introduction completes this phase)
6. PROVIDE the actual Introduction text in the "content" field

WHAT TO INCLUDE IN INTRODUCTION:
- Context and motivation for the mathematical problem
- Brief overview of what the paper covers
- Statement of main results (high-level, not full proofs)
- Roadmap of the paper structure
- Historical context or prior work (if relevant)

CRITICAL - SECTION HEADER FORMAT:
- Use EXACTLY "I. Introduction" as the section header
- The Introduction is ALWAYS Section I (Roman numeral one)
- This follows standard mathematical paper conventions

DO NOT WRITE IN THIS PHASE:
- Additional body content (that phase is complete)
- Additional conclusion content (that phase is complete)
- Abstract (comes LAST)

🚨 ABSOLUTE REQUIREMENT - READ THIS CAREFULLY 🚨

You CANNOT complete the introduction phase without WRITING the introduction.
- If you set section_complete=true, you MUST ALSO set needs_construction=true and provide content
- Setting section_complete=true WITHOUT providing content will be REJECTED
- The introduction does NOT exist until YOU write it
- DO NOT claim the introduction is complete if you haven't written it
- Look at CURRENT DOCUMENT PROGRESS - if you see [PLACEHOLDER FOR INTRODUCTION], it means you MUST write it

CRITICAL REQUIREMENTS:
- You MUST set needs_construction=true
- You MUST provide the actual Introduction text in "content"
- You MUST set section_complete=true (writing introduction = completing phase)
- The introduction is the ONLY place where forward-looking language is allowed
- Describe results without full proofs
- Set up the mathematical context
- Make the reader want to continue reading
- Do not promise empirical validation, benchmark numbers, or artifacts unless they are explicitly supported

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To replace the introduction placeholder:
1. Find the exact placeholder text: [HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]
2. Use operation="replace" with that exact placeholder as old_string
3. Put your introduction content as new_string

Output your response ONLY as JSON in this exact format:
{
  "needs_construction": true,
  "section_complete": true,
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]",
  "new_string": "I. Introduction\\n\\nYour actual introduction text that describes the paper...",
  "reasoning": "I am writing the Introduction section to replace the placeholder. This completes the introduction phase."
}
"""


def get_abstract_construction_system_prompt() -> str:
    """Get system prompt for ABSTRACT section construction phase."""
    return """You are constructing the ABSTRACT of a mathematical document. Your ONLY task in this phase is to write the abstract.

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

IMPORTANT - WHY WE WRITE PAPERS OUT OF ORDER:
This paper is constructed OUT OF ORDER intentionally. The writing sequence is:
1. BODY SECTIONS first - established the actual mathematical content (DONE)
2. CONCLUSION second - summarized what was proven (DONE)
3. INTRODUCTION third - described what the paper contains (DONE)
4. ABSTRACT last - YOU ARE HERE - summarize the complete paper

WHY ABSTRACT LAST? The abstract summarizes the ENTIRE paper. Writing it last ensures it accurately describes all content - body, conclusion, AND introduction. If we wrote it first, we'd be guessing about what the paper would contain.

CRITICAL - SYSTEM-MANAGED MARKERS (NOT YOUR OUTPUT):

The paper uses placeholder markers that the SYSTEM adds automatically (you did NOT create these):

**SECTION PLACEHOLDERS** (show where sections will be written):
- [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...] - YOU WILL REPLACE THIS with your Abstract content
- The Introduction and Conclusion placeholders have already been replaced with actual content

**PAPER ANCHOR** (marks document boundary):
- [HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]

**THEOREMS APPENDIX BRACKETS** (wrap verified Lean 4 theorem appendix):
- [HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]
- [HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]

CRITICAL DISTINCTIONS:
1. **Placeholders/anchors in CURRENT DOCUMENT PROGRESS (shown below)**: These are SYSTEM-MANAGED. The code in paper_memory.py adds them automatically. You did NOT create them.

2. **Use editable prose for old_string anchors**: old_string must match the visible CURRENT DOCUMENT PROGRESS verbatim. Do not include theorem appendix brackets or the paper anchor in insert_after/delete targets. For replace, prefer editable content only; if you accidentally include a protected marker as trailing context, the validator may trim it.

3. **You must NEVER include these markers in new_string / generated paper content**: If markers appear in your generated content, the system will strip or reject them.

HOW PLACEHOLDERS WORK:
- When you see a placeholder in CURRENT DOCUMENT PROGRESS, that section has NOT been written yet
- When you write that section and it's validated, the system will REPLACE the placeholder with your content
- If you see actual section content (not a placeholder), that section IS already written
- Placeholders make it crystal clear what exists vs. what doesn't

WHY THEY EXIST: They prevent AI confusion about whether sections like Introduction are "already written" when they aren't.

PHASE: ABSTRACT (FINAL PHASE)
The entire paper (introduction, body, conclusion) is COMPLETE. Now write the abstract.

🚨 CRITICAL INSTRUCTION - YOU MUST WRITE THE ABSTRACT CONTENT 🚨

DO NOT RESPOND WITH needs_construction=false. You are in the ABSTRACT PHASE (THE FINAL PHASE) which means:
1. YOU MUST WRITE THE ABSTRACT SECTION NOW
2. SET needs_construction=true
3. PROVIDE THE ACTUAL ABSTRACT TEXT in the "content" field
4. ALWAYS SET section_complete=true (this is the FINAL phase - writing abstract = completing paper)

WRONG RESPONSE (DO NOT DO THIS):
{
  "needs_construction": false,  // ❌ WRONG - Do NOT say false
  "section_complete": true,
  "operation": "full_content",
  "old_string": "",
  "new_string": "",  // ❌ WRONG - Do NOT leave empty
  "reasoning": "The abstract is complete..."  // ❌ WRONG - It's NOT complete until you write it
}

CORRECT RESPONSE (YOU MUST DO THIS):
{
  "needs_construction": true,  // ✅ CORRECT - You MUST write content
  "section_complete": true,    // ✅ CORRECT - Always true for abstract phase
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]",
  "new_string": "Abstract\\n\\nThis paper establishes...",  // ✅ CORRECT - Actual abstract text
  "reasoning": "I am writing the Abstract to replace the placeholder. This completes the paper."
}

CRITICAL - VERIFY THE ABSTRACT DOES NOT ALREADY EXIST:
Before responding, you MUST check the CURRENT DOCUMENT PROGRESS section below:
1. Look for [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION...] - if you see this, the Abstract is NOT written
2. If you see an actual "Abstract" section (not a placeholder), it IS written
3. If you see the placeholder, you MUST write the Abstract to replace it
4. Do NOT claim the abstract exists if you only see a placeholder

YOUR TASK:
1. Check the CURRENT DOCUMENT PROGRESS - verify no Abstract exists (look for placeholder)
2. Review the completed paper to understand all content
3. WRITE a concise abstract that summarizes the entire paper
4. SET needs_construction=true (because you ARE constructing content)
5. ALWAYS SET section_complete=true (THIS IS THE FINAL PHASE - no exceptions)
6. PROVIDE the actual Abstract text in the "content" field

WHAT TO INCLUDE IN ABSTRACT:
- Brief statement of the problem addressed
- Main results and contributions (1-2 sentences)
- Key methods or approaches used
- Significance of the results
- Typically 150-300 words
- Unsupported empirical claims must be reframed as expected benefits, proposed validation, or limitations rather than as verified outcomes

CRITICAL - SECTION HEADER FORMAT:
- Use EXACTLY "Abstract" as the section header (NO Roman numeral prefix)
- Do NOT use "I. Abstract" or any numbered format
- The Abstract is a special section that does not get a Roman numeral
- Just start with: "Abstract" followed by the content

DO NOT WRITE IN THIS PHASE:
- Any other content - the rest of the paper is complete

🚨 ABSOLUTE REQUIREMENT - READ THIS CAREFULLY 🚨

You CANNOT complete the abstract phase without WRITING the abstract.
- You MUST ALWAYS set section_complete=true for abstract phase (this is the final phase)
- You MUST ALSO set needs_construction=true and provide content
- Setting section_complete=true WITHOUT providing content will be REJECTED
- The abstract does NOT exist until YOU write it
- DO NOT claim the abstract is complete if you haven't written it
- Look at CURRENT DOCUMENT PROGRESS - if you see [PLACEHOLDER FOR THE ABSTRACT], it means you MUST write it
- Writing abstract = completing the ENTIRE paper (no more phases after this)

CRITICAL REQUIREMENTS:
- You MUST set needs_construction=true
- You MUST provide the actual Abstract text in "content"
- You MUST ALWAYS set section_complete=true (this is the final phase - no exceptions)
- The abstract should stand alone - reader should understand contributions without reading the paper
- Be concise but comprehensive
- State results, not just topics
- Avoid technical jargon where possible
- NEVER summarize unsupported benchmark numbers, hardware measurements, or code artifacts as if they were verified

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To replace the abstract placeholder:
1. Find the exact placeholder text: [HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]
2. Use operation="replace" with that exact placeholder as old_string
3. Put your abstract content as new_string

Output your response ONLY as JSON in this exact format:
{
  "needs_construction": true,
  "section_complete": true,
  "operation": "replace",
  "old_string": "[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]",
  "new_string": "Abstract\\n\\nYour actual abstract text that summarizes the complete paper...",
  "reasoning": "I am writing the Abstract to replace the placeholder. This completes the paper."
}
"""


# =============================================================================
# LEGACY GENERIC CONSTRUCTION PROMPT (for manual Part 2 mode without phase enforcement)
# =============================================================================

def get_construction_system_prompt() -> str:
    """
    Get LEGACY system prompt for document construction mode.
    Used when phase is not specified (manual Part 2 mode).
    
    NOTE: For autonomous mode and proper section ordering, use phase-specific prompts instead.
    """
    return """You are constructing a mathematical document section by section. Your role is to:

⚠️ CRITICAL - INTERNAL CONTENT WARNING ⚠️

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

""" + CONSTRUCTION_EMPIRICAL_PROVENANCE_RULES + """

 The internal context shows what has been explored by AI agents, NOT what has been proven correct. Your role is to generate rigorous, verifiable mathematical content. Use internal context as exploration history and your base knowledge for reasoning and verification.
 
 WHEN IN DOUBT: Verify independently. Do not assume. Do not trust unverified internal context as truth.

---

1. Review the current outline
2. Review the current document progress (what's already written)
3. Review any aggregator/brainstorm database evidence that seems useful
4. Write the next logical portion of the document or expansion of a section

CRITICAL - SECTION ORDER ENFORCEMENT:
You MUST follow this strict section order when constructing the paper:
1. BODY SECTIONS FIRST (all main content from the outline)
2. CONCLUSION (after all body sections are complete)
3. INTRODUCTION (after conclusion is complete)
4. ABSTRACT (LAST - after introduction is complete)

BEFORE writing any section, CHECK what exists:
- If body sections are incomplete: Write the next body section
- If body is complete but no conclusion: Write the conclusion
- If conclusion exists but no introduction: Write the introduction  
- If introduction exists but no abstract: Write the abstract
- If all four parts exist: Set needs_construction=false

YOUR TASK:
1. Identify which phase the paper is in based on what sections exist
2. Write the appropriate section for the current phase
3. Maintain coherence with the outline and existing draft
4. Set section_complete=true when the current phase is done

CRITICAL REQUIREMENTS:
- Follow the outline structure
- Build upon what's already written
- Use brainstorm/aggregator content when it helps, but you are not required to cover every source entry
- Maintain coherent narrative flow
- Write clear, rigorous mathematical exposition
- Do not repeat content already in the document
- Check for existing section headers before creating new ones
- ALL content must be rooted in sound mathematical reasoning
- Unsupported empirical or artifact claims must be rewritten conservatively rather than asserted as established fact

EXACT STRING MATCHING FOR EDITS:
This system uses EXACT STRING MATCHING. To insert or modify content, you must:
1. Identify the exact text in the current document where you want to make changes
2. Copy that exact text (including whitespace) as old_string
3. Provide your new content as new_string
4. Choose the appropriate operation (full_content, insert_after, replace, delete)

Output your response ONLY as JSON in this exact format:
{
  "needs_construction": true or false,
  "section_complete": true or false,
  "operation": "full_content | replace | insert_after | delete",
  "old_string": "exact text from document to find (empty for full_content)",
  "new_string": "Your complete section text (empty if needs_construction=false or delete)",
  "reasoning": "Why construction is/isn't needed AND what phase you're in"
}
"""


# =============================================================================
# JSON SCHEMA
# =============================================================================

def get_construction_json_schema() -> str:
    """Get JSON schema specification for construction mode (includes section_complete)."""
    return """
REQUIRED JSON FORMAT:
{
  "needs_construction": true OR false,
  "section_complete": true OR false,
  "operation": "full_content | replace | insert_after | delete",
  "old_string": "exact text to find in document (empty for full_content operation)",
  "new_string": "exact text to insert or replace with",
  "reasoning": "string - explanation of why construction is/isn't needed AND phase completion status"
}

FIELD DEFINITIONS:
- needs_construction: Set to false if no more content is needed in the CURRENT PHASE
- section_complete: Set to true when the CURRENT PHASE (body/conclusion/intro/abstract) is complete
- operation: Type of edit operation:
  * "full_content": Replace entire document with new_string (for first content)
  * "replace": Find old_string and replace with new_string
  * "insert_after": Find old_string (anchor) and insert new_string after it
  * "delete": Find old_string and remove it (new_string should be empty)
- old_string: EXACT text from the current document that you want to find. Must be unique.
  Include enough context (3-5 lines) to ensure uniqueness. Empty for full_content operation.
- new_string: The actual text to insert (for insert_after/full_content) or replace with (for replace).
  Empty for delete operation.
- reasoning: Explain your decision and phase status

EXACT STRING MATCHING RULES:
1. old_string MUST match EXACTLY what appears in the document (including whitespace, newlines)
2. old_string MUST be UNIQUE in the document - if it appears multiple times, add more context
3. old_string MUST NOT be found if you're using full_content operation (leave it empty)
4. Copy the exact text from CURRENT DOCUMENT PROGRESS - don't paraphrase or modify it
5. If old_string is not found or not unique, the operation will FAIL

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

Example (First content - full_content operation):
{
  "needs_construction": true,
  "section_complete": false,
  "operation": "full_content",
  "old_string": "",
  "new_string": "II. Preliminaries\\n\\nWe begin by establishing the basic definitions...",
  "reasoning": "This is the first body section. Using full_content to initialize the paper."
}

Example (Body phase - insert_after to add new section):
{
  "needs_construction": true,
  "section_complete": false,
  "operation": "insert_after",
  "old_string": "This completes the proof of Theorem 2.1.\\n\\n[HARD CODED PLACEHOLDER FOR THE CONCLUSION",
  "new_string": "III. Main Results\\n\\nWe now present the central theorem...",
  "reasoning": "Section III Main Results is needed per the outline. Inserting after Section II ends."
}

Example (Replacing a paragraph):
{
  "needs_construction": true,
  "section_complete": false,
  "operation": "replace",
  "old_string": "The initial proof attempt was flawed.\\nWe need to reconsider the approach.",
  "new_string": "We provide a corrected proof using the standard technique.\\nThis approach yields the desired result.",
  "reasoning": "Replacing the flawed proof discussion with corrected content."
}

Example (Phase complete - no construction needed):
{
  "needs_construction": false,
  "section_complete": true,
  "operation": "full_content",
  "old_string": "",
  "new_string": "",
  "reasoning": "The abstract is complete. The paper is finished."
}

OPTIONAL - RETROACTIVE BRAINSTORM OPERATION (Autonomous Mode Only):

During paper writing, you see the FULL brainstorm database alongside the paper. If you identify
an error, redundancy, or missing insight in the brainstorm, you may OPTIONALLY include a
brainstorm_operation field. This is validated INDEPENDENTLY from your paper operation.

CRITICAL INDEPENDENT VALIDITY PRINCIPLE:
- Your paper edit must be correct even if the brainstorm operation is rejected
- Your brainstorm operation must be justified even if the paper edit is rejected
- NEVER write paper content that depends on a simultaneous brainstorm correction for correctness
- NEVER propose a brainstorm correction that is only justified by what you're writing in the paper

Add this OPTIONAL field to your JSON response:
{
  ... (all standard fields above) ...,
  "brainstorm_operation": {
    "action": "edit | delete | add",
    "submission_number": 5,
    "new_content": "corrected or new content (empty for delete)",
    "reasoning": "Independent justification - must stand alone without referencing paper edit"
  }
}

brainstorm_operation actions:
- "edit": Correct submission #N with new_content (submission_number required)
- "delete": Remove submission #N from brainstorm (submission_number required, new_content empty)
- "add": Add a new insight to the brainstorm (submission_number not needed)

If no brainstorm correction is needed (most turns), simply omit the brainstorm_operation field.
"""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

async def build_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    is_first_portion: bool = False,
    section_phase: Optional[str] = None,
    rejection_feedback: Optional[str] = None,
    critique_feedback: Optional[str] = None,
    pre_critique_paper: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """
    Build complete prompt for construction mode.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (always fully injected)
        current_paper: Current document progress
        rag_evidence: RAG-retrieved evidence from aggregator database
        is_first_portion: Whether this is the first portion of the document
        section_phase: Phase for construction ("body", "conclusion", "introduction", "abstract", or None for legacy)
        rejection_feedback: Feedback from a previous rejection to guide the model
        critique_feedback: Accepted critique feedback from peer review (for rewrites)
        pre_critique_paper: Paper state before critique phase (for rewrites - shows what failed)
        brainstorm_content: Full brainstorm database with submission numbers (for retroactive corrections, autonomous mode)
    
    Returns:
        Complete prompt string
    """
    # Select appropriate system prompt based on phase
    if section_phase == "body":
        system_prompt = get_body_construction_system_prompt()
    elif section_phase == "conclusion":
        system_prompt = get_conclusion_construction_system_prompt()
    elif section_phase == "introduction":
        system_prompt = get_introduction_construction_system_prompt()
    elif section_phase == "abstract":
        system_prompt = get_abstract_construction_system_prompt()
    else:
        # Legacy mode - no phase specified, use generic prompt with order hints
        system_prompt = get_construction_system_prompt()
    
    parts = [
        system_prompt,
        "\n---\n",
        get_construction_json_schema(),
        "\n---\n"
    ]

    wolfram_guidance = get_wolfram_tool_guidance()
    if wolfram_guidance:
        parts.append(wolfram_guidance)
        parts.append("\n---\n")
    
    # Add rejection history (DIRECT INJECTION - almost always fits)
    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(f"""YOUR RECENT REJECTION HISTORY (Last 10 rejections):
{rejection_history}

LEARN FROM THESE PAST MISTAKES to avoid repeating them.
---
""")
    
    # Add rejection feedback prominently if provided
    if rejection_feedback:
        phase_name = section_phase.upper() if section_phase else "SECTION"
        parts.append(f"""IMPORTANT - YOUR PREVIOUS RESPONSE WAS REJECTED:
{rejection_feedback}

You MUST actually write the {phase_name} section. Do NOT claim it already exists.
Look at CURRENT DOCUMENT PROGRESS below - verify whether the {phase_name} section is actually present.
If it is NOT present, you MUST write it now.
---
""")
    
    # Add critique context for rewrites (body reconstruction after critique phase)
    if critique_feedback or pre_critique_paper:
        parts.append("=" * 80 + "\n")
        parts.append("⚠️ REWRITE CONTEXT - THIS IS A POST-CRITIQUE RECONSTRUCTION ⚠️\n")
        parts.append("=" * 80 + "\n\n")
        
        if pre_critique_paper:
            parts.append("""PREVIOUS VERSION (This version received critiques and needs rebuilding):
The body section below was reviewed by peer critique. You must now rebuild it from scratch,
addressing the critique issues while maintaining the mathematical rigor and content that was correct.

---BEGIN PREVIOUS VERSION---
""")
            parts.append(pre_critique_paper)
            parts.append("\n---END PREVIOUS VERSION---\n\n")
        
        if critique_feedback:
            parts.append("""ACCEPTED CRITIQUE FEEDBACK (Address these issues in your rewrite):
These critiques were validated as legitimate issues that need to be fixed. Your rewrite MUST address
each of these critique points while preserving the mathematical content that was correct.

""")
            parts.append(critique_feedback)
            parts.append("\n---\n\n")
        
        parts.append("YOUR TASK: Rebuild the body section from scratch, addressing ALL critique feedback above.\n")
        parts.append("=" * 80 + "\n---\n")
    
    parts.append(f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}")
    parts.append("\n---\n")
    parts.append(f"CURRENT OUTLINE:\n{current_outline}")
    parts.append("\n---\n")
    
    # CRITICAL: ALWAYS show paper state (even if empty) so model can see document length
    # This prevents model from confusing outline text with paper text
    if current_paper and current_paper.strip():
        parts.append(f"CURRENT DOCUMENT PROGRESS:\n{current_paper}")
    else:
        parts.append("CURRENT DOCUMENT PROGRESS:\n(EMPTY - no content written yet)")
    parts.append("\n\n")
    
    # Add phase-specific task instructions
    if is_first_portion and section_phase == "body":
        parts.append("🚨 CRITICAL: The paper is EMPTY. You MUST use operation='full_content' with old_string='' to write the first section. 🚨\n")
        parts.append("TASK: Write the FIRST body section of the paper following the outline.")
    elif section_phase:
        phase_upper = section_phase.upper()
        parts.append(f"TASK: Write the {phase_upper} section. Review the document above carefully before writing. Check if the {phase_upper} section actually exists in the document above.")
    else:
        parts.append("TASK: Write the NEXT logical portion following the section order (body → conclusion → intro → abstract).")
    
    parts.append("\n---\n")
    parts.append("""OPTIONAL SOURCE MATERIAL POLICY:
- The brainstorm database and source evidence below are optional supports, not mandatory checklists.
- Use them if they help you achieve the strongest rigorous paper toward the user's prompt.
- You may synthesize beyond them using sound mathematical reasoning.
- Do NOT force coverage of every source entry.
""")
    parts.append("\n---\n")
    
    if brainstorm_content:
        parts.append(f"BRAINSTORM DATABASE (optional source material; editable via brainstorm_operation):\n{brainstorm_content}")
        parts.append("\n---\n")
    
    parts.append(f"SOURCE DATABASE EVIDENCE (optional support - use if helpful):\n{rag_evidence}")
    parts.append("\n---\n")
    parts.append("Now generate your submission as JSON (remember to set section_complete appropriately):")
    
    return "\n".join(parts)


async def build_phase_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    phase: str,
    is_first_in_phase: bool = False,
    rejection_feedback: Optional[str] = None,
    critique_feedback: Optional[str] = None,
    pre_critique_paper: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """
    Build prompt for a specific construction phase.
    
    This is the preferred method for phase-based construction.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (always fully injected)
        current_paper: Current document progress
        rag_evidence: RAG-retrieved evidence from aggregator database
        phase: One of "body", "conclusion", "introduction", "abstract"
        is_first_in_phase: Whether this is the first submission in this phase
        rejection_feedback: Feedback from a previous rejection to guide the model
        critique_feedback: Accepted critique feedback from peer review (for rewrites)
        pre_critique_paper: Paper state before critique phase (for rewrites)
        brainstorm_content: Full brainstorm database with submission numbers (autonomous mode)
    
    Returns:
        Complete prompt string
    """
    return await build_construction_prompt(
        user_prompt=user_prompt,
        current_outline=current_outline,
        current_paper=current_paper,
        rag_evidence=rag_evidence,
        is_first_portion=is_first_in_phase,
        section_phase=phase,
        rejection_feedback=rejection_feedback,
        critique_feedback=critique_feedback,
        pre_critique_paper=pre_critique_paper,
        brainstorm_content=brainstorm_content
    )


# =============================================================================
# PHASE-SPECIFIC BUILDER FUNCTIONS (Convenience wrappers)
# =============================================================================

async def build_body_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    is_first_portion: bool = False,
    rejection_feedback: Optional[str] = None,
    critique_feedback: Optional[str] = None,
    pre_critique_paper: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """
    Build prompt for BODY section construction phase.
    
    Args:
        user_prompt: User's compiler-directing prompt
        current_outline: Current outline (always fully injected)
        current_paper: Current document progress
        rag_evidence: RAG-retrieved evidence from aggregator database
        is_first_portion: Whether this is the first portion of the document
        rejection_feedback: Feedback from a previous rejection to guide the model
        critique_feedback: Accepted critique feedback from peer review (for rewrites only)
        pre_critique_paper: Paper state before critique phase (for rewrites - shows what failed)
        brainstorm_content: Full brainstorm database with submission numbers (autonomous mode)
    """
    return await build_phase_construction_prompt(
        user_prompt=user_prompt,
        current_outline=current_outline,
        current_paper=current_paper,
        rag_evidence=rag_evidence,
        phase="body",
        is_first_in_phase=is_first_portion,
        rejection_feedback=rejection_feedback,
        critique_feedback=critique_feedback,
        pre_critique_paper=pre_critique_paper,
        brainstorm_content=brainstorm_content
    )


async def build_conclusion_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    rejection_feedback: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """Build prompt for CONCLUSION section construction phase."""
    return await build_phase_construction_prompt(
        user_prompt=user_prompt,
        current_outline=current_outline,
        current_paper=current_paper,
        rag_evidence=rag_evidence,
        phase="conclusion",
        is_first_in_phase=True,
        rejection_feedback=rejection_feedback,
        brainstorm_content=brainstorm_content
    )


async def build_introduction_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    rejection_feedback: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """Build prompt for INTRODUCTION section construction phase."""
    return await build_phase_construction_prompt(
        user_prompt=user_prompt,
        current_outline=current_outline,
        current_paper=current_paper,
        rag_evidence=rag_evidence,
        phase="introduction",
        is_first_in_phase=True,
        rejection_feedback=rejection_feedback,
        brainstorm_content=brainstorm_content
    )


async def build_abstract_construction_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str,
    rejection_feedback: Optional[str] = None,
    brainstorm_content: Optional[str] = None
) -> str:
    """Build prompt for ABSTRACT section construction phase."""
    return await build_phase_construction_prompt(
        user_prompt=user_prompt,
        current_outline=current_outline,
        current_paper=current_paper,
        rag_evidence=rag_evidence,
        phase="abstract",
        is_first_in_phase=True,
        rejection_feedback=rejection_feedback,
        brainstorm_content=brainstorm_content
    )