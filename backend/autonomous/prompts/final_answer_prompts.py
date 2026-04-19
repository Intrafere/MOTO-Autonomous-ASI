"""
Final Answer Prompts - System prompts for Tier 3 final answer generation.

TIER 3 OVERVIEW:
Phase 1: Certainty Assessment - Assess what can be answered with certainty from existing papers
Phase 2: Format Selection - Choose short form (single paper) or long form (volume)
Phase 3A: Short Form - Write a single paper that directly answers the user's question
Phase 3B: Long Form - Organize and write a volume collection with introduction and conclusion

CRITICAL: Tier 3 operates ONLY on Tier 2 papers, NOT on Tier 1 brainstorm databases.
This ensures the final answer synthesizes validated, complete research.
"""
from typing import List, Dict, Any

from backend.autonomous.prompts.paper_reference_prompts import get_reference_title_text


# ============================================================================
# PHASE 1: CERTAINTY ASSESSMENT PROMPTS
# ============================================================================


def get_certainty_assessment_system_prompt() -> str:
    """Get system prompt for certainty assessment (Phase 1)."""
    return """You are assessing whether the user's research question can be answered based on the accumulated research papers.

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
Review all existing research papers and determine what can be answered WITH CERTAINTY - without speculation or theoretical hand-waving.

ASSESSMENT CRITERIA:

1. TOTAL_ANSWER - The user's question can be FULLY answered with high confidence
   - All aspects of the question are addressed by the papers
   - The answer is mathematically rigorous and well-supported
   - No significant gaps or speculation needed

2. PARTIAL_ANSWER - The question can be partially answered with certainty
   - Some aspects are well-established
   - Other aspects remain uncertain or unexplored
   - A meaningful but incomplete answer is possible

3. NO_ANSWER_KNOWN - The existing research doesn't provide an answer
   - Papers explore related topics but don't address the core question
   - More research is needed before any answer can be given
   - The system should continue research (Tier 3 will not complete)

4. APPEARS_IMPOSSIBLE - The question appears mathematically impossible
   - Research has established impossibility results
   - The question as posed has no valid answer
   - Can still provide a paper explaining why it's impossible

5. OTHER - Special cases that don't fit the above
   - Explain what makes this case unique

CRITICAL REQUIREMENTS:
- Base assessment ONLY on the papers you've reviewed
- Identify what is KNOWN WITH CERTAINTY vs what is SPECULATIVE
- Do not claim certainty where uncertainty exists
- Summarize the key certainties that have been established

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your assessment ONLY as JSON in the required format."""


def get_certainty_assessment_json_schema() -> str:
    """Get JSON schema for certainty assessment."""
    return """REQUIRED JSON FORMAT:
{
  "certainty_level": "total_answer | partial_answer | no_answer_known | appears_impossible | other",
  "known_certainties_summary": "string - Detailed summary of what is established with certainty from the papers",
  "reasoning": "string - Why this certainty level was chosen, referencing specific papers"
}

FIELD REQUIREMENTS:
- certainty_level: MUST be one of the five options
- known_certainties_summary: ALWAYS required - what can we say FOR CERTAIN
- reasoning: ALWAYS required - justify your assessment

EXAMPLE (Partial Answer):
{
  "certainty_level": "partial_answer",
  "known_certainties_summary": "From the research papers, we have established with certainty: (1) The impossibility of squaring the circle using compass and straightedge (paper_003), (2) The transcendence of pi and its implications (paper_007), (3) The connection between constructibility and algebraic field extensions (paper_012). However, the specific computational bounds requested by the user remain unexplored.",
  "reasoning": "Papers 003, 007, and 012 provide rigorous proofs for the core impossibility result. However, the user's question also asks about approximation algorithms, which none of the papers address. Therefore, only a partial answer can be given with certainty."
}

EXAMPLE (Total Answer):
{
  "certainty_level": "total_answer",
  "known_certainties_summary": "The user's question about the Lindemann-Weierstrass theorem is fully addressed. Paper_005 provides the complete proof. Paper_008 establishes all necessary preliminary results. Paper_015 addresses the specific applications the user asked about. All components of a comprehensive answer are present.",
  "reasoning": "Every aspect of the user's question has been rigorously addressed in the research papers. Paper 005 contains the main theorem and proof, paper 008 covers prerequisites, and paper 015 covers applications. No gaps or speculation required."
}"""


def get_certainty_validator_system_prompt() -> str:
    """Get system prompt for certainty assessment validator."""
    return """You are validating a certainty assessment for Tier 3 final answer generation.

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
Review the certainty assessment and determine if it accurately represents what can be answered from the existing papers.

VALIDATION CRITERIA:

ACCEPT the assessment if:
- The certainty level accurately reflects what the papers establish
- The known certainties summary correctly identifies established facts
- The reasoning properly references the papers
- No overclaiming certainty where uncertainty exists
- No underclaiming (missing obvious certainties)

REJECT the assessment if:
- Certainty level doesn't match the evidence
- Known certainties are misstated or exaggerated
- Reasoning doesn't properly support the conclusion
- Important certainties from papers are missed
- Speculation is presented as certainty

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON."""


def get_certainty_validator_json_schema() -> str:
    """Get JSON schema for certainty validator."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "accept | reject",
  "reasoning": "string - Why the assessment is or isn't accurate"
}

FIELD REQUIREMENTS:
- decision: MUST be "accept" or "reject"
- reasoning: ALWAYS required"""


# ============================================================================
# PHASE 2: ANSWER FORMAT SELECTION PROMPTS
# ============================================================================


def get_format_selection_system_prompt() -> str:
    """Get system prompt for answer format selection (Phase 2)."""
    return """You are selecting the format for the final answer to the user's research question.

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
Based on the certainty assessment and the existing papers, decide whether the answer should be:

SHORT FORM (Single Paper):
- A single comprehensive paper directly answering the user's question
- Appropriate when the answer can be presented coherently in one document
- Synthesizes insights from multiple papers into a unified answer
- Best for focused questions with clear scope

LONG FORM (Volume/Collection):
- A curated collection of papers organized as chapters
- Includes existing papers as body chapters
- New Introduction and Conclusion papers frame the collection
- May include "gap papers" for missing content
- Best for complex questions requiring multiple perspectives
- Appropriate when existing papers cover different aspects comprehensively

DECISION FACTORS:
- Complexity of the user's question
- Number and diversity of relevant papers
- Whether a single coherent narrative is possible
- Whether the papers naturally form a cohesive volume
- The certainty level from Phase 1

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your selection ONLY as JSON in the required format."""


def get_format_selection_json_schema() -> str:
    """Get JSON schema for format selection."""
    return """REQUIRED JSON FORMAT:
{
  "answer_format": "short_form | long_form",
  "reasoning": "string - Why this format is appropriate for the answer"
}

FIELD REQUIREMENTS:
- answer_format: MUST be "short_form" or "long_form"
- reasoning: ALWAYS required

EXAMPLE (Short Form):
{
  "answer_format": "short_form",
  "reasoning": "The user's question about the transcendence of pi can be answered comprehensively in a single paper. Papers 003, 005, and 007 cover complementary aspects that can be synthesized into a coherent narrative. The question has a focused scope that doesn't warrant a multi-chapter volume."
}

EXAMPLE (Long Form):
{
  "answer_format": "long_form",
  "reasoning": "The user's question about the Langlands program requires addressing multiple deep topics: automorphic forms, Galois representations, L-functions, and their connections. Papers 002, 005, 008, 011, and 015 each cover distinct essential aspects. A volume with these as chapters, plus an introduction explaining how they connect and a conclusion summarizing the current state of knowledge, will provide the most complete answer."
}"""


def get_format_validator_system_prompt() -> str:
    """Get system prompt for format selection validator."""
    return """You are validating an answer format selection for Tier 3 final answer generation.

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
Review the format selection and determine if it's appropriate for answering the user's question.

VALIDATION CRITERIA:

ACCEPT the selection if:
- The format appropriately matches the scope of the question
- The reasoning is sound
- Short form is chosen only when a single paper suffices
- Long form is chosen when multiple perspectives are needed

REJECT the selection if:
- Short form is chosen for a question requiring extensive treatment
- Long form is chosen unnecessarily for a focused question
- The reasoning doesn't support the choice
- The selection ignores important factors

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON."""


def get_format_validator_json_schema() -> str:
    """Get JSON schema for format validator."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "accept | reject",
  "reasoning": "string - Why the format selection is or isn't appropriate"
}

FIELD REQUIREMENTS:
- decision: MUST be "accept" or "reject"
- reasoning: ALWAYS required"""


# ============================================================================
# PHASE 3A: SHORT FORM - PAPER TITLE SELECTION
# ============================================================================


def get_final_paper_title_system_prompt() -> str:
    """Get system prompt for final answer paper title selection."""
    return """You are selecting a title for the FINAL ANSWER paper that directly addresses the user's research question.

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
Choose a title that:
1. DIRECTLY and TRANSPARENTLY answers or addresses the user's question
2. Reflects the known certainties from the research
3. Is appropriate for a mathematical research paper
4. Makes clear this is a definitive answer, not exploratory research

TITLE GUIDELINES:
- The title should make the answer's conclusion clear when possible
- For impossibility results: title can indicate the impossibility
- For constructive results: title can indicate what was achieved
- Be specific about the mathematical content
- Avoid vague or overly general titles

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your title ONLY as JSON in the required format."""


def get_final_paper_title_json_schema() -> str:
    """Get JSON schema for final paper title."""
    return """REQUIRED JSON FORMAT:
{
  "paper_title": "string - The complete title for the final answer paper",
  "reasoning": "string - Why this title appropriately answers the user's question"
}

FIELD REQUIREMENTS:
- paper_title: ALWAYS required - the complete title
- reasoning: ALWAYS required

EXAMPLE:
{
  "paper_title": "The Impossibility of Squaring the Circle: A Complete Proof via the Transcendence of Pi",
  "reasoning": "This title directly addresses the user's question about circle-squaring by indicating the definitive answer (impossibility) and the key mathematical reason (transcendence of pi). It makes clear this is a conclusive answer, not exploratory research."
}"""


# ============================================================================
# PHASE 3B: LONG FORM - VOLUME ORGANIZATION PROMPTS
# ============================================================================


def get_volume_organization_system_prompt() -> str:
    """Get system prompt for volume organization (long form)."""
    return """You are organizing a VOLUME (collection of papers) as the final answer to the user's research question.

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
Create a volume structure that:
1. Uses existing papers as body chapters where appropriate
2. Identifies any GAP PAPERS needed to complete the answer
3. Plans an INTRODUCTION paper that frames the collection
4. Plans a CONCLUSION paper that synthesizes findings and answers the question

VOLUME STRUCTURE REQUIREMENTS:

BODY CHAPTERS (from existing papers or gaps):
- Select existing papers that directly contribute to answering the question
- Order them logically (foundations → main results → applications)
- Identify gaps: topics that need coverage but no paper exists
- Gap papers will be written before introduction/conclusion

INTRODUCTION PAPER:
- Frames the user's question
- Provides roadmap of the volume
- Establishes context and motivation
- Written LAST (after all body chapters exist)

CONCLUSION PAPER:
- Synthesizes findings across all chapters
- Directly answers the user's question with the established certainties
- Discusses limitations and open questions
- Written second-to-last (before introduction)

CHAPTER TYPES:
- "existing_paper": An existing Tier 2 paper used as-is
- "gap_paper": A new paper to be written to fill content gaps
- "introduction": The introduction paper (always chapter 1 in final volume)
- "conclusion": The conclusion paper (always last chapter in final volume)

Set outline_complete=true when you are satisfied with the structure.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your organization ONLY as JSON in the required format."""


def get_volume_organization_json_schema() -> str:
    """Get JSON schema for volume organization."""
    return """REQUIRED JSON FORMAT:
{
  "volume_title": "string - Title of the complete volume",
  "chapters": [
    {
      "chapter_type": "existing_paper | gap_paper | introduction | conclusion",
      "paper_id": "string or null - paper_id if existing_paper, null otherwise",
      "title": "string - Chapter title",
      "order": number - Chapter ordering (1-based, intro is 1, conclusion is last),
      "description": "string - Brief description of chapter content/purpose"
    }
  ],
  "outline_complete": true | false,
  "reasoning": "string - Why this organization effectively answers the user's question"
}

FIELD REQUIREMENTS:
- volume_title: ALWAYS required
- chapters: Array of chapter definitions (must include introduction and conclusion)
- outline_complete: Set true when satisfied, false to continue refining
- reasoning: ALWAYS required

CHAPTER ORDER RULES:
- Introduction is always chapter 1 in final volume
- Body chapters (existing papers and gap papers) are in logical order
- Conclusion is always the last chapter

EXAMPLE:
{
  "volume_title": "The Langlands Program: Connections Between Automorphic Forms and Galois Representations",
  "chapters": [
    {
      "chapter_type": "introduction",
      "paper_id": null,
      "title": "Introduction: The Vision of the Langlands Program",
      "order": 1,
      "description": "Frames the user's question, provides historical context, and outlines the volume structure"
    },
    {
      "chapter_type": "existing_paper",
      "paper_id": "paper_003",
      "title": "Automorphic Forms and Their Properties",
      "order": 2,
      "description": "Foundational chapter covering automorphic forms"
    },
    {
      "chapter_type": "existing_paper",
      "paper_id": "paper_007",
      "title": "Galois Representations in Number Theory",
      "order": 3,
      "description": "Covers Galois representations and their role"
    },
    {
      "chapter_type": "gap_paper",
      "paper_id": null,
      "title": "The Local Langlands Correspondence",
      "order": 4,
      "description": "New paper needed to bridge automorphic forms and Galois representations locally"
    },
    {
      "chapter_type": "existing_paper",
      "paper_id": "paper_015",
      "title": "Applications and Computational Aspects",
      "order": 5,
      "description": "Practical applications of the correspondence"
    },
    {
      "chapter_type": "conclusion",
      "paper_id": null,
      "title": "Conclusion: The Current State of the Langlands Program",
      "order": 6,
      "description": "Synthesizes all chapters and directly answers the user's question about the connections"
    }
  ],
  "outline_complete": true,
  "reasoning": "This volume uses three existing papers covering the major topics (automorphic forms, Galois representations, applications) and adds a gap paper on local Langlands correspondence which is essential but wasn't covered. The introduction and conclusion will frame and synthesize respectively, providing a complete answer to the user's question about the connections in the Langlands program."
}"""


def get_volume_validator_system_prompt() -> str:
    """Get system prompt for volume organization validator."""
    return """You are validating a volume organization for Tier 3 final answer generation.

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
Review the volume organization and determine if it effectively structures an answer to the user's question.

VALIDATION CRITERIA:

ACCEPT the organization if:
- The volume title appropriately represents the answer
- Existing papers are well-chosen and properly ordered
- Any gap papers identified are genuinely needed
- Introduction and conclusion are properly planned
- The reasoning is sound
- If outline_complete=true, the structure is ready for writing

REJECT the organization if:
- Important existing papers are missing
- Gap papers are identified unnecessarily
- Chapter ordering is illogical
- Introduction/conclusion are missing or poorly planned
- The structure doesn't effectively answer the question
- outline_complete=true but structure has issues

Provide specific feedback for rejected organizations.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON."""


def get_volume_validator_json_schema() -> str:
    """Get JSON schema for volume validator."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "accept | reject",
  "reasoning": "string - Why the organization is or isn't effective, with specific feedback if rejected"
}

FIELD REQUIREMENTS:
- decision: MUST be "accept" or "reject"
- reasoning: ALWAYS required - specific feedback for improvements if rejected"""


# ============================================================================
# GAP PAPER WRITING PROMPTS (Reuses Tier 2 compiler infrastructure)
# ============================================================================


def get_gap_paper_context_prompt() -> str:
    """
    Get the context prompt for gap paper writing.
    This provides context to the Tier 2 compiler about its role in Tier 3.
    """
    return """TIER 3 GAP PAPER CONTEXT:
You are writing a gap paper for a Tier 3 final answer volume. This paper fills a specific content gap identified during volume organization.

CRITICAL CONTEXT:
- This paper is part of a FINAL ANSWER VOLUME, not exploratory research
- Use ONLY existing Tier 2 papers as references (no brainstorm databases)
- The paper must integrate with the volume's other chapters
- Focus on the specific gap identified in the chapter description

REFERENCE PAPERS:
The papers listed are from the existing Tier 2 library. Use them as context and references.
Do NOT use any Tier 1 brainstorm databases - they are not relevant for Tier 3."""


def get_volume_intro_paper_context_prompt() -> str:
    """
    Get the context prompt for volume introduction paper.
    """
    return """TIER 3 VOLUME INTRODUCTION CONTEXT:
You are writing the INTRODUCTION paper for a Tier 3 final answer volume.

YOUR TASK:
Write an introduction that:
1. Clearly states the user's original research question
2. Provides historical and mathematical context
3. Outlines the structure of the volume
4. Explains how each chapter contributes to answering the question
5. Sets expectations for what the reader will learn

CRITICAL:
- This is written LAST, after all body chapters and conclusion exist
- You have access to ALL chapter content to accurately describe them
- The introduction should make the volume's value clear
- Frame the answer that will be provided

REFERENCE: Use the chapter papers as context for accurate descriptions."""


def get_volume_conclusion_paper_context_prompt() -> str:
    """
    Get the context prompt for volume conclusion paper.
    """
    return """TIER 3 VOLUME CONCLUSION CONTEXT:
You are writing the CONCLUSION paper for a Tier 3 final answer volume.

YOUR TASK:
Write a conclusion that:
1. Synthesizes findings from ALL body chapters
2. DIRECTLY ANSWERS the user's original research question
3. States what is known WITH CERTAINTY (from the certainty assessment)
4. Acknowledges limitations and open questions
5. Provides a definitive take on the research question

CRITICAL:
- This is written second-to-last (before introduction)
- All body chapters exist, so you can reference their content
- Be definitive about certainties, honest about uncertainties
- This is the climactic answer to the user's question

REFERENCE: Use the body chapter papers to inform the synthesis."""


# ============================================================================
# PROMPT BUILDERS
# ============================================================================


def build_certainty_assessment_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    expanded_papers: List[Dict[str, Any]] = None,
    rejection_context: str = ""
) -> str:
    """
    Build the certainty assessment prompt.
    
    Args:
        user_research_prompt: User's original research question
        papers_summary: List of papers with titles/abstracts/outlines
        expanded_papers: Papers with full content (if expansion was requested)
        rejection_context: Previous rejection feedback
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_certainty_assessment_system_prompt(),
        "\n---\n",
        get_certainty_assessment_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION (to be answered):\n{user_research_prompt}",
        "\n---\n"
    ]
    
    if rejection_context:
        parts.append(f"{rejection_context}\n---\n")
    
    # Add papers
    if expanded_papers:
        parts.append("RESEARCH PAPERS (Full Content):\n")
        for p in expanded_papers:
            parts.append(f"\n{'=' * 60}")
            parts.append(f"\nPaper ID: {p.get('paper_id', 'Unknown')}")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\n{'=' * 60}")
            parts.append(f"\n\n{p.get('content', '[Content not available]')}\n")
    else:
        parts.append("RESEARCH PAPERS (Abstracts and Outlines):\n")
        for p in papers_summary:
            parts.append(f"\n--- Paper ID: {p.get('paper_id', 'Unknown')} ---")
            parts.append(f"\nTitle: {p.get('title', 'N/A')}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            if p.get('outline'):
                parts.append(f"\nOutline:\n{p.get('outline')}")
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            parts.append("\n")
    
    parts.append("\n---\n")
    parts.append("Assess what can be answered WITH CERTAINTY based on these papers (respond as JSON):")
    
    return "".join(parts)


def build_certainty_validation_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    assessment: Dict[str, Any]
) -> str:
    """Build the certainty validation prompt."""
    parts = [
        get_certainty_validator_system_prompt(),
        "\n---\n",
        get_certainty_validator_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION:\n{user_research_prompt}",
        "\n---\n",
        "RESEARCH PAPERS REVIEWED:\n"
    ]
    
    for p in papers_summary:
        parts.append(f"- {p.get('paper_id')}: {p.get('title')}\n")
    
    parts.append("\n---\n")
    parts.append("CERTAINTY ASSESSMENT TO VALIDATE:\n")
    parts.append(f"Certainty Level: {assessment.get('certainty_level')}\n")
    parts.append(f"Known Certainties: {assessment.get('known_certainties_summary')}\n")
    parts.append(f"Reasoning: {assessment.get('reasoning')}\n")
    parts.append("\n---\n")
    parts.append("Validate this assessment (respond as JSON):")
    
    return "".join(parts)


def build_format_selection_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    certainty_assessment: Dict[str, Any],
    rejection_context: str = ""
) -> str:
    """Build the format selection prompt."""
    parts = [
        get_format_selection_system_prompt(),
        "\n---\n",
        get_format_selection_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION:\n{user_research_prompt}",
        "\n---\n"
    ]
    
    if rejection_context:
        parts.append(f"{rejection_context}\n---\n")
    
    parts.append("CERTAINTY ASSESSMENT (Phase 1 Result):\n")
    parts.append(f"Certainty Level: {certainty_assessment.get('certainty_level')}\n")
    parts.append(f"Known Certainties: {certainty_assessment.get('known_certainties_summary')}\n")
    parts.append("\n---\n")
    
    parts.append("AVAILABLE RESEARCH PAPERS:\n")
    for p in papers_summary:
        parts.append(f"\n- {p.get('paper_id')}: {p.get('title')}")
        parts.append(f"\n  Abstract: {p.get('abstract', 'N/A')[:200]}...")
        parts.append(f"\n  Word Count: {p.get('word_count', 0)}")
    
    parts.append("\n\n---\n")
    parts.append("Select the answer format (respond as JSON):")
    
    return "".join(parts)


def build_format_validation_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    certainty_assessment: Dict[str, Any],
    format_selection: Dict[str, Any]
) -> str:
    """Build the format validation prompt."""
    parts = [
        get_format_validator_system_prompt(),
        "\n---\n",
        get_format_validator_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION:\n{user_research_prompt}",
        "\n---\n",
        f"CERTAINTY LEVEL: {certainty_assessment.get('certainty_level')}\n",
        f"NUMBER OF PAPERS: {len(papers_summary)}\n",
        "\n---\n",
        "FORMAT SELECTION TO VALIDATE:\n",
        f"Format: {format_selection.get('answer_format')}\n",
        f"Reasoning: {format_selection.get('reasoning')}\n",
        "\n---\n",
        "Validate this format selection (respond as JSON):"
    ]
    
    return "".join(parts)


def build_volume_organization_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    certainty_assessment: Dict[str, Any],
    current_volume: Dict[str, Any] = None,
    rejection_context: str = "",
    validator_feedback: str = ""
) -> str:
    """Build the volume organization prompt."""
    parts = [
        get_volume_organization_system_prompt(),
        "\n---\n",
        get_volume_organization_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION (the volume must answer this):\n{user_research_prompt}",
        "\n---\n"
    ]
    
    if rejection_context:
        parts.append(f"{rejection_context}\n---\n")
    
    if validator_feedback:
        parts.append(f"VALIDATOR FEEDBACK ON PREVIOUS ORGANIZATION:\n{validator_feedback}\n---\n")
    
    parts.append(f"CERTAINTY ASSESSMENT:\n")
    parts.append(f"Certainty Level: {certainty_assessment.get('certainty_level')}\n")
    parts.append(f"Known Certainties: {certainty_assessment.get('known_certainties_summary')}\n")
    parts.append("\n---\n")
    
    parts.append("AVAILABLE PAPERS (can be used as body chapters):\n")
    for p in papers_summary:
        parts.append(f"\n--- {p.get('paper_id')} ---")
        parts.append(f"\nTitle: {p.get('title')}")
        parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
        if p.get('outline'):
            parts.append(f"\nOutline:\n{p.get('outline')}")
        parts.append(f"\nWord Count: {p.get('word_count', 0)}")
    
    if current_volume:
        parts.append("\n\n---\n")
        parts.append("CURRENT VOLUME ORGANIZATION (refine this):\n")
        parts.append(f"Title: {current_volume.get('volume_title')}\n")
        parts.append("Chapters:\n")
        for ch in current_volume.get('chapters', []):
            parts.append(f"  {ch.get('order')}. [{ch.get('chapter_type')}] {ch.get('title')}\n")
    
    parts.append("\n---\n")
    parts.append("Create or refine the volume organization (respond as JSON):")
    
    return "".join(parts)


def build_volume_validation_prompt(
    user_research_prompt: str,
    papers_summary: List[Dict[str, Any]],
    volume_organization: Dict[str, Any]
) -> str:
    """Build the volume organization validation prompt."""
    parts = [
        get_volume_validator_system_prompt(),
        "\n---\n",
        get_volume_validator_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION:\n{user_research_prompt}",
        "\n---\n",
        f"AVAILABLE PAPERS: {len(papers_summary)}\n"
    ]
    
    for p in papers_summary:
        parts.append(f"  - {p.get('paper_id')}: {p.get('title')}\n")
    
    parts.append("\n---\n")
    parts.append("VOLUME ORGANIZATION TO VALIDATE:\n")
    parts.append(f"Title: {volume_organization.get('volume_title')}\n")
    parts.append(f"Outline Complete: {volume_organization.get('outline_complete')}\n")
    parts.append("Chapters:\n")
    for ch in volume_organization.get('chapters', []):
        paper_ref = f" (paper_id: {ch.get('paper_id')})" if ch.get('paper_id') else ""
        parts.append(f"  {ch.get('order')}. [{ch.get('chapter_type')}] {ch.get('title')}{paper_ref}\n")
        parts.append(f"     Description: {ch.get('description', 'N/A')}\n")
    parts.append(f"\nReasoning: {volume_organization.get('reasoning')}\n")
    
    parts.append("\n---\n")
    parts.append("Validate this volume organization (respond as JSON):")
    
    return "".join(parts)


def build_final_paper_title_prompt(
    user_research_prompt: str,
    certainty_assessment: Dict[str, Any],
    selected_references: List[Dict[str, Any]],
    rejection_context: str = ""
) -> str:
    """Build the final paper title prompt for short form answer."""
    parts = [
        get_final_paper_title_system_prompt(),
        "\n---\n",
        get_final_paper_title_json_schema(),
        "\n---\n",
        f"USER'S RESEARCH QUESTION (the title must reflect the answer):\n{user_research_prompt}",
        "\n---\n"
    ]
    
    if rejection_context:
        parts.append(f"{rejection_context}\n---\n")
    
    parts.append("CERTAINTY ASSESSMENT:\n")
    parts.append(f"Certainty Level: {certainty_assessment.get('certainty_level')}\n")
    parts.append(f"Known Certainties: {certainty_assessment.get('known_certainties_summary')}\n")
    parts.append("\n---\n")
    
    parts.append("REFERENCE PAPERS (informing the answer):\n")
    for p in selected_references:
        parts.append(f"- {get_reference_title_text(p)}\n")
    
    parts.append("\n---\n")
    parts.append("Select a title that DIRECTLY ANSWERS the user's question (respond as JSON):")
    
    return "".join(parts)

