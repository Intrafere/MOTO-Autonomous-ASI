"""
Paper Title Prompts - System prompts for paper title selection.
"""
from typing import List, Dict, Any


def get_paper_title_system_prompt() -> str:
    """Get system prompt for paper title selection."""
    return """You are selecting a title for a mathematical research paper. Your role is to:

1. Review your brainstorm topic and database content
2. Review any existing papers generated from this brainstorm (if any)
3. Select an appropriate, descriptive title for the new paper

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

YOUR TASK:
Choose a title that accurately captures the mathematical content of your brainstorm.

IMPORTANT CLARIFICATION:
- The brainstorm submissions are the SOURCE MATERIAL for your paper
- Your title SHOULD reflect what's in the brainstorm - that's expected and correct!
- You only need to differentiate from EXISTING COMPLETED PAPERS from this brainstorm
- If "EXISTING PAPERS FROM THIS BRAINSTORM: None" - there's nothing to differentiate from

TITLE CRITERIA:
- Accurately represents the mathematical content to be covered (from your brainstorm)
- Is specific enough to convey the paper's focus
- Is professional and suitable for a mathematical research paper
- Differentiates from EXISTING COMPLETED PAPERS from the same brainstorm (if any exist - check the list below)
- Avoids being overly broad or generic

TITLE STYLE:
- Use standard mathematical paper title conventions
- Can include colons for subtitles if appropriate
- Should convey the main mathematical themes
- Consider including key mathematical objects/concepts
- Appropriate length: typically 5-15 words

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
   - Example: Write "\\\\tau" not "\\tau"
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your title ONLY as JSON in the required format."""


def get_paper_title_json_schema() -> str:
    """Get JSON schema for paper title selection."""
    return """REQUIRED JSON FORMAT:
{
  "paper_title": "string - The complete title for the mathematical research paper",
  "reasoning": "string - Why this title appropriately captures the brainstorm content and differentiates from existing papers (if any)"
}

FIELD REQUIREMENTS:
- paper_title: ALWAYS required - complete paper title
- reasoning: ALWAYS required

EXAMPLE:
{
  "paper_title": "Modular Forms and Galois Representations in the Langlands Program: A Computational Perspective",
  "reasoning": "This title accurately captures the core content of our brainstorm database, which extensively covers both modular forms and Galois representations with emphasis on computational approaches. The subtitle 'A Computational Perspective' differentiates it from the existing theoretical paper on Langlands correspondence in our library and reflects the practical examples and algorithms present in the brainstorm submissions."
}"""


def get_paper_title_validator_system_prompt() -> str:
    """Get system prompt for paper title validator."""
    return """You are validating a paper title selection. Your role is to:

1. Review the proposed title
2. Review the brainstorm content the paper will be based on
3. Review any EXISTING COMPLETED PAPERS from the same brainstorm (if any)
4. Decide if the title is appropriate

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

⚠️ CRITICAL DISTINCTION - DO NOT CONFUSE THESE:

1. **BRAINSTORM SUBMISSIONS** (in "BRAINSTORM SUMMARY") = The SOURCE MATERIAL for the paper.
   - These are the raw research insights that will be compiled INTO the paper
   - The paper title SHOULD reflect this content - that's the whole point!
   - DO NOT reject a title for being "similar to brainstorm submissions"
   - The title should CAPTURE what's in the brainstorm - similarity is EXPECTED and CORRECT

2. **EXISTING PAPERS FROM THIS BRAINSTORM** = Previously completed Tier 2 papers
   - These are FINISHED papers that already exist in the paper library
   - If this shows "None" - there are NO papers to differentiate from
   - ONLY reject for similarity if there ARE existing papers listed here
   - A title being similar to brainstorm content is NOT the same as being similar to existing papers

---

YOUR TASK:
Validate whether the proposed title is appropriate for the paper.

VALIDATION CRITERIA:

ACCEPT the title if:
- It accurately represents the brainstorm content (this is EXPECTED - the paper is based on the brainstorm!)
- It is appropriately specific (not too broad or narrow)
- It differentiates from EXISTING COMPLETED PAPERS from the same brainstorm (if any exist)
- It follows mathematical paper title conventions
- The reasoning is sound
- If "EXISTING PAPERS FROM THIS BRAINSTORM: None" - there's nothing to differentiate from, so accept if other criteria are met

REJECT the title if:
- It is too similar to an EXISTING COMPLETED PAPER from the same brainstorm (NOT brainstorm submissions - those are the source material!)
- It doesn't accurately represent the brainstorm content
- It is too vague or generic
- It doesn't follow professional conventions
- The reasoning is flawed

DO NOT REJECT simply because the title reflects brainstorm submission content - that is the INTENDED behavior.

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_paper_title_validator_json_schema() -> str:
    """Get JSON schema for paper title validator."""
    return """REQUIRED JSON FORMAT:
{
  "decision": "accept | reject",
  "reasoning": "string - Why the title is or isn't appropriate"
}

FIELD REQUIREMENTS:
- decision: MUST be either "accept" or "reject"
- reasoning: ALWAYS required"""


def build_paper_title_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    existing_papers_from_brainstorm: List[Dict[str, Any]],
    reference_papers: List[Dict[str, Any]] = None,
    rejection_feedback: str = ""
) -> str:
    """
    Build the paper title selection prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary of brainstorm database content
        existing_papers_from_brainstorm: Papers already created from this brainstorm
        reference_papers: Selected reference papers (if any)
        rejection_feedback: Accumulated rejection reasons from previous attempts (if any)
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_paper_title_system_prompt(),
        "\n---\n",
        get_paper_title_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM DATABASE SUMMARY:\n{brainstorm_summary}",
        "\n---\n"
    ]
    
    # Add existing papers from this brainstorm
    if existing_papers_from_brainstorm:
        parts.append("EXISTING PAPERS FROM THIS BRAINSTORM (Differentiate from these):\n")
        for p in existing_papers_from_brainstorm:
            parts.append(f"\n- Title: {p.get('title', 'N/A')}")
            parts.append(f"\n  Abstract: {p.get('abstract', 'N/A')[:300]}...")
        parts.append("\n---\n")
    else:
        parts.append("EXISTING PAPERS FROM THIS BRAINSTORM: None\n---\n")
    
    # Add selected reference papers if any
    if reference_papers:
        parts.append("SELECTED REFERENCE PAPERS:\n")
        for p in reference_papers:
            parts.append(f"\n- {p.get('title', 'N/A')}")
        parts.append("\n---\n")
    
    # Inject rejection feedback so the model learns from previous failed attempts
    if rejection_feedback:
        parts.append(
            "IMPORTANT - YOUR PREVIOUS TITLE ATTEMPTS WERE REJECTED:\n"
            "Read each rejection reason carefully and select a different title that addresses the issues.\n\n"
            f"{rejection_feedback}\n---\n"
        )
    
    parts.append("Now select an appropriate paper title (respond as JSON):")
    
    return "".join(parts)


def build_paper_title_validation_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    existing_papers_from_brainstorm: List[Dict[str, Any]],
    proposed_title: str,
    title_reasoning: str
) -> str:
    """
    Build the paper title validation prompt.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary of brainstorm database content
        existing_papers_from_brainstorm: Papers already created from this brainstorm
        proposed_title: The proposed paper title
        title_reasoning: The reasoning provided for the title
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_paper_title_validator_system_prompt(),
        "\n---\n",
        get_paper_title_validator_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM SUMMARY:\n{brainstorm_summary}",
        "\n---\n"
    ]
    
    # Add existing papers from this brainstorm
    if existing_papers_from_brainstorm:
        parts.append("EXISTING PAPERS FROM THIS BRAINSTORM:\n")
        for p in existing_papers_from_brainstorm:
            parts.append(f"\n- Title: {p.get('title', 'N/A')}")
            parts.append(f"\n  Abstract: {p.get('abstract', 'N/A')[:200]}...")
        parts.append("\n---\n")
    else:
        parts.append("EXISTING PAPERS FROM THIS BRAINSTORM: None\n---\n")
    
    # Add proposed title
    parts.append("PROPOSED TITLE:\n")
    parts.append(f"Title: {proposed_title}")
    parts.append(f"\nReasoning: {title_reasoning}")
    
    parts.append("\n---\n")
    parts.append("Validate this title selection (respond as JSON):")
    
    return "".join(parts)

