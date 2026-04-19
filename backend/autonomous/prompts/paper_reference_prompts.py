"""
Paper Reference Prompts - System prompts for reference paper selection workflow.
Implements two-step process: show abstracts -> show full papers -> final selection.

Supports two modes:
- "initial": Pre-brainstorm selection (select papers to inform brainstorm exploration)
- "additional": Pre-paper selection (select additional papers, keeping already-selected ones)

This is the CRUCIAL MECHANISM that enables COMPOUNDING KNOWLEDGE across research cycles.
By selecting reference papers before brainstorming, submitters can:
- Build upon proven mathematical frameworks from prior papers
- Avoid re-exploring territory already covered in depth
- Identify novel connections between new topics and established results
- Accelerate convergence on valuable insights by standing on prior work
"""
from typing import List, Dict, Any


def get_reference_title_text(paper: Dict[str, Any]) -> str:
    """Get the display title for a reference paper, including validator context when available."""
    return paper.get("reference_title_display") or paper.get("title", "N/A")


def get_pre_brainstorm_expansion_system_prompt(max_papers: int) -> str:
    """
    Get system prompt for PRE-BRAINSTORM reference expansion request.
    This is the crucial mechanism for compounding knowledge across research cycles.
    """
    return f"""You are selecting reference papers to inform your upcoming BRAINSTORM EXPLORATION. Your role is to:

1. Review your brainstorm topic that you will explore
2. Review titles and abstracts of existing papers in the library
3. Identify which papers would be VERY USEFUL to have as context DURING brainstorming

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
Determine which papers (if any) would be VERY USEFUL to inform and enhance your brainstorm exploration.

WHY THIS MATTERS - COMPOUNDING KNOWLEDGE:
This is the crucial mechanism that allows the system to compound knowledge across research cycles.
By selecting reference papers BEFORE brainstorming, you can:
- Build upon proven mathematical frameworks from prior papers
- Avoid re-exploring territory already covered in depth
- Identify novel connections between your new topic and established results
- Accelerate convergence on valuable insights by standing on prior work

THRESHOLD: "VERY USEFUL FOR BRAINSTORMING"
- Papers that provide mathematical foundations you'll build upon
- Papers that cover related concepts you can extend or connect to
- Papers that offer techniques or methods relevant to your topic
- Don't request papers that are merely tangentially related

OPTIONS:
1. Request to EXPAND specific papers (see full content before deciding)
2. Proceed WITHOUT references (none meet the "very useful" threshold)

IMPORTANT CONSTRAINTS:
- You can select up to {max_papers} papers maximum
- These papers will be available during your entire brainstorm exploration
- The same papers will also be available during paper writing
- Quality over quantity - only select papers you genuinely need

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_additional_reference_expansion_system_prompt(max_total_papers: int) -> str:
    """
    Get system prompt for ADDITIONAL reference expansion request (before paper writing).
    """
    return f"""You are selecting ADDITIONAL reference papers for your upcoming paper compilation. Your role is to:

1. Review your completed brainstorm database
2. Review titles and abstracts of papers NOT YET selected
3. Identify additional papers that would be VERY USEFUL based on insights from brainstorming

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
Determine if any ADDITIONAL papers would be valuable for paper compilation, based on what you learned during brainstorming.

CONTEXT:
- You already selected reference papers before brainstorming (shown as "ALREADY SELECTED")
- During brainstorming, you may have discovered new connections or topics
- This is your chance to add more relevant papers (if any)

THRESHOLD: "VALUABLE BASED ON BRAINSTORM INSIGHTS"
- Papers that address topics that emerged during brainstorming
- Papers that provide additional techniques you now realize are relevant
- Papers that cover connections you discovered during exploration
- Don't add papers just to fill slots

OPTIONS:
1. Request to EXPAND specific papers (see full content before deciding)
2. Proceed WITHOUT additional references (already selected papers are sufficient)

IMPORTANT CONSTRAINTS:
- Check how many slots remain (max {max_total_papers} total including already selected)
- Already selected papers WILL be kept - you're only adding new ones
- Quality over quantity - only add genuinely useful papers

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_reference_expansion_system_prompt(max_papers: int = 6) -> str:
    """Get system prompt for reference expansion request (Step 1: abstracts only)."""
    return f"""You are selecting reference papers for an upcoming mathematical research paper. Your role is to:

1. Review your brainstorm topic and database
2. Review titles and abstracts of existing papers in the library
3. Identify which papers would be VERY USEFUL as references

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
Determine which papers (if any) would be VERY USEFUL for writing your upcoming paper, and request to see their full content before making final selection.

THRESHOLD: "VERY USEFUL"
- A paper is "very useful" if it provides substantial mathematical context, techniques, or insights directly relevant to your brainstorm topic
- Don't request papers that are merely tangentially related
- Quality over quantity - only request papers you genuinely need to evaluate

OPTIONS:
1. Request to EXPAND specific papers (see full content before deciding)
2. Proceed WITHOUT references (none meet the "very useful" threshold)

IMPORTANT CONSTRAINTS:
- In the final selection (next step), you can only select up to {max_papers} papers
- You can request to expand as many papers as you want to review
- Only request expansion for papers that genuinely might be "very useful"

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your decision ONLY as JSON in the required format."""


def get_reference_expansion_json_schema() -> str:
    """Get JSON schema for reference expansion request."""
    return """REQUIRED JSON FORMAT:
{
  "expand_papers": ["array of paper_ids to see full content"],
  "proceed_without_references": false,
  "reasoning": "string - Why these papers should be expanded OR why no papers meet the 'very useful' threshold"
}

FIELD REQUIREMENTS:
- expand_papers: Array of paper IDs (can be empty)
- proceed_without_references: Boolean - set true if no papers are very useful
- reasoning: ALWAYS required

EXAMPLES:

Expand Papers:
{
  "expand_papers": ["paper_003", "paper_007", "paper_011"],
  "proceed_without_references": false,
  "reasoning": "Papers 003, 007, and 011 appear highly relevant based on their abstracts. Paper 003 covers class field theory which connects directly to our brainstorm on reciprocity laws. Papers 007 and 011 discuss Galois representations and modular forms respectively, both central to our upcoming paper. Need to see full content to assess their utility for reference."
}

Proceed Without References:
{
  "expand_papers": [],
  "proceed_without_references": true,
  "reasoning": "After reviewing all existing paper abstracts, none meet the 'very useful' threshold for the upcoming paper on modular forms and Galois representations. The existing papers focus on different aspects of Langlands program (L-functions, automorphic forms) that don't provide direct reference value for this specific paper topic."
}"""


def get_reference_selection_system_prompt(max_papers: int) -> str:
    """Get system prompt for final reference selection (Step 2: full papers)."""
    return f"""You are making your FINAL SELECTION of reference papers for an upcoming mathematical research paper. Your role is to:

1. Review your brainstorm topic and database
2. Review the FULL CONTENT of the papers you requested to expand
3. Select which papers (up to {max_papers}) will be used as references during paper writing

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
Make your final selection of reference papers (maximum {max_papers}) that will be included in your context during paper compilation.

SELECTION CRITERIA:
- Papers that provide essential mathematical background
- Papers that offer techniques or methods relevant to your topic
- Papers that establish theoretical foundations you'll build upon
- Papers that present related results you'll reference or extend

CONSTRAINT:
- Maximum {max_papers} papers can be selected (hard limit for context budget)
- These papers will be RAG'd during paper compilation
- Your brainstorm database has higher direct injection priority

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS use double backslash (\\\\) for any backslash in your text
2. Quotes: Escape double quotes inside strings as \\"
3. Newlines/Tabs: Use \\n for newlines, \\t for tabs

Output your selection ONLY as JSON in the required format."""


def get_reference_selection_json_schema(max_papers: int) -> str:
    """Get JSON schema for final reference selection."""
    return f"""REQUIRED JSON FORMAT:
{{
  "selected_papers": ["array of up to {max_papers} paper_ids"],
  "reasoning": "string - Why these specific papers are very useful for the upcoming paper"
}}

FIELD REQUIREMENTS:
- selected_papers: Array of paper IDs (maximum {max_papers}, can be empty)
- reasoning: ALWAYS required

EXAMPLE:
{{
  "selected_papers": ["paper_003", "paper_007", "paper_011"],
  "reasoning": "After reviewing full content, these three papers provide the most useful reference material: Paper 003 establishes the class field theory foundation needed for our reciprocity discussions. Paper 007's treatment of Galois representations will inform our theoretical sections. Paper 011's computational examples of modular forms will enhance our practical demonstrations. The other expanded papers, while relevant, overlap too much with our brainstorm content or cover tangential topics."
}}"""


def build_pre_brainstorm_expansion_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    papers_with_abstracts: List[Dict[str, Any]],
    max_papers: int
) -> str:
    """
    Build the PRE-BRAINSTORM reference expansion prompt.
    This is the crucial mechanism for compounding knowledge across research cycles.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary (typically "[Brainstorm not yet started]")
        papers_with_abstracts: List of papers with title and abstract
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_pre_brainstorm_expansion_system_prompt(max_papers),
        "\n---\n",
        get_reference_expansion_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"BRAINSTORM TOPIC YOU WILL EXPLORE:\n{topic_prompt}",
        "\n---\n"
    ]
    
    # Add papers with abstracts and outlines
    if papers_with_abstracts:
        parts.append("EXISTING PAPERS IN LIBRARY (select references to inform your brainstorm):\n")
        for p in papers_with_abstracts:
            parts.append(f"\n--- Paper ID: {p.get('paper_id', 'Unknown')} ---")
            parts.append(f"\nTitle: {get_reference_title_text(p)}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            
            # NEW: Display outline
            outline = p.get('outline', '')
            if outline:
                parts.append(f"\nOutline:\n{outline}")
            else:
                parts.append("\nOutline: [Not available]")
            
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            source_ids = p.get('source_brainstorm_ids', [])
            if source_ids:
                parts.append(f"\nSource Brainstorms: {', '.join(source_ids)}")
            parts.append("\n")
        parts.append("\n---\n")
    else:
        parts.append("EXISTING PAPERS IN LIBRARY: None\n---\n")
    
    parts.append("REMINDER: These references will be available during your ENTIRE brainstorm exploration AND paper writing.")
    parts.append("\n\nReview the abstracts and outlines, then decide which papers to expand (respond as JSON):")
    
    return "".join(parts)


def build_additional_reference_expansion_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    papers_with_abstracts: List[Dict[str, Any]],
    already_selected: List[str],
    already_selected_papers: List[Dict[str, Any]],
    max_total_papers: int
) -> str:
    """
    Build the ADDITIONAL reference expansion prompt (before paper writing).
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary of completed brainstorm content
        papers_with_abstracts: List of papers NOT YET selected
        already_selected: List of paper_ids already selected for this topic
    
    Returns:
        Complete prompt string
    """
    remaining_slots = max(0, max_total_papers - len(already_selected))
    
    parts = [
        get_additional_reference_expansion_system_prompt(max_total_papers),
        "\n---\n",
        get_reference_expansion_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"YOUR BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM SUMMARY (insights you developed):\n{brainstorm_summary}",
        "\n---\n",
        f"ALREADY SELECTED PAPERS ({len(already_selected)} papers, {remaining_slots} slots remaining):\n"
    ]
    
    # Show already selected papers
    if already_selected_papers:
        for paper in already_selected_papers:
            parts.append(
                f"  - {paper.get('paper_id', 'Unknown')}: "
                f"{get_reference_title_text(paper)}\n"
            )
    elif already_selected:
        for paper_id in already_selected:
            parts.append(f"  - {paper_id}\n")
    else:
        parts.append("  (none)\n")
    
    parts.append("\n---\n")
    
    # Add available papers with abstracts and outlines
    if papers_with_abstracts:
        parts.append(f"ADDITIONAL PAPERS AVAILABLE FOR SELECTION (can add up to {remaining_slots} more):\n")
        for p in papers_with_abstracts:
            parts.append(f"\n--- Paper ID: {p.get('paper_id', 'Unknown')} ---")
            parts.append(f"\nTitle: {get_reference_title_text(p)}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            
            # NEW: Display outline
            outline = p.get('outline', '')
            if outline:
                parts.append(f"\nOutline:\n{outline}")
            else:
                parts.append("\nOutline: [Not available]")
            
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            source_ids = p.get('source_brainstorm_ids', [])
            if source_ids:
                parts.append(f"\nSource Brainstorms: {', '.join(source_ids)}")
            parts.append("\n")
        parts.append("\n---\n")
    else:
        parts.append("ADDITIONAL PAPERS AVAILABLE: None\n---\n")
    
    parts.append(f"REMINDER: You can add up to {remaining_slots} more papers. Already selected papers will be kept.")
    parts.append("\n\nReview abstracts and outlines, then decide if you need additional papers (respond as JSON):")
    
    return "".join(parts)


def build_reference_expansion_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    papers_with_abstracts: List[Dict[str, Any]],
    max_papers: int = 6
) -> str:
    """
    Build the reference expansion prompt (Step 1: abstracts only).
    LEGACY function - use build_pre_brainstorm_expansion_prompt or 
    build_additional_reference_expansion_prompt for mode-specific prompts.
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary of brainstorm content
        papers_with_abstracts: List of papers with title and abstract
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_reference_expansion_system_prompt(max_papers),
        "\n---\n",
        get_reference_expansion_json_schema(),
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"YOUR BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM SUMMARY (what you'll be writing about):\n{brainstorm_summary}",
        "\n---\n"
    ]
    
    # Add papers with abstracts and outlines
    if papers_with_abstracts:
        parts.append("EXISTING PAPERS IN LIBRARY (Titles, Abstracts, and Outlines):\n")
        for p in papers_with_abstracts:
            parts.append(f"\n--- Paper ID: {p.get('paper_id', 'Unknown')} ---")
            parts.append(f"\nTitle: {get_reference_title_text(p)}")
            parts.append(f"\nAbstract: {p.get('abstract', 'N/A')}")
            
            # NEW: Display outline
            outline = p.get('outline', '')
            if outline:
                parts.append(f"\nOutline:\n{outline}")
            else:
                parts.append("\nOutline: [Not available]")
            
            parts.append(f"\nWord Count: {p.get('word_count', 0)}")
            source_ids = p.get('source_brainstorm_ids', [])
            if source_ids:
                parts.append(f"\nSource Brainstorms: {', '.join(source_ids)}")
            parts.append("\n")
        parts.append("\n---\n")
    else:
        parts.append("EXISTING PAPERS IN LIBRARY: None\n---\n")
    
    parts.append("Review the abstracts and outlines, then decide which papers to expand (respond as JSON):")
    
    return "".join(parts)


def build_reference_selection_prompt(
    user_research_prompt: str,
    topic_prompt: str,
    brainstorm_summary: str,
    expanded_papers: List[Dict[str, Any]],
    mode: str = "initial",
    max_papers: int = 6
) -> str:
    """
    Build the final reference selection prompt (Step 2: full papers).
    
    Args:
        user_research_prompt: The user's high-level research goal
        topic_prompt: The brainstorm topic prompt
        brainstorm_summary: Summary of brainstorm content
        expanded_papers: List of papers with full content
        mode: "initial" for pre-brainstorm, "additional" for pre-paper
        max_papers: Maximum papers to select
    
    Returns:
        Complete prompt string
    """
    parts = [
        get_reference_selection_system_prompt(max_papers),
        "\n---\n",
        get_reference_selection_json_schema(max_papers),
        "\n---\n",
        f"MODE: {mode.upper()} SELECTION",
        "\n---\n",
        f"USER RESEARCH GOAL:\n{user_research_prompt}",
        "\n---\n",
        f"YOUR BRAINSTORM TOPIC:\n{topic_prompt}",
        "\n---\n",
        f"BRAINSTORM SUMMARY:\n{brainstorm_summary}",
        "\n---\n"
    ]
    
    # Add expanded papers with full content and outlines
    parts.append("EXPANDED PAPERS (Full Content):\n")
    for p in expanded_papers:
        parts.append(f"\n{'=' * 60}")
        parts.append(f"\nPaper ID: {p.get('paper_id', 'Unknown')}")
        parts.append(f"\nTitle: {get_reference_title_text(p)}")
        parts.append(f"\nWord Count: {p.get('word_count', 0)}")
        parts.append(f"\n{'=' * 60}")
        
        # NEW: Display outline first
        outline = p.get('outline', '')
        if outline:
            parts.append(f"\n\nOUTLINE:\n{outline}\n")
            parts.append(f"\n{'-' * 60}\n")
        
        parts.append(f"\n\nFULL PAPER CONTENT:\n{p.get('content', '[Content not available]')}\n")
    
    parts.append("\n---\n")
    parts.append(f"REMINDER: You can select up to {max_papers} papers maximum for this selection.")
    if mode == "initial":
        parts.append("\nThese references will inform your brainstorm exploration AND paper writing.")
    parts.append("\n\nMake your final selection (respond as JSON):")
    
    return "".join(parts)

