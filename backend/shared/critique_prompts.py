"""
Paper Critique Prompts Module.

Contains the default critique prompt and helper functions for building
critique requests to the validator model. Also provides lenient parsing
for critique responses that may be truncated by max_tokens limits.
"""
import json
import re
import logging

logger = logging.getLogger(__name__)

# Default critique prompt that can be customized by users
DEFAULT_CRITIQUE_PROMPT = """You are an expert academic reviewer providing an honest, thorough critique of a research paper.

Evaluate this paper and provide:
1. NOVELTY (1-10): How original and innovative is this work?
2. CORRECTNESS (1-10): How mathematically/logically sound is the content?
3. IMPACT ON RELATED FIELD (1-10): How significant could this contribution be?

For each category, provide the numeric rating (1-10) and detailed feedback explaining your assessment.

Be honest and constructive. Identify both strengths and weaknesses."""


# JSON schema for structured output (always appended, not customizable)
CRITIQUE_JSON_SCHEMA = """
OUTPUT FORMAT (respond ONLY with valid JSON):
{
  "novelty_rating": <integer 1-10>,
  "novelty_feedback": "<detailed feedback on novelty>",
  "correctness_rating": <integer 1-10>,
  "correctness_feedback": "<detailed feedback on correctness>",
  "impact_rating": <integer 1-10>,
  "impact_feedback": "<detailed feedback on potential impact>",
  "full_critique": "<comprehensive summary critique of the paper>"
}

IMPORTANT:
- All ratings MUST be integers from 1 to 10
- All feedback fields MUST be non-empty strings
- Respond ONLY with the JSON object, no additional text
"""


def build_critique_prompt(paper_content: str, paper_title: str = None, custom_prompt: str = None) -> str:
    """
    Build the complete critique prompt for the validator.
    
    Args:
        paper_content: The full paper content to critique
        paper_title: Optional title of the paper being critiqued
        custom_prompt: Optional custom prompt to use instead of default
        
    Returns:
        The complete prompt string to send to the validator
    """
    # Use custom prompt if provided, otherwise use default
    base_prompt = custom_prompt if custom_prompt else DEFAULT_CRITIQUE_PROMPT
    
    # Build title section if provided
    title_section = f"\nPAPER TITLE: {paper_title}\n" if paper_title else ""
    
    # Build the complete prompt
    complete_prompt = f"""{base_prompt}

{CRITIQUE_JSON_SCHEMA}

---
PAPER TO REVIEW:{title_section}
---

{paper_content}

---
END OF PAPER
---

Now provide your honest critique as JSON:"""
    
    return complete_prompt


def get_default_critique_prompt() -> str:
    """
    Get the default critique prompt text.
    
    Returns:
        The default critique prompt string (without JSON schema)
    """
    return DEFAULT_CRITIQUE_PROMPT


def parse_critique_response(response_content: str) -> dict:
    """
    Parse a critique LLM response with lenient fallback for truncated JSON.
    
    Critique responses are especially prone to truncation because reasoning models
    burn tokens on internal thinking before the JSON, and the full_critique field
    (the last and longest field) often gets cut off right before the closing '}'.
    
    Strategy:
    1. Try strict parse_json() first
    2. If truncated, try repairing by appending closing characters
    3. If still fails, extract ratings and feedback via regex
    
    Returns:
        Parsed critique dict with all expected fields
    """
    from backend.shared.json_parser import parse_json, sanitize_json_response

    # Step 1: Try strict parsing
    try:
        return parse_json(response_content)
    except Exception as strict_err:
        logger.info(f"Strict critique parse failed ({strict_err}), attempting truncation repair")

    # Step 2: Try repairing truncated JSON
    # Common case: model wrote all content but ran out of tokens before closing '}'
    try:
        sanitized = sanitize_json_response(response_content)
    except (ValueError, Exception):
        # sanitize_json_response raises ValueError on truncation - that's expected
        # Fall through to repair attempts using raw content
        sanitized = _strip_to_json(response_content)

    repaired = _try_repair_json(sanitized)
    if repaired is not None:
        logger.info("Critique JSON repaired after truncation - recovered all fields")
        return repaired

    # Step 3: Regex extraction fallback
    logger.warning("Critique JSON repair failed, falling back to regex extraction")
    return _regex_extract_critique(response_content)


def _strip_to_json(raw: str) -> str:
    """Strip thinking tokens, markdown, and prefixes to get to the JSON content."""
    content = raw.strip()

    # Strip <think>...</think>
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
    content = re.sub(r'</think\s*>', '', content, flags=re.IGNORECASE).strip()
    content = re.sub(r'<think\s*>', '', content, flags=re.IGNORECASE).strip()

    # Strip markdown code blocks
    if content.startswith('```'):
        lines = content.split('\n')
        if len(lines) > 2:
            closing = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == '```':
                    closing = i
                    break
            if closing > 0:
                content = '\n'.join(lines[1:closing]).strip()

    # Strip prefix before first '{'
    brace = content.find('{')
    if brace > 0:
        content = content[brace:]

    return content


def _try_repair_json(content: str):
    """
    Attempt to repair truncated critique JSON by appending missing closing characters.
    Only repairs simple truncation (missing '}', or string cut off mid-value).
    """
    if not content or '{' not in content:
        return None

    # Try progressively more aggressive repairs
    repairs = [
        '}',        # Missing only closing brace
        '"}',       # String value ended, missing quote + brace  
        '..."}',    # Truncated mid-word in last string value
    ]

    for suffix in repairs:
        candidate = content.rstrip() + suffix
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and result.get("novelty_rating"):
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    # More aggressive: find last complete key-value pair and close from there
    # Handles case where truncation happened mid-field-value
    last_complete = content.rfind('","')
    if last_complete > 0:
        truncated = content[:last_complete + 1] + '}'
        try:
            result = json.loads(truncated)
            if isinstance(result, dict) and result.get("novelty_rating"):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _regex_extract_critique(raw: str) -> dict:
    """
    Last-resort extraction of critique fields from raw text via regex.
    Ratings appear early in the JSON and are almost always present even in
    heavily truncated responses.
    """
    def extract_rating(field: str) -> int:
        m = re.search(rf'"{field}"\s*:\s*(\d+)', raw)
        if m:
            val = int(m.group(1))
            return val if 1 <= val <= 10 else 0
        return 0

    def extract_string(field: str) -> str:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        return m.group(1) if m else ""

    novelty = extract_rating("novelty_rating")
    correctness = extract_rating("correctness_rating")
    impact = extract_rating("impact_rating")

    result = {
        "novelty_rating": novelty,
        "novelty_feedback": extract_string("novelty_feedback") or ("Unable to parse structured response" if novelty == 0 else ""),
        "correctness_rating": correctness,
        "correctness_feedback": extract_string("correctness_feedback") or ("Unable to parse structured response" if correctness == 0 else ""),
        "impact_rating": impact,
        "impact_feedback": extract_string("impact_feedback") or ("Unable to parse structured response" if impact == 0 else ""),
        "full_critique": extract_string("full_critique") or raw,
    }

    recovered = sum(1 for k in ["novelty_rating", "correctness_rating", "impact_rating"] if result[k] > 0)
    logger.info(f"Regex extraction recovered {recovered}/3 ratings: N={novelty}, C={correctness}, I={impact}")

    return result

