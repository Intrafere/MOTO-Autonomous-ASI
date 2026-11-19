"""
Paper Critique Prompts Module.

Contains the default critique prompt and helper functions for building
critique requests to the validator model, plus the strict critique reply parser.
"""
from typing import Optional

from backend.shared.json_key_recovery import KeySpec, ObjectSchema, recover_key_labels
from backend.shared.json_parser import parse_json
from backend.shared.model_output_contract import ModelOutputContractError, OutputRepair

# Default critique prompt that can be customized by users
DEFAULT_CRITIQUE_PROMPT = """You are an expert academic reviewer providing an honest, thorough critique of a research paper.

Evaluate this paper and provide:
1. NOVELTY (1-10): How original and innovative is this work?
2. CORRECTNESS (1-10): How factually, logically, mathematically, technically, or methodologically sound is the content, as appropriate to its claims and field?
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


CRITIQUE_RATING_KEYS = ("novelty_rating", "correctness_rating", "impact_rating")
CRITIQUE_TEXT_KEYS = ("novelty_feedback", "correctness_feedback", "impact_feedback", "full_critique")

# Ordered as CRITIQUE_JSON_SCHEMA asks for them. Only the free-text fields may
# be relabeled; ratings never are.
CRITIQUE_RESPONSE_SCHEMA = ObjectSchema(
    name="paper_critique",
    keys=(
        KeySpec("novelty_rating", value_type="int"),
        KeySpec("novelty_feedback", repairable=True),
        KeySpec("correctness_rating", value_type="int"),
        KeySpec("correctness_feedback", repairable=True),
        KeySpec("impact_rating", value_type="int"),
        KeySpec("impact_feedback", repairable=True),
        KeySpec("full_critique", repairable=True),
    ),
    allow_extra=frozenset({"solution_path_update"}),
)


def parse_critique_response(
    response_content: str,
    *,
    role: str = "paper_critic",
    repairs: Optional[list] = None,
) -> dict:
    """Parse a critique reply strictly against ``CRITIQUE_JSON_SCHEMA``.

    When the caller passes ``repairs`` (and publishes it as Live Activity), the
    shared parser may apply its guarded visible normalizations and one
    mislabeled free-text field may be relabeled by schema-aware key recovery;
    without it the reply must match exactly. Nothing is ever filled in: a
    truncated, malformed, or incomplete reply raises ``ModelOutputContractError``
    so the caller retries or fails visibly instead of storing fabricated ratings
    or placeholder text.
    """
    applied: list[OutputRepair] = []
    collect = repairs is not None
    try:
        data = parse_json(response_content, repairs=applied if collect else None)
    except Exception as exc:
        raise ModelOutputContractError.syntax(exc, role=role) from exc

    if isinstance(data, list):
        if len(data) != 1 or not isinstance(data[0], dict):
            raise ModelOutputContractError(
                "The critique must be one JSON object",
                role=role,
                stage="shape",
                repairs_applied=applied,
            )
        data = data[0]
    if not isinstance(data, dict):
        raise ModelOutputContractError(
            "The critique must be one JSON object",
            role=role,
            stage="shape",
            repairs_applied=applied,
        )

    if collect:
        data, key_repairs = recover_key_labels(data, CRITIQUE_RESPONSE_SCHEMA)
        applied.extend(key_repairs)
        repairs.extend(applied)

    missing = [
        key for key in (*CRITIQUE_RATING_KEYS, *CRITIQUE_TEXT_KEYS) if key not in data
    ]
    invalid = [
        key
        for key in CRITIQUE_RATING_KEYS
        if key in data
        and not (
            isinstance(data[key], int)
            and not isinstance(data[key], bool)
            and 1 <= data[key] <= 10
        )
    ]
    invalid.extend(
        key
        for key in CRITIQUE_TEXT_KEYS
        if key in data and not (isinstance(data[key], str) and data[key].strip())
    )
    if missing or invalid:
        known = {spec.name for spec in CRITIQUE_RESPONSE_SCHEMA.keys}
        unknown = [
            key
            for key in data
            if key not in known and key not in CRITIQUE_RESPONSE_SCHEMA.allow_extra
        ]
        raise ModelOutputContractError(
            "The critique did not match the required JSON format "
            "(ratings must be integers 1-10 and feedback must be non-empty text)",
            role=role,
            stage="missing_key" if missing else "invalid_value",
            missing_keys=missing,
            unknown_keys=unknown,
            invalid_fields=invalid,
            repairs_applied=applied,
        )
    return data


# The JSON shape shown in the one bounded contract retry (the prompt's own schema block).
CRITIQUE_RESPONSE_SHAPE = CRITIQUE_JSON_SCHEMA.split("OUTPUT FORMAT (respond ONLY with valid JSON):", 1)[1].split("\n\n", 1)[0].strip()


async def generate_validated_paper_critique(
    *,
    prompt: str,
    task_id: str,
    model: str,
    context_window: int,
    max_output_tokens: int,
    role_id: str = "paper_critic",
) -> dict:
    """Ask the configured critic role for a critique and parse it strictly.

    A reply that does not match the critique contract gets exactly one bounded
    retry that names the problem; a second failure raises the typed
    ``ModelOutputContractError`` (chained to the first). Provider failures
    propagate unchanged. Applied normalizations are published as
    ``model_output_normalized`` activity.
    """
    from backend.autonomous.agents.output_contract import (
        build_contract_retry_messages,
        response_text,
    )
    from backend.shared.api_client_manager import api_client_manager
    from backend.shared.config import rag_config
    from backend.shared.model_output_contract import report_model_output_normalizations

    response = await api_client_manager.generate_completion(
        task_id=task_id,
        role_id=role_id,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_output_tokens,
        temperature=0.0,
    )
    content = response_text(response)
    report_task_id = task_id
    repairs: list = []
    try:
        critique = parse_critique_response(content, role=role_id, repairs=repairs)
    except ModelOutputContractError as first_error:
        report_task_id = f"{task_id}_retry"
        response = await api_client_manager.generate_completion(
            task_id=report_task_id,
            role_id=role_id,
            model=model,
            messages=build_contract_retry_messages(
                prompt=prompt,
                failed_output=content,
                error=first_error,
                response_noun="critique",
                response_shape=CRITIQUE_RESPONSE_SHAPE,
                max_input_tokens=rag_config.get_available_input_tokens(
                    context_window, max_output_tokens
                ),
            ),
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        repairs = []
        try:
            critique = parse_critique_response(
                response_text(response), role=role_id, repairs=repairs
            )
        except ModelOutputContractError as retry_error:
            raise retry_error from first_error
    await report_model_output_normalizations(
        role_id, repairs, task_id=report_task_id, model=model
    )
    return critique


def critique_contract_failure_detail(error: ModelOutputContractError) -> str:
    """User-facing detail for a critique that still broke its contract after the retry."""
    return (
        "The validator model's critique did not match the required JSON format "
        f"after one retry ({error.field_summary or error.stage}). No critique was saved."
    )
