"""
Pydantic models for the ASI Aggregator System.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONTEXT_WINDOW = 0
DEFAULT_MAX_OUTPUT_TOKENS = 0
DEFAULT_OPENROUTER_REASONING_EFFORT = "auto"
OpenRouterReasoningEffort = Literal["auto", "xhigh", "high", "medium", "low", "minimal", "none"]
ModelProvider = Literal["lm_studio", "openrouter", "openai_codex_oauth", "xai_grok_oauth", "sakana_fugu"]
_LEGACY_WRITER_PREFIX = "high" + "_context"


def _legacy_writer_field(name: str) -> str:
    return f"{_LEGACY_WRITER_PREFIX}_{name}"


class DocumentChunk(BaseModel):
    """Individual text chunk with embeddings."""
    chunk_id: str
    text: str
    source_file: str
    position: int
    chunk_size: int
    chunk_type: Literal["text", "table", "code", "equation", "section"] = "text"
    embedding: Optional[List[float]] = None
    tokens: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_user_file: bool = False
    is_permanent: bool = False  # User files are never evicted


class ContextPack(BaseModel):
    """Main retrieval payload for submitters and validators."""
    text: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    source_map: Dict[str, str] = Field(default_factory=dict)
    coverage: float = 0.0
    answerability: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    needs_more_context: bool = False


class Submission(BaseModel):
    """Submission from a submitter agent."""
    submission_id: str
    submitter_id: int
    content: str
    reasoning: str
    chunk_size_used: int
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_decline: bool = False  # True when critique_needed=false (critique phase only)


class ValidationResult(BaseModel):
    """Result of validation by validator agent."""
    submission_id: str
    decision: Literal["accept", "reject"]
    reasoning: str
    summary: str = ""  # Max 750 chars for rejection logs
    timestamp: datetime = Field(default_factory=datetime.now)
    contradiction_check_passed: bool = True
    json_valid: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CleanupReviewResult(BaseModel):
    """Result of cleanup review by validator."""
    should_remove: bool
    submission_number: Optional[int] = None
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RemovalValidationResult(BaseModel):
    """Result of removal validation by validator."""
    decision: Literal["accept", "reject"]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SubmitterState(BaseModel):
    """State tracking for a submitter agent."""
    submitter_id: int
    current_chunk_size_index: int = 0
    consecutive_rejections: int = 0
    total_submissions: int = 0
    total_acceptances: int = 0
    is_active: bool = True


class SystemStatus(BaseModel):
    """Overall system status."""
    is_running: bool = False
    queue_size: int = 0
    total_submissions: int = 0
    total_acceptances: int = 0
    total_rejections: int = 0
    acceptance_rate: float = 0.0
    submitter_states: List[SubmitterState] = Field(default_factory=list)
    shared_training_size: int = 0
    # Cleanup review stats
    cleanup_reviews_performed: int = 0
    removals_proposed: int = 0
    removals_executed: int = 0
    fatal_error_type: Optional[str] = None
    fatal_error_message: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for a model (can be LM Studio or OpenRouter)."""
    provider: ModelProvider = "lm_studio"
    model_id: str
    openrouter_model_id: Optional[str] = None  # For OpenRouter (different naming)
    openrouter_provider: Optional[str] = None  # Specific OpenRouter provider (e.g., "Anthropic")
    openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    lm_studio_fallback_id: Optional[str] = None  # Fallback LM Studio model if OpenRouter fails
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    supercharge_enabled: bool = False


class BoostConfig(BaseModel):
    """API boost configuration."""
    enabled: bool = False
    openrouter_api_key: str = ""
    boost_model_id: str = ""  # OpenRouter model to use for boost
    boost_provider: Optional[str] = None  # Specific provider, or None to let OpenRouter choose
    boost_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    boost_context_window: int = DEFAULT_CONTEXT_WINDOW
    boost_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


class FreeModelSettings(BaseModel):
    """Settings for free model cooldown handling and rotation."""
    looping_enabled: bool = False
    auto_selector_enabled: bool = False


class WorkflowTask(BaseModel):
    """Represents a predicted API call in the workflow."""
    task_id: str  # Unique ID like "agg_sub1_001"
    sequence_number: int  # 1-20
    role: str  # "Submitter 1", "Validator", "Writing Submitter", etc.
    mode: Optional[str] = None  # "Construction", "Rigor", "Review", etc.
    provider: str = "lm_studio"  # "openrouter" | "lm_studio"
    using_boost: bool = False
    completed: bool = False
    active: bool = False  # Currently executing


class SubmitterConfig(BaseModel):
    """Configuration for a single aggregator submitter agent."""
    submitter_id: int
    provider: ModelProvider = "lm_studio"
    model_id: str  # LM Studio model OR OpenRouter model based on provider
    openrouter_provider: Optional[str] = None  # Specific OpenRouter provider (e.g., "Anthropic")
    openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    lm_studio_fallback_id: Optional[str] = None  # Fallback LM Studio model if OpenRouter fails
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    supercharge_enabled: bool = False


class AggregatorStartRequest(BaseModel):
    """Request to start the aggregator."""
    user_prompt: str
    submitter_configs: List[SubmitterConfig]  # Per-submitter configs (1-10)
    creativity_emphasis_boost_enabled: bool = False
    # Validator config
    validator_provider: ModelProvider = "lm_studio"
    validator_model: str  # LM Studio model OR OpenRouter model based on provider
    validator_openrouter_provider: Optional[str] = None  # Specific OpenRouter provider
    validator_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    validator_lm_studio_fallback: Optional[str] = None  # Fallback if OpenRouter fails
    validator_context_size: int = DEFAULT_CONTEXT_WINDOW
    validator_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    validator_supercharge_enabled: bool = False
    # Parallel Assistant proof-retrieval role (defaults hydrate from Validator in routes/UI)
    assistant_provider: ModelProvider = "lm_studio"
    assistant_model: str = ""
    assistant_openrouter_provider: Optional[str] = None
    assistant_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    assistant_lm_studio_fallback: Optional[str] = None
    assistant_context_size: int = DEFAULT_CONTEXT_WINDOW
    assistant_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    assistant_supercharge_enabled: bool = False
    uploaded_files: List[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Information about an available LM Studio model."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lm-studio"


# ============================================================================
# COMPILER MODELS (Phase 2)
# ============================================================================


class CompilerSubmission(BaseModel):
    """Submission from a compiler submitter agent.
    
    Uses exact string matching for edit operations:
    - content: The full text content being submitted (for display/logging/full replacements)
    - operation: Type of edit (replace, insert_after, delete, or full_content)
    - old_string: Exact text to find (anchor for insert_after, target for replace/delete)
    - new_string: Replacement/insertion text (empty string for delete)
    
    For outline_create mode, uses full_content operation where content is the complete outline.
    For other modes, content stores the submission for logging while old_string/new_string specify the edit.
    
    Retroactive brainstorm operations (optional, autonomous mode only):
    - brainstorm_operation: Optional operation on the source brainstorm database.
      Validated independently from paper operations. Each must stand on its own merits.
    """
    submission_id: str
    mode: Literal["outline_create", "outline_update", "construction", "review", "rigor"]
    content: str  # Full submission content for display/logging/validation
    
    # Exact string matching fields for specifying edit location
    operation: Literal["replace", "insert_after", "delete", "full_content"] = "replace"
    old_string: str = ""  # Exact text to find (empty for full_content operation)
    new_string: str = ""  # New/replacement text (empty for delete)
    
    reasoning: str
    section_complete: bool = False  # Explicit signal that current phase is complete
    outline_complete: Optional[bool] = None  # For outline_create mode: True = lock outline, False = refine further
    needs_construction: Optional[bool] = None  # For construction mode: False = no more content needed
    needs_edit: Optional[bool] = None  # For review mode: False = no edit needed
    needs_enhancement: Optional[bool] = None  # For rigor mode: False = no enhancement needed
    needs_update: Optional[bool] = None  # For outline_update mode: False = no update needed
    
    # Retroactive brainstorm correction (optional, autonomous paper writing only)
    brainstorm_operation: Optional["BrainstormRetroactiveOperation"] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BrainstormRetroactiveOperation(BaseModel):
    """Optional retroactive operation on the source brainstorm database.
    
    Proposed by the compiler submitter during paper writing and validated
    independently from the paper operation. The validator sees ONLY the
    brainstorm context when validating this, never the paper operation.
    Each operation must be independently justified.
    """
    action: Literal["edit", "delete", "add"]
    submission_number: Optional[int] = None  # Required for edit/delete, None for add
    new_content: str = ""  # Required for edit/add, empty for delete
    reasoning: str  # Independent justification (must not depend on paper operation)


CompilerSubmission.model_rebuild()


class CompilerValidationResult(BaseModel):
    """Result of validation by compiler validator."""
    submission_id: str
    decision: Literal["accept", "reject"]
    reasoning: str
    summary: str = ""  # For rejection log (max 750 chars)
    timestamp: datetime = Field(default_factory=datetime.now)
    coherence_check: bool = True
    rigor_check: bool = True
    placement_check: bool = True
    json_valid: bool = True
    validation_stage: str = "llm_validation"  # "pre-validation" | "llm_validation" | "internal_error"


class CompilerState(BaseModel):
    """Compiler system state."""
    is_running: bool = False
    current_mode: str = "idle"
    outline_accepted: bool = False
    paper_word_count: int = 0
    total_submissions: int = 0
    construction_acceptances: int = 0
    construction_rejections: int = 0
    construction_declines: int = 0
    rigor_acceptances: int = 0
    rigor_rejections: int = 0
    rigor_declines: int = 0
    outline_acceptances: int = 0
    outline_rejections: int = 0
    outline_declines: int = 0
    review_acceptances: int = 0
    review_rejections: int = 0
    review_declines: int = 0
    minuscule_edit_count: int = 0
    in_critique_phase: bool = False
    critique_acceptances: int = 0
    paper_version: int = 1


class CompilerStartRequest(BaseModel):
    """Request to start the compiler."""
    compiler_prompt: str
    allow_mathematical_proofs: bool = True
    allow_research_papers: bool = True
    # Validator config
    validator_provider: ModelProvider = "lm_studio"
    validator_model: str
    validator_openrouter_provider: Optional[str] = None
    validator_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    validator_lm_studio_fallback: Optional[str] = None
    validator_context_size: int = DEFAULT_CONTEXT_WINDOW
    validator_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    validator_supercharge_enabled: bool = False
    # Writing submitter config
    writer_provider: ModelProvider = Field(
        default="lm_studio",
        validation_alias=AliasChoices("writer_provider", _legacy_writer_field("provider")),
    )
    writer_model: str = Field(validation_alias=AliasChoices("writer_model", _legacy_writer_field("model")))
    writer_openrouter_provider: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("writer_openrouter_provider", _legacy_writer_field("openrouter_provider")),
    )
    writer_openrouter_reasoning_effort: OpenRouterReasoningEffort = Field(
        default=DEFAULT_OPENROUTER_REASONING_EFFORT,
        validation_alias=AliasChoices(
            "writer_openrouter_reasoning_effort",
            _legacy_writer_field("openrouter_reasoning_effort"),
        ),
    )
    writer_lm_studio_fallback: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("writer_lm_studio_fallback", _legacy_writer_field("lm_studio_fallback")),
    )
    writer_context_size: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        validation_alias=AliasChoices("writer_context_size", _legacy_writer_field("context_size")),
    )
    writer_max_output_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        validation_alias=AliasChoices("writer_max_output_tokens", _legacy_writer_field("max_output_tokens")),
    )
    writer_supercharge_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("writer_supercharge_enabled", _legacy_writer_field("supercharge_enabled")),
    )
    # Rigor & Proofs submitter config (legacy field prefix: high_param_*)
    high_param_provider: ModelProvider = "lm_studio"
    high_param_model: str
    high_param_openrouter_provider: Optional[str] = None
    high_param_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    high_param_lm_studio_fallback: Optional[str] = None
    high_param_context_size: int = DEFAULT_CONTEXT_WINDOW
    high_param_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    high_param_supercharge_enabled: bool = False
    # Deprecated compatibility aliases. Critique generation now uses the
    # Rigor & Proofs submitter config; routes may mirror high_param_* here for
    # older clients that still send/read critique_submitter_* fields.
    critique_submitter_provider: ModelProvider = "lm_studio"
    critique_submitter_model: str = ""
    critique_submitter_openrouter_provider: Optional[str] = None
    critique_submitter_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    critique_submitter_lm_studio_fallback: Optional[str] = None
    critique_submitter_context_window: int = DEFAULT_CONTEXT_WINDOW
    critique_submitter_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    critique_submitter_supercharge_enabled: bool = False
    # Parallel Assistant proof-retrieval role (defaults hydrate from Validator in routes/UI)
    assistant_provider: ModelProvider = "lm_studio"
    assistant_model: str = ""
    assistant_openrouter_provider: Optional[str] = None
    assistant_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    assistant_lm_studio_fallback: Optional[str] = None
    assistant_context_size: int = DEFAULT_CONTEXT_WINDOW
    assistant_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    assistant_supercharge_enabled: bool = False


# ============================================================================
# AUTONOMOUS RESEARCH MODELS (Part 3)
# ============================================================================


class BrainstormMetadata(BaseModel):
    """Metadata for a brainstorm topic."""
    topic_id: str
    topic_prompt: str
    status: Literal["in_progress", "complete"] = "in_progress"
    # Current post-cleanup database size.
    submission_count: int = 0
    # Monotonic accepted-event count used for resume offsets and hard caps.
    total_acceptances: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    last_activity: datetime = Field(default_factory=datetime.now)
    papers_generated: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _restore_legacy_total_acceptances(cls, data: Any) -> Any:
        """Legacy metadata only stored the retained submission count."""
        if isinstance(data, dict) and "total_acceptances" not in data:
            data = dict(data)
            data["total_acceptances"] = max(0, int(data.get("submission_count") or 0))
        return data


class PaperMetadata(BaseModel):
    """Metadata for a completed or in-progress paper."""
    paper_id: str
    title: str
    abstract: str = ""
    word_count: int = 0
    source_brainstorm_ids: List[str] = Field(default_factory=list)
    referenced_papers: List[str] = Field(default_factory=list)
    status: Literal["in_progress", "complete", "archived", "pruned"] = "complete"
    created_at: datetime = Field(default_factory=datetime.now)
    # Per-paper model tracking: model_id -> API call count
    model_usage: Optional[Dict[str, int]] = None
    # Generation date for the paper (separate from created_at for tracking purposes)
    generation_date: Optional[datetime] = None
    # Wolfram Alpha verification count (tracked separately from LLM API calls)
    wolfram_calls: Optional[int] = None
    # Pruned papers are preserved for users but excluded from all model context.
    pruned_at: Optional[datetime] = None
    pruned_reason: Optional[str] = None
    pruned_by: Optional[Literal["system", "user", "legacy"]] = None


class TopicSelectionSubmission(BaseModel):
    """Submission from topic selection agent."""
    action: Literal["new_topic", "continue_existing"]
    topic_id: Optional[str] = None  # Required if action is continue_existing
    topic_prompt: str = ""  # Required if action is new_topic
    reasoning: str


class TopicValidationResult(BaseModel):
    """Result of topic validation."""
    decision: Literal["accept", "reject"]
    reasoning: str
    summary: str = ""  # Rejection feedback (max 750 chars)
    timestamp: datetime = Field(default_factory=datetime.now)


class BrainstormContinuationDecision(BaseModel):
    """Decision on whether to write another paper from the same brainstorm or move on."""
    decision: Literal["write_another_paper", "move_on"]
    reasoning: str


class CompletionReviewResult(BaseModel):
    """Result of brainstorm completion review."""
    decision: Literal["continue_brainstorm", "write_paper"]
    reasoning: str
    suggested_additions: str = ""  # Optional suggestions if continue_brainstorm
    timestamp: datetime = Field(default_factory=datetime.now)


class CompletionSelfValidationResult(BaseModel):
    """Result of self-validation for completion review."""
    validated: bool
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ReferenceExpansionRequest(BaseModel):
    """Request to expand paper abstracts to full content."""
    expand_papers: List[str] = Field(default_factory=list)  # Paper IDs to expand
    proceed_without_references: bool = False
    reasoning: str


class ReferenceSelectionResult(BaseModel):
    """Final selection of reference papers."""
    selected_papers: List[str] = Field(default_factory=list)  # Caller-specific cap
    reasoning: str


class PaperTitleSelection(BaseModel):
    """Selection of paper title."""
    paper_title: str
    reasoning: str


class PaperRedundancyReviewResult(BaseModel):
    """Result of paper redundancy review."""
    should_remove: bool
    paper_id: Optional[str] = None  # Paper to remove, if any
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class AutonomousResearchState(BaseModel):
    """Current state of autonomous research mode."""
    is_running: bool = False
    current_tier: Literal["idle", "tier1_aggregation", "tier2_paper_writing", "tier3_final_answer"] = "idle"
    current_brainstorm: Optional[BrainstormMetadata] = None
    current_paper: Optional[Dict[str, Any]] = None  # Paper being written
    current_phase: Optional[Literal["body_sections", "conclusion", "introduction", "abstract"]] = None
    
    # Tier 3 Final Answer state
    final_answer_state: Optional[Dict[str, Any]] = None  # FinalAnswerState as dict
    
    # Statistics
    total_brainstorms_created: int = 0
    total_brainstorms_completed: int = 0
    total_papers_completed: int = 0
    total_papers_archived: int = 0
    total_papers_pruned: int = 0
    total_submissions_accepted: int = 0
    total_submissions_rejected: int = 0
    topic_selection_rejections: int = 0
    completion_reviews_run: int = 0
    paper_redundancy_reviews_run: int = 0
    tier3_triggers: int = 0  # Number of times Tier 3 has been triggered


class AutonomousTerminalEvent(BaseModel):
    """Durable terminal lifecycle event used to recover missed live updates."""
    terminal_event_id: str
    run_id: str
    lifecycle_generation: int = 0
    event_type: str = "auto_research_stopped"
    reason: str
    message: str
    occurred_at: datetime = Field(default_factory=datetime.now)
    recoverable: bool = True
    fatal: bool = False
    workflow_mode: Literal["autonomous"] = "autonomous"
    notification_kind: Optional[str] = None
    role_id: Optional[str] = None
    current_tier: Optional[str] = None
    current_topic_id: Optional[str] = None
    current_paper_id: Optional[str] = None
    configured_model: Optional[str] = None
    configured_provider: Optional[str] = None
    effective_model: Optional[str] = None
    effective_provider: Optional[str] = None
    effective_host_provider: Optional[str] = None
    route_kind: Optional[str] = None
    resolution: Optional[str] = None
    terminal_guidance: Optional[str] = None
    error_detail: Optional[str] = None
    required_tokens: Optional[int] = None
    available_tokens: Optional[int] = None
    context_window: Optional[int] = None
    output_reserve: Optional[int] = None


class AutonomousResearchStatusResponse(BaseModel):
    """Authoritative lifecycle and progress snapshot for Autonomous Research."""
    is_running: bool
    run_id: str
    lifecycle_generation: int = 0
    resume_available: bool = False
    current_tier: str
    current_brainstorm: Optional[Dict[str, Any]] = None
    current_paper: Optional[Dict[str, Any]] = None
    tier3_status: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    terminal_event: Optional[AutonomousTerminalEvent] = None


class AutonomousResearchStartRequest(BaseModel):
    """Request to start autonomous research mode."""
    user_research_prompt: str
    submitter_configs: List[SubmitterConfig]  # Per-submitter configs for brainstorm aggregation (1-10)
    creativity_emphasis_boost_enabled: bool = False
    allow_mathematical_proofs: bool = True
    allow_research_papers: bool = True
    # Validator config
    validator_provider: ModelProvider = "lm_studio"
    validator_model: str
    validator_openrouter_provider: Optional[str] = None
    validator_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    validator_lm_studio_fallback: Optional[str] = None
    validator_context_window: int = DEFAULT_CONTEXT_WINDOW
    validator_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    validator_supercharge_enabled: bool = False
    # Compiler writer settings (separate from aggregator submitters)
    writer_provider: ModelProvider = Field(
        default="lm_studio",
        validation_alias=AliasChoices("writer_provider", _legacy_writer_field("provider")),
    )
    writer_model: str = Field(
        default="",
        validation_alias=AliasChoices("writer_model", _legacy_writer_field("model")),
    )  # Empty string allowed, will use submitter model as fallback
    writer_openrouter_provider: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("writer_openrouter_provider", _legacy_writer_field("openrouter_provider")),
    )
    writer_openrouter_reasoning_effort: OpenRouterReasoningEffort = Field(
        default=DEFAULT_OPENROUTER_REASONING_EFFORT,
        validation_alias=AliasChoices(
            "writer_openrouter_reasoning_effort",
            _legacy_writer_field("openrouter_reasoning_effort"),
        ),
    )
    writer_lm_studio_fallback: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("writer_lm_studio_fallback", _legacy_writer_field("lm_studio_fallback")),
    )
    writer_context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        validation_alias=AliasChoices("writer_context_window", _legacy_writer_field("context_window")),
    )
    writer_max_tokens: int = Field(
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        validation_alias=AliasChoices("writer_max_tokens", _legacy_writer_field("max_tokens")),
    )
    writer_supercharge_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("writer_supercharge_enabled", _legacy_writer_field("supercharge_enabled")),
    )
    # Compiler Rigor & Proofs settings (legacy field prefix: high_param_*)
    high_param_provider: ModelProvider = "lm_studio"
    high_param_model: str = ""  # Empty string allowed, will use submitter model as fallback
    high_param_openrouter_provider: Optional[str] = None
    high_param_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    high_param_lm_studio_fallback: Optional[str] = None
    high_param_context_window: int = DEFAULT_CONTEXT_WINDOW
    high_param_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    high_param_supercharge_enabled: bool = False
    # Deprecated compatibility aliases. Critique generation now uses the
    # Rigor & Proofs submitter config; routes may mirror high_param_* here for
    # older clients that still send/read critique_submitter_* fields.
    critique_submitter_provider: ModelProvider = "lm_studio"
    critique_submitter_model: str = ""
    critique_submitter_openrouter_provider: Optional[str] = None
    critique_submitter_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    critique_submitter_lm_studio_fallback: Optional[str] = None
    critique_submitter_context_window: int = DEFAULT_CONTEXT_WINDOW
    critique_submitter_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    critique_submitter_supercharge_enabled: bool = False
    # Parallel Assistant proof-retrieval role (defaults hydrate from Validator in routes/UI)
    assistant_provider: ModelProvider = "lm_studio"
    assistant_model: str = ""
    assistant_openrouter_provider: Optional[str] = None
    assistant_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    assistant_lm_studio_fallback: Optional[str] = None
    assistant_context_window: int = DEFAULT_CONTEXT_WINDOW
    assistant_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    assistant_supercharge_enabled: bool = False
    # Tier 3 Final Answer settings
    tier3_enabled: bool = False  # Default OFF — system stops at Tier 2 paper library


# ============================================================================
# LEAN 4 PROOF INTEGRATION MODELS
# ============================================================================


class MathlibLemmaHint(BaseModel):
    """A locally confirmed Mathlib declaration that may help a proof attempt."""
    requested_name: str
    full_name: str = ""
    declaration: str = ""
    file_path: str = ""
    line_number: int = 0


class SmtHint(BaseModel):
    """Optional SMT-derived guidance that can seed Lean proof attempts."""
    result: Literal["sat", "unsat", "unknown"] = "unknown"
    suggested_tactics: List[str] = Field(default_factory=list)
    smtlib: str = ""
    z3_output: str = ""


class ProofCandidate(BaseModel):
    """A theorem candidate extracted from a brainstorm or paper."""
    theorem_id: str
    statement: str
    formal_sketch: str = ""
    expected_novelty_tier: str = ""
    prompt_relevance_rationale: str = ""
    novelty_rationale: str = ""
    why_not_standard_known_result: str = ""
    source_excerpt: str = ""
    origin_source_id: str = ""
    relevant_lemmas: List[MathlibLemmaHint] = Field(default_factory=list)
    smt_hint: Optional[SmtHint] = None


class ProofCandidateNoveltyDecision(BaseModel):
    """Independent pre-Lean novelty decision for one proposed candidate."""
    model_config = ConfigDict(extra="forbid")

    theorem_id: str = Field(min_length=1, max_length=256)
    decision: Literal["approve_novel", "reject_not_novel"]
    reasoning: str = Field(min_length=1, max_length=2000)


class ProofCandidateListValidation(BaseModel):
    """Exact whole-list response contract for the pre-Lean Validator."""
    model_config = ConfigDict(extra="forbid")

    results: List[ProofCandidateNoveltyDecision] = Field(min_length=1)
    feedback: str = Field(min_length=1, max_length=3000)


class ProofCandidateListRejection(BaseModel):
    """Bounded semantic rejection retained only for one proof round."""
    model_config = ConfigDict(extra="forbid")

    list_fingerprint: str = Field(min_length=1, max_length=256)
    generation_attempt: int = Field(ge=1)
    proposed_count: int = Field(ge=1)
    approved_count: int = Field(ge=0)
    rejected_candidate_ids: List[str] = Field(default_factory=list)
    feedback: str = Field(min_length=1, max_length=3000)


class FailedProofCandidate(BaseModel):
    """Persisted failed theorem candidate that can be retried later."""
    source_brainstorm_id: str
    theorem_id: str
    theorem_statement: str
    formal_sketch: str = ""
    expected_novelty_tier: str = ""
    prompt_relevance_rationale: str = ""
    novelty_rationale: str = ""
    why_not_standard_known_result: str = ""
    source_excerpt: str = ""
    error_summary: str = ""
    suggested_lemma_targets: List[str] = Field(default_factory=list)
    retry_count: int = 0
    last_retry_source_id: str = ""
    resolved_proof_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ProofRoleConfigSnapshot(BaseModel):
    """Persisted model/runtime config for proof-related agents."""
    provider: ModelProvider = "lm_studio"
    model_id: str = ""
    openrouter_provider: Optional[str] = None
    openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    lm_studio_fallback_id: Optional[str] = None
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    supercharge_enabled: bool = False


class ProofRuntimeConfigSnapshot(BaseModel):
    """Persisted proof runtime config used for manual proof checks.

    The source role slots are proof-submitter settings. After the Rigor &
    Proofs consolidation, both brainstorm and paper slots should carry the
    configured Rigor & Proofs submitter for proof-solving callers.
    """
    brainstorm: ProofRoleConfigSnapshot
    paper: ProofRoleConfigSnapshot
    validator: ProofRoleConfigSnapshot
    assistant: ProofRoleConfigSnapshot = Field(default_factory=ProofRoleConfigSnapshot)


class ProofDependency(BaseModel):
    """One dependency edge for a verified proof."""
    kind: Literal["mathlib", "moto"]
    name: str
    source_ref: str = ""


ProofDependencyExtractionStatus = Literal[
    "not_attempted",
    "complete",
    "partial",
    "failed",
]


class ProofPruneContextPressure(BaseModel):
    """Safe metadata describing why a proof-context review was requested."""
    model_config = ConfigDict(extra="forbid")

    trigger: Literal["scheduled", "context_pressure", "manual", "test"] = "scheduled"
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    available_input_tokens: Optional[int] = Field(default=None, ge=0)
    active_proof_tokens: Optional[int] = Field(default=None, ge=0)
    mandatory_source_tokens: int = Field(default=0, ge=0)
    candidate_and_feedback_tokens: int = Field(default=0, ge=0)
    active_proof_context_tokens: int = Field(default=0, ge=0)
    output_reserve_tokens: int = Field(default=0, ge=0)
    configured_context_window: int = Field(default=0, ge=0)
    route_config_fingerprint: str = Field(default="", max_length=256)
    proof_set_revision: int = Field(default=0, ge=0)
    measured_at: datetime = Field(default_factory=datetime.now)
    detail: str = Field(default="", max_length=1000)

    def pressure_fingerprint(self) -> str:
        """Return a stable identity for coalescing unchanged pressure."""
        import hashlib
        import json

        payload = self.model_dump(mode="json", exclude={"measured_at", "detail"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class ProofPruneAggregateEntry(BaseModel):
    """Compact deterministic accounting for one active proof occurrence."""
    model_config = ConfigDict(extra="forbid")

    proof_id: str = Field(min_length=1, max_length=256)
    theorem_name: str = Field(default="", max_length=512)
    canonical_theorem_hash: str = Field(min_length=1, max_length=256)
    canonical_lean_hash: str = Field(min_length=1, max_length=256)
    novelty_tier: str = Field(default="", max_length=128)
    independent_novelty_tier: str = Field(default="", max_length=128)
    source_type: str = Field(default="", max_length=128)
    source_id: str = Field(default="", max_length=512)
    created_at: datetime
    dependency_extraction_status: ProofDependencyExtractionStatus = "not_attempted"
    dependency_count: int = Field(default=0, ge=0)
    dependent_count: int = Field(default=0, ge=0)
    dependency_fingerprint: str = Field(default="", max_length=256)
    descriptor_fingerprint: str = Field(default="", max_length=256)
    estimated_context_tokens: int = Field(default=0, ge=0)
    protected_reasons: List[str] = Field(default_factory=list)
    eligible_candidate: bool = False


class ProofPruneProofDescriptor(BaseModel):
    """Bounded hydrated evidence for one candidate or comparator."""
    model_config = ConfigDict(extra="forbid")

    proof_id: str = Field(min_length=1, max_length=256)
    theorem_name: str = Field(default="", max_length=512)
    theorem_statement: str = Field(min_length=1)
    canonical_theorem_hash: str = Field(min_length=1, max_length=256)
    canonical_lean_hash: str = Field(min_length=1, max_length=256)
    novelty_tier: str = Field(default="", max_length=128)
    novelty_reasoning: str = Field(default="", max_length=4000)
    independent_novelty_tier: str = Field(default="", max_length=128)
    independent_novelty_reasoning: str = Field(default="", max_length=4000)
    source_type: str = Field(default="", max_length=128)
    source_id: str = Field(default="", max_length=512)
    source_title: str = Field(default="", max_length=1000)
    created_at: datetime
    dependencies: List[ProofDependency] = Field(default_factory=list)
    dependency_extraction_status: ProofDependencyExtractionStatus = "not_attempted"
    dependency_fingerprint: str = Field(default="", max_length=256)
    descriptor_fingerprint: str = Field(default="", max_length=256)
    protected_reasons: List[str] = Field(default_factory=list)
    role_in_user_objective: str = Field(default="", max_length=2000)
    lean_code: str = ""
    lean_code_included: bool = False


class ProofPruneCoverageClaim(BaseModel):
    """One material contribution and the retained proofs that preserve it."""
    model_config = ConfigDict(extra="forbid")

    target_contribution: str = Field(min_length=1, max_length=2000)
    preserved_by_proof_ids: List[str] = Field(min_length=1, max_length=3)
    explanation: str = Field(min_length=1, max_length=2000)


class ProofPruneProposal(BaseModel):
    """Exact primary response contract for the pruning proposer."""
    model_config = ConfigDict(extra="forbid")

    action: Literal["no_prune", "propose_prune"]
    proof_id: Optional[str] = Field(default=None, max_length=256)
    expected_theorem_hash: Optional[str] = Field(default=None, max_length=256)
    expected_lean_hash: Optional[str] = Field(default=None, max_length=256)
    prune_category: Optional[
        Literal[
            "outdated",
            "contextually_misaligned",
            "redundant",
            "superseded",
            "low_unique_utility",
        ]
    ] = None
    supporting_proof_ids: List[str] = Field(default_factory=list, max_length=3)
    coverage_claims: List[ProofPruneCoverageClaim] = Field(
        default_factory=list,
        max_length=6,
    )
    reasoning: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_target_fields(self):
        target_fields = (
            self.proof_id,
            self.expected_theorem_hash,
            self.expected_lean_hash,
            self.prune_category,
        )
        if self.action == "no_prune":
            if (
                any(value is not None for value in target_fields)
                or self.supporting_proof_ids
                or self.coverage_claims
            ):
                raise ValueError(
                    "no_prune requires null target fields and empty semantic evidence"
                )
            return self
        if any(not str(value or "").strip() for value in target_fields):
            raise ValueError(
                "propose_prune requires proof_id, category, and both expected hashes"
            )
        if not self.supporting_proof_ids or not self.coverage_claims:
            raise ValueError(
                "propose_prune requires retained supporting proofs and coverage claims"
            )
        if self.proof_id in self.supporting_proof_ids:
            raise ValueError("A pruning target cannot support its own removal")
        if len(set(self.supporting_proof_ids)) != len(self.supporting_proof_ids):
            raise ValueError("supporting_proof_ids must be unique")
        allowed_supports = set(self.supporting_proof_ids)
        if any(
            not set(claim.preserved_by_proof_ids).issubset(allowed_supports)
            for claim in self.coverage_claims
        ):
            raise ValueError("Coverage claims may cite only declared supporting proofs")
        return self


class ProofPruneValidation(BaseModel):
    """Exact primary response contract for the independent Validator."""
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    proof_id: str = Field(min_length=1, max_length=256)
    supporting_proof_ids: List[str] = Field(default_factory=list, max_length=3)
    coverage_confirmed: bool = False
    reasoning: str = Field(min_length=1, max_length=4000)


class ProofPruneSnapshot(BaseModel):
    """Immutable whole-set evidence supplied to both pruning roles."""
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1, max_length=256)
    proof_set_revision: int = Field(ge=0)
    proof_store_id: str = Field(min_length=1, max_length=512)
    owning_run_id: str = Field(min_length=1, max_length=512)
    proof_run_id: str = Field(min_length=1, max_length=512)
    proof_run_lifecycle_generation: int = Field(ge=1)
    owning_lifecycle_generation: int = Field(default=1, ge=1)
    scope: Literal["autonomous", "manual"]
    source_type: Literal["brainstorm", "paper"]
    source_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(default="", max_length=512)
    canonical_user_prompt: str = Field(min_length=1)
    trigger_reasons: List[str] = Field(default_factory=list)
    accepted_prompt_novel_total: int = Field(default=0, ge=0)
    context_pressure: ProofPruneContextPressure = Field(
        default_factory=ProofPruneContextPressure
    )
    whole_set: List[ProofPruneAggregateEntry] = Field(default_factory=list)
    candidate_descriptors: List[ProofPruneProofDescriptor] = Field(
        default_factory=list
    )
    evidence_bounded: bool = False
    evidence_policy_version: str = Field(
        default="proof-pruning-semantic-evidence-v1",
        max_length=128,
    )
    evidence_fingerprint: str = Field(default="", max_length=256)
    captured_at: datetime = Field(default_factory=datetime.now)


class ProofPruneGuardResult(BaseModel):
    """Deterministic result produced before semantic validation."""
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    proof_id: Optional[str] = Field(default=None, max_length=256)
    reasons: List[str] = Field(default_factory=list)


class ProofPruneCommitIntent(BaseModel):
    """Validated, non-mutating input for Build 04's commit service."""
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    proof_id: str
    owning_run_id: str
    proof_set_revision: int = Field(ge=0)
    expected_theorem_hash: str
    expected_lean_hash: str
    prune_category: Literal[
        "outdated",
        "contextually_misaligned",
        "redundant",
        "superseded",
        "low_unique_utility",
    ]
    supporting_proof_ids: List[str] = Field(min_length=1, max_length=3)
    supporting_proof_fingerprints: Dict[str, str] = Field(default_factory=dict)
    target_dependency_fingerprint: str
    target_descriptor_fingerprint: str
    evidence_policy_version: str
    evidence_fingerprint: str
    trigger_reasons: List[str] = Field(default_factory=list)
    proposer_reasoning: str = Field(min_length=1, max_length=4000)
    validator_reasoning: str = Field(min_length=1, max_length=4000)


class ProofPruneReviewResult(BaseModel):
    """Side-effect-free terminal result of one Build 03 review."""
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["no_prune", "rejected", "commit_intent"]
    proposal: ProofPruneProposal
    validation: Optional[ProofPruneValidation] = None
    commit_intent: Optional[ProofPruneCommitIntent] = None


class ProofPruningState(BaseModel):
    """Durable, run-scoped Build 04 pruning orchestration state."""
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    policy_version: str
    proof_run_id: str
    run_id: str
    lifecycle_generation: int = Field(ge=1)
    scope: Literal["autonomous", "manual"]
    source_type: Literal["brainstorm", "paper"]
    source_id: str
    proof_store_id: str
    proof_set_revision: int = Field(default=0, ge=0)
    accepted_prompt_novel_total: int = Field(default=0, ge=0)
    last_scheduled_acceptance_baseline: int = Field(default=0, ge=0)
    counted_proof_ids: List[str] = Field(default_factory=list)
    queued_trigger_reasons: List[str] = Field(default_factory=list)
    active_trigger_reasons: List[str] = Field(default_factory=list)
    requested_snapshot_revision: int = Field(default=0, ge=0)
    follow_up_required: bool = False
    round_index: int = Field(default=1, ge=1)
    active_proposal_id: str = ""
    active_proposal_generation: int = Field(default=0, ge=0)
    snapshot_id: str = ""
    snapshot_revision: Optional[int] = Field(default=None, ge=0)
    target_proof_id: str = ""
    status: Literal[
        "disabled",
        "idle",
        "queued",
        "proposing",
        "validating",
        "provider_paused",
        "repair_required",
        "applied",
        "no_prune",
        "rejected",
        "stale",
        "error",
    ] = "idle"
    sanitized_proposal: Dict[str, Any] = Field(default_factory=dict)
    sanitized_validation: Dict[str, Any] = Field(default_factory=dict)
    provider_error_classification: str = ""
    retry_count: int = Field(default=0, ge=0)
    last_error_summary: str = Field(default="", max_length=1800)
    last_applied_proof_id: str = ""
    context_pressure: ProofPruneContextPressure = Field(
        default_factory=ProofPruneContextPressure
    )
    last_reviewed_pressure_fingerprint: str = ""
    last_reviewed_pressure_revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=datetime.now)


@dataclass
class SmtResult:
    """Result of one SMT solver check."""
    success: bool
    result: str = ""
    stdout: str = ""
    stderr: str = ""


class ProofAttemptFeedback(BaseModel):
    """Lean 4 attempt feedback captured for one theorem attempt."""
    attempt: int
    theorem_id: str
    reasoning: str = ""
    lean_code: str = ""
    error_output: str = ""
    diagnostic_output: str = ""
    goal_states: str = ""
    raw_stderr: str = ""
    strategy: Literal["full_script", "tactic_script"] = "full_script"
    tactic_trace: List[str] = Field(default_factory=list)
    success: bool = False
    configured_model: Optional[str] = None
    configured_provider: Optional[str] = None
    effective_model: Optional[str] = None
    effective_provider: Optional[str] = None
    overflow_origin: Optional[Literal["local_preflight", "provider"]] = None
    prompt_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    failure_kind: Optional[Literal[
        "output_truncated",
        "malformed_output",
        "lean_rejected",
        "context_overflow",
        "workspace_error",
        "integrity_rejected",
    ]] = None
    recovery_step: int = 1
    recovery_mode: str = "configured"
    recovery_policy_version: str = "proof-truncation-v1"
    reasoning_effort: Optional[str] = None
    requested_output_tokens: Optional[int] = None
    response_mode: str = "json"
    supercharge_disabled: bool = False
    lean_was_run: bool = False


ProofArtifactPurpose = Literal[
    "verified_occurrence",
    "standalone_exact_duplicate_emphasis",
]


class ProofRecord(BaseModel):
    """Stored proof metadata for the proof library and prompt injection."""
    proof_id: str
    theorem_id: str = ""
    theorem_statement: str
    theorem_name: str = ""
    formal_sketch: str = ""
    source_type: Literal["brainstorm", "paper", "leanoj_subproof", "leanoj_final"]
    source_id: str
    source_title: str = ""
    run_id: str = ""
    user_prompt: str = ""
    solver: str = "Lean 4"
    lean_code: str
    novel: bool = False
    novelty_tier: str = "not_novel"
    novelty_reasoning: str = ""
    independent_novelty_tier: str = ""
    independent_novelty_reasoning: str = ""
    exact_duplicate_proof_id: str = ""
    exact_duplicate_run_id: str = ""
    artifact_purpose: ProofArtifactPurpose = "verified_occurrence"
    canonical_identity_version: str = ""
    canonical_theorem_statement_hash: str = ""
    canonical_lean_code_hash: str = ""
    verification_notes: str = ""
    attempt_count: int = 0
    attempts: List[ProofAttemptFeedback] = Field(default_factory=list)
    dependencies: List[ProofDependency] = Field(default_factory=list)
    dependency_extraction_status: ProofDependencyExtractionStatus = "not_attempted"
    dependency_extraction_detail: str = Field(default="", max_length=1000)
    dependency_extracted_at: Optional[datetime] = None
    solver_hints: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    live_context_status: Literal["active", "pruned"] = "active"
    live_context_owner_run_id: str = ""
    live_context_pruned_at: Optional[datetime] = None
    live_context_pruned_by: Optional[Literal["user", "automatic_proof_pruning"]] = None
    live_context_prune_reason: str = ""
    live_context_prune_validator_reasoning: str = ""
    live_context_prune_snapshot_revision: Optional[int] = None
    live_context_prune_trigger_reasons: List[str] = Field(default_factory=list)
    live_context_prune_supporting_proof_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_live_context_state(self):
        if self.live_context_status == "active":
            self.live_context_owner_run_id = ""
            self.live_context_pruned_at = None
            self.live_context_pruned_by = None
            self.live_context_prune_reason = ""
            self.live_context_prune_validator_reasoning = ""
            self.live_context_prune_snapshot_revision = None
            self.live_context_prune_trigger_reasons = []
            self.live_context_prune_supporting_proof_ids = []
            return self
        if not self.live_context_owner_run_id.strip():
            raise ValueError("Pruned proofs require an owning run ID")
        if self.live_context_pruned_at is None:
            raise ValueError("Pruned proofs require a prune timestamp")
        if self.live_context_pruned_by is None:
            raise ValueError("Pruned proofs require a prune actor")
        if not self.live_context_prune_reason.strip():
            raise ValueError("Pruned proofs require a reason")
        return self


class ProofLibraryEntry(BaseModel):
    """Stable metadata returned by archived proof-library endpoints."""
    library_id: str = ""
    session_id: str = ""
    proof_id: str
    theorem_name: str = ""
    theorem_statement: str = ""
    formal_sketch: str = ""
    source_type: str = ""
    source_id: str = ""
    source_title: str = ""
    run_id: str = ""
    user_prompt: str = ""
    solver: str = "Lean 4"
    novel: bool = False
    novelty_tier: str = "not_novel"
    novelty_reasoning: str = ""
    independent_novelty_tier: str = ""
    independent_novelty_reasoning: str = ""
    exact_duplicate_proof_id: str = ""
    exact_duplicate_run_id: str = ""
    artifact_purpose: ProofArtifactPurpose = "verified_occurrence"
    canonical_identity_version: str = ""
    canonical_theorem_statement_hash: str = ""
    canonical_lean_code_hash: str = ""
    verification_notes: str = ""
    attempt_count: int = 0
    created_at: str = ""
    dependencies: List[ProofDependency] = Field(default_factory=list)
    lean_code: str = ""
    live_context_status: Literal["active", "pruned"] = "active"
    live_context_owner_run_id: str = ""
    live_context_pruned_at: Optional[str] = None
    live_context_pruned_by: Optional[Literal["user", "automatic_proof_pruning"]] = None
    live_context_prune_reason: str = ""
    live_context_prune_validator_reasoning: str = ""
    live_context_prune_snapshot_revision: Optional[int] = None
    live_context_prune_trigger_reasons: List[str] = Field(default_factory=list)


class ProofLibraryCounts(BaseModel):
    total: int = 0
    listed: int = 0
    novel: int = 0
    duplicate_novel: int = 0
    not_novel: int = 0
    live_context_active: int = 0
    live_context_pruned: int = 0


class ProofLibraryResponse(BaseModel):
    proofs: List[ProofLibraryEntry] = Field(default_factory=list)
    counts: ProofLibraryCounts = Field(default_factory=ProofLibraryCounts)
    scope: Literal["autonomous", "manual"]
    category: Literal["novel", "duplicate_novel", "not_novel", "all"]


class CurrentProofListResponse(BaseModel):
    """Current proof-store contents with the revision used by live-context controls."""

    proofs: List[ProofRecord] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    scope: Literal["autonomous", "manual"]
    proof_set_revision: int = Field(ge=0)


class ProofCertificateResponse(BaseModel):
    proof_id: str
    theorem_statement: str
    theorem_name: str = ""
    lean_code: str = ""
    solver: str = "Lean 4"
    lean_version: str = ""
    mathlib_commit: str = ""
    verified_at: Optional[str] = None
    source_type: str = ""
    source_id: str = ""
    source_title: str = ""
    run_id: str = ""
    user_prompt: str = ""
    novel: bool = False
    novelty_tier: str = "not_novel"
    novelty_reasoning: str = ""
    independent_novelty_tier: str = ""
    independent_novelty_reasoning: str = ""
    exact_duplicate_proof_id: str = ""
    exact_duplicate_run_id: str = ""
    artifact_purpose: ProofArtifactPurpose = "verified_occurrence"
    canonical_identity_version: str = ""
    canonical_theorem_statement_hash: str = ""
    canonical_lean_code_hash: str = ""
    attempt_count: int = 0
    solver_hints: List[str] = Field(default_factory=list)
    dependencies: List[ProofDependency] = Field(default_factory=list)
    live_context_status: Literal["active", "pruned"] = "active"
    live_context_owner_run_id: str = ""
    live_context_pruned_at: Optional[str] = None
    live_context_pruned_by: Optional[Literal["user", "automatic_proof_pruning"]] = None
    live_context_prune_reason: str = ""
    live_context_prune_validator_reasoning: str = ""
    live_context_prune_snapshot_revision: Optional[int] = None
    live_context_prune_trigger_reasons: List[str] = Field(default_factory=list)


class ProofAttemptResult(BaseModel):
    """Outcome of one theorem proof-attempt loop."""
    theorem_id: str
    theorem_statement: str
    lean_code: str = ""
    success: bool = False
    novel: bool = False
    attempts_used: int = 0
    proof_id: Optional[str] = None
    error_summary: str = ""
    candidate_fingerprint: str = ""
    truncation_recovery_exhausted: bool = False


class ProofStageResult(BaseModel):
    """Aggregate outcome of one proof-verification stage run."""
    source_type: Literal["brainstorm", "paper"]
    source_id: str
    total_candidates: int = 0
    verified_count: int = 0
    novel_count: int = 0
    results: List[ProofAttemptResult] = Field(default_factory=list)
    had_error: bool = False
    error_message: str = ""
    deferred_candidate_ids: List[str] = Field(default_factory=list)
    context_overflow_payload: Dict[str, Any] = Field(default_factory=dict)
    fatal_stop_reason: str = ""
    fatal_stop_payload: Dict[str, Any] = Field(default_factory=dict)


ProofRunMode = Literal["one_round", "loop_with_pruning"]
ProofRunStatus = Literal[
    "queued",
    "running",
    "provider_paused",
    "stopping",
    "completed",
    "stopped",
    "error",
]
ProofRunIdlePolicy = Literal["provider_reset"]
ProofPruningStatus = Literal[
    "disabled",
    "idle",
    "queued",
    "proposing",
    "validating",
    "provider_paused",
    "repair_required",
    "applied",
    "no_prune",
    "rejected",
    "stale",
    "error",
]


class ProofRunEventContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proof_run_id: str
    run_mode: ProofRunMode
    lifecycle_generation: int = Field(ge=1)
    proof_run_unbounded: bool = False


class ProofRunSnapshot(BaseModel):
    """Current-process lifecycle projection for a manual proof run."""

    model_config = ConfigDict(extra="forbid")

    proof_run_id: str
    run_mode: ProofRunMode
    scope: Literal["autonomous", "manual"]
    source_type: Literal["brainstorm", "paper"]
    source_id: str
    source_title: str = ""
    proof_store_id: str
    run_id: str
    lifecycle_generation: int = Field(ge=1)
    status: ProofRunStatus
    round_limit: Optional[int] = Field(default=1, ge=1)
    unbounded: bool = False
    current_round: int = Field(default=0, ge=0)
    last_completed_round: int = Field(default=0, ge=0)
    last_round_summary: str = Field(default="", max_length=4000)
    last_round_reference: str = Field(default="", max_length=512)
    source_content_fingerprint: str = ""
    source_revision: int = Field(default=0, ge=0)
    proof_set_revision: int = Field(default=0, ge=0)
    candidate_checkpoint_reference: str = Field(default="", max_length=512)
    route_runtime_fingerprint: str = Field(default="", max_length=256)
    idle_reason: str = Field(default="", max_length=1000)
    wake_generation: int = Field(default=0, ge=0)
    idle_policy: Optional[ProofRunIdlePolicy] = None
    provider_state: Optional[Dict[str, Any]] = None
    policy_version: str = "proof-run-policy-v3"
    schema_version: int = Field(default=3, ge=1)
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    terminal_reason: str = ""
    stop_requested: bool = False
    pruning_status: ProofPruningStatus = "disabled"
    pruning_state: Optional[Dict[str, Any]] = None
    last_error_summary: str = ""
    terminal_event_emitted: bool = False
    cleanup_completed: bool = False


class ProofRunQueueResponse(ProofRunSnapshot):
    queued: bool = True


class ProofRunCollectionItem(BaseModel):
    """Bounded proof-run metadata safe for reconnect and queue recovery."""

    model_config = ConfigDict(extra="forbid")

    proof_run_id: str
    run_mode: ProofRunMode
    scope: Literal["autonomous", "manual"]
    source_type: Literal["brainstorm", "paper"]
    source_id: str
    source_title: str = ""
    run_id: str
    lifecycle_generation: int = Field(ge=1)
    status: ProofRunStatus
    current_round: int = Field(default=0, ge=0)
    last_completed_round: int = Field(default=0, ge=0)
    proof_set_revision: int = Field(default=0, ge=0)
    updated_at: datetime
    terminal_reason: str = ""
    pruning_status: ProofPruningStatus = "disabled"


class ProofRunCollectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: List[ProofRunCollectionItem] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    limit: int = Field(ge=1, le=50)
    truncated: bool = False


class ProofRunSourceLookupResponse(ProofRunCollectionResponse):
    scope: Literal["autonomous", "manual"]
    source_type: Literal["brainstorm", "paper"]
    source_id: str
    ambiguous: bool = False
    preferred_proof_run_id: Optional[str] = None


class ProofRunStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_lifecycle_generation: int = Field(ge=1)


class ProofLiveContextMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["active", "pruned"]
    actor: Literal["user"] = "user"
    expected_run_id: str = Field(min_length=1, max_length=256)
    expected_proof_set_revision: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=2000)
    expected_theorem_hash: str = Field(default="", max_length=256)
    expected_lean_hash: str = Field(default="", max_length=256)


class ProofLiveContextMutationResponse(BaseModel):
    success: bool = True
    scope: Literal["autonomous", "manual"]
    proof_id: str
    run_id: str
    live_context_status: Literal["active", "pruned"]
    live_context_pruned_at: Optional[datetime] = None
    proof_search_refresh_scheduled: bool = False
    proof_set_revision: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)


class ProofCheckRequest(BaseModel):
    """Request body for manually triggering a proof check."""
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["brainstorm", "paper"]
    source_id: str
    proof_runtime_config: Optional[Dict[str, Any]] = None
    run_mode: ProofRunMode = "one_round"


class ProofSettingsUpdateRequest(BaseModel):
    """Request body for updating runtime Lean 4 proof settings."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    timeout: int = Field(default=600, ge=10, le=3600)
    lean4_lsp_enabled: Optional[bool] = None
    lean4_lsp_idle_timeout: Optional[int] = Field(default=None, ge=60, le=7200)
    max_parallel_candidates: Optional[int] = Field(default=None, ge=0, le=1000)
    smt_enabled: Optional[bool] = None
    smt_timeout: Optional[int] = Field(default=None, ge=1, le=600)


# ============================================================================
# LEANOJ PROOF SOLVER MODELS
# ============================================================================


class LeanOJRoleConfig(BaseModel):
    """Model/runtime configuration for one LeanOJ proof-solver role."""
    provider: ModelProvider = "lm_studio"
    model_id: str = ""
    openrouter_provider: Optional[str] = None
    openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    lm_studio_fallback_id: Optional[str] = None
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    supercharge_enabled: bool = False


class LeanOJStartRequest(BaseModel):
    """Request to start the LeanOJ proof-solver mode."""
    user_prompt: str
    lean_template: str
    creativity_emphasis_boost_enabled: bool = False
    topic_generator: LeanOJRoleConfig
    topic_validator: LeanOJRoleConfig
    brainstorm_submitters: List[LeanOJRoleConfig] = Field(default_factory=list, min_length=1, max_length=10)
    brainstorm_validator: LeanOJRoleConfig
    path_decider: LeanOJRoleConfig = Field(default_factory=LeanOJRoleConfig)
    final_solver: LeanOJRoleConfig
    assistant: LeanOJRoleConfig = Field(default_factory=LeanOJRoleConfig)
    max_initial_brainstorm_accepts: int = Field(default=30, ge=1, le=200)
    max_recursive_brainstorm_accepts: int = Field(default=10, ge=1, le=100)
    final_attempts_per_cycle: int = Field(default=30, ge=30, le=200)


class LeanOJAttemptRecord(BaseModel):
    """One Lean 4 attempt made by the LeanOJ solver."""
    attempt: int
    target: Literal["subproof", "final"]
    request: str = ""
    lean_code: str = ""
    success: bool = False
    error_output: str = ""
    reasoning: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class LeanOJSubproofRecord(BaseModel):
    """Verified or exhausted subproof produced during one LeanOJ run."""
    subproof_id: str
    request: str
    role: str = ""
    theorem_or_lemma: str = ""
    verified: bool = False
    lean_code: str = ""
    lean_feedback: str = ""
    attempts_used: int = 0
    error_summary: str = ""
    proof_id: str = ""
    novel: bool = False
    novelty_tier: str = "not_novel"
    novelty_reasoning: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class LeanOJState(BaseModel):
    """Current state snapshot for LeanOJ proof-solver mode."""
    is_running: bool = False
    phase: Literal[
        "idle",
        "initial_topic_candidates",
        "initial_brainstorm",
        "path_decision",
        "recursive_brainstorm",
        "proof_storm",
        "final_proof_loop",
        "verified",
        "stopped",
        "error",
    ] = "idle"
    last_active_phase: str = ""
    active_brainstorm_phase: str = ""
    active_brainstorm_start_count: int = 0
    session_id: str = ""
    selected_topic: str = ""
    current_path_decision: str = ""
    accepted_brainstorm_count: int = 0
    rejected_brainstorm_count: int = 0
    brainstorm_acceptance_events: int = 0
    active_brainstorm_last_sufficiency_check_count: int = 0
    active_brainstorm_last_prune_review_count: int = 0
    brainstorm_prune_reviews_performed: int = 0
    brainstorm_prune_operations_applied: int = 0
    recursive_cycle_count: int = 0
    verified_subproofs: List[LeanOJSubproofRecord] = Field(default_factory=list)
    failed_subproofs: List[LeanOJSubproofRecord] = Field(default_factory=list)
    final_attempt_count: int = 0
    final_solution: str = ""
    final_proof_id: str = ""
    final_novel: bool = False
    final_novelty_tier: str = "not_novel"
    final_novelty_reasoning: str = ""
    master_proof_initialized: bool = False
    master_proof_version: int = 0
    master_proof_hash: str = ""
    master_proof_line_count: int = 0
    master_proof_char_count: int = 0
    master_proof_last_edit_summary: str = ""
    master_proof_last_stuck_reason: str = ""
    master_proof_old_attempt_before_redo_version: int = 0
    master_proof_old_attempt_before_redo_hash: str = ""
    master_proof_old_attempt_before_redo_line_count: int = 0
    master_proof_old_attempt_before_redo_char_count: int = 0
    master_proof_old_attempt_before_redo_summary: str = ""
    master_proof_old_attempt_before_redo_validator_justification: str = ""
    master_proof_old_attempt_before_redo_apparent_issue: str = ""
    master_proof_last_shortening_approval_justification: str = ""
    master_proof_last_shortening_apparent_issue: str = ""
    last_error: str = ""
    provider_paused: bool = False
    provider_pause_reason: str = ""
    provider_pause_role_id: str = ""
    provider_pause_message: str = ""
    skip_brainstorm_requested: bool = False
    force_brainstorm_requested: bool = False
    user_forced_final_cycle: bool = False
    updated_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# TIER 3: FINAL ANSWER MODELS (Part 3 - Final Answer Generation)
# ============================================================================


class CertaintyAssessment(BaseModel):
    """
    Assessment of what can be answered with certainty from existing papers.
    Phase 1 of Tier 3 workflow.
    """
    certainty_level: Literal[
        "total_answer",      # User's question can be fully answered with known certainties
        "partial_answer",    # Can provide partial answer with some unknowns
        "no_answer_known",   # Existing research doesn't provide an answer
        "appears_impossible", # The question appears mathematically impossible
        "other"              # Special cases
    ]
    known_certainties_summary: str  # Summary of what is known with certainty
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class AnswerFormatSelection(BaseModel):
    """
    Selection of final answer format (short vs long form).
    Phase 2 of Tier 3 workflow.
    """
    answer_format: Literal["short_form", "long_form"]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class VolumeChapter(BaseModel):
    """
    A single chapter in a long-form volume answer.
    """
    chapter_type: Literal[
        "existing_paper",  # Existing Tier 2 paper used as-is
        "introduction",    # Introduction paper (written last)
        "conclusion",      # Conclusion paper (written second-to-last)
        "gap_paper"        # New paper to fill content gap
    ]
    paper_id: Optional[str] = None  # For existing papers or newly written gap/intro/conclusion papers
    title: str
    order: int  # Chapter ordering in volume (1-based)
    status: Literal["pending", "writing", "complete"] = "pending"
    description: str = ""  # Brief description of chapter content/purpose


class VolumeOrganization(BaseModel):
    """
    Organization structure for a long-form volume answer.
    Iteratively refined until outline_complete=True.
    """
    volume_title: str  # Title of the overall volume
    chapters: List[VolumeChapter] = Field(default_factory=list)
    needs_revision: bool = False  # If True, validator requests changes
    revision_reasoning: str = ""  # Feedback for revision
    outline_complete: bool = False  # Set True when submitter and validator agree
    timestamp: datetime = Field(default_factory=datetime.now)


class VolumeOrganizationSubmission(BaseModel):
    """
    Submission for volume organization (creation or update).
    """
    volume_title: str
    chapters: List[Dict[str, Any]]  # List of chapter definitions
    outline_complete: bool = False  # Submitter signals satisfaction
    reasoning: str


class ModelUsageEntry(BaseModel):
    """
    Tracks usage of a single model during Tier 3 final answer generation.
    Same model used in multiple instances counts as ONE author entry,
    but all API calls are still tallied.
    """
    model_id: str  # The model identifier (e.g., "deepseek-r1:70b")
    api_call_count: int = 0  # Number of API calls made with this model
    first_used: datetime = Field(default_factory=datetime.now)  # When first used


class ModelUsageTracker(BaseModel):
    """
    Tracks all model usage during Tier 3 final answer generation.
    Used to generate author attribution and model credits sections.
    """
    # Dict mapping model_id to its usage entry
    models: Dict[str, ModelUsageEntry] = Field(default_factory=dict)
    
    # The user's original research prompt (for attribution)
    user_prompt: str = ""
    
    # When Tier 3 generation started
    generation_date: datetime = Field(default_factory=datetime.now)
    
    # Total API calls across all models
    total_api_calls: int = 0
    
    def track_call(self, model_id: str) -> None:
        """Record an API call for a model."""
        if model_id not in self.models:
            self.models[model_id] = ModelUsageEntry(model_id=model_id)
        self.models[model_id].api_call_count += 1
        self.total_api_calls += 1
    
    def get_unique_authors(self) -> List[str]:
        """Get list of unique model IDs (authors)."""
        return list(self.models.keys())
    
    def get_models_by_usage(self) -> List[ModelUsageEntry]:
        """Get models sorted by API call count (descending)."""
        return sorted(
            self.models.values(),
            key=lambda x: x.api_call_count,
            reverse=True
        )


class FinalAnswerState(BaseModel):
    """
    Current state of Tier 3 final answer generation.
    Persisted for crash recovery.
    """
    is_active: bool = False
    answer_format: Optional[Literal["short_form", "long_form"]] = None
    certainty_assessment: Optional[CertaintyAssessment] = None
    volume_organization: Optional[VolumeOrganization] = None
    
    # Short form tracking
    short_form_paper_id: Optional[str] = None
    short_form_reference_papers: List[str] = Field(default_factory=list)
    
    # Long form tracking
    current_writing_chapter: Optional[int] = None  # 1-based chapter order being written
    completed_chapters: List[int] = Field(default_factory=list)  # Completed chapter orders
    
    # Model usage tracking for Tier 3
    # Tracks all models used and their API call counts for author attribution and credits
    model_usage: Optional[ModelUsageTracker] = None
    
    # Status tracking
    status: Literal[
        "idle",               # Not active
        "assessing",          # Phase 1: Certainty assessment
        "phase1_assessment",  # Phase 1: Certainty assessment (alias)
        "format_selecting",   # Phase 2: Choosing short/long form
        "phase2_format",      # Phase 2: Choosing short/long form (alias)
        "selecting_references", # Selecting reference papers (short form)
        "phase3a_short_form", # Phase 3A: Short form writing
        "organizing_volume",  # Phase 3B: Creating volume organization (long form)
        "phase3b_long_form",  # Phase 3B: Long form processing
        "writing",            # Writing papers (short or long form)
        "complete"            # Final answer complete - system will stop
    ] = "idle"
    
    # Statistics
    tier3_assessment_rejections: int = 0
    tier3_format_rejections: int = 0
    tier3_volume_rejections: int = 0
    tier3_writing_rejections: int = 0
    
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# PAPER CRITIQUE MODELS (Validator Critique Feature)
# ============================================================================


class PaperCritique(BaseModel):
    """
    A single critique of a paper from the validator model.
    Stores ratings (1-10) and feedback for Novelty, Correctness, and Impact.
    """
    critique_id: str
    model_id: str  # The model that provided this critique
    provider: str = "lm_studio"  # "lm_studio" or "openrouter"
    host_provider: Optional[str] = None  # e.g., "Anthropic", "Google AI" (for OpenRouter)
    date: datetime = Field(default_factory=datetime.now)
    prompt_used: Optional[str] = None  # The prompt used for this critique (for regeneration)
    critique_source: Literal["system_auto", "user_request", "unknown"] = "unknown"
    
    # Ratings (1-10 scale)
    novelty_rating: int = Field(default=0, ge=0, le=10)
    novelty_feedback: str = ""
    correctness_rating: int = Field(default=0, ge=0, le=10)
    correctness_feedback: str = ""
    impact_rating: int = Field(default=0, ge=0, le=10)
    impact_feedback: str = ""
    
    # Overall critique summary
    full_critique: str = ""


class CritiqueRequest(BaseModel):
    """Request body for generating a paper critique."""
    custom_prompt: Optional[str] = None  # User's custom prompt, or None for default
    
    # Optional validator configuration - allows critiques without starting autonomous research
    # If provided, these override the autonomous coordinator's stored config
    validator_model: Optional[str] = None
    validator_context_window: Optional[int] = None
    validator_max_tokens: Optional[int] = None
    validator_provider: Optional[str] = None  # "lm_studio" or "openrouter"
    validator_openrouter_provider: Optional[str] = None  # Specific provider like "Anthropic"
    validator_openrouter_reasoning_effort: OpenRouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT
    validator_supercharge_enabled: bool = False