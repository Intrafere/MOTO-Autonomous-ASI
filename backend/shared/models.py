"""
Pydantic models for the ASI Aggregator System.
"""
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


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


class ModelConfig(BaseModel):
    """Configuration for a model (can be LM Studio or OpenRouter)."""
    provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    model_id: str
    openrouter_model_id: Optional[str] = None  # For OpenRouter (different naming)
    openrouter_provider: Optional[str] = None  # Specific OpenRouter provider (e.g., "Anthropic")
    lm_studio_fallback_id: Optional[str] = None  # Fallback LM Studio model if OpenRouter fails
    context_window: int = 131072
    max_output_tokens: int = 25000


class BoostConfig(BaseModel):
    """API boost configuration."""
    enabled: bool = False
    openrouter_api_key: str = ""
    boost_model_id: str = ""  # OpenRouter model to use for boost
    boost_provider: Optional[str] = None  # Specific provider, or None to let OpenRouter choose
    boost_context_window: int = 131072
    boost_max_output_tokens: int = 25000


class WorkflowTask(BaseModel):
    """Represents a predicted API call in the workflow."""
    task_id: str  # Unique ID like "agg_sub1_001"
    sequence_number: int  # 1-20
    role: str  # "Submitter 1", "Validator", "High-Context", etc.
    mode: Optional[str] = None  # "Construction", "Rigor", "Review", etc.
    provider: str = "lm_studio"  # "openrouter" | "lm_studio"
    using_boost: bool = False
    completed: bool = False
    active: bool = False  # Currently executing


class SubmitterConfig(BaseModel):
    """Configuration for a single aggregator submitter agent."""
    submitter_id: int
    provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    model_id: str  # LM Studio model OR OpenRouter model based on provider
    openrouter_provider: Optional[str] = None  # Specific OpenRouter provider (e.g., "Anthropic")
    lm_studio_fallback_id: Optional[str] = None  # Fallback LM Studio model if OpenRouter fails
    context_window: int = 131072
    max_output_tokens: int = 25000


class AggregatorStartRequest(BaseModel):
    """Request to start the aggregator."""
    user_prompt: str
    submitter_configs: List[SubmitterConfig]  # Per-submitter configs (1-10)
    # Validator config
    validator_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    validator_model: str  # LM Studio model OR OpenRouter model based on provider
    validator_openrouter_provider: Optional[str] = None  # Specific OpenRouter provider
    validator_lm_studio_fallback: Optional[str] = None  # Fallback if OpenRouter fails
    validator_context_size: int = 131072
    validator_max_output_tokens: int = 25000
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
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    miniscule_edit_count: int = 0
    in_critique_phase: bool = False
    critique_acceptances: int = 0
    paper_version: int = 1
    skip_critique_requested: bool = False  # Pre-emptive skip queued


class CompilerStartRequest(BaseModel):
    """Request to start the compiler."""
    compiler_prompt: str
    # Validator config
    validator_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    validator_model: str
    validator_openrouter_provider: Optional[str] = None
    validator_lm_studio_fallback: Optional[str] = None
    validator_context_size: int = 131072
    validator_max_output_tokens: int = 25000
    # High-context submitter config
    high_context_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    high_context_model: str
    high_context_openrouter_provider: Optional[str] = None
    high_context_lm_studio_fallback: Optional[str] = None
    high_context_context_size: int = 131072
    high_context_max_output_tokens: int = 25000
    # High-param submitter config
    high_param_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    high_param_model: str
    high_param_openrouter_provider: Optional[str] = None
    high_param_lm_studio_fallback: Optional[str] = None
    high_param_context_size: int = 131072
    high_param_max_output_tokens: int = 25000
    # Critique submitter config
    critique_submitter_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    critique_submitter_model: str
    critique_submitter_openrouter_provider: Optional[str] = None
    critique_submitter_lm_studio_fallback: Optional[str] = None
    critique_submitter_context_window: int = 131072
    critique_submitter_max_tokens: int = 25000


# ============================================================================
# AUTONOMOUS RESEARCH MODELS (Part 3)
# ============================================================================


class BrainstormMetadata(BaseModel):
    """Metadata for a brainstorm topic."""
    topic_id: str
    topic_prompt: str
    status: Literal["in_progress", "complete"] = "in_progress"
    submission_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    last_activity: datetime = Field(default_factory=datetime.now)
    papers_generated: List[str] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Metadata for a completed or in-progress paper."""
    paper_id: str
    title: str
    abstract: str = ""
    word_count: int = 0
    source_brainstorm_ids: List[str] = Field(default_factory=list)
    referenced_papers: List[str] = Field(default_factory=list)
    status: Literal["in_progress", "complete", "archived"] = "complete"
    created_at: datetime = Field(default_factory=datetime.now)
    # Per-paper model tracking: model_id -> API call count
    model_usage: Optional[Dict[str, int]] = None
    # Generation date for the paper (separate from created_at for tracking purposes)
    generation_date: Optional[datetime] = None
    # Wolfram Alpha verification count (tracked separately from LLM API calls)
    wolfram_calls: Optional[int] = None


class TopicSelectionSubmission(BaseModel):
    """Submission from topic selection agent."""
    action: Literal["new_topic", "continue_existing", "combine_topics"]
    topic_id: Optional[str] = None  # Required if action is continue_existing
    topic_ids: List[str] = Field(default_factory=list)  # Required if action is combine_topics
    topic_prompt: str = ""  # Required if action is new_topic or combine_topics
    reasoning: str


class TopicValidationResult(BaseModel):
    """Result of topic validation."""
    decision: Literal["accept", "reject"]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


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
    selected_papers: List[str] = Field(default_factory=list)  # Max 6 paper IDs
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
    total_submissions_accepted: int = 0
    total_submissions_rejected: int = 0
    topic_selection_rejections: int = 0
    completion_reviews_run: int = 0
    paper_redundancy_reviews_run: int = 0
    tier3_triggers: int = 0  # Number of times Tier 3 has been triggered


class AutonomousResearchStartRequest(BaseModel):
    """Request to start autonomous research mode."""
    user_research_prompt: str
    submitter_configs: List[SubmitterConfig]  # Per-submitter configs for brainstorm aggregation (1-10)
    # Validator config
    validator_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    validator_model: str
    validator_openrouter_provider: Optional[str] = None
    validator_lm_studio_fallback: Optional[str] = None
    validator_context_window: int = 131072
    validator_max_tokens: int = 25000
    # Compiler high-context settings (separate from aggregator submitters)
    high_context_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    high_context_model: str = ""  # Empty string allowed, will use submitter model as fallback
    high_context_openrouter_provider: Optional[str] = None
    high_context_lm_studio_fallback: Optional[str] = None
    high_context_context_window: int = 131072
    high_context_max_tokens: int = 25000
    # Compiler high-param settings
    high_param_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    high_param_model: str = ""  # Empty string allowed, will use submitter model as fallback
    high_param_openrouter_provider: Optional[str] = None
    high_param_lm_studio_fallback: Optional[str] = None
    high_param_context_window: int = 131072
    high_param_max_tokens: int = 25000
    # Critique submitter settings
    critique_submitter_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    critique_submitter_model: str = ""  # For critique generation and rewrite decisions (uses high_context if empty)
    critique_submitter_openrouter_provider: Optional[str] = None
    critique_submitter_lm_studio_fallback: Optional[str] = None
    critique_submitter_context_window: int = 131072
    critique_submitter_max_tokens: int = 25000


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