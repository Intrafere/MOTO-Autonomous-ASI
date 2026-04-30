"""
Configuration for the ASI Aggregator System.
Defines RAG parameters, context allocation, and system constants.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """RAG system configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Chunk size configurations (chars)
    submitter_chunk_intervals: List[int] = [256, 512, 768, 1024]
    validator_chunk_size: int = 512
    chunk_overlap_percentage: float = 0.20
    
    # Quality thresholds
    coverage_threshold: float = 0.25
    answerability_threshold: float = 0.15
    
    # Context allocation (tokens)
    # NOTE: These are DEFAULT values only. User sets actual context via GUI settings.
    # The system will use whatever context the user configured in LM Studio and enters in settings.
    # NO LIMIT is enforced - these defaults are just fallbacks.
    submitter_context_window: int = 131072  # Default if user doesn't specify
    validator_context_window: int = 131072  # Default if user doesn't specify
    context_buffer_tokens: int = 500  # Small buffer for token counting estimation errors
    output_reserve_tokens: int = 25000  # CRITICAL: Reserve for model output generation (matches default max_output_tokens)
    rag_allocation_percentage: float = 0.85  # 85% RAG, 15% direct injection (of remaining context)
    
    # Output token limits (user-configurable)
    submitter_max_output_tokens: int = 25000  # Default for aggregator submitters
    validator_max_output_tokens: int = 25000  # Default for aggregator validator
    
    # Memory limits
    max_documents: int = 10000  # For RAG document cache; user files never evicted; high for infinite runtime
    max_chunks_per_size: int = 10000  # Per-size chunk cap; oldest non-permanent trimmed when exceeded
    max_shared_training_insights: int = 999999  # Effectively unlimited for infinite runtime
    max_local_rejections: int = 5  # Per rules: "last 5 rejections"
    
    # Cache settings
    rewrite_cache_size: int = 500
    rewrite_cache_ttl: int = 1800  # 30 minutes
    bm25_cache_size: int = 1000
    bm25_cache_ttl: int = 3600  # 1 hour
    context_pack_cache_size: int = 300
    
    # Retrieval settings
    query_rewrite_variants: int = 5
    hybrid_recall_top_k: int = 120
    vector_weight: float = 0.60
    bm25_weight: float = 0.40
    mmr_lambda: float = 0.80  # 80% relevance, 20% diversity
    similarity_threshold: float = 0.85
    
    # LM Studio API
    lm_studio_base_url: str = Field(
        default="http://127.0.0.1:1234",
        validation_alias=AliasChoices("MOTO_LM_STUDIO_BASE_URL", "LM_STUDIO_BASE_URL"),
    )
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    
    # OpenRouter API (Global Configuration)
    # This is the default OpenRouter API key used for per-role model selection.
    # API Boost can also reuse it unless the boost modal supplies an override key.
    openrouter_api_key: Optional[str] = None
    openrouter_enabled: bool = False  # True when API key is set and validated
    
    # Debug
    debug_mode: bool = False
    
    def get_available_input_tokens(self, context_window: int, output_tokens: int = None) -> int:
        """
        Calculate available tokens for INPUT prompt (excluding output reserve).
        This is the maximum tokens that can be used for the assembled input prompt.
        
        Formula: context_window - output_reserve - buffer
        
        The buffer accounts for token counting estimation errors.
        
        Args:
            context_window: Total context window size
            output_tokens: Optional output tokens to reserve (defaults to self.output_reserve_tokens)
            
        Returns:
            Available tokens for input prompt assembly
        """
        # Use provided output tokens or fall back to default
        output_reserve = output_tokens if output_tokens is not None else self.output_reserve_tokens
        
        # Fixed buffer for token counting estimation errors (industry standard approach)
        buffer = self.context_buffer_tokens
        
        return context_window - output_reserve - buffer
    
    def get_prompt_assembly_overhead_estimate(self) -> int:
        """
        Estimate additional tokens added during prompt assembly.
        Includes: separators (\n---\n), headers (USER PROMPT:, SUBMISSION TO VALIDATE:, etc.)
        Updated estimate: ~1000 tokens for formatting (increased from 600 for accuracy)
        """
        return 1000  # Realistic: separators (100) + headers (150) + JSON schemas (500) + RAG headers (150) + buffer (100)
    
    def get_minimum_rag_allocation(self, context_window: int, output_tokens: int = None) -> int:
        """Minimum tokens reserved for RAG retrieval for a given context window."""
        available_input = self.get_available_input_tokens(context_window, output_tokens)
        return int(available_input * 0.15)
    
    def get_chunk_overlap(self, chunk_size: int) -> int:
        """Calculate overlap for a given chunk size."""
        return int(chunk_size * self.chunk_overlap_percentage)


class SystemConfig(BaseSettings):
    """System-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Deployment mode
    generic_mode: bool = False

    # Instance/runtime identity
    instance_id: str = Field(
        default="default",
        validation_alias=AliasChoices("MOTO_INSTANCE_ID", "INSTANCE_ID"),
    )
    backend_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("MOTO_BACKEND_HOST", "HOST"),
    )
    backend_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("MOTO_BACKEND_PORT", "PORT"),
    )
    frontend_port: int = Field(
        default=5173,
        validation_alias=AliasChoices("MOTO_FRONTEND_PORT", "FRONTEND_PORT"),
    )
    secret_namespace: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MOTO_SECRET_NAMESPACE", "SECRET_NAMESPACE"),
    )
    internal_proxy_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MOTO_INTERNAL_PROXY_SECRET", "INTERNAL_PROXY_SECRET"),
    )
    frontend_storage_prefix: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MOTO_FRONTEND_STORAGE_PREFIX", "FRONTEND_STORAGE_PREFIX"),
    )
    
    # Aggregator submitter settings (configurable 1-10 submitters)
    default_num_submitters: int = 3  # Default number of submitters
    max_submitters: int = 10  # Hard cap on submitters
    min_submitters: int = 1  # Minimum submitters
    consecutive_rejection_reset_threshold: int = 15
    queue_overflow_threshold: int = 10
    per_submitter_queue_threshold: int = 4  # Pause an individual submitter when it already has more than this many submissions queued (fairness cap)
    
    # Compiler settings (Phase 2)
    # NOTE: Compiler contexts are set by user in GUI, these are just default fallbacks
    # Compiler context windows (separate for each role)
    compiler_validator_context_window: int = 131072
    compiler_high_context_context_window: int = 131072
    compiler_high_param_context_window: int = 131072
    compiler_critique_submitter_context_window: int = 131072  # For critique generation and rewrite decision
    
    # Compiler output token limits (user-configurable)
    compiler_validator_max_output_tokens: int = 25000
    compiler_high_context_max_output_tokens: int = 25000  # For outline_create, outline_update, construction, review
    compiler_high_param_max_output_tokens: int = 25000  # For rigor mode
    compiler_critique_submitter_max_tokens: int = 25000  # For critique and rewrite decision
    
    # Compiler model selections (set at runtime by API)
    compiler_critique_submitter_model: str = ""  # Set by user in GUI
    
    # Autonomous Research settings (Part 3)
    # Context windows (separate for each role)
    autonomous_submitter_context_window: int = 131072
    autonomous_validator_context_window: int = 131072
    autonomous_high_context_context_window: int = 131072
    autonomous_high_param_context_window: int = 131072
    
    # Autonomous output token limits (user-configurable)
    autonomous_submitter_max_tokens: int = 25000
    autonomous_validator_max_tokens: int = 25000
    autonomous_high_context_max_tokens: int = 25000
    autonomous_high_param_max_tokens: int = 25000
    
    # Autonomous workflow settings
    autonomous_completion_review_interval: int = 10  # Every 10 acceptances
    autonomous_paper_redundancy_interval: int = 3  # Every 3 completed papers
    autonomous_topic_cycle_max_reference_papers: int = 3  # Max pre-brainstorm + additional references per topic cycle
    autonomous_tier3_short_form_max_reference_papers: int = 6  # Max references for Tier 3 short-form selection
    
    # Wolfram Alpha integration (optional)
    wolfram_alpha_enabled: bool = False
    wolfram_alpha_api_key: Optional[str] = None

    # Lean 4 proof verification integration (optional)
    lean4_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("MOTO_LEAN4_ENABLED", "LEAN4_ENABLED"),
    )
    lean4_lsp_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("MOTO_LEAN4_LSP_ENABLED", "LEAN4_LSP_ENABLED"),
    )
    lean4_path: str = Field(
        default="",
        validation_alias=AliasChoices("MOTO_LEAN4_PATH", "LEAN4_PATH"),
    )
    lean4_workspace_dir: str = Field(
        default="",
        validation_alias=AliasChoices("MOTO_LEAN4_WORKSPACE_DIR", "LEAN4_WORKSPACE_DIR"),
    )
    lean4_proof_timeout: int = Field(
        default=120,
        validation_alias=AliasChoices("MOTO_LEAN4_PROOF_TIMEOUT", "LEAN4_PROOF_TIMEOUT"),
    )
    lean4_lsp_idle_timeout: int = Field(
        default=600,
        validation_alias=AliasChoices("MOTO_LEAN4_LSP_IDLE_TIMEOUT", "LEAN4_LSP_IDLE_TIMEOUT"),
    )
    # Maximum number of theorem candidates whose Lean 4 formalization attempts
    # may run concurrently within a single proof-verification stage. Novelty
    # assessment and proof-database persistence remain serialized after each
    # candidate's Lean pipeline completes.
    proof_max_parallel_candidates: int = Field(
        default=6,
        validation_alias=AliasChoices(
            "MOTO_PROOF_MAX_PARALLEL_CANDIDATES",
            "PROOF_MAX_PARALLEL_CANDIDATES",
        ),
    )

    # Optional SMT / Z3 integration foundation (Build 4)
    smt_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("MOTO_SMT_ENABLED", "SMT_ENABLED"),
    )
    z3_path: str = Field(
        default="",
        validation_alias=AliasChoices("MOTO_Z3_PATH", "Z3_PATH"),
    )
    smt_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("MOTO_SMT_TIMEOUT", "SMT_TIMEOUT"),
    )
    
    # File paths
    data_dir: str = Field(
        default="backend/data",
        validation_alias=AliasChoices("MOTO_DATA_ROOT", "DATA_DIR"),
    )
    logs_dir: Optional[str] = Field(
        default="backend/logs",
        validation_alias=AliasChoices("MOTO_LOG_ROOT", "LOGS_DIR"),
    )
    user_uploads_dir: Optional[str] = None
    chroma_db_dir: Optional[str] = None
    
    shared_training_file: Optional[str] = None
    compiler_outline_file: Optional[str] = None
    compiler_paper_file: Optional[str] = None
    compiler_rejections_file: Optional[str] = None
    compiler_acceptances_file: Optional[str] = None
    compiler_declines_file: Optional[str] = None
    
    # ========================================================================
    # AUTONOMOUS RESEARCH FILE PATHS (Part 3) - DUAL-PATH ARCHITECTURE
    # ========================================================================
    #
    # The autonomous research system uses TWO storage modes:
    #
    # 1. LEGACY PATHS (for backward compatibility):
    #    - auto_brainstorms_dir, auto_papers_dir, etc.
    #    - Used when existing legacy data is detected
    #    - Preserved for users with existing research
    #
    # 2. SESSION-BASED PATHS (preferred for new sessions):
    #    - auto_sessions_base_dir/{session_id}/brainstorms/
    #    - auto_sessions_base_dir/{session_id}/papers/
    #    - auto_sessions_base_dir/{session_id}/final_answer/
    #    - Created for new research sessions
    #
    # IMPORTANT: Do NOT add helper methods here for path resolution!
    # Path resolution is handled by memory modules (paper_library, brainstorm_memory,
    # etc.) which are session-aware. Using hardcoded paths from config can cause
    # critiques and other data to be stored in wrong locations.
    #
    # ========================================================================
    
    # Legacy paths (backward compatibility - do not use for new features)
    auto_brainstorms_dir: Optional[str] = None
    auto_papers_dir: Optional[str] = None
    auto_papers_archive_dir: Optional[str] = None
    auto_research_metadata_file: Optional[str] = None
    auto_research_stats_file: Optional[str] = None
    auto_workflow_state_file: Optional[str] = None
    auto_research_topic_rejections_file: Optional[str] = None
    
    # Session-based organization (preferred for new features)
    auto_sessions_base_dir: Optional[str] = None

    @model_validator(mode="after")
    def _derive_instance_paths(self) -> "SystemConfig":
        """Derive all mutable runtime paths from the instance data root."""
        default_data_dir = Path("backend/data")
        current_data_dir = Path(self.data_dir)
        current_logs_dir = Path(self.logs_dir) if self.logs_dir else None

        if current_logs_dir is None:
            self.logs_dir = "backend/logs" if current_data_dir == default_data_dir else str(current_data_dir / "_logs")
        elif current_logs_dir == Path("backend/logs") and current_data_dir != default_data_dir:
            self.logs_dir = str(current_data_dir / "_logs")

        def _join_data_path(*parts: str) -> str:
            return str(current_data_dir.joinpath(*parts))

        if not self.user_uploads_dir:
            self.user_uploads_dir = _join_data_path("user_uploads")
        if not self.chroma_db_dir:
            self.chroma_db_dir = _join_data_path("chroma_db")

        if not self.shared_training_file:
            self.shared_training_file = _join_data_path("rag_shared_training.txt")
        if not self.compiler_outline_file:
            self.compiler_outline_file = _join_data_path("compiler_outline.txt")
        if not self.compiler_paper_file:
            self.compiler_paper_file = _join_data_path("compiler_paper.txt")
        if not self.compiler_rejections_file:
            self.compiler_rejections_file = _join_data_path("compiler_last_10_rejections.txt")
        if not self.compiler_acceptances_file:
            self.compiler_acceptances_file = _join_data_path("compiler_last_10_acceptances.txt")
        if not self.compiler_declines_file:
            self.compiler_declines_file = _join_data_path("compiler_last_10_declines.txt")

        if not self.auto_brainstorms_dir:
            self.auto_brainstorms_dir = _join_data_path("auto_brainstorms")
        if not self.auto_papers_dir:
            self.auto_papers_dir = _join_data_path("auto_papers")
        if not self.auto_papers_archive_dir:
            self.auto_papers_archive_dir = _join_data_path("auto_papers", "archive")
        if not self.auto_research_metadata_file:
            self.auto_research_metadata_file = _join_data_path("auto_research_metadata.json")
        if not self.auto_research_stats_file:
            self.auto_research_stats_file = _join_data_path("auto_research_stats.json")
        if not self.auto_workflow_state_file:
            self.auto_workflow_state_file = _join_data_path("auto_workflow_state.json")
        if not self.auto_research_topic_rejections_file:
            self.auto_research_topic_rejections_file = _join_data_path("auto_research_topic_rejections.txt")
        if not self.auto_sessions_base_dir:
            self.auto_sessions_base_dir = _join_data_path("auto_sessions")
        if not self.lean4_workspace_dir:
            self.lean4_workspace_dir = _join_data_path("lean4_workspace")

        if self.secret_namespace is not None:
            self.secret_namespace = self.secret_namespace.strip() or None

        if self.internal_proxy_secret is not None:
            self.internal_proxy_secret = self.internal_proxy_secret.strip() or None

        if self.frontend_storage_prefix is not None:
            self.frontend_storage_prefix = self.frontend_storage_prefix.strip() or None

        return self


# Global configuration instances
rag_config = RAGConfig()
system_config = SystemConfig()

