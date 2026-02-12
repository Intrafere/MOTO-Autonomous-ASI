"""
Configuration for the ASI Aggregator System.
Defines RAG parameters, context allocation, and system constants.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """RAG system configuration."""
    
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
    lm_studio_base_url: str = "http://127.0.0.1:1234"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    
    # OpenRouter API (Global Configuration)
    # This is the global API key used for per-role OpenRouter model selection
    # Separate from boost API key which is stored in BoostConfig
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
    
    # Aggregator submitter settings (configurable 1-10 submitters)
    default_num_submitters: int = 3  # Default number of submitters
    max_submitters: int = 10  # Hard cap on submitters
    min_submitters: int = 1  # Minimum submitters
    consecutive_rejection_reset_threshold: int = 15
    queue_overflow_threshold: int = 10
    
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
    autonomous_max_reference_papers: int = 6  # Max papers for reference context
    autonomous_topic_selection_retry_limit: int = 3
    
    # Wolfram Alpha integration (optional)
    wolfram_alpha_enabled: bool = False
    wolfram_alpha_api_key: Optional[str] = None
    
    # File paths
    data_dir: str = "backend/data"
    logs_dir: str = "backend/logs"
    user_uploads_dir: str = "backend/data/user_uploads"
    chroma_db_dir: str = "backend/data/chroma_db"
    
    shared_training_file: str = "backend/data/rag_shared_training.txt"
    compiler_outline_file: str = "backend/data/compiler_outline.txt"
    compiler_paper_file: str = "backend/data/compiler_paper.txt"
    compiler_rejections_file: str = "backend/data/compiler_last_10_rejections.txt"
    compiler_acceptances_file: str = "backend/data/compiler_last_10_acceptances.txt"
    compiler_declines_file: str = "backend/data/compiler_last_10_declines.txt"
    
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
    auto_brainstorms_dir: str = "backend/data/auto_brainstorms"
    auto_papers_dir: str = "backend/data/auto_papers"
    auto_papers_archive_dir: str = "backend/data/auto_papers/archive"
    auto_research_metadata_file: str = "backend/data/auto_research_metadata.json"
    auto_research_stats_file: str = "backend/data/auto_research_stats.json"
    auto_workflow_state_file: str = "backend/data/auto_workflow_state.json"
    auto_research_topic_rejections_file: str = "backend/data/auto_research_topic_rejections.txt"
    
    # Session-based organization (preferred for new features)
    auto_sessions_base_dir: str = "backend/data/auto_sessions"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instances
rag_config = RAGConfig()
system_config = SystemConfig()

