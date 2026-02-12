"""
Paper Model Tracker - Per-paper API call tracking for author attribution.

Tracks which AI models made API calls during the creation of each paper,
including both brainstorm aggregation (Tier 1) and paper compilation (Tier 2).

This enables per-paper author attribution and model credits, separate from
the global Tier 3 final answer tracking.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class PaperModelTracker:
    """
    Tracks API calls for a single paper from brainstorm through compilation.
    
    Used to generate author attribution headers and model credits for each paper.
    The same model used in multiple instances counts as ONE author,
    but all API calls are still tallied.
    """
    
    def __init__(self, user_prompt: str = "", paper_title: str = ""):
        """
        Initialize a new paper model tracker.
        
        Args:
            user_prompt: The user's research prompt (for attribution)
            paper_title: The paper title (for attribution)
        """
        # model_id -> API call count
        self._models: Dict[str, int] = {}
        self._user_prompt = user_prompt
        self._paper_title = paper_title
        self._generation_date = datetime.now()
        self._total_calls = 0
        
        # Wolfram Alpha tracking
        self._wolfram_calls: int = 0
    
    @property
    def user_prompt(self) -> str:
        return self._user_prompt
    
    @user_prompt.setter
    def user_prompt(self, value: str) -> None:
        self._user_prompt = value
    
    @property
    def paper_title(self) -> str:
        return self._paper_title
    
    @paper_title.setter
    def paper_title(self, value: str) -> None:
        self._paper_title = value
    
    @property
    def generation_date(self) -> datetime:
        return self._generation_date
    
    @property
    def total_calls(self) -> int:
        return self._total_calls
    
    def track_call(self, model_id: str) -> None:
        """
        Record an API call for a model.
        
        Args:
            model_id: The model identifier (e.g., "deepseek-r1:70b")
        """
        if model_id not in self._models:
            self._models[model_id] = 0
        self._models[model_id] += 1
        self._total_calls += 1
        logger.debug(f"Paper tracker: {model_id} call #{self._models[model_id]} (total: {self._total_calls})")
    
    def get_models_dict(self) -> Dict[str, int]:
        """
        Get the model usage dictionary for metadata storage.
        
        Returns:
            Dict mapping model_id to API call count
        """
        return dict(self._models)
    
    def get_author_list(self) -> List[str]:
        """
        Get list of unique model IDs (authors).
        
        Returns:
            List of model IDs used for this paper
        """
        return list(self._models.keys())
    
    def get_models_by_usage(self) -> List[tuple]:
        """
        Get models sorted by API call count (descending).
        
        Returns:
            List of (model_id, call_count) tuples sorted by count
        """
        return sorted(
            self._models.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def has_tracking_data(self) -> bool:
        """Check if any model calls or Wolfram calls have been tracked."""
        return len(self._models) > 0 or self._wolfram_calls > 0
    
    def track_wolfram_call(self, query: str) -> None:
        """
        Record a Wolfram Alpha API call.
        
        Args:
            query: The Wolfram Alpha query (stored for logging, not tracking)
        """
        self._wolfram_calls += 1
        logger.debug(f"Paper tracker: Wolfram Alpha call #{self._wolfram_calls}")
    
    def get_wolfram_call_count(self) -> int:
        """
        Get total Wolfram Alpha API calls.
        
        Returns:
            Number of Wolfram Alpha queries made
        """
        return self._wolfram_calls
    
    def generate_author_attribution(
        self,
        user_prompt: Optional[str] = None,
        paper_title: Optional[str] = None,
        reference_paper_models: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Generate the author attribution section text for the beginning of a paper.
        
        Args:
            user_prompt: Override the stored user prompt (optional)
            paper_title: Override the stored paper title (optional)
            reference_paper_models: Dict of model_id -> count from reference papers (optional)
        
        Returns:
            Formatted author attribution text
        """
        prompt = user_prompt or self._user_prompt
        title = paper_title or self._paper_title
        
        # Get unique authors (model IDs without call counts)
        authors = self.get_author_list()
        
        # Handle case where no tracking data exists
        if not authors:
            author_list = "Model data not available"
        else:
            author_list = ", ".join(authors)
        
        # Format the date
        gen_date = self._generation_date.strftime("%Y-%m-%d")
        
        # Truncate prompt for attribution header to prevent embedding entire uploaded papers.
        # The full prompt is preserved in session_metadata.json for reference.
        MAX_PROMPT_LENGTH = 500
        display_prompt = prompt
        if len(prompt) > MAX_PROMPT_LENGTH:
            display_prompt = prompt[:MAX_PROMPT_LENGTH].rstrip() + "... [truncated]"
        
        # Build the attribution section
        lines = [
            "=" * 80,
            "AUTONOMOUS AI SOLUTION",
            "",
            "Disclaimer: This is an autonomous AI solution generated with the MOTO harness. "
            "This paper was not peer reviewed and was autonomously generated without user "
            "oversight or interaction beyond the original user prompt, therefore, this text "
            "may contain errors. These papers often contain ambitious content and/or "
            "extraordinary claims, all content should be viewed with extreme scrutiny.",
            "",
            f"User's Research Prompt: {display_prompt}",
            "",
            f"Paper Title: {title}",
            "",
            f"AI Model Authors: {author_list}",
        ]
        
        # Add reference models section if provided
        if reference_paper_models and len(reference_paper_models) > 0:
            lines.append("")
            lines.append("Possible Models Used for Additional Reference:")
            reference_section = self.format_reference_models(reference_paper_models)
            lines.append(reference_section)
        
        lines.extend([
            "",
            f"Generated: {gen_date}",
            "=" * 80,
            ""
        ])
        
        return "\n".join(lines)
    
    def generate_model_credits(self) -> str:
        """
        Generate the model credits section text for the end of a paper.
        
        Returns:
            Formatted model credits text, or empty string if no tracking data
        """
        if not self.has_tracking_data():
            return ""
        
        # Get models sorted by usage (descending)
        models_by_usage = self.get_models_by_usage()
        
        # Build the credits section
        lines = [
            "",
            "=" * 80,
            "MODEL CREDITS",
            "",
            "This autonomous solution attempt was generated with the Intrafere LLC AI Harness,",
            "MOTO, and the following model(s):",
            ""
        ]
        
        # Add each model with its call count (if any models tracked)
        if models_by_usage:
            for model_id, call_count in models_by_usage:
                lines.append(f"- {model_id} ({call_count} API calls)")
            
            lines.extend([
                "",
                f"Total AI Model API Calls: {self._total_calls}"
            ])
        else:
            # No model calls tracked, but Wolfram calls exist
            lines.append("(No AI model API calls tracked)")
        
        # Add Wolfram Alpha section if any calls were made
        if self._wolfram_calls > 0:
            lines.extend([
                "",
                f"Wolfram Alpha Verifications: {self._wolfram_calls} queries"
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_reference_models(reference_paper_models: Dict[str, int]) -> str:
        """
        Format reference paper models in alphabetical order with duplicate counts.
        
        Args:
            reference_paper_models: Dict mapping model_id -> count (duplicates across papers)
        
        Returns:
            Formatted string with models in alphabetical order, e.g.:
            "- DeepSeek (3)
             - Llama-3.1 (2)
             - Qwen-2.5 (1)"
        """
        if not reference_paper_models:
            return ""
        
        # Sort alphabetically (case-insensitive)
        sorted_models = sorted(
            reference_paper_models.items(),
            key=lambda x: x[0].lower()
        )
        
        lines = []
        for model_id, count in sorted_models:
            if count > 1:
                lines.append(f"- {model_id} ({count})")
            else:
                lines.append(f"- {model_id}")
        
        return "\n".join(lines)
    
    @staticmethod
    def aggregate_reference_models(paper_model_usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
        """
        Aggregate model usage from multiple reference papers.
        
        Counts how many papers each model appears in (not total API calls).
        This shows "possible models" since we don't know exactly which content
        from each paper was used.
        
        Args:
            paper_model_usages: List of model_usage dicts from reference papers
        
        Returns:
            Dict mapping model_id -> number of papers that model appears in
        """
        model_counts: Counter = Counter()
        
        for model_usage in paper_model_usages:
            if model_usage:
                # Count each unique model once per paper
                for model_id in model_usage.keys():
                    model_counts[model_id] += 1
        
        return dict(model_counts)
    
    def reset(self) -> None:
        """Reset the tracker for a new paper."""
        self._models.clear()
        self._total_calls = 0
        self._wolfram_calls = 0
        self._generation_date = datetime.now()
        self._user_prompt = ""
        self._paper_title = ""
        logger.debug("Paper model tracker reset")


def generate_attribution_for_existing_paper(
    user_prompt: str,
    paper_title: str,
    model_usage: Optional[Dict[str, int]],
    generation_date: Optional[datetime],
    reference_paper_models: Optional[Dict[str, int]] = None
) -> str:
    """
    Generate author attribution for an existing paper with stored model data.
    
    Used when loading papers that already have model_usage stored in metadata.
    
    Args:
        user_prompt: The user's research prompt
        paper_title: The paper title
        model_usage: Dict of model_id -> API call count (may be None)
        generation_date: When the paper was generated (may be None)
        reference_paper_models: Dict of model_id -> count from reference papers
    
    Returns:
        Formatted author attribution text
    """
    # Create a temporary tracker with the stored data
    tracker = PaperModelTracker(user_prompt, paper_title)
    
    if generation_date:
        tracker._generation_date = generation_date
    
    if model_usage:
        tracker._models = dict(model_usage)
        tracker._total_calls = sum(model_usage.values())
    
    return tracker.generate_author_attribution(
        reference_paper_models=reference_paper_models
    )


def generate_credits_for_existing_paper(
    model_usage: Optional[Dict[str, int]],
    wolfram_calls: Optional[int] = None
) -> str:
    """
    Generate model credits for an existing paper with stored model data.
    
    Args:
        model_usage: Dict of model_id -> API call count (may be None)
        wolfram_calls: Number of Wolfram Alpha verifications (may be None)
    
    Returns:
        Formatted model credits text, or empty string if no data
    """
    # Return empty if no tracking data at all
    if not model_usage and not wolfram_calls:
        return ""
    
    # Create a temporary tracker with the stored data
    tracker = PaperModelTracker()
    if model_usage:
        tracker._models = dict(model_usage)
        tracker._total_calls = sum(model_usage.values())
    tracker._wolfram_calls = wolfram_calls or 0
    
    return tracker.generate_model_credits()

