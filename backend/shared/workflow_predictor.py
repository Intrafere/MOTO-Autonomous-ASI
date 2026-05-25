"""
Workflow Predictor - Predicts next 20 API calls based on current workflow state.
Supports Aggregator, Compiler, and Autonomous Research modes.
"""
import logging
from typing import List, Dict, Optional

from backend.shared.models import WorkflowTask

logger = logging.getLogger(__name__)


class WorkflowPredictor:
    """Predicts upcoming API calls based on workflow state."""
    
    @staticmethod
    def predict_aggregator_workflow(
        num_submitters: int,
        single_model_mode: bool,
        current_sequence: int = 0,
        submitter_sequences: Optional[Dict[int, int]] = None,
        validator_sequence: int = 0
    ) -> List[WorkflowTask]:
        """
        Predict next 20 API calls for Aggregator workflow.
        
        Args:
            num_submitters: Number of submitters (1-10)
            single_model_mode: True if all submitters + validator use same model
            current_sequence: Starting sequence number (for display ordering)
            submitter_sequences: Per-submitter task sequence counters {submitter_id: counter}
            validator_sequence: Validator's task sequence counter
            
        Returns:
            List of 20 predicted workflow tasks
        """
        tasks = []
        display_seq = current_sequence
        
        # Initialize per-role sequence counters
        if submitter_sequences is None:
            submitter_sequences = {i: 0 for i in range(1, num_submitters + 1)}
        else:
            # Make a copy to avoid modifying the original
            submitter_sequences = dict(submitter_sequences)
        
        # Make a copy of validator sequence
        val_seq = validator_sequence
        
        if single_model_mode:
            # Sequential mode: S1 → S2 → S3 → V → S1 → S2 → ...
            cycle_length = num_submitters + 1  # submitters + validator
            
            for i in range(20):
                position_in_cycle = i % cycle_length
                
                if position_in_cycle < num_submitters:
                    # Submitter turn
                    submitter_id = position_in_cycle + 1
                    sub_seq = submitter_sequences.get(submitter_id, 0)
                    tasks.append(WorkflowTask(
                        task_id=f"agg_sub{submitter_id}_{sub_seq:03d}",
                        sequence_number=display_seq + 1,
                        role=f"Submitter {submitter_id}" + (" (Main Submitter)" if submitter_id == 1 else ""),
                        mode=None,
                        provider="lm_studio"
                    ))
                    submitter_sequences[submitter_id] = sub_seq + 1
                else:
                    # Validator turn
                    tasks.append(WorkflowTask(
                        task_id=f"agg_val_{val_seq:03d}",
                        sequence_number=display_seq + 1,
                        role="Validator",
                        mode=None,
                        provider="lm_studio"
                    ))
                    val_seq += 1
                
                display_seq += 1
        else:
            # Multi-model mode: S1, S2, S3 (parallel) → V → S1, S2, S3 → V
            for i in range(20):
                position_in_cycle = i % (num_submitters + 1)
                
                if position_in_cycle < num_submitters:
                    # Submitter turn (parallel execution)
                    submitter_id = position_in_cycle + 1
                    sub_seq = submitter_sequences.get(submitter_id, 0)
                    tasks.append(WorkflowTask(
                        task_id=f"agg_sub{submitter_id}_{sub_seq:03d}",
                        sequence_number=display_seq + 1,
                        role=f"Submitter {submitter_id}" + (" (Main Submitter)" if submitter_id == 1 else ""),
                        mode=None,
                        provider="lm_studio"
                    ))
                    submitter_sequences[submitter_id] = sub_seq + 1
                else:
                    # Validator turn (after all submitters)
                    tasks.append(WorkflowTask(
                        task_id=f"agg_val_{val_seq:03d}",
                        sequence_number=display_seq + 1,
                        role="Validator",
                        mode=None,
                        provider="lm_studio"
                    ))
                    val_seq += 1
                
                display_seq += 1
        
        return tasks
    
    @staticmethod
    def predict_compiler_workflow(
        current_mode: str,
        outline_accepted: bool,
        autonomous_section_phase: Optional[str],
        current_sequence: int = 0
    ) -> List[WorkflowTask]:
        """
        Predict next 20 API calls for Compiler workflow.
        
        Args:
            current_mode: Current compiler mode (outline_create, construction, etc.)
            outline_accepted: Whether outline has been accepted
            autonomous_section_phase: Current section phase (body, conclusion, intro, abstract) or None
            current_sequence: Starting sequence number
            
        Returns:
            List of 20 predicted workflow tasks
        """
        tasks = []
        seq = current_sequence
        
        if not outline_accepted:
            # Outline creation phase (iterative): HC → V → HC → V (max 15 iterations)
            for i in range(min(20, 30)):  # 15 iterations max = 30 tasks
                if i % 2 == 0:
                    tasks.append(WorkflowTask(
                        task_id=f"comp_hc_outline_{seq:03d}",
                        sequence_number=seq + 1,
                        role="High-Context",
                        mode="Outline Creation",
                        provider="lm_studio"
                    ))
                else:
                    tasks.append(WorkflowTask(
                        task_id=f"comp_val_outline_{seq:03d}",
                        sequence_number=seq + 1,
                        role="Validator",
                        mode="Outline Review",
                        provider="lm_studio"
                    ))
                seq += 1
                
                if len(tasks) >= 20:
                    break
        else:
            # Paper construction phase - Construction cycle pattern
            # HC(const) → V → HC(const) → V → HC(const) → V → HC(const) → V → 
            # HC(outline) → V → HC(review) → V → HC(review) → V → HP(rigor) → V
            
            cycle_pattern = [
                ("High-Context", "Construction"),
                ("Validator", "Construction Review"),
                ("High-Context", "Construction"),
                ("Validator", "Construction Review"),
                ("High-Context", "Construction"),
                ("Validator", "Construction Review"),
                ("High-Context", "Construction"),
                ("Validator", "Construction Review"),
                ("High-Context", "Outline Update"),
                ("Validator", "Outline Review"),
                ("High-Context", "Paper Review"),
                ("Validator", "Review Validation"),
                ("High-Context", "Paper Review"),
                ("Validator", "Review Validation"),
                ("High-Param", "Rigor Enhancement"),
                ("Validator", "Rigor Review"),
            ]
            
            for i in range(20):
                pattern_idx = i % len(cycle_pattern)
                role, mode = cycle_pattern[pattern_idx]
                
                if role == "High-Context":
                    task_id = f"comp_hc_{seq:03d}"
                elif role == "High-Param":
                    task_id = f"comp_hp_{seq:03d}"
                else:
                    task_id = f"comp_val_{seq:03d}"
                
                tasks.append(WorkflowTask(
                    task_id=task_id,
                    sequence_number=seq + 1,
                    role=role,
                    mode=mode,
                    provider="lm_studio"
                ))
                seq += 1
        
        return tasks[:20]
    
    @staticmethod
    def predict_autonomous_workflow(
        current_tier: str,
        current_phase: Optional[str],
        num_submitters: int,
        single_model_mode: bool,
        current_sequence: int = 0
    ) -> List[WorkflowTask]:
        """
        Predict next 20 API calls for Autonomous Research workflow.
        
        Args:
            current_tier: "tier1_aggregation" or "tier2_paper_writing"
            current_phase: Current paper phase (body_sections, conclusion, etc.) or None
            num_submitters: Number of submitters for brainstorm aggregation
            single_model_mode: True if single-model mode for aggregation
            current_sequence: Starting sequence number
            
        Returns:
            List of 20 predicted workflow tasks
        """
        tasks = []
        seq = current_sequence
        
        if current_tier == "tier1_aggregation":
            # Brainstorm aggregation - Same as Aggregator workflow
            return WorkflowPredictor.predict_aggregator_workflow(
                num_submitters, single_model_mode, current_sequence
            )
        
        elif current_tier == "tier2_paper_writing":
            # Paper compilation - Same as Compiler workflow
            # Note: outline_accepted is True by default for autonomous mode
            return WorkflowPredictor.predict_compiler_workflow(
                current_mode="construction",
                outline_accepted=True,
                autonomous_section_phase=current_phase,
                current_sequence=current_sequence
            )
        
        else:
            # Topic selection or idle
            # Predict generic topic selection workflow
            topic_pattern = [
                ("Topic Selector", "Topic Selection"),
                ("Topic Validator", "Topic Validation"),
            ]
            
            for i in range(20):
                pattern_idx = i % len(topic_pattern)
                role, mode = topic_pattern[pattern_idx]
                
                if "Selector" in role:
                    task_id = f"auto_ts_{seq:03d}"
                else:
                    task_id = f"auto_tv_{seq:03d}"
                
                tasks.append(WorkflowTask(
                    task_id=task_id,
                    sequence_number=seq + 1,
                    role=role,
                    mode=mode,
                    provider="lm_studio"
                ))
                seq += 1
            
            return tasks


# Global instance
workflow_predictor = WorkflowPredictor()

