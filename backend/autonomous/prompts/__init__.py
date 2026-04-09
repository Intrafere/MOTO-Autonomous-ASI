"""
Autonomous Prompts - System prompts and JSON schemas for autonomous research.
"""
from backend.autonomous.prompts import topic_prompts
from backend.autonomous.prompts import topic_exploration_prompts
from backend.autonomous.prompts import completion_prompts
from backend.autonomous.prompts import paper_reference_prompts
from backend.autonomous.prompts import paper_title_exploration_prompts
from backend.autonomous.prompts import paper_title_prompts
from backend.autonomous.prompts import paper_redundancy_prompts
from backend.autonomous.prompts import paper_continuation_prompts

__all__ = [
    'topic_prompts',
    'topic_exploration_prompts',
    'completion_prompts',
    'paper_reference_prompts',
    'paper_title_exploration_prompts',
    'paper_title_prompts',
    'paper_redundancy_prompts',
    'paper_continuation_prompts'
]
