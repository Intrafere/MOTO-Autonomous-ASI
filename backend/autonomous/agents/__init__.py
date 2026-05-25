"""
Autonomous Agents - Topic selection, completion review, reference selection, and title selection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.autonomous.agents.topic_selector import TopicSelectorAgent
    from backend.autonomous.agents.topic_validator import TopicValidatorAgent
    from backend.autonomous.agents.completion_reviewer import CompletionReviewerAgent
    from backend.autonomous.agents.reference_selector import ReferenceSelectorAgent
    from backend.autonomous.agents.paper_title_selector import PaperTitleSelectorAgent

_AGENT_EXPORTS = {
    "TopicSelectorAgent": ("backend.autonomous.agents.topic_selector", "TopicSelectorAgent"),
    "TopicValidatorAgent": ("backend.autonomous.agents.topic_validator", "TopicValidatorAgent"),
    "CompletionReviewerAgent": ("backend.autonomous.agents.completion_reviewer", "CompletionReviewerAgent"),
    "ReferenceSelectorAgent": ("backend.autonomous.agents.reference_selector", "ReferenceSelectorAgent"),
    "PaperTitleSelectorAgent": ("backend.autonomous.agents.paper_title_selector", "PaperTitleSelectorAgent"),
}


def __getattr__(name: str):
    if name not in _AGENT_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _AGENT_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    'TopicSelectorAgent',
    'TopicValidatorAgent',
    'CompletionReviewerAgent',
    'ReferenceSelectorAgent',
    'PaperTitleSelectorAgent'
]
