from backend.shared.models import FreeModelSettings, TopicSelectionSubmission


def test_topic_selection_accepts_new_topic_action() -> None:
    submission = TopicSelectionSubmission(
        action="new_topic",
        topic_prompt="Attack the next direct route to the user's goal",
        reasoning="This opens a distinct useful avenue.",
    )

    assert submission.action == "new_topic"
    assert submission.topic_prompt


def test_topic_selection_accepts_continue_existing_action() -> None:
    submission = TopicSelectionSubmission(
        action="continue_existing",
        topic_id="topic_003",
        reasoning="The existing brainstorm is incomplete and still valuable.",
    )

    assert submission.action == "continue_existing"
    assert submission.topic_id == "topic_003"


def test_free_model_settings_default_fallback_controls_disabled() -> None:
    settings = FreeModelSettings()

    assert settings.looping_enabled is False
    assert settings.auto_selector_enabled is False
