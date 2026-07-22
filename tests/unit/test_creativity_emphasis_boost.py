import unittest

from backend.aggregator.agents.submitter import SubmitterAgent
from backend.aggregator.prompts.submitter_prompts import build_submitter_prompt
from backend.leanoj.prompts import build_brainstorm_prompt, build_topic_candidate_prompt
from backend.shared.models import (
    AggregatorStartRequest,
    AutonomousResearchStartRequest,
    LeanOJRoleConfig,
    LeanOJStartRequest,
    SubmitterConfig,
)


class CreativityEmphasisBoostTests(unittest.TestCase):
    def test_start_requests_default_creativity_boost_off(self) -> None:
        submitter = SubmitterConfig(submitter_id=1, model_id="model-a")
        aggregator = AggregatorStartRequest(
            user_prompt="solve the problem",
            submitter_configs=[submitter],
            validator_model="validator",
        )
        autonomous = AutonomousResearchStartRequest(
            user_research_prompt="advance the problem",
            submitter_configs=[submitter],
            validator_model="validator",
        )
        role = LeanOJRoleConfig(model_id="model-a")
        leanoj = LeanOJStartRequest(
            user_prompt="prove the theorem",
            lean_template="example : True := by trivial",
            topic_generator=role,
            topic_validator=role,
            brainstorm_submitters=[role],
            brainstorm_validator=role,
            final_solver=role,
        )

        self.assertFalse(aggregator.creativity_emphasis_boost_enabled)
        self.assertFalse(autonomous.creativity_emphasis_boost_enabled)
        self.assertFalse(leanoj.creativity_emphasis_boost_enabled)

    def test_aggregator_prompt_adds_creativity_block_only_when_enabled(self) -> None:
        normal_prompt = build_submitter_prompt("user", "context")
        creative_prompt = build_submitter_prompt("user", "context", creativity_emphasized=True)

        self.assertNotIn("CREATIVITY EMPHASIS BOOST", normal_prompt)
        self.assertIn("CREATIVITY EMPHASIS BOOST", creative_prompt)
        self.assertIn("potentially very helpful", creative_prompt)

    def test_aggregator_prompt_hides_lean_proof_schema_when_disabled(self) -> None:
        disabled_prompt = build_submitter_prompt("user", "context", lean4_enabled=False)
        enabled_prompt = build_submitter_prompt("user", "context", lean4_enabled=True)

        self.assertNotIn("lean_proof", disabled_prompt)
        self.assertNotIn("Lean proof candidate", disabled_prompt)
        self.assertIn("lean_proof", enabled_prompt)
        self.assertIn("Lean proof candidate", enabled_prompt)

    def test_brainstorm_submitter_prompts_keep_direct_novelty_pressure(self) -> None:
        aggregator_prompt = build_submitter_prompt("user", "context")
        leanoj_prompt = build_brainstorm_prompt("prompt", "template", "topic", [], [], [])

        self.assertIn(
            "YOUR TASK:\nAggressively pursue the strongest credible and genuinely novel solution "
            "to the user's exact objective.",
            aggregator_prompt,
        )
        self.assertIn("WHOLE question", aggregator_prompt)
        expected_leanoj_line = "Generate a novel mathematical insight that advances the user's goal."
        self.assertIn(f"YOUR TASK:\n{expected_leanoj_line}\nGenerate one concrete idea", leanoj_prompt)

    def test_aggregator_submitter_uses_every_fifth_valid_submission_slot(self) -> None:
        submitter = SubmitterAgent(
            submitter_id=1,
            model_name="model-a",
            user_prompt="solve the problem",
            user_files_content={},
            creativity_emphasis_boost_enabled=True,
        )

        self.assertFalse(submitter._should_use_creativity_emphasis())
        submitter.state.total_submissions = 4
        self.assertTrue(submitter._should_use_creativity_emphasis())
        submitter.state.total_submissions = 5
        self.assertFalse(submitter._should_use_creativity_emphasis())

    def test_leanoj_prompts_add_creativity_block_only_when_enabled(self) -> None:
        normal_topic = build_topic_candidate_prompt("prompt", "template", [])
        creative_topic = build_topic_candidate_prompt(
            "prompt",
            "template",
            [],
            creativity_emphasized=True,
        )
        normal_brainstorm = build_brainstorm_prompt("prompt", "template", "topic", [], [], [])
        creative_brainstorm = build_brainstorm_prompt(
            "prompt",
            "template",
            "topic",
            [],
            [],
            [],
            creativity_emphasized=True,
        )

        self.assertNotIn("CREATIVITY EMPHASIS BOOST", normal_topic)
        self.assertIn("CREATIVITY EMPHASIS BOOST", creative_topic)
        self.assertNotIn("CREATIVITY EMPHASIS BOOST", normal_brainstorm)
        self.assertIn("CREATIVITY EMPHASIS BOOST", creative_brainstorm)


if __name__ == "__main__":
    unittest.main()
