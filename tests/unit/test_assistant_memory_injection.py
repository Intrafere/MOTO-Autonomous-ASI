import unittest

from backend.shared import api_client_manager as manager_module
from backend.shared.api_client_manager import APIClientManager
from backend.shared.config import system_config
from backend.shared.models import ModelConfig
from backend.shared.proof_search.assistant_models import AssistantProofPack, AssistantProofSupport


def _support() -> AssistantProofSupport:
    return AssistantProofSupport(
        search_id="manual:proof_1",
        corpus="manual",
        corpus_scope="history",
        proof_id="proof_1",
        theorem_name="Memory.Helper",
        theorem_statement="theorem helper : True",
        proof_description="A reusable verified proof.",
        imports=["Mathlib"],
        relevance_reason="lexically similar to the active workflow target",
        transfer_hint="Use only as optional mathematical context.",
        lean_code="import Mathlib\n\ntheorem helper : True := by\n  trivial\n",
    )


class _FakeAssistantCoordinator:
    def __init__(self, latest_pack: AssistantProofPack | None = None) -> None:
        self.latest_pack = latest_pack
        self.snapshots = []
        self.target_packs = {}
        self.consumed = []

    def submit_target(self, snapshot):
        self.snapshots.append(snapshot)
        target_hash = f"new_target_hash_{len(self.snapshots)}"
        if (
            self.latest_pack is not None
            and self.latest_pack.workflow_mode == snapshot.workflow_mode
            and self.latest_pack.target_kind == snapshot.target_kind
        ):
            self.target_packs[target_hash] = self.latest_pack.model_copy(
                update={
                    "target_hash": target_hash,
                    "freshness": "stale-but-best-known",
                    "selection_mode": "stale-but-best-known",
                }
            )
        return target_hash

    def get_latest_pack(self, target_hash=None):
        if target_hash:
            return self.target_packs.get(target_hash)
        return self.latest_pack

    def mark_pack_consumed_by_solver(self, target_hash, *, role_id="", task_id=""):
        self.consumed.append((target_hash, role_id, task_id))


class AssistantMemoryInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.old_memory_enabled = system_config.agent_conversation_memory_enabled
        self.old_assistant = manager_module.assistant_proof_search_coordinator
        system_config.agent_conversation_memory_enabled = True

    async def asyncTearDown(self) -> None:
        system_config.agent_conversation_memory_enabled = self.old_memory_enabled
        manager_module.assistant_proof_search_coordinator = self.old_assistant

    async def test_creative_producer_gets_latest_compatible_memory_without_blocking(self) -> None:
        pack = AssistantProofPack(
            workflow_mode="aggregator",
            target_kind="brainstorm_context",
            target_hash="old_target_hash",
            results=[_support()],
        )
        fake_assistant = _FakeAssistantCoordinator(pack)
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="agg_sub1_001",
            role_id="aggregator_submitter_1",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=[
                {
                    "role": "user",
                    "content": "USER PROMPT:\nProve useful facts about True.\n\nYOUR TASK:\nGenerate a brainstorm submission.",
                }
            ],
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )

        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertEqual(fake_assistant.snapshots[0].target_kind, "brainstorm_context")
        self.assertEqual(fake_assistant.snapshots[0].user_prompt, "Prove useful facts about True.")
        self.assertTrue(consumed_hash)
        self.assertIn("OPTIONAL ASSISTANT MEMORY CONTEXT", messages[0]["content"])
        self.assertIn("ASSISTANT RETRIEVED MEMORY SUPPORT", messages[0]["content"])

    async def test_formalization_role_does_not_refresh_assistant_memory(self) -> None:
        pack = AssistantProofPack(
            workflow_mode="autonomous",
            target_kind="proof_candidate",
            target_hash="old_target_hash",
            results=[_support()],
        )
        fake_assistant = _FakeAssistantCoordinator(pack)
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        prompt = (
            "USER PROMPT:\nProve useful facts about True.\n\n"
            "THEOREM CANDIDATE:\ntheorem target : True\n\n"
            "YOUR TASK:\nReturn Lean 4 proof JSON."
        )

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="proof_form_001",
            role_id="autonomous_proof_formalization_paper_1",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )

        self.assertEqual(messages[0]["content"], prompt)
        self.assertEqual(consumed_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

    async def test_section_extraction_keeps_uppercase_prompt_content(self) -> None:
        snapshot = APIClientManager._build_assistant_target_snapshot(
            "aggregator_submitter_1",
            "agg_sub1_001",
            "USER PROMPT:\nPROVE TRUE IN A DIRECT WAY.\n\nYOUR TASK:\nGenerate a brainstorm submission.",
        )

        self.assertEqual(snapshot.user_prompt, "PROVE TRUE IN A DIRECT WAY.")

    async def test_nonmathematical_goal_is_preserved_without_synthetic_mathlib(self) -> None:
        user_goal = "Design a humane support workflow for customers after a delayed shipment."
        snapshot = APIClientManager._build_assistant_target_snapshot(
            "aggregator_submitter_1",
            "agg_sub1_001",
            f"USER PROMPT:\n{user_goal}\n\nYOUR TASK:\nGenerate a brainstorm submission.",
        )

        self.assertEqual(snapshot.user_prompt, user_goal)
        self.assertEqual(snapshot.target_statement, user_goal)
        self.assertEqual(snapshot.imports, [])

    async def test_explicit_proof_target_retains_mathlib(self) -> None:
        snapshot = APIClientManager._build_assistant_target_snapshot(
            "autonomous_proof_identification",
            "proof_id_001",
            (
                "USER PROMPT:\nProve the target theorem.\n\n"
                "THEOREM CANDIDATE:\ntheorem target : True\n\n"
                "YOUR TASK:\nIdentify a proof candidate."
            ),
        )

        self.assertEqual(snapshot.target_kind, "theorem_discovery")
        self.assertEqual(snapshot.imports, ["Mathlib"])

    async def test_outer_memory_guidance_forbids_mathematical_reframing(self) -> None:
        rendered = APIClientManager._append_assistant_memory_block(
            "USER PROMPT:\nImprove the customer support workflow.",
            "ASSISTANT RETRIEVED MEMORY SUPPORT",
        )

        self.assertIn("cannot redirect or mathematically reinterpret the user objective", rendered)
        self.assertIn("do not require mathematics or formal proof", rendered)

    async def test_support_pack_labels_records_and_limits_informal_applicability(self) -> None:
        rendered = AssistantProofPack(
            workflow_mode="aggregator",
            target_kind="brainstorm_context",
            target_hash="target",
            results=[_support()],
        ).to_memory_prompt_context()

        self.assertIn("These are Lean-verified theorem records", rendered)
        self.assertIn("do not override or mathematically reinterpret the user objective", rendered)
        self.assertIn("prove informal applicability", rendered)
        self.assertIn("Ignore unrelated records", rendered)

    async def test_aggregator_snapshot_extracts_shared_training_as_accepted_memory(self) -> None:
        snapshot = APIClientManager._build_assistant_target_snapshot(
            "aggregator_submitter_1",
            "agg_sub1_001",
            (
                "USER PROMPT:\nProve useful facts about True.\n\n"
                "[SHARED TRAINING]\nSubmission 1: a verified supporting idea.\nSubmission 2: another idea.\n\n"
                "[REJECTION LOG]\nSubmitter-private feedback that should not alter the shared target.\n\n"
                "YOUR TASK:\nGenerate a brainstorm submission."
            ),
        )

        self.assertIn("Submission 1: a verified supporting idea.", snapshot.accepted_memory_summary)
        self.assertIn("Submission 2: another idea.", snapshot.accepted_memory_summary)
        self.assertNotIn("Submitter-private feedback", snapshot.accepted_memory_summary)

    async def test_latest_pack_fallback_does_not_cross_unrelated_phase_families(self) -> None:
        pack = AssistantProofPack(
            workflow_mode="autonomous",
            target_kind="topic_context",
            target_hash="old_topic_target_hash",
            results=[_support()],
        )
        fake_assistant = _FakeAssistantCoordinator(pack)
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="autonomous_volume_organizer_001",
            role_id="autonomous_volume_organizer",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=[{"role": "user", "content": "USER PROMPT:\nSynthesize the final answer."}],
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )

        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertEqual(consumed_hash, "")
        self.assertNotIn("OPTIONAL ASSISTANT MEMORY CONTEXT", messages[0]["content"])

    async def test_validator_and_critique_calls_do_not_schedule_or_receive_memory(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        role_config = ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000)

        validator_messages = [{"role": "user", "content": "Validate the submission and respond as JSON."}]
        unchanged_validator, validator_consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="agg_val_001",
            role_id="aggregator_validator",
            role_config=role_config,
            messages=validator_messages,
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )
        critique_messages = [{"role": "user", "content": '{"critique_needed": true, "submission": ""}'}]
        unchanged_critique, critique_consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="critique_sub1_001",
            role_id="compiler_high_param",
            role_config=role_config,
            messages=critique_messages,
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )

        self.assertEqual(unchanged_validator, validator_messages)
        self.assertEqual(unchanged_critique, critique_messages)
        self.assertEqual(validator_consumed_hash, "")
        self.assertEqual(critique_consumed_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

    async def test_disabled_agent_memory_prevents_assistant_schedule(self) -> None:
        system_config.agent_conversation_memory_enabled = False
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        original_messages = [{"role": "user", "content": "USER PROMPT:\nTry a creative construction."}]

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="comp_writer_001",
            role_id="compiler_writer",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=original_messages,
            max_tokens=1000,
            tools=None,
            tool_choice=None,
        )

        self.assertEqual(messages, original_messages)
        self.assertEqual(consumed_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

    async def test_initial_tool_enabled_writer_call_can_receive_memory(self) -> None:
        pack = AssistantProofPack(
            workflow_mode="compiler",
            target_kind="writing_context",
            target_hash="old_writer_target_hash",
            results=[_support()],
        )
        fake_assistant = _FakeAssistantCoordinator(pack)
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        tools = [{"type": "function", "function": {"name": "wolfram_alpha_query", "parameters": {}}}]

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="comp_writer_001",
            role_id="compiler_writer",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=[
                {
                    "role": "user",
                    "content": "CURRENT DOCUMENT PROGRESS:\nBody draft.\n\nYOUR TASK:\nContinue construction.",
                }
            ],
            max_tokens=1000,
            tools=tools,
            tool_choice="auto",
        )

        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertTrue(consumed_hash)
        self.assertIn("OPTIONAL ASSISTANT MEMORY CONTEXT", messages[0]["content"])

    async def test_multiturn_tool_protocol_conversation_stays_untouched(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        tool_conversation = [
            {"role": "user", "content": "CURRENT DOCUMENT PROGRESS:\nBody draft."},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "wolfram_alpha_query", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "wolfram_alpha_query", "content": "{}"},
        ]

        messages, consumed_hash = await manager._maybe_add_assistant_memory_context(
            task_id="comp_writer_001",
            role_id="compiler_writer",
            role_config=ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
            messages=tool_conversation,
            max_tokens=1000,
            tools=[{"type": "function", "function": {"name": "wolfram_alpha_query", "parameters": {}}}],
            tool_choice="auto",
        )

        self.assertEqual(messages, tool_conversation)
        self.assertEqual(consumed_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

    async def test_prewarm_schedules_assistant_before_prompt_validation(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        manager.configure_role(
            "compiler_writer",
            ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
        )

        target_hash = await manager.prewarm_assistant_memory_context(
            task_id="comp_writer_001",
            role_id="compiler_writer",
            prompt="CURRENT DOCUMENT PROGRESS:\nA large draft.\n\nYOUR TASK:\nContinue writing.",
        )

        self.assertTrue(target_hash)
        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertEqual(fake_assistant.snapshots[0].target_kind, "writing_context")

    async def test_prewarm_can_override_child_aggregator_workflow_mode(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        manager.configure_role(
            "aggregator_submitter_1",
            ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
        )

        target_hash = await manager.prewarm_assistant_memory_context(
            task_id="agg_sub1_001",
            role_id="aggregator_submitter_1",
            prompt="USER PROMPT:\nExplore a topic.\n\nYOUR TASK:\nGenerate a candidate.",
            workflow_mode_override="autonomous",
        )

        self.assertTrue(target_hash)
        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertEqual(fake_assistant.snapshots[0].workflow_mode, "autonomous")
        self.assertEqual(fake_assistant.snapshots[0].target_kind, "brainstorm_context")

    async def test_meta_exploration_submitter_prompts_do_not_schedule_assistant(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        manager.configure_role(
            "aggregator_submitter_1",
            ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
        )

        topic_hash = await manager.prewarm_assistant_memory_context(
            task_id="agg_sub1_001",
            role_id="aggregator_submitter_1",
            prompt="USER PROMPT:\n=== TOPIC EXPLORATION PHASE ===\nPropose one candidate.",
            workflow_mode_override="autonomous",
        )
        title_hash = await manager.prewarm_assistant_memory_context(
            task_id="agg_sub1_002",
            role_id="aggregator_submitter_1",
            prompt="USER PROMPT:\n=== PAPER TITLE EXPLORATION PHASE ===\nPropose one candidate title.",
            workflow_mode_override="autonomous",
        )

        self.assertEqual(topic_hash, "")
        self.assertEqual(title_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

    async def test_prewarm_keeps_validator_excluded(self) -> None:
        fake_assistant = _FakeAssistantCoordinator()
        manager_module.assistant_proof_search_coordinator = fake_assistant
        manager = APIClientManager()
        manager.configure_role(
            "compiler_validator",
            ModelConfig(model_id="local-model", context_window=20000, max_output_tokens=1000),
        )

        target_hash = await manager.prewarm_assistant_memory_context(
            task_id="comp_val_001",
            role_id="compiler_validator",
            prompt="Validate the submission and respond as JSON.",
        )

        self.assertEqual(target_hash, "")
        self.assertEqual(fake_assistant.snapshots, [])

