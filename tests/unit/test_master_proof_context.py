"""Verified master proofs in model context: solved template, expansion, and fit order."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import backend.autonomous.memory.proof_database as proof_database_module
from backend.aggregator.core.rag_manager import RAGManager
from backend.shared import master_proof_context as context
from backend.shared.master_proof_context import (
    MASTER_PROOF_LIST_NOTICE,
    PROOFS_BLOCK_END,
    PROOFS_BLOCK_START,
    ContextPart,
    apply_expansion_result,
    build_review_prompt,
    build_solved_template,
    fit_review_prompt,
    list_declarations,
    lookup_master_proof,
    master_proof_handle,
    note_context_part,
    noted_context_parts,
    offered_handles,
    parse_expansion_request,
    parse_review_selection,
    proofs_list_part,
    render_master_proof_entry,
    select_declarations,
    summary_from_state_payload,
)
from backend.shared.master_proof_expansion import (
    MASTER_PROOF_EXPANSION_EVENT,
    ExpansionCall,
    offers_master_proof_expansion,
    resolve_master_proof_expansions,
)
from backend.shared.models import DocumentChunk, MasterProofStep, MasterProofSummary

MASTER_CODE = """import Mathlib

def weight (n : Nat) : Nat := n + 1

lemma helper (n : Nat) : n = n := by
  rfl

lemma step_one (n : Nat) : n + 0 = n := by
  simp

lemma step_two (n : Nat) :
    0 + n = n := by
  simp

theorem master (n : Nat) : n + 0 = n ∧ 0 + n = n := by
  exact ⟨step_one n, step_two n⟩
"""
SUMMARY = MasterProofSummary(
    master_key="key",
    root_theorem_name="master",
    scaffold_revision=2,
    total_attempts=7,
    steps=[
        MasterProofStep(name="step_one", signature="lemma step_one (n : Nat) : n + 0 = n", failed_attempts=2),
        MasterProofStep(name="step_two", signature="lemma step_two (n : Nat) : 0 + n = n"),
    ],
)
HANDLE = master_proof_handle(MASTER_CODE.strip())


def _response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _text(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def _proofs_block(extra: str = "") -> str:
    entry = render_master_proof_entry(
        heading="PROOF 1 [Mathematical Discovery]",
        proof_id="proof_001",
        theorem_name="master",
        theorem_statement="Both additions with zero are the identity.",
        lean_code=MASTER_CODE.strip(),
        summary=SUMMARY,
        source_line="Source: brainstorm topic_001",
    )
    return "\n".join([PROOFS_BLOCK_START, MASTER_PROOF_LIST_NOTICE, *entry, "---", extra, PROOFS_BLOCK_END])


class SolvedTemplateTests(unittest.TestCase):
    def test_template_shows_steps_as_solved_and_hides_every_proof(self):
        template = build_solved_template(MASTER_CODE, SUMMARY)

        self.assertIn("lemma step_one (n : Nat) : n + 0 = n := by", template)
        self.assertIn("lemma step_two (n : Nat) : 0 + n = n := by", template)
        self.assertEqual(template.count("sorry  -- SOLVED: Lean 4 verified, proof hidden"), 2)
        self.assertIn("theorem master (n : Nat) : n + 0 = n ∧ 0 + n = n := by", template)
        self.assertIn("-- VERIFIED: proved from the solved steps above", template)
        self.assertIn("Hidden: 1 definitions, 1 supporting lemmas", template)
        # No proof body, supporting lemma, or definition leaks into the template.
        for hidden in ("simp", "rfl", "helper", "weight", "exact ⟨"):
            self.assertNotIn(hidden, template)

    def test_template_without_a_recorded_scaffold_treats_every_lemma_as_a_step(self):
        template = build_solved_template(MASTER_CODE, None)
        self.assertEqual(template.count("-- SOLVED"), 3)
        self.assertIn("theorem master", template)

    def test_summary_is_read_from_a_master_proof_state_document(self):
        summary = summary_from_state_payload(
            {
                "master_key": "abc",
                "root_name": "master",
                "scaffold_revision": 3,
                "total_attempts": 9,
                "scaffold_attempts": [{}, {}],
                "nodes": {
                    "step_one": {
                        "name": "step_one",
                        "signature": "lemma step_one : True",
                        "parent": "",
                        "depth": 0,
                        "attempts": [{}, {}, {}],
                    }
                },
            }
        )
        self.assertEqual(summary.root_theorem_name, "master")
        self.assertEqual(summary.total_attempts, 9)
        self.assertEqual(summary.scaffold_failed_attempts, 2)
        self.assertEqual(summary.steps[0].failed_attempts, 3)

    def test_declarations_resolve_by_name_and_by_the_definitions_selector(self):
        self.assertEqual(
            [item.name for item in list_declarations(MASTER_CODE)],
            ["weight", "helper", "step_one", "step_two", "master"],
        )
        found, unknown = select_declarations(MASTER_CODE, ["step_two", "definitions", "missing", "Ns.helper"])
        self.assertEqual([item.name for item in found], ["weight", "helper", "step_two"])
        self.assertEqual(unknown, ["missing"])
        self.assertIn("simp", found[-1].text)


class ExpansionContractTests(unittest.TestCase):
    def test_only_a_bare_request_is_an_expansion_request(self):
        request = parse_expansion_request(json.dumps({"expand_master_proof": HANDLE}))
        self.assertEqual((request.handle, request.declarations), (HANDLE, ()))
        partial = parse_expansion_request(
            json.dumps({"expand_master_proof": HANDLE, "declarations": ["step_one", "definitions"]})
        )
        self.assertEqual(partial.declarations, ("step_one", "definitions"))
        # A role answer that merely mentions the key is the role's answer.
        self.assertIsNone(
            parse_expansion_request(json.dumps({"expand_master_proof": HANDLE, "submission": "text"}))
        )
        self.assertIsNone(parse_expansion_request(json.dumps({"expand_master_proof": "proof_001"})))
        self.assertIsNone(parse_expansion_request('{"decision": "accept"}'))
        self.assertIsNone(parse_expansion_request("not json expand_master_proof"))

    def test_review_reply_contract(self):
        selection = parse_review_selection('{"keep_declarations": ["step_one"], "notes": "reuse"}')
        self.assertEqual((selection.keep, selection.notes), (("step_one",), "reuse"))
        self.assertEqual(parse_review_selection('{"keep_declarations": []}').keep, ())
        with self.assertRaises(ValueError):
            parse_review_selection('{"notes": "no list"}')

    def test_entry_registers_the_proof_and_offers_then_withdraws_expansion(self):
        block = _proofs_block()
        self.assertEqual(offered_handles(block), [HANDLE])
        self.assertEqual(lookup_master_proof(HANDLE).lean_code, MASTER_CODE.strip())
        self.assertNotIn("simp", block)
        self.assertIn(f"(MASTER PROOF, id {HANDLE})", block)
        self.assertTrue(offers_master_proof_expansion([{"role": "user", "content": block}]))

        kept = apply_expansion_result(block, HANDLE, "RESULT", keep_offer=True)
        self.assertEqual(offered_handles(kept), [HANDLE])
        closed = apply_expansion_result(block, HANDLE, "RESULT")
        self.assertEqual(offered_handles(closed), [])
        self.assertIn("RESULT\n[end of master proof", closed)
        self.assertFalse(offers_master_proof_expansion([{"role": "user", "content": closed}]))

    def test_noted_context_is_kept_per_kind_and_ignores_small_text(self):
        note_context_part("papers", "short")
        before = len(noted_context_parts())
        note_context_part("papers", "p" * 400)
        note_context_part("papers", "p" * 400)
        note_context_part("brainstorms", "b" * 400)
        parts = noted_context_parts()
        self.assertEqual(len(parts), before + 2)
        self.assertEqual([part.kind for part in parts[-2:]], ["papers", "brainstorms"])


class ReviewFitTests(unittest.IsolatedAsyncioTestCase):
    """Other context makes room in the order proofs list, papers, brainstorms."""

    PROOFS = PROOFS_BLOCK_START + "\n" + "proofs " * 300 + "\n" + PROOFS_BLOCK_END
    PAPER = "paper " * 300
    BRAINSTORM = "brainstorm " * 300

    def _task(self) -> str:
        return f"ROLE INSTRUCTIONS\n{self.PROOFS}\nPAPER:\n{self.PAPER}\nBRAINSTORM:\n{self.BRAINSTORM}\nEND"

    def _parts(self, task: str) -> list[ContextPart]:
        return [
            proofs_list_part(task),
            ContextPart("brainstorms", self.BRAINSTORM),
            ContextPart("papers", self.PAPER),
        ]

    async def _fit(self, limit: int):
        calls: list[str] = []

        async def retrieve(kind, text, budget):
            calls.append(kind)
            return f"<{kind} excerpts>"

        task = self._task()
        fit = await fit_review_prompt(
            task_text=task,
            parts=self._parts(task),
            build=lambda text: "REVIEW\n" + text + "\nEXPANDED",
            fits=lambda prompt: len(prompt) <= limit,
            count_tokens=len,
            retrieve=retrieve,
        )
        return fit, calls

    async def test_nothing_moves_when_the_review_already_fits(self):
        fit, calls = await self._fit(10**6)
        self.assertEqual((fit.actions, calls), ([], []))
        self.assertIn(self.PROOFS, fit.prompt)

    async def test_the_proofs_list_moves_to_retrieval_first(self):
        fit, calls = await self._fit(len(self._task()) - 1500)
        self.assertEqual((fit.actions, calls), (["proofs:retrieval"], ["proofs"]))
        self.assertNotIn(self.PROOFS, fit.prompt)
        self.assertIn("<proofs excerpts>", fit.prompt)
        self.assertIn(self.PAPER, fit.prompt)
        self.assertIn(self.BRAINSTORM, fit.prompt)

    async def test_papers_then_brainstorms_follow_in_order(self):
        fit, calls = await self._fit(len(self.BRAINSTORM) + 900)
        self.assertEqual(calls, ["proofs", "papers"])
        self.assertEqual(fit.actions, ["proofs:retrieval", "papers:retrieval"])
        self.assertIn(self.BRAINSTORM, fit.prompt)

        fit, calls = await self._fit(700)
        self.assertEqual(calls, ["proofs", "papers", "brainstorms"])
        self.assertEqual(fit.actions, ["proofs:retrieval", "papers:retrieval", "brainstorms:retrieval"])
        for kind in ("proofs", "papers", "brainstorms"):
            self.assertIn(f"<{kind} excerpts>", fit.prompt)

    async def test_retrieved_excerpts_are_cut_in_the_same_order_only_if_still_too_large(self):
        all_moved, _calls = await self._fit(700)
        self.assertEqual(
            all_moved.actions, ["proofs:retrieval", "papers:retrieval", "brainstorms:retrieval"]
        )
        # One character too small for all three in retrieval: the proofs excerpts go first.
        first, _calls = await self._fit(len(all_moved.prompt) - 1)
        self.assertEqual(first.actions, ["proofs:cut", "papers:retrieval", "brainstorms:retrieval"])
        self.assertNotIn("<proofs excerpts>", first.prompt)
        self.assertIn("<papers excerpts>", first.prompt)

        second, _calls = await self._fit(len(first.prompt) - 1)
        self.assertEqual(second.actions, ["proofs:cut", "papers:cut", "brainstorms:retrieval"])
        self.assertIn("<brainstorms excerpts>", second.prompt)

        third, calls = await self._fit(len(second.prompt) - 1)
        self.assertEqual(third.actions, ["proofs:cut", "papers:cut", "brainstorms:cut"])
        self.assertEqual(calls, ["proofs", "papers", "brainstorms"])
        self.assertEqual(third.prompt.count("removed to make room"), 3)

        # Smaller than everything cut: nothing fits, and no retrieval is spent finding out.
        nothing, calls = await self._fit(len(third.prompt) - 1)
        self.assertIsNone(nothing.prompt)
        self.assertEqual(calls, [])


class ExpansionFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prompt = f"TASK INSTRUCTIONS\n{_proofs_block()}\nAnswer in JSON."
        self.events: list[tuple[str, dict]] = []
        self.answer_prompts: list[str] = []
        self.review_prompts: list[list[dict]] = []

    def _call(self, answers: list[str], reviews: list[str], *, limit: int = 10**6) -> ExpansionCall:
        async def generate(messages):
            self.answer_prompts.append(messages[0]["content"])
            return _response(answers.pop(0))

        async def generate_review(messages):
            self.review_prompts.append(messages)
            return _response(reviews.pop(0))

        async def broadcast(event, payload):
            self.events.append((event, payload))

        return ExpansionCall(
            task_id="agg_sub1_001",
            role_id="aggregator_submitter_1",
            model="model/x",
            messages=[{"role": "user", "content": self.prompt}],
            generate=generate,
            generate_review=generate_review,
            fits=lambda text: len(text) <= limit,
            response_text=_text,
            broadcast=broadcast,
            route_fields={"workflow_mode": "aggregator"},
        )

    async def test_an_ordinary_answer_passes_through_untouched(self):
        answer = _response('{"submission": "idea"}')
        self.assertIs(await resolve_master_proof_expansions(self._call([], []), answer), answer)
        self.assertEqual((self.answer_prompts, self.events), ([], []))

    async def test_chosen_declarations_are_carried_verbatim_into_the_reissued_task(self):
        call = self._call(
            ['{"submission": "final"}'],
            ['{"keep_declarations": ["step_two", "not_real"], "notes": "reuse the zero-add step"}'],
        )
        result = await resolve_master_proof_expansions(
            call, _response(json.dumps({"expand_master_proof": HANDLE}))
        )

        self.assertEqual(_text(result), '{"submission": "final"}')
        review = self.review_prompts[0][0]["content"]
        self.assertIn("MASTER PROOF EXPANSION REVIEW", review)
        self.assertIn(MASTER_CODE.strip(), review)
        self.assertIn("TASK INSTRUCTIONS", review)
        reissued = self.answer_prompts[0]
        self.assertIn("lemma step_two (n : Nat) :\n    0 + n = n := by\n  simp", reissued)
        self.assertNotIn("lemma step_one (n : Nat) : n + 0 = n := by\n  simp", reissued)
        self.assertIn("Your review note: reuse the zero-add step", reissued)
        self.assertEqual(offered_handles(reissued), [])
        event, payload = self.events[0]
        self.assertEqual(event, MASTER_PROOF_EXPANSION_EVENT)
        self.assertEqual(payload["outcome"], "carried")
        self.assertEqual(payload["kept_declarations"], ["step_two"])
        self.assertEqual(payload["workflow_mode"], "aggregator")
        self.assertIn("carried 1 declaration into its task", payload["message"])

    async def test_partial_view_shows_only_the_requested_declarations(self):
        call = self._call(['{"submission": "final"}'], ['{"keep_declarations": []}'])
        await resolve_master_proof_expansions(
            call,
            _response(json.dumps({"expand_master_proof": HANDLE, "declarations": ["step_one"]})),
        )
        review = self.review_prompts[0][0]["content"]
        self.assertIn("lemma step_one (n : Nat) : n + 0 = n := by\n  simp", review)
        self.assertNotIn("def weight", review)
        self.assertEqual(self.events[0][1]["outcome"], "nothing_carried")
        self.assertIn("carried nothing forward", self.answer_prompts[0])

    async def test_a_malformed_review_gets_one_repair_then_carries_nothing(self):
        call = self._call(['{"submission": "final"}'], ["not json", "still not json"])
        await resolve_master_proof_expansions(
            call, _response(json.dumps({"expand_master_proof": HANDLE}))
        )
        self.assertEqual(len(self.review_prompts), 2)
        self.assertEqual(self.review_prompts[1][1]["role"], "assistant")
        self.assertEqual(self.events[0][1]["outcome"], "nothing_carried")

    async def test_a_second_request_for_the_same_master_proof_is_returned_to_the_role(self):
        repeat = json.dumps({"expand_master_proof": HANDLE})
        call = self._call([repeat], ['{"keep_declarations": ["step_one"]}'])
        result = await resolve_master_proof_expansions(call, _response(repeat))
        self.assertEqual(_text(result), repeat)
        self.assertEqual(len(self.review_prompts), 1)

    async def test_too_large_to_view_keeps_the_offer_for_named_declarations(self):
        # The full file never fits; a single declaration does.
        limit = len(self.prompt) + 900
        partial = json.dumps({"expand_master_proof": HANDLE, "declarations": ["step_one"]})
        call = self._call(
            [partial, '{"submission": "final"}'],
            ['{"keep_declarations": ["step_one"]}'],
            limit=limit,
        )
        long_code = MASTER_CODE.strip() + "\n" + "\n".join(f"-- pad {i}" * 20 for i in range(200))
        handle = context.register_master_proof(
            proof_id="proof_009", theorem_name="master", theorem_statement="s", lean_code=long_code
        )
        self.prompt = self.prompt.replace(HANDLE, handle)
        call.messages = [{"role": "user", "content": self.prompt}]
        partial = partial.replace(HANDLE, handle)
        call_answers = [partial, '{"submission": "final"}']

        async def generate(messages):
            self.answer_prompts.append(messages[0]["content"])
            return _response(call_answers.pop(0))

        call.generate = generate
        result = await resolve_master_proof_expansions(
            call, _response(json.dumps({"expand_master_proof": handle}))
        )

        self.assertEqual(_text(result), '{"submission": "final"}')
        self.assertEqual([payload["outcome"] for _event, payload in self.events], ["too_large", "carried"])
        self.assertIn("does not fit this role's context", self.answer_prompts[0])
        self.assertEqual(offered_handles(self.answer_prompts[0]), [handle])
        self.assertIn("lemma step_one (n : Nat) : n + 0 = n := by\n  simp", self.answer_prompts[1])


class ExpansionAuditTests(unittest.IsolatedAsyncioTestCase):
    """Regressions from the implementation audit."""

    PROMPT_HEAD = "TASK INSTRUCTIONS"

    def _prompt(self) -> str:
        return "\n".join([self.PROMPT_HEAD, _proofs_block(), "Answer in JSON."])

    def _call(self, prompt: str, answers: list[str], reviews: list[str], fits):
        prompts: list[str] = []
        events: list[tuple[str, dict]] = []

        async def generate(messages):
            prompts.append(messages[0]["content"])
            return _response(answers.pop(0))

        async def generate_review(messages):
            return _response(reviews.pop(0))

        async def broadcast(event, payload):
            events.append((event, payload))

        call = ExpansionCall(
            task_id="agg_sub1_001",
            role_id="aggregator_submitter_1",
            model="model/x",
            messages=[{"role": "user", "content": prompt}],
            generate=generate,
            generate_review=generate_review,
            fits=fits,
            response_text=_text,
            broadcast=broadcast,
        )
        return call, prompts, events

    async def test_a_task_with_no_room_closes_viewing_instead_of_asking_forever(self):
        prompt = self._prompt()
        # The review step fits, but the task itself cannot take a single extra character.
        call, prompts, events = self._call(
            prompt,
            ['{"submission": "final"}'],
            ['{"keep_declarations": ["step_one"]}'],
            lambda text: "MASTER PROOF EXPANSION REVIEW" in text or len(text) <= len(prompt),
        )
        result = await resolve_master_proof_expansions(
            call, _response(json.dumps({"expand_master_proof": HANDLE}))
        )

        self.assertEqual(_text(result), '{"submission": "final"}')
        reissued = prompts[0]
        self.assertLessEqual(len(reissued), len(prompt))
        self.assertEqual(offered_handles(reissued), [])
        self.assertNotIn("expand_master_proof", reissued)
        self.assertIn(f"[closed master proof {HANDLE}]", reissued)
        self.assertFalse(offers_master_proof_expansion([{"role": "user", "content": reissued}]))
        self.assertEqual(events[0][1]["outcome"], "no_room")

    async def test_declarations_that_do_not_fit_are_dropped_whole_from_the_end(self):
        prompt = self._prompt()
        counted: list[int] = []

        def fits(text: str) -> bool:
            counted.append(len(text))
            # Room for the result line and two short declarations, not all five.
            return "MASTER PROOF EXPANSION REVIEW" in text or len(text) <= len(prompt) + 330

        call, prompts, events = self._call(
            prompt,
            ['{"submission": "final"}'],
            ['{"keep_declarations": ["weight", "helper", "step_one", "step_two", "master"]}'],
            fits,
        )
        await resolve_master_proof_expansions(
            call, _response(json.dumps({"expand_master_proof": HANDLE}))
        )

        payload = events[0][1]
        self.assertEqual(payload["outcome"], "carried")
        self.assertEqual(payload["kept_declarations"], ["weight", "helper"])
        self.assertEqual(payload["omitted_count"], 3)
        self.assertIn("def weight (n : Nat) : Nat := n + 1", prompts[0])
        self.assertNotIn("exact ⟨step_one n, step_two n⟩", prompts[0])
        self.assertIn("3 further chosen declaration(s) were left out", prompts[0])
        # Bisection, not one count per dropped declaration.
        self.assertLessEqual(len(counted), 8)

    async def test_each_expansion_is_its_own_activity_row(self):
        ids = []
        for _ in range(2):
            call, _prompts, events = self._call(
                self._prompt(), ['{"submission": "final"}'], ['{"keep_declarations": []}'], lambda text: True
            )
            await resolve_master_proof_expansions(
                call, _response(json.dumps({"expand_master_proof": HANDLE}))
            )
            ids.append(events[0][1]["event_id"])
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(HANDLE in event_id for event_id in ids))

    async def test_retrieval_never_indexes_when_it_could_push_real_content_out(self):
        import backend.aggregator.core.rag_manager as rag_module
        from backend.shared import master_proof_expansion as expansion
        from backend.shared.config import rag_config

        calls: list[str] = []

        class FullIndex:
            chunks_by_size = {512: [object()] * rag_config.max_chunks_per_size}
            document_count = 3

            async def add_text(self, *args, **kwargs):
                calls.append("add")

            async def retrieve(self, *args, **kwargs):
                calls.append("retrieve")

            async def remove_document(self, *args, **kwargs):
                calls.append("remove")

        original = rag_module.rag_manager
        rag_module.rag_manager = FullIndex()
        try:
            evidence = await expansion.retrieve_context_evidence(
                "papers", "paper text " * 200, 500, "query"
            )
        finally:
            rag_module.rag_manager = original
        self.assertEqual((evidence, calls), ("", []))

    def test_withdrawing_offers_never_lengthens_the_prompt(self):
        prompt = self._prompt()
        closed = context.withdraw_expansion_offers(prompt)
        self.assertLess(len(closed), len(prompt))
        self.assertEqual(offered_handles(closed), [])
        self.assertIn("Solved master proof template", closed)


class ProofListInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name) / "proofs"
        self.directory.mkdir(parents=True)

    def _record(self, proof_id: str, lean_code: str, **extra) -> dict:
        return {
            "proof_id": proof_id,
            "theorem_statement": f"Statement of {proof_id}.",
            "theorem_name": "master",
            "source_type": "brainstorm",
            "source_id": "topic_001",
            "lean_code": lean_code,
            "novel": True,
            "novelty_tier": "mathematical_discovery",
            "created_at": "2026-11-12T00:00:00",
            **extra,
        }

    def _database(self, records: list[dict]):
        for record in records:
            (self.directory / f"proof_{record['proof_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        database = proof_database_module.ProofDatabase()
        database.set_base_dir(self.directory)
        return database

    async def test_master_proof_enters_as_statement_and_solved_template_only(self):
        individual_code = "theorem small : True := by\n  trivial"
        database = self._database(
            [
                self._record(
                    "proof_001",
                    MASTER_CODE.strip(),
                    proof_kind="master_proof",
                    master_proof=SUMMARY.model_dump(mode="json"),
                ),
                self._record("proof_002", individual_code),
            ]
        )
        await database.initialize()
        block = database.get_novel_proofs_for_injection()

        self.assertIn(MASTER_PROOF_LIST_NOTICE, block)
        self.assertIn(f"(MASTER PROOF, id {HANDLE}): Statement of proof_001.", block)
        self.assertIn("sorry  -- SOLVED: Lean 4 verified, proof hidden", block)
        # The list says outright that the stored proof has no placeholder.
        self.assertIn("Lean 4 verified the complete proof with no `sorry`", block)
        self.assertIn("each `sorry` below only marks where a verified proof is hidden", block)
        self.assertNotIn("def weight", block)
        self.assertNotIn("exact ⟨step_one n, step_two n⟩", block)
        # An individual proof still enters with its full Lean code.
        self.assertIn(individual_code, block)
        self.assertEqual(offered_handles(block), [HANDLE])
        self.assertTrue(block.startswith(PROOFS_BLOCK_START) and block.endswith(PROOFS_BLOCK_END))

    async def test_a_list_without_master_proofs_carries_no_expansion_notice(self):
        database = self._database([self._record("proof_002", "theorem small : True := by\n  trivial")])
        await database.initialize()
        self.assertNotIn("expand_master_proof", database.get_novel_proofs_for_injection())

    async def test_a_stored_proof_built_as_a_master_proof_is_linked_on_load(self):
        database = self._database([self._record("proof_001", MASTER_CODE.strip(), attempt_count=1)])
        masters = self.directory / "master_proofs"
        masters.mkdir()
        state = {
            "master_key": "feedbeef",
            "status": "verified",
            "root_name": "master",
            "scaffold_code": MASTER_CODE,
            "scaffold_revision": 2,
            "total_attempts": 7,
            "scaffold_attempts": [],
            "nodes": {"step_one": {"name": "step_one", "signature": "lemma step_one (n : Nat) : n + 0 = n"}},
        }
        (masters / "master_proof_feedbeef.json").write_text(json.dumps(state), encoding="utf-8")
        (masters / "master_proof_unfinished.json").write_text(
            json.dumps({**state, "master_key": "unfinished", "status": "solving"}), encoding="utf-8"
        )
        await database.initialize()

        proof = await database.get_proof("proof_001")
        self.assertEqual(proof.proof_kind, "master_proof")
        self.assertEqual(proof.master_proof.master_key, "feedbeef")
        self.assertEqual(proof.attempt_count, 7)
        self.assertEqual([step.name for step in proof.master_proof.steps], ["step_one"])
        self.assertEqual(proof.lean_code, MASTER_CODE.strip())
        stored = json.loads((self.directory / "proof_proof_001.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["proof_kind"], "master_proof")
        self.assertIn("(MASTER PROOF, id", database.get_novel_proofs_for_injection())

        # A state file is read once; an unchanged one is not re-read on later loads.
        examined = set(database._master_link_examined)
        self.assertEqual(len(examined), 1)
        database._index_data = None
        await database.initialize()
        self.assertEqual(database._master_link_examined, examined)

    async def test_templates_are_prepared_when_the_store_loads(self):
        context._solved_template.cache_clear()
        database = self._database(
            [
                self._record(
                    "proof_001",
                    MASTER_CODE.strip(),
                    proof_kind="master_proof",
                    master_proof=SUMMARY.model_dump(mode="json"),
                )
            ]
        )
        await database.initialize()
        self.assertEqual(context._solved_template.cache_info().currsize, 1)
        database.get_novel_proofs_for_injection()
        info = context._solved_template.cache_info()
        self.assertEqual((info.currsize, info.misses), (1, 1))
        self.assertGreaterEqual(info.hits, 1)


class OneOffRetrievalSourceTests(unittest.TestCase):
    def test_one_off_sources_are_read_only_by_a_retrieval_that_names_them(self):
        def chunk(source: str) -> DocumentChunk:
            return DocumentChunk.model_construct(source_file=source)

        one_off = f"{context.EPHEMERAL_RAG_SOURCE_PREFIX}papers_abc"
        chunks = [chunk("paper.txt"), chunk(one_off)]
        scope = RAGManager._filter_chunks_by_source_scope

        self.assertEqual([item.source_file for item in scope(chunks)], ["paper.txt"])
        self.assertEqual(
            [item.source_file for item in scope(chunks, include_sources=[one_off])], [one_off]
        )
        self.assertEqual(
            [item.source_file for item in scope(chunks, include_source_prefixes=["master_proof"])], []
        )


class ReviewPromptTests(unittest.TestCase):
    def test_contract_precedes_working_context_and_the_reminder_is_last(self):
        prompt = build_review_prompt(
            handle=HANDLE, expanded_code="lemma x : True := trivial", task_text="TASK", partial=False
        )
        contract = prompt.index("keep_declarations")
        self.assertLess(contract, prompt.index("TASK"))
        self.assertLess(prompt.index("TASK"), prompt.index("lemma x : True"))
        self.assertTrue(prompt.rstrip().endswith('"notes": "optional short note on how the kept declarations help your task"}'))


if __name__ == "__main__":
    unittest.main()
