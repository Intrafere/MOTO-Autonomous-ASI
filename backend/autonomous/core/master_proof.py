"""Master proof solving: one Lean scaffold whose `sorry` obligations close in parallel.

A master proof candidate is first written as a Lean 4 scaffold that Lean accepts
with `sorry` allowed. Every theorem/lemma that still contains `sorry` is an open
obligation. Obligations are attempted in parallel; each attempt may close its
obligation, expand it into further `sorry` obligations (to any depth), or propose
a replacement scaffold that an independent Validator must approve. The scaffold
and every obligation each have a fixed attempt budget; exhausting any one of
them fails the whole master proof. Lean 4 stays authoritative: the master proof
is verified only by a final placeholder-free check of the complete file.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field

from backend.autonomous.agents.master_proof_agent import (
    MasterProofAgent,
    MasterProofContextOverflow,
    MasterProofScaffoldChangeValidator,
)
from backend.autonomous.prompts.master_proof_prompts import MASTER_PROOF_ATTEMPTS_PER_OBLIGATION
from backend.shared.atomic_json import atomic_write_json, atomic_write_text
from backend.shared.config import system_config
from backend.shared.lean4_client import (
    get_lean4_client,
    raise_for_lean_workspace_failure,
    require_lean_workspace,
)
from backend.shared.lean_proof_integrity import validate_lean_proof_integrity
from backend.shared.lean_scaffold import (
    declarations_with_sorry_warning,
    open_obligations,
    parse_blocks,
    plan_obligation_replacement,
    resolve_theorem_name,
    scaffold_structure_problem,
)
from backend.shared.models import ProofAttemptFeedback, ProofCandidate

logger = logging.getLogger(__name__)

MASTER_PROOF_EVENT = "master_proof_progress"
MasterProofEmit = Callable[[str, dict[str, Any]], Awaitable[None]]


class MasterProofNode(BaseModel):
    """One `sorry` obligation of the scaffold and its attempt history."""

    name: str
    signature: str
    parent: str = ""
    depth: int = 0
    status: Literal["open", "closed"] = "open"
    attempts: list[ProofAttemptFeedback] = Field(default_factory=list)


class MasterProofState(BaseModel):
    """Durable state of one master proof; the scaffold file is the working draft."""

    version: int = 1
    master_key: str
    theorem_id: str
    statement: str
    status: Literal["scaffolding", "solving", "verified", "failed"] = "scaffolding"
    root_name: str = ""
    root_signature: str = ""
    scaffold_code: str = ""
    scaffold_revision: int = 0
    scaffold_attempts: list[ProofAttemptFeedback] = Field(default_factory=list)
    nodes: dict[str, MasterProofNode] = Field(default_factory=dict)
    total_attempts: int = 0
    failure_reason: str = ""


def master_proof_key(*, run_id: str, source_type: str, source_id: str, statement: str) -> str:
    material = "\n".join(
        [run_id.strip(), source_type.strip().lower(), source_id.strip().lower(), " ".join(statement.split()).lower()]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class MasterProofStore:
    """Persists master proof state and its scaffold file under the proof store."""

    def __init__(self, directory: Optional[Path]) -> None:
        self._directory = Path(directory) if directory else None
        self._lock = asyncio.Lock()

    def _state_path(self, key: str) -> Path:
        return self._directory / f"master_proof_{key}.json"

    async def load(self, key: str) -> Optional[MasterProofState]:
        if self._directory is None:
            return None
        path = self._state_path(key)

        def read() -> Optional[str]:
            return path.read_text(encoding="utf-8") if path.is_file() else None

        try:
            raw = await asyncio.to_thread(read)
            return MasterProofState.model_validate_json(raw) if raw else None
        except Exception as exc:  # unreadable state restarts the master proof
            logger.warning("Ignoring unreadable master proof state %s: %s", key, type(exc).__name__)
            return None

    async def save(self, state: MasterProofState) -> None:
        if self._directory is None:
            return
        payload = state.model_dump(mode="json")
        scaffold = state.scaffold_code
        state_path = self._state_path(state.master_key)
        lean_path = self._directory / f"master_proof_{state.master_key}.lean"

        def write() -> None:
            atomic_write_json(state_path, payload)
            if scaffold:
                atomic_write_text(lean_path, scaffold if scaffold.endswith("\n") else scaffold + "\n")

        async with self._lock:
            await asyncio.to_thread(write)


@dataclass
class MasterProofOutcome:
    """Result handed back to the proof stage for one master proof candidate."""

    success: bool
    theorem_name: str = ""
    lean_code: str = ""
    final_feedback: Optional[ProofAttemptFeedback] = None
    overflow_feedback: Optional[ProofAttemptFeedback] = None
    # Most recent attempt that reached the model before an overflow, if any.
    reached_feedback: Optional[ProofAttemptFeedback] = None


class MasterProofSolver:
    """Drives one master proof from scaffold to a placeholder-free Lean check."""

    def __init__(
        self,
        *,
        state: MasterProofState,
        store: MasterProofStore,
        agent: MasterProofAgent,
        change_validator: MasterProofScaffoldChangeValidator,
        candidate: ProofCandidate,
        user_prompt: str,
        source_type: str,
        source_title: str,
        source_content: str,
        emit: MasterProofEmit,
        should_stop: Optional[Callable[[], bool]] = None,
        max_parallel: int = 0,
    ) -> None:
        self.state = state
        self.store = store
        self.agent = agent
        self.change_validator = change_validator
        self.candidate = candidate
        self.user_prompt = user_prompt
        self.source_type = source_type
        self.source_title = source_title
        self.source_content = source_content
        self._emit = emit
        self._should_stop = should_stop
        self._scaffold_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_parallel) if max_parallel > 0 else None

    # ------------------------------------------------------------------ helpers

    def _check_stop(self) -> None:
        if self._should_stop and self._should_stop():
            raise asyncio.CancelledError()

    async def _lean_check(self, code: str, *, allow_placeholders: bool):
        result = await get_lean4_client().check_proof(
            code,
            timeout=system_config.lean4_proof_timeout,
            allow_placeholders=allow_placeholders,
        )
        raise_for_lean_workspace_failure(result)
        return result

    def _feedback(
        self,
        *,
        attempt: int,
        lean_code: str,
        reasoning: str,
        error_output: str,
        goal_states: str = "",
        lean_was_run: bool,
        failure_kind: str,
    ) -> ProofAttemptFeedback:
        return ProofAttemptFeedback(
            attempt=attempt,
            theorem_id=self.candidate.theorem_id,
            reasoning=reasoning,
            lean_code=lean_code,
            error_output=error_output,
            goal_states=goal_states,
            strategy="full_script",
            success=False,
            failure_kind=failure_kind,
            lean_was_run=lean_was_run,
        )

    def _open_nodes(self) -> list[MasterProofNode]:
        return [node for node in self.state.nodes.values() if node.status == "open"]

    def _progress(self) -> dict[str, Any]:
        nodes = self.state.nodes.values()
        return {
            "master_proof_id": self.state.master_key,
            "scaffold_revision": self.state.scaffold_revision,
            "open_obligations": sum(1 for node in nodes if node.status == "open"),
            "closed_obligations": sum(1 for node in nodes if node.status == "closed"),
            "attempt_limit": MASTER_PROOF_ATTEMPTS_PER_OBLIGATION,
        }

    async def _announce(self, kind: str, message: str, **extra: Any) -> None:
        # Obligations share one theorem and reuse attempt numbers, so each event
        # carries its own stable identity for Live Activity.
        event_id = ":".join(
            [
                "master-proof",
                self.state.master_key,
                kind,
                str(extra.get("obligation") or extra.get("scope") or ""),
                str(extra.get("attempt") or ""),
                f"r{self.state.scaffold_revision}",
                f"t{self.state.total_attempts}",
            ]
        )
        await self._emit(
            kind, {**self._progress(), "message": message, "event_id": event_id, **extra}
        )

    def _last_reached_feedback(self) -> Optional[ProofAttemptFeedback]:
        """Return evidence that this master proof already reached the model."""
        recorded = [
            *self.state.scaffold_attempts,
            *(attempt for node in self.state.nodes.values() for attempt in node.attempts),
        ]
        if recorded:
            return recorded[-1]
        if self.state.total_attempts <= 0:
            return None
        return self._feedback(
            attempt=self.state.total_attempts,
            lean_code=self.state.scaffold_code,
            reasoning="Lean 4 accepted the master proof scaffold; open obligations remain.",
            error_output="The master proof still has open `sorry` obligations.",
            lean_was_run=True,
            failure_kind="lean_rejected",
        )

    async def _record_failure(
        self,
        attempts: list[ProofAttemptFeedback],
        feedback: ProofAttemptFeedback,
        *,
        scope: str,
        obligation: str = "",
    ) -> None:
        attempts.append(feedback)
        self.state.total_attempts += 1
        await self.store.save(self.state)
        label = f"obligation `{obligation}`" if obligation else "scaffold"
        detail = (feedback.error_output or feedback.reasoning or "").strip()
        await self._announce(
            "attempt_failed",
            (
                f"Master proof {label}: attempt {len(attempts)} failed "
                f"({len(attempts)} of {MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} failures in a row; "
                f"a success never counts)."
            ),
            scope=scope,
            obligation=obligation,
            attempt=len(attempts),
            failure_kind=feedback.failure_kind,
            lean_was_run=feedback.lean_was_run,
            error_summary=detail[:500],
        )

    async def _fail(self, reason: str) -> MasterProofOutcome:
        self.state.status = "failed"
        self.state.failure_reason = reason
        await self.store.save(self.state)
        await self._announce("failed", f"Master proof failed: {reason}")
        return MasterProofOutcome(
            success=False,
            lean_code=self.state.scaffold_code,
            final_feedback=self._feedback(
                attempt=max(1, self.state.total_attempts),
                lean_code=self.state.scaffold_code,
                reasoning="Master proof failed.",
                error_output=f"MASTER PROOF FAILED: {reason}",
                lean_was_run=True,
                failure_kind="lean_rejected",
            ),
        )

    # ----------------------------------------------------------------- scaffold

    async def _build_scaffold(self) -> bool:
        attempts = self.state.scaffold_attempts
        while len(attempts) < MASTER_PROOF_ATTEMPTS_PER_OBLIGATION:
            self._check_stop()
            attempt_number = len(attempts) + 1
            await self._announce(
                "attempt_started",
                f"Master proof scaffold: attempt {attempt_number} started "
                f"({attempt_number - 1} of {MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} "
                f"failures in a row so far).",
                scope="scaffold",
                attempt=attempt_number,
            )
            submission = await self.agent.propose_scaffold(
                user_prompt=self.user_prompt,
                source_type=self.source_type,
                source_title=self.source_title,
                candidate=self.candidate,
                source_content=self.source_content,
                prior_attempts=attempts,
                attempt_number=attempt_number,
            )
            if isinstance(submission, ProofAttemptFeedback):
                await self._record_failure(attempts, submission, scope="scaffold")
                continue
            code = submission.lean_code
            root_name = resolve_theorem_name(code, submission.root_theorem_name)
            problem = scaffold_structure_problem(code, root_name)
            if not problem:
                integrity = validate_lean_proof_integrity(lean_code=code)
                problem = "" if integrity.valid else integrity.reason
            if problem:
                await self._record_failure(
                    attempts,
                    self._feedback(
                        attempt=attempt_number,
                        lean_code=code,
                        reasoning=submission.reasoning,
                        error_output=problem,
                        lean_was_run=False,
                        failure_kind="integrity_rejected",
                    ),
                    scope="scaffold",
                )
                continue
            lean_result = await self._lean_check(code, allow_placeholders=True)
            if not lean_result.success:
                await self._record_failure(
                    attempts,
                    self._feedback(
                        attempt=attempt_number,
                        lean_code=code,
                        reasoning=submission.reasoning,
                        error_output=lean_result.error_output,
                        goal_states=lean_result.goal_states,
                        lean_was_run=True,
                        failure_kind="lean_rejected",
                    ),
                    scope="scaffold",
                )
                continue
            root = next(
                block
                for block in parse_blocks(code)
                if block.is_theorem and block.name == root_name
            )
            self.state.root_name = root.name
            self.state.root_signature = root.signature
            self.state.scaffold_code = code
            self.state.scaffold_revision = 1
            self.state.total_attempts += 1
            self.state.status = "solving"
            self.state.nodes = {
                block.name: MasterProofNode(name=block.name, signature=block.signature)
                for block in open_obligations(code)
            }
            await self.store.save(self.state)
            await self._announce(
                "scaffold_accepted",
                (
                    f"Lean 4 accepted the master proof scaffold: a "
                    f"{len(self.state.nodes)}-step master proof. Each step is an open "
                    f"`sorry` that is now attempted on its own."
                ),
                obligations=sorted(self.state.nodes),
            )
            return True
        return False

    # -------------------------------------------------------------- obligations

    async def _apply_proof(self, node: MasterProofNode, submission, attempt_number: int) -> str:
        plan = plan_obligation_replacement(
            self.state.scaffold_code, node.name, node.signature, submission.lean_code
        )
        problem = plan.problem
        if not problem:
            problem = scaffold_structure_problem(
                plan.code, self.state.root_name, required_root_signature=self.state.root_signature
            )
        if not problem:
            integrity = validate_lean_proof_integrity(lean_code=plan.code)
            problem = "" if integrity.valid else integrity.reason
        if problem:
            await self._record_failure(
                node.attempts,
                self._feedback(
                    attempt=attempt_number,
                    lean_code=submission.lean_code,
                    reasoning=submission.reasoning,
                    error_output=problem,
                    lean_was_run=False,
                    failure_kind="integrity_rejected",
                ),
                scope="obligation",
                obligation=node.name,
            )
            return "retry"
        lean_result = await self._lean_check(plan.code, allow_placeholders=True)
        lean_error = (
            "" if lean_result.success else (lean_result.error_output or "Lean 4 rejected the submission.")
        )
        if lean_result.success and node.name in declarations_with_sorry_warning(
            getattr(lean_result, "checked_source", "") or plan.code,
            getattr(lean_result, "diagnostic_output", "") or "",
        ):
            lean_error = (
                f"SUBMISSION REJECTED: Lean 4 reports that the proof of `{node.name}` still uses "
                "`sorry`. Prove it without any placeholder; put a remaining gap in a new "
                "top-level helper `theorem`/`lemma` whose proof is `sorry`."
            )
        if lean_error:
            await self._record_failure(
                node.attempts,
                self._feedback(
                    attempt=attempt_number,
                    lean_code=submission.lean_code,
                    reasoning=submission.reasoning,
                    error_output=lean_error,
                    goal_states=lean_result.goal_states,
                    lean_was_run=True,
                    failure_kind="lean_rejected",
                ),
                scope="obligation",
                obligation=node.name,
            )
            return "retry"
        node.status = "closed"
        self.state.total_attempts += 1
        self.state.scaffold_code = plan.code
        self.state.scaffold_revision += 1
        children = [block.name for block in plan.new_obligations]
        for block in plan.new_obligations:
            self.state.nodes[block.name] = MasterProofNode(
                name=block.name,
                signature=block.signature,
                parent=node.name,
                depth=node.depth + 1,
            )
        await self.store.save(self.state)
        if children:
            await self._announce(
                "sorry_expanded",
                (
                    f"Master proof obligation `{node.name}` was expanded into "
                    f"{len(children)} new `sorry` obligation(s) at depth {node.depth + 1}."
                ),
                obligation=node.name,
                attempt=attempt_number,
                new_obligations=children,
                depth=node.depth + 1,
            )
        else:
            await self._announce(
                "sorry_closed",
                f"Master proof obligation `{node.name}` was closed on attempt {attempt_number}.",
                obligation=node.name,
                attempt=attempt_number,
            )
        return "closed"

    async def _apply_change(self, node: MasterProofNode, submission, attempt_number: int) -> str:
        code = submission.lean_code
        problem = scaffold_structure_problem(
            code, self.state.root_name, required_root_signature=self.state.root_signature
        )
        if not problem and " ".join(code.split()) == " ".join(self.state.scaffold_code.split()):
            problem = "SCAFFOLD CHANGE REJECTED: the proposed scaffold is identical to the current one."
        if not problem:
            integrity = validate_lean_proof_integrity(lean_code=code)
            problem = "" if integrity.valid else integrity.reason
        lean_was_run = False
        goal_states = ""
        if not problem:
            lean_result = await self._lean_check(code, allow_placeholders=True)
            lean_was_run = True
            if not lean_result.success:
                problem = lean_result.error_output or "Lean 4 rejected the proposed scaffold."
                goal_states = lean_result.goal_states
        if problem:
            await self._record_failure(
                node.attempts,
                self._feedback(
                    attempt=attempt_number,
                    lean_code=code,
                    reasoning=submission.reasoning,
                    error_output=problem,
                    goal_states=goal_states,
                    lean_was_run=lean_was_run,
                    failure_kind="lean_rejected" if lean_was_run else "integrity_rejected",
                ),
                scope="obligation",
                obligation=node.name,
            )
            return "retry"
        await self._announce(
            "scaffold_change_review_started",
            f"Validator is reviewing a scaffold change proposed while closing `{node.name}`.",
            obligation=node.name,
            attempt=attempt_number,
        )
        review = await self.change_validator.review(
            user_prompt=self.user_prompt,
            theorem_statement=self.candidate.statement,
            root_theorem_name=self.state.root_name,
            current_scaffold=self.state.scaffold_code,
            proposed_scaffold=code,
            proposing_obligation=f"{node.name}: {node.signature}",
            proposer_reasoning=submission.reasoning,
        )
        if not review.decided or not review.approved:
            header = (
                "SCAFFOLD CHANGE REJECTED BY VALIDATOR"
                if review.decided
                else "SCAFFOLD CHANGE NOT REVIEWED (no Validator decision; the scaffold is unchanged)"
            )
            await self._record_failure(
                node.attempts,
                self._feedback(
                    attempt=attempt_number,
                    lean_code=code,
                    reasoning=submission.reasoning,
                    error_output=f"{header}: {review.reasoning}",
                    lean_was_run=True,
                    failure_kind="integrity_rejected",
                ),
                scope="obligation",
                obligation=node.name,
            )
            await self._announce(
                "scaffold_change_rejected" if review.decided else "scaffold_change_unreviewed",
                (
                    f"Validator rejected the scaffold change proposed for `{node.name}`."
                    if review.decided
                    else f"The scaffold change proposed for `{node.name}` could not be reviewed; the scaffold is unchanged."
                ),
                obligation=node.name,
                attempt=attempt_number,
                validator_feedback=review.reasoning,
            )
            return "retry"
        previous = self.state.nodes
        rebuilt: dict[str, MasterProofNode] = {}
        for block in open_obligations(code):
            existing = previous.get(block.name)
            if existing is not None and existing.status == "open" and existing.signature == block.signature:
                rebuilt[block.name] = existing
            else:
                rebuilt[block.name] = MasterProofNode(name=block.name, signature=block.signature)
        reset = sorted(
            name
            for name, old in previous.items()
            if old.status == "open" and rebuilt.get(name) is not old
        )
        # Finished obligations the new scaffold still proves stay on record.
        proved = {
            block.name: block.signature
            for block in parse_blocks(code)
            if block.is_theorem and block.name and not block.has_placeholder
        }
        for name, old in previous.items():
            if old.status == "closed" and name not in rebuilt and proved.get(name) == old.signature:
                rebuilt[name] = old
        if rebuilt.get(node.name) is node:
            node.attempts.append(
                self._feedback(
                    attempt=attempt_number,
                    lean_code="",
                    reasoning=submission.reasoning,
                    error_output=(
                        "SCAFFOLD CHANGE ACCEPTED: the scaffold was replaced and this "
                        "obligation is unchanged and still open."
                    ),
                    lean_was_run=True,
                    failure_kind="lean_rejected",
                )
            )
        self.state.nodes = rebuilt
        self.state.scaffold_code = code
        self.state.scaffold_revision += 1
        self.state.total_attempts += 1
        await self.store.save(self.state)
        await self._announce(
            "scaffold_changed",
            (
                f"Validator approved a scaffold change proposed for `{node.name}`. "
                f"{len(reset)} obligation attempt(s) reset to the new scaffold."
            ),
            obligation=node.name,
            attempt=attempt_number,
            validator_feedback=review.reasoning,
            reset_obligations=reset,
            obligations=sorted(rebuilt),
        )
        return "changed"

    async def _attempt_node(self, node: MasterProofNode) -> str:
        attempt_number = len(node.attempts) + 1
        await self._announce(
            "attempt_started",
            (
                f"Master proof obligation `{node.name}`: attempt {attempt_number} started "
                f"({attempt_number - 1} of {MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} "
                f"failures in a row so far)."
            ),
            scope="obligation",
            obligation=node.name,
            attempt=attempt_number,
            depth=node.depth,
        )
        submission = await self.agent.attempt_obligation(
            user_prompt=self.user_prompt,
            source_type=self.source_type,
            source_title=self.source_title,
            candidate=self.candidate,
            source_content=self.source_content,
            root_theorem_name=self.state.root_name,
            scaffold_code=self.state.scaffold_code,
            obligation_name=node.name,
            obligation_signature=node.signature,
            prior_attempts=node.attempts,
            attempt_number=attempt_number,
        )
        async with self._scaffold_lock:
            if self.state.nodes.get(node.name) is not node:
                return "reset"
            if isinstance(submission, ProofAttemptFeedback):
                await self._record_failure(
                    node.attempts, submission, scope="obligation", obligation=node.name
                )
                return "retry"
            if submission.action == "change_scaffold":
                return await self._apply_change(node, submission, attempt_number)
            return await self._apply_proof(node, submission, attempt_number)

    async def _work_node(self, node: MasterProofNode) -> str:
        while True:
            if self.state.nodes.get(node.name) is not node or node.status != "open":
                return "reset"
            if len(node.attempts) >= MASTER_PROOF_ATTEMPTS_PER_OBLIGATION:
                return "exhausted"
            self._check_stop()
            if self._slots is None:
                verdict = await self._attempt_node(node)
            else:
                async with self._slots:
                    verdict = await self._attempt_node(node)
            if verdict != "retry":
                return verdict

    async def _solve(self) -> MasterProofOutcome:
        tasks: dict[str, tuple[MasterProofNode, asyncio.Task]] = {}
        every_task: list[asyncio.Task] = []
        try:
            while True:
                self._check_stop()
                for node in self._open_nodes():
                    if node.name not in tasks:
                        task = asyncio.create_task(self._work_node(node))
                        tasks[node.name] = (node, task)
                        every_task.append(task)
                if not tasks:
                    return await self._finalize()
                done, _pending = await asyncio.wait(
                    [task for _node, task in tasks.values()],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                exhausted = ""
                for name, (node, task) in list(tasks.items()):
                    if task not in done:
                        continue
                    del tasks[name]
                    if task.result() == "exhausted":
                        exhausted = exhausted or name
                if exhausted:
                    return await self._fail(
                        f"obligation `{exhausted}` was not closed within "
                        f"{MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} attempts."
                    )
                # A scaffold change resets every attempt whose obligation it replaced.
                # Taking the scaffold lock first lets a commit in progress, including
                # the change itself, finish before any stale attempt is cancelled.
                async with self._scaffold_lock:
                    stale = [
                        (name, task)
                        for name, (node, task) in tasks.items()
                        if self.state.nodes.get(name) is not node and not task.done()
                    ]
                    for _name, task in stale:
                        task.cancel()
                if stale:
                    await asyncio.gather(*(task for _name, task in stale), return_exceptions=True)
                    for name, _task in stale:
                        del tasks[name]
        finally:
            for task in every_task:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*every_task, return_exceptions=True)

    async def _finalize(self) -> MasterProofOutcome:
        code = self.state.scaffold_code
        lean_result = await self._lean_check(code, allow_placeholders=False)
        if not lean_result.success:
            return await self._fail(
                "every obligation was closed but the final placeholder-free Lean 4 check "
                f"rejected the assembled proof: {(lean_result.error_output or '').strip()[:1500]}"
            )
        final_code = getattr(lean_result, "checked_source", "") or code
        self.state.status = "verified"
        self.state.total_attempts += 1
        await self.store.save(self.state)
        await self._announce(
            "completed",
            "Master proof completed: every `sorry` is closed and Lean 4 accepted the full proof.",
        )
        return MasterProofOutcome(
            success=True,
            theorem_name=self.state.root_name,
            lean_code=final_code,
            final_feedback=ProofAttemptFeedback(
                attempt=self.state.total_attempts,
                theorem_id=self.candidate.theorem_id,
                reasoning="Master proof assembled from its closed scaffold obligations.",
                lean_code=final_code,
                error_output=lean_result.error_output,
                goal_states=lean_result.goal_states,
                strategy="full_script",
                success=True,
                compile_receipt=getattr(lean_result, "compile_receipt", None),
                lean_was_run=True,
            ),
        )

    # --------------------------------------------------------------------- run

    async def run(self) -> MasterProofOutcome:
        require_lean_workspace(get_lean4_client())
        try:
            if self.state.status == "failed":
                return await self._fail(self.state.failure_reason or "the master proof already failed.")
            if not self.state.scaffold_code:
                if not await self._build_scaffold():
                    return await self._fail(
                        "Lean 4 did not accept a scaffold within "
                        f"{MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} attempts."
                    )
            return await self._solve()
        except MasterProofContextOverflow as overflow:
            await self.store.save(self.state)
            return MasterProofOutcome(
                success=False,
                lean_code=self.state.scaffold_code,
                overflow_feedback=overflow.feedback,
                reached_feedback=self._last_reached_feedback(),
            )
