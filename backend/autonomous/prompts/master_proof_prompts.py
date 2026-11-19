"""Prompts for master proof scaffolds: scaffold writing, sorry closing, change review."""
from __future__ import annotations

from typing import Iterable

from backend.autonomous.prompts.proof_prompts import (
    LEAN4_COMMON_PITFALLS,
    _format_attempt_history,
    _format_relevant_lemmas,
    _format_source_title_block,
    _json_only_footer,
    _prepare_user_prompt_context,
    _split_verified_proof_context,
)
from backend.shared.master_proof_context import kind_for_source_type, note_context_part
from backend.shared.models import MathlibLemmaHint, ProofAttemptFeedback

MASTER_PROOF_ATTEMPTS_PER_OBLIGATION = 15


def without_verified_proof_library(user_prompt: str) -> str:
    """Return the research prompt without the injected verified proof library block."""
    return _split_verified_proof_context(user_prompt)[0]


def build_master_proof_scaffold_prompt(
    *,
    user_prompt: str,
    source_type: str,
    source_title: str,
    theorem_statement: str,
    formal_sketch: str,
    full_source_content: str,
    relevant_lemmas: Iterable[MathlibLemmaHint],
    prior_attempts: Iterable[ProofAttemptFeedback],
) -> str:
    """Build the prompt that writes the initial Lean 4 master proof scaffold."""
    note_context_part(kind_for_source_type(source_type), full_source_content)
    user_prompt, verified_proof_context_block = _prepare_user_prompt_context(user_prompt)
    example_json = """{
  "root_theorem_name": "lean_name_of_the_master_theorem",
  "lean_code": "import Mathlib\\n\\nlemma step_one ... := by\\n  sorry\\n\\ntheorem master ... := by\\n  exact ...",
  "reasoning": "brief note about the decomposition"
}"""
    return f"""You are writing a MASTER PROOF SCAFFOLD in Lean 4 for MOTO.

The MASTER THEOREM below is too large to prove in one attempt. Write one complete Lean 4 file that states it and reduces it to smaller obligations whose proofs are left as `sorry`. Lean 4 will immediately compile-check the file with `sorry` allowed, so everything must parse and typecheck. Afterwards, independent solvers each receive the whole scaffold and close one `sorry` at a time; a solver may expand its `sorry` into a further sub-scaffold as often as needed. Each `sorry` receives {MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} Lean 4 attempts, and the whole master proof fails if any one of them cannot be closed.

Requirements:
- Output COMPLETE Lean 4 code, ready to run, including needed imports.
- State the MASTER THEOREM faithfully and at full strength as a top-level `theorem`. Do not weaken, narrow, or trivialize it, and give its Lean name in root_theorem_name.
- Make the MASTER THEOREM the last `theorem`/`lemma` in the file, after every obligation and helper it uses.
- Write every unproved obligation as its own top-level `theorem` or `lemma` that starts at the beginning of a line, has a unique name, and has `sorry` as its entire proof.
- A `sorry` may appear only as the proof of such a theorem or lemma. Never put `sorry` in a definition, structure, instance, or inside a larger proof.
- Prove the MASTER THEOREM itself from those obligations without `sorry` whenever you can, so the decomposition is checked by Lean.
- Make each obligation a precise, true step that is provable directly or through further expansion. Prove easy steps directly instead of leaving them as `sorry`.
- Do not over-detail the scaffold. An obligation may be a larger, more general step that a later solver expands into its own sub-master-proof scaffold; split a step into finer obligations only where you are certain of that decomposition.
- State all assumptions explicitly. Build any definitions or theory Mathlib lacks inside the file.
- Never introduce `axiom`, `constant`, or `opaque` declarations.
- The full source content is mandatory authoritative context for the intended argument.
- If prior attempts failed, use the exact failure history to improve the next attempt.
- Do not describe the code; provide the actual Lean 4 code in JSON.

USER RESEARCH PROMPT:
{user_prompt}
{_format_source_title_block(source_type, source_title)}

VERIFIED PROOF LIBRARY CONTEXT (context only; do not treat this as the user prompt):
{verified_proof_context_block}

SOURCE TYPE:
{source_type}

MASTER THEOREM:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

FULL SOURCE CONTENT FROM WHICH THIS THEOREM WAS DERIVED:
{full_source_content or "[No source content provided.]"}

RELEVANT MATHLIB LEMMAS:
{_format_relevant_lemmas(relevant_lemmas)}

{LEAN4_COMMON_PITFALLS}

PRIOR ATTEMPT HISTORY:
{_format_attempt_history(prior_attempts)}

{_json_only_footer(example_json)}
"""


def build_master_proof_obligation_prompt(
    *,
    user_prompt: str,
    source_type: str,
    source_title: str,
    theorem_statement: str,
    root_theorem_name: str,
    scaffold_code: str,
    obligation_name: str,
    obligation_signature: str,
    full_source_content: str,
    relevant_lemmas: Iterable[MathlibLemmaHint],
    prior_attempts: Iterable[ProofAttemptFeedback],
) -> str:
    """Build the prompt that closes, expands, or restructures around one sorry."""
    note_context_part(kind_for_source_type(source_type), full_source_content)
    user_prompt, verified_proof_context_block = _prepare_user_prompt_context(user_prompt)
    example_json = """{
  "action": "submit_proof",
  "lean_code": "lemma helper ... := by\\n  sorry\\n\\nlemma target_name ... := by\\n  exact ...",
  "reasoning": "brief note about the proof or the change"
}"""
    return f"""You are closing ONE open `sorry` obligation in a MOTO master proof scaffold written in Lean 4.

The CURRENT SCAFFOLD below is the complete Lean 4 file. Other solvers are working on its other `sorry` obligations in parallel. Your obligation is TARGET OBLIGATION. Lean 4 will immediately compile-check the whole scaffold with your submission applied, and `sorry` is allowed only in other open obligations. This obligation has {MASTER_PROOF_ATTEMPTS_PER_OBLIGATION} attempts in total; if they run out, the whole master proof fails. Use the exact failure history to improve the next attempt.

Choose exactly one action:

1. "submit_proof" - lean_code replaces the TARGET OBLIGATION declaration only.
- It must contain the TARGET OBLIGATION exactly once, with the same name and the same statement, and a proof that contains no `sorry` or `admit`.
- It may add new top-level helper declarations before the target. Helper names must not repeat names already in the scaffold.
- If you cannot prove the target directly, EXPAND it: add new top-level `theorem`/`lemma` helpers whose entire proofs are `sorry`, and prove the target from them without `sorry`. Each new `sorry` becomes its own obligation with its own attempts and can be expanded again later, to any depth, so a new helper may itself be a larger, more general step; split it finer only where you are certain of that decomposition.
- Do not include `import` lines and do not restate or edit any other declaration of the scaffold.

2. "change_scaffold" - lean_code is a COMPLETE replacement Lean 4 file for the whole scaffold.
- Use this only when the scaffold itself must change for the master proof to succeed, for example when an obligation is false, is missing a hypothesis, is stated in an unprovable form, or a materially better decomposition exists.
- The master theorem `{root_theorem_name}` must keep exactly its current name and statement and remain the last `theorem`/`lemma` in the file.
- Keep every declaration that is already proved and still valid. Every remaining gap must be a top-level `theorem`/`lemma` whose entire proof is `sorry`.
- An independent Validator decides whether the change is allowed. If it is accepted, solvers working on obligations you changed or removed restart on the new scaffold.

Rules for both actions:
- Never introduce `axiom`, `constant`, or `opaque` declarations.
- A `sorry` may appear only as the entire proof of a named top-level `theorem` or `lemma` that starts at the beginning of a line.
- Do not weaken the master theorem or replace it with an easier statement.
- Do not describe the code; provide the actual Lean 4 code in JSON.

USER RESEARCH PROMPT:
{user_prompt}
{_format_source_title_block(source_type, source_title)}

VERIFIED PROOF LIBRARY CONTEXT (context only; do not treat this as the user prompt):
{verified_proof_context_block}

MASTER THEOREM (natural language):
{theorem_statement}

MASTER THEOREM LEAN NAME:
{root_theorem_name}

TARGET OBLIGATION NAME:
{obligation_name}

TARGET OBLIGATION STATEMENT:
{obligation_signature}

CURRENT SCAFFOLD:
{scaffold_code}

FULL SOURCE CONTENT FROM WHICH THE MASTER THEOREM WAS DERIVED:
{full_source_content or "[No source content provided.]"}

RELEVANT MATHLIB LEMMAS:
{_format_relevant_lemmas(relevant_lemmas)}

{LEAN4_COMMON_PITFALLS}

PRIOR ATTEMPT HISTORY FOR THIS OBLIGATION:
{_format_attempt_history(prior_attempts)}

{_json_only_footer(example_json)}
"""


def build_master_proof_scaffold_change_validation_prompt(
    *,
    user_prompt: str,
    theorem_statement: str,
    root_theorem_name: str,
    current_scaffold: str,
    proposed_scaffold: str,
    proposing_obligation: str,
    proposer_reasoning: str,
) -> str:
    """Build the independent Validator prompt for a proposed scaffold change."""
    user_prompt, _verified_proof_context_block = _prepare_user_prompt_context(user_prompt)
    example_json = """{
  "decision": "approve",
  "reasoning": "specific reason the change should or should not replace the current scaffold"
}"""
    return f"""You are the independent Validator for a proposed change to a MOTO master proof scaffold written in Lean 4.

A solver working on one `sorry` obligation wants to replace the whole scaffold. Other solvers are working on the other obligations in parallel, and any of them whose obligation is changed or removed will restart. Deterministic checks have already confirmed that the proposed scaffold compiles in Lean 4 with `sorry` allowed, that the master theorem `{root_theorem_name}` keeps its exact name and statement, and that no `axiom`, `constant`, or `opaque` declaration was added. Do not re-decide those facts.

Decide whether the change makes the master proof more likely to be completed.

Approve when the change repairs a real defect of the current scaffold, such as a false, underspecified, or unprovable obligation, or gives a materially better decomposition while keeping valid finished work.

Reject when the change discards valid progress without need, merely reshuffles or renames obligations, makes the remaining obligations harder or implausible, hides the whole difficulty in one `sorry` that restates the master theorem, or is not a credible route to proving the master theorem.

Use decision "approve" or "reject". If you reject, the reasoning must state the failed criterion and what the solver should do instead.

USER RESEARCH PROMPT:
{user_prompt}

MASTER THEOREM (natural language):
{theorem_statement}

OBLIGATION THE PROPOSING SOLVER WAS WORKING ON:
{proposing_obligation}

SOLVER'S STATED REASON FOR THE CHANGE:
{proposer_reasoning or "[none given]"}

CURRENT SCAFFOLD:
{current_scaffold}

PROPOSED SCAFFOLD:
{proposed_scaffold}

{_json_only_footer(example_json)}
"""
