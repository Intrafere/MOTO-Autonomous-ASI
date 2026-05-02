"""
Prompt builders for Lean 4 proof integration.
"""
from __future__ import annotations

from typing import Iterable, Any

from backend.shared.models import MathlibLemmaHint, ProofAttemptFeedback, SmtHint


PROOF_FRAMING_CONTEXT = """[PROOF FRAMING CONTEXT -- This research prompt targets formal mathematical proof.
Submissions should aggressively pursue NOVEL, NON-TRIVIAL theorems that push the
boundaries of what is known. The Lean 4 proof assistant is available for formal
verification. Prioritize ambitious conjectures, original results, and theorems that
would represent genuine mathematical contributions over safe restatements of textbook
facts. Standard identities and well-known Mathlib lemmas are NOT valuable targets.]"""


def _json_only_footer(example: str) -> str:
    return (
        "Respond with ONLY valid JSON. Do not use markdown fences. "
        "Escape backslashes correctly for JSON.\n\n"
        f"JSON format:\n{example}"
    )


def _format_attempt_history(prior_attempts: Iterable[ProofAttemptFeedback]) -> str:
    attempts = list(prior_attempts or [])
    if not attempts:
        return "No prior attempts."

    lines = []
    for attempt in attempts:
        if (
            not attempt.lean_code
            and not attempt.error_output
            and "malformed output" in (attempt.reasoning or "").lower()
        ):
            continue
        tactic_trace = "\n".join(
            f"  - {step}"
            for step in (attempt.tactic_trace or [])
        ) or "[none]"
        error_text = attempt.error_output or "[no error output]"
        rejection_banner = ""
        if "PROOF REJECTED: PLACEHOLDER USED" in error_text:
            rejection_banner = (
                "!! PLACEHOLDER REJECTION !! This prior attempt was rejected "
                "because it used `sorry` / `admit` (or an equivalent placeholder). "
                "Do NOT submit another placeholder proof. Either prove the goal "
                "fully, or return a narrower lemma you can actually close."
            )
        block = [
            f"ATTEMPT {attempt.attempt}:",
            f"Strategy: {attempt.strategy}",
            f"Reasoning: {attempt.reasoning}",
            "Lean 4 code:",
            attempt.lean_code or "[none]",
            "Tactic trace:",
            tactic_trace,
            "Lean 4 feedback:",
            error_text,
            f"Goal states: {attempt.goal_states or '[none]'}",
        ]
        if rejection_banner:
            block.append(rejection_banner)
        block.append("---")
        lines.extend(block)
    if not lines:
        return "No prior Lean-checked attempts."
    return "\n".join(lines)


def _format_relevant_lemmas(relevant_lemmas: Iterable[MathlibLemmaHint]) -> str:
    lemmas = list(relevant_lemmas or [])
    if not lemmas:
        return "[No confirmed Mathlib lemmas identified.]"

    lines = []
    for index, lemma in enumerate(lemmas, start=1):
        location = f"{lemma.file_path}:{lemma.line_number}" if lemma.file_path and lemma.line_number else (lemma.file_path or "[path unavailable]")
        lines.extend(
            [
                f"{index}. {lemma.full_name or lemma.requested_name}",
                f"   Declaration: {lemma.declaration or '[declaration unavailable]'}",
                f"   Source: {location}",
            ]
        )
    return "\n".join(lines)


def _truncate_text(value: str, limit: int) -> str:
    text = " ".join((value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_smt_hint(smt_hint: SmtHint | None) -> str:
    if not smt_hint:
        return "[No SMT guidance available.]"

    tactics = ", ".join(smt_hint.suggested_tactics or []) or "[none]"
    sections = [
        f"SMT result: {smt_hint.result}",
        f"Suggested Lean tactics: {tactics}",
    ]
    if smt_hint.smtlib.strip():
        sections.append(f"SMT-LIB encoding sent to Z3:\n{_truncate_text(smt_hint.smtlib, 1500)}")
    if smt_hint.z3_output.strip():
        sections.append(f"Z3 solver output:\n{_truncate_text(smt_hint.z3_output, 1000)}")
    return "\n".join(sections)


LEAN4_COMMON_PITFALLS = """COMMON LEAN 4 PITFALLS TO AVOID:
- NEVER use `sorry` or `admit` in the proof body. MOTO rejects any proof
  that contains `sorry` or `admit` anywhere, even though Lean would only
  emit a warning. A proof with `sorry` is not a proof. If you cannot close
  every goal, return a narrower lemma that you CAN fully prove.
- NEVER introduce new `axiom` declarations that exist only to make the
  target theorem trivial. Axiomatizing the concepts in the statement
  (e.g. `axiom Protocol : Type`, `axiom IC ... : ℝ`) and then proving the
  theorem with `sorry` is a vacuous proof and will be rejected. If a notion
  is not available, model it constructively or use concrete types from
  Mathlib instead.
- STOP writing tactics the instant all goals are closed. Appending ANY
  tactic after the proof is already complete causes Lean to emit
  `error: No goals to be solved`, which counts as a failed attempt. This
  includes: an extra `rfl`, `trivial`, `simp`, `exact`, `decide`, `omega`,
  `norm_num`, or a dangling bullet (`·` / `case _ =>`) after the previous
  branch already finished. If a prior attempt failed with "no goals to be
  solved", do NOT add more tactics -- DELETE the tactic at the reported
  line/column (and any tactics after it) and resubmit.
- Mathlib name collisions: Mathlib already defines names such as `Distribution`,
  `Protocol`, `Relation`, `Graph`, `Set`, `Group`, `Module`, `Order`, and many
  more. Do NOT redeclare these. If you need a local notion, use a unique prefix
  (e.g., `MOTO_Distribution`, `MyDist`, or open a fresh `namespace`), or
  introduce the object as a `variable` of abstract type.
- Missing `Inhabited`/`Nonempty` instances: when you write `∃ x, ...` or use
  tactics like `choose`, `Classical.choice`, or `Exists.intro` on a type with
  no default inhabitant, Lean cannot synthesize the instance. Either assume
  `[Inhabited α]` / `[Nonempty α]` in the theorem header, or construct an
  explicit witness before closing the goal.
- Deprecated tactics: do NOT use `push_neg` as a bare tactic in recent Mathlib.
  Use `push_neg at h` on a hypothesis, or prefer `simp only [not_forall,
  not_exists, not_and, not_or, not_not]` / `by_contra` with explicit rewrites.
  Similarly, avoid legacy aliases like `finish`, `tauto!`, `show_term` in proof
  output.
- Tactic state hygiene: every branch must actually close its goal. Do not rely
  on tactics that may leave unsolved goals (`cases`, `rcases`, `induction`)
  without a closing tactic on each branch.
- Import surface: `import Mathlib` is acceptable but slow; prefer narrower
  imports (e.g., `import Mathlib.Data.Real.Basic`) when you know exactly what
  is needed. When uncertain, fall back to `import Mathlib`."""


def format_failure_hints_for_injection(failure_hints: Iterable[Any]) -> str:
    hints = list(failure_hints or [])
    if not hints:
        return ""

    lines = [
        "=== OPEN LEMMA TARGETS LEAN 4 COULD NOT YET CLOSE ===",
        "[These are recent proof attempts that failed. Prefer brainstorms that generate missing lemmas, stronger assumptions, or cleaner formal theorem statements.]",
        "",
    ]
    for index, hint in enumerate(hints, start=1):
        theorem_statement = ""
        error_summary = ""
        suggested_targets: list[str] = []
        if isinstance(hint, dict):
            theorem_statement = str(hint.get("theorem_statement", "")).strip()
            error_summary = str(hint.get("error_summary", "")).strip()
            suggested_targets = [
                str(target).strip()
                for target in (hint.get("suggested_lemma_targets") or [])
                if str(target).strip()
            ]
        else:
            theorem_statement = str(getattr(hint, "theorem_statement", "")).strip()
            error_summary = str(getattr(hint, "error_summary", "")).strip()
            suggested_targets = [
                str(target).strip()
                for target in (getattr(hint, "suggested_lemma_targets", None) or [])
                if str(target).strip()
            ]
        placeholder_note = ""
        if "PROOF REJECTED: PLACEHOLDER USED" in error_summary:
            placeholder_note = (
                "Note: the previous formalization attempt was rejected because "
                "it used `sorry`/`admit` or axiomatized the theorem's concepts "
                "to make the goal trivial. Prefer brainstorms that state a "
                "narrower, concretely provable lemma instead of the full claim."
            )
        lines.extend(
            [
                f"OPEN TARGET {index}: {_truncate_text(theorem_statement or '[unnamed theorem]', 180)}",
                f"Lean 4 failure summary: {_truncate_text(error_summary or '[no summary available]', 200)}",
                f"Suggested lemma targets: {', '.join(suggested_targets[:6]) if suggested_targets else '[none identified]'}",
            ]
        )
        if placeholder_note:
            lines.append(placeholder_note)
        lines.append("---")
    lines.append("=== END OPEN LEMMA TARGETS ===")
    return "\n".join(lines)


def build_proof_framing_gate_prompt(user_prompt: str) -> str:
    """Ask whether the research goal should be framed toward formal proof."""
    return f"""You are deciding whether a research program should be explicitly framed toward formal mathematical proof and novel theorem discovery.

USER RESEARCH PROMPT:
{user_prompt}

Return TRUE if the prompt would benefit from working toward formally provable theorems in Lean 4, especially novel or non-trivial ones.
Return FALSE only if the prompt is purely empirical, engineering-focused, descriptive, or has no meaningful mathematical content.

Consider:
- Does the research involve mathematical structures, proofs, bounds, or formal reasoning?
- Could novel theorems or formalizations emerge from this research direction?
- Would formal verification add rigor or uncover new results?

Err on the side of TRUE -- if there is any mathematical substance worth formalizing, enable the proof pipeline.

{_json_only_footer('{"is_proof_amenable": true, "reasoning": "brief explanation"}')}
"""


def build_proof_identification_prompt(
    user_prompt: str,
    source_type: str,
    source_id: str,
    source_content: str,
) -> str:
    """Identify novel, non-trivial theorem candidates from a brainstorm or paper."""
    example_json = """{
  "has_provable_theorems": true,
  "theorems": [
    {
      "theorem_id": "thm_1",
      "statement": "natural-language theorem statement",
      "formal_sketch": "optional note about assumptions, notation, or likely Lean formalization strategy",
      "novelty_rationale": "why this theorem is non-trivial and worth formalizing"
    }
  ]
}"""
    return f"""You are a theorem-discovery agent for MOTO. Your mission is to find NOVEL, NON-TRIVIAL mathematical claims in the source below that deserve formal verification in Lean 4.

MOTO's goal is to push the frontier of mathematical knowledge. You are the gatekeeper that decides which theorems are worth the cost of formal verification. Be ambitious -- seek out the most original, surprising, or substantive results the source offers.

WHAT TO EXTRACT (prioritize these):
- Novel theorems, lemmas, or propositions that represent genuine mathematical insight
- Bold conjectures that can be sharpened into provable statements
- Non-obvious connections, bounds, inequalities, or structural results
- Original formalizations of results not yet in Mathlib
- Ambitious claims even if they need narrowing -- the formalization agent can refine them

WHAT TO REJECT (never extract these):
- Trivial identities (e.g. n + 0 = n, a * 1 = a, commutativity of addition)
- Direct restatements of well-known Mathlib lemmas or standard textbook results
- Results closable by a single tactic like `simp`, `omega`, `norm_num`, `decide`, or `rfl`
- Tautologies, definitional equalities, or purely notational rewrites
- Routine algebraic manipulations with no conceptual content

Rules:
- Return TRUE when at least one non-trivial, novel-potential theorem is found.
- Return FALSE only if the source genuinely contains nothing beyond trivial or well-known results.
- Rank candidates by novelty potential. Return at most 5 of the most promising theorems.
- For each candidate, include a brief novelty_rationale explaining why it is worth formalizing.
- Welcome bold or speculative claims -- if the source proposes something ambitious that might be provable with the right formalization, extract it. The downstream formalization agent will handle narrowing if needed.
- Use theorem IDs that are stable strings such as "thm_1", "thm_2", etc.

USER RESEARCH PROMPT:
{user_prompt}

SOURCE TYPE: {source_type}
SOURCE ID: {source_id}

SOURCE CONTENT:
{source_content}

{_json_only_footer(example_json)}
"""


def build_lemma_search_prompt(
    user_prompt: str,
    source_type: str,
    theorem_statement: str,
    formal_sketch: str,
    source_excerpt: str,
) -> str:
    """Suggest existing Mathlib lemmas likely to help prove the target theorem."""
    example_json = """{
  "lemma_names": [
    "Nat.add_comm",
    "Nat.add_assoc"
  ],
  "reasoning": "brief explanation"
}"""
    return f"""You are a Mathlib-lemma suggestion agent for Lean 4 proof generation.

Your job is to suggest EXISTING Mathlib declaration names that are likely useful for proving the target theorem.

Rules:
- Return 5-10 candidate lemma/theorem names when possible.
- Prefer concrete declaration names over descriptions.
- Use familiar Mathlib naming when possible (for example `Nat.add_comm`, `mul_assoc`, `Finset.card_union_add_card_inter`).
- If the theorem is too vague or no good candidates are evident, return an empty list.

USER RESEARCH PROMPT:
{user_prompt}

SOURCE TYPE:
{source_type}

TARGET THEOREM:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

SOURCE EXCERPT:
{source_excerpt}

{_json_only_footer(example_json)}
"""


def build_smt_translation_prompt(
    user_prompt: str,
    source_type: str,
    theorem_statement: str,
    formal_sketch: str,
    source_excerpt: str,
) -> str:
    """Ask the model to translate a conservative arithmetic theorem into SMT-LIB."""
    example_json = """{
  "smtlib": "(set-logic QF_LIA)\\n(declare-const n Int)\\n(assert (not (= (+ n 0) n)))\\n(check-sat)",
  "reasoning": "Negate the target theorem so unsat means the theorem is valid."
}"""
    return f"""You are translating a mathematical theorem into an SMT-LIB v2 check for Z3.

Your job is ONLY to build a conservative SMT-LIB program for a theorem that appears arithmetic or otherwise SMT-amenable.

Rules:
- Encode the NEGATION of the target theorem so that `unsat` means the theorem is valid.
- Prefer quantifier-free arithmetic fragments when possible.
- If the theorem is underspecified, only encode the part that is clearly justified by the theorem statement and notes.
- Do not invent new assumptions that are not strongly implied by the theorem.
- Return an empty `smtlib` string if you cannot produce a faithful SMT translation.
- Use only SMT-LIB text in the `smtlib` field.

USER RESEARCH PROMPT:
{user_prompt}

SOURCE TYPE:
{source_type}

TARGET THEOREM:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

SOURCE EXCERPT:
{source_excerpt}

{_json_only_footer(example_json)}
"""


def build_proof_formalization_prompt(
    user_prompt: str,
    source_type: str,
    theorem_statement: str,
    formal_sketch: str,
    source_excerpt: str,
    prior_attempts: Iterable[ProofAttemptFeedback],
    relevant_lemmas: Iterable[MathlibLemmaHint] = (),
    smt_hint: SmtHint | None = None,
) -> str:
    """Build the Lean 4 formalization prompt for one theorem."""
    attempt_history = _format_attempt_history(prior_attempts)
    relevant_lemmas_block = _format_relevant_lemmas(relevant_lemmas)
    smt_hint_block = _format_smt_hint(smt_hint)
    example_json = """{
  "theorem_name": "optional_lean_identifier",
  "lean_code": "import Mathlib\\n\\n theorem ... := by ...",
  "reasoning": "brief note about the formalization strategy"
}"""
    return f"""You are formalizing a mathematical theorem into Lean 4 code for MOTO.

Lean 4 will immediately compile-check your output. If prior attempts failed, you must use the exact failure history to improve the next attempt.

Requirements:
- Output COMPLETE Lean 4 code, ready to run.
- Include needed imports.
- State assumptions explicitly.
- Prefer correct, minimal, compilable code over stylistic elegance.
- PRESERVE the theorem's non-trivial content. Do not simplify or weaken the
  statement into a trivial identity just to make it compile. The goal is to
  formalize the ACTUAL claim, not a watered-down version of it.
- Your proof MUST close every goal without `sorry` or `admit`. Vacuous
  proofs (e.g. axiomatizing the theorem's own concepts and then closing
  with `sorry`) will be rejected even if Lean compiles them with only a
  warning.
- If the theorem seems invalid or underspecified, still make the strongest faithful formalization attempt you can from the provided source. If the full theorem cannot be proved, prove a narrower concrete lemma that is faithful to the source -- do NOT return a `sorry`-closed stub.
- Do not describe the code; provide the actual Lean 4 code in JSON.

USER RESEARCH PROMPT:
{user_prompt}

SOURCE TYPE:
{source_type}

TARGET THEOREM:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

SOURCE EXCERPT:
{source_excerpt}

RELEVANT MATHLIB LEMMAS:
{relevant_lemmas_block}

OPTIONAL SMT GUIDANCE:
{smt_hint_block}

If SMT guidance is present, treat it as a hint only. Lean 4 must still prove the theorem directly.
If one of the suggested tactics is genuinely appropriate, you may use it. Do not force it when it does not fit the goal.

{LEAN4_COMMON_PITFALLS}

PRIOR ATTEMPT HISTORY:
{attempt_history}

{_json_only_footer(example_json)}
"""


def build_proof_tactic_script_prompt(
    user_prompt: str,
    source_type: str,
    theorem_statement: str,
    formal_sketch: str,
    source_excerpt: str,
    prior_attempts: Iterable[ProofAttemptFeedback],
    relevant_lemmas: Iterable[MathlibLemmaHint] = (),
    smt_hint: SmtHint | None = None,
) -> str:
    """Build a tactic-oriented Lean 4 prompt for one theorem."""
    attempt_history = _format_attempt_history(prior_attempts)
    relevant_lemmas_block = _format_relevant_lemmas(relevant_lemmas)
    smt_hint_block = _format_smt_hint(smt_hint)
    example_json = """{
  "theorem_name": "optional_lean_identifier",
  "theorem_header": "theorem optional_lean_identifier (n : Nat) : n + 0 = n",
  "tactics": [
    {
      "tactic": "simpa using Nat.add_zero n",
      "reasoning": "Close the goal with the standard right-identity lemma."
    }
  ],
  "reasoning": "brief note about the tactic strategy"
}"""
    return f"""You are formalizing a mathematical theorem into Lean 4 using a tactic-by-tactic proof sketch for MOTO.

Lean 4 will immediately compile-check your output. If prior attempts failed, you must use the exact failure history to improve this attempt.

Requirements:
- Return a theorem header ONLY, without a proof body. Do not include `:= by` unless absolutely necessary.
- Return a short, ordered list of tactics that can be appended under a `by` block.
- Each tactic entry must include the Lean tactic string and one short reasoning note.
- Prefer small, composable tactics over a single opaque script.
- PRESERVE the theorem's non-trivial content. Do not simplify or weaken the
  statement into a trivial identity just to make it compile.
- NEVER include `sorry` or `admit` in the tactic list. A script that uses
  `sorry`/`admit` will be rejected even if Lean compiles it.
- Include needed assumptions in the theorem header. Do NOT axiomatize the
  concepts inside the theorem statement just to make the goal trivial.
- If the theorem is underspecified, make the strongest faithful formalization attempt you can from the source. If you cannot close every goal, return a narrower concrete lemma instead of a `sorry`-closed stub.
- Do not describe the code outside the JSON fields.

USER RESEARCH PROMPT:
{user_prompt}

SOURCE TYPE:
{source_type}

TARGET THEOREM:
{theorem_statement}

FORMALIZATION NOTES:
{formal_sketch or "[none]"}

SOURCE EXCERPT:
{source_excerpt}

RELEVANT MATHLIB LEMMAS:
{relevant_lemmas_block}

OPTIONAL SMT GUIDANCE:
{smt_hint_block}

If SMT guidance is present, treat it as a hint only. Lean 4 must still verify the theorem directly.
Suggested tactics are optional and should only be used when they genuinely match the goal.

{LEAN4_COMMON_PITFALLS}

PRIOR ATTEMPT HISTORY:
{attempt_history}

{_json_only_footer(example_json)}
"""


def build_proof_novelty_prompt(
    user_prompt: str,
    theorem_statement: str,
    lean_code: str,
    existing_novel_proofs: str,
) -> str:
    """Ask the validator whether a Lean-verified theorem is novel."""
    existing_proofs_block = existing_novel_proofs or "[No previously stored novel proofs.]"
    return f"""This proof has been FORMALLY VERIFIED by Lean 4. It is mathematically valid.

Your ONLY task: decide whether the verified result is NOVEL in the context of this research program.

A proof is NOVEL if it meets ANY of these criteria:
- It proves a result not already present in Mathlib or standard textbooks
- It establishes a new connection, bound, or structural insight
- It formalizes a conjecture or claim that was previously unverified
- It is a non-trivial composition of known results yielding something new
- It represents original work relative to the existing stored proofs below

A proof is NOT novel if:
- It is a direct restatement of a well-known Mathlib lemma or textbook theorem
- It is a trivial identity, tautology, or definitional equality
- It is closable by a single standard tactic (simp, omega, norm_num, decide, rfl)
- It duplicates a result already in the stored novel proofs below

Rules:
- Do NOT re-check validity. Lean 4 already verified it.
- When uncertain, consider the research prompt context -- a result that is textbook-standard in one field may be a novel formalization contribution in the context of this specific research program.
- Err on the side of recognizing novelty for results that required multi-step reasoning or non-trivial formalization work.

USER RESEARCH PROMPT:
{user_prompt}

VERIFIED THEOREM:
{theorem_statement}

LEAN 4 CODE:
{lean_code}

EXISTING STORED NOVEL PROOFS:
{existing_proofs_block}

{_json_only_footer('{"is_novel": true, "reasoning": "brief explanation"}')}
"""
