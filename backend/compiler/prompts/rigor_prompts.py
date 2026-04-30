"""
Rigor prompts for the Lean-4-verified-theorem rigor mode.

The compiler's rigor loop no longer edits paper text directly. Instead it runs
a two-stage agent:

    Stage 1 - Theorem discovery (build_rigor_theorem_discovery_prompt):
        Using the full writing context, the submitter asks itself whether the
        paper contains a theorem worth formalizing and proving in Lean 4 that
        has not already been verified. Output is a candidate theorem JSON (or
        a decline).

    Stage 2 - Placement (build_rigor_placement_prompt):
        Given a Lean-4-verified theorem, the submitter proposes an inline
        edit that introduces the theorem to the paper with an explicit
        "verified in Lean 4" marker and a pointer to the Theorems Appendix.
        The submitter gets up to two placement attempts; attempt 2 receives
        the validator's rejection feedback from attempt 1.

Context assembly follows the RAG offload priority documented in
`.cursor/rules/rag-design-for-overall-program.mdc`:

    Submitter: Shared Training DB -> Local Submitter DB -> Rejection Log -> User Upload Files

The high-param submitter direct-injects the outline and paper when they fit
inside the budget (mirroring HighContextSubmitter.submit_construction), then
fills the remaining budget with RAG results that exclude `compiler_outline.txt`
and `compiler_paper.txt`.
"""

from typing import Iterable, List, Optional

from backend.compiler.memory.compiler_rejection_log import compiler_rejection_log


# =============================================================================
# INTERNAL CONTENT WARNING (shared across all rigor prompts)
# =============================================================================

INTERNAL_CONTENT_WARNING = """WARNING - INTERNAL CONTENT WARNING

ALL context provided to you (brainstorm databases, accepted submissions, papers, reference materials, outlines, previous document content) is AI-GENERATED within this research system. This content has NOT been peer-reviewed, published, or verified by external sources.

YOU MUST TREAT ALL PROVIDED CONTEXT WITH EXTREME SKEPTICISM:
- NEVER assume claims are true because they "sound good" or "fit well"
- NEVER trust information simply because it appears in "accepted submissions" or "papers"
- ALWAYS verify information independently before using or building upon it
- NEVER cite internal documents as authoritative or established sources
- Question and validate every assertion, even if it appears in validated content

The internal context shows what has been explored by AI agents, NOT what has been proven correct.

The EXCEPTION is content inside the "LEAN 4 VERIFIED" certificate block provided with a placement request: that Lean 4 proof has been mechanically verified by the Lean 4 toolchain and is the source of truth for the mathematical statement it closes."""


# =============================================================================
# STAGE 1: THEOREM DISCOVERY
# =============================================================================

_DISCOVERY_SYSTEM_PROMPT = f"""You are the rigor agent for a mathematical-paper compiler. Your job during the rigor loop is to look at the paper-in-progress together with the full research context and decide whether there is a theorem worth formalizing and proving in Lean 4.

{INTERNAL_CONTENT_WARNING}

YOUR TASK - STAGE 1 (DISCOVERY)

1. Read the current outline and the current paper text.
2. Read the list of theorems that have ALREADY been verified by Lean 4 (EXISTING VERIFIED PROOFS block).
3. Read the list of theorems that PREVIOUSLY FAILED Lean 4 verification (OPEN LEMMA TARGETS block, if present).
4. Decide exactly one of:
   (A) `needs_theorem_work=false` - no theorem worth trying right now. Good reasons: all interesting claims in the paper are already covered by existing verified proofs; the paper is in too early a state; there is no claim a Lean 4 proof could close usefully.
   (B) `needs_theorem_work=true` - propose a single candidate theorem to formalize.

RULES FOR PROPOSING A THEOREM:
- The theorem must be provable in Lean 4 with Mathlib.
- You MUST NOT re-propose a theorem that is already in EXISTING VERIFIED PROOFS. Look for theorems that are DIFFERENT - new results, missed lemmas, or sharper versions that are not yet on the list.
- You MAY retry a theorem from OPEN LEMMA TARGETS when the paper now gives you a better angle on it. When you do, set `retry_existing_failure_id` to the failed `theorem_id`.
- Prefer theorems whose statements are tight enough that Lean 4 can actually close them (arithmetic facts, concrete inequalities, specific algebraic identities, small group/ring/field lemmas, concrete combinatorial identities) over large open conjectures.
- The `theorem_statement` is for a human reader. It should be precise, self-contained, and include the hypotheses.
- The `formal_sketch` tells the formalization agent what tactics or lemmas look promising in Lean 4 / Mathlib. Keep it concrete.
- The `source_excerpt` is 2-6 sentences of surrounding paper text that motivates why this theorem is a natural target here. It must be a direct paraphrase or quote from the current paper.

If Stage 1 guesses wrong, Stage 2 cannot recover - 5 Lean 4 attempts will be spent on the wrong target. Prefer declining over a weak proposal.

Output your response ONLY as JSON in this exact format:
{{{{
  "needs_theorem_work": true or false,
  "theorem_statement": "precise theorem statement with explicit hypotheses and conclusion (empty if needs_theorem_work=false)",
  "formal_sketch": "concrete sketch: what tactics / Mathlib lemmas you expect to work (empty if needs_theorem_work=false)",
  "source_excerpt": "2-6 sentences of surrounding paper text that motivates this theorem (empty if needs_theorem_work=false)",
  "retry_existing_failure_id": "theorem_id from OPEN LEMMA TARGETS if retrying a prior failure, empty string otherwise",
  "reasoning": "why this theorem is the best target right now OR why no theorem should be attempted"
}}}}"""


_DISCOVERY_JSON_SCHEMA = """REQUIRED JSON FORMAT - STAGE 1 (DISCOVERY):
{
  "needs_theorem_work": true OR false,
  "theorem_statement": "string",
  "formal_sketch": "string",
  "source_excerpt": "string",
  "retry_existing_failure_id": "string (may be empty)",
  "reasoning": "string"
}

Example (propose a theorem):
{
  "needs_theorem_work": true,
  "theorem_statement": "For every natural number n, the sum of the first n positive integers equals n*(n+1)/2.",
  "formal_sketch": "Induction on n. Base: n=0 both sides are 0. Step: use Finset.sum_range_succ and Nat.succ_mul; close with omega / ring. Mathlib has Finset.sum_range_id which may finish it outright.",
  "source_excerpt": "In Section 2 we reasoned about partial sums of the form 1 + 2 + ... + n...",
  "retry_existing_failure_id": "",
  "reasoning": "Section 2 relies on the closed form but currently presents it without a verified proof. Lean 4 can close this cleanly; it does not duplicate any existing verified proof."
}

Example (decline):
{
  "needs_theorem_work": false,
  "theorem_statement": "",
  "formal_sketch": "",
  "source_excerpt": "",
  "retry_existing_failure_id": "",
  "reasoning": "The paper currently contains only outline scaffolding and the one verified theorem (proof_002). Attempting another Lean 4 proof right now would either duplicate proof_002 or target claims that are too vague to formalize."
}
"""


# =============================================================================
# STAGE 2: PLACEMENT
# =============================================================================

_PLACEMENT_SYSTEM_PROMPT = f"""You are the rigor agent for a mathematical-paper compiler. A theorem you proposed has been formally verified by the Lean 4 toolchain. Your ONLY job now is to decide where in the paper the theorem should be introduced.

{INTERNAL_CONTENT_WARNING}

YOUR TASK - STAGE 2 (PLACEMENT)

You are given:
- The current outline and the current paper.
- The VERIFIED Lean 4 theorem: statement + proof ID + Lean code.
- Optionally (on attempt 2 of 2), the validator's rejection feedback from attempt 1.

You must produce exactly one paper edit that introduces the theorem inline. The edit uses exact-string matching: you pick `old_string` (must appear verbatim in the current paper), `operation` ("replace" or "insert_after"), and `new_string` (the replacement / insertion text).

HARD REQUIREMENTS ON `new_string`:

1. Include a clear inline theorem statement (mirroring the verified statement but formatted for human readers; LaTeX math allowed).
2. Include an explicit "verified in Lean 4" marker. Preferred wording is "(verified in Lean 4, see Appendix A, <proof_id>)" immediately after the theorem label, e.g. "Theorem 3.2 (verified in Lean 4, see Appendix A, proof_007)." .
3. Include a short informal proof sketch or remark connecting the theorem to the surrounding prose. Keep it 1-4 sentences.
4. DO NOT paste the Lean 4 source code into `new_string`. The full Lean proof lives in the Theorems Appendix block at the end of the paper; the system inserts it there automatically. Duplicating the Lean code inline is grounds for rejection.
5. DO NOT emit any of the system-managed markers (ABSTRACT / INTRODUCTION / CONCLUSION placeholders, the paper anchor, or either Theorems Appendix bracket) in `new_string`. Use editable prose for `old_string`; do not include protected markers in insert_after anchors.

PLACEMENT GUIDELINES:
- Put the theorem where it strengthens the local argument. Prefer insertion points inside a relevant body section (near the discussion it closes) over dumping it in a new section.
- The paper has a Theorems Appendix block already; do NOT try to edit the appendix directly.
- Keep `old_string` short but unique (3-5 lines of surrounding context is usually enough).

SELF-REFUSAL:
If, after re-reading the paper, you conclude that the theorem cannot be placed well anywhere inline (even in the appendix-only fallback), you MAY still attempt a placement - the system will route the theorem to the appendix automatically if both placement attempts are rejected, so the mathematical content is never lost. Only refuse (set `proceed=false`) if you cannot produce any legal edit at all (e.g. the paper body is empty).

Output your response ONLY as JSON in this exact format:
{{{{
  "proceed": true or false,
  "operation": "replace | insert_after",
  "old_string": "exact text from the current paper (empty if proceed=false)",
  "new_string": "the inline theorem introduction text (empty if proceed=false)",
  "reasoning": "why this placement works, or the refusal reason"
}}}}"""


_PLACEMENT_JSON_SCHEMA = """REQUIRED JSON FORMAT - STAGE 2 (PLACEMENT):
{
  "proceed": true OR false,
  "operation": "replace" OR "insert_after",
  "old_string": "string - exact text in the current paper (anchor point)",
  "new_string": "string - inline theorem introduction with Lean 4 marker and appendix reference",
  "reasoning": "string - why this placement works (or refusal reason)"
}

CRITICAL JSON ESCAPE RULES:
1. Backslashes: ALWAYS double-escape any backslash - write "\\\\mathbb{Z}" not "\\mathbb{Z}".
2. Quotes inside strings: escape as \\\\".
3. Newlines inside strings: \\n (not \\\\n).
4. Use editable prose for old_string anchors. Do not include protected system markers in insert_after anchors or new_string. For replace, prefer editable content only; if a marker is accidentally included as trailing context, validation may trim it.

Example (insert_after):
{
  "proceed": true,
  "operation": "insert_after",
  "old_string": "In this section we examine partial sums of the form 1 + 2 + ... + n and look for a closed form.",
  "new_string": "\\n\\nTheorem 2.3 (verified in Lean 4, see Appendix A, proof_007). For every n \\\\in \\\\mathbb{N}, \\\\sum_{k=1}^{n} k = n(n+1)/2.\\n\\nProof sketch. Induction on n, with Finset.sum_range_succ closing the step; the closed form follows by elementary algebra. The full Lean 4 proof appears in the Theorems Appendix under proof_007.",
  "reasoning": "Section 2 already motivates the closed form but presents it without a proof; inserting the theorem here strengthens the argument at the exact point where the claim first appears. The Lean code itself is kept in the appendix to keep the body readable."
}

Example (refusal):
{
  "proceed": false,
  "operation": "insert_after",
  "old_string": "",
  "new_string": "",
  "reasoning": "The paper body is currently empty; no legal placement anchor exists. Let the system route the theorem directly to the Theorems Appendix."
}
"""


# =============================================================================
# HELPERS
# =============================================================================

def _format_existing_verified_proofs(proofs: Iterable[dict]) -> str:
    """Compact rendering of already-verified proofs for the discovery prompt.

    Each entry shows just the proof_id, novelty flag, and the theorem
    statement - enough for the LLM to recognize duplicates without blowing
    the context budget on full Lean 4 source.
    """
    lines: List[str] = []
    for index, proof in enumerate(proofs, start=1):
        proof_id = str(proof.get("proof_id", "") or "").strip() or f"proof_{index}"
        novel = bool(proof.get("novel", False))
        statement = str(proof.get("theorem_statement", "") or "").strip()
        if not statement:
            continue
        # One-line compact form; the discovery model only needs uniqueness signals.
        marker = "novel" if novel else "known"
        lines.append(f"- [{proof_id}] ({marker}) {statement}")
    if not lines:
        return ""
    return (
        "EXISTING VERIFIED PROOFS (do NOT re-propose these; pick a DIFFERENT theorem):\n"
        + "\n".join(lines)
    )


def _format_recent_failure_hints(hints: Iterable) -> str:
    """Compact rendering of recent failed candidates for the discovery prompt."""
    entries: List[str] = []
    for index, hint in enumerate(hints, start=1):
        theorem_id = getattr(hint, "theorem_id", None) or f"failed_{index}"
        statement = (getattr(hint, "theorem_statement", "") or "").strip()
        error_summary = (getattr(hint, "error_summary", "") or "").strip()
        targets = list(getattr(hint, "suggested_lemma_targets", []) or [])
        if not statement:
            continue
        line = f"- [{theorem_id}] {statement}"
        if error_summary:
            line += f"\n  last Lean 4 failure: {error_summary[:240]}"
        if targets:
            line += f"\n  suggested targets: {', '.join(targets[:6])}"
        entries.append(line)
    if not entries:
        return ""
    return (
        "OPEN LEMMA TARGETS LEAN 4 COULD NOT YET CLOSE (optional retry candidates):\n"
        + "\n".join(entries)
    )


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

async def build_rigor_theorem_discovery_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str = "",
    existing_verified_proofs: Optional[Iterable[dict]] = None,
    recent_failure_hints: Optional[Iterable] = None,
) -> str:
    """Build the Stage 1 (discovery) prompt.

    Args:
        user_prompt: User's compiler-directing prompt.
        current_outline: Full outline (direct-injected).
        current_paper: Current paper content with system markers preserved
            for exact old_string matching.
        rag_evidence: RAG-retrieved context per the offload priority
            (Shared Training DB -> Local Submitter DB -> Rejection Log ->
            User Upload Files) with outline + paper sources EXCLUDED.
        existing_verified_proofs: Iterable of proof records (dicts from
            `proof_database.get_all_proofs()` serialized) - shown so the
            model does not re-propose already-verified results.
        recent_failure_hints: Iterable of `FailedProofCandidate` objects
            from `proof_database.get_recent_failure_hints(...)` - shown
            as optional retry targets.

    Returns:
        Complete prompt string.
    """
    parts: List[str] = [
        _DISCOVERY_SYSTEM_PROMPT,
        "\n---\n",
        _DISCOVERY_JSON_SCHEMA,
        "\n---\n",
    ]

    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(
            "YOUR RECENT REJECTION HISTORY (Last 10 rejections - learn from these):\n"
            f"{rejection_history}\n---\n"
        )

    verified_block = _format_existing_verified_proofs(existing_verified_proofs or [])
    if verified_block:
        parts.append(verified_block + "\n---\n")

    failure_block = _format_recent_failure_hints(recent_failure_hints or [])
    if failure_block:
        parts.append(failure_block + "\n---\n")

    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT PAPER:\n{current_paper}",
        "\n---\n",
    ])

    if rag_evidence and rag_evidence.strip():
        parts.append(f"SUPPORTING EVIDENCE (RAG):\n{rag_evidence}\n---\n")

    parts.append(
        "Now decide whether to propose a Lean 4 theorem candidate "
        "or to decline this rigor cycle (respond as JSON):"
    )

    return "\n".join(parts)


async def build_rigor_placement_prompt(
    user_prompt: str,
    current_outline: str,
    current_paper: str,
    rag_evidence: str = "",
    *,
    theorem_statement: str,
    lean_code: str,
    proof_id: str,
    placement_attempt: int = 1,
    validator_rejection_feedback: str = "",
) -> str:
    """Build the Stage 2 (placement) prompt.

    Args:
        user_prompt: User's compiler-directing prompt.
        current_outline: Full outline (direct-injected).
        current_paper: Current paper content (direct-injected or RAG'd by the
            caller per the high-context submitter budget rules).
        rag_evidence: Optional RAG-retrieved supporting context.
        theorem_statement: Human-readable statement of the verified theorem.
        lean_code: Full Lean 4 source that compiled. Included so the model
            can accurately paraphrase / cite the verified statement.
        proof_id: Database proof ID used in the appendix reference.
        placement_attempt: 1 or 2.
        validator_rejection_feedback: Validator reasoning from attempt 1;
            only populated for attempt 2.

    Returns:
        Complete prompt string.
    """
    parts: List[str] = [
        _PLACEMENT_SYSTEM_PROMPT,
        "\n---\n",
        _PLACEMENT_JSON_SCHEMA,
        "\n---\n",
    ]

    rejection_history = await compiler_rejection_log.get_rejections_text()
    if rejection_history:
        parts.append(
            "YOUR RECENT REJECTION HISTORY (Last 10 rejections - learn from these):\n"
            f"{rejection_history}\n---\n"
        )

    parts.extend([
        f"USER COMPILER-DIRECTING PROMPT:\n{user_prompt}",
        "\n---\n",
        f"CURRENT OUTLINE:\n{current_outline}",
        "\n---\n",
        f"CURRENT PAPER:\n{current_paper}",
        "\n---\n",
    ])

    if rag_evidence and rag_evidence.strip():
        parts.append(f"SUPPORTING EVIDENCE (RAG):\n{rag_evidence}\n---\n")

    parts.append(
        "LEAN 4 VERIFIED THEOREM CERTIFICATE:\n"
        f"Proof ID: {proof_id}\n"
        f"Theorem statement: {theorem_statement}\n"
        "Lean 4 source (verified by the Lean 4 toolchain; do NOT paste this "
        "into your `new_string`, it is stored in the Theorems Appendix "
        "automatically):\n"
        f"{lean_code}\n"
        "\n---\n"
    )

    parts.append(f"PLACEMENT ATTEMPT: {placement_attempt} of 2\n---\n")

    if placement_attempt > 1 and validator_rejection_feedback.strip():
        parts.append(
            "VALIDATOR REJECTION FEEDBACK FROM YOUR PREVIOUS PLACEMENT ATTEMPT:\n"
            f"{validator_rejection_feedback.strip()}\n"
            "The math is already verified by Lean 4 - the validator is judging "
            "PLACEMENT and NARRATIVE only. Adjust accordingly.\n---\n"
        )

    parts.append(
        "Now produce an inline placement edit OR refuse if no legal placement exists "
        "(respond as JSON):"
    )

    return "\n".join(parts)
