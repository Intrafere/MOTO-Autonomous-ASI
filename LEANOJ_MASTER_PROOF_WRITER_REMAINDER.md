# LeanOJ Master Proof Writer Remainder

## Audit Result
The master-proof edit loop, persistent `master_proof.lean`, edit history log, resume metadata, stuck-to-brainstorm signal, and focused coordinator tests are implemented.

## Fixed After Audit
### Mandatory Full Proof Direct Injection
The final-solver prompt now directly injects the full `master_proof.lean` as mandatory context. The proof attempt is never truncated, summarized, windowed, or RAG-substituted. If the full master proof cannot fit alongside the other mandatory prompt context, LeanOJ raises a hard mandatory direct context overflow error and stops instead of continuing with partial proof context.

Implemented:
1. Full master proof direct injection.
2. Hard overflow error when the mandatory full proof cannot fit.
3. Token counting before prompt assembly.
4. Test coverage for mandatory context overflow.

## Completed Follow-Ups
1. Added read-only API access for retrieving the current master proof draft on demand, without broadcasting it in normal status payloads.
2. Added compact edit-history summaries for debugging.
3. Added snapshot compaction for large `master_proof_edits.jsonl` logs.
4. Added a conservative progress watchdog for repeated `needs_more_time: true` edits that do not make meaningful progress.
5. Added focused coordinator/API route tests for the follow-up behavior.
6. Added a UI viewer tab for the master proof draft and edit history.

## Remaining Optional Follow-Ups
- None currently known.
