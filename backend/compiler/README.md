# Compiler Tool

The compiler/distillation tool transforms aggregated research submissions into a coherent academic paper.

## Purpose

The compiler tool reads the aggregator's shared training database and systematically builds a complete academic paper through sequential validation cycles.

## Implementation Status

✅ **Fully Implemented**

## Features

- **Sequential Markov Chain Workflow**: One submitter runs at a time, each submission must be validated before proceeding
- **Multiple Submitter Modes**:
  - **Outline Creation/Update**: High-context model creates and maintains paper structure
  - **Paper Construction**: High-context model writes paper sections following the outline
  - **Review/Cleanup**: High-context model reviews and fixes errors (without aggregator DB context)
  - **Rigor Mode (Lean 4)**: High-parameter model proposes one theorem per cycle, runs up to 5 Lean 4 formalization attempts with error-feedback chaining, persists the verified proof into the shared `proof_database`, and places it inline (2 placement attempts) or appends it to the Theorems Appendix on double rejection.
- **Real-time Paper Viewing**: Live updates in the GUI as the paper is constructed
- **Intelligent Placement Logic**: Automatically inserts content at the correct location based on placement context
- **Separate GUI Tabs**: Compiler Interface, Settings, Logs, and Live Paper view

## Architecture

### Core Components

- `compiler_coordinator.py` - Main orchestrator for sequential workflow
- `compiler_rag_manager.py` - Manages RAG retrieval for aggregator database

### Agents

- `high_context_submitter.py` - Low-parameter, high-context model (outline, construction, review)
- `high_param_submitter.py` - High-parameter model. Rigor mode: discovery + 5x Lean 4 attempts + novelty classification + 2-attempt placement + Theorems Appendix fallback.

### Validation

- `compiler_validator.py` - Validates submissions for coherence, rigor, placement, and non-redundancy

### Memory

- `outline_memory.py` - Manages current paper outline
- `paper_memory.py` - Manages current paper state
- `compiler_rejection_log.py` - Tracks last 10 rejections and acceptances

## System-Managed Markers - Critical Architecture

The compiler uses two categories of hard-coded markers:

### 1. Section Placeholders (in paper during construction)

**Markers:**
- `[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]`
- `[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]`
- `[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]`
- `[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]`
- `[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]`

**Management:**
- Added by `paper_memory.py` via `initialize_with_placeholders()`
- Replaced by `paper_memory.replace_placeholder()` when sections are validated
- Theorems appendix bracket pair wraps the Lean-4-verified theorem entries that the rigor loop produces; new entries are appended via `paper_memory.append_to_theorems_appendix(...)`.
- Purpose: Make it crystal clear to AI what sections exist vs. don't exist, and keep Lean-4-verified theorems in a dedicated, stable location.

### 2. Anchors (in paper and outline)

**Markers:**
- Paper: `[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]`
- Outline: Two-line system (end-of-paper reference + end-of-outline mark)

**Management:**
- Added/maintained by `paper_memory.py` and `outline_memory.py`
- Purpose: Non-chronological stop tokens preventing content after endpoint

### Critical Distinction for Prompts

**Markers in CURRENT DOCUMENT/OUTLINE:**
- System-managed, expected, normal during construction
- NOT AI-generated content
- Code automatically adds and removes them

**Markers in SUBMISSION CONTENT:**
- Forbidden, invalid, auto-rejected by pre-validation
- Validator criterion #11 (NO PLACEHOLDER TEXT) checks submissions only
- Pre-validation check at `compiler_validator.py` line 326 catches this before validation

### Why Placeholders Fix Bugs

Previous system had AI falsely claiming sections were "already written" when they weren't. Placeholders make document state explicit:
- If placeholder present → section does NOT exist yet
- If actual content present → section IS written
- Eliminates AI confusion about document state

## Startup

The compiler starts manually via API only (`POST /api/compiler/start`). There is no automatic startup trigger.

## Configuration

Default context window: 131072 tokens (configurable in GUI settings)

## Integration

The compiler continuously reads from the aggregator's shared training database (`backend/data/rag_shared_training.txt`) and re-RAGs every 10 new aggregator acceptances.

## Tools Available to Submitters

- **Wolfram Alpha (construction mode only)**: When `system_config.wolfram_alpha_enabled=true`, the high-context submitter may invoke the `wolfram_alpha_query` OpenAI-compatible tool up to 20 times per construction submission. See `WOLFRAM_TOOL_SCHEMA` in `high_context_submitter.py`. Audit trail attached to `CompilerSubmission.metadata["wolfram_calls"]`. Not available in `outline_create`, `outline_update`, `review`, or rigor mode.
- **Lean 4 (rigor mode only)**: The rigor loop uses `ProofFormalizationAgent.prove_candidate(max_attempts=5)` from `backend/autonomous/agents/proof_formalization_agent.py` backed by the Lean 4 toolchain + Mathlib workspace. Verified proofs are persisted in the shared `proof_database` (same store used by autonomous mode). Novel proofs are automatically injected into the highest-priority direct-injection block on subsequent submitter instantiations.

