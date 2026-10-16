---
description: Cloud provider keys, Boost routing, Supercharge, free-model fallback, and workflow metrics
---

# API Key Controls & Workflow Management System

## Overview

Enables cloud provider access with LM Studio fallback in default mode (OpenRouter API keys plus desktop OAuth providers such as OpenAI Codex and xAI Grok/SuperGrok) and OpenRouter-only operation in generic mode, plus boost controls, research metrics, and the Assistant Memory Bank in the workflow panel.

**Key Features:**
- **Per-Role Cloud Provider Selection**: Each role independently uses LM Studio, OpenRouter, or a desktop-only OAuth/subscription provider where available (OpenAI Codex OAuth, xAI Grok OAuth, Sakana Fugu or Kimi Code API key; default mode); generic mode remains OpenRouter-only.
- **OpenRouter/OAuth**: Header overlay manages OpenRouter API keys and desktop OAuth/provider logins (OpenAI Codex, xAI Grok/SuperGrok OAuth, Sakana Fugu and Kimi Code subscription API keys). The post-disclaimer startup wizard requires OpenRouter or LM Studio first and presents OAuth/provider add-ons later. Providers are registry-driven: backend `subscription_providers/metadata.py` lists every cloud provider (id, label, auth kind, REST slug, capability flag, `max`-effort support) and derives model-provider validation, route labels, error markers, locks, and status sets; frontend `oauthProviders.js` mirrors it (a parity test enforces this) and drives the modal, settings panels, reasoning controls, and auto-fill. A new API-key subscription adds one metadata entry, one `ApiKeyProviderProfile` (endpoints, known models, request shaping, error classifier) registered in `subscription_providers/registry.py`, and one frontend entry; routing, REST routes, and UI need no edits. OAuth attribution identifiers must refer to MOTO Autonomous ASI, never the bare nickname `moto`; compact OAuth identifier fields use `moto-autonomous-asi` unless a provider explicitly requires the full display name.
- **Credential State Refresh**: Adding/removing OpenRouter keys or desktop OAuth/subscription credentials must immediately invalidate mounted settings panels so provider buttons and model lists refresh without tab navigation. Credential presence, authenticated validity, and model-list loading are distinct: OpenRouter validation uses its authenticated key endpoint rather than the public catalog, restored/stale or transiently unreachable keys are rechecked before readiness, and failed OAuth/subscription model loads clear stale model lists and show provider-specific errors. Secret inputs use provider-unique autofill identities and never hydrate from status or browser storage. Per-provider mutations serialize validation, durable commit/clear, and runtime publication; failed commit-pointer writes remove uncommitted keyring slots while preserving the prior committed credential.
- **SyntheticLib4 Access**: SyntheticLib4 backend/search scaffolding remains in place, but the user-facing connectivity pill is currently a `Coming soon` explainer rather than a ready/configuration surface. The modal explains the reciprocal proof-contribution/access model until production corpus access is enabled.
- **SyntheticLib lifecycle gate**: One backend-owned fail-closed product state
  controls promotion. Known prelaunch states retain Coming soon for standard users
  and expose only a read-only developer workspace under Shift+Z+X. Backend credential,
  account, import, mock-service, and local evidence-audit execution additionally
  requires explicit local preview authority and is denied in generic mode; browser
  developer mode is never execution authority. Standard users never see or request
  the SyntheticLib4 proof-search corpus before launch, and the backend enforces it:
  the central corpus filter includes SyntheticLib4 only when launched (either mode)
  or under local preview authority. Developer tooling (ledger status/listing/
  reconcile/cancel/archive-reset/archive-delete, authorization readiness) always
  needs local preview authority, even after launch; the offer panel's evidence,
  snapshot, prepare, and legacy-port migration calls use product authority. Generic mode
  shows no prelaunch integration and never shows proof-trade or developer tooling.
- **SyntheticLib proof identity**: Desktop proof-trade groundwork uses a random
 installation identity held outside configurable instance/keyring namespaces plus
 an explicit non-secret data-root binding. Canonical proof stores persist random
 store IDs; newly stored occurrences receive immutable opaque client proof IDs and
 versioned locators binding installation, root, store, run, local proof ID, and
 canonical theorem/Lean hashes. Manual archival preserves these identities.
 Legacy incomplete records remain unqualified until the explicit legacy-port migration, derived proof-search indexes never
 mint identity, copied/moved roots fail closed until explicit recovery, and generic
 mode gains no trade authority.
- **SyntheticLib proof-time evidence**: Newly verified canonical occurrences
  preserve checker-produced compile receipts bound to exact checked source,
  proof-time toolchain/build metadata, explicit imports, and path-free portable
  dependencies. Legacy evidence is never inferred from current workspace or
  search state. Released-build proofs are ported instead: the explicit legacy-port
  migration (refused during active runs) mints identity plus a digest-bound
  legacy-port receipt, and the auditor's `legacy_port` evidence class waives only
  the compile receipt, novelty receipt, and dependency completeness for them.
- **SyntheticLib eligibility groundwork**: The pre-contract local evidence
  projection reads canonical occurrence files and proof-set revisions without
  proof-search authority. It rejects non-regular, orphaned, ambiguous, or
  revision-racing canonical evidence and reports store-level failures even when
  no occurrence can be loaded. New occurrences carry a canonical digest-bound
  local novelty receipt that distinguishes independent rank, assessment origin,
  group scope, and duplicate overlay without asserting a SyntheticLib rank;
  legacy records are not backfilled. The projection spans every canonical store
  (active/archived manual, Autonomous session, legacy) with cross-store duplicate
  fail-closed checks and applies the frozen `moto-trade-source-selection-v2` policy
  (v2 reads a legacy port's rank from its legacy-port receipt):
  manual, Autonomous, LeanOJ final/subproof, and live-context-pruned occurrences are
  included; nonstandard artifact purposes, `not_novel`, duplicate-overlay occurrences,
  unsupported stores, and later identical copies are excluded. It is metadata-only
  and diagnostics-only: until the SyntheticLib rank mapping and frozen service
  contract exist, no result is eligible, serializable, queued, or transferable, and
  local diagnostics never imply SyntheticLib acceptance or credit.
- **Proof-trade data boundary**: `trade_data_boundary.py` is the single exclusion
  boundary for logs, workflow diagnostics, updater diagnostics, uncaught errors, and
  any future support export/telemetry (which must pass `assert_trade_safe`). The
  base redactor removes credentials and installation/root/batch/selection IDs
  everywhere; trade-namespace loggers also drop paths, local account/host identity,
  occurrence IDs, and tracebacks. Client-visible errors carry the exception type
  only, never its text.
- **Proof-trade ledger**: `proof_trade_ledger.py` is a versioned, metadata-only,
  crash-safe ledger under the active data root (`syntheticlib_trade/`), bound to the
  installation/root binding. Selection snapshots bind canonical hashes, store/record
  keys, and proof-set revision; reconciliation is terminal on delete/move/replace/
  hash/evidence/selection change. Eligibility also fails closed when the Lean
  source embeds credentials, private paths, emails, local identity, or trade IDs.
  Corrupt, unsupported, non-regular, or
  root-mismatched ledgers fail closed and are only recovered by explicit
  archive/reset (no server-deletion claim); confirmed archive deletion unlinks only
  regular archived files. Contract-owned fields (account binding,
  contract version, idempotency key, request digest, receipts, cursors, retry) stay
  null until the frozen contract; batches require canonical eligibility, so none
  can be prepared before launch. Linked/queued/unreadable state blocks installation
  identity reset. Developer-only, authority-gated controls expose metadata-only
  listing, explicit selection, reconcile, prepare (always blocked prelaunch), cancel,
  and confirmed archive/reset; IDs travel in JSON bodies and refusals are reason codes.
- **Proof-trade authorization scaffold**: `proof_trade_authorization.py` freezes only
  plan-fixed facts: public native client, PKCE `S256` via shared `oauth_pkce.py`
  (`plain` rejected), the exact `proof_trade:create`/`proof_trade:read_own` scope set
  (no aliases or corpus scopes), and a distinct keyring record bound to registered
  client plus installation ID (never `syntheticlib4_api_key` or provider OAuth).
  Readiness is metadata-only and authorization stays disabled until the registered
  client profile and frozen contract are ingested; no login, listener, token exchange,
  or credential write exists before then. Proof history's developer-mode **Offer
  proofs for SyntheticLib access** panel selects only canonically `eligible_to_offer`
  proofs after explicit confirmation and records local metadata only.
- **Session History Memory**: The connectivity toggle defaults enabled for new users and maps to local MOTO/manual/LeanOJ proof-history memory used by Assistant workflow-memory search during brainstorming, writing, proof work, and LeanOJ solving. It is not raw provider transcript or chain-of-thought storage. Disabling it persists as non-secret runtime state, removes local proof-history corpora from Assistant retrieval, and must not delete proof records or alter internal retry/rejection/prompt memory.
- **Assistant Role**: Aggregator, Compiler, Autonomous Research, and LeanOJ expose one shared non-blocking Assistant LLM role per workflow surface for verified Lean proof-history support. Autonomous papers-only runs suppress Assistant proof-memory scheduling and injection for the parent and its child Aggregator/Compiler roles without changing the user's persisted Session History setting. Otherwise supports are optional: Assistant cannot redirect or mathematically reinterpret an unrelated parent goal, and no useful proof support is a valid result. Assistant selects up to 7 prior verified proof supports, never replaces validators or submitters, never blocks parent workflows, and is disabled when Session History Memory is disabled. Useful Assistant packs may be reused by two eligible receiver reads before the next refresh. True no-history targets are skipped because Assistant only performs proof-memory retrieval for now. Durable cooldown groups transient task IDs/roles by workflow run while preserving real source/session separation: repeated zero-useful retrieval backs off and may shut down for the run; repeated stagnant same-pack retrieval backs off without shutdown. Top-level Stop clears live/latest pack state, while schema-validated SQLite ranking/goal/pack history can remain rebuildable; explicit Clear/reset or Session History Memory disable also clears durable run-scoped Assistant state. User live activity should show normal Assistant retrieval result logs and explicit Assistant model-output failures, not skip/backoff/shutdown turns. The WorkflowPanel Assistant Memory Bank may show the latest metadata-only up-to-7 pack as novelty-colored proof-history tiles; it must not expose Lean code through pack events.
- **Progressive Solution Path**: Before five cumulative accepted brainstorm events in a top-level run it is absent from prompts and UI. After activation, only existing semantic validators may optionally propose material updates through separately parsed normal-response JSON; this never changes or blocks the primary decision. Updates apply only after transactional serial review by a dedicated role configured exactly from Main Submitter 1. Hard provider/config/context failures enter visible user-repair state rather than retrying forever; after settings repair, the user explicitly retries the generation-fenced proposal through `POST /api/workflow/solution-path/resume`. Stop/crash preserves state, while Clear/new-run is the only count reset and is serialized against reacquisition.
- **Startup provider requirement**: OAuth providers are supplementary role providers, not a standalone startup path, because RAG embeddings route through LM Studio, OpenRouter, or generic-mode FastEmbed. First-run startup and workflow start preflights must require OpenRouter, generic FastEmbed, or LM Studio with an embedding model available before OAuth-only role selection is allowed.
- **OpenRouter Auto-Fill**: OpenRouter selectors fetch provider endpoint metadata and compute host-aware context/output settings from a capable endpoint set. Auto mode ignores known weak hosts (currently Venice) and low/missing-cap outliers before computing context/max-output; manual host selection uses that exact host and its largest exposed endpoint output cap.
- **OAuth Auto-Fill**: OAuth model selectors auto-fill only from provider model metadata, documented provider-specific limits, or curated provider-backed public aliases. Do not invent generic fallback context windows for unknown OAuth models; preserve current settings when metadata is unknown. GPT-5.5 Codex uses the Codex 400K product window, not the 1M regular API window; Codex Spark high is exposed as a curated alias for Codex Spark with high reasoning and its documented 128K window. Grok/SuperGrok OAuth uses xAI model metadata when available and may expose known Grok subscription limits only for known model IDs; model listings must filter xAI catalog entries that are not accepted by the OAuth chat-completions route, such as multi-agent-only models.
- **OpenRouter Reasoning Effort**: Every OpenRouter role exposes a visible reasoning-effort selector. Default `auto` sends maximum OpenRouter reasoning effort (`xhigh`) through the normalized `reasoning.effort` request object; users may lower it or set `none`.
- **Codex Reasoning Effort**: Codex roles reuse the legacy `openrouter_reasoning_effort` fields throughout starts, runtime roles, and proof snapshots. Literal `max` is accepted only with `openai_codex_oauth`; OpenRouter and Boost retain their existing effort enums. Codex effort resolution follows model-supported metadata and must never silently promote an explicit lower selection.
- **xAI Grok Reasoning Effort**: xAI OAuth roles expose `auto`, `high`, `medium`, `low`, and disabled effort choices through the legacy reasoning field. `auto` maps to xAI `high`; disabled omits the request setting.
- **LM Studio Fallback** (default mode only): Optional fallback per role on credit exhaustion
- **Free Model Cooldown Handling**: SERIAL BOTTLENECK pause, free model looping, and auto-selector backup (see below)
- **Boost Mode**: Selective task acceleration via next-count, category, always-prefer, and per-task routing controls, using either an explicit boost override key or the active global OpenRouter key:
  - **Boost Next X Calls**: Counter-based, next X API calls regardless of task ID
  - **Category Boost**: Role-based, boosts all calls for exposed Aggregator, Compiler, autonomous parent-role, and selected LeanOJ category presets. Autonomous proof task IDs and non-exposed LeanOJ helper prefixes are boostable through Boost Next X, Always Prefer, or exact task IDs, but are not separate category presets unless added to `/api/boost/categories`.
  - **Always Prefer Boost**: Attempts boost for every API call, falling back to the primary route on boost failure
  - **Per-Task Toggle**: Legacy task-ID boost controls for individual workflow tasks
- **Supercharge**: Per-role setting that wraps one role answer as 4 parallel diversified full answer attempts plus a 5th same-model deterministic synthesis answer. If Boost applies, all 5 calls use the Boost route/model/provider/settings.
- **Creativity Emphasis Boost**: Developer-gated brainstorm prompt mode for Aggregator, Autonomous Aggregator-backed brainstorm/title/topic exploration, and LeanOJ topic/brainstorm submitters. It is prompt pressure only, not routing or concurrency, and marks accepted/rejected activity with `creativity_emphasized`.
- **System works without LM Studio**: Defaults to OpenRouter when LM Studio unavailable; generic-mode inference and embeddings never route through LM Studio, though shared legacy diagnostics may still exist and should not be used by hosted UI

## Mode-Specific Behavior

| Behavior | Default Mode | Generic Mode |
|----------|-------------|--------------|
| LLM provider | LM Studio + OpenRouter + desktop OAuth providers + fallback | OpenRouter only |
| Embeddings | LM Studio → OpenRouter fallback | FastEmbed (in-process) |
| LM Studio UI | Shown (provider toggle per role) | Hidden (frontend checks `/api/features`) |
| Secret persistence | OS keyring via `secret_store.py` | Env-injected/in-memory; keyring bypassed |
| `Authorization` header | Used for OpenRouter key passthrough | Same; NOT reused for sandbox proxy auth |
| Credential restore | OpenRouter key and desktop OAuth tokens restored from keyring; only OpenRouter or LM Studio satisfies first-run startup | Loaded from `OPENROUTER_API_KEY` env var if present |

---

## Architecture Components

### Boost and Parallel Execution

**Boost is a ROUTING decision, NOT a CONCURRENCY decision.**
- Boost affects which API endpoint is used, NOT whether submitters run in parallel or serial
- Aggregation submitters ALWAYS run in parallel regardless of boost status (unless single-model mode)
- LeanOJ topic and brainstorm submitters ALWAYS run in parallel regardless of boost/provider routing; validation batches up to 3 topics/submissions.
- Single-model mode: triggered when all submitters AND validator use the SAME configured model ID, except when LM Studio has multiple loaded same-base numeric `:#` instances (for example `model:1`, `model:2`) for that model; in that case submitters may run in parallel and the LM Studio client routes independent calls to idle sibling instances. Boost routing does NOT trigger single-model mode.

### Supercharge

**Supercharge is a per-role answer-quality wrapper, NOT a routing mode.**
- `ModelConfig.supercharge_enabled` and related request fields control it per role.
- Frontend Supercharge controls are developer-mode-only; start/generate request payloads must force Supercharge off unless developer mode is active.
- `api_client_manager.generate_completion()` is the only implementation choke point: checked roles run 4 parallel full answer attempts, then a 5th synthesis call that receives the original messages plus the prior 4 outputs and returns the single role answer.
- Calls 1-4 must be full answer attempts in the original role format/schema, not short notes.
- Calls 1-4 intentionally violate the default deterministic temperature rule with the fixed ladder `[0.0, 0.2, 0.4, 0.8]` so concurrent candidates diversify; the synthesis call stays `temperature=0.0`.
- The synthesis call must produce the final answer in the exact original required format/schema and must not mention Supercharge or candidate attempts.
- Tool-call requests (`tools` or `tool_choice`) bypass Supercharge because assistant/tool turn pairing must remain exact.
- If Boost applies to the original task, all 5 Supercharge calls force the same Boost mode and Boost config first; Boost failures are strict for Supercharge and must not silently mix in the primary route. `boost_next_count` is consumed once for the successful boosted overall Supercharge answer, not once per internal attempt.

### Creativity Emphasis Boost

**Creativity Emphasis Boost is prompt steering, NOT a routing or concurrency mode.**
- The developer-gated start flag `creativity_emphasis_boost_enabled` adds the creativity block every fifth valid submission slot per submitter in supported brainstorm loops.
- The extra prompt block must be included in context budgeting; if it alone would overflow a role's configured input budget, skip the creativity block for that slot and run the normal prompt instead.
- It never changes validator prompts, provider selection, Boost routing, Supercharge behavior, or submitter parallelism.
- Frontend payloads must force it off when developer mode is disabled until the control is intentionally promoted to standard mode.

### Backend Core

#### OpenRouterClient (`backend/shared/openrouter_client.py`)
- Async HTTP client. Base URL: `https://openrouter.ai/api/v1`
- App Attribution Headers: `HTTP-Referer: https://intrafere.com/moto-autonomous-home-ai/`, `X-Title: MOTO Autonomous ASI`
- Credit exhaustion detection: HTTP 402 OR error messages containing "credit", "insufficient", "balance", "quota", "key limit", "limit exceeded"
- Raises `CreditExhaustionError` on exhaustion (no retries). Retries transient errors (max 3).
- Temperature=0.0 default except Supercharge candidate attempts and parallel brainstorm submitter lanes. No stop sequences (removed — caused premature truncation with certain models).
- Exposes both model-level metadata (`/models`) and provider endpoint metadata (`/models/{author}/{slug}/endpoints`) so the UI can compute model-context and endpoint-output-cap OpenRouter auto-fill values.
- Auto-routed calls include a provider `ignore` list for known weak hosts so OpenRouter can still fall back across capable providers. Explicit user-selected providers use `provider.order=[provider]` with `allow_fallbacks=false` so requests cannot silently fall back to a host whose limits were not used for settings.

#### APIClientManager (`backend/shared/api_client_manager.py`)
- Central router for all API calls: optional Supercharge wrapper → boost check → role's configured cloud provider (OpenRouter or desktop OAuth providers, with LM Studio fallback) → LM Studio (default mode) or OpenRouter-only (generic mode)
- Temperature policy exceptions live here: Supercharge attempts use `[0.0, 0.2, 0.4, 0.8]`; parallel brainstorm submitter lanes use `[0.0, 0.1, ..., 0.9]`. Validators, compiler roles, proof/final roles, JSON retries, and single-model sequential submitters stay `0.0`.
- LM Studio instance sharing lives below this router in `lm_studio_client.generate_completion()`: only default-mode LM Studio calls can share same-base loaded numeric `:#` siblings, response metadata must preserve both the configured model and effective instance, and state-sensitive workflow ordering must not change.
- OpenAI-compatible assistant text extraction is centralized in `response_extraction.py`: prefer `message.content`, then compatibility fallback fields such as `reasoning`/`thinking` when content is empty so local reasoning-only endpoints still work under one auditable policy. A reply that hit the output-token limit before any answer has only unfinished thinking: the fallback never applies to it, and `_generate_completion_once()` strips those fields so callers, retries, and Supercharge see an empty answer.
- Raw provider/model transport output must never be replayed into MOTO retry prompts, feedback memory, accepted memory, RAG, or durable context. Conversational retries are required, but failed-output context must first pass `sanitize_model_output_for_retry_context()` so only reusable visible answer text remains. The sanitizer strips known private thought/channel/control tokens only as transport scaffolding outside visible JSON/string content, not ordinary visible Lean/math/operator syntax such as `<|` or literal visible marker text such as `<|channel>final` / `<channel|>` inside content.
- Parser exception strings that are inserted into retry prompts must not contain raw response excerpts; raw excerpts are allowed only in logs/observability surfaces that are never reused as model context.
- Persisted observability logs always contain bounded, secret-redacted prompt/response previews, hashes, and metadata; they are not a general PII or confidential-data filter. In desktop mode, `MOTO_API_LOG_STORE_FULL_PAYLOADS` / `API_LOG_STORE_FULL_PAYLOADS` enables full prompt/response inspection only through volatile in-process memory for the current backend process; it never writes full payloads to disk. Generic mode keeps full-payload inspection disabled. Legacy persisted full-payload fields are scrubbed when the logger initializes.
- Tool-call assistant/tool protocol turns are the only exception where exact assistant content/structure may need preservation; ordinary JSON retry assistant turns are not tool protocol turns and must use sanitized retry context.
- Generic mode must normalize or reject LM Studio role configs and must never fall through to `lm_studio_client.generate_completion()`, even if a direct API caller submits legacy `provider="lm_studio"` or an LM fallback value.
- Desktop OAuth providers and Sakana Fugu (`sakana_fugu`) are distinct from regular provider API-key billing. OpenAI Codex OAuth (`openai_codex_oauth`) uses ChatGPT/Codex account tokens against `https://chatgpt.com/backend-api/codex`, streaming each completion over the Responses WebSocket first (the provider closes the HTTP/SSE path at roughly 15 minutes) and falling back to SSE for that request when the handshake is rejected, with unsupported-status rejections disabling WebSocket for the process, and requesting reasoning summaries so long thinking phases keep the stream active; a stream cut before terminal completion adds plain-language retry activity naming the server-side close and the lower-reasoning-effort/other-provider repair hint; xAI Grok OAuth (`xai_grok_oauth`) uses current `auth.x.ai` PKCE login with `grok-cli:access` / `api:access` scopes and subscription tokens against the xAI OpenAI-compatible chat-completions API; API-key subscriptions share `SubscriptionApiKeyClient` (environment override, then one keyring entry; honest `MOTO-Autonomous-ASI/<version>` User-Agent; provider keys scrubbed from all error text): Sakana Fugu (`sakana_fugu`) uses `/responses` with `/chat/completions` for tool turns and prefers `MOTO_SAKANA_FUGU_API_KEY` over legacy `SAKANA_API_KEY` / `FUGU_API_KEY`; Kimi Code (`kimi_code`) uses a Kimi Code console membership key (`MOTO_KIMI_CODE_API_KEY`, never the pay-as-you-go `KIMI_API_KEY`) with streamed `/chat/completions`, must never impersonate another client's User-Agent, detects its international or China endpoint on save and persists that non-secret region in runtime settings, maps effort to `low|high|max` or `thinking: disabled`, and re-attaches in-memory `reasoning_content` to its own tool-call turns (thinking disabled when unknown). xAI Console API keys are a separate pay-as-you-go/credit path, not the subscription-backed OAuth path. OAuth tokens are stored through chunked OS-keyring entries; callback input is bounded, listeners are loopback-only and expire without requiring later API traffic, transient exchange/persistence failure retains pending PKCE state until expiry, and terminal auth failure consumes it. Provider status is non-secret, and these desktop-only providers remain unavailable in generic mode. Hosted/generic settings must not poll desktop OAuth/subscription status/model endpoints when `/api/features` marks those capabilities unavailable.
- Codex model routing must preserve provider-listed model IDs verbatim from selection through the outbound request. Only explicit MOTO-owned aliases may normalize to another provider model, and each such alias must retain distinct metadata and have outbound-payload regression coverage; provider-listed variants must never be silently collapsed into a different model.
- Generic mode: `get_embeddings()` early-returns to `FastEmbedProvider` before the LM Studio → OpenRouter fallback chain
- Tracks fallback state per role: `_role_fallback_state: Dict[str, str]`
- `reset_openrouter_fallbacks()`: Resets all roles originally configured for OpenRouter back from LM Studio fallback. Called automatically on API key set, or manually via reset endpoint.
- Lazy initialization: OpenRouter client initializes from `rag_config.openrouter_api_key` when first needed

**CRITICAL REQUIREMENT - Role Configuration:**
- **EVERY role calling `api_client_manager.generate_completion()` MUST be configured via `api_client_manager.configure_role()`**
- Configuration is exact-role-ID scoped, including source-, mode-, and proof-run-suffixed IDs. Public completion calls fail closed with typed `role_not_configured` repair guidance before Assistant, Boost, Supercharge, logging, or provider access; an unknown role must never default to LM Studio.
- This includes: aggregator submitters/validator/Assistant, compiler Writing/Rigor & Proofs/validator/Assistant, autonomous agents/Assistant, Tier 3 final answer agents, and LeanOJ roles/topic/brainstorm submitters/Assistant
- Role configs must preserve `supercharge_enabled` when copied into proof snapshots, manual proof helpers, child Aggregator/Compiler coordinators, and LeanOJ grouped roles.
- **Proof agents (Part 3/manual, optional)** use the visible Rigor & Proofs Submitter settings for proof-solving submitter work and Validator settings for scoped integrity, novelty, and pruning-validation roles. Every emitted role ID is configured from the matching `ProofRuntimeConfigSnapshot` or caller config, preserving provider, host, reasoning, fallback, context/output, and Supercharge. Semantic pruning copies the Rigor & Proofs `paper` slot and independent validation copies `validator`; its bounded evidence, conservative arbitration, and commit fences remain unchanged. Manual checks require `lean4_enabled=True` and either a stored or request-provided runtime snapshot.
- Assistant is the preferred owner of routine proof-history memory retrieval for eligible non-validator, non-critique flows. Each refresh deterministically fuses concurrent enabled-local-history, versioned canonical theorem/code identity → composite occurrence/run neighborhood, and SyntheticLib4 lanes without quotas; publisher integrity hashes remain separate. Exact neighborhoods are bounded, non-recursive, and current-run/session excluded. A typed standalone exact-duplicate-emphasis purpose is persisted on the exact cross-run overlay occurrence and excluded before fusion; ordinary `duplicate_novel` occurrences stay public-search/Assistant eligible. Reused packs are policy-versioned and run/session re-filtered; mixed supports drop marked occurrences individually and survive when an eligible occurrence remains. Group-scoped novelty anchors are excluded before final Assistant support selection until complete-group atomic hydration is available, so they cannot consume pack capacity or deduplication slots; explicit proof-search tool results retain scope, combined claim, and member IDs. Assistant candidate fusion and every fresh, cached, latest, stale, or provider-cooldown pack reuse apply active-context-run pruning before hydration and injection. A pruned occurrence is excluded only for that context-owning run; future runs remain eligible to retrieve it, and an eligible occurrence in a mixed fused support keeps that support usable. Reciprocal-rank evidence is aggregated per artifact before an at-most-64 → 21 reduction; the normal configured-provider path makes at most one non-Supercharged Assistant selection call for up to 7 supports, while unavailable/no-history/no-candidate/cooldown paths may make none. Two concrete receiver reads, run-scoped cooldown/provider behavior, and narrow emergency-only in-role search remain unchanged.

**Boost Mode Priority** (`should_use_boost(task_id)`):
1. Always Prefer Boost: `boost_always_prefer=True` → True
2. Boost Next X: `boost_next_count > 0` → True
3. Category Boost: `_extract_role_prefix(task_id) in boosted_categories` → True
4. Per-task toggle: exact task ID is enabled → True

**Counter Decrement:** `boost_next_count` decrements ONLY on successful boost API calls. Failed/exhausted calls do NOT decrement.

**Resettable Fallback / Proof Pause:** When a role hits credit exhaustion, it falls back to LM Studio for subsequent calls when fallback is configured; both the immediate fallback call and later calls in persisted fallback state use that role's configured `lm_studio_fallback_id`, never the cloud model ID. Proof workflows (LeanOJ and autonomous proof checkpoints) with no fallback preserve their workflow checkpoint/state and enter a provider-credit pause instead of failing proof progress. `POST /api/openrouter/reset-exhaustion` clears fallback/free-model exhaustion and wakes currently waiting in-process proof workflows; stopped/restarted runs rely on their persisted LeanOJ/proof checkpoint resume paths. Each role has independent fallback state.

Manual `POST /api/proofs/check` requests default to `one_round`; process-local `loop_with_pruning` runs successive rounds until explicit Stop, backend shutdown, hard provider/configuration repair, or ten consecutive valid rounds discover no candidates. A round with candidates resets the streak, and live activity reports each empty round and the normal limit completion. A later request starts a fresh loop. Manual proof runs expose no durable idle, restart-resume, or next-round controls. Provider reset may wake an in-process provider pause without creating durable manual resume state. Their provider/API activity names workflow mode `manual_proof_check`, never the concurrently active workflow's; their retry-wait rows also carry (and persist) the run's source/proof-run IDs so the UI shows each once in the feed that owns the checked source. Autonomous and LeanOJ checkpoint recovery remain unchanged. Autonomous history-paper checks use `source_id="{session_id}:{paper_id}"` and remain bound to that original session's paper and proof store; they must not write into the active session or manual proof database.

**Categories from task ID prefix:**
- `agg_sub{N}` / `agg_val` → Aggregator and autonomous parent-role tasks
- `comp_writer` / `comp_hp` / `comp_val` → Compiler roles; `comp_hp` is Rigor & Proofs, legacy `comp_crit` and `critique_sub*` alias to `comp_hp`, while critique validator/cleanup task IDs route with `comp_val`
- `proof_*` → autonomous proof task IDs (not exposed in the category list unless added explicitly)
- Exposed LeanOJ category presets cover topic generation/validation, brainstorm submitters/validator, sufficiency, path validation, and final solver. `leanoj_path_*` path-decision calls are absorbed into the Final Solver category; other LeanOJ helper prefixes (for example prune review, master-proof edit validation, final review, and proof-novelty tasks) use Boost Next X, Always Prefer, or exact task IDs unless promoted to category presets.

#### BoostManager (`backend/shared/boost_manager.py`)
- Singleton. Key methods: `set_boost_config`, `clear_boost`, `set_boost_next_count`, `toggle_category_boost`, `set_always_prefer`, `toggle_task_boost`, `should_use_boost` (main check for coordinators), `consume_boost_count` (only after successful boost call)
- Boost can use an **explicit override** OpenRouter API key in process memory only, or it falls back to the active global OpenRouter key. Boost state persistence must never write provider key material; legacy persisted boost keys are scrubbed on load. A temporary `OpenRouterClient` is created per boosted task and closed immediately after.
- **Autonomous agent task ID inheritance**: All autonomous orchestration agents use parent role task ID prefixes — Topic Selector/Completion Reviewer/Reference Selector/Paper Title Selector/Tier 3 agents use `agg_sub1_*`; Topic Validator/Redundancy Checker use `agg_val_*`. Boosting a parent role automatically covers all autonomous agents that run on that model. **Proof agents are the exception**: they use their own prefixes (`proof_id_*`, `proof_lemma_*`, `proof_form_*`, `proof_integrity_*`, `proof_novelty_*`, `proof_prune_propose_*`, `proof_prune_validate_*`, `proof_framing_gate_*`) because they run under the `autonomous_proof_*` role IDs with distinct runtime-snapshot configs; Aggregator/Validator category boosts do NOT cover proof agents, and proof prefixes are not currently listed by `/api/boost/categories`.

#### BoostLogger (`backend/shared/boost_logger.py`)
- Singleton. Log file resolves under the active instance data root (default desktop path: `backend/data/boost_api_log.txt`)
- Methods: `log_api_call`, `get_logs(limit)`, `clear_logs`, `get_stats`
- Boost logs are merged into the main API call log view; persisted/default route output must avoid provider keys and raw full prompt/response bodies.

#### SyntheticLib4 Access, Proof Search, And Proof-Trade Compatibility
- SyntheticLib4 integration is an authorized proof-corpus/search surface, not a model route. MOTO may use hosted APIs, downloaded snapshots, and deltas, but must not clone the private Lean repository or scrape website pages.
- Proof-corpus search, model-visible usage attestations, and proof trade are separate authorities. Searchability or local verification never makes a proof trade-eligible, accepted, credited, owned, or authorized for transfer. Until SyntheticLib publishes the frozen proof-trade contract, native-client profile, and canonical fixtures, live proof trade remains disabled and fail-closed.
- MOTO's credential-free proof-trade compatibility preflight binds the advertised package/build/API contract to a tracked canonical matrix, independently reproduced scoped result, normalized receipt/digest, frozen-asset inventory, diagnostic subject/executor attestation, and closed hashed evidence bundle. A passing MOTO preflight is not a passing SyntheticLib wire-contract test. Unknown versions/enums, contradictory or mutated evidence, unexpected files, symlinks, and unpublished frozen assets fail closed; the matrix verifier must validate the complete decision-bearing unpublished state, not only the headline support flag.
- Clean-source release evidence is separate from deterministic checked-in diagnostic evidence and binds an immutable clean commit/tree plus execution provenance. The local workflow-equivalent runner operates only in an isolated allowlisted temporary snapshot, must not mutate or commit the real repository, and remains diagnostic rather than independently controlled release evidence. Evidence publication and optional retained-output handling must be rollback-safe and non-destructive; only explicitly approved compatibility filenames may be publishable.
- Desktop/default credentials use `secret_store.py`; hosted/generic credentials remain env-injected or in process memory. Snapshot files, runtime settings, prompts, and default logs must never contain SyntheticLib4 tokens or API keys.
- The initial unified proof-search backend indexes active and archived canonical MOTO proof records plus an authorized active SyntheticLib4 snapshot through batched local SQLite/FTS rebuilds; search responses enforce the 7-proof combined result cap before workflow/tool integration. Assistant may gather a wider local candidate pool, but final model-visible proof-support packs remain capped at 7. Normal runtime reports zero SyntheticLib4 records until a snapshot is explicitly activated; fixture records are test-only and never a production fallback.
- Duplicate-novel proof records remain searchable and may remain eligible for separately authorized corpus contribution, but proof-trade selection is governed only by its frozen versioned policy. Neither `duplicate_novel` nor `not_novel` may be inferred as trade-eligible; unified proof search and Assistant retrieval continue to include them unless explicitly filtered out.
- Proof-search service calls must centrally filter disabled corpora for REST, tool, and Assistant paths; disabling Session History Memory removes local MOTO/manual/LeanOJ corpora, and disabling SyntheticLib4 or lacking lifecycle corpus authority removes that corpus even if a client requests it explicitly.
- The shared `search_lean_proofs` tool adapter supports overview/search/hydrate/local usage-attestation actions over the same service. Proof formalization exposes the tool through a bounded provider tool-call loop where supported, while preserving the existing prefetch path for providers that do not emit tool calls; local per-attempt exclusions avoid exact repeat results. Proof formalization records local `model_visible_context` attestations when full SyntheticLib4 Lean code is injected into a successful formalization prompt; whole-code-use submission to SyntheticLib4 remains a later live-service step.
- Local snapshot search remains available when lawfully downloaded and allowed by the recorded license/terms, but refresh, hosted retrieval, user-proof browsing, sharing, redistribution, and public serving require active authorization unless enterprise rights explicitly allow more.
- Full proof payloads should enter model context only through the intentional bounded proof-search result block, and default observability should log metadata, counts, IDs, hashes, and short previews rather than full Lean code.

#### Workflow Task Generation (Internal Backend Tracking)
Coordinators track task IDs internally for boost routing. The frontend does NOT display predicted task lists.
- Aggregator: `agg_sub{N}_{seq:03d}`, `agg_val_{seq:03d}`
- Compiler: `comp_writer_{seq:03d}`, `comp_hp_{seq:03d}`, `comp_val_{seq:03d}`
- Autonomous orchestration: parent-role prefixes (`agg_sub1_{seq:03d}` for topic/completion/reference/title/Tier 3 agents; `agg_val_{seq:03d}` for topic/redundancy validators)
- Autonomous proof/framing: `proof_framing_gate_{seq:03d}` for the prompt-framing decision, plus Lean-gated proof work IDs `proof_id_{seq:03d}`, `proof_lemma_{seq:03d}`, `proof_form_{seq:03d}`, `proof_integrity_{seq:03d}`, `proof_novelty_{seq:03d}`

---

## Coordinator Integration

All coordinators track workflow via: `workflow_tasks`, `completed_task_ids`, `current_task_sequence`, `current_task_id`.

Key methods: `refresh_workflow_predictions()`, `get_next_task()`, `mark_task_completed(task_id)`, `mark_task_started(task_id)`.

Predictions refresh: after initialization, each task completion, mode switches, phase transitions.

---

## WebSocket Events

Workflow, boost, fallback/reset, provider pause/resume, hung-connection, rate-limit, privacy-policy, and unrecoverable OAuth provider failures should emit user-visible notifications when the frontend or hosted wrapper depends on them. Every actual provider/API retry wait—including provider-internal retries, workflow backoff, timed usage cooldown waits, and all-free-model exhaustion waits—uses the shared atomic emit-and-stop-aware-wait seam and adds scoped Live Activity with route identity (workflow mode, role, model, provider, non-primary route kind, and for OpenRouter the configured pinned/auto host plus the upstream host OpenRouter reported when known), retry attempt, wait duration, one stable correlation per retry episode, distinct attempt identity, and bounded secret-redacted provider error detail when available, without replacing existing logs or specialized events. Separate API calls must not share a retry correlation. If durable notification persistence fails, live delivery continues with explicit degraded-durability status. Manual-reset pauses remain separately typed because they have no fixed wait duration. OAuth failure notifications must tell the user to check OpenRouter/OAuth, sign in again, and retry; live activity should also include the redacted provider error detail capped at 1800 chars. Recent non-secret OAuth/provider failure notifications are also persisted under the active data root so the UI can recover missed live popups after reconnect or reload. Hung-connection alerts should also appear in the active mode's live activity feed and concisely note that the model may still be thinking, the user can keep waiting, and reasoning effort can be lowered in Settings. Keep consumed event payloads stable enough for UI recovery, but do not treat every internal notification name as a rule-level invariant.

---

## API Endpoints

### Boost (`backend/api/routes/boost.py`)
- `POST /api/boost/enable` — Enable boost (BoostConfig body)
- `POST /api/boost/update-model` — Update boost model/config while preserving boost state
- `POST /api/boost/disable` — Disable boost, clear all modes
- `GET /api/boost/status` — Current active boost config/state and boosted category IDs; available category definitions come from `/api/boost/categories`
- `POST /api/boost/set-always-prefer` — Toggle Always Prefer Boost
- `POST /api/boost/set-next-count` — Set Boost Next X counter `{ "count": int }`
- `POST /api/boost/toggle-category/{category}` — Toggle category boost
- `POST /api/boost/toggle-task/{task_id}` — Toggle a legacy exact task boost
- `GET /api/boost/categories?mode=` — All categories (mode param ignored, always returns all)
- `GET /api/boost/openrouter-models` — Fetch OpenRouter models (Bearer key header)
- `GET /api/boost/model-providers?model_id=` — Providers + endpoint metadata for a model
- `GET /api/boost/logs?limit=` — Recent boost-only logs (debug)
- `GET /api/boost/logs/{index}` — One boost log entry by index
- `POST /api/boost/clear-logs` — Clear logs

### OpenRouter (`backend/api/routes/openrouter.py`)
- `GET /api/openrouter/lm-studio-availability` — LM Studio availability check
- `POST /api/openrouter/set-api-key` — Set and validate global OpenRouter key (auto-resets exhaustion flags)
- `POST /api/openrouter/reset-exhaustion` — Reset all credit exhaustion flags + role fallback states mid-session and wake currently waiting provider-paused proof workflows
- `DELETE /api/openrouter/api-key` — Clear key
- `GET /api/openrouter/api-key-status` — `{ success, has_key, enabled, validation_state }`
- `POST /api/openrouter/validate-current-key` — Explicitly authenticate the currently configured key and refresh its validation state
- `GET /api/openrouter/models` — Available models (also caches free models for rotation); temporary keys must use `Authorization: Bearer`, never URL query parameters
- `GET /api/openrouter/providers/{model_id}` — Providers + endpoint metadata for model
- `GET /api/openrouter/free-model-settings` — `{ looping_enabled, auto_selector_enabled, ... }`
- `POST /api/openrouter/free-model-settings` — Update free model settings (body: `FreeModelSettings`)
- `POST /api/openrouter/test-connection` — Test key without storing
- `GET /api/model-cache` — Cached model ID mapping (display_name → api_id)

### OpenRouter/OAuth (`backend/api/routes/cloud_access.py`)
- `GET /api/cloud-access/status` — Non-secret OAuth provider status used by the OpenRouter/OAuth modal and connectivity panel
- `GET /api/cloud-access/provider-notifications` — Recent non-secret OAuth/provider failure notifications for missed-popup recovery
- `GET /api/cloud-access/provider-recovery/events` — Typed durable alias for the same recovery/notification events; the legacy provider-notifications route remains compatible
- `POST /api/cloud-access/openai-codex/oauth/start` — Start desktop OpenAI Codex OAuth PKCE login and loopback callback listener
- `POST /api/cloud-access/openai-codex/oauth/exchange` — Exchange pasted callback URL/code for Codex OAuth tokens
- `GET /api/cloud-access/openai-codex/status` — Non-secret OpenAI Codex OAuth status
- `GET /api/cloud-access/openai-codex/models` — Codex-backed model list for the signed-in account, including normalized context/output metadata when known
- `DELETE /api/cloud-access/openai-codex` — Revoke best-effort and clear stored OpenAI Codex OAuth tokens
- `POST /api/cloud-access/xai-grok/oauth/start` — Start desktop xAI Grok/SuperGrok OAuth PKCE login and loopback callback listener
- `POST /api/cloud-access/xai-grok/oauth/exchange` — Exchange pasted callback URL/code for xAI Grok OAuth tokens
- `GET /api/cloud-access/xai-grok/status` — Non-secret xAI Grok OAuth status
- `GET /api/cloud-access/xai-grok/models` — Grok model list for the signed-in account, including normalized context/output metadata when known
- `DELETE /api/cloud-access/xai-grok` — Revoke best-effort and clear stored xAI Grok OAuth tokens
- `GET /api/cloud-access/{slug}/status`, `POST /{slug}/api-key`, `GET /{slug}/models`, `DELETE /{slug}` — Mounted per API-key subscription (`sakana-fugu`, `kimi-code`): non-secret status (with detected region), validate-then-store key (validation never mutates state), model list, clear
- `GET /api/cloud-access/kimi-code/usage` — Kimi Code quota pools (used ratio, reset time) from Kimi's `/usages`; fetched only on explicit open/refresh

### Workflow (`backend/api/routes/workflow.py`)
- `GET /api/workflow/predictions` — Current workflow mode (also returns tasks for internal use)
- `GET /api/token-stats` — Cumulative token usage (total_input, total_output, by_model, elapsed_seconds)

### Connectivity (`backend/api/routes/connectivity.py`)
- `GET /api/connectivity/status` — Non-secret grouped status for OpenRouter/OAuth, LM Studio, SyntheticLib4, local proof-history Session History Memory, Wolfram Alpha, and Boost
- `POST /api/connectivity/toggles` — Persist non-secret optional-skill enablement (`syntheticlib4_enabled`, `agent_conversation_memory_enabled`, `wolfram_alpha_enabled`) without clearing credentials, snapshots, indexes, or history

### Proof Search (`backend/api/routes/proof_search.py`)
- `GET /api/proof-search/overview` — Compact corpus/index map for proof navigation, filtered by enabled proof-history/SyntheticLib4 corpus toggles
- `POST /api/proof-search/search` — Search canonical MOTO proof records and SyntheticLib4 fixture/snapshot records, with public request schema and runtime results capped at 7 combined records
- `GET /api/proof-search/proofs/{source}/{proof_id}` — Fetch one indexed proof record, hydrating SyntheticLib4 fixture code when available
- `POST /api/proof-search/reindex` — Rebuild the local proof-search SQLite/FTS index

### SyntheticLib4 (`backend/api/routes/syntheticlib4.py`)
- `GET /api/syntheticlib4/status` — Non-secret auth/snapshot/index status for local SyntheticLib4 corpus access
- `GET /api/syntheticlib4/releases` — Locally available SyntheticLib4 releases
- `POST /api/syntheticlib4/refresh` — Validate local snapshot metadata and rebuild the unified proof index
- `POST /api/syntheticlib4/import-local-snapshot` — Safely activate a local snapshot staged under the active data root, preserving the previous active snapshot if validation/import fails
- `POST /api/syntheticlib4/reindex` — Rebuild the unified proof index from available local corpora
- `POST /api/syntheticlib4/retrieve-batch` — Bounded mock/offline retrieve-batch surface matching the planned SyntheticLib4 corpus contract, with public request schema capped at 7 records
- `GET /api/syntheticlib4/account/proofs`, `/account/proofs/search` — Mock/offline accepted-proof browsing/search surfaces. Current bundled frontend keeps SyntheticLib4 user access as a coming-soon explainer until production corpus access is enabled.
- `POST /api/syntheticlib4/api-key`, `DELETE /auth` — Store/clear SyntheticLib4 API-key credentials through the active mode's secret path while live validation is pending
- `POST /api/syntheticlib4/auth/start`, `/auth/exchange` — Production OAuth placeholders that fail clearly until SyntheticLib.com auth is live
- `GET /api/syntheticlib4/developer/trade-evidence` — Product-authority-gated (shared with the offer panel), read-only, metadata-only canonical-store evidence/source-policy audit; never eligible, never contacts SyntheticLib
- `GET /api/syntheticlib4/developer/trade-authorization` — Developer-tooling-gated, read-only, metadata-only proof-trade authorization readiness (PKCE method, required scopes, keyring/installation/profile presence, reason codes); `authorization_enabled=false` and `upload_enabled=false`
- `GET /api/syntheticlib4/developer/trade-ledger` — Developer-tooling-gated, read-only local proof-trade ledger status (availability/reason code, state and archived-ledger counts); never creates identity or ledger files, uploads always disabled
- `POST /api/syntheticlib4/developer/legacy-port/migrate` — Product-authority-gated explicit legacy-proof migration (`confirm`, `dry_run`); metadata-only per-store counts, 409 reason-code refusals while runs are active, never uploads
- `GET /api/syntheticlib4/developer/trade-ledger/entries`; `POST .../trade-ledger/snapshots`, `/snapshots/reconcile`, `/snapshots/cancel`, `/batches/prepare`, `/batches/cancel`, `/archive-reset`, `/archive/delete` — Ledger controls (snapshot create and batch prepare product-authority-gated; the rest developer-tooling-gated); metadata-only, 409 reason-code refusals, `upload_enabled=false`, and archive/reset/delete never claim server-side deletion

---

## Error Handling

**Credit Exhaustion:** HTTP 402 or keywords "credit"/"insufficient"/"balance"/"quota"/"key limit"/"limit exceeded" → `CreditExhaustionError` → default mode: LM Studio fallback for that role when configured; proof workflows with no fallback checkpoint progress and pause, while currently waiting tasks can be woken by `POST /api/openrouter/reset-exhaustion`; ordinary non-proof generic-mode calls still raise provider/config errors when no fallback exists. Fallback state is resettable via `POST /api/openrouter/reset-exhaustion` or by re-setting the API key.

**Retryable provider failures:** transient gateway/timeout/stream disconnect responses, successful OpenRouter JSON responses without a non-empty completion `choices` list, and OpenRouter's exact opaque upstream `404 Provider returned error` shape (named upstream provider with no raw detail) use provider-adapter retries first, then raise the shared `RetryableProviderError` so workflow loops use one exponential backoff path and preserve checkpoints instead of becoming rejects, `no_candidates`, or stops. Invalid choices must be rejected inside the provider adapter rather than returned to role parsers. Compiler generation loops must preserve typed transient failures for shared recovery while routing provider context-limit failures through the existing context-overflow lifecycle. A wrapper carrying specific raw detail follows normal hard/transient classification and must not be made retryable merely by its outer message. Codex Responses streams (WebSocket or SSE) are consumed incrementally and succeed only after explicit terminal completion; confirmed completion survives later framing failure, while terminal-less disconnects retain an unknown remote outcome and retry sequentially. Codex classification prefers structured status/code/failure-kind metadata: transport, transient-status, interrupted/empty-stream, and metadata-free ambiguous completion failures retry, while deterministic 4xx, auth, model, context, entitlement, and usage-limit failures keep their dedicated hard/cooldown paths. All actual retry waits expose additive scoped activity with bounded secret-safe detail; manual proof runs additionally retain their same-round restart activity. Every provider/API-error Live Activity line carries that route identity in structured fields and one composed `route_identity` prefix on the rendered line; the composed `route_identity` and upstream host/request id never enter exception or classification text, `error_summary` (except quoted provider-reported upstream detail), backoff/correlation keys, or `notification_key`, and non-model workflow-loop retries name mode and role only. Auth/config failures remain hard; when an OAuth model call cannot recover through LM Studio fallback, broadcast a user-visible OAuth reconnect notification with a stable provider/role/reason/model `notification_key` so user-dismissed alerts do not replay after restart.

Validator and proof-classifier calls use this same provider recovery policy for OpenRouter, LM Studio, OpenAI Codex OAuth, xAI Grok OAuth, and every API-key subscription provider. Provider failures must never become semantic rejection decisions. After Lean accepts a proof, transient alignment/novelty failures retry from the preserved post-Lean artifact; credential or subscription repair restores configured roles from LM Studio fallback where applicable.

**Hard provider spending/entitlement failures:** definitive xAI Grok spending/subscription-limit responses are typed as provider-repair-required failures and bypass transient retries. Other providers follow their existing typed credit, usage-limit, authentication, and configuration paths until explicitly classified. With no fallback, a redundant parallel Aggregator submitter lane is disabled while another lane remains; Main Submitter 1 configuration ownership is never transferred when its generation lane is disabled. The last usable standalone Aggregator lane stops with persisted repair activity, while a progression-critical child Aggregator failure propagates into the Autonomous parent's durable terminal lifecycle state. Persisted provider notifications carry workflow scope so reconnect hydration restores the owning Live Activity entry and de-duplicates the popup by stable notification key.

**Provider Context Limits:** OAuth/provider `context_length_exceeded` / context-window rejections, including LM Studio `n_keep` versus `n_ctx` capacity failures, are hard context mismatch failures, not retryable provider errors. When a prompt owns rolling feedback, retry bounded newest-suffix projections by removing one oldest whole entry at a time while preserving newest feedback; after newest-only rejection, try the configured LM Studio fallback once when present or raise the workflow's context-overflow path. Terminal activity exposes the bounded secret-redacted diagnostic when available and uses generic repair text only as fallback. Do not repeatedly broadcast OAuth failure events.

**Provider usage limits:** Codex, xAI, and API-key subscription timed usage/rate-window responses with credible reset metadata enter one provider-wide in-memory cooldown before generic 429 retries. xAI spending-limit without body reset stays repair-required; HTTP Retry-After is a 429 throttle window, not a spending reset. MOTO emits one durable usage-limit activity per role/window, uses LM Studio fallback when configured, otherwise propagates the typed cooldown to the owning workflow loop for a stop-aware wait until reset, and records one confirmed-resume activity only after the configured provider succeeds again. Aggregator submitters must not swallow cooldowns into rapid `None` retries. Ordinary throttling remains bounded and retryable; untimed entitlement denial requires repair. Assistant refreshes reuse the latest cached/deterministic shortlist during cooldown instead of probing the limited provider.

All provider adapters use the shared typed recovery contract for transient retries, timed limits, fallback duration, context/output failures, and operator repair. Credible reset metadata from any provider, including xAI Grok, enters the same stop-aware cooldown path; untimed spending or entitlement denial remains repair-required. API-key subscriptions route through one generic `api_client_manager` block (per-profile switches preserve entitlement and generic-error fallback policy) and emit `subscription_provider_error` for unrecoverable failures. Each registers a `ProviderAdapter` in the provider-recovery registry and must pass `tests/integration/test_subscription_provider_conformance.py`, which runs over every registered adapter. Kimi's message-keyed 403 usage windows read a credible reset from `/usages`; without one they are repair-required and never retried; its 403 concurrency limit and 402 membership verification retry boundedly.

**Boost Exhaustion:** Falls back to primary for that task; boost stays enabled; counter NOT decremented.

**Rate Limits (Free Models):** HTTP 429 or "rate limit" keywords → 1-hour pause per `:free` model. Handled by Free Model Cooldown system (see below).

**Privacy Policy Errors:** HTTP 404 + "data policy" message → `OpenRouterPrivacyPolicyError`. Role-based: falls back to LM Studio if configured; Boost: raises RuntimeError. Fix: https://openrouter.ai/settings/privacy → enable data sharing.

---

## Free Model Cooldown & Rotation System

**Singleton:** `free_model_manager` in `backend/shared/free_model_manager.py`. Two global settings (both default OFF):
- `looping_enabled` — rotate to next available free model on rate limit (highest context first)
- `auto_selector_enabled` — fall back to `openrouter/free` when all free models are exhausted; context/max-output still come from configured model metadata or user settings, not a hidden 131K default
- Free-model rotation and `free_only` model lists are restricted to text-in/text-out chat-capable models; non-text image/audio free models must not be offered as role/fallback targets.

**Rotation chain** (in `api_client_manager._try_free_model_rotation()` called from RateLimitError handler; keep optional `tools` / `tool_choice` passed through when that helper is used):
1. If `looping_enabled`: **iterate through ALL** non-rate-limited free models (highest context first) using `tried_models` set to avoid re-trying. On each `RateLimitError` or other per-model failure (transient route, no endpoint, invalid response), mark that model failed and continue to the next model; unreachable OpenRouter ends looping. On `CreditExhaustionError`, stop looping.
2. If all looping candidates exhausted and `auto_selector_enabled`: try `openrouter/free`
3. If still failed: check LM Studio fallback (default mode only; generic mode skips this)
4. If no fallback: raise the last transient failure as a route-identified `RetryableProviderError`, otherwise `FreeModelExhaustedError` (carrying structured route identity only); current coordinators use a fixed short retry sleep for ordinary all-options-exhausted cooldowns. Proof workflows pause only for account-credit exhaustion, not ordinary per-model cooldown exhaustion.

`get_alternative_free_model()` accepts optional `skip_models: set` parameter to skip already-tried models.

**SERIAL BOTTLENECK:** When ordinary all-free-models `FreeModelExhaustedError` propagates to coordinators:
- Autonomous coordinator: caught inside the loop, sleeps briefly, then re-iterates naturally
- Compiler coordinator: caught at workflow level, sleeps briefly, then spawns new `_main_workflow()` task via `asyncio.create_task()`
- Aggregator submitters: per-submitter brief retry sleep (others continue); validator loop pauses and retries
- Prevents infinite retry loops (the 2000+ attempt bug)

**Account Exhaustion:** HTTP 402 on any `:free` model sets `_account_credits_exhausted` flag. All subsequent free model calls short-circuit immediately, and the UI shows one account-wide popup rather than one per role. Proof workflows with no fallback checkpoint and pause; in-process waiters wake on `POST /api/openrouter/reset-exhaustion`, and restarted workflows resume from their persisted state. The flag clears on next successful free model call, via reset, or automatically when the API key is re-set.

**Error Classes:**
- `FreeModelExhaustedError` — all options exhausted; account-credit exhaustion is separately marked as a provider pause for proof workflows
- Agents re-raise `FreeModelExhaustedError` through generic `except Exception` blocks

**API Endpoints:**
- `GET /api/openrouter/free-model-settings` — current looping/auto-selector state
- `POST /api/openrouter/free-model-settings` — update settings `{looping_enabled, auto_selector_enabled}`

**WebSocket Events:** Free-model rotation, auto-selector use, free-model exhaustion/retry, and account-exhaustion states should be visible to the UI; exact internal event names are not rule-level invariants unless consumed by the hosted wrapper or frontend contract.

**Frontend:** Aggregator, Compiler, and Autonomous settings expose two OpenRouter free-model fallback controls in their settings panels (currently grouped in the OpenRouter fallback area rather than necessarily adjacent to "Free models only"). Both default unchecked, hydrate from the backend singleton, and persist as non-secret runtime settings under the active data root plus localStorage UI state. LeanOJ settings currently expose the text-chat-only "Free models only" filter but not the looping/auto-selector controls.

---

## Configuration Persistence

Autonomous recommended role defaults apply only when no saved settings exist. Saved missing/invalid roles or limits must fail visibly before Start or profile application, never silently become recommended models. Partial UI updates merge into current configuration; credential setup must not replace selected roles. First-launch LM Studio or OAuth startup may apply those provider role defaults only while still on recommended first-launch routes. Legacy field migrations may preserve user-selected routes but must not invent replacements.

**Secure backend storage (OS keyring — default mode):** OpenRouter global API key and Wolfram Alpha API key persist via `backend/shared/secret_store.py` using the OS keychain/keyring. The keyring service name is derived from `MOTO_SECRET_NAMESPACE`; `None` means the shared desktop service `MOTO-Autonomous-ASI`. `moto_launcher.py` MUST keep plain/default launches on the shared default identity (`backend/data`, `backend/logs`, no keyring namespace, no frontend storage prefix, frontend port 5173) so backend keys, browser profiles/prompts, sessions, and proofs remain visible. Never let port availability, Windows `TIME_WAIT`, stale `.moto_last_instance.json`, or Lean/LM startup timing create a new namespace or data root for a plain relaunch; concurrent default launches must be blocked by active-instance state and a data-root runtime lock.

**Data-root ownership:** FastAPI lifespan holds an OS-backed lease for the active data root before mutable-store initialization and releases it only after shutdown. Launcher `.moto_runtime.lock` remains advisory process metadata and is not the authoritative backend lease.

**Startup key detection:** `backend/api/main.py` restores each desktop credential independently before serving status. Restoration is local-only and does not imply authenticated validity; startup explicitly validates a restored OpenRouter key before treating it as ready. Candidate validation must not mutate active state, desktop set/clear commits durable state before runtime publication, and OAuth token replacement preserves the prior committed bundle until the new bundle is verified. Lean/Mathlib bootstrap must not run during backend startup, proof-status reads, or settings changes.

**Hosted generic mode (no keyring):** Provider keys are env-injected at sandbox launch and/or set via proxied MOTO API routes. `secret_store` persistence is bypassed; keys live in sandbox memory only. Re-injection required after sandbox recreation. `OPENROUTER_API_KEY` env var auto-loaded during lifespan if present.

**Non-secret runtime settings:** `runtime_settings.json` under the active data root persists process-level user settings such as OpenRouter free-model looping/auto-selector and, in desktop/default mode, Lean/SMT proof runtime flags/timeouts. Generic mode keeps proof settings unavailable but may still persist non-secret free-model settings. It must never contain provider keys or prompt/response payloads.

**localStorage:** Key families include workflow/UI preferences (`workflow_panel_collapsed`, `developerModeSettingsEnabled`, `banner_shimmer_enabled`, `startup_provider_choice`, high-score critique seen keys), settings/profile keys (`aggregatorConfig` / `aggregator_settings`, `compiler_settings`, `autonomous_research_settings`, `autonomous_research_profiles`, `leanoj_solver_settings`, `leanoj_solver_profiles`, boost modal settings), bounded live-activity histories, prompt helpers, and related free-model/Supercharge fields. Persisted frontend settings must avoid secret-like fields and bulky refetchable provider metadata; quota recovery may clear refetchable caches while keeping essential settings. Active app mode and tab state are not persisted; a fresh frontend mount starts on the autonomous main interface. Browser storage namespacing is driven by `VITE_MOTO_STORAGE_PREFIX`; launch/control-plane config may supply `MOTO_FRONTEND_STORAGE_PREFIX` and project it into the frontend env.

**Session (in-memory):** fallback state per role, boosted task IDs, boost next count, boosted categories, completed task IDs, free model manager state, any explicit Boost override key, and optional bounded desktop debug payloads. Boost override keys must never be persisted to `boost_state.json`; legacy plaintext keys are ignored/scrubbed on load. Boost logs and non-secret boost routing state persist under the active instance data root (`boost_api_log.txt`, `boost_state.json`) and are merged into the main API call log view. API call logs persist bounded previews/metadata by default, and provider/model error logs must report shape/status metadata instead of raw response bodies. Full persisted payloads are desktop opt-in only and must remain disabled in generic mode.
