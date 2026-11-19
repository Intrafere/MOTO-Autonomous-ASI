// Shared helpers for the process-wide Lean proof runtime settings. Several
// surfaces edit the same settings (Allowed Outputs proof options and the
// advanced proof settings), so every save publishes the new status and each
// surface listens for it instead of keeping a stale copy.

export const PROOF_SETTINGS_CHANGED_EVENT = 'moto:proof-settings-changed';

export const DEFAULT_CONCURRENT_PROOF_ATTEMPTS = 6;
export const MAX_CONCURRENT_PROOF_ATTEMPTS = 1000;

// The settings route replaces the core Lean fields on every save, so a partial
// update starts from the latest status and applies only the requested changes.
export function buildProofSettingsPayload(status = {}, overrides = {}) {
  return {
    enabled: Boolean(status.lean4_user_enabled ?? status.lean4_enabled),
    timeout: status.lean4_proof_timeout ?? 2000,
    lean4_lsp_enabled: Boolean(status.lean4_lsp_enabled),
    lean4_lsp_idle_timeout: status.lean4_lsp_idle_timeout ?? 600,
    max_parallel_candidates: status.proof_max_parallel_candidates ?? DEFAULT_CONCURRENT_PROOF_ATTEMPTS,
    smt_enabled: Boolean(status.smt_enabled),
    smt_timeout: status.smt_timeout ?? 300,
    ...overrides,
  };
}

export function publishProofSettings(status) {
  if (!status || typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(PROOF_SETTINGS_CHANGED_EVENT, { detail: status }));
}

export function subscribeToProofSettings(listener) {
  if (typeof window === 'undefined') return () => {};
  const handler = (event) => listener(event.detail);
  window.addEventListener(PROOF_SETTINGS_CHANGED_EVENT, handler);
  return () => window.removeEventListener(PROOF_SETTINGS_CHANGED_EVENT, handler);
}

// Returns an integer in range, or null when the draft is not a usable count.
export function parseConcurrentProofAttempts(value) {
  const text = String(value ?? '').trim();
  if (!/^\d+$/.test(text)) return null;
  const parsed = Number(text);
  return parsed <= MAX_CONCURRENT_PROOF_ATTEMPTS ? parsed : null;
}

export function proofKindsFromStatus(status) {
  return {
    singleShot: status?.proof_single_shot_enabled !== false,
    master: status?.proof_master_proofs_enabled !== false,
  };
}
