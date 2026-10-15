// Shared Live Activity wording for Lean results and Lean setup failures.

export const LEAN_WORKSPACE_REPAIR_REASON = 'lean_workspace_repair_required';
export const LEAN_WORKSPACE_REPAIR_TITLE = 'Lean 4 setup needs repair';
export const LEAN_WORKSPACE_NOT_A_PROOF_RESULT =
  'Lean 4 could not run (setup problem), so this is not a proof result.';

const LEAN_WORKSPACE_ERROR_PATTERN = /LEAN 4 WORKSPACE ERROR/i;
const PROOF_FEEDBACK_PREFIX = 'Lean 4 proof-attempt feedback (not a MOTO system error):';
const FEEDBACK_FOLLOW_UP = 'The model uses these diagnostics if another attempt follows.';

const identity = (text) => String(text || '');

function withTimeoutHint(text) {
  if (/timed out after/i.test(text) && !/Advanced Settings/.test(text)) {
    return `${text} You can change this timeout in Advanced Settings.`;
  }
  return text;
}

/**
 * Describe one proof attempt's Lean result for a Live Activity line.
 *
 * Builds before the Lean repair stop reported setup failures as proof feedback;
 * those persisted events render as a setup problem, not as a Lean verdict.
 */
export function formatProofLeanResponse(
  data = {},
  { formatResponse = identity, formatError = identity, emptyFallback = '' } = {},
) {
  const leanText = [data.lean_response, data.error_summary, data.error_output]
    .filter(Boolean)
    .join(' ');
  if (LEAN_WORKSPACE_ERROR_PATTERN.test(leanText)) return LEAN_WORKSPACE_NOT_A_PROOF_RESULT;
  if (data.lean_response) return withTimeoutHint(formatResponse(data.lean_response));
  if (data.proof_verified === true) return 'Lean 4 response: proof verified.';
  const error = withTimeoutHint(formatError(data.error_summary || data.error_output || data.reason || ''));
  return error
    ? `${PROOF_FEEDBACK_PREFIX} ${error} ${FEEDBACK_FOLLOW_UP} Proof not verified.`
    : emptyFallback;
}

export const DEFAULT_EMPTY_PROOF_FEEDBACK =
  `${PROOF_FEEDBACK_PREFIX} proof not verified. ${FEEDBACK_FOLLOW_UP}`;

/** Live Activity line for a workflow that stopped because Lean cannot run. */
export function formatLeanWorkspaceRepairActivity(data = {}) {
  const detail = [data.message, data.terminal_guidance].filter(Boolean).join(' ');
  return detail ? `${LEAN_WORKSPACE_REPAIR_TITLE}: ${detail}` : LEAN_WORKSPACE_REPAIR_TITLE;
}

/** Persistent repair popup for a workflow that stopped because Lean cannot run. */
export function buildLeanWorkspaceRepairNotification(data = {}) {
  const workflowMode = String(data.workflow_mode || 'workflow');
  return {
    ...data,
    notification_kind: 'model_error',
    notification_key: data.notification_key || `lean-workspace:${workflowMode}`,
    provider: 'lean4',
    reason: LEAN_WORKSPACE_REPAIR_REASON,
    title: LEAN_WORKSPACE_REPAIR_TITLE,
  };
}
