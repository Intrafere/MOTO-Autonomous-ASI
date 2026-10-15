import { describe, expect, it } from 'vitest';
import {
  DEFAULT_EMPTY_PROOF_FEEDBACK,
  LEAN_WORKSPACE_NOT_A_PROOF_RESULT,
  LEAN_WORKSPACE_REPAIR_TITLE,
  buildLeanWorkspaceRepairNotification,
  formatLeanWorkspaceRepairActivity,
  formatProofLeanResponse,
} from './proofLeanResponse';

describe('formatProofLeanResponse', () => {
  it('describes persisted Lean workspace failures as a setup problem, not proof feedback', () => {
    const legacyEvent = {
      lean_response:
        'Lean 4 proof-attempt feedback (not a MOTO system error): LEAN 4 WORKSPACE ERROR: '
        + 'Mathlib cache/workspace repair failed. Proof not verified.',
    };

    expect(formatProofLeanResponse(legacyEvent)).toBe(LEAN_WORKSPACE_NOT_A_PROOF_RESULT);
    expect(formatProofLeanResponse({ error_summary: 'LEAN 4 WORKSPACE ERROR: broken' }))
      .toBe(LEAN_WORKSPACE_NOT_A_PROOF_RESULT);
  });

  it('keeps ordinary Lean feedback, verified results, and timeout hints', () => {
    expect(formatProofLeanResponse({ error_summary: 'unsolved goals' }))
      .toBe(
        'Lean 4 proof-attempt feedback (not a MOTO system error): unsolved goals '
        + 'The model uses these diagnostics if another attempt follows. Proof not verified.',
      );
    expect(formatProofLeanResponse({ proof_verified: true })).toBe('Lean 4 response: proof verified.');
    expect(formatProofLeanResponse({ lean_response: 'Lean timed out after 600 seconds.' }))
      .toContain('Advanced Settings');
    expect(formatProofLeanResponse({}, { emptyFallback: DEFAULT_EMPTY_PROOF_FEEDBACK }))
      .toBe(DEFAULT_EMPTY_PROOF_FEEDBACK);
    expect(formatProofLeanResponse({ error_summary: 'abcdef' }, { formatError: (text) => text.slice(0, 3) }))
      .toContain(': abc The model');
  });
});

describe('Lean workspace repair stop', () => {
  const event = {
    workflow_mode: 'compiler',
    message: 'Compiler stopped because Lean 4 cannot check proofs.',
    terminal_guidance: 'Move the MOTO folder to C:\\MOTO, then press Start again.',
  };

  it('builds a persistent repair notification', () => {
    const notification = buildLeanWorkspaceRepairNotification(event);
    expect(notification.title).toBe(LEAN_WORKSPACE_REPAIR_TITLE);
    expect(notification.notification_kind).toBe('model_error');
    expect(notification.notification_key).toBe('lean-workspace:compiler');
    expect(notification.terminal_guidance).toBe(event.terminal_guidance);
  });

  it('formats one Live Activity line with the problem and the fix', () => {
    expect(formatLeanWorkspaceRepairActivity(event)).toBe(
      `${LEAN_WORKSPACE_REPAIR_TITLE}: ${event.message} ${event.terminal_guidance}`,
    );
  });
});
