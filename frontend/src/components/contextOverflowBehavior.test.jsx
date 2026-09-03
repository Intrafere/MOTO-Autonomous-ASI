import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import {
  compactLiveActivityEvent,
  isProviderNotificationDismissed,
  persistDismissedProviderNotificationId,
  readPersistedLiveActivity,
  shouldRecordWorkflowStoppedActivity,
} from '../utils/liveActivityPersistence';
import {
  compactCompilerActivityEvents,
  formatAggregatorPersistedOverflowMessage,
  shouldIncludeAggregatorProofContextOverflow,
  shouldIncludeAggregatorSolutionPathEvent,
  shouldIncludeCompilerContextOverflow,
  shouldIncludeCompilerProofContextOverflow,
  shouldIncludeCompilerSolutionPathEvent,
} from '../utils/manualLogRouting';
import { formatContextOverflowActivityMessage } from '../utils/activityStyles';
import { buildTerminalModelErrorNotification } from '../App';

describe('context overflow activity behavior', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  test('solution path activity stays in its owning manual workflow log', () => {
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'aggregator' })).toBe(true);
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'compiler' })).toBe(false);
    expect(shouldIncludeAggregatorSolutionPathEvent({ workflow_mode: 'autonomous' })).toBe(false);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'compiler' })).toBe(true);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'aggregator' })).toBe(false);
    expect(shouldIncludeCompilerSolutionPathEvent({ workflow_mode: 'leanoj' })).toBe(false);
  });

  test('Compiler accepts its own events and rejects autonomous, LeanOJ, and Aggregator events', () => {
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'compiler',
      role_id: 'compiler_writer',
    })).toBe(true);
    expect(shouldIncludeCompilerContextOverflow({
      role_id: 'compiler_validator',
    })).toBe(true);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'autonomous',
      role_id: 'compiler_writer',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'leanoj',
      role_id: 'leanoj_final_solver',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({
      workflow_mode: 'aggregator',
      role_id: 'aggregator_validator',
    })).toBe(false);
    expect(shouldIncludeCompilerContextOverflow({})).toBe(false);
  });

  test('App suppresses duplicate overflow terminal entries but retains ordinary stops', () => {
    expect(shouldRecordWorkflowStoppedActivity(
      'auto_research_stopped',
      { reason: 'context_overflow' },
    )).toBe(false);
    expect(shouldRecordWorkflowStoppedActivity(
      'leanoj_stopped',
      { reason: 'context_overflow' },
    )).toBe(false);
    expect(shouldRecordWorkflowStoppedActivity(
      'auto_research_stopped',
      { reason: 'user_stop' },
    )).toBe(true);
  });

  test('App reformats persisted direct and terminal overflow records with model identity', () => {
    localStorage.setItem('activity', JSON.stringify([
      {
        event: 'context_overflow_error',
        message: 'stale message',
        data: {
          message: 'Research stopped.',
          configured_model: 'configured-model',
          configured_provider: 'openrouter',
        },
      },
      {
        event: 'auto_research_stopped',
        message: 'stale terminal message',
        data: {
          reason: 'context_overflow',
          message: 'Research stopped.',
          effective_model: 'fallback-model',
          effective_provider: 'lm_studio',
        },
      },
    ]));

    const restored = readPersistedLiveActivity('activity');
    expect(restored[0].message).toContain('Configured route: configured-model via openrouter');
    expect(restored[1].message).toContain('Route: fallback-model via lm_studio');
  });

  test('App sanitizes credentials in legacy persisted activity records', () => {
    localStorage.setItem('activity', JSON.stringify([{
      event: 'oauth_provider_failure',
      message: 'Authorization: Bearer legacy-token',
      data: {
        provider: 'openai_codex_oauth',
        access_token: 'legacy-access-token',
        error_summary: 'callback https://example.test/?code=legacy-code&state=keep',
      },
    }]));

    const [restored] = readPersistedLiveActivity('activity');
    expect(restored.message).not.toContain('legacy-token');
    expect(restored.data.access_token).toBe('[redacted]');
    expect(restored.data.error_summary).not.toContain('legacy-code');
    expect(restored.data.error_summary).toContain('state=keep');
    expect(restored.data.provider).toBe('openai_codex_oauth');
  });

  test('App drops obsolete manual next-round and three-zero proof activity', () => {
    localStorage.setItem('activity', JSON.stringify([
      {
        event: 'proof_run_round_complete',
        message: 'Round 1 found no candidates and is waiting for Run Next Round.',
        data: { status: 'idle_between_rounds' },
      },
      {
        event: 'proof_run_terminal',
        message: 'Proof run completed after three consecutive valid rounds found no candidates.',
        data: { terminal_reason: 'three_consecutive_zero_candidate_rounds' },
      },
      {
        event: 'proof_run_round_started',
        message: 'Round 4 started.',
        data: { round_index: 4 },
      },
    ]));

    const restored = readPersistedLiveActivity('activity');
    expect(restored).toHaveLength(1);
    expect(restored[0].data.round_index).toBe(4);
  });

  test('App compacts durable activity without discarding sanitized messages', () => {
    const compacted = compactLiveActivityEvent({
      event: 'proof_context_overflow',
      message: 'Proof candidate exceeded its direct-context budget; Authorization: Bearer activity-secret',
      data: {
        workflow_mode: 'autonomous',
        access_token: 'nested-secret',
        error_summary: 'callback https://example.test/?code=oauth-code&state=keep',
      },
    });

    expect(compacted.message).toContain('Proof candidate exceeded its direct-context budget');
    expect(compacted.message).not.toContain('activity-secret');
    expect(compacted.data.access_token).toBe('[redacted]');
    expect(compacted.data.error_summary).not.toContain('oauth-code');
    expect(compacted.data.error_summary).toContain('state=keep');
  });

  test('provider notification dismissals preserve legacy IDs and intermediate fingerprints', async () => {
    localStorage.setItem(
      'dismissedOAuthProviderNotifications',
      JSON.stringify(['legacy-notification']),
    );
    expect(await isProviderNotificationDismissed('legacy-notification')).toBe(true);

    const fingerprint = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode('fingerprinted-notification'),
    );
    const fingerprintHex = Array.from(new Uint8Array(fingerprint), byte => (
      byte.toString(16).padStart(2, '0')
    )).join('');
    localStorage.setItem(
      'dismissedOAuthProviderNotifications',
      JSON.stringify([fingerprintHex]),
    );
    expect(await isProviderNotificationDismissed('fingerprinted-notification')).toBe(true);
  });

  test('provider notification dismissals use independent markers without lost updates', async () => {
    await Promise.all([
      persistDismissedProviderNotificationId('notification-one'),
      persistDismissedProviderNotificationId('notification-two'),
    ]);

    expect(await isProviderNotificationDismissed('notification-one')).toBe(true);
    expect(await isProviderNotificationDismissed('notification-two')).toBe(true);
    expect(
      Object.keys(localStorage).filter(key => (
        key.startsWith('dismissedOAuthProviderNotificationFingerprint:')
      )),
    ).toHaveLength(2);
  });

  test('Aggregator persisted overflow display includes stored model and provider', () => {
    expect(formatAggregatorPersistedOverflowMessage({
      type: 'context_overflow_error',
      message: 'legacy text without identity',
      metadata: {
        message: 'Research stopped.',
        configured_model: 'aggregator-model',
        configured_provider: 'openrouter',
      },
    })).toBe('Research stopped. Configured route: aggregator-model via openrouter.');
  });

  test('manual proof overflow routing assigns each source to exactly one manual feed', () => {
    const aggregator = { source_type: 'brainstorm', source_id: 'manual_aggregator' };
    const compiler = { source_type: 'paper', source_id: 'manual_compiler_current' };
    const autonomous = { source_type: 'paper', source_id: 'paper_123', workflow_mode: 'autonomous' };
    const leanoj = { source_type: 'leanoj_final', source_id: 'leanoj_123' };

    expect([
      shouldIncludeAggregatorProofContextOverflow(aggregator),
      shouldIncludeCompilerProofContextOverflow(aggregator),
    ]).toEqual([true, false]);
    expect([
      shouldIncludeAggregatorProofContextOverflow(compiler),
      shouldIncludeCompilerProofContextOverflow(compiler),
    ]).toEqual([false, true]);
    expect(shouldIncludeAggregatorProofContextOverflow(autonomous)).toBe(false);
    expect(shouldIncludeCompilerProofContextOverflow(autonomous)).toBe(false);
    expect(shouldIncludeAggregatorProofContextOverflow(leanoj)).toBe(false);
    expect(shouldIncludeCompilerProofContextOverflow(leanoj)).toBe(false);
  });

  test('formatter distinguishes changed routes and tolerates partial route metadata', () => {
    expect(formatContextOverflowActivityMessage({
      message: 'Proof context overflow.',
      configured_model: 'primary-model',
      configured_provider: 'openrouter',
      effective_model: 'fallback-model',
      effective_provider: 'lm_studio',
      effective_host_provider: 'local-sibling',
    })).toContain(
      'Effective route: fallback-model via lm_studio, host local-sibling. '
      + 'Configured route: primary-model via openrouter.'
    );
    expect(formatContextOverflowActivityMessage({
      message: 'Proof context overflow.',
      effective_provider: 'openrouter',
      effective_host_provider: 'anthropic',
    })).toContain('Route: openrouter, host anthropic.');
  });

  test('compiler persistence compacts and bounds large event payloads', () => {
    const compacted = compactCompilerActivityEvents(Array.from({ length: 2100 }, (_, index) => ({
      type: 'proof_context_overflow',
      timestamp: `${index}`,
      fullTimestamp: `2026-07-13T00:00:${index}.000Z`,
      data: {
        configured_model: 'configured-model',
        effective_model: 'effective-model',
        effective_host_provider: 'anthropic',
        error_output: 'x'.repeat(5000),
      },
    })));
    expect(compacted).toHaveLength(2000);
    expect(compacted[0].data.configured_model).toBe('configured-model');
    expect(compacted[0].data.effective_host_provider).toBe('anthropic');
    expect(compacted[0].data.error_output.length).toBeLessThan(1300);
  });
});

describe('terminal model error recovery', () => {
  test('adapts provider repair terminal state for the popup', () => {
    expect(buildTerminalModelErrorNotification({
      terminal_event_id: 'terminal-1',
      reason: 'invalid_model',
      notification_kind: 'model_error',
      role_id: 'compiler_writer',
      configured_provider: 'openrouter',
      configured_model: 'configured/model',
      effective_provider: 'lm_studio',
      effective_model: 'fallback/model',
      message: 'Research stopped.',
    })).toMatchObject({
      notification_kind: 'model_error',
      provider: 'lm_studio',
      model: 'fallback/model',
      title: 'Model configuration requires repair',
    });
  });

  test('does not turn ordinary user stops into model errors', () => {
    expect(buildTerminalModelErrorNotification({
      reason: 'user_stop',
      message: 'Research stopped.',
    })).toBeNull();
  });

  test('turns fatal manual continuous proof overflow into a model repair popup', () => {
    expect(buildTerminalModelErrorNotification({
      proof_run_id: 'proof-run-overflow',
      workflow_mode: 'manual_proof_check',
      reason: 'context_overflow',
      fatal: true,
      configured_provider: 'openai_codex_oauth',
      configured_model: 'gpt-5.6-sol',
      message: 'Research stopped.',
    })).toMatchObject({
      notification_kind: 'model_error',
      provider: 'openai_codex_oauth',
      model: 'gpt-5.6-sol',
      title: 'Proof context limit reached',
    });
  });

  test('turns fatal autonomous proof overflow into a model repair popup', () => {
    expect(buildTerminalModelErrorNotification({
      workflow_mode: 'autonomous',
      reason: 'context_overflow',
      fatal: true,
      configured_provider: 'openai_codex_oauth',
      configured_model: 'gpt-5.6-sol',
      message: 'Research stopped.',
    })).toMatchObject({
      notification_kind: 'model_error',
      title: 'Proof context limit reached',
    });
  });

  test('keeps nonfatal candidate proof overflow out of model repair popups', () => {
    expect(buildTerminalModelErrorNotification({
      workflow_mode: 'manual_proof_check',
      reason: 'context_overflow',
      fatal: false,
      message: 'Proof candidate deferred.',
    })).toBeNull();
  });
});
