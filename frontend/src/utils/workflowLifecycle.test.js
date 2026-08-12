import { describe, expect, test } from 'vitest';
import { reconcileAutonomousLifecycle } from './workflowLifecycle';

describe('reconcileAutonomousLifecycle', () => {
  test('authoritatively transitions a known running workflow to stopped', () => {
    const decision = reconcileAutonomousLifecycle({
      wasRunning: true,
      status: {
        is_running: false,
        run_id: 'run-1',
        lifecycle_generation: 3,
        terminal_event: {
          terminal_event_id: 'terminal-1',
          reason: 'context_overflow',
        },
      },
    });

    expect(decision).toMatchObject({
      isRunning: false,
      lifecycleEnded: true,
      shouldRecoverTerminalEvent: true,
      runId: 'run-1',
      lifecycleGeneration: 3,
    });
  });

  test('does not duplicate a terminal event already observed over websocket', () => {
    const decision = reconcileAutonomousLifecycle({
      wasRunning: true,
      seenTerminalEventIds: new Set(['terminal-1']),
      status: {
        is_running: false,
        terminal_event: {
          terminal_event_id: 'terminal-1',
          reason: 'context_overflow',
        },
      },
    });

    expect(decision.shouldRecoverTerminalEvent).toBe(false);
  });

  test('idle status without a terminal record does not invent activity', () => {
    const decision = reconcileAutonomousLifecycle({
      wasRunning: false,
      status: { is_running: false },
    });

    expect(decision.lifecycleEnded).toBe(false);
    expect(decision.shouldRecoverTerminalEvent).toBe(false);
  });

  test('unavailable or malformed status preserves the prior UI state', () => {
    expect(reconcileAutonomousLifecycle({ status: null })).toBeNull();
    expect(reconcileAutonomousLifecycle({ status: {} })).toBeNull();
  });
});
