import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { getScopedProviderActivityFeed } from '../App';
import CompilerLogs, { COMPILER_ACTIVITY_STORAGE_KEY } from './compiler/CompilerLogs';
import AutonomousResearchLogs from './autonomous/AutonomousResearchLogs';
import { normalizeActivityEvent } from '../utils/activityNormalization';
import {
  compactLiveActivityEvent,
  readPersistedLiveActivity,
  selectModelOutputNormalizedActivity,
} from '../utils/liveActivityPersistence';
import { mergeCompilerActivityEvents } from '../utils/manualLogRouting';
import {
  MODEL_OUTPUT_NORMALIZED_EVENT,
  formatModelOutputNormalizedMessage,
} from '../utils/activityStyles';

vi.mock('../services/api', async (importOriginal) => {
  const original = await importOriginal();
  return {
    ...original,
    compilerAPI: {
      ...original.compilerAPI,
      getMetrics: vi.fn().mockResolvedValue({
        data: {
          construction: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
          rigor: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
          outline: { acceptances: 0, rejections: 0, declines: 0 },
          review: { acceptances: 0, rejections: 0, declines: 0 },
          minuscule_edit_count: 0,
          paper_word_count: 0,
          total_submissions: 0,
        },
      }),
      getStatus: vi.fn().mockResolvedValue({ data: { is_running: true, current_mode: 'construction' } }),
    },
    autonomousAPI: {
      ...original.autonomousAPI,
      getProofCompetitionBenchmarks: vi.fn().mockResolvedValue({ reports: [] }),
    },
  };
});

vi.mock('../services/websocket', () => ({
  websocket: {
    on: vi.fn(() => () => {}),
    off: vi.fn(),
  },
}));

vi.mock('./ApiCallLogs', () => ({ default: () => <div>API logs</div> }));
vi.mock('./autonomous/ProofCompetitionBenchmarkReport', () => ({ default: () => <div>Benchmarks</div> }));

const ROUTE_IDENTITY = '[Single Paper Writer · Validator · gemma-3 via LM Studio]';

const normalizedPayload = (overrides = {}) => ({
  event_id: 'model_output_normalized:0001',
  activity_id: 'model_output_normalized:0001',
  role_id: 'compiler_validator',
  task_id: 'comp_val_004',
  workflow_mode: 'compiler',
  route_identity: ROUTE_IDENTITY,
  repairs: [{ kind: 'stray_characters', item_path: '', from_key: '', to_key: '' }],
  repair_count: 1,
  message: `${ROUTE_IDENTITY} Accepted gemma-3's reply after removing a stray character before a field name.`,
  activity_outcome: 'normalized',
  activity_severity: 'info',
  event_type: MODEL_OUTPUT_NORMALIZED_EVENT,
  ...overrides,
});

// Mirrors App.addScopedProviderActivity so the component sees what App hands down.
const feedRow = (payload) => normalizeActivityEvent({
  event: MODEL_OUTPUT_NORMALIZED_EVENT,
  type: MODEL_OUTPUT_NORMALIZED_EVENT,
  timestamp: '2026-11-15T12:00:00Z',
  message: formatModelOutputNormalizedMessage(payload),
  data: payload,
});

describe('model_output_normalized Live Activity routing', () => {
  test('routes by workflow mode exactly like provider retry rows', () => {
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: 'compiler' }))).toBe('compiler');
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: 'aggregator' }))).toBe('aggregator');
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: 'autonomous' }))).toBe('autonomous');
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: 'leanoj' }))).toBe('leanoj');
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: '' }))).toBe('');
    expect(getScopedProviderActivityFeed(normalizedPayload({ workflow_mode: 'translation' }))).toBe('');
  });

  test('manual proof-check normalizations follow the checked source', () => {
    const manual = (sourceId) => normalizedPayload({ workflow_mode: 'manual_proof_check', source_id: sourceId });
    expect(getScopedProviderActivityFeed(manual('manual_aggregator'))).toBe('aggregator');
    expect(getScopedProviderActivityFeed(manual('manual_compiler_current'))).toBe('compiler');
    expect(getScopedProviderActivityFeed(manual('topic_7'))).toBe('autonomous');
    expect(getScopedProviderActivityFeed(manual(''))).toBe('');
  });

  test('other events without a notification key are still dropped', () => {
    expect(getScopedProviderActivityFeed({ event_type: 'something_else', workflow_mode: 'compiler' })).toBe('');
  });
});

describe('model_output_normalized persistence and dedupe', () => {
  beforeEach(() => localStorage.clear());

  test('only normalization rows are selected for Aggregator/Compiler browser storage', () => {
    const retryRow = normalizeActivityEvent({
      event: 'provider_retry_cooldown',
      message: 'retry',
      data: { notification_key: 'provider-retry:x' },
    });
    const rows = selectModelOutputNormalizedActivity([retryRow, feedRow(normalizedPayload())]);
    expect(rows).toHaveLength(1);
    expect(rows[0].event).toBe(MODEL_OUTPUT_NORMALIZED_EVENT);
    expect(selectModelOutputNormalizedActivity(null)).toEqual([]);
  });

  test('a stored row survives reload with its identity and message', () => {
    const row = feedRow(normalizedPayload());
    localStorage.setItem('compiler_model_output_activity_events', JSON.stringify([compactLiveActivityEvent(row)]));
    const [restored] = readPersistedLiveActivity('compiler_model_output_activity_events');
    expect(restored.event).toBe(MODEL_OUTPUT_NORMALIZED_EVENT);
    expect(restored.activityIdentity).toBe(row.activityIdentity);
    expect(restored.activityIdentity).toBe('model_output_normalized:model_output_normalized:0001');
    expect(restored.message).toBe(row.message);
  });

  test('compiler merge keeps one copy per identity and orders newest first', () => {
    const first = feedRow(normalizedPayload());
    const second = feedRow(normalizedPayload({
      event_id: 'model_output_normalized:0002',
      activity_id: 'model_output_normalized:0002',
    }));
    const later = { ...second, timestamp: '2026-11-15T12:05:00Z' };
    const merged = mergeCompilerActivityEvents([first], [first, later]);
    expect(merged.map((event) => event.activityIdentity)).toEqual([
      later.activityIdentity,
      first.activityIdentity,
    ]);
  });
});

describe('model_output_normalized rendering', () => {
  const previousFetch = globalThis.fetch;

  beforeEach(() => {
    localStorage.clear();
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false });
  });

  afterEach(() => {
    cleanup();
    globalThis.fetch = previousFetch;
  });

  test('CompilerLogs shows the row once, as info, and keeps it beside saved events', async () => {
    localStorage.setItem(COMPILER_ACTIVITY_STORAGE_KEY, JSON.stringify([{
      type: 'compiler_submission',
      data: { mode: 'construction' },
      timestamp: '12:00:00 PM',
      fullTimestamp: '2026-11-15T11:59:00Z',
    }]));
    const row = feedRow(normalizedPayload());
    const { container, rerender } = render(<CompilerLogs providerActivity={[row]} />);
    const message = await screen.findByText(row.message);
    expect(message.closest('.activity-item')).toHaveClass('activity-info');
    expect(screen.getByText('Submission: construction mode')).toBeInTheDocument();

    rerender(<CompilerLogs providerActivity={[row, { ...row }]} />);
    await waitFor(() => {
      expect(container.textContent.split(row.message)).toHaveLength(2);
    });
  });

  test('AutonomousResearchLogs labels normalizations as info and errored proof checks as errors', () => {
    render(
      <AutonomousResearchLogs
        stats={{}}
        events={[
          feedRow(normalizedPayload({ workflow_mode: 'autonomous' })),
          {
            event: 'proof_check_complete',
            timestamp: '2026-11-15T12:01:00Z',
            data: {
              verified_count: 0,
              novel_count: 0,
              had_error: true,
              message: "Proof verification encountered an error: 1 of 2 theorem entries was invalid; theorems[0] 'lemma_a': missing required field 'novelty_rationale'. No Lean work was attempted.",
            },
          },
        ]}
      />,
    );
    const normalized = screen.getByText(feedRow(normalizedPayload()).message);
    expect(normalized.closest('.auto-log-entry')).toHaveClass('log-info');
    const stopped = screen.getByText(/Proof check stopped on an error: 1 of 2 theorem entries was invalid/);
    expect(stopped.closest('.auto-log-entry')).toHaveClass('log-reject');
  });
});
