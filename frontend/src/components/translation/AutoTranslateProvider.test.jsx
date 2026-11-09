import React from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import AutoTranslateProvider, { STATUS_REFRESH_DELAY_MS, mergeRecentActivity } from './AutoTranslateProvider';
import { DISPLAY_LANGUAGE_STORAGE_KEY, useAutoTranslate } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';
import { websocket } from '../../services/websocket';

const { listeners } = vi.hoisted(() => ({ listeners: new Map() }));

vi.mock('../../services/websocket', () => ({
  websocket: {
    on: vi.fn((name, callback) => {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(callback);
      return () => listeners.get(name)?.delete(callback);
    }),
    off: vi.fn(),
  },
}));

vi.mock('../../services/api', () => ({
  translationAPI: {
    getStatus: vi.fn(),
    registerDocuments: vi.fn(),
    lookupDocuments: vi.fn(),
  },
}));

function enabledStatus(overrides = {}) {
  return {
    success: true,
    enabled: true,
    paused: false,
    state: 'idle',
    status_label: 'ready',
    languages: [
      {
        key: 'es',
        name: 'Spanish',
        custom: false,
        translated_units: 0,
        pending_units: 0,
        pending_documents: 0,
        complete_documents: 0,
      },
      {
        key: 'x-klingon',
        name: 'Klingon',
        custom: true,
        translated_units: 0,
        pending_units: 0,
        pending_documents: 0,
        complete_documents: 0,
      },
    ],
    recent_activity: [],
    ...overrides,
  };
}

let latest;

function Probe() {
  latest = useAutoTranslate();
  return (
    <div>
      <span data-testid="enabled">{String(latest.enabled)}</span>
      <span data-testid="display">{latest.displayLanguage}</span>
      <span data-testid="languages">{latest.languages.map((language) => language.key).join(',')}</span>
    </div>
  );
}

function emitSocket(name, data = {}) {
  act(() => {
    [...(listeners.get(name) || [])].forEach((callback) => callback(data));
  });
}

async function flushPromises() {
  await act(async () => {
    for (let index = 0; index < 10; index += 1) {
      await Promise.resolve();
    }
  });
}

async function advance(ms) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function renderProvider() {
  const view = render(<AutoTranslateProvider><Probe /></AutoTranslateProvider>);
  await flushPromises();
  return view;
}

beforeEach(() => {
  vi.useFakeTimers();
  listeners.clear();
  localStorage.clear();
  latest = null;
  translationAPI.getStatus.mockReset();
  translationAPI.registerDocuments.mockReset();
  websocket.on.mockClear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
});

describe('AutoTranslateProvider', () => {
  test('the default context is inert without a provider', async () => {
    render(<Probe />);
    await advance(60000);

    expect(latest.available).toBe(false);
    expect(latest.enabled).toBe(false);
    expect(latest.languages).toEqual([]);
    expect(latest.displayLanguage).toBe('original');
    expect(latest.recentActivity).toEqual([]);
    expect(typeof latest.subscribeProgress(() => {})).toBe('function');
    await expect(latest.refreshStatus()).resolves.toBeUndefined();
    expect(translationAPI.getStatus).not.toHaveBeenCalled();
    expect(websocket.on).not.toHaveBeenCalled();
  });

  test('hydrates status on mount and again on websocket connected and status changes', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();

    expect(translationAPI.getStatus).toHaveBeenCalledTimes(1);
    expect(latest.available).toBe(true);
    expect(screen.getByTestId('enabled')).toHaveTextContent('true');
    expect(screen.getByTestId('languages')).toHaveTextContent('es,x-klingon');
    expect(latest.languages[1]).toMatchObject({ key: 'x-klingon', name: 'Klingon', custom: true });

    emitSocket('connected');
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(2);

    emitSocket('translation_status_changed', { state: 'translating' });
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(3);
  });

  test('exposes the status Translate all block, or null from a backend that sends none', async () => {
    render(<Probe />);
    expect(latest.translateAll).toBeNull();
    cleanup();

    const block = { requested: true, state: 'translating', documents: 3, complete_documents: 1 };
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ translate_all: block }));
    const view = await renderProvider();
    expect(latest.translateAll).toEqual(block);
    view.unmount();

    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    expect(latest.translateAll).toBeNull();
  });

  test('keeps the inert values when the status request fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.getStatus.mockRejectedValue(new Error('backend unavailable'));
    await renderProvider();

    expect(latest.available).toBe(true);
    expect(latest.enabled).toBe(false);
    expect(latest.status).toBeNull();
    expect(latest.displayLanguage).toBe('original');
  });

  test('an older status response never replaces a newer one', async () => {
    let resolveFirst;
    translationAPI.getStatus
      .mockReturnValueOnce(new Promise((resolve) => { resolveFirst = resolve; }))
      .mockResolvedValueOnce(enabledStatus({ state: 'translating' }));
    await renderProvider();
    emitSocket('connected');
    await flushPromises();
    expect(latest.status.state).toBe('translating');

    await act(async () => { resolveFirst(enabledStatus({ state: 'idle' })); });
    await flushPromises();
    expect(latest.status.state).toBe('translating');
  });

  test('persists the default display language, falls back when unselected, and keeps it while disabled', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    expect(screen.getByTestId('display')).toHaveTextContent('original');

    act(() => latest.setDisplayLanguage('es'));
    expect(screen.getByTestId('display')).toHaveTextContent('es');
    expect(localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY)).toBe('es');

    act(() => latest.setDisplayLanguage('de'));
    expect(localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY)).toBe('de');
    expect(screen.getByTestId('display')).toHaveTextContent('original');

    act(() => latest.setDisplayLanguage('es'));
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ enabled: false, state: 'disabled' }));
    emitSocket('translation_status_changed');
    await flushPromises();
    // The default can be chosen before translation is turned on; viewers show the original meanwhile.
    expect(screen.getByTestId('display')).toHaveTextContent('es');
    expect(localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY)).toBe('es');
  });

  test('keeps each item\'s own language choice apart from the default', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    act(() => latest.setDisplayLanguage('es'));
    expect(latest.documentLanguages).toEqual({});

    act(() => latest.setDocumentLanguages(['proof:a'], 'original'));
    act(() => latest.setDocumentLanguages(['paper:b', 'paper:c'], 'es'));
    expect(latest.documentLanguages).toEqual({ 'proof:a': 'original', 'paper:b': 'es', 'paper:c': 'es' });
    expect(screen.getByTestId('display')).toHaveTextContent('es');
    expect(localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY)).toBe('es');

    const unchanged = latest.documentLanguages;
    act(() => latest.setDocumentLanguages(['proof:a'], 'original'));
    act(() => latest.setDocumentLanguages([], 'es'));
    expect(latest.documentLanguages).toBe(unchanged);
  });

  test('restores a stored language and survives unavailable storage', async () => {
    localStorage.setItem(DISPLAY_LANGUAGE_STORAGE_KEY, 'x-klingon');
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    expect(screen.getByTestId('display')).toHaveTextContent('x-klingon');

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    act(() => latest.setDisplayLanguage('es'));
    expect(screen.getByTestId('display')).toHaveTextContent('es');
  });

  test('refreshes status two seconds after the first event of a burst, then again for later events', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(1);

    emitSocket('translation_attempt_accepted', { activity_id: 'a1', language_key: 'es' });
    await advance(1500);
    emitSocket('translation_attempt_rejected', { activity_id: 'a2', language_key: 'es' });
    await advance(499);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(1);
    await advance(1);
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(2);

    // A steady stream of events (Batch translation) never postpones the refresh: one
    // refresh every two seconds, and a trailing one after the last event.
    for (let index = 0; index < 20; index += 1) {
      emitSocket(index % 2 ? 'translation_queue_changed' : 'translation_attempt_kept', {
        activity_id: `burst-${index}`,
        queue_version: index,
      });
      await advance(500);
    }
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(7);
    await advance(STATUS_REFRESH_DELAY_MS);
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(7);
  });

  test.each([
    'translation_attempt_accepted',
    'translation_attempt_rejected',
    'translation_attempt_kept',
    'translation_part_finalized',
    'translation_attempt_check_failed',
    'translation_queue_changed',
  ])('%s refreshes status after the delay', async (eventType) => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();

    emitSocket(eventType, { activity_id: `${eventType}-1`, language_key: 'es', queue_version: 1 });
    await advance(STATUS_REFRESH_DELAY_MS);
    await flushPromises();
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(2);
  });

  test.each([
    'translation_validator_contract_retry',
    'translation_repair_required',
    'translation_progress',
  ])('%s does not schedule a status refresh', async (eventType) => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();

    emitSocket(eventType, { activity_id: `${eventType}-1`, language_key: 'es', doc_hashes: ['h1'] });
    await advance(STATUS_REFRESH_DELAY_MS * 2);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(1);
  });

  test('polls every 30 seconds while enabled and visible', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();

    await advance(30000);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(2);

    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    await advance(30000);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(2);

    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ enabled: false, state: 'disabled' }));
    await advance(30000);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(3);
    await advance(90000);
    expect(translationAPI.getStatus).toHaveBeenCalledTimes(3);
  });

  test('forwards progress and queue-change events to subscribers until they unsubscribe', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    const callback = vi.fn();
    const unsubscribe = latest.subscribeProgress(callback);
    const progress = { language_key: 'es', doc_hashes: ['h1'], translated_units: 3, total_units: 9 };
    const queue = { queue_version: 7, activity_id: 'translation_queue_changed:7' };

    emitSocket('translation_progress', progress);
    emitSocket('translation_queue_changed', queue);
    expect(callback.mock.calls).toEqual([
      [progress, 'translation_progress'],
      [queue, 'translation_queue_changed'],
    ]);
    // Queue changes are not user activity.
    expect(latest.recentActivity).toEqual([]);

    unsubscribe();
    emitSocket('translation_progress', progress);
    emitSocket('translation_queue_changed', queue);
    expect(callback).toHaveBeenCalledTimes(2);
  });

  test('a failing subscriber does not stop the others', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    const broken = vi.fn(() => { throw new Error('boom'); });
    const working = vi.fn();
    latest.subscribeProgress(broken);
    latest.subscribeProgress(working);

    emitSocket('translation_queue_changed', { queue_version: 1 });
    expect(broken).toHaveBeenCalledTimes(1);
    expect(working).toHaveBeenCalledWith({ queue_version: 1 }, 'translation_queue_changed');
  });

  test('merges live translation activity with status activity, newest first and de-duplicated', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus({
      recent_activity: [
        { activity_id: 'old', event_type: 'translation_attempt_accepted', timestamp: '2026-10-01T10:00:00Z' },
        { activity_id: 'shared', event_type: 'translation_attempt_rejected', timestamp: '2026-10-01T10:05:00Z' },
      ],
    }));
    await renderProvider();
    expect(latest.recentActivity.map((item) => item.activity_id)).toEqual(['shared', 'old']);

    emitSocket('translation_attempt_rejected', { activity_id: 'shared', round_index: 2 });
    emitSocket('translation_attempt_kept', { activity_id: 'kept', round_index: 3 });
    emitSocket('translation_part_finalized', { activity_id: 'final', reason: 'kept_twice' });
    emitSocket('translation_repair_required', { activity_id: 'repair', reason: 'context_window' });
    emitSocket('provider_retry_cooldown', { activity_id: 'other-workflow', workflow_mode: 'autonomous' });
    emitSocket('provider_retry_cooldown', {
      activity_id: 'retry',
      workflow_mode: 'translation',
      retry_after_seconds: 10,
    });
    emitSocket('hung_connection_alert', { activity_id: 'hung', workflow_mode: 'translation' });

    const ids = latest.recentActivity.map((item) => item.activity_id);
    expect(ids).toEqual(['hung', 'retry', 'repair', 'final', 'kept', 'shared', 'old']);
    expect(latest.recentActivity.find((item) => item.activity_id === 'final')).toMatchObject({
      event_type: 'translation_part_finalized',
      reason: 'kept_twice',
    });
    expect(latest.recentActivity[1]).toMatchObject({ event_type: 'provider_retry_cooldown' });
    expect(ids.filter((id) => id === 'shared')).toHaveLength(1);
  });

  test('Translate all starting, cancelled, and finishing shows live and from status recent_activity', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus({
      recent_activity: [
        {
          activity_id: 'finished',
          event_type: 'translation_translate_all',
          state: 'finished',
          message: 'Translate all finished: 3 items in 1 language.',
          timestamp: '2026-10-01T10:05:00Z',
        },
      ],
    }));
    await renderProvider();
    expect(latest.recentActivity).toEqual([expect.objectContaining({ activity_id: 'finished', state: 'finished' })]);

    emitSocket('translation_translate_all', { activity_id: 'started', state: 'started', message: 'Translate all started.' });
    emitSocket('translation_translate_all', { activity_id: 'cancelled', state: 'cancelled', message: 'Translate all cancelled.' });
    expect(latest.recentActivity.map((item) => item.activity_id)).toEqual(['cancelled', 'started', 'finished']);
    expect(latest.recentActivity[0]).toMatchObject({
      event_type: 'translation_translate_all',
      state: 'cancelled',
      message: 'Translate all cancelled.',
    });
  });

  test('lists adapter retry waits live but leaves service-started waits to the status entry', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ recent_activity: [] }));
    await renderProvider();
    const callsBefore = translationAPI.getStatus.mock.calls.length;

    emitSocket('provider_retry_cooldown', {
      activity_id: 'service-wait',
      workflow_mode: 'translation',
      retry_layer: 'workflow',
      retry_after_seconds: 10,
    });
    emitSocket('provider_retry_cooldown', {
      activity_id: 'adapter-wait',
      workflow_mode: 'translation',
      retry_layer: 'adapter',
      retry_after_seconds: 5,
    });

    expect(latest.recentActivity.map((item) => item.activity_id)).toEqual(['adapter-wait']);
    await advance(STATUS_REFRESH_DELAY_MS);
    expect(translationAPI.getStatus.mock.calls.length).toBeGreaterThan(callsBefore);
  });

  test('a usage-limit cooldown wait shows once, from its status entry with the route', async () => {
    const route = '[Foreign Language Translation w/ Top-p · Translator Submitter · kimi-for-coding via Kimi Code]';
    // The service records the wait under the cooldown error's reason, with the route.
    const statusEntry = {
      activity_id: 'status-usage-wait',
      event_type: 'translation_provider_retry',
      role_id: 'translation_submitter',
      reason: 'usage_limit_reached',
      retry_after_seconds: 3600,
      route_identity: route,
      message: `${route} Usage limit reached; retrying in 3600s.`,
      timestamp: '2026-11-02T10:00:00Z',
    };
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ recent_activity: [] }));
    await renderProvider();
    translationAPI.getStatus.mockResolvedValue(enabledStatus({ recent_activity: [statusEntry] }));

    // The shared cooldown seam emits one adapter-layer event per waiting chunk.
    [1, 2].forEach((attempt) => emitSocket('provider_retry_cooldown', {
      activity_id: `usage-wait-${attempt}`,
      workflow_mode: 'translation',
      retry_layer: 'adapter',
      reason: 'usage_limit_cooldown',
      retry_attempt: attempt,
      retry_after_seconds: 300,
      route_identity: route,
      message: `${route} Kimi Code usage limit reached; waiting 300 seconds before retry.`,
    }));
    emitSocket('provider_retry_cooldown', {
      activity_id: 'other-usage-wait',
      workflow_mode: 'autonomous',
      reason: 'usage_limit_cooldown',
    });
    expect(latest.recentActivity).toEqual([]);

    await advance(STATUS_REFRESH_DELAY_MS);
    await flushPromises();
    expect(latest.recentActivity.map((item) => item.activity_id)).toEqual(['status-usage-wait']);
    expect(latest.recentActivity[0].route_identity).toBe(route);
  });

  test('provider_retry_resumed is not translation activity', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    await renderProvider();
    emitSocket('provider_retry_resumed', { activity_id: 'resumed', workflow_mode: 'translation' });
    expect(latest.recentActivity).toEqual([]);
    expect(listeners.get('provider_retry_resumed')?.size || 0).toBe(0);
  });

  test('de-duplicates activity stored without an activity_id by event, part, attempt, language, and time', () => {
    const stored = (overrides) => ({
      event_type: 'translation_attempt_rejected',
      target_hash: 'doc-1:es:7',
      round_index: 1,
      language_key: 'es',
      timestamp: '2026-10-01T10:00:00Z',
      ...overrides,
    });
    const merged = mergeRecentActivity(
      [{ activity_id: 'live-1', event_type: 'translation_attempt_accepted', received_at: '2026-10-01T10:10:00Z' }],
      [
        stored(),
        stored(),
        stored({ round_index: 2 }),
        stored({ language_key: 'fr' }),
        stored({ timestamp: '2026-10-01T10:01:00Z' }),
        stored({ event_type: 'translation_attempt_check_failed' }),
        stored({ target_hash: 'doc-1:es:9' }),
        stored({ target_hash: 'doc-1:es:9' }),
        { activity_id: 'live-1', event_type: 'translation_attempt_accepted', timestamp: '2026-10-01T10:10:00Z' },
      ],
    );

    expect(merged).toHaveLength(7);
    expect(merged.filter((item) => item.activity_id === 'live-1')).toHaveLength(1);
    expect(merged.filter((item) => item.target_hash === 'doc-1:es:9')).toHaveLength(1);
    expect(merged.filter((item) => (
      item.event_type === 'translation_attempt_rejected' && item.round_index === 1 && item.language_key === 'es'
        && item.timestamp === '2026-10-01T10:00:00Z' && item.target_hash === 'doc-1:es:7'
    ))).toHaveLength(1);
  });

  test('bounds recent activity to 100 items', () => {
    const live = Array.from({ length: 80 }, (_, index) => ({
      activity_id: `live-${index}`,
      received_at: new Date(Date.UTC(2026, 9, 2, 0, 0, index)).toISOString(),
    }));
    const stored = Array.from({ length: 80 }, (_, index) => ({
      activity_id: `stored-${index}`,
      timestamp: new Date(Date.UTC(2026, 9, 1, 0, 0, index)).toISOString(),
    }));
    const merged = mergeRecentActivity(live, stored);
    expect(merged).toHaveLength(100);
    expect(merged[0].activity_id).toBe('live-79');
    expect(merged[99].activity_id).toBe('stored-60');
  });

  test('unmounting removes every websocket listener', async () => {
    translationAPI.getStatus.mockResolvedValue(enabledStatus());
    const { unmount } = await renderProvider();
    expect([...listeners.values()].some((set) => set.size > 0)).toBe(true);
    unmount();
    expect([...listeners.values()].every((set) => set.size === 0)).toBe(true);
  });
});
