import { describe, expect, test } from 'vitest';
import {
  isSensitivePersistedActivityField,
  redactPersistedActivityText,
  sanitizePersistedActivityValue,
} from './activityPersistence';
import {
  compactLiveActivityEvent,
  readPersistedLiveActivity,
} from './liveActivityPersistence';
import { formatProviderRetryCooldownMessage } from './activityStyles';

describe('activity persistence sanitization', () => {
  test('redacts nested credential fields without mutating useful metadata', () => {
    const input = {
      provider: 'openrouter',
      role_id: 'validator',
      session_id: 'session-123',
      nested: [{ access_token: 'access-secret', model: 'model-a' }],
      apiKey: 'key-secret',
    };

    const output = sanitizePersistedActivityValue(input);

    expect(output).toEqual({
      provider: 'openrouter',
      role_id: 'validator',
      session_id: 'session-123',
      nested: [{ access_token: '[redacted]', model: 'model-a' }],
      apiKey: '[redacted]',
    });
    expect(input.nested[0].access_token).toBe('access-secret');
  });

  test('redacts credentials embedded in messages and callback URLs', () => {
    const text = [
      'Authorization: Bearer abc.def.ghi',
      'api_key=plain-secret',
      'sk-or-v1-supersecret',
      'https://callback.test/?code=oauth-code&state=keep&refresh_token=refresh-secret',
    ].join(' ');
    const redacted = redactPersistedActivityText(text);

    expect(redacted).not.toContain('abc.def.ghi');
    expect(redacted).not.toContain('plain-secret');
    expect(redacted).not.toContain('supersecret');
    expect(redacted).not.toContain('oauth-code');
    expect(redacted).not.toContain('refresh-secret');
    expect(redacted).toContain('state=keep');
  });

  test('does not classify provenance session IDs as session tokens', () => {
    expect(isSensitivePersistedActivityField('session_id')).toBe(false);
    expect(isSensitivePersistedActivityField('session_token')).toBe(true);
  });

  test('handles circular values without throwing', () => {
    const input = { provider: 'openrouter' };
    input.self = input;
    expect(sanitizePersistedActivityValue(input)).toEqual({
      provider: 'openrouter',
      self: '[omitted circular value]',
    });
  });

  test('compact live events preserve assurance fields and safe bounded detail', () => {
    const compacted = compactLiveActivityEvent({
      event: 'compiler_rejection',
      timestamp: '2026-10-05T12:00:00Z',
      data: {
        submission_id: 'sub-1',
        reasoning: `api_key=secret ${'x'.repeat(2500)}`,
      },
    });

    expect(compacted.activityIdentity).toContain('sub-1');
    expect(compacted.category).toBe('validation');
    expect(compacted.severity).toBe('error');
    expect(compacted.outcome).toBe('rejected');
    expect(compacted.validatorFeedback).not.toContain('secret');
    expect(compacted.validatorFeedback.length).toBeLessThanOrEqual(2000);
  });

  test('persists two correlated provider retry attempts across reload distinctly and secret-safe', () => {
    const storageKey = 'test:provider-retry-live-activity';
    const storedValues = new Map();
    const localStorageStub = {
      getItem: (key) => storedValues.get(key) ?? null,
      setItem: (key, value) => storedValues.set(key, String(value)),
      removeItem: (key) => storedValues.delete(key),
    };
    const previousLocalStorage = globalThis.localStorage;
    globalThis.localStorage = localStorageStub;
    const attempts = [1, 2].map((attempt) => compactLiveActivityEvent({
      event: 'provider_retry_wait',
      event_id: `retry-event-${attempt}`,
      timestamp: `2026-10-05T12:00:0${attempt}Z`,
      message: `Retry ${attempt}: Authorization: Bearer bearer-secret-${attempt}`,
      data: {
        activity_id: `retry-event-${attempt}`,
        correlation_id: 'provider-call-123',
        retry_attempt: attempt,
        provider: 'openrouter',
        role_id: 'validator',
        error_summary: `api_key=api-secret-${attempt} transient upstream failure`,
      },
    }));

    let persistedBytes;
    let restored;
    try {
      localStorage.setItem(storageKey, JSON.stringify(attempts));
      persistedBytes = localStorage.getItem(storageKey);
      restored = readPersistedLiveActivity(storageKey);
    } finally {
      if (previousLocalStorage === undefined) {
        delete globalThis.localStorage;
      } else {
        globalThis.localStorage = previousLocalStorage;
      }
    }

    expect(restored).toHaveLength(2);
    expect(restored.map((event) => event.activityIdentity)).toEqual([
      'provider_retry_wait:retry-event-1',
      'provider_retry_wait:retry-event-2',
    ]);
    expect(restored.map((event) => event.data.correlation_id)).toEqual([
      'provider-call-123',
      'provider-call-123',
    ]);
    expect(restored.map((event) => event.data.retry_attempt)).toEqual([1, 2]);
    expect(persistedBytes).not.toContain('bearer-secret');
    expect(persistedBytes).not.toContain('api-secret');
    expect(restored.every((event) => event.message.includes('[redacted]'))).toBe(true);
    expect(restored.every((event) => event.errorDetail.includes('[redacted]'))).toBe(true);
  });

  test('persists provider retry route identity across reload', () => {
    const storageKey = 'test:provider-route-identity-live-activity';
    const routeIdentity = '[Autonomous · Submitter 2 · vendor/test-free:free via OpenRouter · host auto → TestHost]';
    const storedValues = new Map();
    const localStorageStub = {
      getItem: (key) => storedValues.get(key) ?? null,
      setItem: (key, value) => storedValues.set(key, String(value)),
      removeItem: (key) => storedValues.delete(key),
    };
    const previousLocalStorage = globalThis.localStorage;
    globalThis.localStorage = localStorageStub;
    const notification = {
      event_type: 'provider_retry_cooldown',
      event_id: 'provider-retry-route-1',
      workflow_mode: 'autonomous',
      role_id: 'aggregator_submitter_2',
      role_label: 'Submitter 2',
      model: 'vendor/test-free:free',
      provider: 'openrouter',
      provider_label: 'OpenRouter',
      host_routing: 'auto',
      configured_host_provider: '',
      effective_host_provider: 'TestHost',
      provider_request_id: 'gen-test-1',
      route_identity: routeIdentity,
      retry_attempt: 1,
      retry_after_seconds: 2,
      message: `${routeIdentity} OpenRouter server-side error: OpenRouter returned a successful `
        + 'response without completion choices. OpenRouter API call will retry after a 2-second cooldown (retry 1).',
    };
    const event = compactLiveActivityEvent({
      event: 'provider_retry_cooldown',
      type: 'provider_retry_cooldown',
      event_id: notification.event_id,
      timestamp: '2026-10-05T12:00:01Z',
      message: formatProviderRetryCooldownMessage(notification),
      data: notification,
    });

    let restored;
    try {
      localStorage.setItem(storageKey, JSON.stringify([event]));
      restored = readPersistedLiveActivity(storageKey);
    } finally {
      if (previousLocalStorage === undefined) {
        delete globalThis.localStorage;
      } else {
        globalThis.localStorage = previousLocalStorage;
      }
    }

    expect(restored).toHaveLength(1);
    expect(restored[0].event).toBe('provider_retry_cooldown');
    expect(restored[0].message).toBe(notification.message);
    expect(restored[0].message.startsWith(routeIdentity)).toBe(true);
    expect(restored[0].message.split(routeIdentity)).toHaveLength(2);
    expect(restored[0].data).toMatchObject({
      workflow_mode: 'autonomous',
      role_id: 'aggregator_submitter_2',
      model: 'vendor/test-free:free',
      provider: 'openrouter',
      host_routing: 'auto',
      effective_host_provider: 'TestHost',
      provider_request_id: 'gen-test-1',
      route_identity: routeIdentity,
    });
    expect(restored[0].activityIdentity).toBe('provider_retry_cooldown:provider-retry-route-1');
  });
});
