import { describe, expect, test } from 'vitest';
import {
  isSensitivePersistedActivityField,
  redactPersistedActivityText,
  sanitizePersistedActivityValue,
} from './activityPersistence';

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
});
