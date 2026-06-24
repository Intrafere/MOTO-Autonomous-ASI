import { describe, expect, test, beforeEach, vi } from 'vitest';
import { readBooleanStorage } from './safeStorage';

describe('readBooleanStorage', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  test('falls back instead of throwing on malformed localStorage values', () => {
    localStorage.setItem('banner_shimmer_enabled', 'not-json');

    expect(readBooleanStorage('banner_shimmer_enabled', true)).toBe(true);
    expect(readBooleanStorage('banner_shimmer_enabled', false)).toBe(false);
  });

  test('reads stored boolean strings and JSON booleans', () => {
    localStorage.setItem('plain_true', 'true');
    localStorage.setItem('json_false', JSON.stringify(false));

    expect(readBooleanStorage('plain_true', false)).toBe(true);
    expect(readBooleanStorage('json_false', true)).toBe(false);
  });
});
