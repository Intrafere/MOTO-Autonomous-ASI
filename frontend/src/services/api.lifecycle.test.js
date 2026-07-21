import { afterEach, describe, expect, test, vi } from 'vitest';
import {
  API_ERROR_KINDS,
  MotoApiError,
  autonomousAPI,
  requestJson,
} from './api';

describe('lifecycle API errors', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('classifies unreachable reads as backend unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')));

    await expect(requestJson('/api/health')).rejects.toMatchObject({
      kind: API_ERROR_KINDS.BACKEND_UNAVAILABLE,
    });
  });

  test('classifies mutation transport failures as ambiguous', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('connection reset')));

    await expect(autonomousAPI.stop()).rejects.toMatchObject({
      kind: API_ERROR_KINDS.AMBIGUOUS_TRANSPORT,
    });
  });

  test.each([
    [401, API_ERROR_KINDS.STALE_TOKEN],
    [422, API_ERROR_KINDS.BACKEND_VALIDATION],
  ])('classifies HTTP %s responses', async (status, kind) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: 'request rejected' }),
      { status, headers: { 'Content-Type': 'application/json' } },
    )));

    try {
      await autonomousAPI.start({});
      throw new Error('expected request to fail');
    } catch (error) {
      expect(error).toBeInstanceOf(MotoApiError);
      expect(error).toMatchObject({ kind, status });
    }
  });
});
