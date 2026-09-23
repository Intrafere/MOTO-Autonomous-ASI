import { afterEach, describe, expect, test, vi } from 'vitest';
import { API_ERROR_KINDS, autonomousAPI } from './api';

describe('paper batch prune API', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('posts session-aware targets as one confirmed atomic request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: true,
      pruned_count: 2,
      results: [],
    }), { status: 200 })));
    const targets = [
      { session_id: 'current-session', paper_id: 'paper-1' },
      { session_id: 'older-session', paper_id: 'paper-2' },
    ];

    await autonomousAPI.prunePapersBatch(targets);

    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0][0]).toContain('/api/auto-research/papers/prune-batch');
    expect(fetch.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      targets,
      confirm: true,
    });
  });

  test('reports an all-or-none failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: 'One target is no longer eligible' }),
      { status: 409, headers: { 'Content-Type': 'application/json' } },
    )));

    await expect(autonomousAPI.prunePapersBatch([
      { session_id: 'session', paper_id: 'paper' },
    ])).rejects.toMatchObject({
      message: expect.stringContaining('One target is no longer eligible'),
      status: 409,
      kind: API_ERROR_KINDS.CONFLICT,
    });
  });

  test('rejects a non-success response body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: false,
      pruned_count: 0,
      results: [],
    }), { status: 200 })));

    await expect(autonomousAPI.prunePapersBatch([
      { session_id: 'session', paper_id: 'paper' },
    ])).rejects.toThrow('unsupported response');
  });
});
