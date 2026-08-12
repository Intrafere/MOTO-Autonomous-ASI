import { afterEach, describe, expect, test, vi } from 'vitest';
import {
  API_ERROR_KINDS,
  MotoApiError,
  autonomousAPI,
  buildProofSourceKey,
  normalizeProofRunList,
  normalizeProofRunSnapshot,
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

  test('queues proof runs with explicit mode and normalizes identity', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      proof_run_id: 'proof-run-1',
      run_mode: 'loop_with_pruning',
      scope: 'manual',
      source_type: 'paper',
      source_id: 'manual_compiler_current',
      lifecycle_generation: 3,
      status: 'queued',
    }), { status: 200 })));

    const result = await autonomousAPI.runProofCheck({
      sourceType: 'paper',
      sourceId: 'manual_compiler_current',
      scope: 'manual',
      runMode: 'loop_with_pruning',
    });

    const [, options] = fetch.mock.calls[0];
    expect(JSON.parse(options.body)).toMatchObject({
      source_type: 'paper',
      source_id: 'manual_compiler_current',
      run_mode: 'loop_with_pruning',
    });
    expect(result).toMatchObject({
      proof_run_id: 'proof-run-1',
      lifecycle_generation: 3,
      source_key: 'manual:paper:manual_compiler_current',
    });
  });

  test('sends lifecycle generation for proof-run stop', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      proof_run_id: 'proof-run-2',
      run_mode: 'loop_with_pruning',
      scope: 'autonomous',
      source_type: 'brainstorm',
      source_id: 'topic-1',
      lifecycle_generation: 5,
      status: 'stopping',
    }), { status: 200 })));

    await autonomousAPI.stopProofRun('proof-run-2', 4);

    expect(fetch.mock.calls[0][0]).toContain('/proofs/runs/proof-run-2/stop');
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      expected_lifecycle_generation: 4,
    });
  });

  test('classifies stale-generation conflicts and old proof-run contracts actionably', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Lifecycle generation changed' }), {
        status: 409,
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Not Found' }), {
        status: 404,
      })));

    await expect(autonomousAPI.stopProofRun('run', 1)).rejects.toMatchObject({
      kind: API_ERROR_KINDS.CONFLICT,
      status: 409,
    });
    await expect(autonomousAPI.listProofRuns()).rejects.toMatchObject({
      kind: API_ERROR_KINDS.OLD_CONTRACT,
      status: 404,
    });
  });

  test('normalizes minimal run discovery shapes', () => {
    const snapshot = normalizeProofRunSnapshot({
      proof_run_id: 'proof-run-minimal',
      run_id: 'owning-run-minimal',
      source_type: 'paper',
      source_id: 'paper-1',
      generation: 2,
      status: 'running',
    });
    expect(snapshot.proof_run_id).toBe('proof-run-minimal');
    expect(snapshot.run_id).toBe('owning-run-minimal');
    expect(snapshot.source_key).toBe(buildProofSourceKey('autonomous', 'paper', 'paper-1'));
    expect(normalizeProofRunList({ proof_runs: [snapshot] }).runs).toHaveLength(1);
  });

  test('uses backend-resolved proof-run scope ahead of caller fallback', () => {
    const snapshot = normalizeProofRunSnapshot({
      proof_run_id: 'proof-run-backend-scope',
      source_type: 'paper',
      source_id: 'session-1:paper-1',
      scope: 'autonomous',
      status: 'queued',
    }, {
      scope: 'manual',
      sourceType: 'paper',
      sourceId: 'session-1:paper-1',
    });

    expect(snapshot.scope).toBe('autonomous');
    expect(snapshot.source_key).toBe(buildProofSourceKey('autonomous', 'paper', 'session-1:paper-1'));
  });

  test('preserves unbounded continuous mode through reload normalization', () => {
    const snapshot = normalizeProofRunSnapshot({
      proof_run_id: 'proof-run-progress',
      source_type: 'paper',
      source_id: 'paper-1',
      run_mode: 'loop_with_pruning',
      lifecycle_generation: 3,
      status: 'running',
    });
    expect(snapshot).toMatchObject({
      run_mode: 'loop_with_pruning',
      unbounded: true,
      status: 'running',
    });
  });

  test('fences live-context mutations with run and proof-set identity', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: true,
      scope: 'manual',
      proof_id: 'proof-1',
      run_id: 'run-1',
      live_context_status: 'pruned',
      proof_set_revision: 8,
    }), { status: 200 })));

    await autonomousAPI.updateProofLiveContext({
      proofId: 'proof-1',
      scope: 'manual',
      status: 'pruned',
      runId: 'run-1',
      proofSetRevision: 7,
      reason: 'Removed from this run context by the operator.',
      theoremHash: 'theorem-hash',
      leanHash: 'lean-hash',
    });

    expect(fetch.mock.calls[0][0]).toContain('/proofs/proof-1/live-context?scope=manual');
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({
      expected_run_id: 'run-1',
      expected_proof_set_revision: 7,
      expected_theorem_hash: 'theorem-hash',
      expected_lean_hash: 'lean-hash',
    });
  });
});
