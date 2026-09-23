import { afterEach, expect, test, vi } from 'vitest';
import { proofCompetitionBenchmarksAPI } from './api';
afterEach(() => vi.unstubAllGlobals());

test('report helpers encode identity and disable caching', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ reports: [] }) });
  vi.stubGlobal('fetch', fetch);
  await proofCompetitionBenchmarksAPI.list({ current: true, offset: 25 });
  expect(fetch).toHaveBeenCalledWith('/api/proof-competition/benchmarks?current=true&offset=25&limit=25', expect.objectContaining({ cache: 'no-store' }));
  await proofCompetitionBenchmarksAPI.get('session name', 'report name');
  expect(fetch).toHaveBeenLastCalledWith('/api/proof-competition/benchmarks/session%20name/report%20name', expect.objectContaining({ cache: 'no-store' }));
});
