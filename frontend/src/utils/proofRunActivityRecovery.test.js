import { describe, expect, it, vi } from 'vitest';
import {
  PROOF_RUN_ACTIVITY_REPLAY_FLAG,
  acceptLiveProofRunActivity,
  createProofRunActivityTracker,
  isReplayedProofRunActivity,
  recoverProofRunActivity,
} from './proofRunActivityRecovery';

const memoryTracker = (initial = {}) => {
  let stored = initial;
  return createProofRunActivityTracker({
    readState: () => stored,
    writeState: (state) => {
      stored = JSON.parse(JSON.stringify(state));
    },
  });
};

const activityEvent = (sequence, type = 'proof_candidate_list_review_rejected') => ({
  sequence,
  type,
  timestamp: `2026-10-11T09:0${sequence}:00Z`,
  data: { proof_run_id: 'run-a', proof_run_activity_sequence: sequence, list_attempt: sequence },
});

const activityPage = (events, extra = {}) => ({
  latest_sequence: events.length ? events[events.length - 1].sequence : 0,
  oldest_retained_sequence: events.length ? events[0].sequence : 0,
  events,
  gap: false,
  truncated: false,
  ...extra,
});

describe('proof-run activity tracker', () => {
  it('passes non-journaled payloads and drops already processed sequences', () => {
    const tracker = memoryTracker();
    expect(acceptLiveProofRunActivity({ source_id: 'x' }, tracker)).toBe(true);
    const event = { proof_run_id: 'run-a', proof_run_activity_sequence: 1 };
    expect(acceptLiveProofRunActivity(event, tracker)).toBe(true);
    expect(acceptLiveProofRunActivity(event, tracker)).toBe(false);
    expect(tracker.resumeFrom('run-a')).toBe(1);
  });

  it('keeps out-of-order sequences until the gap is filled', () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 1);
    tracker.accept('run-a', 4);
    expect(tracker.resumeFrom('run-a')).toBe(1);
    tracker.accept('run-a', 2);
    tracker.accept('run-a', 3);
    expect(tracker.resumeFrom('run-a')).toBe(4);
    expect(tracker.accept('run-a', 3)).toBe(false);
  });

  it('survives a reload through stored cursor state', () => {
    let stored = {};
    const options = {
      readState: () => stored,
      writeState: (state) => {
        stored = JSON.parse(JSON.stringify(state));
      },
    };
    const first = createProofRunActivityTracker(options);
    first.accept('run-a', 1);
    first.accept('run-a', 2);
    const reloaded = createProofRunActivityTracker(options);
    expect(reloaded.accept('run-a', 2)).toBe(false);
    expect(reloaded.resumeFrom('run-a')).toBe(2);
  });

  it('marks replayed payloads', () => {
    expect(isReplayedProofRunActivity({ [PROOF_RUN_ACTIVITY_REPLAY_FLAG]: true })).toBe(true);
    expect(isReplayedProofRunActivity({})).toBe(false);
  });
});

describe('recoverProofRunActivity', () => {
  it('replays only events the tab missed, in order, with original timestamps', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 1);
    tracker.accept('run-a', 2);
    const dispatch = vi.fn();
    const getActivity = vi.fn(async (_runId, { afterSequence }) => {
      expect(afterSequence).toBe(2);
      return activityPage([activityEvent(3), activityEvent(4, 'proof_run_round_started')]);
    });

    const replayed = await recoverProofRunActivity({
      listRuns: async () => ({ runs: [{ proof_run_id: 'run-a' }] }),
      getActivity,
      dispatch,
      tracker,
    });

    expect(replayed).toBe(2);
    expect(dispatch.mock.calls.map(([type]) => type)).toEqual([
      'proof_candidate_list_review_rejected',
      'proof_run_round_started',
    ]);
    expect(dispatch.mock.calls[0][1]).toMatchObject({
      list_attempt: 3,
      _serverTimestamp: '2026-10-11T09:03:00Z',
      [PROOF_RUN_ACTIVITY_REPLAY_FLAG]: true,
    });
    // A late live copy of a replayed event is now a duplicate.
    expect(tracker.accept('run-a', 4)).toBe(false);
  });

  it('baselines a run this tab never observed instead of replaying its history', async () => {
    const tracker = memoryTracker();
    const dispatch = vi.fn();
    const getActivity = vi.fn(async () => activityPage([], { latest_sequence: 9 }));

    await recoverProofRunActivity({
      listRuns: async () => ({ runs: [{ proof_run_id: 'run-a' }] }),
      getActivity,
      dispatch,
      tracker,
    });

    expect(getActivity).toHaveBeenCalledWith('run-a', { afterSequence: 0, limit: 0 });
    expect(dispatch).not.toHaveBeenCalled();
    expect(tracker.accept('run-a', 9)).toBe(false);
    expect(tracker.accept('run-a', 10)).toBe(true);
  });

  it('replays from the first event shown when a run was first seen mid-stream', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 5);
    const dispatch = vi.fn();
    const getActivity = vi.fn(async (_runId, { afterSequence }) => {
      expect(afterSequence).toBe(5);
      return activityPage([activityEvent(6)]);
    });

    await recoverProofRunActivity({
      listRuns: async () => ({ runs: [{ proof_run_id: 'run-a' }] }),
      getActivity,
      dispatch,
      tracker,
    });

    expect(dispatch).toHaveBeenCalledTimes(1);
  });

  it('pages through truncated responses and skips past evicted events', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 1);
    const dispatch = vi.fn();
    const getActivity = vi
      .fn()
      .mockResolvedValueOnce(activityPage(
        [activityEvent(4), activityEvent(5)],
        { gap: true, truncated: true, latest_sequence: 6 },
      ))
      .mockResolvedValueOnce(activityPage([activityEvent(6)]));

    await recoverProofRunActivity({
      listRuns: async () => ({ runs: [{ proof_run_id: 'run-a' }] }),
      getActivity,
      dispatch,
      tracker,
    });

    expect(getActivity.mock.calls.map(([, options]) => options.afterSequence)).toEqual([1, 5]);
    expect(dispatch.mock.calls.map(([, data]) => data.list_attempt)).toEqual([4, 5, 6]);
    expect(tracker.resumeFrom('run-a')).toBe(6);
  });

  it('stops querying finished runs once caught up', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 1);
    const getActivity = vi.fn(async () => activityPage([activityEvent(2)]));
    const listRuns = async () => ({ runs: [{ proof_run_id: 'run-a', status: 'completed' }] });

    await recoverProofRunActivity({ listRuns, getActivity, dispatch: vi.fn(), tracker });
    await recoverProofRunActivity({ listRuns, getActivity, dispatch: vi.fn(), tracker });

    expect(getActivity).toHaveBeenCalledTimes(1);
  });

  it('keeps querying active runs', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-a', 1);
    const getActivity = vi.fn(async () => activityPage([]));
    const listRuns = async () => ({ runs: [{ proof_run_id: 'run-a', status: 'running' }] });

    await recoverProofRunActivity({ listRuns, getActivity, dispatch: vi.fn(), tracker });
    await recoverProofRunActivity({ listRuns, getActivity, dispatch: vi.fn(), tracker });

    expect(getActivity).toHaveBeenCalledTimes(2);
  });

  it('forgets runs the backend no longer reports', async () => {
    const tracker = memoryTracker();
    tracker.accept('run-old', 1);
    await recoverProofRunActivity({
      listRuns: async () => ({ runs: [] }),
      getActivity: vi.fn(),
      dispatch: vi.fn(),
      tracker,
    });
    expect(tracker.resumeFrom('run-old')).toBeNull();
  });
});
