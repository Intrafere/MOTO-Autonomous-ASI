/**
 * Recovers proof-run Live Activity this tab never processed.
 *
 * The backend numbers every activity event a proof run publishes and keeps a
 * bounded process-local journal. A frozen, discarded, or reloaded tab can miss
 * live WebSocket messages, so this module tracks which sequence numbers this
 * tab has processed (per tab, surviving reloads) and replays the rest through
 * the normal WebSocket listeners after reconnect or resume.
 */
import { getNamespacedStorageKey } from './runtimeConfig';

export const PROOF_RUN_ACTIVITY_SEQUENCE_FIELD = 'proof_run_activity_sequence';
export const PROOF_RUN_ACTIVITY_REPLAY_FLAG = '_proof_run_replay';
const CURSOR_STORAGE_KEY = 'proof_run_activity_cursors';
const MAX_TRACKED_RUNS = 50;
const MAX_PAGES_PER_RUN = 20;
const TERMINAL_RUN_STATUSES = new Set(['completed', 'stopped', 'error']);

const memoryStore = { value: null };

function defaultReadState() {
  try {
    const raw = window.sessionStorage.getItem(getNamespacedStorageKey(CURSOR_STORAGE_KEY));
    return raw ? JSON.parse(raw) : {};
  } catch {
    return memoryStore.value || {};
  }
}

function defaultWriteState(state) {
  memoryStore.value = state;
  try {
    window.sessionStorage.setItem(
      getNamespacedStorageKey(CURSOR_STORAGE_KEY),
      JSON.stringify(state),
    );
  } catch {
    // Session storage may be unavailable; the in-memory copy still dedupes.
  }
}

function sanitizeEntry(entry) {
  if (!entry || typeof entry !== 'object') return null;
  const watermark = Number.isInteger(entry.watermark) && entry.watermark >= 0
    ? entry.watermark
    : null;
  const seen = Array.isArray(entry.seen)
    ? entry.seen.filter((value) => Number.isInteger(value) && value > 0)
    : [];
  return { watermark, seen, settled: entry.settled === true };
}

/**
 * Tracks processed sequences per run. A run has a known watermark (every
 * sequence at or below it was processed or is unrecoverable) plus the processed
 * sequences above it. An unknown watermark means the tab first observed the run
 * mid-stream.
 */
export function createProofRunActivityTracker({
  readState = defaultReadState,
  writeState = defaultWriteState,
} = {}) {
  let state = null;

  const load = () => {
    if (state) return state;
    const raw = readState();
    state = {};
    if (raw && typeof raw === 'object') {
      Object.entries(raw).forEach(([runId, entry]) => {
        const clean = sanitizeEntry(entry);
        if (clean) state[runId] = clean;
      });
    }
    return state;
  };

  const save = () => {
    const runIds = Object.keys(state);
    if (runIds.length > MAX_TRACKED_RUNS) {
      runIds.slice(0, runIds.length - MAX_TRACKED_RUNS).forEach((runId) => {
        delete state[runId];
      });
    }
    writeState(state);
  };

  const compact = (entry) => {
    if (entry.watermark === null) return;
    const seen = new Set(entry.seen.filter((value) => value > entry.watermark));
    while (seen.has(entry.watermark + 1)) {
      entry.watermark += 1;
      seen.delete(entry.watermark);
    }
    entry.seen = [...seen].sort((a, b) => a - b);
  };

  return {
    /** Record one sequence; returns false when this tab already processed it. */
    accept(runId, sequence) {
      if (!runId || !Number.isInteger(sequence) || sequence < 1) return true;
      const runs = load();
      let entry = runs[runId];
      if (!entry) {
        // The first event of a run proves nothing earlier existed to miss.
        entry = { watermark: sequence === 1 ? 0 : null, seen: [], settled: false };
        runs[runId] = entry;
      }
      if (entry.watermark !== null && sequence <= entry.watermark) return false;
      if (entry.seen.includes(sequence)) return false;
      entry.seen.push(sequence);
      compact(entry);
      save();
      return true;
    },

    /**
     * Sequence to replay after, or null when the run was never observed here
     * and only needs a baseline.
     */
    resumeFrom(runId) {
      const entry = load()[runId];
      if (!entry) return null;
      if (entry.watermark === null) {
        if (!entry.seen.length) return null;
        // Replay from the first event this tab showed for the run.
        entry.watermark = Math.max(0, Math.min(...entry.seen) - 1);
        compact(entry);
        save();
      }
      return entry.watermark;
    },

    /** Mark every sequence through `sequence` as processed or unrecoverable. */
    advanceTo(runId, sequence) {
      const runs = load();
      const entry = runs[runId] || { watermark: null, seen: [], settled: false };
      entry.watermark = Math.max(entry.watermark ?? 0, Math.max(0, sequence));
      compact(entry);
      runs[runId] = entry;
      save();
    },

    /** A finished run whose retained activity has all been processed. */
    isSettled(runId) {
      return load()[runId]?.settled === true;
    },

    markSettled(runId) {
      const entry = load()[runId];
      if (!entry || entry.settled) return;
      entry.settled = true;
      save();
    },

    /** Drop tracking for runs the backend no longer reports. */
    retainOnly(runIds) {
      const runs = load();
      const keep = new Set(runIds);
      let changed = false;
      Object.keys(runs).forEach((runId) => {
        if (!keep.has(runId)) {
          delete runs[runId];
          changed = true;
        }
      });
      if (changed) save();
    },
  };
}

const defaultTracker = createProofRunActivityTracker();

/** Gate for live WebSocket payloads: false means this tab already processed it. */
export function acceptLiveProofRunActivity(data, tracker = defaultTracker) {
  if (!data || typeof data !== 'object') return true;
  const sequence = data[PROOF_RUN_ACTIVITY_SEQUENCE_FIELD];
  if (!data.proof_run_id || sequence === undefined || sequence === null) return true;
  return tracker.accept(String(data.proof_run_id), Number(sequence));
}

export function isReplayedProofRunActivity(data) {
  return Boolean(data && data[PROOF_RUN_ACTIVITY_REPLAY_FLAG]);
}

/**
 * Replay retained events this tab has not processed, for every run the
 * backend still holds. Events are dispatched in sequence order.
 */
export async function recoverProofRunActivity({
  listRuns,
  getActivity,
  dispatch,
  tracker = defaultTracker,
}) {
  const { runs = [] } = await listRuns();
  tracker.retainOnly(runs.map((run) => run.proof_run_id).filter(Boolean));
  let replayed = 0;
  for (const run of runs) {
    const runId = run?.proof_run_id;
    if (!runId) continue;
    // Finished runs publish nothing more once this tab has caught up.
    const finished = TERMINAL_RUN_STATUSES.has(String(run.status || ''));
    if (finished && tracker.isSettled(runId)) continue;
    try {
      let after = tracker.resumeFrom(runId);
      if (after === null) {
        const baseline = await getActivity(runId, { afterSequence: 0, limit: 0 });
        tracker.advanceTo(runId, Number(baseline?.latest_sequence || 0));
        if (finished) tracker.markSettled(runId);
        continue;
      }
      let caughtUp = false;
      for (let page = 0; page < MAX_PAGES_PER_RUN; page += 1) {
        const activity = await getActivity(runId, { afterSequence: after });
        if (activity?.gap) {
          const oldestRetained = Number(activity.oldest_retained_sequence || 0);
          const lostThrough = Math.max(
            after,
            oldestRetained > 0 ? oldestRetained - 1 : Number(activity.latest_sequence || 0),
          );
          console.warn(
            `Proof run ${runId} activity ${after + 1}-${lostThrough} is no longer retained; replaying the rest.`,
          );
          tracker.advanceTo(runId, lostThrough);
        }
        const events = Array.isArray(activity?.events) ? activity.events : [];
        for (const event of events) {
          const sequence = Number(event.sequence);
          if (!tracker.accept(runId, sequence)) continue;
          dispatch(event.type, {
            ...(event.data || {}),
            _serverTimestamp: event.timestamp,
            [PROOF_RUN_ACTIVITY_REPLAY_FLAG]: true,
          });
          replayed += 1;
        }
        if (!activity?.truncated || !events.length) {
          caughtUp = !activity?.truncated;
          break;
        }
        after = Number(events[events.length - 1].sequence);
      }
      if (finished && caughtUp) tracker.markSettled(runId);
    } catch (error) {
      console.debug(`Could not recover activity for proof run ${runId}:`, error);
    }
  }
  return replayed;
}

/**
 * Run recovery after every WebSocket (re)connect and whenever the page becomes
 * visible or resumes from a browser freeze. Returns an uninstall function.
 */
export function installProofRunActivityRecovery({ websocket, api }) {
  let running = false;
  let rerun = false;

  const recover = async () => {
    if (running) {
      rerun = true;
      return;
    }
    running = true;
    try {
      do {
        rerun = false;
        try {
          await recoverProofRunActivity({
            listRuns: () => api.listProofRuns(),
            getActivity: (runId, options) => api.getProofRunActivity(runId, options),
            dispatch: (type, data) => websocket.emit(type, data),
          });
        } catch (error) {
          console.debug('Could not recover proof-run activity:', error);
        }
      } while (rerun);
    } finally {
      running = false;
    }
  };

  const handleVisible = () => {
    if (document.visibilityState === 'visible') void recover();
  };
  const handleResume = () => void recover();

  const unsubscribe = websocket.on('connected', handleResume);
  document.addEventListener('visibilitychange', handleVisible);
  // Page Lifecycle `resume` fires when a frozen tab is thawed.
  document.addEventListener('resume', handleResume);
  window.addEventListener('pageshow', handleResume);

  return () => {
    unsubscribe();
    document.removeEventListener('visibilitychange', handleVisible);
    document.removeEventListener('resume', handleResume);
    window.removeEventListener('pageshow', handleResume);
  };
}
