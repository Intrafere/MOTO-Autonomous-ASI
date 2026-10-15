import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeWebSocket {
  static OPEN = 1;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
    FakeWebSocket.instances.push(this);
  }

  send() {}

  close() {}

  receive(type, data, timestamp = '2026-10-11T09:00:00Z') {
    this.onmessage({ data: JSON.stringify({ type, data, timestamp }) });
  }
}

const rejection = (sequence) => ({
  proof_run_id: 'proof-run-a',
  proof_run_activity_sequence: sequence,
  list_attempt: sequence,
});

describe('WebSocket proof-run activity replay', () => {
  beforeEach(() => {
    vi.resetModules();
    window.sessionStorage.clear();
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('drops duplicate live events and replays missed ones after reconnect', async () => {
    const { websocket } = await import('./websocket');
    const { installProofRunActivityRecovery } = await import('../utils/proofRunActivityRecovery');
    const received = [];
    websocket.on('proof_candidate_list_review_rejected', (data) => received.push(data));

    await websocket.connect();
    const socket = FakeWebSocket.instances[0];
    socket.receive('proof_candidate_list_review_rejected', rejection(1));
    socket.receive('proof_candidate_list_review_rejected', rejection(1));
    expect(received.map((data) => data.list_attempt)).toEqual([1]);

    const api = {
      listProofRuns: vi.fn(async () => ({ runs: [{ proof_run_id: 'proof-run-a' }] })),
      getProofRunActivity: vi.fn(async (_runId, { afterSequence }) => ({
        latest_sequence: 3,
        oldest_retained_sequence: 1,
        after_sequence: afterSequence,
        events: [1, 2, 3]
          .filter((sequence) => sequence > afterSequence)
          .map((sequence) => ({
            sequence,
            type: 'proof_candidate_list_review_rejected',
            timestamp: `2026-10-11T09:0${sequence}:00Z`,
            data: rejection(sequence),
          })),
        gap: false,
        truncated: false,
      })),
    };
    const uninstall = installProofRunActivityRecovery({ websocket, api });

    websocket.emit('connected', {});
    await vi.waitFor(() => expect(received).toHaveLength(3));
    expect(api.getProofRunActivity).toHaveBeenCalledWith('proof-run-a', { afterSequence: 1 });
    expect(received.map((data) => data.list_attempt)).toEqual([1, 2, 3]);
    expect(received[2]._proof_run_replay).toBe(true);
    expect(received[2]._serverTimestamp).toBe('2026-10-11T09:03:00Z');

    // A late live copy of a replayed event is not shown twice.
    socket.receive('proof_candidate_list_review_rejected', rejection(3));
    socket.receive('proof_candidate_list_review_rejected', rejection(4));
    expect(received.map((data) => data.list_attempt)).toEqual([1, 2, 3, 4]);
    uninstall();
  });
});
