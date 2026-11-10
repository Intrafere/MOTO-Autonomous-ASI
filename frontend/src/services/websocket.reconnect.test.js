import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    FakeWebSocket.instances.push(this);
  }

  send() {}

  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  fail() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe('WebSocket reconnect', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.resetModules();
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('keeps retrying after a reconnect attempt fails before the socket opens', async () => {
    const { websocket: service } = await import('./websocket.js');

    await service.connect();
    expect(FakeWebSocket.instances).toHaveLength(1);
    FakeWebSocket.instances[0].open();

    // Backend goes away: the socket closes and a reconnect is scheduled.
    FakeWebSocket.instances[0].fail();
    await vi.advanceTimersByTimeAsync(3000);
    expect(FakeWebSocket.instances).toHaveLength(2);

    // The backend is still down, so the retry closes before it ever opens.
    FakeWebSocket.instances[1].fail();
    await vi.advanceTimersByTimeAsync(3000);
    expect(FakeWebSocket.instances).toHaveLength(3);

    FakeWebSocket.instances[2].fail();
    await vi.advanceTimersByTimeAsync(3000);
    expect(FakeWebSocket.instances).toHaveLength(4);

    service.disconnect();
  });

  it('does not open a second socket while one is already live', async () => {
    const { websocket: service } = await import('./websocket.js');

    await service.connect();
    FakeWebSocket.instances[0].open();
    await service.connect();
    expect(FakeWebSocket.instances).toHaveLength(1);

    service.disconnect();
  });
});
