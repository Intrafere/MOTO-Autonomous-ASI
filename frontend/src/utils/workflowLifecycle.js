export function reconcileAutonomousLifecycle({
  status,
  wasRunning = false,
  seenTerminalEventIds = new Set(),
} = {}) {
  if (!status || typeof status.is_running !== 'boolean') {
    return null;
  }

  const isRunning = status.is_running;
  const terminalEvent = !isRunning && status.terminal_event
    ? status.terminal_event
    : null;
  const terminalEventId = terminalEvent?.terminal_event_id || '';
  const shouldRecoverTerminalEvent = Boolean(
    terminalEventId
    && !seenTerminalEventIds.has(terminalEventId)
  );

  return {
    isRunning,
    clearPendingLifecycle: true,
    lifecycleEnded: wasRunning && !isRunning,
    runId: status.run_id || '',
    lifecycleGeneration: Number(status.lifecycle_generation || 0),
    terminalEvent,
    shouldRecoverTerminalEvent,
  };
}
