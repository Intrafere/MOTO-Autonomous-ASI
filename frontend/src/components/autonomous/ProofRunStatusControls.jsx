import React, { useState } from 'react';
import { proofRunStatusLabel } from '../../hooks/useProofCheckRuntime';
import './ProofRunStatusControls.css';

const TERMINAL_STATUSES = new Set(['completed', 'error', 'stopped', 'repair_required']);

function readable(value, fallback = 'Idle') {
  return value ? String(value).replaceAll('_', ' ') : fallback;
}

export default function ProofRunStatusControls({
  run,
  sourceLabel = '',
  onStop,
}) {
  const [pendingAction, setPendingAction] = useState('');
  const [error, setError] = useState('');
  if (!run?.proof_run_id) return null;

  const status = String(run.status || 'unknown');
  const round = Number(
    run.current_round || run.round_index || run.proof_round_index || run.rounds_completed || 0
  );
  const pruningStatus = run.pruning_status || run.pruning?.status || 'idle';
  const isContinuous = run.run_mode === 'loop_with_pruning' || run.unbounded === true;
  const isStopping = status === 'stopping';
  const isTerminal = TERMINAL_STATUSES.has(status);
  const canStop = !isTerminal;

  const invoke = async (action, callback) => {
    if (!callback || pendingAction) return;
    try {
      setPendingAction(action);
      setError('');
      await callback(run, run.lifecycle_generation);
    } catch (actionError) {
      setError(actionError?.message || `Failed to ${action}.`);
    } finally {
      setPendingAction('');
    }
  };

  return (
    <section
      className="proof-run-status-controls"
      aria-label="Proof run status"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="proof-run-status-controls__facts">
        <span><strong>Source</strong> {sourceLabel || `${run.source_type} ${run.source_id}`}</span>
        <span><strong>Round</strong> {round || 'Preparing'}</span>
        <span><strong>Status</strong> {proofRunStatusLabel(run)}</span>
        <span><strong>Pruning</strong> {readable(pruningStatus)}</span>
      </div>
      {isContinuous && status === 'running' && (
        <div className="proof-run-status-controls__progress" role="status" aria-live="polite">
          Continuous search is active. Every completed round starts the next round automatically until you press Stop.
        </div>
      )}
      {status === 'repair_required' && (
        <div className="proof-run-status-controls__progress">
          Repair the provider or model settings, then start a new proof loop.
        </div>
      )}
      {isTerminal && run.terminal_reason && (
        <div className="proof-run-status-controls__progress">
          <strong>Ended because</strong> {readable(run.terminal_reason, 'Unknown reason')}
        </div>
      )}
      <div className="proof-run-status-controls__actions">
        {canStop && (
          <button
            type="button"
            className="proof-run-status-controls__stop"
            disabled={Boolean(pendingAction) || isStopping}
            onClick={() => invoke('stop proof run', onStop)}
          >
            {isStopping || pendingAction === 'stop proof run'
              ? 'Stopping…'
              : (isContinuous ? 'Stop Loop' : 'Stop Proof Check')}
          </button>
        )}
      </div>
      {error && <div className="proof-run-status-controls__error" role="alert">{error}</div>}
    </section>
  );
}
