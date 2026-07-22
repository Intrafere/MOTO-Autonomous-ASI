import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { workflowAPI } from '../services/api';
import { websocket } from '../services/websocket';
import {
  getSolutionPathSettingsMode,
  isSolutionPathSnapshotAtLeast,
  solutionPathEventMatches,
} from '../utils/solutionPathPresentation';
import './SolutionPathModal.css';

export default function SolutionPathModal({ snapshot, onClose, onSnapshotChange, onOpenSettings }) {
  const dialogRef = useRef(null);
  const closeButtonRef = useRef(null);
  const snapshotRef = useRef(snapshot);
  const retryButtonRefs = useRef(new Map());
  const [retryingProposalId, setRetryingProposalId] = useState('');
  const [retryError, setRetryError] = useState('');
  const [retryStatus, setRetryStatus] = useState('');

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    let requestSequence = 0;
    const refresh = async (event = {}) => {
      const current = snapshotRef.current;
      if (!solutionPathEventMatches(event, current, current?.mode || '')) return;
      const sequence = ++requestSequence;
      try {
        const nextSnapshot = await workflowAPI.getSolutionPath();
        if (
          sequence === requestSequence
          && nextSnapshot?.run_id === current?.run_id
          && isSolutionPathSnapshotAtLeast(nextSnapshot, snapshotRef.current)
        ) {
          snapshotRef.current = nextSnapshot;
          onSnapshotChange?.(nextSnapshot);
        }
      } catch {
        // The next workflow event or explicit retry will refresh the modal.
      }
    };
    const events = [
      'solution_path_proposal_queued',
      'solution_path_proposal_reviewing',
      'solution_path_updated',
      'solution_path_proposal_rejected',
      'solution_path_proposal_retry_queued',
      'solution_path_proposal_user_repair_required',
      'solution_path_proposal_resumed',
    ];
    events.forEach((eventName) => websocket.on(eventName, refresh));
    return () => {
      requestSequence += 1;
      events.forEach((eventName) => websocket.off(eventName, refresh));
    };
  }, [onSnapshotChange]);
  useEffect(() => {
    const previouslyFocused = document.activeElement;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable.length) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      previouslyFocused?.focus?.();
    };
  }, [onClose]);

  if (!snapshot || typeof document === 'undefined') return null;
  const steps = Array.isArray(snapshot.steps) ? snapshot.steps : [];
  const repairs = Array.isArray(snapshot.repairs) ? snapshot.repairs : [];

  const retryRepair = async (repair) => {
    setRetryError('');
    setRetryStatus('');
    setRetryingProposalId(repair.proposal_id);
    const retryFence = {
      runId: snapshot.run_id,
      lifecycleGeneration: repair.lifecycle_generation || snapshot.lifecycle_generation,
      proposalId: repair.proposal_id,
    };
    try {
      await workflowAPI.resumeSolutionPathProposal({
        runId: retryFence.runId,
        proposalId: retryFence.proposalId,
        lifecycleGeneration: retryFence.lifecycleGeneration,
      });
      const nextSnapshot = await workflowAPI.getSolutionPath();
      if (
        nextSnapshot?.run_id !== retryFence.runId
        || Number(nextSnapshot?.lifecycle_generation) !== Number(retryFence.lifecycleGeneration)
      ) {
        throw new Error('The active solution-path run changed. Reopen the current route before retrying.');
      }
      if (isSolutionPathSnapshotAtLeast(nextSnapshot, snapshotRef.current)) {
        snapshotRef.current = nextSnapshot;
        onSnapshotChange?.(nextSnapshot);
      }
      setRetryStatus('Review retry queued. The route will refresh as the review progresses.');
    } catch (error) {
      setRetryError(error.message || 'The solution-path update could not be retried.');
    } finally {
      setRetryingProposalId('');
      requestAnimationFrame(() => retryButtonRefs.current.get(repair.proposal_id)?.focus());
    }
  };

  return createPortal(
    <div className="solution-path-modal__backdrop" role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className="solution-path-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="solution-path-title"
        aria-describedby="solution-path-guidance"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="solution-path-modal__header">
          <div>
            <div className="solution-path-modal__eyebrow">{snapshot.mode || 'Workflow'} · Solution Path</div>
            <h2 id="solution-path-title">Current solution path</h2>
          </div>
          <button ref={closeButtonRef} type="button" onClick={onClose} aria-label="Close solution path">×</button>
        </header>
        <div className="solution-path-modal__meta">
          {snapshot.revision && <span>Revision {snapshot.revision}</span>}
          {snapshot.run_id && <span>Run {snapshot.run_id}</span>}
          {snapshot.lifecycle_generation != null && <span>Generation {snapshot.lifecycle_generation}</span>}
          {snapshot.ownership && <span>{snapshot.ownership === 'resumable' ? 'Resumable run' : 'Active run'}</span>}
          <span>{snapshot.ordering === 'unordered' ? 'Flexible order' : 'Ordered route'}</span>
          {snapshot.pending_proposals > 0 && <span>{snapshot.pending_proposals} update(s) under review</span>}
        </div>
        <p id="solution-path-guidance" className="solution-path-modal__guidance">
          This is a distillation attempt at the best currently known path from the available context.
          It may be wrong or incomplete, is optional guidance, and may be deviated from whenever a
          better route serves the user prompt or current subgoal.
        </p>
        {repairs.length > 0 && (
          <section className="solution-path-modal__repair" aria-labelledby="solution-path-repair-title">
            <h3 id="solution-path-repair-title">Settings repair required</h3>
            <p>
              Update the Main Submitter 1 provider, credentials, or context settings, then retry the
              blocked review. The workflow remains resumable.
            </p>
            {repairs.map((repair) => (
              <div key={repair.proposal_id} className="solution-path-modal__repair-item">
                <strong>{String(repair.reason || 'reviewer failure').replaceAll('_', ' ')}</strong>
                {repair.detail && <p>{repair.detail}</p>}
                <div className="solution-path-modal__repair-actions">
                  <button type="button" onClick={() => onOpenSettings?.(getSolutionPathSettingsMode(snapshot))}>
                    Open Main Submitter 1 settings
                  </button>
                  <button
                    ref={(node) => {
                      if (node) retryButtonRefs.current.set(repair.proposal_id, node);
                      else retryButtonRefs.current.delete(repair.proposal_id);
                    }}
                    type="button"
                    onClick={() => retryRepair(repair)}
                    disabled={Boolean(retryingProposalId)}
                  >
                    {retryingProposalId === repair.proposal_id ? 'Retrying…' : 'Retry review'}
                  </button>
                </div>
              </div>
            ))}
            {retryError && <p className="solution-path-modal__repair-error" role="alert">{retryError}</p>}
            {retryStatus && <p className="solution-path-modal__repair-status" role="status" aria-live="polite">{retryStatus}</p>}
          </section>
        )}
        {snapshot.main_route && (
          <section className="solution-path-modal__main-route" aria-label="Main route to solution">
            <strong>Main route to solution</strong>
            <p>{snapshot.main_route}</p>
          </section>
        )}
        {!snapshot.enabled || steps.length === 0 ? (
          <div className="solution-path-modal__empty">
            <strong>No approved route yet</strong>
            <p>{snapshot.message || 'No solution path is available yet.'}</p>
            {snapshot.acceptance_count >= 5 && (
              <small>{snapshot.acceptance_count} accepted brainstorm ideas</small>
            )}
          </div>
        ) : (
          React.createElement(
            snapshot.ordering === 'unordered' ? 'ul' : 'ol',
            {
              className: `solution-path-modal__steps ${snapshot.ordering === 'unordered' ? 'unordered' : ''}`,
              'aria-label': snapshot.ordering === 'unordered'
                ? 'Flexible solution route steps'
                : 'Ordered solution route steps',
            },
            steps.map((step) => (
              <li
                key={step.step_id}
                className={`solution-path-modal__step status-${step.status}`}
                aria-label={`${step.title}: ${step.status}`}
              >
                <span className="solution-path-modal__marker" aria-hidden="true" />
                <div>
                  <div className="solution-path-modal__step-title">{step.title}</div>
                  {step.description && <p>{step.description}</p>}
                  <span className="solution-path-modal__status">
                    <span aria-hidden="true">{step.status === 'complete' ? '✓ ' : ''}</span>
                    {step.status}
                  </span>
                </div>
              </li>
            ))
          )
        )}
      </section>
    </div>,
    document.body
  );
}
