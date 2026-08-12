import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import HelpTooltip from '../HelpTooltip';
import './ProofCheckModeModal.css';

const MODE_OPTIONS = [
  {
    value: 'one_round',
    title: 'Run One Round',
    description: 'Identify and attempt prompt-relevant candidates once, verify with Lean, store verified proofs, then finish without automatic pruning.',
    help: 'Runs one candidate-identification and Lean-verification round, stores verified proofs, then finishes. Automatic pruning does not run.',
  },
  {
    value: 'loop_with_pruning',
    title: 'Run Rounds on Loop With Pruning',
    description: 'Run proof-search rounds continuously until you press Stop, the backend restarts, or a hard repair condition ends the run. A round with no candidates immediately proceeds to the next round.',
    help: 'Runs proof rounds continuously in the current backend process with no automatic no-candidate limit. Press Stop when you want the loop to end; a backend restart or hard provider, model, source, or runtime repair condition also ends it. After every three accepted novel proofs, or material proof-memory context pressure, Rigor & Proofs may propose at most one redundant occurrence and Validator must approve. Nothing is deleted; no-prune is normal.',
  },
];

const SOURCE_TITLE_PREVIEW_LIMIT = 180;

function getSourceTitlePreview(sourceTitle) {
  const normalized = String(sourceTitle || '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= SOURCE_TITLE_PREVIEW_LIMIT) return normalized;
  return `${normalized.slice(0, SOURCE_TITLE_PREVIEW_LIMIT - 1).trimEnd()}…`;
}

export default function ProofCheckModeModal({
  sourceTitle = '',
  initialMode = 'one_round',
  busy = false,
  error = '',
  onClose,
  onConfirm,
}) {
  const titleId = useId();
  const descriptionId = useId();
  const statusId = useId();
  const dialogRef = useRef(null);
  const closeButtonRef = useRef(null);
  const [mode, setMode] = useState(
    MODE_OPTIONS.some((option) => option.value === initialMode) ? initialMode : 'one_round',
  );

  useEffect(() => {
    const previouslyFocused = document.activeElement;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && !busy) {
        event.preventDefault();
        onClose?.();
        return;
      }
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll(
        'button:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex="-1"])',
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
  }, [busy, onClose]);

  if (typeof document === 'undefined') return null;

  const sourceTitlePreview = getSourceTitlePreview(sourceTitle);

  const confirm = async (event) => {
    event.preventDefault();
    if (!busy) await onConfirm?.(mode);
  };

  return createPortal(
    <div
      className="proof-check-mode-modal__backdrop"
      role="presentation"
      onMouseDown={() => {
        if (!busy) onClose?.();
      }}
    >
      <form
        ref={dialogRef}
        className="proof-check-mode-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={`${descriptionId} ${statusId}`}
        tabIndex={-1}
        onSubmit={confirm}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="proof-check-mode-modal__header">
          <div>
            <span className="proof-check-mode-modal__eyebrow">Mathematical proofs</span>
            <h2 id={titleId}>Choose proof check mode</h2>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            aria-label="Close proof check mode"
            disabled={busy}
            onClick={onClose}
          >
            ×
          </button>
        </header>

        <div id={descriptionId} className="proof-check-mode-modal__description">
          <span>Run a Lean proof search for this source:</span>
          <strong
            className="proof-check-mode-modal__source-title"
            title={sourceTitle || undefined}
          >
            {sourceTitlePreview || 'Current proof source'}
          </strong>
        </div>

        <fieldset disabled={busy}>
          <legend>
            Proof run behavior
          </legend>
          {MODE_OPTIONS.map((option) => (
            <label
              key={option.value}
              className={`proof-check-mode-modal__option ${mode === option.value ? 'is-selected' : ''}`}
            >
              <input
                type="radio"
                name="proof-check-mode"
                value={option.value}
                checked={mode === option.value}
                onChange={() => setMode(option.value)}
              />
              <span>
                <span className="proof-check-mode-modal__option-title">
                  <strong>{option.title}</strong>
                  <HelpTooltip
                    label={`About ${option.title}`}
                    anchorClassName="help-tooltip-anchor--inline"
                    useFixedPosition
                  >
                    {option.help}
                  </HelpTooltip>
                </span>
                <small>{option.description}</small>
              </span>
            </label>
          ))}
        </fieldset>

        <div
          id={statusId}
          className={`proof-check-mode-modal__status ${error ? 'is-error' : ''}`}
          role={error ? 'alert' : 'status'}
          aria-live="polite"
          aria-atomic="true"
        >
          {error || (busy ? 'Starting proof run…' : '')}
        </div>

        <footer className="proof-check-mode-modal__actions">
          <button type="button" disabled={busy} onClick={onClose}>Cancel</button>
          <button type="submit" className="proof-check-mode-modal__confirm" disabled={busy}>
            {busy ? 'Starting…' : 'Start proof check'}
          </button>
        </footer>
      </form>
    </div>,
    document.body,
  );
}
