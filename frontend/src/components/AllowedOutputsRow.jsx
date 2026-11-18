import React, { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import AllowedOutputOption from './AllowedOutputOption';
import UsageLevelBadge from './UsageLevelBadge';
import {
  MAX_CONCURRENT_PROOF_ATTEMPTS,
  buildProofSettingsPayload,
  parseConcurrentProofAttempts,
  proofKindsFromStatus,
  publishProofSettings,
  subscribeToProofSettings,
} from '../utils/proofRuntimeSettings';
import './AllowedOutputsRow.css';

const KIND_FIELDS = {
  singleShot: 'single_shot_proofs_enabled',
  master: 'master_proofs_enabled',
};

// Pixels kept between the pointer and the card's rounded corners.
const CARET_INSET = 18;

// Allowed Outputs pill shared by Autonomous Research and the Single Paper
// Writer. An output may carry an arrow that opens an options card under it;
// only one card is open at a time, so opening one arrow closes any other.
// Mathematical Proofs has the only card today: which kinds of proof discovery
// may propose, and Concurrent Proof Attempts. These are process-wide proof
// settings, the same ones the advanced proof settings edit, so each change
// saves immediately and is published to other surfaces.
export default function AllowedOutputsRow({
  rowTitle,
  proofOutputsAvailable,
  proofs,
  papers,
  optionsLocked = false,
  getProofStatus,
  updateProofSettings,
}) {
  const baseId = useId();
  const panelIds = { proofs: `${baseId}-proof-options` };
  const headingId = `${baseId}-proof-options-heading`;
  const concurrencyInputId = `${baseId}-concurrency`;
  const concurrencyUsageId = `${baseId}-concurrency-usage`;
  const toggleRefs = useRef({});
  const mountedRef = useRef(true);
  // The one open options card: null, or the key of the output it belongs to.
  const [openPanel, setOpenPanel] = useState(null);
  const expanded = openPanel === 'proofs';
  const [proofStatus, setProofStatus] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [concurrencyDraft, setConcurrencyDraft] = useState('');

  const showProofOptions = Boolean(proofOutputsAvailable && getProofStatus && updateProofSettings);

  const applyStatus = useCallback((status) => {
    if (!status || !mountedRef.current) return;
    setProofStatus(status);
    if (status.proof_max_parallel_candidates !== undefined) {
      setConcurrencyDraft(String(status.proof_max_parallel_candidates));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => subscribeToProofSettings(applyStatus), [applyStatus]);

  useEffect(() => {
    if (!expanded || proofStatus || !showProofOptions) return;
    let cancelled = false;
    setLoadError('');
    getProofStatus()
      .then((status) => { if (!cancelled) applyStatus(status); })
      .catch((error) => {
        if (!cancelled) setLoadError(`Could not load proof settings: ${error.message}`);
      });
    return () => { cancelled = true; };
  }, [expanded, proofStatus, showProofOptions, getProofStatus, applyStatus]);

  // One save at a time: a blur that follows Enter, or a click that follows a
  // blur, must not send a second request while the first is in flight.
  const savingRef = useRef(false);
  const saveProofSettings = async (overrides) => {
    if (savingRef.current) return null;
    savingRef.current = true;
    setSaving(true);
    setMessage('');
    try {
      const latest = await getProofStatus();
      const updated = await updateProofSettings(buildProofSettingsPayload(latest, overrides));
      applyStatus(updated);
      publishProofSettings(updated);
      return true;
    } catch (error) {
      if (mountedRef.current) setMessage(`Could not save proof settings: ${error.message}`);
      return false;
    } finally {
      savingRef.current = false;
      if (mountedRef.current) setSaving(false);
    }
  };

  const kinds = proofKindsFromStatus(proofStatus);
  const proofsOn = Boolean(proofs.checked);
  const settingsReady = Boolean(proofStatus);
  const controlsDisabled = optionsLocked || saving || !settingsReady || !proofsOn;

  const toggleKind = (kind, checked) => {
    const next = { ...kinds, [kind]: checked };
    if (!next.singleShot && !next.master) {
      setMessage('Keep at least one proof type on.');
      return;
    }
    saveProofSettings({ [KIND_FIELDS[kind]]: checked });
  };

  const savedConcurrency = proofStatus?.proof_max_parallel_candidates;
  const commitConcurrency = (value) => {
    const parsed = parseConcurrentProofAttempts(value);
    if (parsed === null) {
      setMessage(`Enter a whole number from 0 to ${MAX_CONCURRENT_PROOF_ATTEMPTS}.`);
      setConcurrencyDraft(savedConcurrency === undefined ? '' : String(savedConcurrency));
      return;
    }
    setConcurrencyDraft(String(parsed));
    if (parsed === savedConcurrency) return;
    saveProofSettings({ max_parallel_candidates: parsed }).then((saved) => {
      // A failed save shows the value that is actually in effect again.
      if (saved === false && mountedRef.current && savedConcurrency !== undefined) {
        setConcurrencyDraft(String(savedConcurrency));
      }
    });
  };

  const stepConcurrency = (delta) => {
    const current = parseConcurrentProofAttempts(concurrencyDraft) ?? savedConcurrency ?? 0;
    const next = Math.min(MAX_CONCURRENT_PROOF_ATTEMPTS, Math.max(0, current + delta));
    commitConcurrency(String(next));
  };

  const togglePanel = (key) => setOpenPanel((current) => (current === key ? null : key));
  const closePanel = () => {
    const key = openPanel;
    setOpenPanel(null);
    if (key) toggleRefs.current[key]?.focus();
  };
  const toggleProps = (key, label) => ({
    label,
    expanded: openPanel === key,
    panelId: panelIds[key],
    buttonRef: (node) => { toggleRefs.current[key] = node; },
    onToggle: () => togglePanel(key),
  });

  const draftCount = parseConcurrentProofAttempts(concurrencyDraft);
  let statusNote = '';
  if (loadError) statusNote = loadError;
  else if (!settingsReady) statusNote = 'Loading proof settings...';
  else if (!proofsOn) statusNote = 'Turn on Mathematical Proofs to change these options.';
  else if (optionsLocked) statusNote = 'Stop the run to change these options.';

  return (
    <>
      <div className="allowed-outputs-row allowed-outputs-row--with-options" title={rowTitle}>
        <span className="allowed-outputs-label">Allowed Outputs:</span>
        <span className="allowed-output-group">
          <AllowedOutputOption
            usageFeatureId="allowed_outputs.mathematical_proofs"
            title={proofs.title}
            checked={proofs.checked}
            onChange={proofs.onChange}
            disabled={proofs.disabled}
          >
            Mathematical Proofs
          </AllowedOutputOption>
          {showProofOptions && <OptionExpandToggle {...toggleProps('proofs', 'Proof options')} />}
        </span>
        <span className="allowed-outputs-divider" aria-hidden="true" />
        <span className="allowed-output-group">
          <AllowedOutputOption
            usageFeatureId="allowed_outputs.research_papers"
            title={papers.title}
            checked={papers.checked}
            onChange={papers.onChange}
            disabled={papers.disabled}
          >
            Research Papers
          </AllowedOutputOption>
        </span>
      </div>

      {showProofOptions && expanded && (
        <OptionPanel
          id={panelIds.proofs}
          anchor={toggleRefs.current.proofs}
          className={`proof-options-panel${proofsOn ? '' : ' is-inactive'}`}
          labelledBy={headingId}
          onClose={closePanel}
        >
          <header className="proof-options-panel__header">
            <h4 id={headingId} className="proof-options-panel__title">Proof options</h4>
            {saving && <span className="proof-options-panel__saving" role="status">Saving...</span>}
          </header>

          <fieldset className="proof-options-panel__group" disabled={controlsDisabled}>
            <legend className="proof-options-panel__legend">Proof types</legend>
            <ProofKindOption
              featureId="allowed_outputs.single_shot_proofs"
              title="Single-shot proofs"
              description="Each theorem is proved in one budget of up to five Lean attempts."
              checked={kinds.singleShot}
              onChange={(event) => toggleKind('singleShot', event.target.checked)}
            />
            <ProofKindOption
              featureId="allowed_outputs.master_proofs"
              title="Master proofs"
              description="Large theorems are built as a scaffold of steps solved in parallel, splitting steps as needed."
              checked={kinds.master}
              onChange={(event) => toggleKind('master', event.target.checked)}
            />
          </fieldset>

          <div className="proof-options-panel__concurrency">
            <div className="proof-options-panel__concurrency-copy">
              <label htmlFor={concurrencyInputId} className="proof-options-panel__option-title">
                Concurrent proof attempts
              </label>
              <span className="proof-options-panel__option-description">
                Default 6. Set 0 for unlimited. Same setting as Advanced Settings.
              </span>
            </div>
            <div className="proof-options-panel__concurrency-controls">
              <div className="proof-options-stepper">
                <button
                  type="button"
                  className="proof-options-stepper__button"
                  data-proof-stepper=""
                  aria-label="Fewer concurrent proof attempts"
                  onClick={() => stepConcurrency(-1)}
                  disabled={controlsDisabled || draftCount === 0}
                >
                  −
                </button>
                <input
                  id={concurrencyInputId}
                  className="proof-options-stepper__input"
                  type="text"
                  inputMode="numeric"
                  aria-describedby={concurrencyUsageId}
                  value={concurrencyDraft}
                  onChange={(event) => setConcurrencyDraft(event.target.value)}
                  onBlur={(event) => {
                    // Moving to + or − lets that click commit the typed value, so
                    // the blur save cannot disable the button under the click.
                    if (event.relatedTarget?.hasAttribute?.('data-proof-stepper')) return;
                    commitConcurrency(event.target.value);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      commitConcurrency(event.currentTarget.value);
                    }
                  }}
                  disabled={controlsDisabled}
                />
                <button
                  type="button"
                  className="proof-options-stepper__button"
                  data-proof-stepper=""
                  aria-label="More concurrent proof attempts"
                  onClick={() => stepConcurrency(1)}
                  disabled={controlsDisabled || draftCount === MAX_CONCURRENT_PROOF_ATTEMPTS}
                >
                  +
                </button>
              </div>
              <UsageLevelBadge
                id={concurrencyUsageId}
                featureId="proofs.concurrent_attempts"
                context={{ count: draftCount ?? '' }}
              />
            </div>
          </div>

          {(message || statusNote) && (
            <p
              className={`proof-options-panel__note${message ? ' is-error' : ''}`}
              role={message ? 'alert' : undefined}
            >
              {message || statusNote}
            </p>
          )}
        </OptionPanel>
      )}
    </>
  );
}

function ProofKindOption({ featureId, title, description, checked, onChange }) {
  const baseId = useId();
  const titleId = `${baseId}-title`;
  const descriptionId = `${baseId}-description`;
  const usageId = `${baseId}-usage`;
  return (
    <label className="proof-kind-option">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        aria-labelledby={titleId}
        aria-describedby={`${descriptionId} ${usageId}`}
      />
      <span className="proof-kind-option__copy">
        <span className="proof-kind-option__heading">
          <span id={titleId} className="proof-options-panel__option-title">{title}</span>
          <UsageLevelBadge id={usageId} featureId={featureId} />
        </span>
        <span id={descriptionId} className="proof-options-panel__option-description">{description}</span>
      </span>
    </label>
  );
}

// The arrow beside an output that opens or closes its options card.
function OptionExpandToggle({ label, expanded, panelId, buttonRef, onToggle }) {
  const lowered = label.charAt(0).toLowerCase() + label.slice(1);
  return (
    <button
      ref={buttonRef}
      type="button"
      className={`allowed-output-expand${expanded ? ' is-expanded' : ''}`}
      aria-expanded={expanded}
      aria-controls={panelId}
      aria-label={label}
      title={expanded ? `Hide ${lowered}` : `Show ${lowered}`}
      onClick={onToggle}
    >
      <svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true" focusable="false">
        <path d="M2.5 4.25 6 7.75l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.6"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

// Whether the items after the pill (Clear and similar buttons) fit on the
// pill's own line. Measured from widths alone, so moving the card cannot
// change the answer.
function sharesLineWithFollowing(pill, slot) {
  const row = slot.parentElement;
  if (!pill || !row) return true;
  const items = Array.from(row.children).filter((node) => node !== slot);
  const following = items.slice(items.indexOf(pill) + 1);
  if (!following.length) return true;
  const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
  const needed = following.reduce((sum, node) => sum + gap + node.offsetWidth, pill.offsetWidth);
  return needed <= row.clientWidth;
}

// An options card on its own line of the composer options row. When the pill
// shares its line with the buttons after it, the card goes below them so
// opening it never moves them; when they have already wrapped, it goes right
// under the pill. The card centres itself under the arrow that opened it,
// stays inside the row, and points at the arrow.
function OptionPanel({ id, anchor, className, labelledBy, onClose, children }) {
  const slotRef = useRef(null);
  const panelRef = useRef(null);
  const [placement, setPlacement] = useState(null);

  useLayoutEffect(() => {
    const slot = slotRef.current;
    const panel = panelRef.current;
    if (!slot || !panel || !anchor) return undefined;
    const pill = anchor.closest('.allowed-outputs-row');
    const update = () => {
      const width = panel.offsetWidth;
      const slotBox = slot.getBoundingClientRect();
      if (!width || !slotBox.width) return;
      const anchorBox = anchor.getBoundingClientRect();
      const center = anchorBox.left + anchorBox.width / 2 - slotBox.left;
      const left = Math.round(Math.min(Math.max(0, slotBox.width - width), Math.max(0, center - width / 2)));
      const caret = Math.round(Math.min(width - CARET_INSET, Math.max(CARET_INSET, center - left)));
      const afterRow = sharesLineWithFollowing(pill, slot);
      setPlacement((current) => (
        current && current.left === left && current.caret === caret && current.afterRow === afterRow
          ? current
          : { left, caret, afterRow }
      ));
    };
    update();
    window.addEventListener('resize', update);
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : null;
    [slot, slot.parentElement, pill].forEach((node) => {
      if (node) observer?.observe(node);
    });
    return () => {
      window.removeEventListener('resize', update);
      observer?.disconnect();
    };
  }, [anchor]);

  return (
    <div
      ref={slotRef}
      className="allowed-output-panel-slot"
      style={placement && !placement.afterRow ? { order: 0 } : undefined}
    >
      <section
        ref={panelRef}
        id={id}
        className={`allowed-output-panel ${className}${placement ? ' is-placed' : ''}`}
        style={placement ? {
          marginLeft: `${placement.left}px`,
          '--allowed-output-caret': `${placement.caret}px`,
        } : undefined}
        aria-labelledby={labelledBy}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            onClose();
          }
        }}
      >
        {children}
      </section>
    </div>
  );
}
