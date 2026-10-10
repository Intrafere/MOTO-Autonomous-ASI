import React, { useEffect, useRef, useState } from 'react';
import { syntheticLib4API } from '../../services/api';
import useExplicitRequest from '../../hooks/useExplicitRequest';
import {
  describeTradeReasonCode,
  previewErrorMessage,
} from '../../utils/syntheticLibPresentation';

// Planned default from the canonical plan; must be server-sourced at launch and
// never treated as permanent policy.
export const PLANNED_QUALIFYING_PROOF_THRESHOLD = 20;

const MAX_OFFER_SELECTION = 500;

const describeEligibilityError = (error) => previewErrorMessage(
  error,
  'check which proofs could be offered',
  'The local offer-eligibility check failed.',
);
const describeOfferError = (error) => previewErrorMessage(
  error,
  'record a local offer selection',
  'Recording the local offer selection failed.',
);

const describePrepareError = (error) => previewErrorMessage(
  error,
  'prepare existing proofs',
  'Preparing existing proofs failed.',
);

const fetchEligibility = () => syntheticLib4API.getTradeEvidenceDiagnostics();
const runLegacyMigration = (dryRun) => syntheticLib4API.migrateLegacyProofs(dryRun);

const LEGACY_SKIP_LABELS = {
  legacy_port_rank_unknown: 'no recognizable stored novelty rank',
  legacy_port_not_legacy_record: 'not from a released build',
  legacy_port_identity_conflict: 'stored identity does not match the proof',
  legacy_port_lean_source_missing: 'no Lean source',
  legacy_port_record_invalid: 'unreadable record',
  legacy_port_store_not_regular: 'store is not a regular folder',
  legacy_port_store_unreadable: 'store could not be read',
};

function describeMigration(result) {
  const skipped = Object.entries(result.skipped || {})
    .map(([reason, count]) => `${count} ${LEGACY_SKIP_LABELS[reason] || reason}`)
    .join('; ');
  const skippedText = skipped ? ` Skipped: ${skipped}.` : '';
  if (result.dry_run) {
    return `${result.eligible_count} existing proof${result.eligible_count === 1 ? '' : 's'} `
      + `can be prepared; ${result.already_qualified_count} already prepared.${skippedText}`;
  }
  return `Prepared ${result.migrated_count} existing proof${result.migrated_count === 1 ? '' : 's'} `
    + `locally. Proof code was not changed and nothing was uploaded.${skippedText}`;
}

// Records the explicit selection locally and asks for a local batch. Before the
// frozen contract exists nothing is canonically eligible, so this never uploads.
// If the panel closes after the snapshot, the batch request is never sent.
async function recordOffer(clientProofIds, { isCurrent } = {}) {
  const { snapshot } = await syntheticLib4API.createTradeSelectionSnapshot(clientProofIds);
  if (isCurrent && !isCurrent()) return null;
  const result = await syntheticLib4API.prepareTradeBatch(snapshot.snapshot_id);
  if (result.outcome === 'blocked' || result.outcome === 'invalidated') {
    const reasons = result.reason_codes?.length ? `: ${result.reason_codes.join(', ')}` : '';
    return `Selection recorded locally; batch ${result.outcome}${reasons}. Nothing was uploaded.`;
  }
  return 'Selection recorded as a local batch. Upload is not available before launch; nothing was uploaded.';
}

function EligibilityLabel({ item }) {
  const eligible = item.eligible_to_offer === true;
  return (
    <>
      <span aria-hidden="true">{eligible ? '✓ ' : '✕ '}</span>
      {eligible ? 'Eligible to offer' : 'Not eligible'}
    </>
  );
}

function ReasonList({ codes, label }) {
  if (!codes?.length) return '—';
  return (
    <ul aria-label={label} style={{ margin: 0, paddingLeft: '1.1em' }}>
      {codes.map((code) => (
        <li key={code}>
          {describeTradeReasonCode(code)} <code className="wrap-anywhere">{code}</code>
        </li>
      ))}
    </ul>
  );
}

export default function SyntheticLibOfferPanel({ visible = false, launched = false }) {
  const [eligibilityState, runEligibilityCheck] = useExplicitRequest(
    visible,
    fetchEligibility,
    describeEligibilityError,
  );
  const [offerState, runOffer] = useExplicitRequest(visible, recordOffer, describeOfferError);
  const [prepareState, runPrepare] = useExplicitRequest(
    visible,
    runLegacyMigration,
    describePrepareError,
  );
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [confirming, setConfirming] = useState(false);
  const [confirmingPrepare, setConfirmingPrepare] = useState(false);
  const offerButtonRef = useRef(null);
  const checkButtonRef = useRef(null);
  const confirmButtonRef = useRef(null);
  const prepareCheckRef = useRef(null);
  const prepareTriggerRef = useRef(null);
  const prepareConfirmRef = useRef(null);

  useEffect(() => {
    if (!visible) {
      setSelectedIds(new Set());
      setConfirming(false);
      setConfirmingPrepare(false);
    }
  }, [visible]);

  useEffect(() => {
    if (confirming) confirmButtonRef.current?.focus();
  }, [confirming]);

  useEffect(() => {
    if (confirmingPrepare) prepareConfirmRef.current?.focus();
  }, [confirmingPrepare]);

  if (!visible) return null;

  const audit = eligibilityState.data;
  const diagnostics = audit?.diagnostics || [];
  const offerable = diagnostics.filter(
    (item) => item.eligible_to_offer === true && item.client_proof_id,
  );
  const busy = eligibilityState.status === 'running'
    || offerState.status === 'running'
    || prepareState.status === 'running';
  const preparePreview = prepareState.data?.dry_run ? prepareState.data : null;
  const selectionTooLarge = selectedIds.size > MAX_OFFER_SELECTION;

  const checkEligibility = () => {
    setSelectedIds(new Set());
    setConfirming(false);
    return runEligibilityCheck();
  };

  const toggleSelected = (clientProofId) => {
    setConfirming(false);
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(clientProofId)) next.delete(clientProofId);
      else next.add(clientProofId);
      return next;
    });
  };

  const cancelConfirmation = () => {
    setConfirming(false);
    offerButtonRef.current?.focus();
  };

  const confirmOffer = async () => {
    setConfirming(false);
    // Keep focus on a control that stays enabled while the offer is recorded.
    checkButtonRef.current?.focus();
    const succeeded = await runOffer([...selectedIds]);
    if (succeeded) setSelectedIds(new Set());
  };

  const checkExistingProofs = () => {
    setConfirmingPrepare(false);
    return runPrepare(true);
  };

  const cancelPrepare = () => {
    setConfirmingPrepare(false);
    prepareTriggerRef.current?.focus();
  };

  const confirmPrepare = async () => {
    setConfirmingPrepare(false);
    prepareCheckRef.current?.focus();
    const succeeded = await runPrepare(false);
    if (succeeded) await runEligibilityCheck();
  };

  const handlePrepareKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      cancelPrepare();
    }
  };

  const handleConfirmationKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      cancelConfirmation();
    }
  };

  return (
    <section className="proof-search-panel" aria-labelledby="syntheticlib-offer-title">
      <div className="proof-search-panel__header">
        <div>
          <h3 id="syntheticlib-offer-title">Offer proofs for SyntheticLib access</h3>
          {!launched && (
            <span className="connectivity-status connectivity-status--pending">
              Private development
            </span>
          )}
        </div>
      </div>

      <ul aria-label="How offering proofs works">
        <li>This is an exchange for SyntheticLib access, not a donation.</li>
        <li>
          SyntheticLib independently reruns compile, uniqueness, policy, and novelty checks.
          A local MOTO compile or novelty rank is never SyntheticLib acceptance or credit.
        </li>
        <li>
          Access is planned to require {PLANNED_QUALIFYING_PROOF_THRESHOLD} qualifying proofs,
          not {PLANNED_QUALIFYING_PROOF_THRESHOLD} uploaded files; SyntheticLib sets the final
          requirement.
        </li>
        <li>
          An offered proof may end up pending, rejected, or in cooldown; only proofs
          SyntheticLib accepts count toward access.
        </li>
        <li>
          Only the proofs you select are sent, with minimized provenance plus authorization
          and status metadata. Nothing else leaves your computer.
        </li>
        <li>Legal review and acceptance happen only on SyntheticLib&apos;s own site.</li>
        <li>Nothing is sent in the background. You choose each proof and can cancel before upload.</li>
      </ul>
      {!launched && (
        <p>
          Before launch, eligibility, selection, and batches are local metadata only. Uploading,
          authorization, and account status are not available.
        </p>
      )}

      <h4 id="syntheticlib-prepare-title">Existing proofs</h4>
      <p>
        Proofs saved by earlier MOTO versions have no local trade identity yet. Preparing them
        adds an ID to each proof file in every proof store, including archived manual runs and
        past research sessions, so they can be offered. Their Lean code is not changed, nothing
        is uploaded, and SyntheticLib re-checks every proof it receives.
      </p>
      <button
        type="button"
        className="btn-ghost"
        ref={prepareCheckRef}
        onClick={() => {
          if (!busy) checkExistingProofs();
        }}
        aria-disabled={busy}
        aria-describedby="syntheticlib-prepare-title"
      >
        {prepareState.status === 'running' ? 'Checking existing proofs…' : 'Check existing proofs'}
      </button>
      {preparePreview && preparePreview.eligible_count > 0 && !confirmingPrepare && (
        <button
          type="button"
          className="btn-ghost"
          ref={prepareTriggerRef}
          onClick={() => setConfirmingPrepare(true)}
          disabled={busy}
          style={{ marginLeft: '8px' }}
        >
          Prepare {preparePreview.eligible_count} existing proof
          {preparePreview.eligible_count === 1 ? '' : 's'}
        </button>
      )}
      {confirmingPrepare && preparePreview && (
        <div
          role="group"
          aria-label="Confirm preparing existing proofs"
          style={{ marginTop: '8px' }}
          onKeyDown={handlePrepareKeyDown}
        >
          <p>
            Add local trade IDs to {preparePreview.eligible_count} existing proof
            {preparePreview.eligible_count === 1 ? '' : 's'} across all proof stores? This writes
            to the proof files on this computer only; Lean code is unchanged and nothing is
            uploaded.
          </p>
          <button
            type="button"
            className="btn-ghost"
            ref={prepareConfirmRef}
            onClick={confirmPrepare}
          >
            Confirm prepare
          </button>
          <button type="button" className="btn-ghost" onClick={cancelPrepare}>
            Cancel
          </button>
        </div>
      )}

      <h4>Offer</h4>
      <button
        type="button"
        className="btn-ghost"
        ref={checkButtonRef}
        onClick={() => {
          if (!busy) checkEligibility();
        }}
        aria-disabled={busy}
      >
        {eligibilityState.status === 'running' ? 'Checking eligibility…' : 'Check offer eligibility'}
      </button>

      <div role="status" aria-live="polite">
        {prepareState.status === 'running' && <p>Checking existing proofs…</p>}
        {prepareState.status === 'error' && <p>{prepareState.error}</p>}
        {prepareState.status === 'complete' && prepareState.data && (
          <p>{describeMigration(prepareState.data)}</p>
        )}
        {eligibilityState.status === 'running' && <p>Checking which proofs could be offered…</p>}
        {offerState.status === 'running' && <p>Recording the local offer selection…</p>}
        {selectionTooLarge && <p>At most {MAX_OFFER_SELECTION} proofs can be offered at once.</p>}
        {eligibilityState.status === 'error' && <p>{eligibilityState.error}</p>}
        {offerState.status === 'error' && <p>{offerState.error}</p>}
        {offerState.status === 'complete' && offerState.data && <p>{offerState.data}</p>}
        {audit && offerable.length === 0 && (
          <p>
            No proofs are eligible to offer yet. Eligibility opens only after SyntheticLib
            publishes its novelty-rank mapping and frozen proof-trade contract.
          </p>
        )}
        {audit && offerable.length > 0 && (
          <p>{offerable.length} proof{offerable.length === 1 ? '' : 's'} eligible to offer.</p>
        )}
      </div>

      {audit && offerable.length === 0 && audit.reason_codes?.length > 0 && (
        <ReasonList codes={audit.reason_codes} label="Why no proofs are eligible yet" />
      )}

      {audit && diagnostics.length > 0 && (
        <div className="table-scroll">
          <table aria-label="Proof offer eligibility" style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th scope="col">Select</th>
                <th scope="col">Proof</th>
                <th scope="col">Status</th>
                <th scope="col">Reasons</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.map((item) => {
                const canSelect = item.eligible_to_offer === true && Boolean(item.client_proof_id);
                const proofLabel = item.client_proof_id || item.record_key;
                return (
                  <tr key={item.record_key}>
                    <td>
                      {canSelect ? (
                        <input
                          type="checkbox"
                          aria-label={`Offer proof ${proofLabel}`}
                          checked={selectedIds.has(item.client_proof_id)}
                          onChange={() => toggleSelected(item.client_proof_id)}
                          disabled={busy}
                        />
                      ) : (
                        <span className="visually-hidden">Not selectable</span>
                      )}
                    </td>
                    <th scope="row" style={{ fontWeight: 'normal', textAlign: 'left' }}>
                      <code className="wrap-anywhere">{proofLabel}</code>
                    </th>
                    <td><EligibilityLabel item={item} /></td>
                    <td><ReasonList codes={item.reason_codes} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {offerable.length > 0 && (
        <div style={{ marginTop: '8px' }}>
          <button
            type="button"
            className="btn-ghost"
            ref={offerButtonRef}
            onClick={() => setConfirming(true)}
            disabled={selectedIds.size === 0 || selectionTooLarge || busy}
          >
            Offer selected proofs ({selectedIds.size})
          </button>
        </div>
      )}

      {confirming && (
        <div
          role="group"
          aria-label="Confirm proof offer"
          style={{ marginTop: '8px' }}
          onKeyDown={handleConfirmationKeyDown}
        >
          <p>
            Record {selectedIds.size} selected proof{selectedIds.size === 1 ? '' : 's'} as a local
            offer? {launched
              ? 'SyntheticLib will independently re-check every proof.'
              : 'Before launch this records local metadata only and uploads nothing.'}
          </p>
          <button type="button" className="btn-ghost" ref={confirmButtonRef} onClick={confirmOffer}>
            Confirm offer
          </button>
          <button type="button" className="btn-ghost" onClick={cancelConfirmation}>
            Cancel
          </button>
        </div>
      )}
    </section>
  );
}
