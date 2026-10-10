import React, { useEffect, useRef, useState } from 'react';
import { syntheticLib4API } from '../services/api';
import useExplicitRequest from '../hooks/useExplicitRequest';
import useModalDialogFocus from '../hooks/useModalDialogFocus';
import { previewErrorMessage } from '../utils/syntheticLibPresentation';
import './settings-common.css';

const STORE_ROLE_LABELS = {
  active_manual: 'Active manual',
  archived_manual: 'Archived manual',
  autonomous_session: 'Autonomous session',
  autonomous_legacy: 'Autonomous legacy',
};

const MAX_SELECTION_SIZE = 500;

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' };
const cellStyle = { textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid #2c2c48', verticalAlign: 'top' };
const actionRowStyle = { display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' };

function countReasons(audit) {
  const counts = new Map();
  for (const item of audit.diagnostics || []) {
    for (const reason of item.reason_codes || []) {
      counts.set(reason, (counts.get(reason) || 0) + 1);
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

// Only evidence-complete occurrences that pass the frozen local source policy can
// be bound into a selection snapshot; the backend re-checks this independently.
function isSelectable(item) {
  return Boolean(item.evidence_complete && item.source_selected && item.client_proof_id);
}

const EVIDENCE_CLASS_LABELS = {
  full: 'Full receipts',
  legacy_port: 'Legacy port',
  none: 'None',
};

const LEDGER_REASON_LABELS = {
  ledger_corrupt: 'The ledger file is unreadable or fails schema validation.',
  ledger_schema_unsupported: 'The ledger was written by an unsupported schema version.',
  ledger_not_regular: 'The ledger path is not a regular file.',
  ledger_installation_binding_mismatch:
    'The ledger belongs to a different installation or data-root binding.',
  ledger_generic_mode_unavailable: 'The ledger is unavailable in hosted generic mode.',
};

const describeAuditError = (error) => previewErrorMessage(
  error,
  'run this read-only audit',
  'The local proof-trade evidence audit failed.',
);
const describeLedgerError = (error) => previewErrorMessage(
  error,
  'check the local trade ledger',
  'The local proof-trade ledger check failed.',
);
const describeLedgerActionError = (error) => previewErrorMessage(
  error,
  'change the local trade ledger',
  'The local proof-trade ledger action failed.',
);
const describeReadinessError = (error) => previewErrorMessage(
  error,
  'check proof-trade authorization readiness',
  'The proof-trade authorization readiness check failed.',
);
const fetchReadiness = () => syntheticLib4API.getTradeAuthorizationReadiness();
const fetchAudit = () => syntheticLib4API.getTradeEvidenceDiagnostics();

async function fetchLedger({ isCurrent } = {}) {
  const status = await syntheticLib4API.getTradeLedgerStatus();
  if (!status.available || !status.exists) {
    return { status, entries: null };
  }
  // Closing the workspace stops the follow-up read instead of only ignoring it.
  if (isCurrent && !isCurrent()) return { status, entries: null };
  const entries = await syntheticLib4API.listTradeLedgerEntries();
  return { status, entries };
}

function reasonSuffix(reasonCodes) {
  return reasonCodes?.length ? `: ${reasonCodes.join(', ')}` : '';
}

// Each action resolves to the message announced after the ledger is refreshed.
const LEDGER_ACTIONS = {
  createSnapshot: async (clientProofIds) => {
    const { snapshot } = await syntheticLib4API.createTradeSelectionSnapshot(clientProofIds);
    return `Selection snapshot ${snapshot.snapshot_id} holds ${snapshot.entry_count} proof`
      + `${snapshot.entry_count === 1 ? '' : 's'} (local metadata only; nothing uploads).`;
  },
  reconcile: async (snapshotId) => {
    const { snapshot } = await syntheticLib4API.reconcileTradeSelectionSnapshot(snapshotId);
    return snapshot.reconciliation_outcome === 'ok'
      ? `Snapshot ${snapshotId} still matches the canonical proof stores.`
      : `Reconciliation failed; snapshot ${snapshotId} is now invalidated${reasonSuffix(snapshot.reason_codes)}.`;
  },
  prepareBatch: async (snapshotId) => {
    const result = await syntheticLib4API.prepareTradeBatch(snapshotId);
    if (result.outcome === 'blocked') {
      return `Batch blocked${reasonSuffix(result.reason_codes)}. Nothing is queued for upload.`;
    }
    if (result.outcome === 'invalidated') {
      return `Snapshot invalidated during reconciliation${reasonSuffix(result.reason_codes)}.`;
    }
    return `Local batch ${result.client_batch_id} recorded. Uploads remain disabled.`;
  },
  cancelSnapshot: async (snapshotId) => {
    await syntheticLib4API.cancelTradeSelectionSnapshot(snapshotId);
    return `Snapshot ${snapshotId} and its prepared local batches were cancelled.`;
  },
  cancelBatch: async (clientBatchId) => {
    await syntheticLib4API.cancelTradeBatch(clientBatchId);
    return `Local batch ${clientBatchId} was cancelled.`;
  },
  archiveReset: async () => {
    const result = await syntheticLib4API.archiveResetTradeLedger();
    return result.archived
      ? 'The local ledger was archived and reset. No server-side deletion is claimed.'
      : 'No local ledger existed; nothing was archived.';
  },
  deleteArchives: async () => {
    const result = await syntheticLib4API.deleteArchivedTradeLedgers();
    const skipped = result.skipped_count
      ? ` ${result.skipped_count} non-regular entr${result.skipped_count === 1 ? 'y was' : 'ies were'} left untouched.`
      : '';
    return `Deleted ${result.deleted_count} archived local ledger`
      + `${result.deleted_count === 1 ? '' : 's'} from this machine.${skipped}`
      + ' No server-side deletion is claimed.';
  },
};

const CONFIRMATION_LABELS = {
  snapshot: 'Confirm selection snapshot',
  reset: 'Confirm ledger archive and reset',
  deleteArchives: 'Confirm archived ledger deletion',
};

const runLedgerAction = (action, ...args) => LEDGER_ACTIONS[action](...args);

function ledgerSummary(ledger) {
  if (!ledger.available) {
    const reason = LEDGER_REASON_LABELS[ledger.reason_code] || ledger.reason_code;
    return `Ledger unavailable (fails closed): ${reason}`;
  }
  if (!ledger.exists) {
    return 'No ledger yet. Nothing is selected, queued, or uploaded.';
  }
  return `Ledger available (generation ${ledger.ledger_generation}). Uploads: disabled.`;
}

export default function SyntheticLib4DeveloperWorkspace({
  isOpen,
  onClose,
  productState,
  launched = false,
}) {
  const [auditState, runAuditRequest] = useExplicitRequest(isOpen, fetchAudit, describeAuditError);
  const [ledgerState, runLedgerCheck] = useExplicitRequest(
    isOpen,
    fetchLedger,
    describeLedgerError,
  );
  const [actionState, runActionRequest] = useExplicitRequest(
    isOpen,
    runLedgerAction,
    describeLedgerActionError,
  );
  const [readinessState, runReadinessCheck] = useExplicitRequest(
    isOpen,
    fetchReadiness,
    describeReadinessError,
  );
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [pendingConfirmation, setPendingConfirmation] = useState(null);
  // Last completed ledger read, kept mounted during a refresh so focused row
  // controls survive; cleared on error and on close.
  const [ledgerView, setLedgerView] = useState(null);
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const checkLedgerRef = useRef(null);
  const confirmRef = useRef(null);
  const snapshotTriggerRef = useRef(null);
  const resetTriggerRef = useRef(null);
  const deleteTriggerRef = useRef(null);
  const triggerRefs = {
    snapshot: snapshotTriggerRef,
    reset: resetTriggerRef,
    deleteArchives: deleteTriggerRef,
  };

  // The trigger may only remount after the confirmation closes, so focus it
  // after that render rather than immediately.
  const returnFocusRef = useRef(null);
  const cancelConfirmation = () => {
    returnFocusRef.current = triggerRefs[pendingConfirmation] || null;
    setPendingConfirmation(null);
  };

  useModalDialogFocus({
    active: Boolean(isOpen),
    dialogRef,
    initialFocusRef: closeRef,
    onEscape: () => (pendingConfirmation ? cancelConfirmation() : onClose()),
  });

  useEffect(() => {
    if (!isOpen) {
      setSelectedIds(new Set());
      setPendingConfirmation(null);
      setLedgerView(null);
    }
  }, [isOpen]);

  useEffect(() => {
    if (ledgerState.status === 'complete') setLedgerView(ledgerState.data);
    if (ledgerState.status === 'error') setLedgerView(null);
  }, [ledgerState]);

  useEffect(() => {
    if (pendingConfirmation) {
      confirmRef.current?.focus();
    } else if (returnFocusRef.current) {
      returnFocusRef.current.current?.focus();
      returnFocusRef.current = null;
    }
  }, [pendingConfirmation]);

  if (!isOpen) return null;

  const { status, data: audit, error } = auditState;
  const ledger = ledgerView?.status;
  const ledgerEntries = ledgerView?.entries;
  const ledgerRefreshing = ledgerState.status === 'running';
  const diagnostics = audit?.diagnostics || [];
  const reasonCounts = audit ? countReasons(audit) : [];
  const selectedCount = diagnostics.filter((item) => item.source_selected).length;
  const completeCount = diagnostics.filter((item) => item.evidence_complete).length;
  const actionRunning = actionState.status === 'running';
  const selectionTooLarge = selectedIds.size > MAX_SELECTION_SIZE;
  const archivedCount = ledger?.archived_ledger_count || 0;

  // Busy controls stay focusable (aria-disabled) so keyboard focus is never
  // dropped while a request they started is running.
  const guarded = (busy, handler) => () => {
    if (!busy) handler();
  };

  const runAudit = () => {
    setSelectedIds(new Set());
    setPendingConfirmation(null);
    return runAuditRequest();
  };

  const toggleSelected = (clientProofId) => {
    setPendingConfirmation(null);
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(clientProofId)) next.delete(clientProofId);
      else next.add(clientProofId);
      return next;
    });
  };

  // Refresh only after a fenced success, so closing mid-action never repopulates.
  // Focus lands on the stable Check ledger control rather than a removed button.
  const performAction = async (action, ...args) => {
    setPendingConfirmation(null);
    checkLedgerRef.current?.focus();
    const succeeded = await runActionRequest(action, ...args);
    if (succeeded) {
      if (action === 'createSnapshot') setSelectedIds(new Set());
      await runLedgerCheck();
    }
  };

  const renderConfirmation = (kind, message, confirmLabel, onConfirm, keepLabel) => (
    pendingConfirmation === kind && (
      <div role="group" aria-label={CONFIRMATION_LABELS[kind]} style={{ marginTop: '8px' }}>
        <p>{message}</p>
        <div style={actionRowStyle}>
          <button type="button" className="btn-ghost" ref={confirmRef} onClick={onConfirm}>
            {confirmLabel}
          </button>
          <button type="button" className="btn-ghost" onClick={cancelConfirmation}>
            {keepLabel}
          </button>
        </div>
      </div>
    )
  );

  const rowHeader = { ...cellStyle, fontWeight: 'normal' };

  return (
    <div
      className="inline-modal-overlay"
      style={{ zIndex: 10000 }}
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        className="inline-modal-content inline-modal-content--fluid"
        role="dialog"
        aria-modal="true"
        aria-labelledby="syntheticlib-developer-title"
        style={{ width: '860px', maxWidth: '94vw', maxHeight: '90vh', overflowY: 'auto', background: '#1a1a2e' }}
      >
        <div className="settings-header-row">
          <div>
            <h2 id="syntheticlib-developer-title">
              SyntheticLib4 {launched ? 'Developer Tools' : 'Private Development Workspace'}
            </h2>
            {!launched && (
              <span className="connectivity-status connectivity-status--pending">
                Not launched
              </span>
            )}
          </div>
          <button
            type="button"
            ref={closeRef}
            onClick={onClose}
            className="modal-close-btn"
            aria-label="Close SyntheticLib4 workspace"
          >
            ×
          </button>
        </div>

        <div className="settings-section">
          <h3>Lifecycle</h3>
          <p>
            Product state: <strong>{productState}</strong>
          </p>
          <p>
            This workspace does not inspect credentials, contact SyntheticLib, poll
            account state, or transfer proof data. Ledger controls change only local
            metadata, and only when you click them.
          </p>
        </div>

        <div className="settings-section">
          <h3>Proof status boundary</h3>
          <p>
            A proof being stored, compiled, ranked, or searchable in MOTO does not mean
            SyntheticLib has accepted, credited, owned, or declared it trade-eligible.
          </p>
          <p>
            These developer tools also require separate local development authority on the
            backend, even after launch; this browser-only developer view cannot grant it.
          </p>
        </div>

        <div className="settings-section">
          <h3 id="syntheticlib-readiness-title">Authorization readiness (read-only)</h3>
          <p>
            Desktop proof-trade authorization will use a public native client with PKCE
            S256 and the exact proof-trade scopes. No login, browser return, token exchange,
            or credential storage exists until SyntheticLib publishes its registered client
            profile and frozen contract. Checking readiness reads nothing secret and creates
            nothing.
          </p>
          <div style={actionRowStyle}>
            <button
              type="button"
              className="btn-ghost"
              onClick={guarded(readinessState.status === 'running', () => runReadinessCheck())}
              aria-disabled={readinessState.status === 'running'}
              aria-describedby="syntheticlib-readiness-title"
            >
              {readinessState.status === 'running' ? 'Checking readiness…' : 'Check readiness'}
            </button>
          </div>
          <div role="status" aria-live="polite">
            {readinessState.status === 'error' && <p>{readinessState.error}</p>}
            {readinessState.data && (
              <p>
                Authorization: disabled. PKCE {readinessState.data.pkce_method}; scopes{' '}
                {readinessState.data.required_scopes.join(', ')}; keyring{' '}
                {readinessState.data.keyring_available ? 'available' : 'unavailable'}; installation{' '}
                {readinessState.data.installation_bound ? 'bound' : 'not bound'}; client profile{' '}
                {readinessState.data.profile_published ? 'present' : 'unpublished'}.
              </p>
            )}
            {readinessState.data?.reason_codes?.length > 0 && (
              <ul aria-label="Authorization readiness reason codes">
                {readinessState.data.reason_codes.map((reason) => (
                  <li key={reason}><code className="wrap-anywhere">{reason}</code></li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="settings-section">
          <h3 id="syntheticlib-audit-title">Local evidence audit (read-only)</h3>
          <p>
            Audits every canonical local proof store against MOTO&apos;s frozen
            source-selection policy and recorded proof evidence. It reads metadata only and
            never uses proof-search indexes. Every proof stays ineligible until SyntheticLib
            publishes its rank mapping and frozen trade contract.
          </p>
          <button
            type="button"
            className="btn-ghost"
            onClick={guarded(status === 'running', runAudit)}
            aria-disabled={status === 'running'}
            aria-describedby="syntheticlib-audit-title"
          >
            {status === 'running' ? 'Running local audit…' : 'Run local audit'}
          </button>
          <div role="status" aria-live="polite" style={{ marginTop: '8px' }}>
            {status === 'running' && 'Auditing canonical proof stores…'}
            {status === 'error' && <span>{error}</span>}
            {status === 'complete' && audit && (
              <span>
                Audit complete: {diagnostics.length} occurrence
                {diagnostics.length === 1 ? '' : 's'} across {audit.stores.length} store
                {audit.stores.length === 1 ? '' : 's'}. Eligible to offer: none.
              </span>
            )}
          </div>

          {status === 'complete' && audit && (
            <div style={{ marginTop: '12px' }}>
              <p>
                Policy <code>{audit.source_selection_policy_version}</code>. Evidence
                complete: {completeCount}. Passing the local source policy: {selectedCount}.
              </p>

              <h4>Stores</h4>
              {audit.stores.length === 0 ? (
                <p>No canonical proof stores exist yet.</p>
              ) : (
                <div className="table-scroll">
                  <table style={tableStyle} aria-label="Canonical proof stores">
                    <thead>
                      <tr>
                        <th style={cellStyle} scope="col">Store</th>
                        <th style={cellStyle} scope="col">Revision</th>
                        <th style={cellStyle} scope="col">Records</th>
                        <th style={cellStyle} scope="col">Evidence complete</th>
                        <th style={cellStyle} scope="col">Policy pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {audit.stores.map((store) => (
                        <tr key={store.store_key}>
                          <th style={rowHeader} scope="row">
                            {STORE_ROLE_LABELS[store.store_role] || store.store_role}
                            {' '}
                            <code className="wrap-anywhere">{store.store_key}</code>
                          </th>
                          <td style={cellStyle}>{store.proof_set_revision ?? 'unknown'}</td>
                          <td style={cellStyle}>{store.record_count}</td>
                          <td style={cellStyle}>{store.evidence_complete_count}</td>
                          <td style={cellStyle}>{store.source_selected_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {reasonCounts.length > 0 && (
                <>
                  <h4>Reason codes</h4>
                  <div className="table-scroll">
                    <table style={tableStyle} aria-label="Reason code counts">
                      <thead>
                        <tr>
                          <th style={cellStyle} scope="col">Reason</th>
                          <th style={cellStyle} scope="col">Occurrences</th>
                        </tr>
                      </thead>
                      <tbody>
                        {reasonCounts.map(([reason, count]) => (
                          <tr key={reason}>
                            <th style={rowHeader} scope="row">
                              <code className="wrap-anywhere">{reason}</code>
                            </th>
                            <td style={cellStyle}>{count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}

              {diagnostics.length > 0 && (
                <>
                  <h4>Occurrences</h4>
                  <p>
                    Evidence-complete occurrences that pass the local source policy can be
                    selected into a local ledger snapshot below.
                  </p>
                  <div className="table-scroll" style={{ maxHeight: '320px', overflowY: 'auto' }}>
                    <table style={tableStyle} aria-label="Proof occurrence diagnostics">
                      <thead>
                        <tr>
                          <th style={cellStyle} scope="col">Select</th>
                          <th style={cellStyle} scope="col">Record</th>
                          <th style={cellStyle} scope="col">Category</th>
                          <th style={cellStyle} scope="col">Evidence</th>
                          <th style={cellStyle} scope="col">Evidence class</th>
                          <th style={cellStyle} scope="col">Policy</th>
                          <th style={cellStyle} scope="col">Reasons</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diagnostics.map((item) => (
                          <tr key={`${item.store_key}:${item.record_key}`}>
                            <td style={cellStyle}>
                              {isSelectable(item) ? (
                                <input
                                  type="checkbox"
                                  aria-label={`Select record ${item.record_key}`}
                                  checked={selectedIds.has(item.client_proof_id)}
                                  onChange={() => toggleSelected(item.client_proof_id)}
                                  disabled={actionRunning}
                                />
                              ) : (
                                <span className="visually-hidden">Not selectable</span>
                              )}
                            </td>
                            <th style={rowHeader} scope="row">
                              <code className="wrap-anywhere">{item.record_key}</code>
                            </th>
                            <td style={cellStyle}>{item.source_category}</td>
                            <td style={cellStyle}>{item.evidence_complete ? 'Complete' : 'Incomplete'}</td>
                            <td style={cellStyle}>
                              {EVIDENCE_CLASS_LABELS[item.evidence_class] || EVIDENCE_CLASS_LABELS.none}
                            </td>
                            <td style={cellStyle}>{item.source_selected ? 'Included' : 'Excluded'}</td>
                            <td style={cellStyle} className="wrap-anywhere">
                              {(item.reason_codes || []).join(', ')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        <div className="settings-section">
          <h3 id="syntheticlib-ledger-title">Local trade ledger (developer controls)</h3>
          <p>
            The crash-safe local ledger holds explicit proof selections as metadata only
            (no proof text, prompts, or paths). Checking it never creates it. Prepare batch
            reports blocked until SyntheticLib publishes its contract; nothing is queued for
            upload and nothing leaves this machine.
          </p>
          <div style={actionRowStyle}>
            <button
              type="button"
              className="btn-ghost"
              ref={checkLedgerRef}
              onClick={guarded(ledgerRefreshing || actionRunning, () => runLedgerCheck())}
              aria-disabled={ledgerRefreshing || actionRunning}
              aria-describedby="syntheticlib-ledger-title"
            >
              {ledgerRefreshing ? 'Checking ledger…' : 'Check ledger'}
            </button>
            <button
              type="button"
              className="btn-ghost"
              ref={snapshotTriggerRef}
              onClick={() => setPendingConfirmation('snapshot')}
              disabled={selectedIds.size === 0 || selectionTooLarge || actionRunning}
            >
              Select proofs into snapshot ({selectedIds.size})
            </button>
          </div>

          {renderConfirmation(
            'snapshot',
            `Bind ${selectedIds.size} selected proof${selectedIds.size === 1 ? '' : 's'} into a `
              + 'local selection snapshot? This records metadata only and uploads nothing. The '
              + 'first ledger write binds this data root to the installation identity.',
            'Confirm selection snapshot',
            () => performAction('createSnapshot', [...selectedIds]),
            'Keep editing',
          )}

          <div role="status" aria-live="polite" style={{ marginTop: '8px' }}>
            {selectionTooLarge && (
              <p>A snapshot can hold at most {MAX_SELECTION_SIZE} proofs.</p>
            )}
            {ledgerRefreshing && 'Reading the local trade ledger…'}
            {ledgerState.status === 'error' && <span>{ledgerState.error}</span>}
            {!ledgerRefreshing && ledger && <span>{ledgerSummary(ledger)}</span>}
          </div>
          <div role="status" aria-live="polite" aria-label="Ledger action result">
            {actionRunning && 'Updating the local trade ledger…'}
            {actionState.status === 'error' && <span>{actionState.error}</span>}
            {actionState.status === 'complete' && <span>{actionState.data}</span>}
          </div>

          <div aria-busy={ledgerRefreshing}>
            {ledger?.available && ledger.exists && (
              <div className="table-scroll">
                <table style={{ ...tableStyle, marginTop: '12px' }} aria-label="Trade ledger state counts">
                  <thead>
                    <tr>
                      <th style={cellStyle} scope="col">Item</th>
                      <th style={cellStyle} scope="col">State</th>
                      <th style={cellStyle} scope="col">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ledger.snapshot_counts || {}).map(([state, count]) => (
                      <tr key={`snapshot-${state}`}>
                        <th style={rowHeader} scope="row">Selection snapshots</th>
                        <td style={cellStyle}>{state}</td>
                        <td style={cellStyle}>{count}</td>
                      </tr>
                    ))}
                    {Object.entries(ledger.batch_counts || {}).map(([state, count]) => (
                      <tr key={`batch-${state}`}>
                        <th style={rowHeader} scope="row">Local batches</th>
                        <td style={cellStyle}>{state}</td>
                        <td style={cellStyle}>{count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {ledgerEntries?.snapshots?.length > 0 && (
              <>
                <h4>Selection snapshots</h4>
                <div className="table-scroll">
                  <table style={tableStyle} aria-label="Selection snapshots">
                    <thead>
                      <tr>
                        <th style={cellStyle} scope="col">Snapshot</th>
                        <th style={cellStyle} scope="col">State</th>
                        <th style={cellStyle} scope="col">Proofs</th>
                        <th style={cellStyle} scope="col">Reconciliation</th>
                        <th style={cellStyle} scope="col">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ledgerEntries.snapshots.map((snapshot) => (
                        <tr key={snapshot.snapshot_id}>
                          <th style={rowHeader} scope="row">
                            <code className="wrap-anywhere">{snapshot.snapshot_id}</code>
                          </th>
                          <td style={cellStyle}>{snapshot.state}</td>
                          <td style={cellStyle}>{snapshot.entry_count}</td>
                          <td style={cellStyle} className="wrap-anywhere">
                            {snapshot.reconciliation_outcome || 'not yet'}
                            {reasonSuffix(snapshot.reason_codes)}
                          </td>
                          <td style={cellStyle}>
                            {snapshot.state === 'selected' && (
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                <button
                                  type="button"
                                  className="btn-ghost"
                                  aria-disabled={actionRunning}
                                  aria-label={`Reconcile snapshot ${snapshot.snapshot_id}`}
                                  onClick={guarded(actionRunning, () => performAction('reconcile', snapshot.snapshot_id))}
                                >
                                  Reconcile
                                </button>
                                <button
                                  type="button"
                                  className="btn-ghost"
                                  aria-disabled={actionRunning}
                                  aria-label={`Prepare batch for snapshot ${snapshot.snapshot_id}`}
                                  onClick={guarded(actionRunning, () => performAction('prepareBatch', snapshot.snapshot_id))}
                                >
                                  Prepare batch
                                </button>
                                <button
                                  type="button"
                                  className="btn-ghost"
                                  aria-disabled={actionRunning}
                                  aria-label={`Cancel snapshot ${snapshot.snapshot_id}`}
                                  onClick={guarded(actionRunning, () => performAction('cancelSnapshot', snapshot.snapshot_id))}
                                >
                                  Cancel
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {ledgerEntries?.batches?.length > 0 && (
              <>
                <h4>Local batches</h4>
                <div className="table-scroll">
                  <table style={tableStyle} aria-label="Local batches">
                    <thead>
                      <tr>
                        <th style={cellStyle} scope="col">Batch</th>
                        <th style={cellStyle} scope="col">Snapshot</th>
                        <th style={cellStyle} scope="col">State</th>
                        <th style={cellStyle} scope="col">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ledgerEntries.batches.map((batch) => (
                        <tr key={batch.client_batch_id}>
                          <th style={rowHeader} scope="row">
                            <code className="wrap-anywhere">{batch.client_batch_id}</code>
                          </th>
                          <td style={cellStyle}><code className="wrap-anywhere">{batch.snapshot_id}</code></td>
                          <td style={cellStyle}>{batch.state}</td>
                          <td style={cellStyle}>
                            {batch.state === 'prepared' && (
                              <button
                                type="button"
                                className="btn-ghost"
                                aria-disabled={actionRunning}
                                aria-label={`Cancel batch ${batch.client_batch_id}`}
                                onClick={guarded(actionRunning, () => performAction('cancelBatch', batch.client_batch_id))}
                              >
                                Cancel
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {ledger?.exists && (
              <div style={{ marginTop: '12px' }}>
                {pendingConfirmation !== 'reset' && (
                  <button
                    type="button"
                    className="btn-ghost"
                    ref={resetTriggerRef}
                    aria-disabled={actionRunning}
                    onClick={guarded(actionRunning, () => setPendingConfirmation('reset'))}
                  >
                    Archive and reset local ledger
                  </button>
                )}
                {renderConfirmation(
                  'reset',
                  'Move the local ledger into its archive folder and start empty? This clears '
                    + 'local pending selections only. It does not claim any server-side deletion.',
                  'Confirm archive and reset',
                  () => performAction('archiveReset'),
                  'Keep ledger',
                )}
              </div>
            )}

            {archivedCount > 0 && (
              <div style={{ marginTop: '12px' }}>
                {pendingConfirmation !== 'deleteArchives' && (
                  <button
                    type="button"
                    className="btn-ghost"
                    ref={deleteTriggerRef}
                    aria-disabled={actionRunning}
                    onClick={guarded(actionRunning, () => setPendingConfirmation('deleteArchives'))}
                  >
                    Delete archived local ledgers ({archivedCount})
                  </button>
                )}
                {renderConfirmation(
                  'deleteArchives',
                  `Permanently delete ${archivedCount} archived local ledger`
                    + `${archivedCount === 1 ? '' : 's'} from this machine? The current ledger is `
                    + 'not touched, and no server-side deletion is claimed.',
                  'Confirm deletion',
                  () => performAction('deleteArchives'),
                  'Keep archives',
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
