import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import SyntheticLib4DeveloperWorkspace from './SyntheticLib4DeveloperWorkspace';
import { syntheticLib4API } from '../services/api';

const PROOF_ONE = `moto_proof_${'1'.repeat(32)}`;
const PROOF_TWO = `moto_proof_${'2'.repeat(32)}`;
const SNAPSHOT_ID = `moto_selection_${'a'.repeat(32)}`;
const BATCH_ID = `moto_batch_${'b'.repeat(32)}`;

const AUDIT = {
  success: true,
  product_state: 'developer_preview',
  schema_version: 'moto-trade-evidence-installation-projection-v1',
  source_selection_policy_version: 'moto-trade-source-selection-v2',
  projection_fingerprint: 'f'.repeat(64),
  evidence_complete: true,
  eligible_to_offer: false,
  reason_codes: ['frozen_syntheticlib_contract_unpublished', 'novelty_trade_mapping_unavailable'],
  stores: [
    {
      store_key: 'store_key_manual',
      store_role: 'active_manual',
      stable_store_id: 'store_fixture',
      proof_set_revision: 4,
      projection_fingerprint: 'a'.repeat(64),
      record_count: 2,
      evidence_complete_count: 2,
      source_selected_count: 1,
      evidence_complete: true,
      reason_codes: [],
    },
  ],
  diagnostics: [
    {
      record_key: 'record_one',
      store_key: 'store_key_manual',
      client_proof_id: PROOF_ONE,
      source_category: 'active_manual',
      source_selected: true,
      evidence_complete: true,
      eligible_to_offer: false,
      reason_codes: ['novelty_trade_mapping_unavailable'],
      evidence_class: 'legacy_port',
    },
    {
      record_key: 'record_two',
      store_key: 'store_key_manual',
      client_proof_id: PROOF_TWO,
      source_category: 'active_manual',
      source_selected: false,
      evidence_complete: true,
      eligible_to_offer: false,
      reason_codes: ['novelty_trade_mapping_unavailable', 'source_local_rank_not_novel'],
    },
  ],
};

const LEDGER = {
  success: true,
  product_state: 'developer_preview',
  schema_version: 'moto-proof-trade-ledger-v1',
  available: true,
  reason_code: '',
  exists: true,
  ledger_generation: 3,
  snapshot_counts: { selected: 1, invalidated: 1, cancelled: 0 },
  batch_counts: { prepared: 0, cancelled: 0, invalidated: 0 },
  account_binding_present: false,
  contract_version_present: false,
  upload_enabled: false,
};

const ENTRIES = {
  success: true,
  product_state: 'developer_preview',
  schema_version: 'moto-proof-trade-ledger-v1',
  available: true,
  reason_code: '',
  exists: true,
  ledger_generation: 3,
  snapshots: [
    {
      snapshot_id: SNAPSHOT_ID,
      state: 'selected',
      entry_count: 1,
      selection_digest: 'd'.repeat(64),
      reconciliation_outcome: null,
      reason_codes: [],
      created_at: '2026-10-08T00:00:00+00:00',
      updated_at: '2026-10-08T00:00:00+00:00',
      reconciled_at: null,
      revision_advanced: false,
    },
  ],
  batches: [
    {
      client_batch_id: BATCH_ID,
      snapshot_id: SNAPSHOT_ID,
      state: 'prepared',
      created_at: '2026-10-08T00:00:00+00:00',
      updated_at: '2026-10-08T00:00:00+00:00',
    },
  ],
  upload_enabled: false,
};

function mockLedger() {
  return {
    status: vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockResolvedValue(LEDGER),
    entries: vi.spyOn(syntheticLib4API, 'listTradeLedgerEntries').mockResolvedValue(ENTRIES),
  };
}

describe('SyntheticLib4DeveloperWorkspace', () => {
  let fetchSpy;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('makes no request until an action is explicitly clicked', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <SyntheticLib4DeveloperWorkspace
        isOpen
        onClose={onClose}
        productState="developer_preview"
      />,
    );

    expect(screen.getByRole('heading', { name: /private development workspace/i })).toBeInTheDocument();
    expect(screen.getByText(/does not inspect credentials/i)).toBeInTheDocument();
    expect(screen.getByText(/does not mean SyntheticLib has accepted/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run local audit/i })).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /close SyntheticLib4 workspace/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('runs one audit on request and renders metadata-only results', async () => {
    const user = userEvent.setup();
    const audit = vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    await user.click(screen.getByRole('button', { name: /run local audit/i }));

    expect(audit).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Eligible to offer: none/i)).toBeInTheDocument();
    expect(screen.getByRole('table', { name: /canonical proof stores/i })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: /reason code counts/i })).toHaveTextContent(
      'source_local_rank_not_novel',
    );
    expect(screen.getByRole('table', { name: /proof occurrence diagnostics/i })).toHaveTextContent(
      'Excluded',
    );
  });

  test('checks authorization readiness only on click and clears it on close', async () => {
    const user = userEvent.setup();
    const readiness = vi.spyOn(syntheticLib4API, 'getTradeAuthorizationReadiness').mockResolvedValue({
      success: true,
      product_state: 'developer_preview',
      readiness_version: 'moto-proof-trade-authorization-readiness-v1',
      pkce_method: 'S256',
      redirect_policy: 'profile_defined',
      required_scopes: ['proof_trade:create', 'proof_trade:read_own'],
      profile_published: false,
      keyring_available: true,
      installation_bound: false,
      authorization_enabled: false,
      upload_enabled: false,
      reason_codes: ['frozen_syntheticlib_contract_unpublished', 'native_client_profile_unpublished'],
    });
    const { rerender } = render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    expect(readiness).not.toHaveBeenCalled();
    expect(screen.getByText(/No login, browser return, token exchange/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /check readiness/i }));

    expect(readiness).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Authorization: disabled\. PKCE S256/)).toBeInTheDocument();
    expect(
      screen.getByRole('list', { name: /authorization readiness reason codes/i }),
    ).toHaveTextContent('native_client_profile_unpublished');

    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen={false} onClose={vi.fn()} productState="developer_preview" />,
    );
    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    expect(screen.queryByText(/Authorization: disabled/)).not.toBeInTheDocument();
    expect(readiness).toHaveBeenCalledTimes(1);
  });

  test('explains missing backend preview authority on 403', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockRejectedValue(
      Object.assign(new Error('forbidden'), { status: 403 }),
    );
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="coming_soon" />,
    );

    await user.click(screen.getByRole('button', { name: /run local audit/i }));

    expect(
      await screen.findByText(/MOTO_SYNTHETICLIB_DEVELOPER_PREVIEW_AUTHORIZED/),
    ).toHaveTextContent(/run this read-only audit/);
  });

  test('ignores an audit response that arrives after the workspace closes', async () => {
    const user = userEvent.setup();
    let resolveAudit;
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockReturnValue(
      new Promise((resolve) => {
        resolveAudit = resolve;
      }),
    );
    const { rerender } = render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /run local audit/i }));

    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen={false} onClose={vi.fn()} productState="developer_preview" />,
    );
    resolveAudit(AUDIT);
    await Promise.resolve();
    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    expect(screen.queryByText(/Eligible to offer: none/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run local audit/i })).toBeEnabled();
  });

  test('checks the ledger only on request and shows counts without uploads', async () => {
    const user = userEvent.setup();
    const { status: ledger, entries } = mockLedger();
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    expect(ledger).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    expect(ledger).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/generation 3\). Uploads: disabled/i)).toBeInTheDocument();
    expect(screen.getByRole('table', { name: /trade ledger state counts/i })).toHaveTextContent(
      'invalidated',
    );
    expect(entries).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('table', { name: /selection snapshots/i })).toHaveTextContent(SNAPSHOT_ID);
    expect(screen.getByRole('table', { name: /local batches/i })).toHaveTextContent('prepared');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('reports an unreadable ledger as failing closed', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockResolvedValue({
      ...LEDGER,
      available: false,
      reason_code: 'ledger_corrupt',
      ledger_generation: null,
    });
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    expect(await screen.findByText(/fails closed/i)).toHaveTextContent(/unreadable/);
    expect(screen.queryByRole('table', { name: /trade ledger state counts/i })).not.toBeInTheDocument();
  });

  test('explains missing preview authority for the ledger check', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockRejectedValue(
      Object.assign(new Error('forbidden'), { status: 403 }),
    );
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="coming_soon" />,
    );

    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    expect(await screen.findByText(/check the local trade ledger/)).toHaveTextContent(
      /MOTO_SYNTHETICLIB_DEVELOPER_PREVIEW_AUTHORIZED/,
    );
  });

  test('ignores a ledger response that arrives after the workspace closes', async () => {
    const user = userEvent.setup();
    let resolveLedger;
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockReturnValue(
      new Promise((resolve) => {
        resolveLedger = resolve;
      }),
    );
    const { rerender } = render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen={false} onClose={vi.fn()} productState="developer_preview" />,
    );
    resolveLedger(LEDGER);
    await Promise.resolve();
    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    expect(screen.queryByText(/Uploads: disabled/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /check ledger/i })).toBeEnabled();
  });

  test('offers selection only for evidence-complete policy-passing proofs', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    const create = vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot');
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    await user.click(screen.getByRole('button', { name: /run local audit/i }));
    await screen.findByText(/Eligible to offer: none/i);

    expect(screen.getByRole('checkbox', { name: /select record record_one/i })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: /select record record_two/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /select proofs into snapshot \(0\)/i })).toBeDisabled();
    expect(create).not.toHaveBeenCalled();
  });

  test('creates a snapshot only after confirmation and refreshes the ledger', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    const { status: ledger } = mockLedger();
    const create = vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot').mockResolvedValue({
      success: true,
      snapshot: { ...ENTRIES.snapshots[0] },
      upload_enabled: false,
    });
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /run local audit/i }));
    await user.click(await screen.findByRole('checkbox', { name: /select record record_one/i }));
    await user.click(screen.getByRole('button', { name: /select proofs into snapshot \(1\)/i }));

    expect(create).not.toHaveBeenCalled();
    expect(screen.getByRole('group', { name: /confirm selection snapshot/i })).toHaveTextContent(
      /uploads nothing/i,
    );

    await user.click(screen.getByRole('button', { name: /confirm selection snapshot/i }));

    expect(create).toHaveBeenCalledTimes(1);
    expect(create).toHaveBeenCalledWith([PROOF_ONE]);
    expect(
      await screen.findByText(/holds 1 proof \(local metadata only; nothing uploads\)/i),
    ).toBeInTheDocument();
    expect(ledger).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: /select proofs into snapshot \(0\)/i })).toBeDisabled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('prepare batch reports blocked and nothing queued', async () => {
    const user = userEvent.setup();
    mockLedger();
    const prepare = vi.spyOn(syntheticLib4API, 'prepareTradeBatch').mockResolvedValue({
      success: true,
      outcome: 'blocked',
      client_batch_id: null,
      reason_codes: ['batch_blocked_ineligible'],
      upload_enabled: false,
    });
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await user.click(
      await screen.findByRole('button', { name: `Prepare batch for snapshot ${SNAPSHOT_ID}` }),
    );

    expect(prepare).toHaveBeenCalledWith(SNAPSHOT_ID);
    expect(
      await screen.findByText(/Batch blocked: batch_blocked_ineligible\. Nothing is queued for upload/),
    ).toBeInTheDocument();
  });

  test('cancels snapshots and batches by explicit click', async () => {
    const user = userEvent.setup();
    mockLedger();
    const cancelSnapshot = vi
      .spyOn(syntheticLib4API, 'cancelTradeSelectionSnapshot')
      .mockResolvedValue({});
    const cancelBatch = vi.spyOn(syntheticLib4API, 'cancelTradeBatch').mockResolvedValue({});
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    await user.click(await screen.findByRole('button', { name: `Cancel batch ${BATCH_ID}` }));
    expect(cancelBatch).toHaveBeenCalledWith(BATCH_ID);
    expect(await screen.findByText(`Local batch ${BATCH_ID} was cancelled.`)).toBeInTheDocument();

    await user.click(await screen.findByRole('button', { name: `Cancel snapshot ${SNAPSHOT_ID}` }));
    expect(cancelSnapshot).toHaveBeenCalledWith(SNAPSHOT_ID);
  });

  test('archive and reset requires confirmation and claims no server deletion', async () => {
    const user = userEvent.setup();
    mockLedger();
    const reset = vi.spyOn(syntheticLib4API, 'archiveResetTradeLedger').mockResolvedValue({
      success: true,
      archived: true,
      server_side_deletion_claimed: false,
    });
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await user.click(await screen.findByRole('button', { name: /archive and reset local ledger/i }));

    expect(reset).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /keep ledger/i }));
    expect(reset).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /archive and reset local ledger/i }));
    await user.click(screen.getByRole('button', { name: /confirm archive and reset/i }));

    expect(reset).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/No server-side deletion is claimed/)).toBeInTheDocument();
  });

  test('renders ledger refusal reason codes and preview-authority guidance', async () => {
    const user = userEvent.setup();
    mockLedger();
    const reconcile = vi
      .spyOn(syntheticLib4API, 'reconcileTradeSelectionSnapshot')
      .mockRejectedValueOnce(
        Object.assign(
          new Error('HTTP 409 Conflict: Proof-trade ledger request rejected: snapshot_not_active'),
          { status: 409 },
        ),
      );
    render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await user.click(
      await screen.findByRole('button', { name: `Reconcile snapshot ${SNAPSHOT_ID}` }),
    );
    expect(await screen.findByText(/snapshot_not_active/)).toBeInTheDocument();

    reconcile.mockRejectedValueOnce(Object.assign(new Error('forbidden'), { status: 403 }));
    await user.click(screen.getByRole('button', { name: `Reconcile snapshot ${SNAPSHOT_ID}` }));
    expect(await screen.findByText(/change the local trade ledger/)).toHaveTextContent(
      /MOTO_SYNTHETICLIB_DEVELOPER_PREVIEW_AUTHORIZED/,
    );
  });

  test('closing mid-action ignores the response, clears selection, and skips the refresh', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    const { status: ledger } = mockLedger();
    let resolveCreate;
    vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot').mockReturnValue(
      new Promise((resolve) => {
        resolveCreate = resolve;
      }),
    );
    const { rerender } = render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /run local audit/i }));
    await user.click(await screen.findByRole('checkbox', { name: /select record record_one/i }));
    await user.click(screen.getByRole('button', { name: /select proofs into snapshot \(1\)/i }));
    await user.click(screen.getByRole('button', { name: /confirm selection snapshot/i }));

    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen={false} onClose={vi.fn()} productState="developer_preview" />,
    );
    resolveCreate({ success: true, snapshot: { ...ENTRIES.snapshots[0] }, upload_enabled: false });
    await Promise.resolve();
    await Promise.resolve();
    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );

    expect(ledger).not.toHaveBeenCalled();
    expect(screen.queryByText(/holds 1 proof/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /select proofs into snapshot \(0\)/i })).toBeDisabled();
  });

  test('renders nothing while closed', () => {
    const { container } = render(
      <SyntheticLib4DeveloperWorkspace
        isOpen={false}
        onClose={vi.fn()}
        productState="coming_soon"
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

describe('SyntheticLib4DeveloperWorkspace accessibility and request safety', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function renderWorkspace(props = {}) {
    return render(
      <SyntheticLib4DeveloperWorkspace
        isOpen
        onClose={vi.fn()}
        productState="developer_preview"
        {...props}
      />,
    );
  }

  test('is a modal dialog that takes focus, wraps Tab, closes on Escape, and restores focus', async () => {
    const user = userEvent.setup();
    const trigger = document.createElement('button');
    trigger.textContent = 'Open workspace';
    document.body.appendChild(trigger);
    trigger.focus();
    const onClose = vi.fn();
    const { rerender } = renderWorkspace({ onClose });

    const dialog = screen.getByRole('dialog', { name: /SyntheticLib4 Private Development Workspace/i });
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    const close = screen.getByRole('button', { name: /close SyntheticLib4 workspace/i });
    expect(close).toHaveFocus();

    await user.keyboard('{Shift>}{Tab}{/Shift}');
    expect(dialog).toContainElement(document.activeElement);
    await user.keyboard('{Tab}');
    expect(close).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);

    rerender(
      <SyntheticLib4DeveloperWorkspace isOpen={false} onClose={onClose} productState="developer_preview" />,
    );
    expect(trigger).toHaveFocus();
    trigger.remove();
  });

  test('confirmation takes focus; Escape and Keep return it without closing the dialog', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    mockLedger();
    const onClose = vi.fn();
    renderWorkspace({ onClose });

    await user.click(screen.getByRole('button', { name: /run local audit/i }));
    await user.click(await screen.findByRole('checkbox', { name: /select record record_one/i }));
    const trigger = screen.getByRole('button', { name: /select proofs into snapshot \(1\)/i });
    await user.click(trigger);
    expect(screen.getByRole('button', { name: /confirm selection snapshot/i })).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.queryByRole('group', { name: /confirm selection snapshot/i })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();

    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    const reset = await screen.findByRole('button', { name: /archive and reset local ledger/i });
    await user.click(reset);
    expect(screen.getByRole('button', { name: /confirm archive and reset/i })).toHaveFocus();
    await user.click(screen.getByRole('button', { name: /keep ledger/i }));
    expect(screen.getByRole('button', { name: /archive and reset local ledger/i })).toHaveFocus();
  });

  test('keeps focus on Check ledger after an action and keeps tables mounted while refreshing', async () => {
    const user = userEvent.setup();
    const { status } = mockLedger();
    vi.spyOn(syntheticLib4API, 'reconcileTradeSelectionSnapshot').mockResolvedValue({
      success: true,
      snapshot: { ...ENTRIES.snapshots[0], reconciliation_outcome: 'ok' },
      upload_enabled: false,
    });
    renderWorkspace();
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await user.click(await screen.findByRole('button', { name: `Reconcile snapshot ${SNAPSHOT_ID}` }));

    let resolveRefresh;
    status.mockReturnValueOnce(new Promise((resolve) => {
      resolveRefresh = resolve;
    }));
    expect(await screen.findByText(/still matches the canonical proof stores/i)).toBeInTheDocument();
    expect(screen.getByRole('table', { name: /selection snapshots/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /check ledger/i })).toHaveFocus();
    resolveRefresh(LEDGER);
    expect(await screen.findByText(/Ledger available \(generation 3\)/)).toBeInTheDocument();
  });

  test('closing during a ledger check never sends the follow-up entries request', async () => {
    const user = userEvent.setup();
    let resolveStatus;
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockReturnValue(
      new Promise((resolve) => {
        resolveStatus = resolve;
      }),
    );
    const entries = vi.spyOn(syntheticLib4API, 'listTradeLedgerEntries').mockResolvedValue(ENTRIES);
    const { unmount } = renderWorkspace();
    await user.click(screen.getByRole('button', { name: /check ledger/i }));

    unmount();
    resolveStatus(LEDGER);
    await Promise.resolve();
    await Promise.resolve();

    expect(entries).not.toHaveBeenCalled();
  });

  test('announces readiness reasons and selection limits in live regions', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeAuthorizationReadiness').mockResolvedValue({
      success: true,
      product_state: 'developer_preview',
      readiness_version: 'moto-proof-trade-authorization-readiness-v1',
      pkce_method: 'S256',
      redirect_policy: 'profile_defined',
      required_scopes: ['proof_trade:create', 'proof_trade:read_own'],
      profile_published: false,
      keyring_available: true,
      installation_bound: false,
      authorization_enabled: false,
      upload_enabled: false,
      reason_codes: ['native_client_profile_unpublished'],
    });
    renderWorkspace();
    await user.click(screen.getByRole('button', { name: /check readiness/i }));

    const reasons = await screen.findByRole('list', { name: /authorization readiness reason codes/i });
    expect(reasons.closest('[role="status"]')).not.toBeNull();
  });

  test('uses row headers and announces non-selectable rows', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    renderWorkspace();
    await user.click(screen.getByRole('button', { name: /run local audit/i }));

    const table = await screen.findByRole('table', { name: /proof occurrence diagnostics/i });
    expect(within(table).getByRole('rowheader', { name: 'record_one' })).toBeInTheDocument();
    expect(within(table).getByText('Not selectable')).toBeInTheDocument();
  });

  test('deletes archived ledgers only after confirmation and claims no server deletion', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockResolvedValue({
      ...LEDGER,
      exists: false,
      archived_ledger_count: 2,
    });
    const remove = vi.spyOn(syntheticLib4API, 'deleteArchivedTradeLedgers').mockResolvedValue({
      success: true,
      deleted_count: 2,
      skipped_count: 0,
      server_side_deletion_claimed: false,
    });
    renderWorkspace();
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await user.click(await screen.findByRole('button', { name: /delete archived local ledgers \(2\)/i }));
    expect(remove).not.toHaveBeenCalled();
    expect(screen.getByRole('group', { name: /confirm archived ledger deletion/i })).toHaveTextContent(
      /no server-side deletion is claimed/i,
    );

    await user.click(screen.getByRole('button', { name: /^confirm deletion$/i }));

    expect(remove).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Deleted 2 archived local ledgers/i)).toHaveTextContent(
      /No server-side deletion is claimed/,
    );
  });

  test('shows the evidence class of each occurrence', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    renderWorkspace();
    await user.click(screen.getByRole('button', { name: /run local audit/i }));

    const table = await screen.findByRole('table', { name: /proof occurrence diagnostics/i });
    expect(within(table).getByRole('columnheader', { name: 'Evidence class' })).toBeInTheDocument();
    expect(within(table).getByText('Legacy port')).toBeInTheDocument();
    expect(within(table).getByText('None')).toBeInTheDocument();
  });
});
