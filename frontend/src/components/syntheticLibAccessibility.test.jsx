import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axe from 'axe-core';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { expectNoAxeViolations } from '../test/axe';
import ConnectivityPanel from './ConnectivityPanel';
import SolutionPathModal from './SolutionPathModal';
import SyntheticLib4AccessModal from './SyntheticLib4AccessModal';
import SyntheticLib4DeveloperWorkspace from './SyntheticLib4DeveloperWorkspace';
import SyntheticLibOfferPanel from './autonomous/SyntheticLibOfferPanel';
import { syntheticLib4API, workflowAPI } from '../services/api';

const PROOF_ID = `moto_proof_${'1'.repeat(32)}`;
const SNAPSHOT_ID = `moto_selection_${'a'.repeat(32)}`;

const AUDIT = {
  success: true,
  product_state: 'developer_preview',
  schema_version: 'moto-trade-evidence-installation-projection-v1',
  source_selection_policy_version: 'moto-trade-source-selection-v2',
  projection_fingerprint: 'f'.repeat(64),
  evidence_complete: true,
  eligible_to_offer: false,
  reason_codes: ['frozen_syntheticlib_contract_unpublished'],
  stores: [
    {
      store_key: 'store_key_manual',
      store_role: 'active_manual',
      stable_store_id: 'store_fixture',
      proof_set_revision: 4,
      projection_fingerprint: 'a'.repeat(64),
      record_count: 2,
      evidence_complete_count: 1,
      source_selected_count: 1,
      evidence_complete: true,
      reason_codes: [],
    },
  ],
  diagnostics: [
    {
      record_key: 'record_one',
      store_key: 'store_key_manual',
      client_proof_id: PROOF_ID,
      source_category: 'active_manual',
      source_selected: true,
      evidence_complete: true,
      eligible_to_offer: false,
      evidence_class: 'legacy_port',
      reason_codes: ['novelty_trade_mapping_unavailable'],
    },
    {
      record_key: 'record_two',
      store_key: 'store_key_manual',
      client_proof_id: '',
      source_category: 'unresolved',
      source_selected: false,
      evidence_complete: false,
      eligible_to_offer: false,
      evidence_class: 'none',
      reason_codes: ['client_proof_id_missing'],
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
  ledger_generation: 2,
  snapshot_counts: { selected: 1, invalidated: 0, cancelled: 0 },
  batch_counts: { prepared: 0, cancelled: 0, invalidated: 0 },
  account_binding_present: false,
  contract_version_present: false,
  archived_ledger_count: 1,
  upload_enabled: false,
};

const ENTRIES = {
  ...LEDGER,
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
  batches: [],
};

describe('SyntheticLib UI accessibility (axe)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('the axe helper detects the nested-control pattern it guards against', async () => {
    const { container } = render(
      <div role="button" tabIndex={0}>
        <input type="checkbox" aria-label="Enable thing" />
        Thing
      </div>,
    );
    const results = await axe.run(container, { runOnly: ['nested-interactive'] });
    expect(results.violations.map((violation) => violation.id)).toContain('nested-interactive');
  });

  test('developer workspace with audit and ledger data', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    vi.spyOn(syntheticLib4API, 'getTradeLedgerStatus').mockResolvedValue(LEDGER);
    vi.spyOn(syntheticLib4API, 'listTradeLedgerEntries').mockResolvedValue(ENTRIES);
    const { container } = render(
      <SyntheticLib4DeveloperWorkspace isOpen onClose={vi.fn()} productState="developer_preview" />,
    );
    await user.click(screen.getByRole('button', { name: /run local audit/i }));
    await screen.findByRole('table', { name: /proof occurrence diagnostics/i });
    await user.click(screen.getByRole('button', { name: /check ledger/i }));
    await screen.findByRole('table', { name: /selection snapshots/i });

    await expectNoAxeViolations(container);
  });

  test('offer panel with eligibility results and a prepare preview', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(AUDIT);
    vi.spyOn(syntheticLib4API, 'migrateLegacyProofs').mockResolvedValue({
      success: true,
      migration_version: 'moto-legacy-port-migration-v1',
      dry_run: true,
      eligible_count: 3,
      migrated_count: 0,
      already_qualified_count: 0,
      skipped: {},
      reason_codes: [],
      stores: [],
      upload_enabled: false,
      server_side_claim: false,
    });
    const { container } = render(<SyntheticLibOfferPanel visible />);
    await user.click(screen.getByRole('button', { name: /check existing proofs/i }));
    await screen.findByRole('button', { name: /prepare 3 existing proofs/i });
    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));
    await screen.findByRole('table', { name: /proof offer eligibility/i });

    await expectNoAxeViolations(container);
  });

  test('Coming soon access modal', async () => {
    const { container } = render(<SyntheticLib4AccessModal isOpen onClose={vi.fn()} />);
    await expectNoAxeViolations(container);
  });

  test('connectivity panel in standard and developer presentations', async () => {
    const status = {
      inference: { openrouter_oauth: { status: 'ACTIVE' }, lm_studio: { status: 'inactive' } },
      skills: {
        wolfram_alpha: { status: 'inactive', enabled: false },
        agent_conversation_memory: { status: 'ACTIVE', enabled: true },
      },
    };
    const standard = render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled={false}
        connectivityStatus={status}
        capabilities={{ lmStudioEnabled: true, syntheticLibProductState: 'coming_soon' }}
        onModeChange={vi.fn()}
      />,
    );
    await expectNoAxeViolations(standard.container);
    standard.unmount();

    const developer = render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled
        connectivityStatus={status}
        capabilities={{ lmStudioEnabled: true, syntheticLibProductState: 'developer_preview' }}
        onModeChange={vi.fn()}
      />,
    );
    await expectNoAxeViolations(developer.container);
  });

  test('solution path modal', async () => {
    vi.spyOn(workflowAPI, 'getSolutionPath').mockResolvedValue(null);
    render(
      <SolutionPathModal
        snapshot={{
          run_id: 'run_1',
          lifecycle_generation: 1,
          active: true,
          revision: 1,
          goal: 'Prove the target.',
          steps: [{ step_id: 's1', title: 'First step', status: 'pending', detail: '' }],
          repairs: [],
        }}
        onClose={vi.fn()}
        onSnapshotChange={vi.fn()}
        onOpenSettings={vi.fn()}
      />,
    );
    await expectNoAxeViolations(document.body);
  });
});
