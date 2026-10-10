import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import SyntheticLibOfferPanel, { PLANNED_QUALIFYING_PROOF_THRESHOLD } from './SyntheticLibOfferPanel';
import { syntheticLib4API } from '../../services/api';

const PROOF_ONE = `moto_proof_${'1'.repeat(32)}`;
const PROOF_TWO = `moto_proof_${'2'.repeat(32)}`;
const SNAPSHOT_ID = `moto_selection_${'a'.repeat(32)}`;

function audit(eligibleOne = false) {
  return {
    success: true,
    product_state: 'developer_preview',
    eligible_to_offer: false,
    reason_codes: ['frozen_syntheticlib_contract_unpublished', 'novelty_trade_mapping_unavailable'],
    stores: [],
    diagnostics: [
      {
        record_key: 'record_one',
        client_proof_id: PROOF_ONE,
        source_selected: true,
        evidence_complete: true,
        eligible_to_offer: eligibleOne,
        reason_codes: eligibleOne ? [] : ['novelty_trade_mapping_unavailable'],
      },
      {
        record_key: 'record_two',
        client_proof_id: PROOF_TWO,
        source_selected: true,
        evidence_complete: true,
        eligible_to_offer: false,
        reason_codes: ['novelty_trade_mapping_unavailable'],
      },
    ],
  };
}

describe('SyntheticLibOfferPanel', () => {
  let fetchSpy;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('renders nothing and requests nothing when the integration is hidden', () => {
    const { container } = render(<SyntheticLibOfferPanel visible={false} />);
    expect(container).toBeEmptyDOMElement();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('shows required disclosures without auto-requesting or promising a multiplier', () => {
    render(<SyntheticLibOfferPanel visible />);

    expect(screen.getByRole('heading', { name: /offer proofs for SyntheticLib access/i })).toBeInTheDocument();
    expect(screen.getByText('Private development')).toBeInTheDocument();
    expect(screen.getByText(/not a donation/i)).toBeInTheDocument();
    expect(screen.getByText(/reruns compile, uniqueness, policy, and novelty checks/i)).toBeInTheDocument();
    expect(
      screen.getByText(new RegExp(`${PLANNED_QUALIFYING_PROOF_THRESHOLD} qualifying proofs`)),
    ).toHaveTextContent(/uploaded files/);
    expect(screen.getByText(/only on SyntheticLib/i)).toBeInTheDocument();
    expect(screen.getByText(/pending, rejected, or in cooldown/i)).toBeInTheDocument();
    expect(
      screen.getByText(/minimized provenance plus authorization\s+and status metadata/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/multiplier/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/balance/i)).not.toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('offers no checkbox while nothing is canonically eligible', async () => {
    const user = userEvent.setup();
    const diagnostics = vi
      .spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics')
      .mockResolvedValue(audit(false));
    render(<SyntheticLibOfferPanel visible />);

    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));

    expect(diagnostics).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/No proofs are eligible to offer yet/i)).toBeInTheDocument();
    const reasons = screen.getByRole('list', { name: /why no proofs are eligible yet/i });
    expect(reasons).toHaveTextContent('novelty_trade_mapping_unavailable');
    expect(reasons).toHaveTextContent(/has not published how MOTO novelty ranks map/i);
    expect(screen.getByRole('table', { name: /proof offer eligibility/i })).toHaveTextContent(
      'Not eligible',
    );
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /offer selected proofs/i })).not.toBeInTheDocument();
  });

  test('requires explicit selection and confirmation; cancel sends nothing', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(audit(true));
    const snapshot = vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot').mockResolvedValue({
      snapshot: { snapshot_id: SNAPSHOT_ID, entry_count: 1 },
    });
    const prepare = vi.spyOn(syntheticLib4API, 'prepareTradeBatch').mockResolvedValue({
      outcome: 'blocked',
      reason_codes: ['batch_blocked_ineligible'],
    });
    render(<SyntheticLibOfferPanel visible />);

    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));
    const checkboxes = await screen.findAllByRole('checkbox');
    expect(checkboxes).toHaveLength(1);
    expect(screen.getByRole('button', { name: /offer selected proofs \(0\)/i })).toBeDisabled();

    await user.click(screen.getByRole('checkbox', { name: `Offer proof ${PROOF_ONE}` }));
    await user.click(screen.getByRole('button', { name: /offer selected proofs \(1\)/i }));
    expect(screen.getByRole('group', { name: /confirm proof offer/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /confirm offer/i })).toHaveFocus();

    await user.click(screen.getByRole('button', { name: /^cancel$/i }));
    expect(screen.queryByRole('group', { name: /confirm proof offer/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /offer selected proofs \(1\)/i })).toHaveFocus();
    expect(snapshot).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /offer selected proofs \(1\)/i }));
    await user.click(screen.getByRole('button', { name: /confirm offer/i }));

    expect(snapshot).toHaveBeenCalledWith([PROOF_ONE]);
    expect(prepare).toHaveBeenCalledWith(SNAPSHOT_ID);
    expect(await screen.findByText(/batch blocked: batch_blocked_ineligible/i)).toHaveTextContent(
      /Nothing was uploaded/,
    );
    // The offer button is disabled once the selection clears, so focus stays usable.
    expect(screen.getByRole('button', { name: /check offer eligibility/i })).toHaveFocus();
  });

  test('Escape cancels the confirmation and returns focus without sending', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(audit(true));
    const snapshot = vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot');
    render(<SyntheticLibOfferPanel visible />);

    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));
    await user.click(await screen.findByRole('checkbox', { name: `Offer proof ${PROOF_ONE}` }));
    await user.click(screen.getByRole('button', { name: /offer selected proofs \(1\)/i }));
    expect(screen.getByRole('button', { name: /confirm offer/i })).toHaveFocus();

    await user.keyboard('{Escape}');

    expect(screen.queryByRole('group', { name: /confirm proof offer/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /offer selected proofs \(1\)/i })).toHaveFocus();
    expect(snapshot).not.toHaveBeenCalled();
  });

  test('hiding the panel mid-offer never sends the batch request', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockResolvedValue(audit(true));
    let resolveSnapshot;
    vi.spyOn(syntheticLib4API, 'createTradeSelectionSnapshot').mockReturnValue(
      new Promise((resolve) => {
        resolveSnapshot = resolve;
      }),
    );
    const prepare = vi.spyOn(syntheticLib4API, 'prepareTradeBatch');
    const { rerender } = render(<SyntheticLibOfferPanel visible />);

    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));
    await user.click(await screen.findByRole('checkbox', { name: `Offer proof ${PROOF_ONE}` }));
    await user.click(screen.getByRole('button', { name: /offer selected proofs \(1\)/i }));
    await user.click(screen.getByRole('button', { name: /confirm offer/i }));
    expect(await screen.findByText(/Recording the local offer selection/i)).toBeInTheDocument();

    rerender(<SyntheticLibOfferPanel visible={false} />);
    resolveSnapshot({ snapshot: { snapshot_id: SNAPSHOT_ID, entry_count: 1 } });
    await Promise.resolve();
    await Promise.resolve();

    expect(prepare).not.toHaveBeenCalled();
  });

  test('ignores an eligibility response that arrives after the panel is hidden', async () => {
    const user = userEvent.setup();
    let resolveAudit;
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockReturnValue(
      new Promise((resolve) => {
        resolveAudit = resolve;
      }),
    );
    const { rerender } = render(<SyntheticLibOfferPanel visible />);
    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));

    rerender(<SyntheticLibOfferPanel visible={false} />);
    resolveAudit(audit(true));
    await Promise.resolve();
    rerender(<SyntheticLibOfferPanel visible />);

    expect(screen.queryByRole('table', { name: /proof offer eligibility/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  test('drops the private-development label once launched', () => {
    render(<SyntheticLibOfferPanel visible launched />);
    expect(screen.queryByText('Private development')).not.toBeInTheDocument();
    expect(screen.queryByText(/Before launch, eligibility/i)).not.toBeInTheDocument();
  });

  test('explains missing backend preview authority on 403', async () => {
    const user = userEvent.setup();
    vi.spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics').mockRejectedValue(
      Object.assign(new Error('forbidden'), { status: 403 }),
    );
    render(<SyntheticLibOfferPanel visible />);
    await user.click(screen.getByRole('button', { name: /check offer eligibility/i }));
    expect(await screen.findByText(/MOTO_SYNTHETICLIB_DEVELOPER_PREVIEW_AUTHORIZED/)).toBeInTheDocument();
  });

  test('prepares existing proofs only after a dry run and confirmation', async () => {
    const user = userEvent.setup();
    const migrate = vi.spyOn(syntheticLib4API, 'migrateLegacyProofs').mockImplementation(
      async (dryRun) => ({
        success: true,
        migration_version: 'moto-legacy-port-migration-v1',
        dry_run: dryRun,
        eligible_count: 2,
        migrated_count: dryRun ? 0 : 2,
        already_qualified_count: 1,
        skipped: { legacy_port_rank_unknown: 1 },
        reason_codes: [],
        stores: [],
        upload_enabled: false,
        server_side_claim: false,
      }),
    );
    const eligibility = vi
      .spyOn(syntheticLib4API, 'getTradeEvidenceDiagnostics')
      .mockResolvedValue(audit(false));
    render(<SyntheticLibOfferPanel visible />);
    expect(migrate).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /check existing proofs/i }));
    expect(migrate).toHaveBeenCalledWith(true);
    expect(await screen.findByText(/2 existing proofs can be prepared; 1 already prepared/i))
      .toHaveTextContent(/1 no recognizable stored novelty rank/);

    await user.click(screen.getByRole('button', { name: /prepare 2 existing proofs/i }));
    const group = screen.getByRole('group', { name: /confirm preparing existing proofs/i });
    expect(group).toHaveTextContent(/nothing is\s+uploaded/i);
    expect(screen.getByRole('button', { name: /confirm prepare/i })).toHaveFocus();

    await user.keyboard('{Escape}');
    expect(screen.queryByRole('group', { name: /confirm preparing existing proofs/i })).not.toBeInTheDocument();
    expect(migrate).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: /prepare 2 existing proofs/i }));
    await user.click(screen.getByRole('button', { name: /confirm prepare/i }));

    expect(migrate).toHaveBeenLastCalledWith(false);
    expect(eligibility).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: /check existing proofs/i })).toHaveFocus();
  });
});
