import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MasterProofDetails from './MasterProofDetails';
import * as downloadHelpers from '../../utils/downloadHelpers';

const PROOF = {
  proof_id: 'proof_040',
  theorem_name: 'master',
  proof_kind: 'master_proof',
  attempt_count: 9,
  master_proof: {
    total_attempts: 9,
    steps: [
      { name: 'step_one', signature: 'lemma step_one : True', depth: 0, failed_attempts: 2 },
      { name: 'step_one_aux', signature: 'lemma step_one_aux : True', depth: 1, parent: 'step_one', failed_attempts: 0 },
    ],
  },
};
const BUNDLE = {
  proof_id: 'proof_040',
  solved_template: 'lemma step_one : True := by\n  sorry  -- SOLVED',
  steps: [
    { name: 'step_one', lean_code: 'lemma step_one : True := by\n  trivial' },
    { name: 'step_one_aux', lean_code: 'lemma step_one_aux : True := trivial' },
  ],
  lean_code: 'full file',
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('MasterProofDetails', () => {
  it('renders nothing for an individual proof', () => {
    const { container } = render(
      <MasterProofDetails proof={{ proof_id: 'p', proof_kind: 'individual' }} loadBundle={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('lists the solved steps without loading the full proof', () => {
    const loadBundle = vi.fn();
    render(<MasterProofDetails proof={PROOF} loadBundle={loadBundle} />);

    expect(screen.getByText(/Built from 2 solved steps in 9 Lean 4 submissions/)).toBeTruthy();
    const steps = within(screen.getByRole('list', { name: 'Solved steps of this master proof' }))
      .getAllByRole('listitem');
    expect(steps).toHaveLength(2);
    expect(within(steps[0]).getByText(/Solved after 2 failed attempts/)).toBeTruthy();
    expect(within(steps[1]).getByText(/Solved first try · expanded from step_one/)).toBeTruthy();
    expect(loadBundle).not.toHaveBeenCalled();
  });

  it('loads the bundle once to show the solved template and a step proof', async () => {
    const loadBundle = vi.fn().mockResolvedValue(BUNDLE);
    render(<MasterProofDetails proof={PROOF} loadBundle={loadBundle} />);

    const templateToggle = screen.getByRole('button', { name: 'View Solved Template' });
    expect(templateToggle.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(templateToggle);
    await waitFor(() => expect(screen.getByText(/sorry -- SOLVED/)).toBeTruthy());
    expect(screen.getByRole('button', { name: 'Hide Solved Template' }).getAttribute('aria-expanded')).toBe('true');

    fireEvent.click(screen.getAllByRole('button', { name: 'View proof' })[0]);
    await waitFor(() => expect(screen.getByText(/trivial$/m)).toBeTruthy());
    expect(loadBundle).toHaveBeenCalledTimes(1);
    expect(loadBundle).toHaveBeenCalledWith(PROOF);
  });

  it('downloads the whole master proof as one bundle and the template as Lean', async () => {
    const download = vi.spyOn(downloadHelpers, 'downloadTextFile').mockImplementation(() => {});
    const loadBundle = vi.fn().mockResolvedValue(BUNDLE);
    render(<MasterProofDetails proof={PROOF} loadBundle={loadBundle} />);

    fireEvent.click(screen.getByRole('button', { name: 'Download Master Proof Bundle (JSON)' }));
    await waitFor(() => expect(download).toHaveBeenCalledTimes(1));
    expect(download.mock.calls[0][1]).toBe('proof_040_master_proof.json');
    expect(JSON.parse(download.mock.calls[0][0]).lean_code).toBe('full file');

    fireEvent.click(screen.getByRole('button', { name: 'Download Solved Template (.lean)' }));
    await waitFor(() => expect(download).toHaveBeenCalledTimes(2));
    expect(download.mock.calls[1][1]).toBe('master_solved_template.lean');
  });

  it('reports a load failure instead of showing an empty proof', async () => {
    const loadBundle = vi.fn().mockRejectedValue(new Error('Proof not found'));
    render(<MasterProofDetails proof={PROOF} loadBundle={loadBundle} />);

    fireEvent.click(screen.getByRole('button', { name: 'View Solved Template' }));
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('Proof not found'));
  });
});
