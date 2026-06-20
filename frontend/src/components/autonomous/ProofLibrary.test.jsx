import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ProofLibrary from './ProofLibrary';
import { autonomousAPI, proofSearchAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  autonomousAPI: {
    getProofLibrary: vi.fn(),
    getSessions: vi.fn(),
    getLibraryProof: vi.fn(),
  },
  proofSearchAPI: {
    getOverview: vi.fn(),
    search: vi.fn(),
    getProof: vi.fn(),
    reindex: vi.fn(),
  },
}));

vi.mock('../../utils/researchRunHistory', () => ({
  buildResearchRunGroups: vi.fn(() => []),
  formatRunPromptPreview: vi.fn((value) => value || 'Unknown run'),
}));

vi.mock('../../utils/downloadHelpers', () => ({
  downloadTextFile: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  autonomousAPI.getProofLibrary.mockResolvedValue({ proofs: [] });
  autonomousAPI.getSessions.mockResolvedValue({ sessions: [] });
  proofSearchAPI.getOverview.mockResolvedValue({
    total_records: 99,
    result_cap: 7,
    corpora: [],
  });
  proofSearchAPI.search.mockResolvedValue({
    results: [
      {
        search_id: 'syntheticlib4:sl4-proof-1',
        corpus: 'syntheticlib4',
        proof_id: 'sl4-proof-1',
        theorem_name: 'synthetic_comm_monoid',
        theorem_statement: 'Every synthetic commutative monoid has the mocked identity property.',
        corpus_scope: 'stable-2026-06-12',
        source_kind: 'snapshot',
        lean_code_hash: 'lean-hash-1',
      },
    ],
    ranking_notes: 'ranked with dependency and novelty filters',
  });
  proofSearchAPI.getProof.mockResolvedValue({
    corpus: 'syntheticlib4',
    proof_id: 'sl4-proof-1',
    theorem_name: 'synthetic_comm_monoid',
    theorem_statement: 'Every synthetic commutative monoid has the mocked identity property.',
    proof_description: 'Hydrated SyntheticLib4 proof record.',
    imports: ['Mathlib.Algebra.Group.Basic'],
    dependency_names: ['Monoid.mul_assoc'],
    lean_code: 'theorem synthetic_comm_monoid : True := by trivial',
    theorem_statement_hash: 'statement-hash-1',
    lean_code_hash: 'lean-hash-1',
  });
  proofSearchAPI.reindex.mockResolvedValue({
    overview: {
      total_records: 101,
      result_cap: 7,
      corpora: [],
    },
  });
});

test('submits unified proof-search checklist and hydrates SyntheticLib4 results', async () => {
  const user = userEvent.setup();
  render(<ProofLibrary />);

  expect(await screen.findByText('Unified Proof Search')).toBeInTheDocument();
  expect(screen.getByText('99 indexed')).toBeInTheDocument();
  expect(screen.getByText('7 result cap')).toBeInTheDocument();

  await user.click(screen.getByLabelText(/Include partial artifacts/i));
  await user.click(screen.getByLabelText(/Include failed attempts/i));
  await user.click(screen.getByRole('button', { name: /Search Proofs/i }));

  await waitFor(() => {
    expect(proofSearchAPI.search).toHaveBeenCalledWith({
      query: '',
      corpora: ['moto', 'manual', 'leanoj', 'syntheticlib4'],
      verified_only: false,
      include_partial: true,
      include_failed: true,
      dependency_names: [],
      novelty_filters: [],
      module_filters: [],
      source_filters: [],
      limit: 7,
      hydrate_lean_code: false,
    });
  });
  expect(screen.getByText('synthetic_comm_monoid')).toBeInTheDocument();
  expect(screen.getByText(/ranked with dependency and novelty filters/i)).toBeInTheDocument();

  await user.click(screen.getByText('synthetic_comm_monoid').closest('button'));

  await waitFor(() => {
    expect(proofSearchAPI.getProof).toHaveBeenCalledWith('syntheticlib4', 'sl4-proof-1', {
      sessionId: null,
    });
  });
  expect(await screen.findByText(/Hydrated SyntheticLib4 proof record/i)).toBeInTheDocument();
  expect(screen.getByText(/theorem synthetic_comm_monoid : True := by trivial/i)).toBeInTheDocument();
});

test('rebuilds the unified proof-search index from the frontend', async () => {
  const user = userEvent.setup();
  render(<ProofLibrary />);

  await screen.findByText('Unified Proof Search');
  await user.click(screen.getByRole('button', { name: /Rebuild Index/i }));

  expect(await screen.findByText('Unified proof-search index rebuilt.')).toBeInTheDocument();
  expect(proofSearchAPI.reindex).toHaveBeenCalledTimes(1);
  expect(screen.getByText('101 indexed')).toBeInTheDocument();
});

test('keeps proof-search errors visible and clears stale results', async () => {
  const user = userEvent.setup();
  proofSearchAPI.search.mockRejectedValueOnce(new Error('HTTP 503: SyntheticLib4 search temporarily unavailable'));
  render(<ProofLibrary />);

  await screen.findByText('Unified Proof Search');
  await user.click(screen.getByRole('button', { name: /Search Proofs/i }));

  expect(await screen.findByText(/SyntheticLib4 search temporarily unavailable/i)).toBeInTheDocument();
  expect(screen.queryByText('synthetic_comm_monoid')).not.toBeInTheDocument();
});
