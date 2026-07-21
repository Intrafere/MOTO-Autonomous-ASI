import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ProofLibrary from './ProofLibrary';
import { autonomousAPI, proofSearchAPI } from '../../services/api';
import { buildResearchRunGroups } from '../../utils/researchRunHistory';

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
  buildResearchRunGroups.mockReturnValue([]);
  Element.prototype.scrollIntoView = vi.fn();
  autonomousAPI.getProofLibrary.mockResolvedValue({ proofs: [], counts: {} });
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
  expect(screen.queryByLabelText(/Include partial artifacts/i)).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/Include failed attempts/i)).not.toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: /Search Proofs/i }));

  await waitFor(() => {
    expect(proofSearchAPI.search).toHaveBeenCalledWith({
      query: '',
      corpora: ['moto', 'manual', 'leanoj', 'syntheticlib4'],
      verified_only: true,
      include_partial: false,
      include_failed: false,
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
      searchId: 'syntheticlib4:sl4-proof-1',
      runId: null,
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

test('renders proof library category tabs and requests selected category', async () => {
  const user = userEvent.setup();
  render(<ProofLibrary />);

  expect(await screen.findByRole('button', { name: /^Novel Proofs$/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^Duplicate Novel Proofs$/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^Not Novel Proofs$/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^All Proofs$/i })).toBeInTheDocument();
  expect(autonomousAPI.getProofLibrary).toHaveBeenCalledWith('novel', 'autonomous');

  await user.click(screen.getByRole('button', { name: /^Duplicate Novel Proofs$/i }));

  await waitFor(() => {
    expect(autonomousAPI.getProofLibrary).toHaveBeenLastCalledWith('duplicate_novel', 'autonomous');
  });
});

test('renders duplicate novel proof badge', async () => {
  const user = userEvent.setup();
  autonomousAPI.getProofLibrary.mockResolvedValue({
    counts: { duplicate_novel: 1 },
    proofs: [
      {
        proof_id: 'duplicate-novel-proof-1',
        session_id: 'session-a',
        theorem_name: 'DuplicateNovel.Helper',
        theorem_statement: 'theorem duplicate_novel_helper : True',
        source_title: 'Duplicate source',
        novelty_tier: 'duplicate_novel',
        novel: true,
        created_at: '2026-06-12T00:00:00+00:00',
      },
    ],
  });
  buildResearchRunGroups.mockReturnValue([
    {
      sessionId: 'session-a',
      userPrompt: 'Duplicate proof run',
      createdAt: '2026-06-12T00:00:00+00:00',
    },
  ]);

  render(<ProofLibrary />);

  const groupButton = await screen.findByRole('button', { name: /Duplicate proof run/i });
  expect(groupButton).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByText('DuplicateNovel.Helper')).not.toBeInTheDocument();
  await user.click(groupButton);
  expect(await screen.findByText('DuplicateNovel.Helper')).toBeInTheDocument();
  expect(screen.getByText('Duplicate Novel')).toBeInTheDocument();
});

test('selected proof id switches filters, expands, and hydrates matching proof history card', async () => {
  autonomousAPI.getProofLibrary.mockResolvedValue({
    counts: { not_novel: 1 },
    proofs: [
      {
        proof_id: 'known-proof-1',
        session_id: 'session-a',
        theorem_name: 'Known.Helper',
        theorem_statement: 'theorem known_helper : True',
        source_title: 'Known source',
        novelty_tier: 'not_novel',
        novel: false,
        created_at: '2026-06-12T00:00:00+00:00',
      },
    ],
  });
  autonomousAPI.getSessions.mockResolvedValue({
    sessions: [
      {
        session_id: 'session-a',
        user_prompt: 'Known proof run',
        created_at: '2026-06-12T00:00:00+00:00',
      },
    ],
  });
  autonomousAPI.getLibraryProof.mockResolvedValue({
    proof_id: 'known-proof-1',
    session_id: 'session-a',
    theorem_name: 'Known.Helper',
    theorem_statement: 'theorem known_helper : True',
    novelty_reasoning: 'Known but useful.',
    lean_code: 'theorem known_helper : True := by trivial',
  });
  proofSearchAPI.getProof.mockRejectedValueOnce(new Error('Legacy record is not indexed'));
  buildResearchRunGroups.mockReturnValue([
    {
      sessionId: 'session-a',
      userPrompt: 'Known proof run',
      createdAt: '2026-06-12T00:00:00+00:00',
    },
  ]);

  render(<ProofLibrary selectedProofId="known-proof-1" selectedSessionId="session-a" />);

  expect(await screen.findByText('Known.Helper')).toBeInTheDocument();
  await waitFor(() => {
    expect(proofSearchAPI.getProof).toHaveBeenCalledWith('autonomous', 'known-proof-1', {
      searchId: null,
      runId: null,
      sessionId: 'session-a',
    });
    expect(autonomousAPI.getLibraryProof).toHaveBeenCalledWith('session-a', 'known-proof-1', 'autonomous');
  });
  expect(await screen.findByText('Known but useful.')).toBeInTheDocument();
  expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
});
