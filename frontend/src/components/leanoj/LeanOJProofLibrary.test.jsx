import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LeanOJProofLibrary from './LeanOJProofLibrary';

vi.mock('../../utils/downloadHelpers', () => ({
  downloadTextFile: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
});

test('selected proof id expands and hydrates matching LeanOJ proof history card', async () => {
  const api = {
    getProofLibrary: vi.fn().mockResolvedValue({
      proofs: [
        {
          proof_id: 'leanoj-proof-1',
          session_id: 'leanoj-session-1',
          theorem_name: 'LeanOJ.Helper',
          theorem_statement: 'theorem leanoj_helper : True',
          proof_kind: 'subproof',
          source_title: 'LeanOJ source',
          created_at: '2026-06-12T00:00:00+00:00',
        },
      ],
      sessions: [
        {
          session_id: 'leanoj-session-1',
          user_prompt: 'LeanOJ run',
          updated_at: '2026-06-12T00:00:00+00:00',
        },
      ],
    }),
    getLibraryProof: vi.fn().mockResolvedValue({
      proof_id: 'leanoj-proof-1',
      session_id: 'leanoj-session-1',
      theorem_name: 'LeanOJ.Helper',
      theorem_statement: 'Hydrated LeanOJ proof statement.',
      lean_code: 'theorem leanoj_helper : True := by trivial',
    }),
  };

  render(
    <LeanOJProofLibrary
      api={api}
      selectedProofId="leanoj-proof-1"
      selectedSessionId="leanoj-session-1"
    />
  );

  expect(await screen.findByText('LeanOJ.Helper')).toBeInTheDocument();
  await waitFor(() => {
    expect(api.getLibraryProof).toHaveBeenCalledWith('leanoj-session-1', 'leanoj-proof-1');
  });
  expect(await screen.findByText('Hydrated LeanOJ proof statement.')).toBeInTheDocument();
  expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
});

test('keeps LeanOJ session proofs collapsed until the prompt is expanded', async () => {
  const user = userEvent.setup();
  const api = {
    getProofLibrary: vi.fn().mockResolvedValue({
      proofs: [{
        proof_id: 'leanoj-proof-2',
        session_id: 'leanoj-session-2',
        theorem_name: 'LeanOJ.Collapsed',
        theorem_statement: 'theorem collapsed : True',
        proof_kind: 'final',
      }],
      sessions: [{
        session_id: 'leanoj-session-2',
        user_prompt: 'Solve the LeanOJ prompt',
        updated_at: '2026-06-12T00:00:00+00:00',
      }],
    }),
    getLibraryProof: vi.fn(),
  };

  render(<LeanOJProofLibrary api={api} />);

  const groupButton = await screen.findByRole('button', { name: /Solve the LeanOJ prompt/i });
  expect(groupButton).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByText('LeanOJ.Collapsed')).not.toBeInTheDocument();
  await user.click(groupButton);
  expect(screen.getByText('LeanOJ.Collapsed')).toBeInTheDocument();
});

test('bounds long prompt text in historical group headers', async () => {
  const longPrompt = `Solve ${'a'.repeat(1000)}`;
  const api = {
    getProofLibrary: vi.fn().mockResolvedValue({
      proofs: [{
        proof_id: 'leanoj-proof-3',
        session_id: 'leanoj-session-3',
        theorem_name: 'LeanOJ.LongPrompt',
        theorem_statement: 'theorem long_prompt : True',
        proof_kind: 'final',
      }],
      sessions: [{
        session_id: 'leanoj-session-3',
        user_prompt: longPrompt,
        updated_at: '2026-06-12T00:00:00+00:00',
      }],
    }),
    getLibraryProof: vi.fn(),
  };

  render(<LeanOJProofLibrary api={api} />);

  const groupButton = await screen.findByRole('button', { name: /Solve a+/i });
  expect(groupButton).toHaveTextContent('...');
  expect(groupButton).not.toHaveTextContent(longPrompt);
});

