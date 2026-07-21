import React from 'react';
import { render, screen } from '@testing-library/react';
import SyntheticLib4AccessModal from './SyntheticLib4AccessModal';

function renderModal() {
  return render(
    <SyntheticLib4AccessModal
      isOpen
      onClose={vi.fn()}
    />
  );
}

test('shows SyntheticLib4 coming-soon explainer copy', () => {
  renderModal();

  expect(screen.getByRole('heading', { name: 'SyntheticLib4' })).toBeInTheDocument();
  expect(screen.getByText("The Inventor's Dictionary")).toBeInTheDocument();
  expect(screen.getByText(/Coming soon/i)).toBeInTheDocument();
  expect(screen.getByText(/Stay tuned for the imminent SyntheticLib release/i)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /Follow us on X for up-to-date information/i })).toHaveAttribute(
    'href',
    'https://x.com/IntrafereLLC'
  );
  expect(screen.getByText(/contribution-based Lean 4 proof ecosystem/i)).toBeInTheDocument();
  expect(screen.getByText(/unused proofs can now provide value back to you/i)).toBeInTheDocument();
});

test('explains contribution access and redistribution limits', () => {
  renderModal();

  expect(screen.getByText(/20 novel proofs/i)).toBeInTheDocument();
  expect(screen.getByText(/one month of access/i)).toBeInTheDocument();
  expect(screen.getByText(/pay a small monthly fee/i)).toBeInTheDocument();
  expect(screen.getByText(/cite, use, and reference individual proofs/i)).toBeInTheDocument();
  expect(screen.getByText(/not be redistributable as a whole/i)).toBeInTheDocument();
});
