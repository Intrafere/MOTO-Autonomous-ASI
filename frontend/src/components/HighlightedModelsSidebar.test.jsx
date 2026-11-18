import React from 'react';
import { render, screen } from '@testing-library/react';
import HighlightedModelsSidebar from './HighlightedModelsSidebar';

test('silver pick is Claude Haiku 5.5, Supercharged, with the PS tag and a Supercharge explanation', () => {
  render(<HighlightedModelsSidebar />);
  expect(screen.queryByText('Grok 4.7')).not.toBeInTheDocument();

  const tile = screen.getByText('Claude Haiku 5.5').closest('.model-item');
  expect(tile).toHaveClass('model-item--silver', 'model-item--supercharged');
  expect(tile.querySelector('.ranking-badge--silver')).not.toBeNull();
  expect(tile.querySelector('.ps-badge')).toHaveTextContent('PS');
  expect(tile).toHaveTextContent('Low cost knowledge, generous token API allotment w/ subscription');

  const label = screen.getByLabelText('Supercharged: learn about the Supercharge option');
  expect(label).toHaveAttribute('tabindex', '0');
  expect(label).toHaveTextContent('SUPERCHARGED');
  expect(screen.getByRole('tooltip', { hidden: true })).toHaveTextContent(/4 full answers in parallel/);
});
