import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';
import ProofCheckModeModal from './ProofCheckModeModal';

describe('ProofCheckModeModal', () => {
  test('offers one-confirm mode selection', async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(<ProofCheckModeModal sourceTitle="Paper one" onClose={() => {}} onConfirm={onConfirm} />);

    expect(screen.getByRole('dialog', { name: 'Choose proof check mode' })).toBeTruthy();
    expect(screen.getByText('Run a Lean proof search for this source:')).toBeTruthy();
    expect(screen.getByText('Paper one')).toBeTruthy();
    expect(screen.getByText('Run One Round')).toBeTruthy();
    expect(screen.getByText('Run Rounds on Loop With Pruning')).toBeTruthy();
    expect(screen.getByText(/A round with no candidates immediately proceeds to the next round/i)).toBeTruthy();
    expect(screen.getByText(/backend restarts/i)).toBeTruthy();
    expect(screen.getByText(/until you press Stop/i)).toBeTruthy();
    await user.click(screen.getByRole('radio', { name: /Run Rounds on Loop With Pruning/ }));
    await user.click(screen.getByRole('button', { name: 'Start proof check' }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onConfirm).toHaveBeenCalledWith('loop_with_pruning');
  });

  test('shows a bounded source prompt preview with the full prompt available as a title', () => {
    const sourceTitle = `Solve this exact objective ${'with detailed supporting context '.repeat(12)}`.trim();
    render(<ProofCheckModeModal sourceTitle={sourceTitle} onClose={() => {}} onConfirm={() => {}} />);

    const sourcePreview = document.querySelector('.proof-check-mode-modal__source-title');
    expect(sourcePreview.textContent.length).toBeLessThanOrEqual(180);
    expect(sourcePreview.textContent.endsWith('…')).toBe(true);
    expect(sourcePreview).toHaveAttribute('title', sourceTitle);
  });

  test('closes on Escape and restores focus', async () => {
    const onClose = vi.fn();
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const { unmount } = render(
      <ProofCheckModeModal onClose={onClose} onConfirm={() => {}} />,
    );

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    unmount();
    await waitFor(() => expect(document.activeElement).toBe(opener));
  });

  test('does not dismiss while a confirmation is in flight', () => {
    const onClose = vi.fn();
    render(<ProofCheckModeModal busy onClose={onClose} onConfirm={() => {}} />);

    fireEvent.mouseDown(document.querySelector('.proof-check-mode-modal__backdrop'));
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('status').textContent).toContain('Starting proof run');
  });
});
