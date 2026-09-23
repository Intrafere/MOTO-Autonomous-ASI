import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';
import PaperBatchPruneControls, {
  isPaperBatchPruneEligible,
  paperTargetKey,
} from './PaperBatchPruneControls';

describe('PaperBatchPruneControls', () => {
  const targets = [
    { session_id: 'session-a', paper_id: 'paper-1' },
    { session_id: 'session-b', paper_id: 'paper-1' },
  ];

  test('keeps same paper IDs in different sessions distinct', () => {
    expect(paperTargetKey(targets[0])).not.toBe(paperTargetKey(targets[1]));
  });

  test('excludes pruned and in-progress papers', () => {
    expect(isPaperBatchPruneEligible({ ...targets[0], status: 'complete' })).toBe(true);
    expect(isPaperBatchPruneEligible({ ...targets[0], status: 'in_progress' })).toBe(false);
    expect(isPaperBatchPruneEligible({ ...targets[0], status: 'complete', is_pruned: true })).toBe(false);
  });

  test('shows a count-aware all-or-none confirmation', async () => {
    const onPruneSelected = vi.fn().mockResolvedValue(true);
    render(
      <PaperBatchPruneControls
        visibleTargets={targets}
        selectedKeys={new Set(targets.map(paperTargetKey))}
        onSelectAllVisible={vi.fn()}
        onClear={vi.fn()}
        onPruneSelected={onPruneSelected}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Prune selected (2)' }));
    expect(screen.getByText(/Prune 2 selected papers from model context/)).toBeInTheDocument();
    expect(screen.getByText(/all-or-none/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Prune 2' }));
    expect(onPruneSelected).toHaveBeenCalledOnce();
    await waitFor(() => {
      expect(screen.queryByText(/Prune 2 selected papers from model context/)).not.toBeInTheDocument();
    });
  });
});
