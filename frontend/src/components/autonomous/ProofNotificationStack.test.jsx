import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import ProofNotificationStack from './ProofNotificationStack';

describe('ProofNotificationStack', () => {
  it('states that a standalone rank applies to the accepted Lean statement', () => {
    render(
      <ProofNotificationStack
        notifications={[{
          id: 'standalone-1',
          proof_id: 'proof_001',
          theorem_statement: 'A standalone theorem',
          novelty_tier: 'mathematical_discovery',
          novelty_scope: 'standalone',
        }]}
        onDismiss={vi.fn()}
        onClickNotification={vi.fn()}
        panelCollapsed
      />
    );

    expect(screen.getByText('Ranking applies to this accepted Lean statement alone.')).toBeInTheDocument();
  });

  it('shows the finite group scope and combined claim without breaking click or dismiss', () => {
    const onDismiss = vi.fn();
    const onClickNotification = vi.fn();
    render(
      <ProofNotificationStack
        notifications={[{
          id: 'group-1',
          proof_id: 'proof_002',
          theorem_statement: 'The second case',
          novelty_tier: 'major_mathematical_discovery',
          novelty_scope: 'group',
          novelty_group_claim: 'Together the declarations cover both cases.',
          novelty_group_proof_ids: ['proof_001', 'proof_002'],
          proof_id: 'proof_002',
        }]}
        onDismiss={onDismiss}
        onClickNotification={onClickNotification}
        panelCollapsed={false}
      />
    );

    expect(screen.getByText('Ranking applies to a finite verified group of 2 proofs.')).toBeInTheDocument();
    expect(screen.getByText(/Combined claim: Together the declarations cover both cases/)).toBeInTheDocument();

    fireEvent.click(screen.getByText('The second case'));
    expect(onClickNotification).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss proof notification' }));
    expect(onDismiss).toHaveBeenCalledWith('group-1');
  });
});
