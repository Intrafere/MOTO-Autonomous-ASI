import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import CodexOAuthNotificationStack from './CodexOAuthNotificationStack';

describe('CodexOAuthNotificationStack', () => {
  it('renders a Sakana waiting cooldown as provider attention', () => {
    render(
      <CodexOAuthNotificationStack
        notifications={[{
          id: 'sakana-wait',
          provider: 'sakana_fugu',
          provider_label: 'Sakana Fugu',
          role_id: 'proof_formalization',
          reason: 'usage_limit_reached',
          message: 'Sakana Fugu usage limit reached. This role is waiting for the provider cooldown to end.',
        }]}
        onDismiss={vi.fn()}
        onOpenCloudAccess={vi.fn()}
      />,
    );

    expect(screen.getByText('Provider Cooldown')).toBeTruthy();
    expect(screen.getByText('Sakana Fugu')).toBeTruthy();
    expect(screen.getByText(/waiting for the provider cooldown/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Open Provider Settings' })).toBeTruthy();
  });
});
