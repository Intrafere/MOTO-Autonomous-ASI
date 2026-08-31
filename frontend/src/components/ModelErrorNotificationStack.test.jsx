import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import ModelErrorNotificationStack from './ModelErrorNotificationStack';

describe('ModelErrorNotificationStack', () => {
  it('renders terminal guidance and dismisses by stable notification key', () => {
    const onDismiss = vi.fn();
    render(
      <ModelErrorNotificationStack
        embedded
        notifications={[{
          notification_key: 'proof-truncation:run-a',
          title: 'Proof model output repeatedly truncated',
          message: 'Autonomous research stopped.',
          terminal_guidance: 'Increase the output allowance and restart.',
        }]}
        onDismiss={onDismiss}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('Autonomous research stopped.');
    expect(screen.getByText('Increase the output allowance and restart.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {
      name: /dismiss proof model output repeatedly truncated/i,
    }));
    expect(onDismiss).toHaveBeenCalledWith('proof-truncation:run-a');
  });

  it('uses the shared embedded lane without fixed positioning', () => {
    const { container } = render(
      <ModelErrorNotificationStack
        embedded
        notifications={[{
          notification_key: 'proof-truncation:run-b',
          message: 'Stopped.',
        }]}
        onDismiss={() => {}}
      />,
    );
    expect(container.firstChild).toHaveStyle({ position: 'static' });
  });

  it('displays provider model repair failures with the affected route', () => {
    render(
      <ModelErrorNotificationStack
        embedded
        notifications={[{
          notification_key: 'model-repair:run-a:proof-id:retired-model',
          title: 'Model configuration requires repair',
          message: "Research stopped because OpenRouter could not serve model 'retired/model'.",
          terminal_guidance: 'Choose an available model or configure a fallback, then retry.',
          model: 'retired/model',
          provider: 'openrouter',
        }]}
        onDismiss={() => {}}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('Model configuration requires repair');
    expect(screen.getByRole('status')).toHaveTextContent('retired/model');
    expect(screen.getByRole('status')).toHaveTextContent('Choose an available model');
  });
});
