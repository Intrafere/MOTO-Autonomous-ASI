import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';
import ConnectivityPanel from './ConnectivityPanel';

const connectivityStatus = {
  inference: {
    openrouter_oauth: { status: 'active' },
    lm_studio: { status: 'ready' },
  },
  skills: {
    syntheticlib4: { enabled: true, status: 'ready' },
    agent_conversation_memory: { enabled: true, status: 'active' },
    wolfram_alpha: { enabled: false, status: 'inactive' },
  },
};

describe('ConnectivityPanel', () => {
  test('shows startup state before connectivity status hydrates', () => {
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled={false}
        connectivityStatus={null}
        capabilities={{ lmStudioEnabled: true }}
        onModeChange={vi.fn()}
      />
    );

    expect(screen.getAllByText('starting').length).toBeGreaterThanOrEqual(3);
  });

  test('uses workflow-memory language for Session History Memory tooltip', () => {
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled={false}
        connectivityStatus={connectivityStatus}
        capabilities={{ lmStudioEnabled: true }}
        onModeChange={vi.fn()}
      />
    );

    expect(screen.getByTitle(/Configure local proof-history memory for Assistant workflow-memory search/i)).toBeInTheDocument();
  });

  test('shows SyntheticLib4 as coming soon even when backend has snapshot state', async () => {
    const user = userEvent.setup();
    const onOpenSyntheticLib4 = vi.fn();

    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled={false}
        connectivityStatus={{
          ...connectivityStatus,
          skills: {
            ...connectivityStatus.skills,
            syntheticlib4: { enabled: true, status: 'outdated', outdated: true },
          },
        }}
        capabilities={{ lmStudioEnabled: true }}
        onModeChange={vi.fn()}
        onOpenSyntheticLib4={onOpenSyntheticLib4}
      />
    );

    const status = screen.getByText('Coming soon');
    expect(status).toBeInTheDocument();
    expect(status).toHaveClass('connectivity-status--pending');
    const syntheticLibCheckbox = screen.getByRole('checkbox', { name: /SyntheticLib4 coming soon/i });
    expect(syntheticLibCheckbox).toHaveClass('connectivity-checkbox');
    expect(syntheticLibCheckbox).toHaveClass('connectivity-checkbox--x');
    expect(syntheticLibCheckbox).not.toBeChecked();

    await user.click(screen.getByText('SyntheticLib4'));
    expect(onOpenSyntheticLib4).toHaveBeenCalledTimes(1);
  });
});

