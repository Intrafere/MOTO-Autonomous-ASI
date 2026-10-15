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
    const syntheticLibButton = screen.getByRole('button', { name: /SyntheticLib4\s*Coming soon/i });
    expect(syntheticLibButton).toHaveAttribute('aria-haspopup', 'dialog');
    const indicator = screen.getByTestId('connectivity-coming-soon-indicator');
    expect(indicator).toHaveAttribute('aria-hidden', 'true');
    expect(indicator).toHaveClass('connectivity-checkbox');
    expect(indicator).toHaveClass('connectivity-checkbox--x');
    expect(screen.queryByRole('checkbox', { name: /SyntheticLib4/i })).not.toBeInTheDocument();

    await user.click(screen.getByText('SyntheticLib4'));
    expect(onOpenSyntheticLib4).toHaveBeenCalledTimes(1);
  });

  test('shows the private workspace row only in developer mode', async () => {
    const user = userEvent.setup();
    const onOpenSyntheticLib4 = vi.fn();
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled
        connectivityStatus={connectivityStatus}
        capabilities={{
          lmStudioEnabled: true,
          syntheticLibProductState: 'developer_preview',
        }}
        onModeChange={vi.fn()}
        onOpenSyntheticLib4={onOpenSyntheticLib4}
      />,
    );

    expect(screen.queryByTestId('connectivity-coming-soon-indicator')).not.toBeInTheDocument();
    expect(screen.getByText('SyntheticLib4 — Private development')).toBeInTheDocument();
    expect(screen.getByText('Developer preview')).toBeInTheDocument();
    await user.click(screen.getByText('SyntheticLib4 — Private development'));
    expect(onOpenSyntheticLib4).toHaveBeenCalledTimes(1);
  });

  test('fails an unknown lifecycle state closed even in developer mode', () => {
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled
        connectivityStatus={connectivityStatus}
        capabilities={{
          lmStudioEnabled: true,
          syntheticLibProductState: 'unknown_future_state',
        }}
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: /SyntheticLib4\s*Coming soon/i })).toBeInTheDocument();
    expect(screen.queryByText('SyntheticLib4 — Private development')).not.toBeInTheDocument();
  });

  test('keeps Coming soon in hosted generic mode even with developer mode on', () => {
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled
        connectivityStatus={connectivityStatus}
        capabilities={{
          lmStudioEnabled: false,
          genericMode: true,
          syntheticLibProductState: 'developer_preview',
        }}
        onModeChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: /SyntheticLib4\s*Coming soon/i })).toBeInTheDocument();
    expect(screen.queryByText('SyntheticLib4 — Private development')).not.toBeInTheDocument();
  });

  test('rows are real buttons with sibling enable checkboxes, never nested controls', async () => {
    const user = userEvent.setup();
    const onToggleWolfram = vi.fn();
    const onOpenWolfram = vi.fn();
    render(
      <ConnectivityPanel
        appMode="autonomous"
        developerModeEnabled={false}
        connectivityStatus={connectivityStatus}
        capabilities={{ lmStudioEnabled: true }}
        onModeChange={vi.fn()}
        onToggleWolfram={onToggleWolfram}
        onOpenWolfram={onOpenWolfram}
      />,
    );

    const toggle = screen.getByRole('checkbox', { name: 'Enable Wolfram Alpha' });
    const open = screen.getByRole('button', { name: /^Wolfram Alpha/ });
    expect(open).not.toContainElement(toggle);
    expect(document.querySelector('[role="button"] input, button input')).toBeNull();

    await user.click(toggle);
    expect(onToggleWolfram).toHaveBeenCalledTimes(1);
    expect(onOpenWolfram).not.toHaveBeenCalled();

    open.focus();
    await user.keyboard('{Enter}');
    expect(onOpenWolfram).toHaveBeenCalledTimes(1);
  });
});
