import { render, screen } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';
import WolframAlphaAccessModal from './WolframAlphaAccessModal';

vi.mock('../services/api', () => ({
  api: {
    getWolframStatus: vi.fn(() => new Promise(() => {})),
  },
  connectivityAPI: {
    updateToggles: vi.fn(),
    getStatus: vi.fn(),
  },
}));

describe('WolframAlphaAccessModal', () => {
  test('does not claim the App ID is missing while status is loading', () => {
    render(
      <WolframAlphaAccessModal
        isOpen
        onClose={vi.fn()}
        connectivityStatus={null}
        capabilities={{ genericMode: false }}
      />
    );

    expect(screen.getByText('Checking Wolfram Alpha App ID status...')).toBeInTheDocument();
    expect(screen.queryByText('No Wolfram Alpha App ID is configured.')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Clear App ID' })).toBeDisabled();
  });
});
