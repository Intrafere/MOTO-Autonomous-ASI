import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import StartupDisclaimerModal from './StartupDisclaimerModal';
import StartupProviderSetupModal from './StartupProviderSetupModal';

function setScrollableSize(element, { clientHeight, scrollHeight }) {
  Object.defineProperties(element, {
    clientHeight: { configurable: true, value: clientHeight },
    scrollHeight: { configurable: true, value: scrollHeight },
  });
}

function installAnimationFrameHarness() {
  const callbacks = [];
  const requestAnimationFrame = vi.fn((callback) => {
    callbacks.push(callback);
    return callbacks.length;
  });
  const cancelAnimationFrame = vi.fn();
  vi.stubGlobal('requestAnimationFrame', requestAnimationFrame);
  vi.stubGlobal('cancelAnimationFrame', cancelAnimationFrame);
  return { callbacks, requestAnimationFrame, cancelAnimationFrame };
}

describe('responsive startup modals', () => {
  let resizeCallback;

  beforeEach(() => {
    resizeCallback = null;
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback) {
        resizeCallback = callback;
      }

      observe() {}

      disconnect() {}
    });
    vi.stubGlobal('matchMedia', vi.fn(() => ({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  test('uses an accessible bounded shell and shows only applicable provider cards', () => {
    const { rerender } = render(
      <StartupProviderSetupModal
        isOpen
        capabilities={{ lmStudioEnabled: false }}
        onChooseOpenRouter={vi.fn()}
        onConfirmLmStudio={vi.fn()}
      />,
    );

    expect(screen.getByRole('dialog', { name: 'Choose Your Startup Setup' }))
      .toHaveClass('startup-modal-shell');
    expect(screen.getByText('OpenRouter Setup')).toBeInTheDocument();
    expect(screen.queryByText('LM Studio Setup')).not.toBeInTheDocument();
    expect(screen.queryByText('Cloud Provider Add-On')).not.toBeInTheDocument();

    rerender(
      <StartupProviderSetupModal
        isOpen
        capabilities={{
          lmStudioEnabled: true,
          openAICodexOauthAvailable: true,
        }}
        onChooseOpenRouter={vi.fn()}
        onConfirmLmStudio={vi.fn()}
      />,
    );

    expect(screen.getByText('LM Studio Setup')).toBeInTheDocument();
    expect(screen.getByText('Cloud Provider Add-On')).toBeInTheDocument();
  });

  test('does not show a scroll hint when the disclaimer content fits', () => {
    const { container } = render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled
        onAcknowledge={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 600, scrollHeight: 600 });

    act(() => resizeCallback?.());

    expect(screen.queryByText('More below')).not.toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: 'I Have Read and Acknowledge This Disclaimer',
    })).toBeVisible();
  });

  test('shows an overflow affordance until the user reaches the end', () => {
    const { container } = render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled
        onAcknowledge={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 300, scrollHeight: 800 });

    act(() => resizeCallback?.());
    expect(screen.getByText('More below')).toBeInTheDocument();
    expect(container.querySelector('.startup-modal-overflow-fade')).toBeInTheDocument();

    Object.defineProperty(scrollRegion, 'scrollTop', {
      configurable: true,
      writable: true,
      value: 500,
    });
    fireEvent.scroll(scrollRegion);
    expect(screen.queryByText('More below')).not.toBeInTheDocument();
    expect(container.querySelector('.startup-modal-overflow-fade')).not.toBeInTheDocument();
  });

  test('assisted scrolling starts after the delay and moves at low speed', () => {
    vi.useFakeTimers();
    const { callbacks, requestAnimationFrame } = installAnimationFrameHarness();

    const { container } = render(
      <StartupProviderSetupModal
        isOpen
        capabilities={{ lmStudioEnabled: true }}
        onChooseOpenRouter={vi.fn()}
        onConfirmLmStudio={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 300, scrollHeight: 900 });
    Object.defineProperty(scrollRegion, 'scrollTop', {
      configurable: true,
      writable: true,
      value: 0,
    });
    act(() => resizeCallback?.());

    act(() => vi.advanceTimersByTime(1799));
    expect(requestAnimationFrame).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(1));
    expect(requestAnimationFrame).toHaveBeenCalledTimes(1);

    act(() => callbacks[0](1000));
    act(() => callbacks[1](1100));
    expect(scrollRegion.scrollTop).toBeCloseTo(1.4);
  });

  test('a global interaction cancels running assistance and it never restarts', () => {
    vi.useFakeTimers();
    const {
      callbacks,
      requestAnimationFrame,
      cancelAnimationFrame,
    } = installAnimationFrameHarness();

    const { container } = render(
      <StartupProviderSetupModal
        isOpen
        capabilities={{ lmStudioEnabled: true }}
        onChooseOpenRouter={vi.fn()}
        onConfirmLmStudio={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 300, scrollHeight: 900 });
    act(() => resizeCallback?.());
    act(() => vi.advanceTimersByTime(1800));
    act(() => callbacks[0](1000));
    expect(requestAnimationFrame).toHaveBeenCalledTimes(2);

    fireEvent.pointerDown(container.querySelector('.startup-modal-overlay'));
    expect(cancelAnimationFrame).toHaveBeenCalledWith(2);

    act(() => resizeCallback?.());
    act(() => vi.advanceTimersByTime(5000));
    expect(requestAnimationFrame).toHaveBeenCalledTimes(2);
  });

  test('global interaction cancels delayed assistance before it starts', () => {
    vi.useFakeTimers();
    const { requestAnimationFrame } = installAnimationFrameHarness();

    const { container } = render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled
        onAcknowledge={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 300, scrollHeight: 900 });
    act(() => resizeCallback?.());

    fireEvent.keyDown(document, { key: 'Tab' });
    act(() => vi.advanceTimersByTime(2500));

    expect(requestAnimationFrame).not.toHaveBeenCalled();
  });

  test('reduced motion prevents assisted scrolling while retaining the hint', () => {
    vi.useFakeTimers();
    const requestAnimationFrame = vi.fn();
    vi.stubGlobal('requestAnimationFrame', requestAnimationFrame);
    window.matchMedia.mockReturnValue({
      matches: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });

    const { container } = render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled={false}
        onAcknowledge={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 250, scrollHeight: 700 });
    act(() => resizeCallback?.());
    act(() => vi.advanceTimersByTime(2500));

    expect(requestAnimationFrame).not.toHaveBeenCalled();
    expect(screen.getByText('More below')).toBeInTheDocument();
  });

  test('a hidden document prevents assisted scrolling', () => {
    vi.useFakeTimers();
    const { requestAnimationFrame } = installAnimationFrameHarness();
    const originalHidden = document.hidden;
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      value: true,
    });

    try {
      const { container } = render(
        <StartupDisclaimerModal
          isOpen
          lmStudioEnabled
          onAcknowledge={vi.fn()}
        />,
      );
      const scrollRegion = container.querySelector('.startup-modal-scroll-region');
      setScrollableSize(scrollRegion, { clientHeight: 250, scrollHeight: 700 });
      act(() => resizeCallback?.());
      act(() => vi.advanceTimersByTime(2500));

      expect(requestAnimationFrame).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(document, 'hidden', {
        configurable: true,
        value: originalHidden,
      });
    }
  });

  test('closing the modal clears pending and active assisted-scroll work', () => {
    vi.useFakeTimers();
    const {
      callbacks,
      requestAnimationFrame,
      cancelAnimationFrame,
    } = installAnimationFrameHarness();

    const { container, rerender } = render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled
        onAcknowledge={vi.fn()}
      />,
    );
    const scrollRegion = container.querySelector('.startup-modal-scroll-region');
    setScrollableSize(scrollRegion, { clientHeight: 250, scrollHeight: 700 });
    act(() => resizeCallback?.());
    act(() => vi.advanceTimersByTime(1800));
    act(() => callbacks[0](1000));
    expect(requestAnimationFrame).toHaveBeenCalledTimes(2);

    rerender(
      <StartupDisclaimerModal
        isOpen={false}
        lmStudioEnabled
        onAcknowledge={vi.fn()}
      />,
    );
    expect(cancelAnimationFrame).toHaveBeenCalledWith(2);

    act(() => vi.advanceTimersByTime(5000));
    expect(requestAnimationFrame).toHaveBeenCalledTimes(2);
  });

  test('acknowledgement keeps the existing callback behavior', () => {
    const onAcknowledge = vi.fn();
    render(
      <StartupDisclaimerModal
        isOpen
        lmStudioEnabled
        onAcknowledge={onAcknowledge}
      />,
    );

    fireEvent.click(screen.getByRole('button', {
      name: 'I Have Read and Acknowledge This Disclaimer',
    }));
    expect(onAcknowledge).toHaveBeenCalledTimes(1);
  });
});
