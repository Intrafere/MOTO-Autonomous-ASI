import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import useWhatsNew from './useWhatsNew';

afterEach(() => { cleanup(); localStorage.clear(); });

it('orders waiver, release highlights, then startup setup and supports reopening', () => {
  const { result, rerender, unmount } = renderHook(useWhatsNew, {
    initialProps: { backendVersion: '1.1.06', showDisclaimer: true },
  });
  expect(result.current.whatsNewVersion).toBeNull();
  expect(result.current.startupBlocked).toBe(true);
  rerender({ backendVersion: '1.1.06', showDisclaimer: false });
  expect(result.current.whatsNewVersion).toBe('1.1.06');
  expect(result.current.startupBlocked).toBe(true);
  act(() => result.current.closeWhatsNew());
  expect(result.current.startupBlocked).toBe(false);
  act(() => result.current.openWhatsNew());
  expect(result.current.whatsNewVersion).toBe('1.1.06');
  act(() => result.current.closeWhatsNew());
  unmount();
  const next = renderHook(useWhatsNew, { initialProps: { backendVersion: '1.1.06', showDisclaimer: false } });
  expect(next.result.current.whatsNewVersion).toBeNull();
  next.rerender({ backendVersion: '1.1.07', showDisclaimer: false });
  expect(next.result.current.whatsNewVersion).toBe('1.1.07');
});

it('waits for confirmed identity and retains it across failed feature refreshes', () => {
  const { result, rerender } = renderHook(useWhatsNew, {
    initialProps: { backendVersion: '', showDisclaimer: false },
  });
  expect(result.current.whatsNewVersion).toBeNull();
  rerender({ backendVersion: 'v1.1.07', showDisclaimer: false });
  expect(result.current.whatsNewVersion).toBe('1.1.07');
  act(() => result.current.closeWhatsNew());
  rerender({ backendVersion: '', showDisclaimer: false });
  expect(result.current.currentVersion).toBe('1.1.07');
  expect(result.current.whatsNewVersion).toBeNull();
  act(() => result.current.openWhatsNew());
  expect(result.current.whatsNewVersion).toBe('1.1.07');
});
