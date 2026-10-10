import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import useExplicitRequest from './useExplicitRequest';
import useSyntheticLibPrivateSurface from './useSyntheticLibPrivateSurface';

describe('useSyntheticLibPrivateSurface', () => {
  it('opens the private workspace only while it is allowed', () => {
    const { result, rerender } = renderHook(
      ({ allowed }) => useSyntheticLibPrivateSurface(allowed),
      { initialProps: { allowed: false } },
    );
    act(() => result.current.open());
    expect(result.current.modalOpen).toBe(true);
    expect(result.current.privateSurfaceOpen).toBe(false);

    act(() => result.current.close());
    rerender({ allowed: true });
    act(() => result.current.open());
    expect(result.current.modalOpen).toBe(true);
    expect(result.current.privateSurfaceOpen).toBe(true);
  });

  it('closes the private workspace at once when developer mode is switched off', () => {
    const { result, rerender } = renderHook(
      ({ allowed }) => useSyntheticLibPrivateSurface(allowed),
      { initialProps: { allowed: true } },
    );
    act(() => result.current.open());
    expect(result.current.privateSurfaceOpen).toBe(true);

    rerender({ allowed: false });

    expect(result.current.privateSurfaceOpen).toBe(false);
    expect(result.current.modalOpen).toBe(false);
  });

  it('leaves the Coming soon modal open when the gate was never granted', () => {
    const { result, rerender } = renderHook(
      ({ allowed }) => useSyntheticLibPrivateSurface(allowed),
      { initialProps: { allowed: false } },
    );
    act(() => result.current.open());
    rerender({ allowed: false });
    expect(result.current.modalOpen).toBe(true);
  });
});

describe('useExplicitRequest', () => {
  it('passes isCurrent so multi-step fetchers stop once the surface closes', async () => {
    let resolveFirst;
    const second = vi.fn();
    const fetcher = vi.fn(async (value, { isCurrent }) => {
      await new Promise((resolve) => {
        resolveFirst = resolve;
      });
      if (!isCurrent()) return null;
      second(value);
      return value;
    });
    const { result, rerender } = renderHook(
      ({ active }) => useExplicitRequest(active, fetcher, () => 'failed'),
      { initialProps: { active: true } },
    );

    let pending;
    act(() => {
      pending = result.current[1]('payload');
    });
    rerender({ active: false });
    resolveFirst();
    await act(async () => {
      expect(await pending).toBe(false);
    });

    expect(fetcher).toHaveBeenCalledWith('payload', expect.objectContaining({ isCurrent: expect.any(Function) }));
    expect(second).not.toHaveBeenCalled();
    expect(result.current[0].status).toBe('idle');
  });

  it('reports isCurrent true for an uninterrupted request', async () => {
    const fetcher = vi.fn(async (_value, { isCurrent }) => isCurrent());
    const { result } = renderHook(() => useExplicitRequest(true, fetcher, () => 'failed'));
    await act(async () => {
      expect(await result.current[1]('x')).toBe(true);
    });
    expect(result.current[0]).toMatchObject({ status: 'complete', data: true });
  });
});
