import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import LatexRenderer from './LatexRenderer';
const mocks = vi.hoisted(() => ({ render: vi.fn() }));
vi.mock('../utils/latexRenderService', () => ({ renderLatexAsync: (...args) => mocks.render(...args), scheduleLatexMainTask: async task => task() }));
let observers = [];
class Observer {
  constructor(callback) { this.callback = callback; observers.push(this); }
  observe() {}
  disconnect() {}
  visible(value) { this.callback([{ isIntersecting: value }]); }
}
afterEach(() => { window.getSelection()?.removeAllRanges(); vi.useRealTimers(); observers = []; vi.unstubAllGlobals(); mocks.render.mockReset(); });
it('retains later DOM and measured height after an earlier edit and unchanged polling', async () => {
  vi.useFakeTimers(); vi.stubGlobal('IntersectionObserver', Observer);
  const resizeCallbacks = [];
  vi.stubGlobal('ResizeObserver', class { constructor(callback) { resizeCallbacks.push(callback); } observe() {} disconnect() {} });
  mocks.render.mockImplementation(async text => `<b>${text}</b>`);
  const first = 'a'.repeat(3100) + '\n\n', later = 'later'.repeat(700);
  const { container, rerender } = render(<LatexRenderer content={first + later} />);
  await act(async () => observers.forEach(observer => observer.visible(true)));
  const nodes = [...container.querySelectorAll('[data-chunk]')];
  const laterMarkup = nodes[1].firstChild;
  nodes[1].getBoundingClientRect = () => ({ height: 777 });
  act(() => resizeCallbacks[1]());
  // Resize observer updates retained geometry without replacing the chunk shell.
  const calls = mocks.render.mock.calls.length;
  rerender(<LatexRenderer content={first + later} />);
  await act(async () => vi.advanceTimersByTimeAsync(1500));
  expect(mocks.render).toHaveBeenCalledTimes(calls);
  rerender(<LatexRenderer content={'changed ' + first + later} />);
  await act(async () => vi.advanceTimersByTimeAsync(1500));
  expect(container.querySelectorAll('[data-chunk]')[1]).toBe(nodes[1]);
  expect(nodes[1].firstChild).toBe(laterMarkup);
  expect(observers).toHaveLength(2);
  await act(async () => observers[1].visible(false));
  expect(nodes[1].style.minHeight).toBe('777px');
});
it('publishes continuously streamed updates within the bounded interval', async () => {
  vi.useFakeTimers(); vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockImplementation(async text => text);
  const { rerender } = render(<LatexRenderer content="v0" />);
  await act(async () => observers[0].visible(true));
  for (let i = 1; i <= 3; i++) {
    rerender(<LatexRenderer content={`v${i}`} />);
    await act(async () => vi.advanceTimersByTimeAsync(500));
  }
  expect(mocks.render.mock.calls.at(-1)[0]).toBe('v3');
});
it('uses latest raw content immediately when returning to rendered mode', async () => {
  vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockImplementation(async text => text);
  const { rerender } = render(<LatexRenderer content="old" showLatex={false} />);
  rerender(<LatexRenderer content="latest" showLatex={false} />);
  rerender(<LatexRenderer content="latest" showLatex />);
  await act(async () => observers.at(-1).visible(true));
  expect(mocks.render.mock.calls.at(-1)[0]).toBe('latest');
});
it('defers edited selected content until selection ends', async () => {
  vi.useFakeTimers(); vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockImplementation(async text => `<b>${text}</b>`);
  const { rerender } = render(<LatexRenderer content="original" />);
  await act(async () => observers[0].visible(true));
  const node = screen.getByText('original'), range = document.createRange(); range.selectNodeContents(node);
  act(() => { window.getSelection().addRange(range); document.dispatchEvent(new Event('selectionchange')); });
  rerender(<LatexRenderer content="replacement" />);
  await act(async () => vi.advanceTimersByTimeAsync(1600));
  expect(screen.getByText('original')).toBe(node);
  expect(mocks.render).toHaveBeenCalledTimes(1);
  await act(async () => { window.getSelection().removeAllRanges(); document.dispatchEvent(new Event('selectionchange')); });
  expect(screen.getByText('replacement')).toBeInTheDocument();
});
it('defers updates while a rendered block owns focus', async () => {
  vi.useFakeTimers(); vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockImplementation(async text => `<b>${text}</b>`);
  const { container, rerender } = render(<LatexRenderer content="focused old" />);
  await act(async () => observers[0].visible(true));
  const block = container.querySelector('[data-chunk]'); block.tabIndex = 0;
  act(() => block.focus());
  rerender(<LatexRenderer content="focused new" />);
  await act(async () => vi.advanceTimersByTimeAsync(1600));
  expect(screen.getByText('focused old')).toBeInTheDocument();
  await act(async () => block.blur());
  expect(screen.getByText('focused new')).toBeInTheDocument();
});
it('offers a visible retry and viewport search limitation notice', async () => {
  vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockRejectedValueOnce(new Error('busy')).mockResolvedValue('recovered');
  render(<LatexRenderer content="source" />);
  act(() => observers[0].visible(true));
  fireEvent.click(await screen.findByRole('button', { name: 'Retry rendered block' }));
  await screen.findByText('recovered');
  expect(screen.getByText(/Browser Find and Select All/)).toBeInTheDocument();
});
it('sanitizes worker results and evicts offscreen markup', async () => {
  vi.stubGlobal('IntersectionObserver', Observer);
  mocks.render.mockResolvedValue('<b>safe</b><img src=x onerror=alert(1)><script>bad()</script>');
  const { container } = render(<LatexRenderer content="$x$" />);
  act(() => observers[0].visible(true));
  await screen.findByText('safe');
  expect(container.querySelector('script,img')).toBeNull();
  act(() => observers[0].visible(false));
  await waitFor(() => expect(screen.queryByText('safe')).toBeNull());
});
it('shows full escaped raw source when worker fails', async () => {
  vi.stubGlobal('IntersectionObserver', Observer);
  mocks.render.mockRejectedValue(new Error('Worker unavailable'));
  const source = '<script>unsafe()</script> $x$';
  const { container } = render(<LatexRenderer content={source} />);
  act(() => observers[0].visible(true));
  await screen.findByText(source);
  expect(container.querySelector('script')).toBeNull();
  expect(screen.getByText(/Full raw block shown/)).toBeInTheDocument();
});
it('fences old document promises and resets identity immediately', async () => {
  vi.stubGlobal('IntersectionObserver', Observer);
  let resolveOld;
  mocks.render.mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve; })).mockResolvedValue('new output');
  const { rerender } = render(<LatexRenderer content="old" documentId="a" />);
  act(() => observers[0].visible(true));
  rerender(<LatexRenderer content="new" documentId="b" />);
  act(() => observers.at(-1).visible(true));
  await screen.findByText('new output');
  await act(async () => resolveOld('old output'));
  expect(screen.queryByText('old output')).toBeNull();
});
it('pins selected markup until selection is released', async () => {
  vi.stubGlobal('IntersectionObserver', Observer); mocks.render.mockResolvedValue('<b>selected text</b>');
  render(<LatexRenderer content="source" />);
  act(() => observers[0].visible(true)); const node = await screen.findByText('selected text');
  const selection = window.getSelection(), range = document.createRange(); range.selectNodeContents(node);
  act(() => { selection.addRange(range); document.dispatchEvent(new Event('selectionchange')); observers[0].visible(false); });
  expect(screen.getByText('selected text')).toBeInTheDocument();
  act(() => { selection.removeAllRanges(); document.dispatchEvent(new Event('selectionchange')); });
  await waitFor(() => expect(screen.queryByText('selected text')).toBeNull());
});
