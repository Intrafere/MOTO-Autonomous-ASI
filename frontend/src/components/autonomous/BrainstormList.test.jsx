import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import BrainstormList from './BrainstormList';

const { listeners } = vi.hoisted(() => ({ listeners: new Map() }));
vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn((name, callback) => { listeners.set(name, callback); return () => listeners.delete(name); }) } }));
vi.mock('../LatexRenderer', () => ({ default: ({ content, documentId, showLatex }) => <div data-testid="renderer" data-document={documentId} data-rendered={String(showLatex)}>{content}</div> }));
vi.mock('../../hooks/useProofCheckRuntime', () => ({ isProofRunBusy: () => false, useProofCheckRuntime: () => ({ getSourceState: () => null, manualCheckEnabled: false }) }));
vi.mock('./ProofRunStatusControls', () => ({ default: () => null }));
vi.mock('./ProofCheckModeModal', () => ({ default: () => null }));

const brainstorms = ['a', 'b'].map((topic_id) => ({ topic_id, topic_prompt: `Topic ${topic_id}`, status: 'complete', submission_count: 1 }));
const deferred = () => { let resolve; let reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
beforeEach(() => { vi.useFakeTimers(); listeners.clear(); });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); });

test('loads each document once and ignores stale successes when switching', async () => {
  const a = deferred(); const b = deferred();
  const api = { getBrainstorm: vi.fn((id) => id === 'a' ? a.promise : b.promise) };
  render(<BrainstormList brainstorms={brainstorms} api={api} onRefresh={vi.fn()} />);
  fireEvent.click(screen.getByText('Topic a'));
  expect(api.getBrainstorm).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByText('Topic b'));
  await act(async () => { b.resolve({ content: 'Current B' }); });
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', 'brainstorm:b');
  expect(screen.getByLabelText('LaTeX Rendering')).toBeChecked();
  await act(async () => { a.resolve({ content: 'Stale A' }); });
  expect(screen.getByTestId('renderer')).toHaveTextContent('Current B');
  expect(screen.queryByText('Stale A')).not.toBeInTheDocument();
  fireEvent.click(screen.getByLabelText('LaTeX Rendering'));
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-rendered', 'false');
});

test('ignores stale rejection and does not end the new document loading state', async () => {
  const a = deferred(); const b = deferred();
  const api = { getBrainstorm: vi.fn((id) => id === 'a' ? a.promise : b.promise) };
  render(<BrainstormList brainstorms={brainstorms} api={api} onRefresh={vi.fn()} />);
  fireEvent.click(screen.getByText('Topic a')); fireEvent.click(screen.getByText('Topic b'));
  await act(async () => { a.reject(new Error('old failure')); });
  expect(screen.getByText('Loading...')).toBeInTheDocument();
  await act(async () => { b.resolve({ content: 'Current B' }); });
  expect(screen.getByTestId('renderer')).toHaveTextContent('Current B');
});

test('latest websocket refresh wins over older polling requests', async () => {
  const poll = deferred(); const live = deferred();
  const api = { getBrainstorm: vi.fn().mockResolvedValueOnce({ content: 'Initial' }).mockReturnValueOnce(poll.promise).mockReturnValueOnce(live.promise) };
  render(<BrainstormList brainstorms={brainstorms} api={api} onRefresh={vi.fn()} />);
  fireEvent.click(screen.getByText('Topic a'));
  await act(async () => {});
  act(() => { vi.advanceTimersByTime(5000); });
  act(() => { listeners.get('brainstorm_submission_accepted')({ topic_id: 'a' }); });
  await act(async () => { live.resolve({ content: 'Newest' }); });
  await act(async () => { poll.resolve({ content: 'Old poll' }); });
  expect(screen.getByTestId('renderer')).toHaveTextContent('Newest');
});

test('parent refreshes preserve the mounted reader and polling cadence with fresh API wrappers', async () => {
  const poll = deferred();
  const getBrainstorm = vi.fn().mockResolvedValueOnce({ content: 'Stable content' }).mockReturnValueOnce(poll.promise);
  const props = { brainstorms, onRefresh: vi.fn() };
  const { rerender } = render(<BrainstormList {...props} api={{ getBrainstorm }} />);
  fireEvent.click(screen.getByText('Topic a'));
  await act(async () => {});
  const reader = screen.getByTestId('renderer');
  const subscription = listeners.get('brainstorm_submission_accepted');
  reader.scrollTop = 250;

  for (let i = 0; i < 4; i++) {
    act(() => { vi.advanceTimersByTime(1000); });
    rerender(<BrainstormList {...props} brainstorms={[...brainstorms]} api={{ getBrainstorm }} />);
    expect(screen.getByTestId('renderer')).toBe(reader);
    expect(reader.scrollTop).toBe(250);
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
    expect(getBrainstorm).toHaveBeenCalledTimes(1);
    expect(listeners.get('brainstorm_submission_accepted')).toBe(subscription);
  }
  act(() => { vi.advanceTimersByTime(1000); });
  expect(getBrainstorm).toHaveBeenCalledTimes(2);
  expect(screen.getByTestId('renderer')).toBe(reader);
  await act(async () => { poll.resolve({ content: 'Updated content' }); });
  expect(screen.getByTestId('renderer')).toBe(reader);
  expect(reader).toHaveTextContent('Updated content');
});

test('collapse invalidates requests and unmount removes subscriptions', async () => {
  const pending = deferred();
  const api = { getBrainstorm: vi.fn(() => pending.promise) };
  const { unmount } = render(<BrainstormList brainstorms={brainstorms} api={api} onRefresh={vi.fn()} />);
  fireEvent.click(screen.getByText('Topic a')); fireEvent.click(screen.getByText('Topic a'));
  await act(async () => { pending.resolve({ content: 'Hidden' }); });
  expect(screen.queryByTestId('renderer')).not.toBeInTheDocument();
  expect(listeners.size).toBe(0);
  unmount();
});
