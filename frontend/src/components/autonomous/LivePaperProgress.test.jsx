import React from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import LivePaperProgress from './LivePaperProgress';

const { listeners } = vi.hoisted(() => ({ listeners: new Map() }));
vi.mock('../../services/websocket', () => ({ websocket: { on: (name, callback) => { listeners.set(name, callback); return () => listeners.delete(name); } } }));
vi.mock('../PaperProofViewer', () => ({ default: ({ content, documentId }) => <div data-testid="paper" data-document={documentId}>{content}</div>, PaperProofMetrics: () => null }));
vi.mock('../../utils/downloadHelpers', () => ({ isPDFDownloadAvailable: () => true, sanitizeFilename: value => value, downloadRawText: vi.fn(), downloadPDFViaBackend: vi.fn(), PDF_UNAVAILABLE_MESSAGE: 'Unavailable' }));
const deferred = () => { let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve }; };
beforeEach(() => { vi.useFakeTimers(); listeners.clear(); });
afterEach(() => { cleanup(); vi.useRealTimers(); });

test('newer live paper responses win and carry durable identity', async () => {
  const first = deferred(); const second = deferred();
  const api = { getCurrentPaperProgress: vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise) };
  render(<LivePaperProgress api={api} isCompiling />);
  act(() => { listeners.get('paper_updated')(); });
  await act(async () => { second.resolve({ paper_id: 'b', session_id: 'run', title: 'B', content: 'New body' }); });
  await act(async () => { first.resolve({ paper_id: 'a', title: 'A', content: 'Old body' }); });
  expect(screen.getByTestId('paper')).toHaveTextContent('New body');
  expect(screen.getByTestId('paper')).toHaveAttribute('data-document', 'paper:run:b');
});

test('stop and unmount invalidate pending requests', async () => {
  const pending = deferred();
  const api = { getCurrentPaperProgress: vi.fn(() => pending.promise) };
  const { rerender, unmount } = render(<LivePaperProgress api={api} isCompiling />);
  rerender(<LivePaperProgress api={api} isCompiling={false} />);
  await act(async () => { pending.resolve({ paper_id: 'a', content: 'Old body' }); });
  expect(screen.queryByTestId('paper')).not.toBeInTheDocument();
  expect(listeners.size).toBe(0);
  unmount();
});
