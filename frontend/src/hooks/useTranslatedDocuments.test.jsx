import React from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import useTranslatedDocuments, { VIEWING_REFRESH_MS, fnv1aHash } from './useTranslatedDocuments';
import { AutoTranslateContext } from './useAutoTranslate';
import { translationAPI } from '../services/api';
import { websocket } from '../services/websocket';

const { listeners } = vi.hoisted(() => ({ listeners: new Map() }));

vi.mock('../services/websocket', () => ({
  websocket: {
    on: vi.fn((name, callback) => {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(callback);
      return () => listeners.get(name)?.delete(callback);
    }),
    off: vi.fn(),
  },
}));

vi.mock('../services/api', () => ({
  translationAPI: {
    getStatus: vi.fn(),
    registerDocuments: vi.fn(),
    lookupDocuments: vi.fn(),
    documentProgress: vi.fn(),
  },
}));

const LANGUAGES = [
  { key: 'es', name: 'Spanish', custom: false },
  { key: 'fr', name: 'French', custom: false },
];

let progressSubscribers;
let contextValue;

function makeContext(overrides = {}) {
  return {
    available: true,
    status: { enabled: true, state: 'translating' },
    enabled: true,
    languages: LANGUAGES,
    displayLanguage: 'es',
    setDisplayLanguage: vi.fn(),
    refreshStatus: vi.fn(),
    subscribeProgress: (callback) => {
      progressSubscribers.add(callback);
      return () => progressSubscribers.delete(callback);
    },
    recentActivity: [],
    ...overrides,
  };
}

function wrapper({ children }) {
  return <AutoTranslateContext.Provider value={contextValue}>{children}</AutoTranslateContext.Provider>;
}

function emitProgress(data, eventType = 'translation_progress') {
  act(() => {
    [...progressSubscribers].forEach((callback) => callback(data, eventType));
  });
}

function emitQueueChanged(queueVersion = 1) {
  emitProgress({ queue_version: queueVersion }, 'translation_queue_changed');
}

function emitSocket(name, data = {}) {
  act(() => {
    [...(listeners.get(name) || [])].forEach((callback) => callback(data));
  });
}

async function advance(ms) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function flushPromises() {
  await act(async () => {
    for (let index = 0; index < 10; index += 1) {
      await Promise.resolve();
    }
  });
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function proofDoc(id, parts = { theorem_statement: `Statement ${id}` }) {
  return { docKey: `proof:autonomous:run:${id}`, kind: 'proof', title: `Proof ${id}`, profile: 'fields', parts };
}

function statusFor(document, overrides = {}) {
  return {
    doc_key: document.docKey ?? document.doc_key,
    doc_hash: `hash-${document.docKey ?? document.doc_key}`,
    language: 'es',
    status: 'pending',
    translated_units: 0,
    total_units: 2,
    queue_position: 1,
    parts: null,
    ...overrides,
  };
}

function echoPending() {
  translationAPI.registerDocuments.mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => statusFor(document)),
  }));
}

beforeEach(() => {
  vi.useFakeTimers();
  listeners.clear();
  progressSubscribers = new Set();
  contextValue = makeContext();
  translationAPI.registerDocuments.mockReset();
  translationAPI.lookupDocuments.mockReset();
  translationAPI.documentProgress.mockReset();
  translationAPI.documentProgress.mockResolvedValue({ success: true, documents: [] });
  translationAPI.getStatus.mockReset();
  websocket.on.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('useTranslatedDocuments', () => {
  test('is inert without a provider and never calls the translation API', async () => {
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents));

    await advance(5000);

    expect(result.current.active).toBe(false);
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'original', parts: null });
    expect(result.current.overall.status).toBe('original');
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    expect(websocket.on).not.toHaveBeenCalled();
  });

  test('stays inactive for the original language, the enabled option, or no documents', async () => {
    echoPending();
    contextValue = makeContext({ displayLanguage: 'original' });
    const documents = [proofDoc('a')];
    const original = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    contextValue = makeContext();
    const optedOut = renderHook(() => useTranslatedDocuments(documents, { enabled: false }), { wrapper });
    const empty = renderHook(() => useTranslatedDocuments([]), { wrapper });
    contextValue = makeContext({ enabled: false });
    const disabled = renderHook(() => useTranslatedDocuments(documents), { wrapper });

    emitQueueChanged(1);
    await advance(VIEWING_REFRESH_MS * 3);

    [original, optedOut, empty, disabled].forEach(({ result }) => expect(result.current.active).toBe(false));
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
  });

  test('registers documents after the debounce in chunks of at most 100 with only non-empty parts', async () => {
    echoPending();
    const documents = Array.from({ length: 150 }, (_, index) => proofDoc(index, {
      theorem_statement: `Statement ${index}`,
      formal_sketch: '   ',
      novelty_reasoning: '',
    }));
    documents.push(proofDoc('blank', { theorem_statement: ' ', formal_sketch: '' }));

    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    expect(result.current.active).toBe(true);
    expect(result.current.byKey['proof:autonomous:run:0'].status).toBe('loading');

    await advance(399);
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    await advance(1);
    await flushPromises();

    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
    const [first, second] = translationAPI.registerDocuments.mock.calls.map(([payload]) => payload);
    expect(first.language).toBe('es');
    expect(first.viewing).toBe(true);
    expect(first.documents).toHaveLength(100);
    expect(second.documents).toHaveLength(50);
    expect(first.documents[0]).toEqual({
      docKey: 'proof:autonomous:run:0',
      kind: 'proof',
      title: 'Proof 0',
      profile: 'fields',
      parts: { theorem_statement: 'Statement 0' },
    });
    expect([...first.documents, ...second.documents].map((document) => document.docKey))
      .not.toContain('proof:autonomous:run:blank');
    expect(result.current.byKey['proof:autonomous:run:149']).toMatchObject({
      status: 'pending',
      docHash: 'hash-proof:autonomous:run:149',
      total: 2,
    });
    expect(result.current.byKey['proof:autonomous:run:blank']).toMatchObject({
      status: 'empty',
      parts: { theorem_statement: ' ', formal_sketch: '' },
    });
  });

  test('maps statuses per document and aggregates the overall state', async () => {
    const documents = [
      proofDoc('done', { theorem_statement: 'Original statement', formal_sketch: 'Original sketch' }),
      proofDoc('waiting'),
      proofDoc('math', { theorem_statement: '$x^2$' }),
    ];
    translationAPI.registerDocuments.mockResolvedValue({
      success: true,
      documents: [
        statusFor(documents[0], {
          status: 'complete',
          translated_units: 2,
          total_units: 2,
          // A position on a finished document is meaningless and ignored.
          queue_position: 0,
          parts: { theorem_statement: 'Enunciado traducido' },
        }),
        statusFor(documents[1], { translated_units: 2, total_units: 5, queue_position: 3 }),
        statusFor(documents[2], {
          status: 'empty',
          total_units: 0,
          queue_position: null,
          parts: { theorem_statement: '$x^2$' },
        }),
      ],
    });

    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();

    expect(result.current.byKey['proof:autonomous:run:done']).toMatchObject({
      status: 'complete',
      queuePosition: null,
      parts: { theorem_statement: 'Enunciado traducido', formal_sketch: 'Original sketch' },
    });
    expect(result.current.byKey['proof:autonomous:run:waiting']).toMatchObject({
      status: 'pending',
      translated: 2,
      total: 5,
      queuePosition: 3,
      parts: null,
    });
    expect(result.current.byKey['proof:autonomous:run:math']).toMatchObject({
      status: 'empty',
      queuePosition: null,
      parts: { theorem_statement: '$x^2$' },
    });
    expect(result.current.overall).toEqual({
      status: 'pending', translated: 4, total: 7, queuePosition: 3, requestable: true,
    });
    expect(result.current.language).toBe('es');
    expect(result.current.languageName).toBe('Spanish');

    translationAPI.lookupDocuments.mockResolvedValue({
      success: true,
      documents: [statusFor(documents[1], {
        status: 'complete',
        translated_units: 5,
        total_units: 5,
        parts: { theorem_statement: 'Listo' },
      })],
    });
    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:waiting'] });
    await advance(800);
    await flushPromises();

    expect(result.current.overall).toEqual({
      status: 'complete', translated: 7, total: 7, queuePosition: null, requestable: false,
    });
  });

  test('re-looks-up only matching pending hashes on progress and ignores other languages', async () => {
    echoPending();
    const documents = [proofDoc('a'), proofDoc('b'), proofDoc('c')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => ({
        doc_key: docHash.replace('hash-', ''),
        doc_hash: docHash,
        language: 'es',
        status: 'complete',
        translated_units: 2,
        total_units: 2,
        parts: { theorem_statement: `Traducido ${docHash}` },
      })),
    }));

    emitProgress({ language_key: 'fr', doc_hashes: ['hash-proof:autonomous:run:a'] });
    emitProgress({ language_key: 'es', doc_hashes: ['someone-else'] });
    await advance(1000);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();

    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:a', 'unrelated'] });
    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:b'] });
    await advance(799);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
    await advance(1);
    await flushPromises();

    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a', 'hash-proof:autonomous:run:b'],
      viewing: true,
    });
    expect(result.current.byKey['proof:autonomous:run:a'].status).toBe('complete');
    expect(result.current.byKey['proof:autonomous:run:c'].status).toBe('pending');
  });

  test('re-looks up every non-complete document on queue changes so positions update live', async () => {
    const documents = [proofDoc('a'), proofDoc('b'), proofDoc('c')];
    translationAPI.registerDocuments.mockResolvedValue({
      success: true,
      documents: [
        statusFor(documents[0], { queue_position: 2 }),
        statusFor(documents[1], { status: 'complete', translated_units: 2, parts: { theorem_statement: 'B' } }),
        // Only non-negative integers are queue positions.
        statusFor(documents[2], { queue_position: -1 }),
      ],
    });
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a'].queuePosition).toBe(2);
    expect(result.current.byKey['proof:autonomous:run:c'].queuePosition).toBeNull();
    expect(result.current.overall.queuePosition).toBe(2);

    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => ({
        doc_key: docHash.replace('hash-', ''),
        doc_hash: docHash,
        language: 'es',
        status: 'pending',
        translated_units: docHash.endsWith(':a') ? 1 : 0,
        total_units: 2,
        queue_position: docHash.endsWith(':a') ? 0 : 1,
        parts: null,
      })),
    }));
    emitQueueChanged(1);
    await advance(300);
    emitQueueChanged(2);
    await advance(499);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
    await advance(1);
    await flushPromises();

    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a', 'hash-proof:autonomous:run:c'],
      viewing: true,
    });
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ queuePosition: 0, translated: 1 });
    expect(result.current.byKey['proof:autonomous:run:c']).toMatchObject({ queuePosition: 1 });
    expect(result.current.overall).toEqual({
      status: 'pending', translated: 3, total: 6, queuePosition: 0, requestable: true,
    });

    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => statusFor({ doc_key: docHash.replace('hash-', '') }, {
        status: 'complete',
        translated_units: 2,
        parts: { theorem_statement: 'Hecho' },
      })),
    }));
    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:a', 'hash-proof:autonomous:run:c'] });
    await advance(800);
    await flushPromises();
    expect(result.current.overall).toEqual({
      status: 'complete', translated: 6, total: 6, queuePosition: null, requestable: false,
    });

    // Nothing left to translate: queue changes no longer cause lookups.
    emitQueueChanged(3);
    await advance(2000);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(2);
  });

  test('a queue change during an in-flight registration triggers a follow-up lookup', async () => {
    const pending = deferred();
    translationAPI.registerDocuments.mockReturnValueOnce(pending.promise);
    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => statusFor({ doc_key: docHash.replace('hash-', '') }, {
        queue_position: 0,
      })),
    }));
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);

    emitQueueChanged(4);
    await act(async () => {
      pending.resolve({ success: true, documents: [statusFor(documents[0], { queue_position: 2 })] });
    });
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a'].queuePosition).toBe(2);

    await advance(800);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a'],
      viewing: true,
    });
    expect(result.current.byKey['proof:autonomous:run:a'].queuePosition).toBe(0);
  });

  test('re-registers on websocket connected and when document text changes', async () => {
    echoPending();
    let documents = [proofDoc('a')];
    const { result, rerender } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);

    emitSocket('connected');
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);

    documents = [proofDoc('a', { theorem_statement: 'Edited statement' })];
    rerender();
    expect(result.current.byKey['proof:autonomous:run:a'].status).toBe('loading');
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(3);
    expect(translationAPI.registerDocuments.mock.calls[2][0].documents[0].parts)
      .toEqual({ theorem_statement: 'Edited statement' });
    expect(result.current.byKey['proof:autonomous:run:a'].status).toBe('pending');
  });

  test('re-registers documents shown as complete when the service generation changes, as after a clear', async () => {
    const document = proofDoc('a');
    translationAPI.registerDocuments.mockResolvedValueOnce({
      success: true,
      documents: [statusFor(document, {
        status: 'complete',
        translated_units: 2,
        queue_position: null,
        parts: { theorem_statement: 'Enunciado a' },
      })],
    });
    contextValue = makeContext({ status: { enabled: true, state: 'idle', generation: 3 } });
    const { result, rerender } = renderHook(() => useTranslatedDocuments([document]), { wrapper });
    await advance(400);
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a'].status).toBe('complete');

    // A status refresh with the same generation sends nothing.
    contextValue = makeContext({ status: { enabled: true, state: 'idle', generation: 3 } });
    rerender();
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);

    // Clearing stored translations advances the generation; the document is translated again.
    echoPending();
    contextValue = makeContext({ status: { enabled: true, state: 'translating', generation: 4 } });
    rerender();
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
    expect(result.current.byKey['proof:autonomous:run:a'].status).toBe('pending');

    // A restarted backend counts from 0 again; a clear after the restart still re-registers.
    contextValue = makeContext({ status: { enabled: true, state: 'idle', generation: 0 } });
    rerender();
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(3);
    contextValue = makeContext({ status: { enabled: true, state: 'translating', generation: 2 } });
    rerender();
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(4);
  });

  test('maps registration rows to documents in request order so identical documents each get a status', async () => {
    const parts = { theorem_statement: 'The same statement', novelty_reasoning: 'Same reasoning' };
    const documents = [proofDoc('first', parts), proofDoc('second', parts), proofDoc('other')];
    const sharedHash = 'hash-shared';
    translationAPI.registerDocuments.mockResolvedValue({
      success: true,
      documents: [
        // Identical content shares one backend record, which may carry either key.
        statusFor({ docKey: documents[1].docKey }, { doc_hash: sharedHash, translated_units: 1 }),
        statusFor({ docKey: documents[1].docKey }, { doc_hash: sharedHash, translated_units: 1 }),
        statusFor(documents[2]),
      ],
    });

    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();

    expect(result.current.byKey['proof:autonomous:run:first']).toMatchObject({ status: 'pending', translated: 1, docHash: sharedHash });
    expect(result.current.byKey['proof:autonomous:run:second']).toMatchObject({ status: 'pending', translated: 1, docHash: sharedHash });
    expect(result.current.byKey['proof:autonomous:run:other']).toMatchObject({ status: 'pending' });
    expect(result.current.overall.status).toBe('pending');

    translationAPI.lookupDocuments.mockResolvedValue({
      success: true,
      documents: [{
        doc_key: documents[1].docKey,
        doc_hash: sharedHash,
        language: 'es',
        status: 'complete',
        translated_units: 2,
        total_units: 2,
        parts: { theorem_statement: 'El mismo enunciado' },
      }],
    });
    emitProgress({ language_key: 'es', doc_hashes: [sharedHash] });
    await advance(800);
    await flushPromises();

    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({ language: 'es', docHashes: [sharedHash], viewing: true });
    ['first', 'second'].forEach((id) => {
      expect(result.current.byKey[`proof:autonomous:run:${id}`]).toMatchObject({
        status: 'complete',
        parts: { theorem_statement: 'El mismo enunciado', novelty_reasoning: 'Same reasoning' },
      });
    });
  });

  test('falls back to exact keys when registration rows are out of order', async () => {
    const documents = [proofDoc('a'), proofDoc('b')];
    translationAPI.registerDocuments.mockResolvedValue({
      success: true,
      documents: [
        statusFor(documents[1], { status: 'complete', parts: { theorem_statement: 'B traducido' } }),
        statusFor(documents[0], { translated_units: 1 }),
      ],
    });

    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();

    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'pending', translated: 1 });
    expect(result.current.byKey['proof:autonomous:run:b']).toMatchObject({
      status: 'complete',
      parts: { theorem_statement: 'B traducido' },
    });
  });

  test('progress reported while a registration is in flight triggers a lookup once its reply is applied', async () => {
    const pending = deferred();
    translationAPI.registerDocuments.mockReturnValueOnce(pending.promise);
    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => ({
        doc_key: docHash.replace('hash-', ''),
        doc_hash: docHash,
        language: 'es',
        status: 'complete',
        translated_units: 2,
        total_units: 2,
        parts: { theorem_statement: 'Traducido' },
      })),
    }));
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);

    // The last part is finalized after the backend computed the registration reply
    // but before the client applies it; the event names a hash the client cannot know yet.
    emitProgress({ language_key: 'fr', doc_hashes: ['hash-proof:autonomous:run:a'] });
    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:a'] });
    await act(async () => {
      pending.resolve({ success: true, documents: [statusFor(documents[0], { translated_units: 1 })] });
    });
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'pending', translated: 1 });

    await advance(800);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a'],
      viewing: true,
    });
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'complete' });
  });

  test('no follow-up lookup without current-language progress during the registration', async () => {
    const pending = deferred();
    translationAPI.registerDocuments.mockReturnValueOnce(pending.promise);
    const documents = [proofDoc('a')];
    renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);

    emitProgress({ language_key: 'fr', doc_hashes: ['hash-proof:autonomous:run:a'] });
    await act(async () => {
      pending.resolve({ success: true, documents: [statusFor(documents[0])] });
    });
    await flushPromises();
    await advance(2000);

    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
  });

  test('a lookup that reports an unknown document re-registers it', async () => {
    echoPending();
    const documents = [proofDoc('a')];
    renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    translationAPI.lookupDocuments.mockResolvedValue({
      success: true,
      documents: [statusFor(documents[0], { status: 'unknown' })],
    });

    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:a'] });
    await advance(800);
    await flushPromises();
    await advance(400);
    await flushPromises();

    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
  });

  test('fences stale responses after a language change', async () => {
    const spanish = deferred();
    translationAPI.registerDocuments
      .mockReturnValueOnce(spanish.promise)
      .mockImplementationOnce(async ({ documents }) => ({
        success: true,
        documents: documents.map((document) => statusFor(document, { language: 'fr', translated_units: 1 })),
      }));
    const documents = [proofDoc('a')];
    const { result, rerender } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);

    contextValue = makeContext({ displayLanguage: 'fr' });
    rerender();
    await act(async () => {
      spanish.resolve({
        success: true,
        documents: [statusFor(documents[0], { status: 'complete', parts: { theorem_statement: 'Enunciado' } })],
      });
    });
    await flushPromises();

    expect(result.current.language).toBe('fr');
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'loading', parts: null });

    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenLastCalledWith(expect.objectContaining({ language: 'fr' }));
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'pending', translated: 1 });
  });

  test('ignores responses that arrive after unmount', async () => {
    const pending = deferred();
    translationAPI.registerDocuments.mockReturnValueOnce(pending.promise);
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    const documents = [proofDoc('a')];
    const { result, unmount } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    const lastSeen = result.current;
    unmount();

    await act(async () => {
      pending.resolve({ success: true, documents: [statusFor(documents[0])] });
    });
    await advance(5000);

    expect(result.current).toBe(lastSeen);
    expect(progressSubscribers.size).toBe(0);
    expect(listeners.get('connected')?.size || 0).toBe(0);
    expect(consoleError).not.toHaveBeenCalled();
  });

  test('refreshes viewing priority at least every four minutes while a document is unfinished', async () => {
    echoPending();
    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => statusFor({ doc_key: docHash.replace('hash-', '') })),
    }));
    const documents = [proofDoc('a'), proofDoc('b')];
    const everything = {
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a', 'hash-proof:autonomous:run:b'],
      viewing: true,
    };
    renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();

    await advance(VIEWING_REFRESH_MS - 1);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
    await advance(1);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).toHaveBeenLastCalledWith(everything);

    // A lookup of one document does not refresh the other, so the cadence keeps running.
    await advance(60000);
    emitProgress({ language_key: 'es', doc_hashes: ['hash-proof:autonomous:run:a'] });
    await advance(800);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenLastCalledWith({
      language: 'es',
      docHashes: ['hash-proof:autonomous:run:a'],
      viewing: true,
    });
    await advance(VIEWING_REFRESH_MS - 60800 - 1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(2);
    await advance(1);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(3);
    expect(translationAPI.lookupDocuments).toHaveBeenLastCalledWith(everything);

    // A registration refreshes every document and restarts the cadence.
    await advance(100000);
    emitSocket('connected');
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
    await advance(VIEWING_REFRESH_MS - 1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(3);
    await advance(1);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(4);
  });

  test('stops refreshing viewing priority once everything is complete, when disabled, and on unmount', async () => {
    echoPending();
    translationAPI.lookupDocuments.mockImplementation(async ({ docHashes }) => ({
      success: true,
      documents: docHashes.map((docHash) => statusFor({ doc_key: docHash.replace('hash-', '') }, {
        status: 'complete',
        translated_units: 2,
        parts: { theorem_statement: 'Hecho' },
      })),
    }));
    const complete = renderHook(() => useTranslatedDocuments([proofDoc('a')]), { wrapper });
    await advance(400);
    await flushPromises();
    await advance(VIEWING_REFRESH_MS);
    await flushPromises();
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(complete.result.current.overall.status).toBe('complete');
    await advance(VIEWING_REFRESH_MS * 3);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    complete.unmount();

    translationAPI.lookupDocuments.mockClear();
    const disabledDocuments = [proofDoc('b')];
    const disabled = renderHook(() => useTranslatedDocuments(disabledDocuments), { wrapper });
    await advance(400);
    await flushPromises();
    contextValue = makeContext({ enabled: false });
    disabled.rerender();
    await advance(VIEWING_REFRESH_MS * 3);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
    disabled.unmount();

    contextValue = makeContext();
    const unmountedDocuments = [proofDoc('c')];
    const unmounted = renderHook(() => useTranslatedDocuments(unmountedDocuments), { wrapper });
    await advance(400);
    await flushPromises();
    unmounted.unmount();
    await advance(VIEWING_REFRESH_MS * 3);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(3);
  });

  test('an older reply never overwrites a newer queue position', async () => {
    echoPending();
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    const position = () => result.current.byKey['proof:autonomous:run:a'].queuePosition;
    const reply = (queuePosition) => ({
      success: true,
      documents: [statusFor(documents[0], { queue_position: queuePosition })],
    });

    const first = deferred();
    const second = deferred();
    translationAPI.lookupDocuments.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    emitQueueChanged(1);
    await advance(800);
    emitQueueChanged(2);
    await advance(800);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(2);
    await act(async () => { second.resolve(reply(1)); });
    await flushPromises();
    expect(position()).toBe(1);
    await act(async () => { first.resolve(reply(2)); });
    await flushPromises();
    expect(position()).toBe(1);

    // A registration sent before a lookup that has already answered is older too.
    const registration = deferred();
    translationAPI.registerDocuments.mockReturnValueOnce(registration.promise);
    emitSocket('connected');
    await advance(400);
    translationAPI.lookupDocuments.mockResolvedValueOnce(reply(0));
    emitQueueChanged(3);
    await advance(800);
    await flushPromises();
    expect(position()).toBe(0);
    await act(async () => { registration.resolve(reply(3)); });
    await flushPromises();
    expect(position()).toBe(0);
  });

  test('re-registers documents whose registration failed on a doubling backoff until it succeeds', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.registerDocuments.mockRejectedValue(new Error('HTTP 500'));
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'error', docHash: null });

    let calls = 1;
    for (const delay of [5000, 10000, 20000, 40000, 60000, 60000]) {
      await advance(delay - 1);
      expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(calls);
      await advance(1);
      await flushPromises();
      calls += 1;
      expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(calls);
    }

    echoPending();
    await advance(60000);
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({
      status: 'pending',
      docHash: 'hash-proof:autonomous:run:a',
    });
    await advance(VIEWING_REFRESH_MS * 2);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(calls + 1);
  });

  test('a progress or queue event re-registers only the documents whose registration failed', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const documents = [proofDoc('a'), proofDoc('b')];
    // The reply names only the first document; the second was never registered.
    translationAPI.registerDocuments
      .mockResolvedValueOnce({ success: true, documents: [statusFor(documents[0])] })
      .mockRejectedValueOnce(new Error('HTTP 500'));
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:b']).toMatchObject({ status: 'error', docHash: null });

    await advance(5000);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
    expect(translationAPI.registerDocuments.mock.calls[1][0].documents.map((document) => document.docKey))
      .toEqual(['proof:autonomous:run:b']);

    // The next backoff retry is 10 s away; an event of any language brings it forward,
    // but no sooner than 5 s after the last failure.
    echoPending();
    await advance(600);
    emitProgress({ language_key: 'fr', doc_hashes: ['someone-else'] });
    await advance(4399);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(2);
    await advance(1);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(3);
    expect(translationAPI.registerDocuments.mock.calls[2][0].documents.map((document) => document.docKey))
      .toEqual(['proof:autonomous:run:b']);
    expect(result.current.byKey['proof:autonomous:run:b']).toMatchObject({ status: 'pending' });

    emitQueueChanged(1);
    await advance(60000);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(3);
  });

  test('never retries a failed registration while translation is disabled', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.registerDocuments.mockRejectedValue(new Error('HTTP 500'));
    const documents = [proofDoc('a')];
    const { rerender } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);

    contextValue = makeContext({ enabled: false });
    rerender();
    emitQueueChanged(1);
    await advance(120000);
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).not.toHaveBeenCalled();
  });

  test('reports error without throwing when the API fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.registerDocuments.mockRejectedValue(new Error('backend unavailable'));
    const documents = [proofDoc('a'), proofDoc('b')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });

    await advance(400);
    await flushPromises();

    expect(result.current.byKey['proof:autonomous:run:a']).toMatchObject({ status: 'error', parts: null });
    expect(result.current.overall.status).toBe('error');
  });

  test('request saves the unfinished documents in the chosen language and hides once requested', async () => {
    const documents = [proofDoc('done'), proofDoc('waiting')];
    translationAPI.registerDocuments.mockImplementation(async ({ documents: sent, request }) => ({
      success: true,
      documents: sent.map((document) => (document.docKey === 'proof:autonomous:run:done'
        ? statusFor(document, { status: 'complete', translated_units: 2, queue_position: null, parts: {} })
        : statusFor(document, { requested: Boolean(request) }))),
    }));
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(400);
    await flushPromises();
    expect(result.current.overall.requestable).toBe(true);

    await act(async () => {
      await result.current.request();
    });
    const call = translationAPI.registerDocuments.mock.calls.at(-1)[0];
    expect(call).toMatchObject({ language: 'es', viewing: true, request: true });
    expect(call.documents.map((document) => document.docKey)).toEqual(['proof:autonomous:run:waiting']);
    expect(result.current.byKey['proof:autonomous:run:waiting'].requested).toBe(true);
    expect(result.current.overall.requestable).toBe(false);
  });

  test('a document a current run translates is not requestable', async () => {
    translationAPI.registerDocuments.mockImplementation(async ({ documents }) => ({
      success: true,
      documents: documents.map((document) => statusFor(document, { automatic: true })),
    }));
    const { result } = renderHook(() => useTranslatedDocuments([proofDoc('a')]), { wrapper });
    await advance(400);
    await flushPromises();
    expect(result.current.byKey['proof:autonomous:run:a'].automatic).toBe(true);
    expect(result.current.overall.requestable).toBe(false);
  });

  test('while the original is shown, request asks for every selected language without registering for viewing', async () => {
    contextValue = makeContext({ displayLanguage: 'original' });
    translationAPI.registerDocuments.mockResolvedValue({ success: true, documents: [] });
    const { result } = renderHook(() => useTranslatedDocuments([proofDoc('a'), proofDoc('blank', {})]), { wrapper });
    await advance(400);
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    expect(result.current.overall.requestable).toBe(true);

    await act(async () => {
      await result.current.request();
    });
    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);
    const call = translationAPI.registerDocuments.mock.calls[0][0];
    expect(call).toMatchObject({ language: null, viewing: false, request: true });
    expect(call.documents.map((document) => document.docKey)).toEqual(['proof:autonomous:run:a']);
  });

  test('translationAPI sends the backend document and lookup contracts', async () => {
    const { translationAPI: realTranslationAPI } = await vi.importActual('../services/api');
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({ success: true, documents: [] }) }));
    vi.stubGlobal('fetch', fetchMock);
    try {
      await realTranslationAPI.registerDocuments({
        language: 'es',
        documents: [{
          docKey: 'paper:s1:p1',
          kind: 'paper',
          title: 'Paper',
          profile: 'document',
          parts: { content: 'Body', summary: '  ', note: 7 },
        }],
      });
      await realTranslationAPI.lookupDocuments({ language: 'es', docHashes: ['h1'] });
      await realTranslationAPI.clearCache();
      await realTranslationAPI.lookupDocuments({ language: 'es', docHashes: ['h2'], viewing: true });
      await realTranslationAPI.registerDocuments({
        language: null,
        viewing: false,
        request: true,
        documents: [{ docKey: 'paper:s1:p1', kind: 'paper', parts: { content: 'Body' } }],
      });
      await realTranslationAPI.documentProgress({
        documents: [{ docKey: 'paper:s1:p1', kind: 'paper', title: 'Paper', parts: { content: 'Body' } }],
      });
    } finally {
      vi.unstubAllGlobals();
    }

    // Progress is read-only: it sends documents only, with no language, viewing, or request.
    const [progressUrl, progressOptions] = fetchMock.mock.calls[5];
    expect(progressUrl).toMatch(/\/translation\/documents\/progress$/);
    expect(progressOptions).toMatchObject({ method: 'POST', cache: 'no-store' });
    expect(JSON.parse(progressOptions.body)).toEqual({
      documents: [{
        doc_key: 'paper:s1:p1',
        kind: 'paper',
        title: 'Paper',
        profile: 'document',
        parts: [{ name: 'content', text: 'Body' }],
      }],
    });
    // A plain registration leaves `request` out, so backends without it still accept it.
    const requested = JSON.parse(fetchMock.mock.calls[4][1].body);
    expect(requested).toMatchObject({ language: null, viewing: false, request: true });
    const [registerUrl, registerOptions] = fetchMock.mock.calls[0];
    expect(registerUrl).toMatch(/\/translation\/documents$/);
    expect(registerOptions).toMatchObject({ method: 'POST', cache: 'no-store' });
    expect(JSON.parse(registerOptions.body)).toEqual({
      language: 'es',
      viewing: true,
      documents: [{
        doc_key: 'paper:s1:p1',
        kind: 'paper',
        title: 'Paper',
        profile: 'document',
        parts: [{ name: 'content', text: 'Body' }],
      }],
    });
    const [lookupUrl, lookupOptions] = fetchMock.mock.calls[1];
    expect(lookupUrl).toMatch(/\/translation\/documents\/lookup$/);
    // Lookups refresh viewing priority only when asked to.
    expect(JSON.parse(lookupOptions.body)).toEqual({ language: 'es', doc_hashes: ['h1'], viewing: false });
    const [clearUrl, clearOptions] = fetchMock.mock.calls[2];
    expect(clearUrl).toMatch(/\/translation\/cache\?confirm=true$/);
    expect(clearOptions).toMatchObject({ method: 'DELETE', cache: 'no-store' });
    expect(JSON.parse(fetchMock.mock.calls[3][1].body)).toEqual({ language: 'es', doc_hashes: ['h2'], viewing: true });
  });

  test('registers each document in its own chosen language and leaves original ones unregistered', async () => {
    echoPending();
    const [a, b, c] = [proofDoc('a'), proofDoc('b'), proofDoc('c')];
    contextValue = makeContext({
      setDocumentLanguages: vi.fn(),
      documentLanguages: { [b.docKey]: 'fr', [c.docKey]: 'original' },
    });
    const documents = [a, b, c];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(500);
    await flushPromises();

    const calls = translationAPI.registerDocuments.mock.calls.map(([body]) => [
      body.language,
      body.documents.map((document) => document.docKey),
    ]);
    expect(calls).toEqual([['es', [a.docKey]], ['fr', [b.docKey]]]);
    expect(result.current.forKey(a.docKey)).toMatchObject({ active: true, language: 'es', selectedLanguage: 'es' });
    expect(result.current.forKey(b.docKey)).toMatchObject({ active: true, language: 'fr', selectedLanguage: 'fr' });
    expect(result.current.forKey(c.docKey)).toMatchObject({ active: false, selectedLanguage: 'original' });
    expect(result.current.byKey[c.docKey].status).toBe('original');
    // Mixed languages have no single language for the whole view.
    expect(result.current.language).toBeNull();

    result.current.forKey(c.docKey).setLanguage('es');
    expect(contextValue.setDocumentLanguages).toHaveBeenCalledWith([c.docKey], 'es');
    // A card the viewer has not loaded keeps its choice too.
    expect(result.current.forKey('proof:autonomous:run:unloaded')).toMatchObject({ selectedLanguage: 'es' });
    result.current.forKey('proof:autonomous:run:unloaded').setLanguage('fr');
    expect(contextValue.setDocumentLanguages).toHaveBeenCalledWith(['proof:autonomous:run:unloaded'], 'fr');
  });

  test('a progress event in one language looks up only the documents shown in it', async () => {
    echoPending();
    const [a, b] = [proofDoc('a'), proofDoc('b')];
    contextValue = makeContext({ documentLanguages: { [b.docKey]: 'fr' } });
    const documents = [a, b];
    renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(500);
    await flushPromises();
    translationAPI.lookupDocuments.mockResolvedValue({ success: true, documents: [] });

    emitProgress({ language_key: 'fr', doc_hashes: [`hash-${a.docKey}`, `hash-${b.docKey}`] });
    await advance(1000);
    await flushPromises();

    expect(translationAPI.lookupDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.lookupDocuments).toHaveBeenCalledWith({
      language: 'fr',
      docHashes: [`hash-${b.docKey}`],
      viewing: true,
    });
  });

  test('reads every language\'s progress without registering, so switches grey out unfinished ones', async () => {
    const document = proofDoc('a');
    translationAPI.documentProgress.mockResolvedValue({
      success: true,
      documents: [{
        doc_key: document.docKey,
        doc_hash: 'hash-a',
        languages: [
          { language: 'es', status: 'complete', translated_units: 2, total_units: 2 },
          { language: 'fr', status: 'pending', translated_units: 1, total_units: 4 },
        ],
      }],
    });
    contextValue = makeContext({ displayLanguage: 'original' });
    const documents = [document];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(500);
    await flushPromises();

    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    expect(translationAPI.documentProgress).toHaveBeenCalledWith({
      documents: [expect.objectContaining({ docKey: document.docKey, parts: document.parts })],
    });
    expect(result.current.forKey(document.docKey).languageOptions).toEqual([
      { key: 'es', name: 'Spanish', ready: true, status: 'complete', translated: 2, total: 2 },
      { key: 'fr', name: 'French', ready: false, status: 'pending', translated: 1, total: 4 },
    ]);

    // A finished part for this content refreshes the progress; other documents' parts do not.
    emitProgress({ language_key: 'fr', doc_hashes: ['hash-other'] });
    await advance(1000);
    expect(translationAPI.documentProgress).toHaveBeenCalledTimes(1);
    emitProgress({ language_key: 'fr', doc_hashes: ['hash-a'] });
    await advance(1000);
    await flushPromises();
    expect(translationAPI.documentProgress).toHaveBeenCalledTimes(2);
  });

  test('never reads progress while translation is off, and greys out every language', async () => {
    contextValue = makeContext({ enabled: false });
    const documents = [proofDoc('a')];
    const { result } = renderHook(() => useTranslatedDocuments(documents), { wrapper });
    await advance(5000);

    expect(translationAPI.documentProgress).not.toHaveBeenCalled();
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    expect(result.current.forKey('proof:autonomous:run:a').languageOptions.map((option) => [option.key, option.status]))
      .toEqual([['es', 'disabled'], ['fr', 'disabled']]);
  });

  test('fingerprints text with 32-bit FNV-1a', () => {
    expect(fnv1aHash('')).toBe('811c9dc5');
    expect(fnv1aHash('a')).toBe('e40c292c');
    expect(fnv1aHash('foobar')).toBe('bf9cf968');
  });
});
