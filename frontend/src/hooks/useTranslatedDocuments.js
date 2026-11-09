import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { translationAPI } from '../services/api';
import { subscribeTranslationSocket } from '../components/translation/AutoTranslateProvider';
import { ORIGINAL_LANGUAGE, useAutoTranslate } from './useAutoTranslate';

// Viewer events forwarded by AutoTranslateProvider.subscribeProgress.
const QUEUE_CHANGED_EVENT = 'translation_queue_changed';

export const REGISTER_DEBOUNCE_MS = 400;
export const PROGRESS_LOOKUP_DELAY_MS = 800;
// The backend keeps a document's viewing priority for 600 s after a viewing request.
export const VIEWING_REFRESH_MS = 240000;
// Documents whose registration failed are registered again on this doubling backoff.
export const REGISTRATION_RETRY_BASE_MS = 5000;
export const REGISTRATION_RETRY_MAX_MS = 60000;
export const MAX_DOCUMENTS_PER_REQUEST = 100;
// The backend accepts at most 4,000,000 characters per registration request.
const MAX_TEXT_CHARS_PER_REQUEST = 3500000;
const MAX_HASHES_PER_LOOKUP = 500;

const READY_STATUSES = new Set(['complete', 'empty']);
const BACKEND_STATUS_MAP = {
  complete: 'complete',
  pending: 'pending',
  empty: 'empty',
  unavailable: 'unavailable',
  disabled: 'disabled',
  // The backend no longer knows this document (cache cleared or restarted):
  // re-register it and keep showing the original meanwhile.
  unknown: 'loading',
};
const AGGREGATE_PRIORITY = ['error', 'pending', 'loading', 'disabled', 'unavailable'];
// How far each language is for a viewer's documents, from the read-only progress route.
const LANGUAGE_PROGRESS_STATUSES = new Set(['complete', 'pending', 'unknown', 'disabled']);

const FNV_OFFSET_BASIS = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

function fnv1aUpdate(hash, text) {
  let next = hash;
  for (let index = 0; index < text.length; index += 1) {
    next ^= text.charCodeAt(index);
    next = Math.imul(next, FNV_PRIME);
  }
  return next;
}

// Cheap 32-bit FNV-1a content fingerprint; identity only, never security.
export function fnv1aHash(text) {
  return (fnv1aUpdate(FNV_OFFSET_BASIS, String(text ?? '')) >>> 0).toString(16).padStart(8, '0');
}

function normalizeDocuments(documents) {
  if (!Array.isArray(documents)) {
    return [];
  }
  const seen = new Set();
  const normalized = [];
  documents.forEach((document) => {
    const docKey = typeof document?.docKey === 'string' ? document.docKey.trim() : '';
    if (!docKey || seen.has(docKey)) {
      return;
    }
    seen.add(docKey);
    const kind = typeof document.kind === 'string' && document.kind ? document.kind : 'other';
    const profile = document.profile === 'fields' ? 'fields' : 'document';
    const parts = {};
    const sendParts = {};
    let hash = fnv1aUpdate(FNV_OFFSET_BASIS, `${kind}\u0001${profile}`);
    Object.keys(document.parts || {}).sort().forEach((name) => {
      const text = document.parts[name];
      if (typeof text !== 'string') {
        return;
      }
      parts[name] = text;
      if (text.trim()) {
        sendParts[name] = text;
        hash = fnv1aUpdate(fnv1aUpdate(hash, `\u0002${name}\u0003`), text);
      }
    });
    normalized.push({
      docKey,
      kind,
      profile,
      title: typeof document.title === 'string' ? document.title : '',
      parts,
      sendParts,
      hasText: Object.keys(sendParts).length > 0,
      contentHash: (hash >>> 0).toString(16).padStart(8, '0'),
    });
  });
  return normalized;
}

function chunkDocuments(documents) {
  const chunks = [];
  let current = [];
  let currentChars = 0;
  documents.forEach((document) => {
    const chars = Object.values(document.sendParts).reduce((total, text) => total + text.length, 0);
    if (current.length && (
      current.length >= MAX_DOCUMENTS_PER_REQUEST
      || currentChars + chars > MAX_TEXT_CHARS_PER_REQUEST
    )) {
      chunks.push(current);
      current = [];
      currentChars = 0;
    }
    current.push(document);
    currentChars += chars;
  });
  if (current.length) {
    chunks.push(current);
  }
  return chunks;
}

// Documents grouped by the language they are shown in, in first-seen order.
function groupByLanguage(documents) {
  const groups = new Map();
  documents.forEach((document) => {
    const language = document.language || null;
    if (!groups.has(language)) {
      groups.set(language, []);
    }
    groups.get(language).push(document);
  });
  return groups;
}

function toCount(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

// 0 = translating now, n = n-th in line; anything else means no queue position.
function toQueuePosition(value) {
  return Number.isInteger(value) && value >= 0 ? value : null;
}

function stringParts(rawParts) {
  if (!rawParts || typeof rawParts !== 'object' || Array.isArray(rawParts)) {
    return {};
  }
  return Object.fromEntries(Object.entries(rawParts).filter(([, text]) => typeof text === 'string'));
}

function toResultEntry(document, item, language) {
  const status = BACKEND_STATUS_MAP[String(item?.status || '')] || 'error';
  return {
    language,
    contentHash: document.contentHash,
    status,
    translated: toCount(item?.translated_units),
    total: toCount(item?.total_units),
    queuePosition: status === 'pending' ? toQueuePosition(item?.queue_position) : null,
    // Untranslatable parts (and `empty` documents) keep their original text.
    parts: READY_STATUSES.has(status) ? { ...document.parts, ...stringParts(item?.parts) } : null,
    docHash: typeof item?.doc_hash === 'string' && item.doc_hash ? item.doc_hash : null,
    // The user requested it, or a checked mode's current run translates it automatically.
    requested: item?.requested === true,
    automatic: item?.automatic === true,
  };
}

function errorEntry(document, language, previous) {
  const same = previous?.language === language && previous?.contentHash === document.contentHash;
  return {
    language,
    contentHash: document.contentHash,
    status: 'error',
    translated: 0,
    total: 0,
    queuePosition: null,
    parts: null,
    docHash: same ? previous.docHash : null,
    requested: same ? Boolean(previous.requested) : false,
    automatic: same ? Boolean(previous.automatic) : false,
  };
}

function toRegistrationDocument(document) {
  return {
    docKey: document.docKey,
    kind: document.kind,
    title: document.title,
    profile: document.profile,
    parts: document.sendParts,
  };
}

function toLanguageProgress(row) {
  const languages = {};
  (Array.isArray(row?.languages) ? row.languages : []).forEach((item) => {
    if (typeof item?.language !== 'string' || !item.language) {
      return;
    }
    const status = LANGUAGE_PROGRESS_STATUSES.has(item.status) ? item.status : 'unknown';
    languages[item.language] = {
      status,
      translated: toCount(item.translated_units),
      total: toCount(item.total_units),
    };
  });
  return languages;
}

// A pending document only a viewer keeps translating: neither requested nor in a current run.
function isRequestable(entry) {
  return entry.status === 'pending' && !entry.requested && !entry.automatic;
}

// The backend answers registrations in request order. Each row is checked against its
// document's key; a row naming another document of this request with identical content
// still belongs here, because identical content shares one backend record whose key
// can be either document's. Anything else falls back to an exact key match.
function registrationRowFor(chunk, rows, index, rowsByKey) {
  const document = chunk[index];
  const row = rows[index];
  if (row && typeof row === 'object') {
    if (row.doc_key === document.docKey) {
      return row;
    }
    const named = chunk.find((candidate) => candidate.docKey === row.doc_key);
    if (named && named.contentHash === document.contentHash) {
      return row;
    }
  }
  return rowsByKey.get(document.docKey) || null;
}

// Lookups are answered in request order too; each row is checked against its hash.
function lookupRowFor(hashes, rows, index, rowsByHash) {
  const row = rows[index];
  if (row && typeof row === 'object' && row.doc_hash === hashes[index]) {
    return row;
  }
  return rowsByHash.get(hashes[index]) || null;
}

// An entry describes a document as it is shown now: the same content in its chosen language.
function entryMatches(entry, document) {
  return Boolean(
    entry
    && document.language
    && entry.language === document.language
    && entry.contentHash === document.contentHash,
  );
}

// The backend hash a lookup can refresh: known once registered, until the document is ready.
function unfinishedHash(entry, document) {
  return entryMatches(entry, document) && entry.docHash && !READY_STATUSES.has(entry.status)
    ? entry.docHash
    : null;
}

// Registration failed before the backend named the document, so lookups and progress cannot reach it.
function isUnregisteredError(entry, document) {
  return entryMatches(entry, document) && entry.status === 'error' && !entry.docHash;
}

function hasUnregisteredErrors(documents, results) {
  return documents.some((document) => isUnregisteredError(results[document.docKey], document));
}

function hasUnfinishedDocuments(documents, results) {
  return documents.some((document) => {
    const entry = results[document.docKey];
    return document.hasText && document.language && !(entryMatches(entry, document) && READY_STATUSES.has(entry.status));
  });
}

function lookupKey(language, docHash) {
  return `${language}\u0001${docHash}`;
}

function aggregateStatus(entries) {
  if (!entries.length) {
    return ORIGINAL_LANGUAGE;
  }
  const statuses = new Set(entries.map((entry) => entry.status));
  if ([...statuses].every((status) => READY_STATUSES.has(status))) {
    return 'complete';
  }
  return AGGREGATE_PRIORITY.find((status) => statuses.has(status)) || 'pending';
}

// The pending document closest to translation: 0 if any is translating now.
function aggregateQueuePosition(entries) {
  const positions = entries
    .filter((entry) => entry.status === 'pending' && entry.queuePosition !== null)
    .map((entry) => entry.queuePosition);
  return positions.length ? Math.min(...positions) : null;
}

const ORIGINAL_ENTRY = Object.freeze({
  status: ORIGINAL_LANGUAGE,
  translated: 0,
  total: 0,
  queuePosition: null,
  parts: null,
  docHash: null,
  requested: false,
  automatic: false,
});

// The single language every listed choice agrees on, or null when they differ.
function commonValue(values) {
  if (!values.length) {
    return null;
  }
  return values.every((value) => value === values[0]) ? values[0] : null;
}

/**
 * Translated parts for the documents a viewer shows, each in its own language.
 *
 * documents: [{ docKey, kind, title, profile: 'document' | 'fields', parts: { name: text } }]
 * A document is shown in the language its viewer's switch chose for it, otherwise in the
 * default display language set in the Auto Top-p Translate window.
 *
 * Returns a view of every document: { available, active, language, languageName,
 * selectedLanguage, languageOptions, byKey, overall, request, setLanguage, forKey }.
 * `forKey(docKey)` returns the same view for one document, for per-item switches.
 * byKey entries: { status, translated, total, queuePosition, parts, docHash, requested,
 * automatic, language }. `translated`/`total` count units. `queuePosition` is set only for
 * pending documents: 0 = translating now, n = n-th in line, null = no position. Only
 * `complete` and `empty` entries carry parts; every other status means "show the original".
 * `language` / `languageName` name the language every document of the view is shown in
 * (null when they differ or the original is shown); `selectedLanguage` is the switch value.
 * `languageOptions` lists every selected language with `ready` (finished for every document
 * of the view) and progress, so switches can grey out unfinished languages. `request()`
 * saves the user's request for the view's documents (in their chosen language, or every
 * selected language while the original is shown); `setLanguage(key)` chooses the language
 * for the view's documents only.
 * While any document is unfinished its viewing priority is refreshed at least every
 * VIEWING_REFRESH_MS; documents whose registration failed are registered again on the next
 * progress or queue event and on a doubling backoff. Nothing is sent while translation is
 * off; while it is on, documents shown in the original are only read for progress.
 */
export default function useTranslatedDocuments(documents, { enabled = true } = {}) {
  const {
    available,
    enabled: translationEnabled,
    languages,
    displayLanguage,
    documentLanguages,
    setDocumentLanguages,
    subscribeProgress,
    status: serviceStatus,
  } = useAutoTranslate();
  const baseDocs = useMemo(() => normalizeDocuments(documents), [documents]);
  const languageList = useMemo(() => (Array.isArray(languages) ? languages : []), [languages]);
  const viewerEnabled = Boolean(translationEnabled && enabled !== false);

  // A document's chosen language: its viewer's choice, else the default display language.
  const chooseLanguage = useCallback((docKey) => {
    const choice = documentLanguages?.[docKey] ?? displayLanguage;
    return viewerEnabled && choice && choice !== ORIGINAL_LANGUAGE
      ? languageList.find((language) => language?.key === choice) || null
      : null;
  }, [documentLanguages, displayLanguage, viewerEnabled, languageList]);
  const docs = useMemo(() => baseDocs.map((document) => {
    const entry = chooseLanguage(document.docKey);
    return {
      ...document,
      language: entry ? entry.key : null,
      languageName: entry ? (entry.name || entry.key) : '',
    };
  }), [baseDocs, chooseLanguage]);
  const signature = useMemo(
    () => docs.map((document) => (
      `${document.docKey}\u0001${document.kind}\u0001${document.contentHash}\u0001${document.language || ''}`
    )).join('\n'),
    [docs],
  );
  const contentSignature = useMemo(
    () => baseDocs.map((document) => `${document.docKey}\u0001${document.kind}\u0001${document.contentHash}`).join('\n'),
    [baseDocs],
  );
  const active = viewerEnabled && docs.some((document) => document.language);
  const progressActive = viewerEnabled && baseDocs.some((document) => document.hasText);

  const [results, setResults] = useState({});
  const [languageProgress, setLanguageProgress] = useState({});
  const resultsRef = useRef(results);
  const languageProgressRef = useRef(languageProgress);
  const docsRef = useRef(docs);
  const activeRef = useRef(active);
  const progressActiveRef = useRef(progressActive);
  const generationRef = useRef(0);
  const progressGenerationRef = useRef(0);
  // Every registration and lookup request gets the next number; a document keeps the
  // number of the reply it shows, so an older reply never overwrites a newer one.
  const requestSequenceRef = useRef(0);
  const registerTimerRef = useRef(null);
  const lookupTimerRef = useRef(null);
  const progressTimerRef = useRef(null);
  const viewingTimerRef = useRef(null);
  const retryTimerRef = useRef(null);
  const retryDueAtRef = useRef(0);
  const retryAttemptsRef = useRef(0);
  const retryNotBeforeRef = useRef(0);
  // Language-qualified hashes (lookupKey) progress events named since the last lookup.
  const pendingLookupKeysRef = useRef(new Set());
  const lookupAllRef = useRef(false);
  const lookupNowRef = useRef(null);
  const registerNowRef = useRef(null);
  const progressNowRef = useRef(null);
  const registrationsRef = useRef(new Set());
  const serviceGeneration = Number.isInteger(serviceStatus?.generation) ? serviceStatus.generation : null;
  const seenGenerationRef = useRef(serviceGeneration);

  // Latest-value refs for timers and event handlers; declared before the effects that read them.
  useEffect(() => {
    docsRef.current = docs;
    activeRef.current = active;
    progressActiveRef.current = progressActive;
  });

  const clearViewingRefresh = useCallback(() => {
    if (viewingTimerRef.current) {
      clearTimeout(viewingTimerRef.current);
      viewingTimerRef.current = null;
    }
  }, []);

  // While a displayed document is unfinished, a lookup of every unfinished document keeps
  // its viewing priority at most VIEWING_REFRESH_MS old. `restart` marks a request that
  // just refreshed every unfinished document.
  const syncViewingRefresh = useCallback(({ restart = false } = {}) => {
    const unfinished = activeRef.current && hasUnfinishedDocuments(docsRef.current, resultsRef.current);
    if (restart || !unfinished) {
      clearViewingRefresh();
    }
    if (!unfinished || viewingTimerRef.current) {
      return;
    }
    viewingTimerRef.current = setTimeout(() => {
      viewingTimerRef.current = null;
      lookupAllRef.current = true;
      lookupNowRef.current?.();
    }, VIEWING_REFRESH_MS);
  }, [clearViewingRefresh]);

  const applyResults = useCallback((updates, sequence) => {
    const currentKeys = new Set(docsRef.current.map((document) => document.docKey));
    const next = {};
    Object.entries(resultsRef.current).forEach(([docKey, entry]) => {
      if (currentKeys.has(docKey)) {
        next[docKey] = entry;
      }
    });
    Object.entries(updates).forEach(([docKey, entry]) => {
      const previous = next[docKey];
      if (!previous || previous.sequence <= sequence) {
        next[docKey] = { ...entry, sequence };
      }
    });
    resultsRef.current = next;
    setResults(next);
    syncViewingRefresh();
  }, [syncViewingRefresh]);

  const clearRetryRegistration = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  // Re-register failed documents after `delay` unless a retry is already due sooner.
  const scheduleRetryRegistration = useCallback((delay) => {
    const dueAt = Date.now() + delay;
    if (retryTimerRef.current && retryDueAtRef.current <= dueAt) {
      return;
    }
    clearRetryRegistration();
    retryDueAtRef.current = dueAt;
    retryTimerRef.current = setTimeout(() => {
      retryTimerRef.current = null;
      registerNowRef.current?.({ failedOnly: true });
    }, delay);
  }, [clearRetryRegistration]);

  // Coalesce lookups (progress and queue-change bursts, post-registration follow-ups) into one
  // per delay window.
  const scheduleLookup = useCallback(() => {
    if (lookupTimerRef.current) {
      return;
    }
    lookupTimerRef.current = setTimeout(() => {
      lookupTimerRef.current = null;
      lookupNowRef.current?.();
    }, PROGRESS_LOOKUP_DELAY_MS);
  }, []);

  // Coalesce reads of every language's progress into one per delay window.
  const scheduleProgress = useCallback((delay = PROGRESS_LOOKUP_DELAY_MS) => {
    if (progressTimerRef.current) {
      return;
    }
    progressTimerRef.current = setTimeout(() => {
      progressTimerRef.current = null;
      progressNowRef.current?.();
    }, delay);
  }, []);

  // Registers every document shown in a language, or with `failedOnly` just those whose
  // registration failed. Each language's documents go in their own requests.
  const registerNow = useCallback(async ({ failedOnly = false } = {}) => {
    const generation = generationRef.current;
    if (!activeRef.current) {
      return;
    }
    const documents = docsRef.current.filter((document) => document.hasText && document.language && (
      !failedOnly || isUnregisteredError(resultsRef.current[document.docKey], document)
    ));
    if (!documents.length) {
      return;
    }
    // Progress or queue changes reported while this registration is in flight can postdate the
    // replies the backend already computed, so they trigger a lookup once those are applied.
    const registration = { languages: new Set(documents.map((document) => document.language)), progressSeen: false };
    registrationsRef.current.add(registration);
    if (!failedOnly) {
      syncViewingRefresh({ restart: true });
    }
    try {
      for (const [requestLanguage, group] of groupByLanguage(documents)) {
        for (const chunk of chunkDocuments(group)) {
          if (generation !== generationRef.current) {
            return;
          }
          requestSequenceRef.current += 1;
          const sequence = requestSequenceRef.current;
          try {
            const response = await translationAPI.registerDocuments({
              language: requestLanguage,
              viewing: true,
              documents: chunk.map(toRegistrationDocument),
            });
            if (generation !== generationRef.current) {
              return;
            }
            const rows = Array.isArray(response?.documents) ? response.documents : [];
            const rowsByKey = new Map(rows.map((item) => [item?.doc_key, item]));
            applyResults(Object.fromEntries(chunk.map((document, index) => {
              const row = registrationRowFor(chunk, rows, index, rowsByKey);
              return [
                document.docKey,
                row
                  ? toResultEntry(document, row, requestLanguage)
                  : errorEntry(document, requestLanguage, resultsRef.current[document.docKey]),
              ];
            })), sequence);
          } catch (error) {
            if (generation !== generationRef.current) {
              return;
            }
            console.warn('Translation registration failed:', error?.message || error);
            applyResults(Object.fromEntries(chunk.map((document) => [
              document.docKey,
              errorEntry(document, requestLanguage, resultsRef.current[document.docKey]),
            ])), sequence);
          }
        }
      }
      if (generation !== generationRef.current) {
        return;
      }
      if (registration.progressSeen) {
        lookupAllRef.current = true;
        scheduleLookup();
      }
      if (hasUnregisteredErrors(docsRef.current, resultsRef.current)) {
        const delay = Math.min(
          REGISTRATION_RETRY_BASE_MS * (2 ** retryAttemptsRef.current),
          REGISTRATION_RETRY_MAX_MS,
        );
        retryAttemptsRef.current += 1;
        retryNotBeforeRef.current = Date.now() + REGISTRATION_RETRY_BASE_MS;
        scheduleRetryRegistration(delay);
      } else {
        retryAttemptsRef.current = 0;
        clearRetryRegistration();
      }
    } finally {
      registrationsRef.current.delete(registration);
    }
  }, [applyResults, clearRetryRegistration, scheduleLookup, scheduleRetryRegistration, syncViewingRefresh]);

  // Saves the user's request for the given documents (all by default) not yet translated: in
  // each document's chosen language, or in every selected language while it shows the
  // original. Rejects when the backend could not save it, so the caller can say so.
  const request = useCallback(async (docKeys = null) => {
    const generation = generationRef.current;
    const wanted = Array.isArray(docKeys) ? new Set(docKeys) : null;
    const documents = docsRef.current.filter((document) => {
      const entry = resultsRef.current[document.docKey];
      return document.hasText
        && (!wanted || wanted.has(document.docKey))
        && !(entryMatches(entry, document) && READY_STATUSES.has(entry.status));
    });
    for (const [requestLanguage, group] of groupByLanguage(documents)) {
      for (const chunk of chunkDocuments(group)) {
        requestSequenceRef.current += 1;
        const sequence = requestSequenceRef.current;
        const response = await translationAPI.registerDocuments({
          language: requestLanguage,
          viewing: Boolean(requestLanguage),
          request: true,
          documents: chunk.map(toRegistrationDocument),
        });
        if (!requestLanguage || generation !== generationRef.current) {
          continue;
        }
        const rows = Array.isArray(response?.documents) ? response.documents : [];
        const rowsByKey = new Map(rows.map((item) => [item?.doc_key, item]));
        const updates = {};
        chunk.forEach((document, index) => {
          const row = registrationRowFor(chunk, rows, index, rowsByKey);
          if (row) {
            updates[document.docKey] = toResultEntry(document, row, requestLanguage);
          }
        });
        applyResults(updates, sequence);
      }
    }
  }, [applyResults]);

  const scheduleRegister = useCallback(() => {
    if (registerTimerRef.current) {
      clearTimeout(registerTimerRef.current);
    }
    registerTimerRef.current = setTimeout(() => {
      registerTimerRef.current = null;
      registerNow();
    }, REGISTER_DEBOUNCE_MS);
  }, [registerNow]);

  const lookupNow = useCallback(async () => {
    const generation = generationRef.current;
    const wantedKeys = pendingLookupKeysRef.current;
    const lookupAll = lookupAllRef.current;
    pendingLookupKeysRef.current = new Set();
    lookupAllRef.current = false;
    if (!activeRef.current) {
      return;
    }
    // language -> hash -> documents shown with that content in that language.
    const byLanguage = new Map();
    docsRef.current.forEach((document) => {
      const docHash = unfinishedHash(resultsRef.current[document.docKey], document);
      if (!docHash || !(lookupAll || wantedKeys.has(lookupKey(document.language, docHash)))) {
        return;
      }
      if (!byLanguage.has(document.language)) {
        byLanguage.set(document.language, new Map());
      }
      const docsByHash = byLanguage.get(document.language);
      docsByHash.set(docHash, [...(docsByHash.get(docHash) || []), document]);
    });
    if (lookupAll && byLanguage.size) {
      syncViewingRefresh({ restart: true });
    }
    for (const [requestLanguage, docsByHash] of byLanguage) {
      const hashes = [...docsByHash.keys()];
      for (let start = 0; start < hashes.length; start += MAX_HASHES_PER_LOOKUP) {
        const chunk = hashes.slice(start, start + MAX_HASHES_PER_LOOKUP);
        requestSequenceRef.current += 1;
        const sequence = requestSequenceRef.current;
        try {
          const response = await translationAPI.lookupDocuments({
            language: requestLanguage,
            docHashes: chunk,
            viewing: true,
          });
          if (generation !== generationRef.current) {
            return;
          }
          const updates = {};
          let needsRegistration = false;
          const rows = Array.isArray(response?.documents) ? response.documents : [];
          const rowsByHash = new Map(rows.map((item) => [item?.doc_hash, item]));
          chunk.forEach((docHash, index) => {
            const item = lookupRowFor(chunk, rows, index, rowsByHash);
            if (!item) {
              return;
            }
            (docsByHash.get(docHash) || []).forEach((document) => {
              const entry = toResultEntry(document, item, requestLanguage);
              if (item.status === 'unknown') {
                needsRegistration = true;
              }
              updates[document.docKey] = { ...entry, docHash: entry.docHash || docHash };
            });
          });
          applyResults(updates, sequence);
          if (needsRegistration) {
            scheduleRegister();
          }
        } catch (error) {
          if (generation !== generationRef.current) {
            return;
          }
          console.warn('Translation lookup failed:', error?.message || error);
          const updates = {};
          chunk.forEach((docHash) => {
            (docsByHash.get(docHash) || []).forEach((document) => {
              updates[document.docKey] = errorEntry(document, requestLanguage, resultsRef.current[document.docKey]);
            });
          });
          applyResults(updates, sequence);
        }
      }
    }
  }, [applyResults, scheduleRegister, syncViewingRefresh]);

  // Reads how far every selected language is for each document. Read-only: it never
  // registers, views, or requests anything, so documents shown in the original stay idle.
  const progressNow = useCallback(async () => {
    const generation = progressGenerationRef.current;
    if (!progressActiveRef.current) {
      return;
    }
    const documents = docsRef.current.filter((document) => document.hasText);
    for (const chunk of chunkDocuments(documents)) {
      let response;
      try {
        response = await translationAPI.documentProgress({ documents: chunk.map(toRegistrationDocument) });
      } catch {
        // Progress only greys out unfinished languages; the next refresh tries again.
        return;
      }
      if (generation !== progressGenerationRef.current) {
        return;
      }
      const rows = Array.isArray(response?.documents) ? response.documents : [];
      const rowsByKey = new Map(rows.map((item) => [item?.doc_key, item]));
      const currentKeys = new Set(docsRef.current.map((document) => document.docKey));
      const next = Object.fromEntries(
        Object.entries(languageProgressRef.current).filter(([docKey]) => currentKeys.has(docKey)),
      );
      chunk.forEach((document, index) => {
        const row = registrationRowFor(chunk, rows, index, rowsByKey);
        if (row) {
          next[document.docKey] = {
            contentHash: document.contentHash,
            docHash: typeof row.doc_hash === 'string' ? row.doc_hash : null,
            languages: toLanguageProgress(row),
          };
        }
      });
      languageProgressRef.current = next;
      setLanguageProgress(next);
    }
  }, []);

  useEffect(() => {
    lookupNowRef.current = lookupNow;
    registerNowRef.current = registerNow;
    progressNowRef.current = progressNow;
  }, [lookupNow, registerNow, progressNow]);

  useEffect(() => {
    if (!active) {
      return undefined;
    }
    generationRef.current += 1;
    scheduleRegister();
    return () => {
      // Language/document changes and unmount invalidate in-flight responses and stop every
      // timer, so nothing is sent while translation is off.
      generationRef.current += 1;
      if (registerTimerRef.current) {
        clearTimeout(registerTimerRef.current);
        registerTimerRef.current = null;
      }
      clearViewingRefresh();
      clearRetryRegistration();
      retryAttemptsRef.current = 0;
      retryNotBeforeRef.current = 0;
    };
  }, [active, signature, scheduleRegister, clearViewingRefresh, clearRetryRegistration]);

  // Language progress follows the documents' content, whichever language each one shows.
  useEffect(() => {
    if (!progressActive) {
      return undefined;
    }
    progressGenerationRef.current += 1;
    scheduleProgress(REGISTER_DEBOUNCE_MS);
    return () => {
      progressGenerationRef.current += 1;
      if (progressTimerRef.current) {
        clearTimeout(progressTimerRef.current);
        progressTimerRef.current = null;
      }
    };
  }, [progressActive, contentSignature, scheduleProgress]);

  useEffect(() => {
    if (!progressActive) {
      return undefined;
    }
    return subscribeTranslationSocket('connected', () => {
      if (activeRef.current) {
        scheduleRegister();
      }
      scheduleProgress(REGISTER_DEBOUNCE_MS);
    });
  }, [progressActive, scheduleRegister, scheduleProgress]);

  // A control action, settings save, repair, or cleared cache changes the service
  // generation and can drop stored translations, so documents shown as complete are
  // registered again to show what the service has now. Any change counts: status replies
  // arrive in order, and a restarted backend counts its generation from 0 again.
  useEffect(() => {
    if (serviceGeneration === null) {
      return;
    }
    const seen = seenGenerationRef.current;
    seenGenerationRef.current = serviceGeneration;
    if (seen === null || serviceGeneration === seen) {
      return;
    }
    if (active) {
      scheduleRegister();
    }
    if (progressActive) {
      scheduleProgress(REGISTER_DEBOUNCE_MS);
    }
  }, [active, progressActive, serviceGeneration, scheduleRegister, scheduleProgress]);

  useEffect(() => {
    if (!progressActive || typeof subscribeProgress !== 'function') {
      return undefined;
    }
    const unsubscribe = subscribeProgress((event, eventType) => {
      if (!event) {
        return;
      }
      const queueChanged = eventType === QUEUE_CHANGED_EVENT;
      const eventHashes = new Set(Array.isArray(event.doc_hashes) ? event.doc_hashes : []);
      // A finished part changes how far that language is for any of these documents.
      if (!queueChanged && Object.values(languageProgressRef.current).some((item) => eventHashes.has(item.docHash))) {
        scheduleProgress();
      }
      if (!activeRef.current) {
        return;
      }
      // Any progress or queue event shows the backend is reachable: retry failed registrations,
      // but no sooner than the backoff base after the last failure.
      if (!registrationsRef.current.size && hasUnregisteredErrors(docsRef.current, resultsRef.current)) {
        scheduleRetryRegistration(Math.max(REGISTER_DEBOUNCE_MS, retryNotBeforeRef.current - Date.now()));
      }
      const shownLanguages = new Set(docsRef.current.map((document) => document.language).filter(Boolean));
      if (!queueChanged && !shownLanguages.has(event.language_key)) {
        return;
      }
      registrationsRef.current.forEach((registration) => {
        if (queueChanged || registration.languages.has(event.language_key)) {
          registration.progressSeen = true;
        }
      });
      let matched = false;
      docsRef.current.forEach((document) => {
        const docHash = unfinishedHash(resultsRef.current[document.docKey], document);
        if (!docHash) {
          return;
        }
        if (queueChanged) {
          // Queue positions of every waiting document may have moved.
          lookupAllRef.current = true;
          matched = true;
        } else if (document.language === event.language_key && eventHashes.has(docHash)) {
          pendingLookupKeysRef.current.add(lookupKey(document.language, docHash));
          matched = true;
        }
      });
      if (matched) {
        scheduleLookup();
      }
    });
    return () => {
      if (typeof unsubscribe === 'function') {
        unsubscribe();
      }
      if (lookupTimerRef.current) {
        clearTimeout(lookupTimerRef.current);
        lookupTimerRef.current = null;
      }
      pendingLookupKeysRef.current = new Set();
      lookupAllRef.current = false;
    };
  }, [progressActive, signature, subscribeProgress, scheduleLookup, scheduleProgress, scheduleRetryRegistration]);

  return useMemo(() => {
    const byKey = {};
    docs.forEach((document) => {
      if (!document.language) {
        byKey[document.docKey] = ORIGINAL_ENTRY;
        return;
      }
      if (!document.hasText) {
        byKey[document.docKey] = {
          status: 'empty',
          translated: 0,
          total: 0,
          queuePosition: null,
          parts: { ...document.parts },
          docHash: null,
          requested: false,
          automatic: false,
          language: document.language,
        };
        return;
      }
      const entry = results[document.docKey];
      byKey[document.docKey] = entryMatches(entry, document)
        ? {
          status: entry.status,
          translated: entry.translated,
          total: entry.total,
          queuePosition: entry.queuePosition,
          parts: entry.parts,
          docHash: entry.docHash,
          requested: entry.requested,
          automatic: entry.automatic,
          language: document.language,
        }
        : {
          status: 'loading',
          translated: 0,
          total: 0,
          queuePosition: null,
          parts: null,
          docHash: null,
          requested: false,
          automatic: false,
          language: document.language,
        };
    });

    // How far one language is for one document: a fresher registration reply for the language
    // it is shown in wins over the read-only progress.
    const languageState = (document, languageKey) => {
      if (!translationEnabled) {
        return { status: 'disabled', translated: 0, total: 0 };
      }
      if (!document.hasText) {
        return { status: 'complete', translated: 0, total: 0 };
      }
      const entry = byKey[document.docKey];
      if (document.language === languageKey && READY_STATUSES.has(entry.status)) {
        return { status: 'complete', translated: entry.total, total: entry.total };
      }
      if (document.language === languageKey && entry.status === 'pending') {
        return { status: 'pending', translated: entry.translated, total: entry.total };
      }
      const progress = languageProgress[document.docKey];
      const item = progress && progress.contentHash === document.contentHash ? progress.languages[languageKey] : null;
      return item || { status: 'unknown', translated: 0, total: 0 };
    };

    // `keys` are the documents the view's switch chooses for; `viewDocs` the ones it shows.
    const buildView = (viewDocs, keys, fallbackSelected = ORIGINAL_LANGUAGE) => {
      const shown = viewDocs.filter((document) => document.language);
      const entries = shown.map((document) => byKey[document.docKey]);
      const viewLanguage = commonValue(shown.map((document) => document.language));
      const viewActive = viewerEnabled && shown.length > 0;
      const languageOptions = languageList
        .filter((language) => language && typeof language.key === 'string')
        .map((language) => {
          const states = viewDocs.map((document) => languageState(document, language.key));
          const ready = states.length > 0 && states.every((state) => state.status === 'complete');
          const statusOf = () => {
            if (!states.length) return translationEnabled ? 'unknown' : 'disabled';
            if (ready) return 'complete';
            if (states.some((state) => state.status === 'disabled')) return 'disabled';
            if (states.some((state) => state.status === 'pending')) return 'pending';
            return 'unknown';
          };
          return {
            key: language.key,
            name: language.name || language.key,
            ready,
            status: statusOf(),
            translated: states.reduce((total, state) => total + state.translated, 0),
            total: states.reduce((total, state) => total + state.total, 0),
          };
        });
      const selected = viewDocs.length
        ? commonValue(viewDocs.map((document) => document.language || ORIGINAL_LANGUAGE))
        : fallbackSelected;
      return {
        available: Boolean(available),
        active: viewActive,
        language: viewActive ? viewLanguage : null,
        languageName: viewActive && viewLanguage ? shown[0].languageName : '',
        selectedLanguage: selected ?? ORIGINAL_LANGUAGE,
        languageOptions,
        byKey,
        overall: viewActive
          ? {
            status: aggregateStatus(entries),
            translated: entries.reduce((total, entry) => total + entry.translated, 0),
            total: entries.reduce((total, entry) => total + entry.total, 0),
            queuePosition: aggregateQueuePosition(entries),
            requestable: entries.some(isRequestable),
          }
          : {
            status: ORIGINAL_LANGUAGE,
            translated: 0,
            total: 0,
            queuePosition: null,
            requestable: Boolean(
              viewerEnabled && languageList.length && viewDocs.some((document) => document.hasText),
            ),
          },
        request: () => request(keys),
        setLanguage: (languageKey) => setDocumentLanguages?.(keys, languageKey),
      };
    };

    const docsByKey = new Map(docs.map((document) => [document.docKey, document]));
    return {
      ...buildView(docs, docs.map((document) => document.docKey)),
      // A key this viewer has not loaded (a collapsed library card) still gets a switch: its
      // choice is kept and applies once the document is shown.
      forKey: (docKey) => (docsByKey.has(docKey)
        ? buildView([docsByKey.get(docKey)], [docKey])
        : buildView([], [docKey], chooseLanguage(docKey)?.key || ORIGINAL_LANGUAGE)),
    };
  }, [
    available,
    chooseLanguage,
    docs,
    languageList,
    languageProgress,
    request,
    results,
    setDocumentLanguages,
    translationEnabled,
    viewerEnabled,
  ]);
}

export { useTranslatedDocuments };
