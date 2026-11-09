import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { translationAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import {
  AutoTranslateContext,
  DISPLAY_LANGUAGE_STORAGE_KEY,
  ORIGINAL_LANGUAGE,
} from '../../hooks/useAutoTranslate';

export const STATUS_POLL_INTERVAL_MS = 30000;
// Status refreshes at most this long after the first event of a burst.
export const STATUS_REFRESH_DELAY_MS = 2000;
export const MAX_RECENT_ACTIVITY = 100;

const TRANSLATION_ACTIVITY_EVENTS = [
  'translation_attempt_accepted',
  'translation_attempt_rejected',
  'translation_attempt_kept',
  'translation_part_finalized',
  'translation_attempt_check_failed',
  'translation_validator_contract_retry',
  'translation_repair_required',
  'translation_translate_all',
];

// Each attempt changes the active part's attempt, accepted, and keep counts shown in
// status, so these also refresh status (throttled).
const STATUS_REFRESH_ACTIVITY_EVENTS = new Set([
  'translation_attempt_accepted',
  'translation_attempt_rejected',
  'translation_attempt_kept',
  'translation_part_finalized',
  'translation_attempt_check_failed',
]);

// Forwarded to subscribeProgress subscribers as (data, eventType) so viewers can
// refresh their translated units and queue positions.
const VIEWER_PROGRESS_EVENTS = ['translation_progress', 'translation_queue_changed'];

// Shared provider events count as translation activity only when they name the
// translation workflow.
const PROVIDER_ACTIVITY_EVENTS = [
  'provider_retry_cooldown',
  'hung_connection_alert',
];

// Retry waits the service starts itself: its workflow-layer backoffs (retry_layer
// "workflow") and the shared usage-limit cooldown it enters (reason usage_limit_cooldown,
// emitted at the adapter layer). Each is recorded in status recent_activity as one
// readable line carrying its route, so these refresh status rather than listing the
// same wait twice; other adapter waits show live.
function isServiceRecordedWait(eventType, data) {
  return eventType === 'provider_retry_cooldown'
    && (data.retry_layer === 'workflow' || data.reason === 'usage_limit_cooldown');
}

// websocket.on() returns an unsubscribe function; tolerate partial mocks or
// listeners that do not, and never let a subscription failure break rendering.
export function subscribeTranslationSocket(eventType, callback) {
  try {
    const unsubscribe = websocket.on(eventType, callback);
    return () => {
      try {
        if (typeof unsubscribe === 'function') {
          unsubscribe();
        } else if (typeof websocket.off === 'function') {
          websocket.off(eventType, callback);
        }
      } catch {
        // Listener already gone.
      }
    };
  } catch {
    return () => {};
  }
}

function readStoredDisplayLanguage() {
  try {
    const stored = window.localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY);
    return typeof stored === 'string' && stored.trim() ? stored.trim() : ORIGINAL_LANGUAGE;
  } catch {
    return ORIGINAL_LANGUAGE;
  }
}

function writeStoredDisplayLanguage(languageKey) {
  try {
    window.localStorage.setItem(DISPLAY_LANGUAGE_STORAGE_KEY, languageKey);
  } catch {
    // Storage unavailable (private window, quota, blocked site data): keep in memory only.
  }
}

function activityTime(item) {
  const raw = item?.timestamp ?? item?._serverTimestamp ?? item?.received_at;
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return raw < 1e12 ? raw * 1000 : raw;
  }
  const parsed = Date.parse(typeof raw === 'string' ? raw : '');
  return Number.isFinite(parsed) ? parsed : 0;
}

// Identity of one activity item: its activity_id, or for items stored without one
// (older backends) the event, part, attempt, language, and time it describes.
function activityKey(item) {
  if (item.activity_id) {
    return `id:${item.activity_id}`;
  }
  if (item.timestamp === undefined || item.timestamp === null || item.timestamp === '') {
    return null;
  }
  return [
    'event',
    item.event_type ?? '',
    item.target_hash ?? '',
    item.round_index ?? '',
    item.language_key ?? '',
    item.timestamp,
  ].join('\u0001');
}

// Newest first, de-duplicated by activity_id (or the fallback identity), bounded.
export function mergeRecentActivity(liveItems = [], statusItems = []) {
  const seen = new Set();
  const merged = [];
  [...liveItems, ...(Array.isArray(statusItems) ? statusItems : [])].forEach((item) => {
    if (!item || typeof item !== 'object') {
      return;
    }
    const key = activityKey(item);
    if (key) {
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
    }
    merged.push(item);
  });
  return merged
    .map((item, index) => ({ item, index, time: activityTime(item) }))
    .sort((left, right) => (right.time - left.time) || (left.index - right.index))
    .slice(0, MAX_RECENT_ACTIVITY)
    .map((entry) => entry.item);
}

function normalizeLanguages(rawLanguages) {
  if (!Array.isArray(rawLanguages)) {
    return [];
  }
  return rawLanguages.filter((language) => (
    language && typeof language === 'object' && typeof language.key === 'string' && language.key
  ));
}

export default function AutoTranslateProvider({ children }) {
  const [status, setStatus] = useState(null);
  const [storedLanguage, setStoredLanguage] = useState(readStoredDisplayLanguage);
  // Each viewer's language choice for its own documents; the default applies to the rest.
  const [documentLanguages, setDocumentLanguagesState] = useState({});
  const [liveActivity, setLiveActivity] = useState([]);
  const mountedRef = useRef(false);
  const requestSequenceRef = useRef(0);
  const appliedSequenceRef = useRef(0);
  const statusRefreshTimerRef = useRef(null);
  const progressSubscribersRef = useRef(new Set());

  const refreshStatus = useCallback(async () => {
    const sequence = requestSequenceRef.current + 1;
    requestSequenceRef.current = sequence;
    try {
      const nextStatus = await translationAPI.getStatus();
      if (!mountedRef.current || sequence < appliedSequenceRef.current) {
        return;
      }
      appliedSequenceRef.current = sequence;
      if (nextStatus && typeof nextStatus === 'object') {
        setStatus(nextStatus);
      }
    } catch (error) {
      // Older backends or a restarting backend: keep the last known status.
      console.warn('Auto Top-p Translate status unavailable:', error?.message || error);
    }
  }, []);

  // Viewer subscription to translation_progress and translation_queue_changed;
  // callbacks receive (data, eventType).
  const subscribeProgress = useCallback((callback) => {
    if (typeof callback !== 'function') {
      return () => {};
    }
    progressSubscribersRef.current.add(callback);
    return () => {
      progressSubscribersRef.current.delete(callback);
    };
  }, []);

  const appendLiveActivity = useCallback((eventType, data) => {
    if (!data || typeof data !== 'object') {
      return;
    }
    const item = {
      ...data,
      event_type: data.event_type || eventType,
      received_at: new Date().toISOString(),
    };
    setLiveActivity((previous) => mergeRecentActivity([item], previous));
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    refreshStatus();

    // Throttled: a pending refresh already covers later events of the burst, and the
    // first event after it fires starts the trailing refresh.
    const scheduleStatusRefresh = () => {
      if (statusRefreshTimerRef.current) {
        return;
      }
      statusRefreshTimerRef.current = setTimeout(() => {
        statusRefreshTimerRef.current = null;
        refreshStatus();
      }, STATUS_REFRESH_DELAY_MS);
    };

    const unsubscribers = [
      subscribeTranslationSocket('connected', () => { refreshStatus(); }),
      subscribeTranslationSocket('translation_status_changed', () => { refreshStatus(); }),
      ...VIEWER_PROGRESS_EVENTS.map((eventType) => subscribeTranslationSocket(eventType, (data) => {
        progressSubscribersRef.current.forEach((callback) => {
          try {
            callback(data, eventType);
          } catch (error) {
            console.warn('Translation progress subscriber failed:', error);
          }
        });
        // The order and positions of the status queue changed.
        if (eventType === 'translation_queue_changed') {
          scheduleStatusRefresh();
        }
      })),
      ...TRANSLATION_ACTIVITY_EVENTS.map((eventType) => subscribeTranslationSocket(eventType, (data) => {
        appendLiveActivity(eventType, data);
        if (STATUS_REFRESH_ACTIVITY_EVENTS.has(eventType)) {
          scheduleStatusRefresh();
        }
      })),
      ...PROVIDER_ACTIVITY_EVENTS.map((eventType) => subscribeTranslationSocket(eventType, (data) => {
        if (data?.workflow_mode !== 'translation') {
          return;
        }
        if (isServiceRecordedWait(eventType, data)) {
          scheduleStatusRefresh();
          return;
        }
        appendLiveActivity(eventType, data);
      })),
    ];

    return () => {
      mountedRef.current = false;
      unsubscribers.forEach((unsubscribe) => unsubscribe());
      if (statusRefreshTimerRef.current) {
        clearTimeout(statusRefreshTimerRef.current);
        statusRefreshTimerRef.current = null;
      }
    };
  }, [appendLiveActivity, refreshStatus]);

  const enabled = Boolean(status?.enabled);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    const intervalId = setInterval(() => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }
      refreshStatus();
    }, STATUS_POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [enabled, refreshStatus]);

  const languages = useMemo(() => normalizeLanguages(status?.languages), [status?.languages]);

  // The default stays as chosen while translation is off (viewers then show the original
  // regardless) so it can be set before translation is turned on.
  const displayLanguage = languages.some((language) => language.key === storedLanguage)
    ? storedLanguage
    : ORIGINAL_LANGUAGE;

  const setDisplayLanguage = useCallback((languageKey) => {
    const nextLanguage = typeof languageKey === 'string' && languageKey.trim()
      ? languageKey.trim()
      : ORIGINAL_LANGUAGE;
    writeStoredDisplayLanguage(nextLanguage);
    setStoredLanguage(nextLanguage);
  }, []);

  const setDocumentLanguages = useCallback((docKeys, languageKey) => {
    const keys = (Array.isArray(docKeys) ? docKeys : [docKeys])
      .filter((docKey) => typeof docKey === 'string' && docKey);
    if (!keys.length) {
      return;
    }
    const nextLanguage = typeof languageKey === 'string' && languageKey.trim()
      ? languageKey.trim()
      : ORIGINAL_LANGUAGE;
    setDocumentLanguagesState((current) => {
      if (keys.every((docKey) => current[docKey] === nextLanguage)) {
        return current;
      }
      const next = { ...current };
      keys.forEach((docKey) => {
        next[docKey] = nextLanguage;
      });
      return next;
    });
  }, []);

  const recentActivity = useMemo(
    () => mergeRecentActivity(liveActivity, status?.recent_activity),
    [liveActivity, status?.recent_activity],
  );

  // Translate all progress; older backends send no block.
  const translateAll = status?.translate_all && typeof status.translate_all === 'object'
    ? status.translate_all
    : null;

  const value = useMemo(() => ({
    available: true,
    status,
    enabled,
    languages,
    displayLanguage,
    setDisplayLanguage,
    documentLanguages,
    setDocumentLanguages,
    refreshStatus,
    subscribeProgress,
    recentActivity,
    translateAll,
  }), [
    status,
    enabled,
    languages,
    displayLanguage,
    setDisplayLanguage,
    documentLanguages,
    setDocumentLanguages,
    refreshStatus,
    subscribeProgress,
    recentActivity,
    translateAll,
  ]);

  return (
    <AutoTranslateContext.Provider value={value}>
      {children}
    </AutoTranslateContext.Provider>
  );
}
