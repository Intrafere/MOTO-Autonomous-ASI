import { formatContextOverflowActivityMessage } from './activityStyles';
import { sanitizePersistedActivityValue } from './activityPersistence';

const DISMISSED_PROVIDER_NOTIFICATION_IDS_STORAGE_KEY = 'dismissedOAuthProviderNotifications';
const DISMISSED_PROVIDER_NOTIFICATION_FINGERPRINT_PREFIX = 'dismissedOAuthProviderNotificationFingerprint:';
const MAX_DISMISSED_PROVIDER_NOTIFICATION_IDS = 500;
export const MAX_LIVE_ACTIVITY_EVENTS = 5000;
const MAX_PERSISTED_ACTIVITY_STRING_LENGTH = 2000;
const MAX_PERSISTED_ACTIVITY_ARRAY_ITEMS = 20;
const MAX_PERSISTED_ACTIVITY_OBJECT_KEYS = 60;

async function fingerprintProviderNotificationId(value) {
  const input = new TextEncoder().encode(String(value || ''));
  const digest = await window.crypto.subtle.digest('SHA-256', input);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

function readLegacyDismissedProviderNotificationIds() {
  if (typeof window === 'undefined') {
    return new Set();
  }

  try {
    const raw = window.localStorage.getItem(DISMISSED_PROVIDER_NOTIFICATION_IDS_STORAGE_KEY);
    const values = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(values) ? values.filter(value => typeof value === 'string') : []);
  } catch (error) {
    console.warn('Could not read legacy dismissed provider notification IDs:', error);
    return new Set();
  }
}

function dismissedProviderNotificationFingerprintKey(fingerprint) {
  return `${DISMISSED_PROVIDER_NOTIFICATION_FINGERPRINT_PREFIX}${fingerprint}`;
}

function trimDismissedProviderNotificationMarkers() {
  const markers = [];
  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index);
    if (key?.startsWith(DISMISSED_PROVIDER_NOTIFICATION_FINGERPRINT_PREFIX)) {
      markers.push({
        key,
        createdAt: Number(window.localStorage.getItem(key)) || 0,
      });
    }
  }
  markers
    .sort((left, right) => left.createdAt - right.createdAt)
    .slice(0, Math.max(0, markers.length - MAX_DISMISSED_PROVIDER_NOTIFICATION_IDS))
    .forEach(({ key }) => window.localStorage.removeItem(key));
}

export async function isProviderNotificationDismissed(notificationId) {
  if (typeof window === 'undefined') {
    return false;
  }

  const normalizedId = String(notificationId || '');
  const fingerprint = await fingerprintProviderNotificationId(normalizedId);
  if (window.localStorage.getItem(dismissedProviderNotificationFingerprintKey(fingerprint)) !== null) {
    return true;
  }

  const legacyIds = readLegacyDismissedProviderNotificationIds();
  return legacyIds.has(normalizedId) || legacyIds.has(fingerprint);
}

export async function persistDismissedProviderNotificationId(notificationId) {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const fingerprint = await fingerprintProviderNotificationId(notificationId);
    window.localStorage.setItem(
      dismissedProviderNotificationFingerprintKey(fingerprint),
      String(Date.now()),
    );
    trimDismissedProviderNotificationMarkers();
  } catch (error) {
    console.warn('Could not save dismissed provider notification IDs:', error);
  }
}

export function readPersistedLiveActivity(storageKey) {
  try {
    const savedEvents = localStorage.getItem(storageKey);
    if (!savedEvents) {
      return [];
    }
    const parsed = sanitizePersistedActivityValue(JSON.parse(savedEvents));
    return Array.isArray(parsed)
      ? parsed
        .filter((event) => event && typeof event === 'object')
        .filter((event) => {
          const eventName = event.event || event.type;
          const message = String(event.message || '');
          return (
            eventName !== 'proof_run_idle'
            && eventName !== 'proof_run_next_round_required'
            && event?.data?.status !== 'idle_between_rounds'
            && event?.data?.terminal_reason !== 'three_consecutive_zero_candidate_rounds'
            && !/waiting for Run Next Round|three consecutive valid.*no candidates/i.test(message)
          );
        })
        .map((event) => {
          const eventName = event.event || event.type;
          const isOverflow = eventName === 'context_overflow_error'
            || (
              (eventName === 'auto_research_stopped' || eventName === 'leanoj_stopped')
              && event?.data?.reason === 'context_overflow'
            );
          return isOverflow
            ? { ...event, message: formatContextOverflowActivityMessage(event.data || {}) }
            : event;
        })
        .slice(-MAX_LIVE_ACTIVITY_EVENTS)
      : [];
  } catch (error) {
    console.error(`Failed to load ${storageKey}:`, error);
    return [];
  }
}

export function shouldRecordWorkflowStoppedActivity(eventName, data = {}) {
  return !(
    (eventName === 'auto_research_stopped' || eventName === 'leanoj_stopped')
    && data?.reason === 'context_overflow'
  );
}

function compactPersistedActivityValue(value, depth = 0) {
  if (value == null || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    return value.length > MAX_PERSISTED_ACTIVITY_STRING_LENGTH
      ? `${value.slice(0, MAX_PERSISTED_ACTIVITY_STRING_LENGTH)}...`
      : value;
  }
  if (depth >= 3) {
    return '[omitted]';
  }
  if (Array.isArray(value)) {
    return value
      .slice(0, MAX_PERSISTED_ACTIVITY_ARRAY_ITEMS)
      .map((item) => compactPersistedActivityValue(item, depth + 1));
  }
  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, MAX_PERSISTED_ACTIVITY_OBJECT_KEYS)
        .map(([key, nestedValue]) => [key, compactPersistedActivityValue(nestedValue, depth + 1)])
    );
  }
  return String(value);
}

export function compactLiveActivityEvent(event) {
  if (!event || typeof event !== 'object') {
    return null;
  }
  const sanitizedEvent = sanitizePersistedActivityValue(event);
  return {
    event: sanitizedEvent.event || sanitizedEvent.type || '',
    type: sanitizedEvent.type,
    timestamp: sanitizedEvent.timestamp || sanitizedEvent.fullTimestamp || '',
    fullTimestamp: sanitizedEvent.fullTimestamp,
    // Persist the user-visible message after recursive credential redaction so
    // live activity remains useful across reloads without storing secrets.
    message: typeof sanitizedEvent.message === 'string'
      ? compactPersistedActivityValue(sanitizedEvent.message)
      : '',
    data: compactPersistedActivityValue(sanitizedEvent.data || {}),
  };
}
