import { sanitizePersistedActivityValue } from './activityPersistence';

const MAX_ACTIVITY_DETAIL_LENGTH = 1200;

const EVENT_ALIASES = {
  accept: 'submission_accepted',
  reject: 'submission_rejected',
  'cleanup-remove': 'cleanup_submission_removed',
  'cleanup-error': 'cleanup_review_error',
  warning: 'hung_connection_alert',
};

const firstPresent = (source, keys) => {
  for (const key of keys) {
    const value = source?.[key];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return '';
};

export function boundedActivityDetail(value, maxLength = MAX_ACTIVITY_DETAIL_LENGTH) {
  const sanitized = sanitizePersistedActivityValue(value);
  const text = String(sanitized ?? '').replace(/\s+/g, ' ').trim();
  if (!text || text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

export function normalizeActivityEventName(eventName = '') {
  const normalized = String(eventName || '').trim();
  return EVENT_ALIASES[normalized] || normalized;
}

export function inferActivityOutcome(eventName = '', data = {}) {
  const explicit = String(firstPresent(data, ['activity_outcome', 'outcome', 'decision', 'status'])).toLowerCase();
  if (['failed', 'failure', 'invalid'].includes(explicit)) return 'error';
  if (['error', 'informational', 'retrying', 'paused', 'cancelled'].includes(explicit)) return explicit;
  if (['accepted', 'accept', 'approved', 'success', 'verified', 'complete', 'completed'].includes(explicit)) {
    return 'accepted';
  }
  if (['rejected', 'reject'].includes(explicit)) {
    return 'rejected';
  }
  if (['declined', 'decline', 'skipped', 'no_change'].includes(explicit)) return 'declined';
  const event = normalizeActivityEventName(eventName).toLowerCase();
  if (/(?:accepted|acceptance|approved|verified|_complete$|_completed$|_success$)/.test(event)) return 'accepted';
  if (/(?:rejected|rejection|failed|failure|_error$)/.test(event)) return 'rejected';
  if (/(?:decline|skipped|no_change)/.test(event)) return 'declined';
  return 'informational';
}

export function inferActivityCategory(eventName = '', data = {}) {
  const explicit = firstPresent(data, ['activity_category', 'category']);
  if (explicit) return String(explicit);
  const event = normalizeActivityEventName(eventName).toLowerCase();
  if (event.startsWith('proof_') || event.startsWith('smt_')) return 'proof';
  if (/(?:provider|oauth|openrouter|hung_connection)/.test(event)) return 'provider';
  if (/(?:submission|compiler_|critique|validator|validation)/.test(event)) return 'validation';
  if (event.startsWith('solution_path_')) return 'solution_path';
  if (/(?:topic|brainstorm|paper|tier3|phase_)/.test(event)) return 'workflow';
  return 'system';
}

export function inferActivitySeverity(eventName = '', data = {}, outcome = '') {
  const explicit = firstPresent(data, ['activity_severity', 'severity', 'level']);
  if (explicit) return String(explicit).toLowerCase();
  const event = normalizeActivityEventName(eventName).toLowerCase();
  if (outcome === 'rejected' || outcome === 'error' || /(?:fatal|repair_required|overflow|corruption)/.test(event)) return 'error';
  if (/(?:warning|retry|cooldown|paused|stopping)/.test(event)) return 'warning';
  if (outcome === 'accepted') return 'success';
  return 'info';
}

export function getActivityIdentity(event = {}) {
  const data = event.data || {};
  const eventName = normalizeActivityEventName(event.event || event.type || '');
  const explicit = firstPresent(event, ['activity_id', 'event_id'])
    || firstPresent(data, ['activity_id', 'manual_event_id', 'terminal_event_id', 'notification_key', 'notification_id']);
  if (explicit) return `${eventName}:${explicit}`;

  const contextParts = [
    firstPresent(data, ['activity_workflow', 'workflow_mode', 'scope']),
    firstPresent(data, ['activity_run_id', 'run_id', 'proof_run_id', 'session_id']),
    firstPresent(data, ['activity_generation', 'lifecycle_generation']),
    firstPresent(data, ['activity_source_type', 'source_type']),
    firstPresent(data, ['activity_source_id', 'source_id', 'topic_id', 'paper_id']),
    firstPresent(data, ['activity_role_id', 'role_id']),
  ].map((value) => boundedActivityDetail(value, 180));
  const distinguishingParts = [
    firstPresent(data, ['submission_id', 'proof_id', 'candidate_id', 'proposal_id']),
    firstPresent(data, [
      'sequence',
      'attempt',
      'attempt_index',
      'list_attempt',
      'iteration',
      'proof_round_index',
      'round_index',
      'retry_attempt',
    ]),
    firstPresent(data, ['total_acceptances', 'total_rejections', 'submission_number']),
  ].map((value) => boundedActivityDetail(value, 180));
  if (distinguishingParts.some(Boolean)) {
    return [eventName, ...contextParts, ...distinguishingParts].join(':');
  }

  return [
    eventName,
    event.timestamp || event.fullTimestamp || '',
    boundedActivityDetail(event.message || data.message || '', 240),
  ].join(':');
}

export function normalizeActivityEvent(event = {}, defaults = {}) {
  const sanitized = sanitizePersistedActivityValue(event) || {};
  const data = sanitized.data && typeof sanitized.data === 'object' ? sanitized.data : {};
  const eventName = normalizeActivityEventName(
    sanitized.event || sanitized.type || defaults.event || defaults.type || '',
  );
  const validatorFeedback = boundedActivityDetail(firstPresent(data, [
    'activity_feedback',
    'validator_feedback',
    'validation_feedback',
    'feedback',
    'reasoning',
  ]));
  const errorDetail = boundedActivityDetail(firstPresent(data, [
    'activity_detail',
    'error_summary',
    'error_detail',
    'error_message',
    'error',
    'warning',
  ]));
  const outcome = inferActivityOutcome(eventName, data);
  const normalized = {
    ...defaults,
    ...sanitized,
    event: eventName,
    type: eventName,
    data,
    category: sanitized.category || inferActivityCategory(eventName, data),
    severity: sanitized.severity || inferActivitySeverity(eventName, data, outcome),
    outcome: sanitized.outcome || outcome,
    validatorFeedback: sanitized.validatorFeedback || validatorFeedback,
    errorDetail: sanitized.errorDetail || errorDetail,
  };
  normalized.activityIdentity = sanitized.activityIdentity || getActivityIdentity(normalized);
  return normalized;
}

export function hasRecentActivityDuplicate(events = [], candidate = {}, limit = 500) {
  const identity = normalizeActivityEvent(candidate).activityIdentity;
  return Boolean(identity) && events.slice(-limit).some(
    (event) => normalizeActivityEvent(event).activityIdentity === identity,
  );
}

export function formatValidatorDecisionActivity(event = {}, options = {}) {
  const normalized = normalizeActivityEvent(event);
  const data = normalized.data;
  const mode = boundedActivityDetail(data.mode || options.defaultMode || 'unknown', 100);
  const feedback = normalized.validatorFeedback;
  const accepted = normalized.outcome === 'accepted';
  const label = accepted ? 'ACCEPTED' : normalized.outcome === 'declined' ? 'DECLINED' : 'REJECTED';
  const subject = options.subject || 'Submission';
  const modeText = options.includeMode === false ? '' : ` (${mode} mode)`;
  return `${subject} ${label}${modeText}.${feedback ? ` Validator feedback: ${feedback}` : ''}`;
}
