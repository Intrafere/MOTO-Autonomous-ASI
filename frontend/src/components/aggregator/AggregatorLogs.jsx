import React, { useState, useEffect } from 'react';
import { websocket } from '../../services/websocket';
import { api } from '../../services/api';
import LiveActivityFeed from '../LiveActivityFeed';
import { MANUAL_AGGREGATOR_PROOF_SOURCE_ID } from '../../hooks/useProofCheckRuntime';
import { getActivityClass, getActivityIcon } from '../../utils/activityStyles';
import '../settings-common.css';

const MAX_EVENT_LOG_ENTRIES = 5000;
const AGGREGATOR_LIVE_ACTIVITY_STORAGE_KEY = 'aggregator_live_activity_manual_events';
const MANUAL_PROOF_EVENTS = [
  'proof_check_started',
  'proof_check_no_candidates',
  'proof_check_candidates_found',
  'proof_attempt_started',
  'proof_lean_accepted',
  'proof_attempt_failed',
  'proof_attempts_exhausted',
  'proof_integrity_rejected',
  'proof_verified',
  'known_proof_verified',
  'proof_registration_duplicate',
  'novel_proof_discovered',
  'proof_dependency_added',
  'proof_check_complete',
];

const normalizeAggregatorEventName = (eventName = '') => {
  switch (eventName) {
    case 'accept':
      return 'submission_accepted';
    case 'reject':
      return 'submission_rejected';
    case 'cleanup-remove':
      return 'cleanup_submission_removed';
    case 'cleanup-error':
      return 'cleanup_review_error';
    case 'warning':
      return 'hung_connection_alert';
    default:
      return eventName;
  }
};

const getEventStorageKey = (event = {}) => {
  const data = event.data || {};
  const normalizedType = normalizeAggregatorEventName(event.type || '');
  if (MANUAL_PROOF_EVENTS.includes(normalizedType) && data.manual_event_id) {
    return `manual-proof:${data.manual_event_id}`;
  }
  if (
    data.submission_id &&
    ['new_submission', 'submission_accepted', 'submission_rejected', 'submission'].includes(normalizedType)
  ) {
    return `submission:${normalizedType}:${data.submission_id}`;
  }
  if (normalizedType === 'submission_accepted' && data.total_acceptances && data.submitter_id) {
    return `accepted:${data.total_acceptances}:${data.submitter_id}`;
  }
  if (normalizedType === 'submission_rejected' && data.total_rejections && data.submitter_id) {
    return `rejected:${data.total_rejections}:${data.submitter_id}`;
  }
  if (normalizedType === 'cleanup_submission_removed' && data.submission_number) {
    return `cleanup-remove:${data.submission_number}`;
  }
  if (event.persisted && event.id) {
    return `persisted:${event.id}`;
  }
  return [
    event.type || '',
    event.timestamp || '',
    event.message || '',
    data.source_type || '',
    data.source_id || '',
    data.theorem_id || '',
    data.proof_id || '',
  ].join('|');
};

const getEventSortTime = (event = {}) => {
  const parsed = new Date(event.timestamp || event.fullTimestamp || '').getTime();
  if (!Number.isNaN(parsed)) {
    return parsed;
  }
  return typeof event.id === 'number' ? event.id : 0;
};

const mergeEventLists = (...eventLists) => {
  const seen = new Set();
  const merged = [];
  eventLists.flat().forEach((event) => {
    if (!event) {
      return;
    }
    const key = getEventStorageKey(event);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    merged.push(event);
  });
  return merged
    .sort((left, right) => getEventSortTime(right) - getEventSortTime(left))
    .slice(0, MAX_EVENT_LOG_ENTRIES);
};

export default function AggregatorLogs() {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const [recoveryStatus, setRecoveryStatus] = useState(null);

  useEffect(() => {
    fetchStatus();
    fetchRecoveryStatus();
    loadInitialEvents(); // Load local manual proof events + backend events together
    const statusInterval = setInterval(fetchStatus, 2000);
    const recoveryInterval = setInterval(fetchRecoveryStatus, 1000); // More frequent for recovery
    const eventRefreshInterval = setInterval(refreshBackendPersistedEvents, 2000);

    // Subscribe to WebSocket events
    const unsubscribers = [
      websocket.on('new_submission', handleNewSubmission),
      websocket.on('submission_accepted', handleAcceptance),
      websocket.on('submission_rejected', handleRejection),
      websocket.on('model_corruption_detected', handleCorruptionDetected),
      websocket.on('model_recovery_initiated', handleRecoveryInitiated),
      websocket.on('model_recovery_success', handleRecoverySuccess),
      websocket.on('model_recovery_failed', handleRecoveryFailed),
      websocket.on('cleanup_review_started', handleCleanupStarted),
      websocket.on('cleanup_removal_proposed', handleRemovalProposed),
      websocket.on('cleanup_submission_removed', handleSubmissionRemoved),
      websocket.on('cleanup_review_complete', handleCleanupComplete),
      websocket.on('cleanup_review_error', handleCleanupError),
      websocket.on('hung_connection_alert', handleHungConnectionAlert),
      ...MANUAL_PROOF_EVENTS.map((eventName) => (
        websocket.on(eventName, (data) => handleManualProofEvent(eventName, data))
      )),
    ];

    return () => {
      clearInterval(statusInterval);
      clearInterval(recoveryInterval);
      clearInterval(eventRefreshInterval);
      unsubscribers.forEach(unsub => unsub());
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const data = await api.getStatus();
      setStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const fetchRecoveryStatus = async () => {
    try {
      const response = await fetch('/api/aggregator/status/recovery');
      if (response.ok) {
        const data = await response.json();
        setRecoveryStatus(data);
      }
    } catch (error) {
      console.error('Failed to fetch recovery status:', error);
    }
  };

  const readStoredManualEvents = () => {
    try {
      const savedEvents = localStorage.getItem(AGGREGATOR_LIVE_ACTIVITY_STORAGE_KEY);
      if (!savedEvents) {
        return [];
      }
      const parsed = JSON.parse(savedEvents);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      console.error('Failed to load manual Aggregator live activity:', error);
      return [];
    }
  };

  const fetchBackendPersistedEvents = async () => {
    try {
      const response = await fetch('/api/aggregator/events');
      if (response.ok) {
        const data = await response.json();
        if (data.events && data.events.length > 0) {
          // Convert persisted events to display format
          return data.events.map(event => ({
            id: event.id,
            type: event.type || 'event',
            message: formatPersistedEventMessage(event),
            data: event.metadata || {},
            timestamp: event.timestamp,
            persisted: true,
          })).reverse();
        }
      }
    } catch (error) {
      console.error('Failed to fetch persisted events:', error);
    }
    return [];
  };

  const formatPersistedEventMessage = (event = {}) => {
    switch (event.type) {
      case 'submission_accepted':
        return `✓ ${event.message}`;
      case 'submission_rejected':
        return `✗ ${event.message}`;
      default:
        return event.message || event.type || 'Aggregator event';
    }
  };

  const loadInitialEvents = async () => {
    const [manualEvents, backendEvents] = await Promise.all([
      Promise.resolve(readStoredManualEvents()),
      fetchBackendPersistedEvents(),
    ]);
    setEvents(mergeEventLists(backendEvents, manualEvents));
  };

  const refreshBackendPersistedEvents = async () => {
    const backendEvents = await fetchBackendPersistedEvents();
    if (backendEvents.length === 0) {
      return;
    }
    setEvents(prev => mergeEventLists(backendEvents, prev));
  };

  const persistManualEvents = (nextEvents) => {
    try {
      const manualEvents = nextEvents.filter((event) => MANUAL_PROOF_EVENTS.includes(event.type));
      localStorage.setItem(
        AGGREGATOR_LIVE_ACTIVITY_STORAGE_KEY,
        JSON.stringify(manualEvents.slice(0, MAX_EVENT_LOG_ENTRIES))
      );
    } catch (error) {
      console.error('Failed to persist manual Aggregator live activity:', error);
    }
  };

  const handleNewSubmission = (data) => {
    const prefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
    addEvent('new_submission', `${prefix}New submission from Submitter ${data.submitter_id}`, data);
  };

  const handleAcceptance = (data) => {
    const prefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
    addEvent('submission_accepted', `✓ ${prefix}Submission from Submitter ${data.submitter_id} ACCEPTED`, data);
  };

  const handleRejection = (data) => {
    const prefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
    addEvent('submission_rejected', `✗ ${prefix}Submission from Submitter ${data.submitter_id} REJECTED: ${data.reasoning.substring(0, 100)}...`, data);
  };

  const handleCorruptionDetected = (data) => {
    addEvent('corruption', `MODEL CORRUPTION DETECTED: ${data.model_id} (${data.failure_count} failures)`);
  };

  const handleRecoveryInitiated = (data) => {
    addEvent('recovery', `Recovery started for ${data.model_id} - ejecting and reloading...`);
  };

  const handleRecoverySuccess = (data) => {
    addEvent('recovery-success', `✓ Model ${data.model_id} recovered successfully! Operations resumed.`);
  };

  const handleRecoveryFailed = (data) => {
    addEvent('recovery-fail', `✗ Model ${data.model_id} recovery FAILED: ${data.error_message}`);
  };

  const handleCleanupStarted = (data) => {
    addEvent('cleanup', `🧹 Cleanup review #${data.review_number} started (${data.total_acceptances} total acceptances)`);
  };

  const handleRemovalProposed = (data) => {
    addEvent('cleanup', `Cleanup proposes removing submission #${data.submission_number}`, data);
  };

  const handleSubmissionRemoved = (data) => {
    addEvent('cleanup_submission_removed', `Submission #${data.submission_number} REMOVED (total removals: ${data.total_removals})`, data);
  };

  const handleCleanupComplete = (data) => {
    if (data.removal_executed) {
      addEvent('cleanup', `✓ Cleanup review #${data.review_number} complete - submission #${data.submission_number} removed`);
    } else if (data.removal_proposed) {
      addEvent('cleanup', `✓ Cleanup review #${data.review_number} complete - removal rejected, keeping submission`);
    } else {
      addEvent('cleanup', `✓ Cleanup review #${data.review_number} complete - no removal needed`);
    }
  };

  const handleCleanupError = (data) => {
    addEvent('cleanup-error', `Cleanup review #${data.review_number} error: ${data.error}`);
  };

  const handleHungConnectionAlert = (data) => {
    const roleId = String(data.role_id || '').toLowerCase();
    if (!roleId.startsWith('aggregator_')) {
      return;
    }
    addEvent('warning', formatHungConnectionMessage(data));
  };

  const handleManualProofEvent = (eventName, data = {}) => {
    if (data.source_type !== 'brainstorm' || data.source_id !== MANUAL_AGGREGATOR_PROOF_SOURCE_ID) {
      return;
    }
    addEvent(eventName, formatProofEvent(eventName, data), data);
  };

  const formatProofEvent = (eventName, data = {}) => {
    switch (eventName) {
      case 'proof_check_started':
        return 'Proof check started for the manual Aggregator database';
      case 'proof_check_no_candidates':
        return 'No formal theorem candidates found in the manual Aggregator database';
      case 'proof_check_candidates_found':
        return `Proof candidates found: ${data.count || 0}`;
      case 'proof_attempt_started':
        return `Lean proof attempt started: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
      case 'proof_lean_accepted':
        return `Lean accepted proof: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
      case 'proof_attempt_failed':
        return `Proof attempt failed: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
      case 'proof_attempts_exhausted':
        return `Proof attempts exhausted: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
      case 'proof_integrity_rejected':
        return `Proof integrity rejected: ${data.reason || data.message || data.theorem_name || 'candidate'}`;
      case 'proof_verified':
        return `Proof verified: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
      case 'known_proof_verified':
        return `Known proof verified: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
      case 'proof_registration_duplicate':
        return `Duplicate proof reused: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
      case 'novel_proof_discovered':
        return `Novel proof discovered: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
      case 'proof_dependency_added':
        return `Proof dependency added: ${data.theorem_name || data.proof_label || data.proof_id || 'verified proof'}`;
      case 'proof_check_complete':
        return `Proof check complete: ${data.verified_count || 0} verified, ${data.novel_count || 0} novel`;
      default:
        return `Proof event: ${eventName}`;
    }
  };

  const formatHungConnectionMessage = (data = {}) => {
    const model = data.model || 'model';
    const provider = data.provider || 'provider';
    const elapsed = data.elapsed_minutes || 15;
    return `Possible hung model call: ${model} via ${provider} (${elapsed}+ min). It may still be thinking; you can keep waiting or lower reasoning effort in Settings if this repeats.`;
  };

  const addEvent = (type, message, data = {}) => {
    const event = {
      id: Date.now(),
      type,
      message,
      data,
      timestamp: new Date().toISOString(),
    };
    setEvents(prev => {
      const updated = mergeEventLists([event], prev);
      if (MANUAL_PROOF_EVENTS.includes(type)) {
        persistManualEvents(updated);
      }
      return updated;
    });
  };

  const getAggregatorActivityClass = (eventName = '', item = {}) => {
    const normalizedEventName = normalizeAggregatorEventName(eventName);
    const data = item?.data || {};
    if (normalizedEventName === 'proof_check_complete') {
      if (data.message) {
        return 'activity-reject';
      }
      return (data.verified_count || data.novel_count) ? 'activity-success' : 'activity-info';
    }
    return getActivityClass(normalizedEventName, { ...item, type: normalizedEventName });
  };

  const getAggregatorActivityIcon = (eventName = '', item = {}) => (
    getActivityIcon(normalizeAggregatorEventName(eventName), item)
  );

  const chronologicalEvents = events.slice().reverse();

  return (
    <div>
      <h1>Aggregator Logs</h1>

      {status && (
        <>
          <div className="grid-3">
            <div className="metric-card">
              <div className="metric-label">Queue Size</div>
              <div className="metric-value">{status.queue_size}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Acceptance Rate</div>
              <div className="metric-value">
                {(status.acceptance_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Accepted Submissions</div>
              <div className="metric-value">
                {status.shared_training_size} / {status.total_acceptances || status.shared_training_size}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                remaining / total
              </div>
            </div>
          </div>

          {/* Cleanup Review Stats */}
          {(status.cleanup_reviews_performed > 0 || status.removals_executed > 0) && (
            <div style={{ 
              backgroundColor: '#e8f5e9', 
              border: '1px solid #4CAF50',
              borderRadius: '8px',
              padding: '1rem',
              margin: '1rem 0'
            }}>
              <h3 style={{ margin: '0 0 0.5rem 0', color: '#2e7d32' }}>🧹 Database Cleanup Stats</h3>
              <div className="grid-3">
                <div className="metric-card" style={{ backgroundColor: '#fff' }}>
                  <div className="metric-label">Cleanup Reviews</div>
                  <div className="metric-value">{status.cleanup_reviews_performed || 0}</div>
                </div>
                <div className="metric-card" style={{ backgroundColor: '#fff' }}>
                  <div className="metric-label">Removals Proposed</div>
                  <div className="metric-value">{status.removals_proposed || 0}</div>
                </div>
                <div className="metric-card" style={{ backgroundColor: '#fff' }}>
                  <div className="metric-label">Removals Executed</div>
                  <div className="metric-value" style={{ color: status.removals_executed > 0 ? '#f44336' : 'inherit' }}>
                    {status.removals_executed || 0}
                  </div>
                </div>
              </div>
            </div>
          )}

          {recoveryStatus && recoveryStatus.in_recovery && (
            <div style={{ 
              backgroundColor: 'rgba(30, 255, 28, 0.1)', 
              border: '2px solid #1eff1c',
              borderRadius: '8px',
              padding: '1rem',
              margin: '1rem 0'
            }}>
              <h2 style={{ color: '#1eff1c', margin: '0 0 0.5rem 0' }}>
                Model Recovery in Progress
              </h2>
              <div style={{ color: '#c6ffc5' }}>
                <div><strong>Model:</strong> {recoveryStatus.recovering_model}</div>
                <div><strong>Stage:</strong> {recoveryStatus.recovery_stage}</div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                  All operations are paused while the model is being ejected and reloaded...
                </div>
              </div>
            </div>
          )}

          {recoveryStatus && !recoveryStatus.in_recovery && Object.keys(recoveryStatus.failure_counts || {}).length > 0 && (
            <div style={{ 
              backgroundColor: '#f8f9fa', 
              border: '1px solid #dee2e6',
              borderRadius: '8px',
              padding: '1rem',
              margin: '1rem 0'
            }}>
              <h3 style={{ margin: '0 0 0.5rem 0' }}>Model Health Status</h3>
              <div className="grid-3">
                {Object.entries(recoveryStatus.failure_counts).map(([model, count]) => (
                  <div key={model} className="metric-card" style={{ 
                    borderColor: count >= recoveryStatus.corruption_threshold ? '#f44336' : '#ff9800'
                  }}>
                    <div className="metric-label label--sm">{model}</div>
                    <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                      <div style={{ color: count >= recoveryStatus.corruption_threshold ? '#f44336' : '#ff9800' }}>
                        Failures: {count}/{recoveryStatus.corruption_threshold}
                      </div>
                      {recoveryStatus.recovery_attempts && recoveryStatus.recovery_attempts[model] > 0 && (
                        <div style={{ color: '#18cc17' }}>
                          Recoveries: {recoveryStatus.recovery_attempts[model]}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <h2>Submitter Status</h2>
          <div className="grid-3">
            {status.submitter_states?.map(submitter => (
              <div key={submitter.submitter_id} className="metric-card">
                <div className="metric-label">Submitter {submitter.submitter_id}</div>
                <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  <div>Submissions: {submitter.total_submissions}</div>
                  <div style={{ color: '#4CAF50' }}>Acceptances: {submitter.total_acceptances}</div>
                  <div className="error-text">Consecutive Rejections: {submitter.consecutive_rejections}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <LiveActivityFeed
        title={`Live Activity${events.length > 0 ? ` (${events.length})` : ''}`}
        items={chronologicalEvents}
        emptyMessage="No events yet. Start the aggregator to see activity."
        getEventName={(event) => event.type || ''}
        getMessage={(event) => event.message || ''}
        getTimestamp={(event) => event.timestamp}
        getClassName={getAggregatorActivityClass}
        getIcon={getAggregatorActivityIcon}
      />
    </div>
  );
}

