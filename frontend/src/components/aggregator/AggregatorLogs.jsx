import React, { useState, useEffect } from 'react';
import { websocket } from '../../services/websocket';
import { api } from '../../services/api';
import '../settings-common.css';

export default function AggregatorLogs() {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const [recoveryStatus, setRecoveryStatus] = useState(null);

  useEffect(() => {
    fetchStatus();
    fetchRecoveryStatus();
    fetchPersistedEvents(); // Load persisted events on mount
    const statusInterval = setInterval(fetchStatus, 2000);
    const recoveryInterval = setInterval(fetchRecoveryStatus, 1000); // More frequent for recovery

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
    ];

    return () => {
      clearInterval(statusInterval);
      clearInterval(recoveryInterval);
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
      const response = await fetch('http://localhost:8000/api/aggregator/status/recovery');
      if (response.ok) {
        const data = await response.json();
        setRecoveryStatus(data);
      }
    } catch (error) {
      console.error('Failed to fetch recovery status:', error);
    }
  };

  const fetchPersistedEvents = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/aggregator/events');
      if (response.ok) {
        const data = await response.json();
        if (data.events && data.events.length > 0) {
          // Convert persisted events to display format
          const persistedEvents = data.events.map(event => ({
            id: event.id,
            type: event.type === 'submission_accepted' ? 'accept' 
                : event.type === 'submission_rejected' ? 'reject'
                : event.type === 'cleanup_submission_removed' ? 'cleanup-remove'
                : 'event',
            message: event.type === 'submission_accepted' 
              ? `✓ ${event.message}`
              : event.type === 'submission_rejected'
              ? `✗ ${event.message}`
              : event.type === 'cleanup_submission_removed'
              ? `${event.message}`
              : event.message,
            timestamp: new Date(event.timestamp).toLocaleTimeString(),
            persisted: true  // Mark as persisted so we don't duplicate
          }));
          // Show persisted events in reverse chronological order (newest first)
          setEvents(persistedEvents.reverse());
        }
      }
    } catch (error) {
      console.error('Failed to fetch persisted events:', error);
    }
  };

  const handleNewSubmission = (data) => {
    addEvent('submission', `New submission from Submitter ${data.submitter_id}`);
  };

  const handleAcceptance = (data) => {
    addEvent('accept', `✓ Submission from Submitter ${data.submitter_id} ACCEPTED`);
  };

  const handleRejection = (data) => {
    addEvent('reject', `✗ Submission from Submitter ${data.submitter_id} REJECTED: ${data.reasoning.substring(0, 100)}...`);
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
    addEvent('cleanup', `Cleanup proposes removing submission #${data.submission_number}`);
  };

  const handleSubmissionRemoved = (data) => {
    addEvent('cleanup-remove', `Submission #${data.submission_number} REMOVED (total removals: ${data.total_removals})`);
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

  const addEvent = (type, message) => {
    const event = {
      id: Date.now(),
      type,
      message,
      timestamp: new Date().toLocaleTimeString(),
    };
    setEvents(prev => [event, ...prev].slice(0, 100)); // Keep last 100 events
  };

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
                        <div style={{ color: '#2196F3' }}>
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

      <h2>Event Log</h2>
      <div className="event-log">
        {events.length === 0 ? (
          <div style={{ color: '#666' }}>No events yet. Start the aggregator to see activity.</div>
        ) : (
          events.map(event => (
            <div key={event.id} className="event-item">
              <div className="event-time">{event.timestamp}</div>
              <div className={`event-${event.type}`}>{event.message}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

