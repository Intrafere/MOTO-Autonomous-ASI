import React, { useState, useEffect } from 'react';
import { compilerAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import LiveActivityFeed from '../LiveActivityFeed';
import { getActivityClass, getActivityIcon } from '../../utils/activityStyles';
import {
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
  MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID,
} from '../../hooks/useProofCheckRuntime';
import '../autonomous/AutonomousResearch.css';

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

function CompilerLogs() {
  const [metrics, setMetrics] = useState({
    construction: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
    rigor: { acceptances: 0, rejections: 0, declines: 0, acceptance_rate: 0 },
    outline: { acceptances: 0, rejections: 0, declines: 0 },
    review: { acceptances: 0, rejections: 0, declines: 0 },
    minuscule_edit_count: 0,
    paper_word_count: 0,
    total_submissions: 0
  });
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState({ current_mode: 'idle' });
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [recoveryStatus, setRecoveryStatus] = useState(null);
  const [critiqueStats, setCritiqueStats] = useState({ accepted: 0, rejected: 0, total: 0, version: 1 });

  useEffect(() => {
    loadMetrics();
    loadStatus();
    loadRecoveryStatus();
    
    // Poll metrics every 5 seconds
    const interval = setInterval(() => {
      loadMetrics();
      loadStatus();
    }, 5000);

    // Poll recovery status more frequently
    const recoveryInterval = setInterval(loadRecoveryStatus, 1000);

    // Listen for WebSocket events
    const handleCompilerEvent = (event) => {
      addEvent(event);
      loadMetrics(); // Refresh metrics on events
    };

    const handleCompilerError = (data) => {
      console.error('Compiler error:', data);
      setError({
        message: data.error,
        traceback: data.traceback,
        mode: data.mode,
        source: data.source || 'compiler'
      });
      addEvent({ type: 'compiler_error', data });
      loadStatus(); // Update status to reflect error state
    };

    const handleCompilerWarning = (data) => {
      console.warn('Compiler warning:', data);
      setWarning({
        message: data.warning,
        mode: data.current_mode,
        lastActivity: data.last_activity
      });
      addEvent({ type: 'compiler_warning', data });
    };

    const handleCorruptionDetected = (data) => {
      addEvent({ type: 'model_corruption_detected', data: { message: `Model ${data.model_id} corrupted (${data.failure_count} failures)` } });
    };

    const handleRecoveryInitiated = (data) => {
      addEvent({ type: 'model_recovery_initiated', data: { message: `Recovery started for ${data.model_id}` } });
    };

    const handleRecoverySuccess = (data) => {
      addEvent({ type: 'model_recovery_success', data: { message: `✓ Model ${data.model_id} recovered successfully!` } });
    };

    const handleRecoveryFailed = (data) => {
      addEvent({ type: 'model_recovery_failed', data: { message: `✗ Model ${data.model_id} recovery FAILED` } });
    };

    const handleCompilerDecline = (data) => {
      addEvent({ type: 'compiler_decline', data });
      loadMetrics(); // Refresh metrics on declines
    };

    const handleManualProofEvent = (eventName, data = {}) => {
      const isCurrentPaperProof = data.source_type === 'paper'
        && data.source_id === MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID;
      const isAggregatorProofOnly = data.source_type === 'brainstorm'
        && data.source_id === MANUAL_AGGREGATOR_PROOF_SOURCE_ID;
      if (!isCurrentPaperProof && !isAggregatorProofOnly) {
        return;
      }
      addEvent({ type: eventName, data });
    };

    // Handler for critique progress to update stats display
    const handleCritiqueProgress = (data) => {
      setCritiqueStats({
        accepted: data.acceptances || 0,
        rejected: data.rejections || 0,
        total: data.total_attempts || 0,
        version: data.version || 1
      });
      addEvent({ type: 'critique_progress', data });
    };

    // Handler for critique phase ended to reset stats
    const handleCritiquePhaseEnded = (data) => {
      addEvent({ type: 'critique_phase_ended', data });
      loadMetrics();
    };

    // Handler for critique phase started
    const handleCritiquePhaseStarted = (data) => {
      setCritiqueStats({ accepted: 0, rejected: 0, total: 0, version: data.paper_version || 1 });
      addEvent({ type: 'critique_phase_started', data });
    };

    websocket.on('compiler_submission', handleCompilerEvent);
    websocket.on('compiler_acceptance', handleCompilerEvent);
    websocket.on('compiler_rejection', handleCompilerEvent);
    websocket.on('compiler_decline', handleCompilerDecline);
    websocket.on('paper_updated', handleCompilerEvent);
    websocket.on('outline_updated', handleCompilerEvent);
    websocket.on('compiler_error', handleCompilerError);
    websocket.on('compiler_warning', handleCompilerWarning);
    websocket.on('model_corruption_detected', handleCorruptionDetected);
    websocket.on('model_recovery_initiated', handleRecoveryInitiated);
    websocket.on('model_recovery_success', handleRecoverySuccess);
    websocket.on('model_recovery_failed', handleRecoveryFailed);
    websocket.on('hung_connection_alert', handleHungConnectionAlert);

    // Critique phase events
    websocket.on('critique_phase_started', handleCritiquePhaseStarted);
    websocket.on('critique_progress', handleCritiqueProgress);
    websocket.on('critique_accepted', handleCompilerEvent);
    websocket.on('critique_rejected', handleCompilerEvent);
    websocket.on('critique_decline_accepted', handleCompilerEvent);
    websocket.on('critique_decline_rejected', handleCompilerEvent);
    websocket.on('critique_removed', handleCompilerEvent);
    websocket.on('critique_phase_ended', handleCritiquePhaseEnded);
    websocket.on('self_review_appended', handleCompilerEvent);

    // Phase transition events
    websocket.on('phase_transition', handleCompilerEvent);
    websocket.on('phase_completion_signal', handleCompilerEvent);

    // Wolfram Alpha tool call events (Phase 3) - main writer invoked Wolfram
    websocket.on('compiler_wolfram_call', handleCompilerEvent);

    const manualProofUnsubscribers = MANUAL_PROOF_EVENTS.map((eventName) => (
      websocket.on(eventName, (data) => handleManualProofEvent(eventName, data))
    ));

    return () => {
      clearInterval(interval);
      clearInterval(recoveryInterval);
      websocket.off('compiler_submission', handleCompilerEvent);
      websocket.off('compiler_acceptance', handleCompilerEvent);
      websocket.off('compiler_rejection', handleCompilerEvent);
      websocket.off('compiler_decline', handleCompilerDecline);
      websocket.off('paper_updated', handleCompilerEvent);
      websocket.off('outline_updated', handleCompilerEvent);
      websocket.off('compiler_error', handleCompilerError);
      websocket.off('compiler_warning', handleCompilerWarning);
      websocket.off('model_corruption_detected', handleCorruptionDetected);
      websocket.off('model_recovery_initiated', handleRecoveryInitiated);
      websocket.off('model_recovery_success', handleRecoverySuccess);
      websocket.off('model_recovery_failed', handleRecoveryFailed);
      websocket.off('hung_connection_alert', handleHungConnectionAlert);

      // Critique phase events cleanup
      websocket.off('critique_phase_started', handleCritiquePhaseStarted);
      websocket.off('critique_progress', handleCritiqueProgress);
      websocket.off('critique_accepted', handleCompilerEvent);
      websocket.off('critique_rejected', handleCompilerEvent);
      websocket.off('critique_decline_accepted', handleCompilerEvent);
      websocket.off('critique_decline_rejected', handleCompilerEvent);
      websocket.off('critique_removed', handleCompilerEvent);
      websocket.off('critique_phase_ended', handleCritiquePhaseEnded);
      websocket.off('self_review_appended', handleCompilerEvent);

      // Phase transition events cleanup
      websocket.off('phase_transition', handleCompilerEvent);
      websocket.off('phase_completion_signal', handleCompilerEvent);

      // Wolfram tool cleanup
      websocket.off('compiler_wolfram_call', handleCompilerEvent);

      manualProofUnsubscribers.forEach((unsubscribe) => unsubscribe());
    };
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await compilerAPI.getMetrics();
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  };

  const loadStatus = async () => {
    try {
      const response = await compilerAPI.getStatus();
      setStatus(response.data);
    } catch (error) {
      console.error('Failed to load status:', error);
    }
  };

  const loadRecoveryStatus = async () => {
    try {
      const response = await fetch('/api/compiler/status/recovery');
      if (response.ok) {
        const data = await response.json();
        setRecoveryStatus(data);
      }
    } catch (error) {
      console.error('Failed to load recovery status:', error);
    }
  };

  const addEvent = (event) => {
    const timestamp = new Date().toLocaleTimeString();
    const fullTimestamp = new Date().toISOString();
    const newEvent = {
      ...event,
      timestamp,
      fullTimestamp
    };
    
    setEvents(prev => {
      const updated = [newEvent, ...prev].slice(0, 10000); // Keep last 10k events
      
      // Save to localStorage for persistence
      try {
        localStorage.setItem('compiler_events_log', JSON.stringify(updated));
      } catch (e) {
        console.error('Failed to save events to localStorage:', e);
      }
      
      return updated;
    });
  };

  const handleHungConnectionAlert = (data) => {
    const roleId = String(data.role_id || '').toLowerCase();
    if (!roleId.startsWith('compiler_')) {
      return;
    }
    addEvent({ type: 'hung_connection_alert', data });
  };

  // Load events from localStorage on mount
  useEffect(() => {
    try {
      const savedEvents = localStorage.getItem('compiler_events_log');
      if (savedEvents) {
        setEvents(JSON.parse(savedEvents));
      }
    } catch (e) {
      console.error('Failed to load events from localStorage:', e);
    }
  }, []);

  const formatRate = (rate) => {
    return (rate * 100).toFixed(1) + '%';
  };

  const clearEventsLog = () => {
    setEvents([]);
    localStorage.removeItem('compiler_events_log');
  };

  // Format event data for user-friendly display
  const formatEventDisplay = (event) => {
    const data = event.data || {};
    const type = event.type || '';

    // Critique phase events
    if (type === 'critique_phase_started') {
      return `Critique phase started (paper v${data.paper_version || 1}, target: ${data.target_critiques || 3} attempts)`;
    }
    if (type === 'critique_progress') {
      return `Progress: ${data.acceptances || 0} accepted, ${data.rejections || 0} rejected, ${data.total_attempts || 0}/${data.target || 3} total`;
    }
    if (type === 'critique_accepted') {
      return `Critique ACCEPTED (${data.count || '?'}/${data.target || 3}): ${(data.preview || '').substring(0, 80)}...`;
    }
    if (type === 'critique_rejected') {
      return `Critique REJECTED: ${(data.reasoning || '').substring(0, 100)}...`;
    }
    if (type === 'critique_decline_accepted') {
      return `Decline ACCEPTED - body is academically acceptable`;
    }
    if (type === 'critique_decline_rejected') {
      return `Decline REJECTED - validator found issues: ${(data.reasoning || '').substring(0, 80)}...`;
    }
    if (type === 'critique_removed') {
      return `Critique #${data.critique_number} removed via cleanup`;
    }
    if (type === 'critique_phase_ended') {
      return `Critique phase ended (self-review appended: ${data.self_review_appended ? 'YES' : 'NO'})`;
    }
    if (type === 'self_review_appended') {
      return `AI self-review appended (${data.critique_count || 0} accepted critique${data.critique_count === 1 ? '' : 's'})`;
    }

    // Phase transitions
    if (type === 'phase_transition') {
      return `Phase transition: ${data.from_phase || '?'} → ${data.to_phase || '?'}`;
    }
    if (type === 'phase_completion_signal') {
      return `Section complete: ${data.previous_phase || '?'} → ${data.new_phase || '?'}`;
    }

    // Existing events - keep simple for now
    if (type === 'compiler_submission') {
      return `Submission: ${data.mode || 'unknown'} mode`;
    }
    if (type === 'compiler_acceptance') {
      return `ACCEPTED: ${data.mode || 'unknown'} mode`;
    }
    if (type === 'compiler_rejection') {
      return `REJECTED: ${data.mode || 'unknown'} - ${(data.reasoning || '').substring(0, 80)}...`;
    }
    if (type === 'compiler_decline') {
      return `Declined: ${data.mode || 'unknown'} - ${(data.reasoning || '').substring(0, 60)}...`;
    }
    if (type === 'paper_updated') {
      return `Paper updated: ${data.word_count?.toLocaleString() || '?'} words`;
    }
    if (type === 'outline_updated') {
      return `Outline updated`;
    }

    // Wolfram Alpha tool call (Phase 3)
    if (type === 'compiler_wolfram_call') {
      const n = data.calls_used ?? '?';
      const cap = data.max_calls ?? 20;
      const query = (data.query || '').substring(0, 80);
      const preview = (data.result_preview || '').substring(0, 80);
      const previewSuffix = preview ? ` - ${preview}` : '';
      return `[Wolfram ${n}/${cap}] ${query}${previewSuffix}`;
    }
    if (type === 'proof_check_started') {
      return 'Proof check started for the current manual Compiler paper';
    }
    if (type === 'proof_check_no_candidates') {
      return 'No formal theorem candidates found in the current manual Compiler paper';
    }
    if (type === 'proof_check_candidates_found') {
      return `Proof candidates found: ${data.count || 0}`;
    }
    if (type === 'proof_attempt_started') {
      return `Lean proof attempt started: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
    }
    if (type === 'proof_lean_accepted') {
      return `Lean accepted proof: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
    }
    if (type === 'proof_attempt_failed') {
      return `Proof attempt failed: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
    }
    if (type === 'proof_attempts_exhausted') {
      return `Proof attempts exhausted: ${data.theorem_name || data.proof_label || data.theorem_id || 'candidate'}`;
    }
    if (type === 'proof_integrity_rejected') {
      return `Proof integrity rejected: ${data.reason || data.message || data.theorem_name || 'candidate'}`;
    }
    if (type === 'proof_verified') {
      return `Proof verified: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
    }
    if (type === 'known_proof_verified') {
      return `Known proof verified: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
    }
    if (type === 'proof_registration_duplicate') {
      return `Duplicate proof reused: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
    }
    if (type === 'novel_proof_discovered') {
      return `Novel proof discovered: ${data.theorem_name || data.proof_label || data.proof_id || 'candidate'}`;
    }
    if (type === 'proof_dependency_added') {
      return `Proof dependency added: ${data.theorem_name || data.proof_label || data.proof_id || 'verified proof'}`;
    }
    if (type === 'proof_check_complete') {
      return `Proof check complete: ${data.verified_count || 0} verified, ${data.novel_count || 0} novel`;
    }
    if (type === 'hung_connection_alert') {
      const model = data.model || 'model';
      const provider = data.provider || 'provider';
      const elapsed = data.elapsed_minutes || 15;
      return `Possible hung model call: ${model} via ${provider} (${elapsed}+ min). It may still be thinking; you can keep waiting or lower reasoning effort in Settings if this repeats.`;
    }

    // Default: show raw JSON
    return JSON.stringify(data, null, 2);
  };

  const getCompilerActivityClass = (eventName = '', item = {}) => {
    if (eventName === 'proof_check_complete') {
      const data = item?.data || {};
      if (data.message) {
        return 'activity-reject';
      }
      return (data.verified_count || data.novel_count) ? 'activity-success' : 'activity-info';
    }
    return getActivityClass(eventName, item);
  };

  const chronologicalEvents = events.slice().reverse();

  return (
    <div className="autonomous-logs compiler-logs">
      <div className="autonomous-header">
        <h2>Single Paper Writer Logs</h2>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="alert alert-error">
          <h3>Compiler Error</h3>
          <p><strong>Source:</strong> {error.source}</p>
          <p><strong>Mode:</strong> {error.mode}</p>
          <p><strong>Error:</strong> {error.message}</p>
          {error.traceback && (
            <details>
              <summary>Show Traceback</summary>
              <pre className="traceback">{error.traceback}</pre>
            </details>
          )}
          <button onClick={() => setError(null)} className="dismiss-btn">Dismiss</button>
        </div>
      )}

      {/* Warning Alert */}
      {warning && (
        <div className="alert alert-warning">
          <h3>Compiler Warning</h3>
          <p><strong>Message:</strong> {warning.message}</p>
          <p><strong>Mode:</strong> {warning.mode}</p>
          {warning.lastActivity && (
            <p><strong>Last Activity:</strong> {new Date(warning.lastActivity * 1000).toLocaleString()}</p>
          )}
          <button onClick={() => setWarning(null)} className="dismiss-btn">Dismiss</button>
        </div>
      )}

      {/* Recovery Alert */}
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

      {!status.is_running && metrics.total_submissions === 0 ? (
        <div className="empty-state">
          <h3>Compiler Not Running</h3>
          <p>The compiler has not been started yet.</p>
          <p>Go to the <strong>Compiler Interface</strong> tab to start compiling the aggregator database into a paper.</p>
        </div>
      ) : (
        <>
          <div className="current-mode">
            <h3>Current Mode: <span className="mode-highlight">{status.current_mode}</span></h3>
            {error && <span className="compiler-status-badge status-error">Error</span>}
            {warning && !error && <span className="compiler-status-badge status-warning">Stalled</span>}
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Construction</h3>
              <div className="metric-value">{metrics.construction.acceptances} / {metrics.construction.rejections} / {metrics.construction.declines}</div>
              <div className="metric-label">Accept / Reject / Decline</div>
              <div className="metric-rate">Rate: {formatRate(metrics.construction.acceptance_rate)}</div>
            </div>

            <div className="metric-card">
              <h3>Rigor Enhancement</h3>
              <div className="metric-value">{metrics.rigor.acceptances} / {metrics.rigor.rejections} / {metrics.rigor.declines}</div>
              <div className="metric-label">Accept / Reject / Decline</div>
              <div className="metric-rate">Rate: {formatRate(metrics.rigor.acceptance_rate)}</div>
            </div>

            <div className="metric-card">
              <h3>Outline</h3>
              <div className="metric-value">{metrics.outline.acceptances} / {metrics.outline.rejections} / {metrics.outline.declines}</div>
              <div className="metric-label">Accept / Reject / Decline</div>
            </div>

            <div className="metric-card">
              <h3>Review</h3>
              <div className="metric-value">{metrics.review.acceptances} / {metrics.review.rejections} / {metrics.review.declines}</div>
              <div className="metric-label">Accept / Reject / Decline</div>
            </div>

            <div className="metric-card">
              <h3>Total Submissions</h3>
              <div className="metric-value">{metrics.total_submissions}</div>
              <div className="metric-label">All modes combined</div>
            </div>

            <div className="metric-card">
              <h3>Minuscule Edits</h3>
              <div className="metric-value">{metrics.minuscule_edit_count}</div>
              <div className="metric-label">Convergence indicator</div>
            </div>

            <div className="metric-card">
              <h3>Paper Word Count</h3>
              <div className="metric-value">{metrics.paper_word_count.toLocaleString()}</div>
              <div className="metric-label">Current paper size</div>
            </div>

            {/* Critique Phase Stats - Show when in critique mode or has activity */}
            {(status.current_mode === 'critique' || critiqueStats.total > 0) && (
              <div className="metric-card" style={{ borderColor: '#1eff1c', borderWidth: '2px' }}>
                <h3>Critique Phase (v{critiqueStats.version})</h3>
                <div className="metric-value">{critiqueStats.accepted} / {critiqueStats.rejected} / {critiqueStats.total}</div>
                <div className="metric-label">Accept / Reject / Total Attempts</div>
                <div className="metric-rate" style={{ color: '#1eff1c' }}>Target: 5 attempts</div>
              </div>
            )}
          </div>

          <LiveActivityFeed
            title={`Live Activity${events.length > 0 ? ` (${events.length} saved)` : ''}`}
            items={chronologicalEvents}
            emptyMessage="No events yet"
            getEventName={(event) => event.type || ''}
            getMessage={formatEventDisplay}
            getTimestamp={(event) => event.fullTimestamp || event.timestamp}
            getClassName={getCompilerActivityClass}
            getIcon={getActivityIcon}
            headerAction={(
              <button onClick={clearEventsLog} className="btn-clear">
                Clear Events Log
              </button>
            )}
          />

          <div className="info-section">
            <h4>Convergence Indicators</h4>
            <p>The paper is approaching completion when:</p>
            <ul>
              <li>Construction <strong>declines</strong> increase (paper already covers all topics)</li>
              <li>Construction rejection rate increases (no more novel content to add)</li>
              <li>Minuscule edit count increases (only tiny improvements found)</li>
              <li>Review <strong>declines</strong> increase (paper is already clean)</li>
              <li>Rigor <strong>declines</strong> increase (rigor already adequate)</li>
            </ul>
            <p><strong>Note:</strong> A <em>decline</em> means the submitter chose not to make changes because the guide is already complete/perfect for that aspect, while a <em>rejection</em> means the validator rejected a proposed change.</p>
          </div>
        </>
      )}
    </div>
  );
}

export default CompilerLogs;

