/**
 * AutonomousResearchLogs - Metrics and event log for autonomous research.
 * Shows submission accept/reject statistics broken down by each submitter role.
 * Includes API call logging with full request/response details.
 */
import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { autonomousAPI } from '../../services/api';
import './AutonomousResearch.css';

const EMPTY_API_STATS = Object.freeze({
  total_calls: 0,
  successful_calls: 0,
  failed_calls: 0,
  success_rate: 0,
  boosted_calls: 0,
  by_phase: {},
  by_model: {},
  by_provider: {},
  by_source: {},
  by_boost_mode: {},
});

const AutonomousResearchLogs = ({ stats, events }) => {
  const eventsContainerRef = useRef(null);
  const prevEventsLengthRef = useRef(0);
  const [expandedSubmitters, setExpandedSubmitters] = useState({});
  
  // API Logs state
  const [apiLogs, setApiLogs] = useState([]);
  const [apiStats, setApiStats] = useState(null);
  const [apiLogsLoading, setApiLogsLoading] = useState(true);
  const [expandedApiLogIdx, setExpandedApiLogIdx] = useState(null);
  const [apiAutoRefresh, setApiAutoRefresh] = useState(true);
  const abortControllerRef = useRef(null);

  // Auto-scroll event log only when new events are added (not on mount/tab switch)
  useEffect(() => {
    const currentLength = events ? events.length : 0;
    if (currentLength > prevEventsLengthRef.current && eventsContainerRef.current) {
      eventsContainerRef.current.scrollTop = eventsContainerRef.current.scrollHeight;
    }
    prevEventsLengthRef.current = currentLength;
  }, [events]);

  // Fetch API logs
  const fetchApiLogs = useCallback(async () => {
    // Abort previous request if still pending
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Create new abort controller for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;
    
    try {
      const response = await autonomousAPI.getApiLogs(100, { signal: controller.signal });
      if (abortControllerRef.current !== controller) {
        return;
      }

      if (response.success) {
        setApiLogs(response.logs || []);
        setApiStats(response.stats || EMPTY_API_STATS);
      }
    } catch (error) {
      if (abortControllerRef.current !== controller) {
        return;
      }

      // Don't log abort errors as they're expected on cleanup
      if (error.name !== 'AbortError') {
        console.error('Failed to fetch autonomous API logs:', error);
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setApiLogsLoading(false);
      }
    }
  }, []);

  // Initial fetch and auto-refresh for API logs
  useEffect(() => {
    fetchApiLogs();

    let interval;
    if (apiAutoRefresh) {
      // Set interval to refresh every 5 seconds (skip first call since we already called above)
      interval = setInterval(fetchApiLogs, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
      // Cancel any pending requests on unmount
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [fetchApiLogs, apiAutoRefresh]);

  // Handle clear API logs
  const handleClearApiLogs = async () => {
    if (!window.confirm('Are you sure you want to clear all API logs?')) {
      return;
    }

    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      await autonomousAPI.clearApiLogs();
      setApiLogs([]);
      setApiStats(EMPTY_API_STATS);
      setExpandedApiLogIdx(null);
      setApiLogsLoading(false);
    } catch (error) {
      console.error('Failed to clear API logs:', error);
    }
  };

  // Toggle API log expansion
  const toggleApiLogExpand = (index) => {
    setExpandedApiLogIdx(expandedApiLogIdx === index ? null : index);
  };

  // Copy to clipboard
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  // Format duration
  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  // Get phase label
  const getPhaseLabel = (phase) => {
    switch (phase) {
      case 'topic_selection': return 'Topic';
      case 'brainstorm': return 'Brainstorm';
      case 'paper_compilation': return 'Paper';
      case 'tier3': return 'Tier 3';
      case 'boost': return 'Boost';
      default: return phase || 'Unknown';
    }
  };

  const getSourceLabel = (source) => {
    switch (source) {
      case 'api+boost': return 'Boosted';
      case 'boost': return 'Boost Only';
      default: return 'Standard';
    }
  };

  const getBoostModeLabel = (mode) => {
    switch (mode) {
      case 'next_count': return 'Next X';
      case 'category': return 'Category';
      case 'task_id': return 'Task ID';
      default: return mode || 'Unknown';
    }
  };

  const getProviderLabel = (provider) => {
    switch (provider) {
      case 'openrouter': return 'OR';
      case 'lm_studio': return 'LMS';
      default: return provider || 'UNK';
    }
  };

  // Calculate per-submitter statistics from individual events
  // These come from the aggregator's direct 'submission_accepted'/'submission_rejected' events
  const submitterStats = useMemo(() => {
    const stats = {};
    
    if (!events || events.length === 0) return stats;
    
    events.forEach((event) => {
      // Listen for aggregator's direct events (submission_accepted/rejected)
      if (event.event === 'submission_accepted' || event.event === 'submission_rejected') {
        const data = event.data || {};
        const submitterId = data.submitter_id;
        const isAccepted = event.event === 'submission_accepted';
        
        // Use explicit check for submitter_id to handle edge cases (0 is valid but falsy)
        if (submitterId !== undefined && submitterId !== null) {
          const key = `${submitterId}`;
          
          if (!stats[key]) {
            stats[key] = {
              submitter_id: submitterId,
              model: data.submitter_model || 'N/A',
              provider: data.submitter_provider || 'lm_studio',
              accepted: 0,
              rejected: 0,
              all_events: []
            };
          }
          
          // Update model info if available (latest event wins)
          if (data.submitter_model) {
            stats[key].model = data.submitter_model;
          }
          if (data.submitter_provider) {
            stats[key].provider = data.submitter_provider;
          }
          
          if (isAccepted) {
            stats[key].accepted++;
          } else {
            stats[key].rejected++;
          }
          
          // Store the event for this specific submitter
          stats[key].all_events.unshift({
            type: isAccepted ? 'accepted' : 'rejected',
            timestamp: event.timestamp,
            total: isAccepted ? data.total_acceptances : data.total_rejections
          });
        }
      }
    });
    
    return stats;
  }, [events]);

  const formatEventMessage = (event) => {
    const data = event.data || {};
    
    switch (event.event) {
      case 'auto_research_started':
        return 'Autonomous research started';
      case 'auto_research_stopped':
        return `Research stopped. Total: ${data.final_stats?.total_papers_completed || 0} papers`;
      // Topic exploration events (pre-brainstorm)
      case 'topic_exploration_started':
        return `Topic exploration started (target: ${data.target || 5} candidates${data.resumed_count ? `, resumed: ${data.resumed_count}` : ''})`;
      case 'topic_exploration_progress': {
        const question = data.latest_question ? data.latest_question.substring(0, 80) + '...' : '';
        return `Exploration candidate ${data.accepted}/${data.target} accepted${question ? `: ${question}` : ''}`;
      }
      case 'topic_exploration_rejected':
        return `Exploration candidate rejected (${data.accepted_so_far || 0}/${data.target || 5} accepted)`;
      case 'topic_exploration_complete':
        return `Topic exploration complete: ${data.accepted_count} candidates (${data.total_attempts} attempts)`;
      // Paper title exploration events
      case 'paper_title_exploration_started':
        return `Title exploration started (target: ${data.target || 5} candidate titles)`;
      case 'paper_title_exploration_progress':
        return `Title candidate ${data.accepted}/${data.target} accepted`;
      case 'paper_title_exploration_complete':
        return `Title exploration complete: ${data.accepted_count} candidates (${data.total_attempts} attempts)`;
      case 'topic_selected':
        return `Topic selected: ${data.action} - ${data.topic_prompt || data.topic_id}`;
      case 'topic_selection_rejected':
        return `Topic rejected: ${data.reasoning?.substring(0, 100)}...`;
      // Aggregator's direct per-submission events
      case 'submission_accepted': {
        const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : '';
        return `Submitter ${data.submitter_id} [${modelName}]: ✓ ACCEPTED (total: ${data.total_acceptances})`;
      }
      case 'submission_rejected': {
        const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : '';
        return `Submitter ${data.submitter_id} [${modelName}]: ✗ REJECTED (total: ${data.total_rejections})`;
      }
      case 'completion_review_started':
        return `[${data.topic_id}] Completion review at ${data.submission_count} submissions`;
      case 'completion_review_result':
        return `[${data.topic_id}] Review decision: ${data.decision}`;
      case 'paper_writing_started':
        return `Starting paper: "${data.title}"`;
      case 'paper_section_completed':
        return `Section complete: ${data.section_type}`;
      case 'paper_completed':
        return `Paper complete: "${data.title}" (${data.word_count?.toLocaleString()} words)`;
      case 'paper_redundancy_review':
        return data.should_remove 
          ? `Redundancy: Removed ${data.paper_id}` 
          : 'Redundancy: No removal needed';
      case 'proof_framing_decided':
        return data.is_proof_amenable
          ? 'Proof framing enabled for this run'
          : 'Proof framing not applied for this run';
      case 'proof_check_started':
        if (data.trigger === 'manual') {
          return `Manual proof check started for ${data.source_type} ${data.source_id}`;
        }
        if (data.trigger === 'retry') {
          return `Paper-stage proof retry started for ${data.source_type} ${data.source_id}`;
        }
        return `Proof check started for ${data.source_type} ${data.source_id}`;
      case 'proof_retry_scheduled':
        return `Scheduled ${data.count || 0} proof retry candidate(s) for paper ${data.source_id}`;
      case 'proof_retry_started':
        return `Retrying ${data.count || 0} failed proof candidate(s) against paper ${data.source_id}`;
      case 'proof_check_no_candidates':
        return `No formal theorem candidates found in ${data.source_type} ${data.source_id}`;
      case 'proof_check_candidates_found':
        return `Proof candidates found: ${data.count || 0}`;
      case 'proof_attempt_started':
        return `Proof attempt ${data.attempt || 1}: ${data.theorem_statement || data.theorem_id}`;
      case 'proof_attempt_failed':
        return `Proof attempt ${data.attempt || '?'} failed: ${data.error_summary || data.theorem_statement || data.theorem_id}`;
      case 'proof_verified':
        return `Lean 4 verified: ${data.theorem_statement || data.theorem_id}`;
      case 'proof_attempts_exhausted':
        return `Proof attempts exhausted: ${data.theorem_statement || data.theorem_id}`;
      case 'novel_proof_discovered':
        return `Novel proof discovered: ${data.theorem_statement}`;
      case 'known_proof_verified':
        return `Known proof verified for ${data.source_type} ${data.source_id}`;
      case 'proof_check_complete':
        return `Proof check complete: ${data.verified_count || 0} verified, ${data.novel_count || 0} novel`;
      default:
        return event.event;
    }
  };

  const getEventClass = (event) => {
    const eventName = event.event || '';
    if (eventName === 'proof_attempt_failed' || eventName === 'proof_attempts_exhausted') {
      return 'log-reject';
    }
    if (
      eventName === 'proof_verified' ||
      eventName === 'novel_proof_discovered' ||
      eventName === 'known_proof_verified' ||
      eventName === 'proof_check_complete'
    ) {
      return 'log-success';
    }
    if (
      eventName === 'proof_framing_decided' ||
      eventName === 'proof_check_started' ||
      eventName === 'proof_retry_scheduled' ||
      eventName === 'proof_retry_started' ||
      eventName === 'proof_check_no_candidates' ||
      eventName === 'proof_check_candidates_found' ||
      eventName === 'proof_attempt_started'
    ) {
      return 'log-info';
    }
    if (eventName.includes('completed') || eventName.includes('accepted') || eventName === 'submission_accepted' || eventName === 'topic_exploration_complete' || eventName === 'paper_title_exploration_complete') {
      return 'log-success';
    }
    if (eventName.includes('rejected') || eventName === 'submission_rejected' || eventName === 'topic_exploration_rejected') {
      return 'log-reject';
    }
    if (eventName.includes('started') || eventName.includes('review') || eventName.includes('progress')) {
      return 'log-info';
    }
    return '';
  };

  const toggleSubmitterExpanded = (submitterId) => {
    setExpandedSubmitters(prev => ({
      ...prev,
      [submitterId]: !prev[submitterId]
    }));
  };

  return (
    <div className="autonomous-logs">
      {/* Metrics Grid */}
      <div className="logs-metrics">
        <div className="metric-card">
          <span className="metric-value">{stats?.total_brainstorms_created || 0}</span>
          <span className="metric-label">Brainstorms</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.total_brainstorms_completed || 0}</span>
          <span className="metric-label">Completed</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.total_papers_completed || 0}</span>
          <span className="metric-label">Papers</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.total_papers_archived || 0}</span>
          <span className="metric-label">Archived</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.total_submissions_accepted || 0}</span>
          <span className="metric-label">Accepted</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.total_submissions_rejected || 0}</span>
          <span className="metric-label">Rejected</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">
            {((stats?.acceptance_rate || 0) * 100).toFixed(1)}%
          </span>
          <span className="metric-label">Accept Rate</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.completion_reviews_run || 0}</span>
          <span className="metric-label">Reviews</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.topic_selection_rejections || 0}</span>
          <span className="metric-label">Topic Rejects</span>
        </div>
        
        <div className="metric-card">
          <span className="metric-value">{stats?.paper_redundancy_reviews_run || 0}</span>
          <span className="metric-label">Redundancy</span>
        </div>
      </div>

      {/* Per-Submitter Statistics */}
      <h4 style={{ marginTop: '20px' }}>Per-Submitter Statistics</h4>
      <div className="submitter-stats-container">
        {Object.keys(submitterStats).length === 0 ? (
          <div className="auto-empty-state">
            No submission data yet.
          </div>
        ) : (
          Object.values(submitterStats)
            .sort((a, b) => parseInt(a.submitter_id) - parseInt(b.submitter_id))
            .map((submitter) => {
              const total = submitter.accepted + submitter.rejected;
              const acceptRate = total > 0 ? ((submitter.accepted / total) * 100).toFixed(1) : 0;
              const isExpanded = expandedSubmitters[submitter.submitter_id];
              
              return (
                <div 
                  key={submitter.submitter_id} 
                  className="submitter-stat-card"
                >
                  <div 
                    className="submitter-header"
                    onClick={() => toggleSubmitterExpanded(submitter.submitter_id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <span className="submitter-title">
                      Submitter {submitter.submitter_id}
                      {submitter.submitter_id === '1' && ' (Main)'}
                      {isExpanded ? ' ▼' : ' ▶'}
                    </span>
                    <span className="submitter-model">
                      {submitter.model ? (submitter.model.split('/')[1] || submitter.model.substring(0, 15)) : 'N/A'}
                    </span>
                  </div>
                  
                  <div className="submitter-stats-line">
                    <span className="stat-item success">
                      ✓ {submitter.accepted} Accepted
                    </span>
                    <span className="stat-item reject">
                      ✗ {submitter.rejected} Rejected
                    </span>
                    <span className="stat-item info">
                      {acceptRate}% Rate
                    </span>
                  </div>

                  {/* Expanded Details - Show ALL events for this submitter */}
                  {isExpanded && (
                    <div className="submitter-expanded">
                      <div className="detail-section">
                        <strong>Model Details:</strong>
                        <div className="detail-item">
                          <span>Model:</span> {submitter.model}
                        </div>
                        <div className="detail-item">
                          <span>Provider:</span> {submitter.provider}
                        </div>
                      </div>
                      
                      <div className="detail-section">
                        <strong>Event Log for Submitter {submitter.submitter_id}:</strong>
                        {submitter.all_events.length === 0 ? (
                          <div style={{ fontSize: '0.9em', color: '#999' }}>No events</div>
                        ) : (
                          <div className="recent-events-list">
                            {submitter.all_events.map((evt, idx) => (
                              <div key={idx} className={`recent-event ${evt.type}`}>
                                <span className="event-type">
                                  {evt.type === 'accepted' ? '✓' : '✗'}
                                </span>
                                <span className="event-count">#{evt.total}</span>
                                <span className="event-time">
                                  {new Date(evt.timestamp).toLocaleTimeString()}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })
        )}
      </div>

      {/* API Call Logs Section */}
      <div className="api-logs-section" style={{ marginTop: '30px' }}>
        <div className="api-logs-header">
          <h3>API Call Logs</h3>
          <div className="api-logs-actions">
            <label className="auto-refresh-toggle">
              <input
                type="checkbox"
                checked={apiAutoRefresh}
                onChange={(e) => setApiAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
            <button onClick={fetchApiLogs} className="refresh-btn" title="Refresh now">
              Refresh
            </button>
            <button 
              onClick={handleClearApiLogs} 
              className="clear-btn"
              disabled={apiLogs.length === 0}
            >
              Clear Logs
            </button>
          </div>
        </div>

        {/* API Stats Summary */}
        {apiStats && (
          <div className="api-stats">
            <div className="stat-card">
              <span className="stat-value">{apiStats.total_calls}</span>
              <span className="stat-label">Total API Calls</span>
            </div>
            <div className="stat-card success">
              <span className="stat-value">{apiStats.successful_calls}</span>
              <span className="stat-label">Successful</span>
            </div>
            <div className="stat-card error">
              <span className="stat-value">{apiStats.failed_calls}</span>
              <span className="stat-label">Failed</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">
                {(apiStats.success_rate * 100).toFixed(1)}%
              </span>
              <span className="stat-label">Success Rate</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{apiStats.boosted_calls || 0}</span>
              <span className="stat-label">Boosted Calls</span>
            </div>
          </div>
        )}

        {/* Stats by Phase */}
        {apiStats && apiStats.by_phase && Object.keys(apiStats.by_phase).length > 0 && (
          <div className="phase-stats">
            <span className="phase-stats-label">By Phase:</span>
            {Object.entries(apiStats.by_phase).map(([phase, count]) => (
              <span key={phase} className="phase-stat-badge">
                {getPhaseLabel(phase)}: {count}
              </span>
            ))}
          </div>
        )}

        {apiStats && apiStats.by_source && Object.keys(apiStats.by_source).length > 0 && (
          <div className="phase-stats">
            <span className="phase-stats-label">By Source:</span>
            {Object.entries(apiStats.by_source).map(([source, count]) => (
              <span key={source} className="phase-stat-badge">
                {getSourceLabel(source)}: {count}
              </span>
            ))}
          </div>
        )}

        {apiStats && apiStats.by_boost_mode && Object.keys(apiStats.by_boost_mode).length > 0 && (
          <div className="phase-stats">
            <span className="phase-stats-label">Boost Modes:</span>
            {Object.entries(apiStats.by_boost_mode).map(([mode, count]) => (
              <span key={mode} className="phase-stat-badge">
                {getBoostModeLabel(mode)}: {count}
              </span>
            ))}
          </div>
        )}

        {/* API Logs List */}
        <div className="api-logs-list">
          {apiLogsLoading ? (
            <div className="logs-loading">Loading API logs...</div>
          ) : apiLogs.length === 0 ? (
            <div className="logs-empty">
              <p>No API calls logged yet.</p>
              <p className="logs-empty-hint">
                Run a workflow and make API calls to see the combined logs here.
              </p>
            </div>
          ) : (
            apiLogs.map((log, index) => (
              <div 
                key={index} 
                className={`api-log-entry ${log.success ? 'success' : 'error'} ${expandedApiLogIdx === index ? 'expanded' : ''}`}
              >
                <div 
                  className="log-summary"
                  onClick={() => toggleApiLogExpand(index)}
                >
                  <div className="log-status">
                    {log.success ? '✓' : '✗'}
                  </div>
                  <div className="log-info">
                    <div className="log-task">
                      <span className="log-task-id">{log.task_id}</span>
                      <span className="log-phase-badge">{getPhaseLabel(log.phase)}</span>
                      <span className={`log-source-badge ${log.boosted ? 'boosted' : 'standard'}`}>
                        {getSourceLabel(log.source)}
                      </span>
                      {log.boost_mode && (
                        <span className="log-boost-mode-badge">{getBoostModeLabel(log.boost_mode)}</span>
                      )}
                    </div>
                    <div className="log-meta">
                      <span className="log-model">{log.model}</span>
                      <span className="log-provider-badge">{getProviderLabel(log.provider)}</span>
                      <span className="log-duration">{formatDuration(log.duration_ms)}</span>
                      {log.tokens_used && (
                        <span className="log-tokens">{log.tokens_used} tokens</span>
                      )}
                    </div>
                  </div>
                  <div className="log-timestamp">{formatTimestamp(log.timestamp)}</div>
                  <div className="log-expand-icon">{expandedApiLogIdx === index ? '▼' : '▶'}</div>
                </div>

                {expandedApiLogIdx === index && (
                  <div className="log-details">
                    <div className="log-detail-section">
                      <h4>Role</h4>
                      <pre>{log.role_id}</pre>
                    </div>

                    <div className="log-detail-section">
                      <h4>Source</h4>
                      <pre>{getSourceLabel(log.source)}{log.boost_mode ? ` (${getBoostModeLabel(log.boost_mode)})` : ''}</pre>
                    </div>

                    {log.error && (
                      <div className="log-detail-section error">
                        <h4>Error</h4>
                        <pre>{log.error}</pre>
                      </div>
                    )}

                    <div className="log-detail-section">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h4>Sent Prompt</h4>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(log.prompt_full || log.prompt_preview || '');
                          }}
                          className="copy-btn"
                          title="Copy full prompt to clipboard"
                        >
                          Copy Full
                        </button>
                      </div>
                      <pre className="log-preview">{log.prompt_preview || '(empty)'}</pre>
                    </div>

                    <div className="log-detail-section">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h4>Received Response</h4>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            copyToClipboard(log.response_full || log.response_preview || '');
                          }}
                          className="copy-btn"
                          title="Copy full response to clipboard"
                        >
                          Copy Full
                        </button>
                      </div>
                      <pre className="log-response">{log.response_full || log.response_preview || '(empty)'}</pre>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Event Log */}
      <h4 style={{ marginTop: '20px' }}>Event Log</h4>
      <div className="logs-events" ref={eventsContainerRef}>
        {(!events || events.length === 0) ? (
          <div className="auto-empty-state">
            No events recorded yet.
          </div>
        ) : (
          events.map((event, index) => (
            <div 
              key={index} 
              className={`auto-log-entry ${getEventClass(event)}`}
            >
              <span className="log-time">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
              <span className="log-event">
                {event.event.replace(/_/g, ' ')}
              </span>
              <span className="log-message">
                {formatEventMessage(event)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AutonomousResearchLogs;

