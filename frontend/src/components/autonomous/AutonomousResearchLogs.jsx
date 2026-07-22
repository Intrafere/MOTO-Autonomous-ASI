/**
 * AutonomousResearchLogs - Metrics and event log for autonomous research.
 * Shows submission accept/reject statistics broken down by each submitter role.
 * Includes API call logging with full request/response details.
 */
import React, { useRef, useEffect, useMemo, useState } from 'react';
import { autonomousAPI } from '../../services/api';
import ApiCallLogs from '../ApiCallLogs';
import './AutonomousResearch.css';

const AutonomousResearchLogs = ({ stats, events }) => {
  const eventsContainerRef = useRef(null);
  const prevEventsLengthRef = useRef(0);
  const [expandedSubmitters, setExpandedSubmitters] = useState({});

  // Auto-scroll event log only when new events are added (not on mount/tab switch)
  useEffect(() => {
    const currentLength = events ? events.length : 0;
    if (currentLength > prevEventsLengthRef.current && eventsContainerRef.current) {
      eventsContainerRef.current.scrollTop = eventsContainerRef.current.scrollHeight;
    }
    prevEventsLengthRef.current = currentLength;
  }, [events]);

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
    const proofName = data.proof_label ? `Proof ${data.proof_label}` : 'Proof';
    const proofTarget = data.theorem_statement || data.theorem_id || '';
    const proofRoundLabel = () => {
      const round = Number(data.proof_round_index || 0);
      const maxRounds = Number(data.proof_max_rounds || 0);
      if (round <= 0 || maxRounds <= 1) return '';
      return `Proof round ${round}/${maxRounds}`;
    };
    const proofLeanResponse = () => {
      if (data.lean_response) {
        let response = data.lean_response;
        if (/timed out after/i.test(response) && !/Advanced Settings/.test(response)) {
          response = `${response} You can change this timeout in Advanced Settings.`;
        }
        return response;
      }
      if (data.proof_verified === true) return 'Lean 4 response: proof verified.';
      let error = data.error_summary || data.error_output || data.reason || '';
      if (/timed out after/i.test(error) && !/Advanced Settings/.test(error)) {
        error = `${error} You can change this timeout in Advanced Settings.`;
      }
      return error ? `Lean 4 response: ${error} - proof not verified.` : 'Lean 4 response: proof not verified.';
    };
    const formatProofNoveltyTier = (tier) => {
      switch (tier) {
        case 'major_mathematical_discovery':
          return 'Major mathematical discovery';
        case 'mathematical_discovery':
          return 'Mathematical discovery';
        case 'novel_variant':
          return 'Novel variant';
        case 'novel_formulation':
          return 'Novel formulation';
        case 'not_novel':
          return 'Not novel';
        case 'novel':
          return 'Novel';
        default:
          return tier ? String(tier).replace(/_/g, ' ') : 'Not rated';
      }
    };
    const proofNoveltyMessage = () => {
      const tierLabel = formatProofNoveltyTier(data.novelty_tier || (data.is_novel ? 'novel' : 'not_novel'));
      const duplicateNote = data.duplicate ? ' (duplicate proof reused)' : '';
      const rawReason = String(data.novelty_reasoning || data.reasoning || '').replace(/\s+/g, ' ').trim();
      const reason = rawReason.length > 240 ? `${rawReason.slice(0, 240)}...` : rawReason;
      return `${proofName} Lean 4 novelty validator rating: ${tierLabel}${duplicateNote}${reason ? ` - ${reason}` : ''}${proofTarget ? ` (${proofTarget})` : ''}`;
    };
    
    switch (event.event) {
      case 'auto_research_started':
        return 'Autonomous research started';
      case 'auto_research_stopped':
        return data.message || `Research stopped. Total: ${data.final_stats?.total_papers_completed || 0} papers`;
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
        const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
        return `${creativityPrefix}Submitter ${data.submitter_id} [${modelName}]: ✓ ACCEPTED (total: ${data.total_acceptances})`;
      }
      case 'submission_rejected': {
        const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : '';
        const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
        return `${creativityPrefix}Submitter ${data.submitter_id} [${modelName}]: ✗ REJECTED WITH FEEDBACK (total: ${data.total_rejections})`;
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
        return `${proofRoundLabel() || 'Proof check'} started for ${data.source_type} ${data.source_id}`;
      case 'proof_retry_scheduled':
        return `Scheduled ${data.count || 0} proof retry candidate(s) for paper ${data.source_id}`;
      case 'proof_retry_started':
        return `Retrying ${data.count || 0} failed proof candidate(s) against paper ${data.source_id}`;
      case 'proof_check_no_candidates':
        return `${proofRoundLabel() ? `${proofRoundLabel()} discovery` : 'Proof discovery'} found 0 proof candidates; no proofs will be attempted`;
      case 'proof_check_candidates_found': {
        const count = Number(data.count || 0);
        const subject = count === 1 ? 'proof candidate' : 'proof candidates';
        const prefix = proofRoundLabel() ? `${proofRoundLabel()} discovery` : 'Proof discovery';
        return `${prefix} found ${count} ${subject}; ${count} will be attempted`;
      }
      case 'proof_attempt_started':
        return `${proofName}, Attempt ${data.attempt || 1} started: ${proofTarget}`;
      case 'proof_attempt_failed':
        return `${proofName}, Attempt ${data.attempt || '?'} final: ${proofLeanResponse()}`;
      case 'proof_lean_accepted':
        return `${proofName}, Attempt ${data.attempt || '?'} final: ${proofLeanResponse()}`;
      case 'proof_integrity_rejected':
        return `${proofName} error: integrity rejected - ${data.reason || proofTarget}`;
      case 'proof_verified':
        return `${proofName} verified and accepted: ${proofTarget}`;
      case 'proof_attempts_exhausted':
        return `${proofName} terminated: proof attempts exhausted for ${proofTarget}`;
      case 'novel_proof_discovered':
        return proofNoveltyMessage();
      case 'known_proof_verified':
        return proofNoveltyMessage();
      case 'proof_registration_duplicate':
        return proofNoveltyMessage();
      case 'proof_check_complete': {
        const detail = data.message ? ` - ${String(data.message).replace(/\s+/g, ' ').trim()}` : '';
        return `${proofRoundLabel() || 'Proof check'} complete: ${data.verified_count || 0} verified, ${data.novel_count || 0} novel${detail}`;
      }
      case 'hung_connection_alert': {
        const model = data.model || 'model';
        const provider = data.provider || 'provider';
        const elapsed = data.elapsed_minutes || 15;
        return `Possible hung model call: ${model} via ${provider} (${elapsed}+ min). It may still be thinking; you can keep waiting or lower reasoning effort in Settings if this repeats.`;
      }
      default:
        return event.event;
    }
  };

  const getEventClass = (event) => {
    const eventName = event.event || '';
    if (
      eventName === 'proof_attempt_failed' ||
      eventName === 'proof_attempts_exhausted' ||
      eventName === 'proof_integrity_rejected' ||
      eventName === 'smt_check_error'
    ) {
      return 'log-reject';
    }
    if (
      eventName === 'proof_lean_accepted' ||
      eventName === 'proof_verified' ||
      eventName === 'novel_proof_discovered' ||
      eventName === 'known_proof_verified' ||
      eventName === 'proof_registration_duplicate' ||
      eventName === 'proof_check_complete'
    ) {
      return 'log-success';
    }
    if (
      eventName === 'hung_connection_alert'
    ) {
      return 'log-warning';
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
          <span className="metric-value">{stats?.total_papers_pruned || stats?.total_papers_archived || 0}</span>
          <span className="metric-label">Pruned</span>
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

      <ApiCallLogs
        api={autonomousAPI}
        workflow="autonomous"
        style={{ marginTop: '30px' }}
      />

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

