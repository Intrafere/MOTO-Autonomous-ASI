/**
 * AutonomousResearchInterface - Main UI for autonomous research mode.
 * Controls research workflow and displays real-time activity.
 */
import React, { useState, useEffect, useRef } from 'react';
import './AutonomousResearch.css';
import LivePaperProgress from './LivePaperProgress';
import LiveTier3Progress from './LiveTier3Progress';
import TextFileUploader from '../TextFileUploader';

const AutonomousResearchInterface = ({
  isRunning,
  status,
  activity,
  onStart,
  onStop,
  onClear,
  config,
  api
}) => {
  const [researchPrompt, setResearchPrompt] = useState(() => {
    const saved = localStorage.getItem('autonomous_research_prompt');
    return saved || '';
  });
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [showForceConfirm, setShowForceConfirm] = useState(false);
  const [isForcing, setIsForcing] = useState(false);
  const [showTier3Dialog, setShowTier3Dialog] = useState(false);
  const [isForcingTier3, setIsForcingTier3] = useState(false);
  const [critiquePhaseActive, setCritiquePhaseActive] = useState(false);
  const [isSkipping, setIsSkipping] = useState(false);
  const [skipQueued, setSkipQueued] = useState(false);  // Skip has been queued pre-emptively
  const activityEndRef = useRef(null);

  // Save research prompt to localStorage
  useEffect(() => {
    localStorage.setItem('autonomous_research_prompt', researchPrompt);
  }, [researchPrompt]);

  // Auto-scroll activity feed
  useEffect(() => {
    if (activityEndRef.current) {
      activityEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activity]);

  // Listen for critique phase events in activity feed
  useEffect(() => {
    if (!activity || activity.length === 0) return;
    
    const lastEvent = activity[activity.length - 1];
    if (lastEvent.event === 'critique_phase_started') {
      setCritiquePhaseActive(true);
    } else if (lastEvent.event === 'critique_phase_ended') {
      setCritiquePhaseActive(false);
      // Only reset if critique ended without skip (e.g., rewrite happened)
      // If skipQueued is true, the skip worked, so keep showing checkmark
    } else if (lastEvent.event === 'critique_phase_skipped') {
      setCritiquePhaseActive(false);
      // Skip worked! Keep skipQueued=true to show checkmark
    } else if (lastEvent.event === 'paper_writing_started' || lastEvent.event === 'paper_completed') {
      setSkipQueued(false);  // Reset skip state for new paper
    }
  }, [activity]);

  // Reset skip state when tier changes away from paper writing
  useEffect(() => {
    if (status?.current_tier !== 'tier2_paper_writing') {
      setSkipQueued(false);
      setCritiquePhaseActive(false);
    }
  }, [status?.current_tier]);

  const handleTextFileLoaded = (content) => {
    // Append to existing prompt with separator
    const separator = researchPrompt.trim() ? '\n\n' : '';
    const newPrompt = researchPrompt + separator + content;
    setResearchPrompt(newPrompt);
  };

  const handleStart = () => {
    if (!researchPrompt.trim()) {
      alert('Please enter a research prompt');
      return;
    }
    onStart(researchPrompt);
  };

  const handleClear = async () => {
    if (showClearConfirm) {
      setIsClearing(true);
      try {
        await onClear();
        setShowClearConfirm(false);
      } finally {
        setIsClearing(false);
      }
    } else {
      setShowClearConfirm(true);
    }
  };

  const handleForcePaperWriting = async () => {
    if (showForceConfirm) {
      setIsForcing(true);
      try {
        const response = await api.forcePaperWriting();
        if (response.success) {
          alert(`Success! Brainstorm will now transition to paper writing.\nTopic: ${response.topic_prompt}`);
          setShowForceConfirm(false);
        }
      } catch (error) {
        alert(`Failed to force paper writing: ${error.details || error.message}`);
      } finally {
        setIsForcing(false);
      }
    } else {
      setShowForceConfirm(true);
    }
  };

  const handleForceTier3 = async (mode) => {
    // Close dialog immediately - don't wait for API
    setShowTier3Dialog(false);
    
    try {
      // Fire and forget - API returns immediately, Tier 3 runs in background
      const response = await api.forceTier3(mode);
      if (!response.success) {
        alert(`Failed to start Tier 3: ${response.message || 'Unknown error'}`);
      }
      // Success message will come through WebSocket activity feed
    } catch (error) {
      alert(`Failed to force Tier 3: ${error.details || error.message}`);
    }
  };

  const handleSkipCritique = async () => {
    if (!confirm('Skip the critique phase and continue to writing the conclusion? This cannot be undone.')) {
      return;
    }
    
    setIsSkipping(true);
    try {
      await api.skipCritique();
      setSkipQueued(true);  // Mark skip as successfully queued
    } catch (error) {
      alert('Failed to skip critique: ' + error.message);
    } finally {
      setIsSkipping(false);
    }
  };

  const getTier3DialogContext = () => {
    if (!status) return '';
    
    if (status.current_tier === 'tier1_aggregation') {
      const count = status.current_brainstorm?.acceptance_count || 
                   status.current_brainstorm?.submission_count || 0;
      return `Currently in brainstorm aggregation with ${count} submissions.`;
    }
    if (status.current_tier === 'tier2_paper_writing') {
      return `Currently writing paper: ${status.current_paper?.paper_id || 'unknown'}`;
    }
    return 'Ready to start final answer generation.';
  };

  const getTierLabel = (tier, isTier3Active) => {
    // Check for Tier 3 first
    if (isTier3Active) {
      return 'Tier 3 - Final Answer Generation';
    }
    switch (tier) {
      case 'tier1_aggregation':
        return 'Tier 1 - Brainstorm Aggregation';
      case 'tier2_paper_writing':
        return 'Tier 2 - Paper Compilation';
      case 'tier3_final_answer':
        return 'Tier 3 - Final Answer Generation';
      default:
        return 'Idle';
    }
  };

  const getActivityIcon = (event) => {
    switch (event) {
      case 'brainstorm_submission_accepted':
        return '✓';
      case 'brainstorm_submission_rejected':
        return '✗';
      case 'topic_selected':
        return '»';
      case 'topic_selection_rejected':
        return '⚠';
      case 'completion_review_started':
        return '◎';
      case 'completion_review_result':
        return '□';
      case 'manual_paper_writing_triggered':
        return '▶';
      case 'brainstorm_hard_limit_reached':
        return '⊘';
      case 'paper_writing_started':
      case 'paper_writing_resumed':
        return '▬';
      case 'critique_phase_started':
        return '◎';
      case 'critique_progress':
        return '⊟';
      case 'body_rewrite_started':
        return '▬';
      case 'partial_revision_complete':
        return '◈';
      case 'critique_phase_ended':
        return '✓';
      case 'critique_phase_skipped':
        return '↷';
      case 'phase_transition':
        return '□';
      case 'paper_completed':
        return '⊟';
      case 'paper_redundancy_review':
        return '◇';
      // Reference selection events
      case 'reference_selection_started':
        return '▭';
      case 'reference_selection_complete':
        return '✓';
      // Research lifecycle events
      case 'auto_research_resumed':
        return '↻';
      // Tier 3 events
      case 'tier3_started':
        return '★';
      case 'tier3_result':
        return '⊟';
      case 'tier3_format_selected':
        return '▬';
      case 'tier3_volume_organized':
        return '▭';
      case 'tier3_chapter_started':
        return '✎';
      case 'tier3_chapter_complete':
        return '✓';
      case 'tier3_complete':
        return '◆';
      case 'tier3_rejection':
        return '⚠';
      case 'tier3_forced':
        return '▶';
      case 'tier3_phase_changed':
        return '↻';
      case 'tier3_paper_started':
        return '▬';
      case 'tier3_short_form_complete':
      case 'tier3_long_form_complete':
        return '✓';
      case 'final_answer_complete':
        return '◆';
      default:
        return '•';
    }
  };

  const getActivityClass = (event) => {
    // Tier 3 completion events are special highlights
    if (event === 'tier3_complete' || event === 'final_answer_complete') {
      return 'activity-tier3-complete';
    }
    // Success events
    if (event.includes('accepted') || 
        event === 'paper_completed' || 
        event === 'partial_revision_complete' ||
        event === 'tier3_chapter_complete' ||
        event === 'tier3_short_form_complete' ||
        event === 'tier3_long_form_complete' ||
        event === 'reference_selection_complete') {
      return 'activity-success';
    }
    // Rejection events
    if (event.includes('rejected') || event === 'tier3_rejection') {
      return 'activity-reject';
    }
    // Info events (reviews, starts, tier3 progress, etc.)
    if (event.includes('review') || 
        event.includes('started') || 
        event.includes('resumed') ||
        event.includes('progress') ||
        event.includes('transition') ||
        event === 'manual_paper_writing_triggered' ||
        event === 'brainstorm_hard_limit_reached' ||
        event === 'tier3_forced' ||
        event === 'tier3_phase_changed' ||
        event === 'tier3_result' ||
        event === 'tier3_format_selected' ||
        event === 'tier3_volume_organized' ||
        event === 'topic_selected' ||
        event === 'reference_selection_started' ||
        event === 'critique_phase_ended' ||
        event === 'critique_phase_skipped') {
      return 'activity-info';
    }
    return 'activity-neutral';
  };

  return (
    <div className="autonomous-interface">
      {/* Header */}
      <div className="autonomous-header">
        <h2>Autonomous Research</h2>
        <div className="autonomous-controls">
          {!isRunning ? (
            <button 
              className="btn-start"
              onClick={handleStart}
              disabled={!config?.submitter_configs?.some(s => s.modelId)}
            >
              Start Research
            </button>
          ) : (
            <button className="btn-stop" onClick={onStop}>
              Stop Research
            </button>
          )}
          <button 
            className={`btn-clear ${showClearConfirm ? 'btn-confirm' : ''}`}
            onClick={handleClear}
            disabled={isRunning || isClearing}
          >
            {isClearing ? 'Clearing...' : (showClearConfirm ? 'Confirm Clear' : 'Clear All')}
          </button>
          {showClearConfirm && !isClearing && (
            <button 
              className="btn-cancel"
              onClick={() => setShowClearConfirm(false)}
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Research Prompt Input */}
      <div className="research-prompt-section">
        <label htmlFor="research-prompt">Research Goal</label>
        <textarea
          id="research-prompt"
          value={researchPrompt}
          onChange={(e) => setResearchPrompt(e.target.value)}
          placeholder="Enter your high level research goal on any topic that related to S.T.E.M. mathematics, anything event remotely related to mathematics (e.g., 'Explore the connections between modular forms and the Langlands program' or )"
          disabled={isRunning}
          rows={3}
        />
        <TextFileUploader 
          onFileLoaded={handleTextFileLoaded}
          disabled={isRunning}
          maxSizeMB={5}
          showCharCount={true}
          confirmIfNotEmpty={true}
          existingPromptLength={researchPrompt.length}
        />
      </div>

      {/* Status Display */}
      <div className="status-section">
        <div className="status-tier">
          <span className="status-label">Current Status:</span>
          <span className={`status-value ${isRunning ? (status?.is_tier3_active ? 'status-tier3' : 'status-running') : 'status-idle'}`}>
            {isRunning ? getTierLabel(status?.current_tier, status?.is_tier3_active) : 'Not Running'}
          </span>
        </div>

        {status?.current_brainstorm && (
          <div className="current-brainstorm">
            <span className="status-label">Current Brainstorm:</span>
            <span className="brainstorm-id">{status.current_brainstorm.topic_id}</span>
            <p className="brainstorm-prompt">"{status.current_brainstorm.topic_prompt}"</p>
            <div className="brainstorm-stats">
              <span className="submission-count accepted">
                ✓ {status.current_brainstorm.acceptance_count || status.current_brainstorm.submission_count || 0} accepted
              </span>
              <span className="submission-count pruned">
                ◇ {status.current_brainstorm.cleanup_removals || 0} pruned
              </span>
              <span className="submission-count queue">
                ⧗ {status.current_brainstorm.queue_size || 0} in queue
              </span>
            </div>
            
            {/* Manual Control Button */}
            {status?.current_tier === 'tier1_aggregation' && (
              <div className="manual-controls">
                <button
                  className={`btn-force-paper ${showForceConfirm ? 'btn-confirm' : ''}`}
                  onClick={handleForcePaperWriting}
                  disabled={isForcing || !isRunning}
                >
                  {showForceConfirm 
                    ? (isForcing ? 'Forcing...' : 'Confirm Force Paper Writing') 
                    : (
                      <span className="force-paper-text">
                        <span className="force-paper-action">Skip AI Autonomy and Force Paper Writing</span>
                        <span className="force-paper-hint">(We recommend at minimum 5 ACCEPTED submissions - it is normal for a very low % acceptance rate as the validator is only seeking novel solutions - higher parameter models may help submission acceptance rate, however optimizing for both speed (rapid submissions) and knowledge can also work well. The validator provides feedback on rejections to avoid rejection-loops. Harder problems may require hundreds or more of rejections before a single submission acceptance - the first submission acceptance often takes the longest. View brainstorms in the brainstorm tab.)</span>
                      </span>
                    )}
                </button>
                {showForceConfirm && !isForcing && (
                  <button 
                    className="btn-cancel"
                    onClick={() => setShowForceConfirm(false)}
                  >
                    Cancel
                  </button>
                )}
                <p className="manual-help-text">
                  Skip completion review and begin paper writing immediately
                </p>
              </div>
            )}
          </div>
        )}

        {status?.current_paper && (
          <div className="current-paper">
            <span className="status-label">Writing Paper:</span>
            <span className="paper-id">{status.current_paper.paper_id}</span>
            <p className="paper-title">"{status.current_paper.title}"</p>
          </div>
        )}

        {/* Paper Status Banner - Show all the time during Tier 2 paper writing */}
        {status?.current_tier === 'tier2_paper_writing' && (
          <div className="paper-status-banner" style={{
            backgroundColor: critiquePhaseActive ? '#2a2a2a' : '#1a1a1a',
            border: critiquePhaseActive ? '2px solid #ffd700' : '2px solid #666',
            borderRadius: '8px',
            padding: '1rem',
            marginTop: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <span className="status-icon" style={{ fontSize: '2rem' }}>
              {critiquePhaseActive ? '◎' : '▬'}
            </span>
            <div style={{ flex: 1 }}>
              <strong style={{ color: critiquePhaseActive ? '#ffd700' : '#ccc', fontSize: '1.1rem' }}>
                {critiquePhaseActive ? 'Critique Phase in Progress' : 'Paper Writing in Progress'}
              </strong>
              {critiquePhaseActive ? (
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Collecting peer review feedback on body section...
                </p>
              ) : (
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Constructing paper sections...
                </p>
              )}
            </div>
            {/* Skip button - ALWAYS visible during Tier 2 paper writing */}
            <button
              onClick={handleSkipCritique}
              className={`btn ${skipQueued ? 'btn-success' : 'btn-warning'}`}
              style={{ marginLeft: 'auto' }}
              disabled={isSkipping || skipQueued}
            >
              {isSkipping ? 'Skipping...' : skipQueued ? '✓ Skip Queued' : (critiquePhaseActive ? 'Skip Critique Now' : 'Skip Critique (Pre-emptive)')}
            </button>
          </div>
        )}
      </div>

      {/* Force Tier 3 Button - Always visible when running and not already in Tier 3 */}
      {isRunning && !status?.is_tier3_active && status?.stats?.total_papers_completed > 0 && (
        <div className="force-tier3-section">
          <button
            className="btn-force-tier3"
            onClick={() => setShowTier3Dialog(true)}
            disabled={isForcingTier3}
          >
            Force Final Answer (Tier 3) (We recommend at minimum 4 accepted papers in the paper library)
          </button>
          <p className="force-tier3-help">
            Generate the final answer based on completed papers ({status.stats.total_papers_completed} available)
          </p>
        </div>
      )}

      {/* Force Tier 3 Confirmation Dialog */}
      {showTier3Dialog && (
        <div className="tier3-dialog-overlay">
          <div className="tier3-dialog">
            <h3>Force Final Answer Generation</h3>
            <p className="tier3-warning">
              <strong>Warning:</strong> Your system will review for final answer writing; however, it may autonomously decide to override the user and continue paper generation. This system is designed this way because Tier 2 answers are currently the better answers. Given the hallucinatory nature of Tier 3 answers, we want Tier 3 to have the best potential possible. You may find it efficient to skip tier 3 unless you require a 30K+ word answer.
            </p>
            <p className="tier3-context">{getTier3DialogContext()}</p>
            <p className="tier3-papers-count">
              {status?.stats?.total_papers_completed || 0} completed papers available for synthesis
            </p>
            
            <div className="tier3-dialog-options">
              <button
                className="btn-tier3-complete"
                onClick={() => handleForceTier3('complete_current')}
                disabled={isForcingTier3}
              >
                {isForcingTier3 ? 'Processing...' : 'Complete Current Work First'}
                <span className="btn-description">
                  Finish current brainstorm/paper, then generate final answer
                </span>
              </button>
              
              <button
                className="btn-tier3-skip"
                onClick={() => handleForceTier3('skip_incomplete')}
                disabled={isForcingTier3}
              >
                {isForcingTier3 ? 'Processing...' : 'Skip Incomplete & Proceed'}
                <span className="btn-description">
                  Abandon incomplete work, use only completed papers
                </span>
              </button>
              
              <button
                className="btn-tier3-cancel"
                onClick={() => setShowTier3Dialog(false)}
                disabled={isForcingTier3}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Live Paper Progress (Tier 2) */}
      {status?.current_tier === 'tier2_paper_writing' && api && (
        <LivePaperProgress 
          api={api} 
          isCompiling={status?.current_tier === 'tier2_paper_writing'}
        />
      )}

      {/* Live Tier 3 Progress (Final Answer) */}
      {status?.current_tier === 'tier3_final_answer' && api && (
        <LiveTier3Progress 
          api={api}
          status={status}
        />
      )}

      {/* Statistics */}
      {status?.stats && (
        <div className="stats-section">
          <div className="stat-item">
            <span className="stat-value">{status.stats.total_brainstorms_created || 0}</span>
            <span className="stat-label">Brainstorms</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{status.stats.total_brainstorms_completed || 0}</span>
            <span className="stat-label">Completed</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{status.stats.total_papers_completed || 0}</span>
            <span className="stat-label">Papers</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">
              {((status.stats.acceptance_rate || 0) * 100).toFixed(1)}%
            </span>
            <span className="stat-label">Accept Rate</span>
          </div>
        </div>
      )}

      {/* Activity Feed */}
      <div className="activity-section">
        <h3>Live Activity</h3>
        <div className="activity-feed">
          {activity.length === 0 ? (
            <div className="activity-empty">
              No activity yet. Start autonomous research to see updates.
            </div>
          ) : (
            activity.map((item, index) => (
              <div 
                key={index} 
                className={`activity-item ${getActivityClass(item.event)}`}
              >
                <span className="activity-icon">{getActivityIcon(item.event)}</span>
                <span className="activity-time">
                  {new Date(item.timestamp).toLocaleTimeString()}
                </span>
                <span className="activity-message">{item.message}</span>
              </div>
            ))
          )}
          <div ref={activityEndRef} />
        </div>
      </div>
    </div>
  );
};

export default AutonomousResearchInterface;

