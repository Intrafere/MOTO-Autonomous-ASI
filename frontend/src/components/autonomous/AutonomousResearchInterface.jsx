/**
 * AutonomousResearchInterface - Main UI for autonomous research mode.
 * Controls research workflow and displays real-time activity.
 */
import React, { useState, useEffect, useRef } from 'react';
import './AutonomousResearch.css';
import LivePaperProgress from './LivePaperProgress';
import LiveTier3Progress from './LiveTier3Progress';
import TextFileUploader from '../TextFileUploader';
import '../settings-common.css';
import { getActivityClass as getSharedActivityClass, getActivityIcon as getSharedActivityIcon } from '../../utils/activityStyles';

const AutonomousResearchInterface = ({
  isRunning,
  isStopping = false,
  anyWorkflowRunning,
  status,
  activity,
  onStart,
  onStop,
  onClear,
  config,
  onConfigChange,
  developerModeEnabled = false,
  capabilities = {},
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
  const [explorationProgress, setExplorationProgress] = useState(null);  // Topic exploration phase tracking
  const [titleExplorationProgress, setTitleExplorationProgress] = useState(null);  // Paper title exploration tracking
  const [proofOutputUpdating, setProofOutputUpdating] = useState(false);
  const activityFeedRef = useRef(null);
  const prevActivityLengthRef = useRef(0);
  const proofOutputsAvailable = !capabilities?.genericMode;

  // Save research prompt to localStorage
  useEffect(() => {
    localStorage.setItem('autonomous_research_prompt', researchPrompt);
  }, [researchPrompt]);

  // Auto-scroll activity feed only when new items are added (not on mount/tab switch)
  useEffect(() => {
    const currentLength = activity ? activity.length : 0;
    if (currentLength > prevActivityLengthRef.current && activityFeedRef.current) {
      activityFeedRef.current.scrollTop = activityFeedRef.current.scrollHeight;
    }
    prevActivityLengthRef.current = currentLength;
  }, [activity]);

  // Listen for critique phase events in activity feed
  useEffect(() => {
    if (!activity || activity.length === 0) return;
    
    const lastEvent = activity[activity.length - 1];
    if (lastEvent.event === 'critique_phase_started') {
      setCritiquePhaseActive(true);
    } else if (lastEvent.event === 'critique_phase_ended') {
      setCritiquePhaseActive(false);
    }
    
    // Topic exploration phase tracking
    if (lastEvent.event === 'topic_exploration_started') {
      setExplorationProgress({ accepted: lastEvent.data?.resumed_count || 0, target: lastEvent.data?.target || 5 });
    } else if (lastEvent.event === 'topic_exploration_progress') {
      setExplorationProgress({ accepted: lastEvent.data?.accepted || 0, target: lastEvent.data?.target || 5 });
    } else if (lastEvent.event === 'topic_exploration_complete' || lastEvent.event === 'topic_selected') {
      setExplorationProgress(null);
    }
    
    // Paper title exploration phase tracking
    if (lastEvent.event === 'paper_title_exploration_started') {
      setTitleExplorationProgress({ accepted: lastEvent.data?.resumed_count || 0, target: lastEvent.data?.target || 5 });
    } else if (lastEvent.event === 'paper_title_exploration_progress') {
      setTitleExplorationProgress({ accepted: lastEvent.data?.accepted || 0, target: lastEvent.data?.target || 5 });
    } else if (lastEvent.event === 'paper_title_exploration_complete' || lastEvent.event === 'paper_writing_started') {
      setTitleExplorationProgress(null);
    }
  }, [activity]);

  // Reset critique phase state when tier changes away from paper writing
  useEffect(() => {
    if (status?.current_tier !== 'tier2_paper_writing') {
      setCritiquePhaseActive(false);
    }
  }, [status?.current_tier]);

  const handleTextFileLoaded = (content) => {
    // Append to existing prompt with separator
    const separator = researchPrompt.trim() ? '\n\n' : '';
    const newPrompt = researchPrompt + separator + content;
    setResearchPrompt(newPrompt);
  };

  const handleStart = async () => {
    if (anyWorkflowRunning && !isRunning) {
      alert('Another workflow is already running. Stop it before starting Autonomous Research.');
      return;
    }

    if (!researchPrompt.trim()) {
      alert('Please enter a research prompt');
      return;
    }
    const mathematicalProofsAllowed = proofOutputsAvailable && (config?.allow_mathematical_proofs ?? true);
    const researchPapersAllowed = config?.allow_research_papers ?? true;
    if (!mathematicalProofsAllowed && !researchPapersAllowed) {
      alert('Please allow at least one output: Mathematical Proofs or Research Papers.');
      return;
    }
    const proofOnlyRequested = mathematicalProofsAllowed && !researchPapersAllowed;
    const shouldSyncProofRuntime = mathematicalProofsAllowed && !capabilities?.genericMode;
    if (proofOnlyRequested || shouldSyncProofRuntime) {
      const enabled = await updateProofRuntimeSetting(true);
      if (!enabled) {
        return;
      }
    }
    onStart(researchPrompt);
  };

  const updateProofRuntimeSetting = async (enabled) => {
    if (!api?.getProofStatus || !api?.updateProofSettings || capabilities?.genericMode) {
      if (enabled) {
        alert('Mathematical proof output is unavailable in this runtime.');
        return false;
      }
      return true;
    }

    setProofOutputUpdating(true);
    try {
      const status = await api.getProofStatus();
      const updatedStatus = await api.updateProofSettings({
        enabled,
        timeout: status.lean4_proof_timeout ?? 120,
        lean4_lsp_enabled: Boolean(status.lean4_lsp_enabled),
        lean4_lsp_idle_timeout: status.lean4_lsp_idle_timeout ?? 600,
        max_parallel_candidates: status.proof_max_parallel_candidates ?? 6,
        smt_enabled: Boolean(status.smt_enabled),
        smt_timeout: status.smt_timeout ?? 30,
      });
      if (enabled) {
        const leanVersion = String(updatedStatus.lean4_version || updatedStatus.lean_version || '').trim();
        const leanVersionUnavailable = !leanVersion || /not found|no such file|not recognized/i.test(leanVersion);
        // A cold Mathlib sanity check can exceed the short status timeout even when
        // Lean is usable. Workflow proof stages wait on the real workspace check.
        if (!updatedStatus.lean4_enabled || leanVersionUnavailable) {
          alert(updatedStatus.manual_check_message || 'Lean 4 proof output is not ready. Check Lean 4 runtime settings before starting proof output.');
          return false;
        }
      }
      return true;
    } catch (error) {
      alert(`Failed to update Lean 4 proof setting: ${error.message}`);
      return false;
    } finally {
      setProofOutputUpdating(false);
    }
  };

  const updateAllowedOutput = async (key, checked) => {
    const nextConfig = {
      ...config,
      allow_mathematical_proofs: config?.allow_mathematical_proofs ?? true,
      allow_research_papers: config?.allow_research_papers ?? true,
      [key]: checked
    };

    if (!nextConfig.allow_mathematical_proofs && !nextConfig.allow_research_papers) {
      alert('At least one allowed output must remain enabled.');
      return;
    }

    if (key === 'allow_mathematical_proofs') {
      const updated = await updateProofRuntimeSetting(checked);
      if (!updated) {
        return;
      }
    }

    onConfigChange?.(nextConfig);
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
    setIsForcingTier3(true);
    
    try {
      // Fire and forget - API returns immediately, Tier 3 runs in background
      const response = await api.forceTier3(mode);
      if (!response.success) {
        alert(`Failed to start Tier 3: ${response.message || 'Unknown error'}`);
      }
      // Success message will come through WebSocket activity feed
    } catch (error) {
      alert(`Failed to force Tier 3: ${error.details || error.message}`);
    } finally {
      setIsForcingTier3(false);
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

  return (
    <div className={`autonomous-interface workflow-main-interface ${isRunning || isStopping ? 'workflow-main-interface--running' : ''}`}>
      {/* Header */}
      <div className="autonomous-header">
        <h2>Autonomous Research</h2>
        <div className="autonomous-controls-stack">
          <div className="autonomous-controls">
            {!isRunning && !isStopping ? (
              <button
                className="btn-start"
                onClick={handleStart}
                disabled={
                  !config?.submitter_configs?.some(s => s.modelId) ||
                  (anyWorkflowRunning && !isRunning)
                }
              >
                Start Research
              </button>
            ) : (
              <>
                <span
                  className="runtime-indicator"
                  role="status"
                  aria-live="polite"
                  title={isStopping ? "Autonomous research is stopping" : "Autonomous research is currently running"}
                >
                  <span className="runtime-indicator-dot" aria-hidden="true"></span>
                  <span className="runtime-indicator-label">{isStopping ? 'Stopping' : 'Running'}</span>
                </span>
                <button className="btn-stop" onClick={onStop} disabled={isStopping}>
                  {isStopping ? 'Stopping...' : 'Stop Research'}
                </button>
              </>
            )}
            {developerModeEnabled && (
              <label className="settings-checkbox-label">
                <input
                  type="checkbox"
                  checked={Boolean(config?.creativity_emphasis_boost_enabled)}
                  onChange={(event) => onConfigChange?.({
                    ...config,
                    creativity_emphasis_boost_enabled: event.target.checked
                  })}
                  disabled={isRunning || isStopping}
                />
                Creativity Emphasis Boost
              </label>
            )}
              <button
              className={`btn-clear ${showClearConfirm ? 'btn-confirm' : ''}`}
              onClick={handleClear}
              disabled={isRunning || isStopping || isClearing}
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
          <div
            className="allowed-outputs-row"
            title="Allowed Outputs controls which products this workflow may generate. At least one output must remain enabled."
          >
            <span className="allowed-outputs-label">Allowed Outputs:</span>
            <label
              className="allowed-output-option"
              title="Mathematical Proofs enables Lean 4 proof verification and proof-library output for this run."
            >
              <input
                type="checkbox"
                checked={proofOutputsAvailable && Boolean(config?.allow_mathematical_proofs ?? true)}
                onChange={(event) => updateAllowedOutput('allow_mathematical_proofs', event.target.checked)}
                disabled={isRunning || isStopping || proofOutputUpdating || !proofOutputsAvailable}
              />
              <span className="allowed-output-text">Mathematical Proofs</span>
            </label>
            <label
              className="allowed-output-option"
              title="Research Papers enables paper compilation output. When disabled, autonomous research loops through brainstorms and proof checks without writing papers."
            >
              <input
                type="checkbox"
                checked={Boolean(config?.allow_research_papers ?? true)}
                onChange={(event) => updateAllowedOutput('allow_research_papers', event.target.checked)}
                disabled={isRunning || isStopping}
              />
              <span className="allowed-output-text">Research Papers</span>
            </label>
          </div>
        </div>
      </div>

      {/* Research Prompt Input */}
      <div className="research-prompt-section">
        <label htmlFor="research-prompt">Research Goal</label>
        <textarea
          id="research-prompt"
          value={researchPrompt}
          onChange={(e) => setResearchPrompt(e.target.value)}
          placeholder="Enter your high level research goal on any topic that relates to S.T.E.M. mathematics, anything remotely related to mathematics (e.g., 'Advance desalination technology' or 'Solve physics unification')"
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

        {explorationProgress && (
          <div className="current-brainstorm" style={{ borderLeft: '3px solid #18cc17' }}>
            <span className="status-label">Topic Exploration:</span>
            <p className="brainstorm-prompt" style={{ color: '#c4b5fd' }}>
              Brainstorming candidate directions ({explorationProgress.accepted}/{explorationProgress.target} accepted)
            </p>
            <div className="brainstorm-stats">
              <span className="submission-count accepted">
                ◈ {explorationProgress.accepted} / {explorationProgress.target} candidates validated
              </span>
            </div>
          </div>
        )}

        {titleExplorationProgress && (
          <div className="current-brainstorm" style={{ borderLeft: '3px solid #f59e0b' }}>
            <span className="status-label">Title Exploration:</span>
            <p className="brainstorm-prompt" style={{ color: '#7dff6f' }}>
              Exploring candidate paper titles ({titleExplorationProgress.accepted}/{titleExplorationProgress.target} accepted)
            </p>
            <div className="brainstorm-stats">
              <span className="submission-count accepted">
                ◈ {titleExplorationProgress.accepted} / {titleExplorationProgress.target} titles validated
              </span>
            </div>
          </div>
        )}

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
                        <span className="force-paper-action">Skip AI Autonomy</span>
                        <span className="force-paper-hint" role="tooltip">
                          We recommend at minimum 5 ACCEPTED submissions. A very low acceptance rate is normal because the validator is seeking novel solutions. Higher parameter models may improve acceptance, though optimizing for both speed and knowledge can also work well. Validator feedback on rejections helps avoid rejection loops. Harder problems may require hundreds or more rejections before a single acceptance, and the first acceptance often takes the longest. View brainstorms in the brainstorm tab.
                        </span>
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
            border: critiquePhaseActive ? '2px solid #1eff1c' : '2px solid #666',
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
              <strong style={{ color: critiquePhaseActive ? '#1eff1c' : '#ccc', fontSize: '1.1rem' }}>
                {critiquePhaseActive ? 'Critique Phase in Progress' : 'Paper Writing in Progress'}
              </strong>
              {critiquePhaseActive ? (
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Collecting peer review feedback on the body section...
                </p>
              ) : (
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Constructing paper sections...
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Force Tier 3 Button - Only visible when running, tier3 enabled, and not already in Tier 3 */}
      {isRunning && config?.tier3_enabled && !status?.is_tier3_active && status?.stats?.total_papers_completed > 0 && (
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
          capabilities={capabilities}
        />
      )}

      {/* Live Tier 3 Progress (Final Answer) */}
      {status?.current_tier === 'tier3_final_answer' && api && (
        <LiveTier3Progress 
          api={api}
          status={status}
          capabilities={capabilities}
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
        <div className="activity-feed" ref={activityFeedRef}>
          {activity.length === 0 ? (
            <div className="activity-empty">
              No activity yet. Wait about 20 to 30 minutes. If you have not yet, press the start button above your prompt entry to begin research.
            </div>
          ) : (
            activity.map((item, index) => (
              <div 
                key={index} 
                className={`activity-item ${getSharedActivityClass(item.event)}`}
              >
                <span className="activity-icon">{getSharedActivityIcon(item.event)}</span>
                <span className="activity-time">
                  {new Date(item.timestamp).toLocaleTimeString()}
                </span>
                <span className="activity-message">{item.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default AutonomousResearchInterface;

