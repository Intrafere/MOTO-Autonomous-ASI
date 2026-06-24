import React, { useEffect, useState } from 'react';
import { persistLeanOJSettings, settingsToLeanOJRequest } from '../../utils/leanojProfiles';
import LiveActivityFeed from '../LiveActivityFeed';
import '../autonomous/AutonomousResearch.css';
import '../settings-common.css';

export default function LeanOJInterface({
  isRunning,
  anyWorkflowRunning,
  status,
  activity,
  settings,
  onSettingsChange,
  onStart,
  onStop,
  onClear,
  onSkipBrainstorm,
  onForceBrainstorm,
  developerModeEnabled = false,
  assistantMemoryEnabled = true,
}) {
  const [prompt, setPrompt] = useState(settings.prompt || '');
  const [leanTemplate, setLeanTemplate] = useState(settings.leanTemplate || '');

  useEffect(() => {
    setPrompt(settings.prompt || '');
    setLeanTemplate(settings.leanTemplate || '');
  }, [settings.prompt, settings.leanTemplate]);

  const persistDraft = (nextPrompt, nextLeanTemplate) => {
    const nextSettings = persistLeanOJSettings({
      ...settings,
      prompt: nextPrompt,
      leanTemplate: nextLeanTemplate,
    });
    onSettingsChange(nextSettings);
    return nextSettings;
  };

  const handlePromptChange = (value) => {
    setPrompt(value);
    persistDraft(value, leanTemplate);
  };

  const handleLeanTemplateChange = (value) => {
    setLeanTemplate(value);
    persistDraft(prompt, value);
  };

  const handleStart = async () => {
    try {
      const nextSettings = persistDraft(prompt, leanTemplate);
      await onStart(settingsToLeanOJRequest(nextSettings, prompt, leanTemplate, { assistantEnabled: assistantMemoryEnabled }));
    } catch (error) {
      alert(error.message || 'Failed to start Proof Solver');
    }
  };

  const canStart = !isRunning && !anyWorkflowRunning && prompt.trim() && leanTemplate.trim();
  const disabledReason = anyWorkflowRunning && !isRunning
    ? 'Another workflow is already running.'
    : 'Enter a problem prompt and Lean template.';

  return (
    <div className={`autonomous-interface workflow-main-interface ${isRunning ? 'workflow-main-interface--running' : ''}`}>
      <div className="autonomous-header leanoj-header">
        <div className="leanoj-header-copy">
          <h2>Proof Solver</h2>
          <p className="settings-hint leanoj-proof-solver-intro">
            Paste a proof problem statement and Lean template. MOTO will build cumulative brainstorm context, allow Lean-verified proof fragments during brainstorming, and keep trying the final Lean 4 submission until Lean verifies it or you stop the run.
          </p>
        </div>
        <div className="autonomous-controls">
          {!isRunning ? (
            <button className="btn-start" onClick={handleStart} disabled={!canStart} title={!canStart ? disabledReason : ''}>
              Start Proof Solver
            </button>
          ) : (
            <>
              <span className="runtime-indicator" role="status" aria-live="polite" title="Proof Solver is running">
                <span className="runtime-indicator-dot" aria-hidden="true"></span>
                <span className="runtime-indicator-label">Running</span>
              </span>
              <button className="btn-stop" onClick={onStop}>
                Stop Proof Solver
              </button>
            </>
          )}
          {developerModeEnabled && (
            <label className="settings-checkbox-label">
              <input
                type="checkbox"
                checked={Boolean(settings.creativityEmphasisBoostEnabled)}
                onChange={(event) => onSettingsChange(persistLeanOJSettings({
                  ...settings,
                  creativityEmphasisBoostEnabled: event.target.checked,
                }))}
                disabled={isRunning}
              />
              Creativity Emphasis Boost
            </label>
          )}
          <button className="btn-clear" onClick={onSkipBrainstorm} disabled={!isRunning}>
            Skip Brainstorm
          </button>
          <button className="btn-clear" onClick={onForceBrainstorm} disabled={!isRunning}>
            Force Brainstorm
          </button>
          <button className="btn-clear" onClick={onClear} disabled={isRunning}>
            Clear Progress
          </button>
        </div>
      </div>

      <div className="research-prompt-section">
        <label htmlFor="leanoj-problem-prompt">Problem Prompt</label>
        <textarea
          id="leanoj-problem-prompt"
          value={prompt}
          onChange={(event) => handlePromptChange(event.target.value)}
          disabled={isRunning}
          rows={6}
          placeholder="Describe the proof problem, constraints, and what the template expects."
        />
      </div>

      <div className="research-prompt-section">
        <label htmlFor="leanoj-template">Lean Template</label>
        <textarea
          id="leanoj-template"
          value={leanTemplate}
          onChange={(event) => handleLeanTemplateChange(event.target.value)}
          disabled={isRunning}
          rows={14}
          spellCheck={false}
          placeholder={'import Mathlib\n\nexample ... := by\n  sorry'}
        />
      </div>

      <div className="status-section">
        <div className="status-tier">
          <span className="status-label">Current Status:</span>
          <span className={`status-value ${isRunning ? 'status-running' : 'status-idle'}`}>
            {isRunning ? status?.phase || 'running' : 'Not Running'}
          </span>
        </div>
        {status?.current_path_decision && (
          <div className="current-brainstorm">
            <span className="status-label">Current Path:</span>
            <p className="brainstorm-prompt">{status.current_path_decision}</p>
          </div>
        )}
        {status?.provider_paused && (
          <div className="settings-notice">
            Proof Solver is paused until OpenRouter credits are reset. Add credits, then press Retry OpenRouter.
            {status.provider_pause_message ? ` ${status.provider_pause_message}` : ''}
          </div>
        )}
        {status?.last_error && (
          <div className="error-message">{status.last_error}</div>
        )}
      </div>

      <div className="stats-section">
        <div className="stat-item">
          <span className="stat-value">{status?.accepted_brainstorm_count || 0}</span>
          <span className="stat-label">Accepted Ideas</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{(status?.validated_topics || []).length}</span>
          <span className="stat-label">Validated Topics</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{(status?.verified_subproofs || []).length}</span>
          <span className="stat-label">Verified Proof Fragments</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{status?.final_attempt_count || 0}</span>
          <span className="stat-label">Final Attempts</span>
        </div>
      </div>

      {status?.final_solution && (
        <div className="status-section">
          <h3>Verified Proof Solver Submission</h3>
          <pre className="code-block">{status.final_solution}</pre>
        </div>
      )}

      <LiveActivityFeed
        title="Live Activity"
        items={activity || []}
        emptyMessage="No activity yet."
      />
    </div>
  );
}
