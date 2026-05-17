import React, { useState, useEffect } from 'react';
import { compilerAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
} from '../../utils/openRouterSelection';
import TextFileUploader from '../TextFileUploader';
import { getRuntimeDataPath } from '../../utils/runtimeConfig';
import '../autonomous/AutonomousResearch.css';

function CompilerInterface({
  activeTab,
  capabilities,
  anyWorkflowRunning = false,
  onWorkflowRunningChange = null,
  developerModeEnabled = false,
}) {
  const [compilerPrompt, setCompilerPrompt] = useState('');
  const [status, setStatus] = useState({ is_running: false });
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highContextContextSize, setHighContextContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highParamContextSize, setHighParamContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [critiqueSubmitterContextSize, setCritiqueSubmitterContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [critiquePhaseActive, setCritiquePhaseActive] = useState(false);
  const [critiqueAcceptances, setCritiqueAcceptances] = useState(0);
  const [paperVersion, setPaperVersion] = useState(1);
  const [isSkipping, setIsSkipping] = useState(false);
  const [skipQueued, setSkipQueued] = useState(false);
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;

  const normalizeCompilerSettingsForCapabilities = (settings = {}) => {
    if (lmStudioEnabled) {
      return settings;
    }

    const nextSettings = { ...settings };
    const rolePrefixes = ['validator', 'highContext', 'highParam', 'critiqueSubmitter'];

    rolePrefixes.forEach((rolePrefix) => {
      const providerKey = `${rolePrefix}Provider`;
      const modelKey = `${rolePrefix}Model`;
      const openRouterProviderKey = `${rolePrefix}OpenrouterProvider`;
      const fallbackKey = `${rolePrefix}LmStudioFallback`;
      const keepOpenRouterState = nextSettings[providerKey] === 'openrouter';

      nextSettings[providerKey] = 'openrouter';
      nextSettings[modelKey] = keepOpenRouterState ? (nextSettings[modelKey] || '') : '';
      nextSettings[openRouterProviderKey] = keepOpenRouterState
        ? (nextSettings[openRouterProviderKey] || null)
        : null;
      nextSettings[fallbackKey] = null;
    });

    return nextSettings;
  };

  useEffect(() => {
    loadStatus();
    loadSettings();
    
    // Poll status every 5 seconds
    const interval = setInterval(loadStatus, 5000);
    
    // Listen for critique phase events via WebSocket
    const handleCritiquePhaseStarted = (data) => {
      setCritiquePhaseActive(true);
      setPaperVersion(data.paper_version || 1);
      setCritiqueAcceptances(0);
    };
    
    const handleCritiqueProgress = (data) => {
      setCritiqueAcceptances(data.acceptances || 0);
      setPaperVersion(data.version || 1);
    };
    
    const handleCritiquePhaseEnded = (data) => {
      setCritiquePhaseActive(false);
      // Don't reset skipQueued - if skip was queued, it worked
    };
    
    const handleCritiquePhaseSkipped = (data) => {
      setCritiquePhaseActive(false);
      // Skip worked! Keep skipQueued=true to show checkmark
    };
    
    websocket.on('critique_phase_started', handleCritiquePhaseStarted);
    websocket.on('critique_progress', handleCritiqueProgress);
    websocket.on('critique_phase_ended', handleCritiquePhaseEnded);
    websocket.on('critique_phase_skipped', handleCritiquePhaseSkipped);
    
    return () => {
      clearInterval(interval);
      websocket.off('critique_phase_started', handleCritiquePhaseStarted);
      websocket.off('critique_progress', handleCritiqueProgress);
      websocket.off('critique_phase_ended', handleCritiquePhaseEnded);
      websocket.off('critique_phase_skipped', handleCritiquePhaseSkipped);
    };
  }, []);

  // Reload settings when tab becomes active
  useEffect(() => {
    if (activeTab === 'compiler-interface') {
      loadSettings();
    }
  }, [activeTab, lmStudioEnabled]);

  const loadSettings = () => {
    const savedSettings = localStorage.getItem('compiler_settings');
    if (savedSettings) {
      try {
        const settings = normalizeCompilerSettingsForCapabilities(JSON.parse(savedSettings));
        if (settings.validatorContextSize) setValidatorContextSize(settings.validatorContextSize);
        if (settings.highContextContextSize) setHighContextContextSize(settings.highContextContextSize);
        if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
        if (settings.critiqueSubmitterContextSize) setCritiqueSubmitterContextSize(settings.critiqueSubmitterContextSize);
        // Store for use in handleStart
        window.compilerSettings = settings;
        if (!lmStudioEnabled) {
          localStorage.setItem('compiler_settings', JSON.stringify(settings));
        }
      } catch (error) {
        console.error('Failed to load compiler settings:', error);
      }
    }
  };

  const loadStatus = async () => {
    try {
      const response = await compilerAPI.getStatus();
      setStatus(response.data);
      if (response.data.is_running) {
        onWorkflowRunningChange?.(true);
      }
      // Update critique phase state from status
      if (response.data.in_critique_phase !== undefined) {
        setCritiquePhaseActive(response.data.in_critique_phase);
        setCritiqueAcceptances(response.data.critique_acceptances || 0);
        setPaperVersion(response.data.paper_version || 1);
      }
      // Reset skip state when not running
      if (!response.data.is_running) {
        setSkipQueued(false);
      }
    } catch (error) {
      console.error('Failed to load status:', error);
    }
  };

  const handleTextFileLoaded = (content) => {
    // Append to existing prompt with separator
    const separator = compilerPrompt.trim() ? '\n\n' : '';
    const newPrompt = compilerPrompt + separator + content;
    setCompilerPrompt(newPrompt);
  };

  const handleStart = async () => {
    if (anyWorkflowRunning && !status.is_running) {
      setError({
        error: 'workflow_conflict',
        reason: 'Another workflow is already running. Stop it before starting the Compiler.',
        suggestion: 'Only one workflow mode may be active at a time.'
      });
      return;
    }

    if (!compilerPrompt.trim()) {
      alert('Please enter a compiler-directing prompt');
      return;
    }

    const settings = window.compilerSettings || {};
    
    // Check if models are configured in settings
    if (!settings.validatorModel || !settings.highContextModel || !settings.highParamModel || !settings.critiqueSubmitterModel) {
      alert('Please configure all four models in the Compiler Settings tab first (validator, high-context, high-param, critique submitter)');
      return;
    }

    setIsStarting(true);
    setError(null);
    try {
      await compilerAPI.start({
        compiler_prompt: compilerPrompt,
        // Validator config with OpenRouter support
        validator_provider: lmStudioEnabled ? (settings.validatorProvider || 'lm_studio') : 'openrouter',
        validator_model: settings.validatorModel,
        validator_openrouter_provider: settings.validatorOpenrouterProvider || null,
        validator_openrouter_reasoning_effort: settings.validatorOpenrouterReasoningEffort || 'auto',
        validator_lm_studio_fallback: lmStudioEnabled ? (settings.validatorLmStudioFallback || null) : null,
        validator_context_size: settings.validatorContextSize || validatorContextSize,
        validator_max_output_tokens: settings.validatorMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS,
        validator_supercharge_enabled: developerModeEnabled && Boolean(settings.validatorSuperchargeEnabled),
        // High-context submitter config with OpenRouter support
        high_context_provider: lmStudioEnabled ? (settings.highContextProvider || 'lm_studio') : 'openrouter',
        high_context_model: settings.highContextModel,
        high_context_openrouter_provider: settings.highContextOpenrouterProvider || null,
        high_context_openrouter_reasoning_effort: settings.highContextOpenrouterReasoningEffort || 'auto',
        high_context_lm_studio_fallback: lmStudioEnabled ? (settings.highContextLmStudioFallback || null) : null,
        high_context_context_size: settings.highContextContextSize || highContextContextSize,
        high_context_max_output_tokens: settings.highContextMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS,
        high_context_supercharge_enabled: developerModeEnabled && Boolean(settings.highContextSuperchargeEnabled),
        // High-param submitter config with OpenRouter support
        high_param_provider: lmStudioEnabled ? (settings.highParamProvider || 'lm_studio') : 'openrouter',
        high_param_model: settings.highParamModel,
        high_param_openrouter_provider: settings.highParamOpenrouterProvider || null,
        high_param_openrouter_reasoning_effort: settings.highParamOpenrouterReasoningEffort || 'auto',
        high_param_lm_studio_fallback: lmStudioEnabled ? (settings.highParamLmStudioFallback || null) : null,
        high_param_context_size: settings.highParamContextSize || highParamContextSize,
        high_param_max_output_tokens: settings.highParamMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS,
        high_param_supercharge_enabled: developerModeEnabled && Boolean(settings.highParamSuperchargeEnabled),
        // Critique submitter config with OpenRouter support
        critique_submitter_provider: lmStudioEnabled
          ? (settings.critiqueSubmitterProvider || 'lm_studio')
          : 'openrouter',
        critique_submitter_model: settings.critiqueSubmitterModel,
        critique_submitter_openrouter_provider: settings.critiqueSubmitterOpenrouterProvider || null,
        critique_submitter_openrouter_reasoning_effort: settings.critiqueSubmitterOpenrouterReasoningEffort || 'auto',
        critique_submitter_lm_studio_fallback: lmStudioEnabled
          ? (settings.critiqueSubmitterLmStudioFallback || null)
          : null,
        critique_submitter_context_window: settings.critiqueSubmitterContextSize || critiqueSubmitterContextSize,
        critique_submitter_max_tokens: settings.critiqueSubmitterMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS,
        critique_submitter_supercharge_enabled: developerModeEnabled && Boolean(settings.critiqueSubmitterSuperchargeEnabled)
      });
      onWorkflowRunningChange?.(true);
      
      await loadStatus();
    } catch (err) {
      console.error('Failed to start compiler:', err);
      
      // Handle structured error response
      if (err.details && typeof err.details === 'object') {
        setError(err.details);
      } else if (typeof err.details === 'string' && err.details.trim()) {
        setError({
          error: 'workflow_conflict',
          reason: err.details,
          suggestion: 'Stop the active workflow and try again.'
        });
      } else {
        setError({
          error: 'unknown',
          reason: err.message,
          suggestion: 'Please check your model selection in the Compiler Settings tab and try again.'
        });
      }
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    try {
      await compilerAPI.stop();
      setSkipQueued(false);  // Reset skip state when compiler stops
      onWorkflowRunningChange?.(false);
      await loadStatus();
    } catch (error) {
      console.error('Failed to stop compiler:', error);
      alert('Failed to stop compiler: ' + error.message);
    }
  };

  const handleSkipCritique = async () => {
    if (!confirm('Skip the critique phase and continue to writing the conclusion? This cannot be undone.')) {
      return;
    }
    
    setIsSkipping(true);
    try {
      await compilerAPI.skipCritique();
      setSkipQueued(true);  // Mark skip as successfully queued
      await loadStatus(); // Reload status to reflect phase transition
    } catch (error) {
      alert('Failed to skip critique: ' + error.message);
    } finally {
      setIsSkipping(false);
    }
  };

  const getModelTypeLabel = (type) => {
    const labels = {
      validator: 'Validator Model',
      high_context: 'High-Context Model',
      high_param: 'High-Parameter Model'
    };
    return labels[type] || type;
  };

  return (
    <div className={`autonomous-interface compiler-interface workflow-main-interface ${status.is_running ? 'workflow-main-interface--running' : ''}`}>
      <div className="autonomous-header">
        <div>
          <h2>Single Paper Writer</h2>
          <p className="settings-hint">
            Compile the accepted aggregator database into one live mathematical paper.
          </p>
        </div>
        <div className="autonomous-controls">
          {!status.is_running ? (
            <button
              onClick={handleStart}
              className="btn-start"
              disabled={isStarting || (anyWorkflowRunning && !status.is_running)}
            >
              {isStarting ? 'Starting...' : 'Start Writer'}
            </button>
          ) : (
            <>
              <span className="runtime-indicator" role="status" aria-live="polite" title="Single paper writer is running">
                <span className="runtime-indicator-dot" aria-hidden="true"></span>
                <span className="runtime-indicator-label">Running</span>
              </span>
              <button
                onClick={handleStop}
                className="btn-stop"
              >
                Stop Writer
              </button>
            </>
          )}
        </div>
      </div>

      <div className="status-section">
        <div className="status-tier">
          <span className="status-label">Current Status:</span>
          <span className={`status-value ${status.is_running ? 'status-running' : 'status-idle'}`}>
            {status.is_running ? 'Paper Writing' : 'Not Running'}
          </span>
          {status.current_mode && status.current_mode !== 'idle' && (
            <span className="mode-badge">Mode: {status.current_mode}</span>
          )}
        </div>
      </div>

      {status.is_running && (
        <div className="critique-phase-banner" style={{
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
              {critiquePhaseActive ? `Critique Phase (Version ${paperVersion})` : 'Paper Writing in Progress'}
            </strong>
            {critiquePhaseActive ? (
              <>
                <p style={{ margin: '0.25rem 0 0 0', color: '#ccc' }}>
                  {critiqueAcceptances} / 10 critiques accepted
                </p>
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Collecting peer review feedback on the body section...
                </p>
              </>
            ) : (
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                Mode: {status.current_mode || 'constructing'}
              </p>
            )}
          </div>
          {/* Skip button - ALWAYS visible during paper writing */}
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

      {error && (
        <div className="error-box">
          <div className="error-header">
            <h3 className="error-title">Model Compatibility Error</h3>
            <button onClick={() => setError(null)} className="error-dismiss">×</button>
          </div>
          <div className="error-details">
            {error.failed_model_type && (
              <p><strong>Failed Model:</strong> {getModelTypeLabel(error.failed_model_type)} ({error.failed_model_name})</p>
            )}
            <p><strong>Reason:</strong> {error.reason}</p>
          </div>
          {error.suggestion && (
            <div className="error-suggestion">
              <p><strong>Suggestion:</strong> {error.suggestion}</p>
            </div>
          )}
        </div>
      )}

      <div className="research-prompt-section">
        <label htmlFor="compilerPrompt">Compiler-Directing Prompt:</label>
        <textarea
          id="compilerPrompt"
          value={compilerPrompt}
          onChange={(e) => setCompilerPrompt(e.target.value)}
          placeholder='Create a final prompt that exactly relates to a solution your aggregation database helps solve, i.e. "Tell me the most theoretically advanced perspective on how squaring the circle works."'
          rows={6}
          disabled={status.is_running}
        />
        <TextFileUploader 
          onFileLoaded={handleTextFileLoaded}
          disabled={status.is_running}
          maxSizeMB={5}
          showCharCount={true}
          confirmIfNotEmpty={true}
          existingPromptLength={compilerPrompt.length}
        />
        <small>This prompt directs the compiler on what kind of mathematical document to create from the aggregated database. View your in-progress and final answer in the "Live Paper" tab.</small>
      </div>

      <div className="stats-section">
        <div className="stat-item">
          <span className="stat-value">{validatorContextSize.toLocaleString()}</span>
          <span className="stat-label">Validator Tokens</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{highContextContextSize.toLocaleString()}</span>
          <span className="stat-label">High-Context Tokens</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{highParamContextSize.toLocaleString()}</span>
          <span className="stat-label">High-Param Tokens</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{critiqueSubmitterContextSize.toLocaleString()}</span>
          <span className="stat-label">Critique Tokens</span>
        </div>
      </div>

      <div className="status-section">
        <h3>Aggregator Database</h3>
        <p>The compiler will read from the aggregator's accepted submissions database.</p>
        <p>Location: <code>{getRuntimeDataPath('rag_shared_training.txt')}</code></p>
      </div>
    </div>
  );
}

export default CompilerInterface;
