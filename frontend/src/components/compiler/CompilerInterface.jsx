import React, { useState, useEffect } from 'react';
import { compilerAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import TextFileUploader from '../TextFileUploader';

function CompilerInterface({ activeTab }) {
  const [compilerPrompt, setCompilerPrompt] = useState('');
  const [status, setStatus] = useState({ is_running: false });
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(131072);
  const [highContextContextSize, setHighContextContextSize] = useState(131072);
  const [highParamContextSize, setHighParamContextSize] = useState(131072);
  const [critiqueSubmitterContextSize, setCritiqueSubmitterContextSize] = useState(131072);
  const [critiquePhaseActive, setCritiquePhaseActive] = useState(false);
  const [critiqueAcceptances, setCritiqueAcceptances] = useState(0);
  const [paperVersion, setPaperVersion] = useState(1);
  const [isSkipping, setIsSkipping] = useState(false);
  const [skipQueued, setSkipQueued] = useState(false);

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
    
    const handleBodyRewriteStarted = (data) => {
      setPaperVersion(data.version || 1);
      setSkipQueued(false);  // Reset skip state for new paper version
    };
    
    websocket.on('critique_phase_started', handleCritiquePhaseStarted);
    websocket.on('critique_progress', handleCritiqueProgress);
    websocket.on('critique_phase_ended', handleCritiquePhaseEnded);
    websocket.on('critique_phase_skipped', handleCritiquePhaseSkipped);
    websocket.on('body_rewrite_started', handleBodyRewriteStarted);
    
    return () => {
      clearInterval(interval);
      websocket.off('critique_phase_started', handleCritiquePhaseStarted);
      websocket.off('critique_progress', handleCritiqueProgress);
      websocket.off('critique_phase_ended', handleCritiquePhaseEnded);
      websocket.off('critique_phase_skipped', handleCritiquePhaseSkipped);
      websocket.off('body_rewrite_started', handleBodyRewriteStarted);
    };
  }, []);

  // Reload settings when tab becomes active
  useEffect(() => {
    if (activeTab === 'compiler-interface') {
      loadSettings();
    }
  }, [activeTab]);

  const loadSettings = () => {
    const savedSettings = localStorage.getItem('compiler_settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        if (settings.validatorContextSize) setValidatorContextSize(settings.validatorContextSize);
        if (settings.highContextContextSize) setHighContextContextSize(settings.highContextContextSize);
        if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
        if (settings.critiqueSubmitterContextSize) setCritiqueSubmitterContextSize(settings.critiqueSubmitterContextSize);
        // Store for use in handleStart
        window.compilerSettings = settings;
      } catch (error) {
        console.error('Failed to load compiler settings:', error);
      }
    }
  };

  const loadStatus = async () => {
    try {
      const response = await compilerAPI.getStatus();
      setStatus(response.data);
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
        validator_provider: settings.validatorProvider || 'lm_studio',
        validator_model: settings.validatorModel,
        validator_openrouter_provider: settings.validatorOpenrouterProvider || null,
        validator_lm_studio_fallback: settings.validatorLmStudioFallback || null,
        validator_context_size: settings.validatorContextSize || validatorContextSize,
        validator_max_output_tokens: settings.validatorMaxOutput || 25000,
        // High-context submitter config with OpenRouter support
        high_context_provider: settings.highContextProvider || 'lm_studio',
        high_context_model: settings.highContextModel,
        high_context_openrouter_provider: settings.highContextOpenrouterProvider || null,
        high_context_lm_studio_fallback: settings.highContextLmStudioFallback || null,
        high_context_context_size: settings.highContextContextSize || highContextContextSize,
        high_context_max_output_tokens: settings.highContextMaxOutput || 25000,
        // High-param submitter config with OpenRouter support
        high_param_provider: settings.highParamProvider || 'lm_studio',
        high_param_model: settings.highParamModel,
        high_param_openrouter_provider: settings.highParamOpenrouterProvider || null,
        high_param_lm_studio_fallback: settings.highParamLmStudioFallback || null,
        high_param_context_size: settings.highParamContextSize || highParamContextSize,
        high_param_max_output_tokens: settings.highParamMaxOutput || 25000,
        // Critique submitter config with OpenRouter support
        critique_submitter_provider: settings.critiqueSubmitterProvider || 'lm_studio',
        critique_submitter_model: settings.critiqueSubmitterModel,
        critique_submitter_openrouter_provider: settings.critiqueSubmitterOpenrouterProvider || null,
        critique_submitter_lm_studio_fallback: settings.critiqueSubmitterLmStudioFallback || null,
        critique_submitter_context_window: settings.critiqueSubmitterContextSize || critiqueSubmitterContextSize,
        critique_submitter_max_tokens: settings.critiqueSubmitterMaxOutput || 25000
      });
      
      await loadStatus();
    } catch (err) {
      console.error('Failed to start compiler:', err);
      
      // Handle structured error response
      if (err.details && typeof err.details === 'object') {
        setError(err.details);
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
    <div className="compiler-interface">
      <h2>Compiler Interface</h2>
      
      <div className="status-indicator">
        <span className={`status-badge ${status.is_running ? 'running' : 'stopped'}`}>
          {status.is_running ? '● Running' : '○ Stopped'}
        </span>
        {status.current_mode && status.current_mode !== 'idle' && (
          <span className="mode-badge">Mode: {status.current_mode}</span>
        )}
      </div>

      {status.is_running && (
        <div className="critique-phase-banner" style={{
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
              {critiquePhaseActive ? `Critique Phase (Version ${paperVersion})` : 'Paper Writing in Progress'}
            </strong>
            {critiquePhaseActive ? (
              <>
                <p style={{ margin: '0.25rem 0 0 0', color: '#ccc' }}>
                  {critiqueAcceptances} / 10 critiques accepted
                </p>
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                  Collecting peer review feedback on body section...
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
              <p><strong>💡 Suggestion:</strong> {error.suggestion}</p>
            </div>
          )}
        </div>
      )}

      <div className="form-group">
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

      <div className="form-group">
        <label htmlFor="contextSizeDisplay">Context Window Sizes:</label>
        <div className="context-size-display">
          <div><strong>Validator:</strong> {validatorContextSize.toLocaleString()} tokens</div>
          <div><strong>High-Context:</strong> {highContextContextSize.toLocaleString()} tokens</div>
          <div><strong>High-Parameter:</strong> {highParamContextSize.toLocaleString()} tokens</div>
          <div><strong>Critique Submitter:</strong> {critiqueSubmitterContextSize.toLocaleString()} tokens</div>
          <small style={{marginTop: '0.5rem', display: 'block', color: '#666'}}>
            (Change these in the Compiler Settings tab)
          </small>
        </div>
      </div>

      <div className="button-group">
        {!status.is_running ? (
          <button 
            onClick={handleStart} 
            className="btn btn-primary"
            disabled={isStarting}
          >
            {isStarting ? 'Starting...' : 'Start Compiler'}
          </button>
        ) : (
          <button 
            onClick={handleStop} 
            className="btn btn-danger"
          >
            Stop Compiler
          </button>
        )}
      </div>

      <div className="info-section">
        <h3>Aggregator Database</h3>
        <p>The compiler will read from the aggregator's accepted submissions database.</p>
        <p>Location: <code>backend/data/rag_shared_training.txt</code></p>
      </div>
    </div>
  );
}

export default CompilerInterface;
