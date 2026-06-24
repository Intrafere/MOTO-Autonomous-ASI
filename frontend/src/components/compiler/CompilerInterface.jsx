import React, { useState, useEffect } from 'react';
import { autonomousAPI, compilerAPI } from '../../services/api';
import { websocket } from '../../services/websocket';
import {
  DEFAULT_CONTEXT_WINDOW,
} from '../../utils/openRouterSelection';
import TextFileUploader from '../TextFileUploader';
import { getRuntimeDataPath } from '../../utils/runtimeConfig';
import { readPromptDraft, readPromptDraftSync, savePromptDraft } from '../../utils/promptDraftStorage';
import '../autonomous/AutonomousResearch.css';

const COMPILER_PROMPT_STORAGE_KEY = 'compiler_prompt';
const LEGACY_WRITER_CAMEL_PREFIX = ['high', 'Context'].join('');

const isMeaningfulWriterSetting = (value, suffix) => {
  if (value === undefined || value === null) return false;
  if (typeof value === 'string' && value.trim() === '') return false;
  if ((suffix === 'ContextSize' || suffix === 'MaxOutput') && Number(value) <= 0) return false;
  return true;
};

const readWriterSetting = (settings, suffix) => {
  const current = settings[`writer${suffix}`];
  if (isMeaningfulWriterSetting(current, suffix)) {
    return current;
  }
  return settings[`${LEGACY_WRITER_CAMEL_PREFIX}${suffix}`];
};

const migrateCompilerWriterSettings = (settings = {}) => ({
  ...settings,
  writerProvider: readWriterSetting(settings, 'Provider') || settings.writerProvider,
  writerModel: readWriterSetting(settings, 'Model') || settings.writerModel,
  writerOpenrouterProvider: readWriterSetting(settings, 'OpenrouterProvider') ?? settings.writerOpenrouterProvider,
  writerOpenrouterReasoningEffort: readWriterSetting(settings, 'OpenrouterReasoningEffort') || settings.writerOpenrouterReasoningEffort,
  writerLmStudioFallback: readWriterSetting(settings, 'LmStudioFallback') ?? settings.writerLmStudioFallback,
  writerContextSize: readWriterSetting(settings, 'ContextSize') ?? settings.writerContextSize,
  writerMaxOutput: readWriterSetting(settings, 'MaxOutput') ?? settings.writerMaxOutput,
  writerSuperchargeEnabled: readWriterSetting(settings, 'SuperchargeEnabled') ?? settings.writerSuperchargeEnabled,
});

function CompilerInterface({
  activeTab,
  capabilities,
  anyWorkflowRunning = false,
  onWorkflowRunningChange = null,
  developerModeEnabled = false,
  connectivityStatus = null,
}) {
  const [compilerPrompt, setCompilerPrompt] = useState(() => (
    readPromptDraftSync(COMPILER_PROMPT_STORAGE_KEY)
  ));
  const [compilerPromptDraftLoaded, setCompilerPromptDraftLoaded] = useState(false);
  const [status, setStatus] = useState({ is_running: false });
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [writerContextSize, setWritingContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highParamContextSize, setHighParamContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [critiquePhaseActive, setCritiquePhaseActive] = useState(false);
  const [critiqueAcceptances, setCritiqueAcceptances] = useState(0);
  const [paperVersion, setPaperVersion] = useState(1);
  const [proofOutputUpdating, setProofOutputUpdating] = useState(false);
  const [allowedOutputs, setAllowedOutputs] = useState(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem('compiler_allowed_outputs') || '{}');
      return {
        mathematicalProofs: parsed.mathematicalProofs ?? true,
        researchPapers: parsed.researchPapers ?? true,
      };
    } catch {
      return { mathematicalProofs: true, researchPapers: true };
    }
  });
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const proofOutputsAvailable = !capabilities?.genericMode;

  const normalizeCompilerSettingsForCapabilities = (settings = {}) => {
    if (lmStudioEnabled) {
      return settings;
    }

    const nextSettings = { ...settings };
    const rolePrefixes = ['validator', 'assistant', 'writer', 'highParam'];

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
    
    const handleCritiquePhaseEnded = () => {
      setCritiquePhaseActive(false);
    };
    
    websocket.on('critique_phase_started', handleCritiquePhaseStarted);
    websocket.on('critique_progress', handleCritiqueProgress);
    websocket.on('critique_phase_ended', handleCritiquePhaseEnded);
    
    return () => {
      clearInterval(interval);
      websocket.off('critique_phase_started', handleCritiquePhaseStarted);
      websocket.off('critique_progress', handleCritiqueProgress);
      websocket.off('critique_phase_ended', handleCritiquePhaseEnded);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem('compiler_allowed_outputs', JSON.stringify(allowedOutputs));
  }, [allowedOutputs]);

  useEffect(() => {
    if (compilerPromptDraftLoaded) {
      savePromptDraft(COMPILER_PROMPT_STORAGE_KEY, compilerPrompt);
    }
  }, [compilerPrompt, compilerPromptDraftLoaded]);

  useEffect(() => {
    let cancelled = false;
    const hydrateCompilerPrompt = async () => {
      try {
        const savedDraft = await readPromptDraft(COMPILER_PROMPT_STORAGE_KEY);
        if (savedDraft && !cancelled) {
          setCompilerPrompt((current) => (
            current.trim() ? current : savedDraft
          ));
        }

        const response = await compilerAPI.getPrompt();
        const persistedPrompt = response.data?.prompt || '';
        if (!persistedPrompt.trim() || cancelled) {
          return;
        }
        setCompilerPrompt((current) => (
          current.trim() ? current : persistedPrompt
        ));
      } catch (error) {
        console.debug('Could not hydrate manual Compiler prompt:', error);
      } finally {
        if (!cancelled) {
          setCompilerPromptDraftLoaded(true);
        }
      }
    };

    hydrateCompilerPrompt();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (proofOutputsAvailable || !allowedOutputs.mathematicalProofs) {
      return;
    }
    setAllowedOutputs((current) => ({
      ...current,
      mathematicalProofs: false,
      researchPapers: true,
    }));
  }, [proofOutputsAvailable, allowedOutputs.mathematicalProofs]);

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
        const settings = normalizeCompilerSettingsForCapabilities(
          migrateCompilerWriterSettings(JSON.parse(savedSettings))
        );
        if (settings.validatorContextSize) setValidatorContextSize(settings.validatorContextSize);
        if (settings.writerContextSize) setWritingContextSize(settings.writerContextSize);
        if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
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
    const mathematicalProofsAllowed = proofOutputsAvailable && allowedOutputs.mathematicalProofs;
    const researchPapersAllowed = allowedOutputs.researchPapers;
    if (!mathematicalProofsAllowed && !researchPapersAllowed) {
      alert('Please allow at least one output: Mathematical Proofs or Research Papers.');
      return;
    }
    const proofOnlyRequested = mathematicalProofsAllowed && !researchPapersAllowed;
    const shouldSyncProofRuntime = mathematicalProofsAllowed;
    if (proofOnlyRequested || shouldSyncProofRuntime) {
      const enabled = await updateProofRuntimeSetting(true);
      if (!enabled) {
        return;
      }
    }

    const settings = window.compilerSettings || {};
    
    // Check if models are configured in settings
    if (!settings.validatorModel || !settings.writerModel || !settings.highParamModel) {
      alert('Please configure Validator, Writing Submitter, and Rigor & Proofs Submitter in the Compiler Settings tab first.');
      return;
    }

    setIsStarting(true);
    setError(null);
    try {
      const assistantMemoryEnabled = connectivityStatus?.skills?.agent_conversation_memory?.enabled !== false;
      await compilerAPI.start({
        compiler_prompt: compilerPrompt,
        // Validator config with OpenRouter support
        validator_provider: lmStudioEnabled ? (settings.validatorProvider || 'lm_studio') : 'openrouter',
        validator_model: settings.validatorModel,
        validator_openrouter_provider: settings.validatorOpenrouterProvider || null,
        validator_openrouter_reasoning_effort: settings.validatorOpenrouterReasoningEffort || 'auto',
        validator_lm_studio_fallback: lmStudioEnabled ? (settings.validatorLmStudioFallback || null) : null,
        validator_context_size: settings.validatorContextSize ?? validatorContextSize,
        validator_max_output_tokens: settings.validatorMaxOutput,
        validator_supercharge_enabled: developerModeEnabled && Boolean(settings.validatorSuperchargeEnabled),
        // Writing submitter config with OpenRouter support
        writer_provider: lmStudioEnabled ? (settings.writerProvider || 'lm_studio') : 'openrouter',
        writer_model: settings.writerModel,
        writer_openrouter_provider: settings.writerOpenrouterProvider || null,
        writer_openrouter_reasoning_effort: settings.writerOpenrouterReasoningEffort || 'auto',
        writer_lm_studio_fallback: lmStudioEnabled ? (settings.writerLmStudioFallback || null) : null,
        writer_context_size: settings.writerContextSize ?? writerContextSize,
        writer_max_output_tokens: settings.writerMaxOutput,
        writer_supercharge_enabled: developerModeEnabled && Boolean(settings.writerSuperchargeEnabled),
        // Rigor & Proofs Submitter config with OpenRouter support
        high_param_provider: lmStudioEnabled ? (settings.highParamProvider || 'lm_studio') : 'openrouter',
        high_param_model: settings.highParamModel,
        high_param_openrouter_provider: settings.highParamOpenrouterProvider || null,
        high_param_openrouter_reasoning_effort: settings.highParamOpenrouterReasoningEffort || 'auto',
        high_param_lm_studio_fallback: lmStudioEnabled ? (settings.highParamLmStudioFallback || null) : null,
        high_param_context_size: settings.highParamContextSize ?? highParamContextSize,
        high_param_max_output_tokens: settings.highParamMaxOutput,
        high_param_supercharge_enabled: developerModeEnabled && Boolean(settings.highParamSuperchargeEnabled),
        // Deprecated critique fields mirror Rigor & Proofs for compatibility.
        critique_submitter_provider: lmStudioEnabled ? (settings.highParamProvider || 'lm_studio') : 'openrouter',
        critique_submitter_model: settings.highParamModel,
        critique_submitter_openrouter_provider: settings.highParamOpenrouterProvider || null,
        critique_submitter_openrouter_reasoning_effort: settings.highParamOpenrouterReasoningEffort || 'auto',
        critique_submitter_lm_studio_fallback: lmStudioEnabled ? (settings.highParamLmStudioFallback || null) : null,
        critique_submitter_context_window: settings.highParamContextSize ?? highParamContextSize,
        critique_submitter_max_tokens: settings.highParamMaxOutput,
        critique_submitter_supercharge_enabled: developerModeEnabled && Boolean(settings.highParamSuperchargeEnabled),
        assistant_provider: assistantMemoryEnabled
          ? (lmStudioEnabled ? (settings.assistantProvider || settings.validatorProvider || 'lm_studio') : 'openrouter')
          : (lmStudioEnabled ? (settings.validatorProvider || 'lm_studio') : 'openrouter'),
        assistant_model: assistantMemoryEnabled ? (settings.assistantModel || settings.validatorModel) : '',
        assistant_openrouter_provider: assistantMemoryEnabled
          ? (settings.assistantOpenrouterProvider || settings.validatorOpenrouterProvider || null)
          : null,
        assistant_openrouter_reasoning_effort: assistantMemoryEnabled
          ? (settings.assistantOpenrouterReasoningEffort || settings.validatorOpenrouterReasoningEffort || 'auto')
          : 'auto',
        assistant_lm_studio_fallback: assistantMemoryEnabled && lmStudioEnabled
          ? (settings.assistantLmStudioFallback || settings.validatorLmStudioFallback || null)
          : null,
        assistant_context_size: assistantMemoryEnabled ? (settings.assistantContextSize || settings.validatorContextSize || validatorContextSize) : 0,
        assistant_max_output_tokens: assistantMemoryEnabled ? (settings.assistantMaxOutput || settings.validatorMaxOutput) : 0,
        assistant_supercharge_enabled: assistantMemoryEnabled && developerModeEnabled && Boolean(settings.assistantSuperchargeEnabled),
        allow_mathematical_proofs: Boolean(mathematicalProofsAllowed),
        allow_research_papers: Boolean(researchPapersAllowed)
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

  const updateProofRuntimeSetting = async (enabled) => {
    if (capabilities?.genericMode) {
      if (enabled) {
        alert('Mathematical proof output is unavailable in this runtime.');
        return false;
      }
      return true;
    }

    setProofOutputUpdating(true);
    try {
      const status = await autonomousAPI.getProofStatus();
      const updatedStatus = await autonomousAPI.updateProofSettings({
        enabled,
        timeout: status.lean4_proof_timeout ?? 600,
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
    const nextOutputs = { ...allowedOutputs, [key]: checked };
    if (!nextOutputs.mathematicalProofs && !nextOutputs.researchPapers) {
      alert('At least one allowed output must remain enabled.');
      return;
    }
    if (key === 'mathematicalProofs') {
      const updated = await updateProofRuntimeSetting(checked);
      if (!updated) {
        return;
      }
    }
    setAllowedOutputs(nextOutputs);
  };

  const handleStop = async () => {
    try {
      await compilerAPI.stop();
      onWorkflowRunningChange?.(false);
      await loadStatus();
    } catch (error) {
      console.error('Failed to stop compiler:', error);
      alert('Failed to stop compiler: ' + error.message);
    }
  };

  const getModelTypeLabel = (type) => {
    const labels = {
      validator: 'Validator Model',
      writer: 'Writing Submitter',
      high_param: 'Rigor & Proofs Submitter'
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
        <div className="autonomous-controls-stack">
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
                checked={proofOutputsAvailable && Boolean(allowedOutputs.mathematicalProofs)}
                onChange={(event) => updateAllowedOutput('mathematicalProofs', event.target.checked)}
                disabled={status.is_running || proofOutputUpdating || !proofOutputsAvailable}
              />
              <span className="allowed-output-text">Mathematical Proofs</span>
            </label>
            <label
              className="allowed-output-option"
              title="Research Papers enables the Single Paper Writer compilation output. When disabled, the writer runs proof extraction over the aggregator database instead of compiling a paper."
            >
              <input
                type="checkbox"
                checked={Boolean(allowedOutputs.researchPapers)}
                onChange={(event) => updateAllowedOutput('researchPapers', event.target.checked)}
                disabled={status.is_running}
              />
              <span className="allowed-output-text">Research Papers</span>
            </label>
          </div>
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
                  {critiqueAcceptances} accepted critique{critiqueAcceptances === 1 ? '' : 's'}
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
          placeholder='Create a final prompt that exactly relates to a solution your aggregation database helps solve, i.e. "Tell me the most theoretically advanced perspective on the "squaring the circle" problem."'
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
          <span className="stat-value">{writerContextSize.toLocaleString()}</span>
          <span className="stat-label">Writing Submitter Tokens</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">{highParamContextSize.toLocaleString()}</span>
          <span className="stat-label">Rigor & Proofs Tokens</span>
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
