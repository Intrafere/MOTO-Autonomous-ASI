import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  API_ERROR_KINDS,
  autonomousAPI,
  buildProofSourceKey,
  normalizeProofRunSnapshot,
} from '../services/api';
import { websocket } from '../services/websocket';
import {
  getStoredAutonomousSettings,
  settingsToAutonomousConfig,
} from '../utils/autonomousProfiles';
import {
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
  MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID,
} from '../utils/manualProofSources';
import { isCloudAccessProvider } from '../utils/oauthProviders';

const DEVELOPER_MODE_STORAGE_KEY = 'developerModeSettingsEnabled';
export {
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID,
  MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID,
};
const PROOF_STATUS_STARTUP_POLL_MS = 30000;

function isDeveloperModeEnabled() {
  return localStorage.getItem(DEVELOPER_MODE_STORAGE_KEY) === 'true';
}

function inferProofScope(sourceType, sourceId, requestedScope = null) {
  if (requestedScope) return requestedScope;
  if (
    (sourceType === 'brainstorm' && sourceId === MANUAL_AGGREGATOR_PROOF_SOURCE_ID)
    || (sourceType === 'paper' && sourceId === MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID)
  ) {
    return 'manual';
  }
  return 'autonomous';
}

function buildSourceKey(sourceType, sourceId, scope = null) {
  return buildProofSourceKey(inferProofScope(sourceType, sourceId, scope), sourceType, sourceId);
}

const TERMINAL_PROOF_RUN_STATUSES = new Set(['completed', 'error', 'stopped', 'repair_required']);
const NON_BUSY_PROOF_RUN_STATUSES = new Set([
  'completed',
  'error',
  'stopped',
  'repair_required',
]);
const PROOF_RUN_EVENTS = [
  'proof_run_queued',
  'proof_run_round_started',
  'proof_run_round_complete',
  'proof_run_provider_paused',
  'proof_run_provider_resumed',
  'proof_run_repair_required',
  'proof_run_terminal',
  'context_overflow_error',
  'proof_prune_review_queued',
  'proof_prune_review_started',
  'proof_prune_proposed',
  'proof_prune_validation_started',
  'proof_prune_provider_paused',
  'proof_prune_repair_required',
  'proof_prune_applied',
  'proof_prune_no_change',
  'proof_prune_rejected',
  'proof_prune_stale',
  'proof_prune_error',
  'proof_check_started',
  'proof_candidate_list_review_started',
  'proof_candidate_list_review_accepted',
  'proof_candidate_list_review_rejected',
  'proof_candidate_list_review_interrupted',
  'proof_candidate_list_regeneration_started',
  'proof_check_candidates_found',
  'proof_check_no_candidates',
  'proof_check_complete',
];

export function proofRunStatusLabel(run) {
  if (!run) return '';
  if (run.status === 'provider_paused') return 'Provider paused';
  if (run.status === 'repair_required') return 'Settings repair required';
  if (run.status === 'stopping') return 'Stopping';
  if (run.status === 'queued') return 'Queued';
  if (run.status === 'running') {
    return run.pruning_status && !['disabled', 'idle'].includes(run.pruning_status)
      ? `Running · pruning ${String(run.pruning_status).replaceAll('_', ' ')}`
      : 'Running';
  }
  return String(run.status || 'unknown').replaceAll('_', ' ');
}

export function mergeProofRun(current, incoming) {
  if (!incoming) return current;
  if (!current) return incoming;
  const currentGeneration = Number(current.lifecycle_generation || 0);
  const incomingGeneration = Number(incoming.lifecycle_generation || 0);
  if (incomingGeneration < currentGeneration) return current;
  return { ...current, ...incoming };
}

export function isProofRunBusy(run) {
  return Boolean(run && !NON_BUSY_PROOF_RUN_STATUSES.has(String(run.status || 'unknown')));
}

export function selectSourceProofRun(runs, sourceKey, preferredProofRunId = null) {
  const matchingRuns = Object.values(runs || {})
    .filter((run) => run?.source_key === sourceKey)
    .sort((a, b) => (
      Number(b.lifecycle_generation || 0) - Number(a.lifecycle_generation || 0)
      || String(b.updated_at || '').localeCompare(String(a.updated_at || ''))
    ));
  if (preferredProofRunId) {
    const preferred = matchingRuns.find((run) => run.proof_run_id === preferredProofRunId);
    if (preferred) return preferred;
  }
  return matchingRuns.find((run) => !TERMINAL_PROOF_RUN_STATUSES.has(run.status))
    || matchingRuns[0]
    || null;
}

function normalizeProvider(provider) {
  if (provider === 'openrouter' || isCloudAccessProvider(provider)) {
    return provider;
  }
  return 'lm_studio';
}

function toPositiveInteger(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : null;
}

function readStoredJson(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    console.warn(`Failed to read ${key}:`, error);
    return null;
  }
}

function roleFromSubmitterConfig(config = {}) {
  const superchargeAllowed = isDeveloperModeEnabled();
  return {
    provider: normalizeProvider(config.provider),
    model_id: config.modelId || config.model_id || '',
    openrouter_provider: config.openrouterProvider ?? config.openrouter_provider ?? null,
    openrouter_reasoning_effort: config.openrouterReasoningEffort ?? config.openrouter_reasoning_effort ?? 'auto',
    lm_studio_fallback_id: config.lmStudioFallbackId ?? config.lm_studio_fallback_id ?? null,
    context_window: toPositiveInteger(config.contextWindow ?? config.context_window),
    max_output_tokens: toPositiveInteger(config.maxOutputTokens ?? config.max_output_tokens),
    supercharge_enabled: superchargeAllowed && Boolean(config.superchargeEnabled ?? config.supercharge_enabled),
  };
}

function roleFromAggregatorValidatorSettings(settings = {}) {
  const superchargeAllowed = isDeveloperModeEnabled();
  return {
    provider: normalizeProvider(settings.validatorProvider),
    model_id: settings.validatorModel || settings.validator_model || '',
    openrouter_provider: settings.validatorOpenrouterProvider ?? settings.validator_openrouter_provider ?? null,
    openrouter_reasoning_effort: settings.validatorOpenrouterReasoningEffort ?? settings.validator_openrouter_reasoning_effort ?? 'auto',
    lm_studio_fallback_id: settings.validatorLmStudioFallback ?? settings.validator_lm_studio_fallback ?? null,
    context_window: toPositiveInteger(settings.validatorContextSize ?? settings.validator_context_size),
    max_output_tokens: toPositiveInteger(settings.validatorMaxOutput ?? settings.validator_max_output_tokens),
    supercharge_enabled: superchargeAllowed && Boolean(settings.validatorSuperchargeEnabled ?? settings.validator_supercharge_enabled),
  };
}

function roleFromCompilerSettings(settings = {}, prefix) {
  const superchargeAllowed = isDeveloperModeEnabled();
  return {
    provider: normalizeProvider(settings[`${prefix}Provider`]),
    model_id: settings[`${prefix}Model`] || '',
    openrouter_provider: settings[`${prefix}OpenrouterProvider`] ?? null,
    openrouter_reasoning_effort: settings[`${prefix}OpenrouterReasoningEffort`] ?? 'auto',
    lm_studio_fallback_id: settings[`${prefix}LmStudioFallback`] ?? null,
    context_window: toPositiveInteger(settings[`${prefix}ContextSize`]),
    max_output_tokens: toPositiveInteger(settings[`${prefix}MaxOutput`]),
    supercharge_enabled: superchargeAllowed && Boolean(settings[`${prefix}SuperchargeEnabled`]),
  };
}

function roleFromAutonomousConfig(config, rolePrefix, fallbackModelId = '') {
  const superchargeAllowed = isDeveloperModeEnabled();
  return {
    provider: normalizeProvider(config[`${rolePrefix}_provider`]),
    model_id: config[`${rolePrefix}_model`] || fallbackModelId || '',
    openrouter_provider: config[`${rolePrefix}_openrouter_provider`] ?? null,
    openrouter_reasoning_effort: config[`${rolePrefix}_openrouter_reasoning_effort`] ?? 'auto',
    lm_studio_fallback_id: config[`${rolePrefix}_lm_studio_fallback`] ?? null,
    context_window: toPositiveInteger(config[`${rolePrefix}_context_window`]),
    max_output_tokens: toPositiveInteger(config[`${rolePrefix}_max_tokens`]),
    supercharge_enabled: superchargeAllowed && Boolean(config[`${rolePrefix}_supercharge_enabled`]),
  };
}

function roleWithDefaults(role = {}, defaults = {}) {
  const hasExplicitModel = Boolean(role?.model_id);
  return {
    provider: hasExplicitModel ? role.provider : defaults.provider,
    model_id: role.model_id || defaults.model_id || '',
    openrouter_provider: hasExplicitModel ? (role.openrouter_provider ?? null) : (defaults.openrouter_provider ?? null),
    openrouter_reasoning_effort: hasExplicitModel
      ? (role.openrouter_reasoning_effort || 'auto')
      : (defaults.openrouter_reasoning_effort || 'auto'),
    lm_studio_fallback_id: hasExplicitModel
      ? (role.lm_studio_fallback_id ?? null)
      : (defaults.lm_studio_fallback_id ?? null),
    context_window: role.context_window || defaults.context_window || null,
    max_output_tokens: role.max_output_tokens || defaults.max_output_tokens || null,
    supercharge_enabled: hasExplicitModel
      ? Boolean(role.supercharge_enabled)
      : Boolean(defaults.supercharge_enabled),
  };
}

function roleFromAggregatorAssistantSettings(settings = {}, defaults = {}) {
  const superchargeAllowed = isDeveloperModeEnabled();
  return roleWithDefaults({
    provider: normalizeProvider(settings.assistantProvider),
    model_id: settings.assistantModel || '',
    openrouter_provider: settings.assistantOpenrouterProvider ?? null,
    openrouter_reasoning_effort: settings.assistantOpenrouterReasoningEffort ?? 'auto',
    lm_studio_fallback_id: settings.assistantLmStudioFallback ?? null,
    context_window: toPositiveInteger(settings.assistantContextSize),
    max_output_tokens: toPositiveInteger(settings.assistantMaxOutput),
    supercharge_enabled: superchargeAllowed && Boolean(settings.assistantSuperchargeEnabled),
  }, defaults);
}

export function buildCurrentProofRuntimeConfig() {
  try {
    const config = settingsToAutonomousConfig(getStoredAutonomousSettings());
    const rigor = roleFromAutonomousConfig(config, 'high_param');
    const validator = roleFromAutonomousConfig(config, 'validator');
    const assistant = roleWithDefaults(
      roleFromAutonomousConfig(config, 'assistant'),
      validator
    );
    return {
      brainstorm: rigor,
      paper: rigor,
      validator,
      assistant,
    };
  } catch (error) {
    console.warn('Failed to build current proof runtime config:', error);
    return null;
  }
}

export function buildManualAggregatorProofRuntimeConfig() {
  const settings = {
    ...(readStoredJson('aggregatorConfig') || {}),
    ...(readStoredJson('aggregator_settings') || {}),
  };
  const firstSubmitter = roleFromSubmitterConfig(settings.submitterConfigs?.[0]);
  const validator = roleFromAggregatorValidatorSettings(settings);
  const assistant = roleFromAggregatorAssistantSettings(settings, validator);
  return {
    brainstorm: firstSubmitter,
    paper: firstSubmitter,
    validator,
    assistant,
  };
}

export function buildManualCompilerProofRuntimeConfig() {
  const settings = readStoredJson('compiler_settings') || {};
  const rigor = roleFromCompilerSettings(settings, 'highParam');
  const validator = roleFromCompilerSettings(settings, 'validator');
  const assistant = roleWithDefaults(roleFromCompilerSettings(settings, 'assistant'), validator);
  return {
    brainstorm: rigor,
    paper: rigor,
    validator,
    assistant,
  };
}

export function buildProofRuntimeConfigForSource(sourceType, sourceId) {
  if (sourceType === 'brainstorm' && sourceId === MANUAL_AGGREGATOR_PROOF_SOURCE_ID) {
    return buildManualAggregatorProofRuntimeConfig();
  }
  if (sourceType === 'paper' && sourceId === MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID) {
    return buildManualCompilerProofRuntimeConfig();
  }
  return buildCurrentProofRuntimeConfig();
}

export function isProofRuntimeConfigComplete(config) {
  return Boolean(
    config?.brainstorm?.model_id &&
    config?.brainstorm?.context_window &&
    config?.brainstorm?.max_output_tokens &&
    config?.paper?.model_id &&
    config?.paper?.context_window &&
    config?.paper?.max_output_tokens &&
    config?.validator?.model_id &&
    config?.validator?.context_window &&
    config?.validator?.max_output_tokens &&
    (
      !config?.assistant?.model_id ||
      Boolean(config.assistant.context_window && config.assistant.max_output_tokens)
    )
  );
}

function hasProofRuntimeConfigForSource(sourceType, sourceId) {
  return isProofRuntimeConfigComplete(buildProofRuntimeConfigForSource(sourceType, sourceId));
}

function getLeanRuntimeUnavailableMessage(proofStatus) {
  if (!proofStatus?.lean4_enabled) {
    return 'Lean 4 proof checks are disabled.';
  }

  const version = (proofStatus.lean4_version || proofStatus.lean_version || '').trim().toLowerCase();
  const versionUnavailable = (
    version.includes('not found') ||
    version.includes('no such file') ||
    version.includes('not recognized')
  );
  if (versionUnavailable) {
    return 'Lean 4 executable is not available.';
  }
  if (proofStatus.workspace_state === 'failed') {
    return proofStatus.workspace_error || 'Lean 4 workspace preparation failed.';
  }
  return '';
}

function canUseLocalProofRuntimeConfig(proofStatus, sourceType, sourceId) {
  return (
    hasProofRuntimeConfigForSource(sourceType, sourceId) &&
    !getLeanRuntimeUnavailableMessage(proofStatus)
  );
}

export function useProofCheckRuntime() {
  const [proofStatus, setProofStatus] = useState(null);
  const [runtimeError, setRuntimeError] = useState('');
  const [activeChecks, setActiveChecks] = useState({});
  const [queuedChecks, setQueuedChecks] = useState({});
  const [proofRuns, setProofRuns] = useState({});
  const [runDiscoveryError, setRunDiscoveryError] = useState('');

  const rememberRun = useCallback((runLike, fallback = {}) => {
    let incoming;
    try {
      incoming = normalizeProofRunSnapshot(runLike, fallback);
    } catch {
      return null;
    }
    setProofRuns((previous) => ({
      ...previous,
      [incoming.proof_run_id]: mergeProofRun(previous[incoming.proof_run_id], incoming),
    }));
    return incoming;
  }, []);

  const discoverProofRuns = useCallback(async () => {
    try {
      const response = await autonomousAPI.listProofRuns();
      setProofRuns((previous) => {
        const discovered = {};
        response.runs.forEach((run) => {
          discovered[run.proof_run_id] = mergeProofRun(previous[run.proof_run_id], run);
        });
        return discovered;
      });
      setRunDiscoveryError('');
      return response.runs;
    } catch (error) {
      const message = error.kind === API_ERROR_KINDS.OLD_CONTRACT
        ? 'Proof-run controls require a newer backend contract. Update MOTO, then reload.'
        : (error.message || 'Failed to discover proof runs');
      setRunDiscoveryError(message);
      return [];
    }
  }, []);

  const discoverSourceProofRuns = useCallback(async (sourceType, sourceId, scope = null) => {
    const resolvedScope = inferProofScope(sourceType, sourceId, scope);
    const response = await autonomousAPI.listProofRuns({
      scope: resolvedScope,
      sourceType,
      sourceId,
    });
    response.runs.forEach((run) => rememberRun(run));
    if (response.ambiguous && !response.preferred_proof_run_id) {
      throw new Error(
        'Multiple proof runs match this source and no authoritative run could be selected. Refresh proof runs before starting another.',
      );
    }
    const preferred = response.preferred_proof_run_id
      ? response.runs.find((run) => run.proof_run_id === response.preferred_proof_run_id)
      : response.runs[0];
    return preferred || null;
  }, [rememberRun]);

  const refreshProofStatus = useCallback(async () => {
    try {
      const status = await autonomousAPI.getProofStatus();
      setProofStatus(status);
      setRuntimeError('');
      return status;
    } catch (err) {
      setRuntimeError(err.message || 'Failed to load proof status');
      return null;
    }
  }, []);

  useEffect(() => {
    refreshProofStatus();
    discoverProofRuns();
  }, [discoverProofRuns, refreshProofStatus]);

  useEffect(() => {
    const handleRefresh = () => {
      refreshProofStatus();
      discoverProofRuns();
    };
    const handleOnline = () => discoverProofRuns();

    window.addEventListener('focus', handleRefresh);
    window.addEventListener('online', handleOnline);
    document.addEventListener('visibilitychange', handleRefresh);

    return () => {
      window.removeEventListener('focus', handleRefresh);
      window.removeEventListener('online', handleOnline);
      document.removeEventListener('visibilitychange', handleRefresh);
    };
  }, [discoverProofRuns, refreshProofStatus]);

  useEffect(() => {
    const shouldPollProofStatus = (
      !proofStatus ||
      (proofStatus.lean4_enabled && Boolean(getLeanRuntimeUnavailableMessage(proofStatus)))
    );
    if (!shouldPollProofStatus) {
      return undefined;
    }

    const interval = setInterval(refreshProofStatus, PROOF_STATUS_STARTUP_POLL_MS);
    return () => clearInterval(interval);
  }, [proofStatus, refreshProofStatus]);

  useEffect(() => {
    const reconcileRunEvent = (data = {}) => {
      const proofRunId = data.proof_run_id;
      if (!proofRunId) return;
      if (data.source_type && data.source_id) {
        const sourceKey = buildSourceKey(data.source_type, data.source_id, data.scope);
        if (data.status && data.status !== 'queued') {
          setQueuedChecks((previous) => {
            if (!previous[sourceKey]) return previous;
            const next = { ...previous };
            delete next[sourceKey];
            return next;
          });
        }
        if (NON_BUSY_PROOF_RUN_STATUSES.has(data.status)) {
          setActiveChecks((previous) => {
            if (!previous[sourceKey]) return previous;
            const next = { ...previous };
            delete next[sourceKey];
            return next;
          });
        }
      }
      setProofRuns((previous) => {
        const current = previous[proofRunId];
        const eventGeneration = Number(data.lifecycle_generation || 0);
        if (
          current
          && eventGeneration
          && eventGeneration < Number(current.lifecycle_generation || 0)
        ) {
          return previous;
        }
        const next = mergeProofRun(current, {
          ...(current || {}),
          ...data,
          proof_run_id: proofRunId,
          lifecycle_generation: eventGeneration || current?.lifecycle_generation || 1,
          status: data.status || current?.status || 'running',
        });
        return { ...previous, [proofRunId]: next };
      });
      autonomousAPI.getProofRun(proofRunId)
        .then((run) => rememberRun(run))
        .catch(() => {
          // The compact event remains useful if detail hydration races cleanup.
        });
    };
    PROOF_RUN_EVENTS.forEach((eventName) => websocket.on(eventName, reconcileRunEvent));
    return () => {
      PROOF_RUN_EVENTS.forEach((eventName) => websocket.off(eventName, reconcileRunEvent));
    };
  }, [rememberRun]);

  useEffect(() => {
    const unsubscribeStarted = websocket.on('proof_check_started', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id, data.scope);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: null,
        },
      }));
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
    });

    const unsubscribeCandidates = websocket.on('proof_check_candidates_found', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id, data.scope);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: data.count ?? null,
        },
      }));
    });

    const unsubscribeNoCandidates = websocket.on('proof_check_no_candidates', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id, data.scope);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: 0,
        },
      }));
    });

    const unsubscribeComplete = websocket.on('proof_check_complete', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id, data.scope);
      setActiveChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      refreshProofStatus();
    });

    return () => {
      unsubscribeStarted();
      unsubscribeCandidates();
      unsubscribeNoCandidates();
      unsubscribeComplete();
    };
  }, [refreshProofStatus]);

  const queueManualProofCheck = useCallback(async ({
    sourceType,
    sourceId,
    scope = null,
    runMode = 'one_round',
  }) => {
    const resolvedScope = inferProofScope(sourceType, sourceId, scope);
    const sourceKey = buildSourceKey(sourceType, sourceId, resolvedScope);
    setQueuedChecks((prev) => ({
      ...prev,
      [sourceKey]: true,
    }));

    try {
      const proofRuntimeConfig = buildProofRuntimeConfigForSource(sourceType, sourceId);
      const run = await autonomousAPI.runProofCheck({
        sourceType,
        sourceId,
        scope: resolvedScope,
        runMode,
        proofRuntimeConfig: isProofRuntimeConfigComplete(proofRuntimeConfig) ? proofRuntimeConfig : null,
      });
      rememberRun(run);
      return run;
    } catch (err) {
      if (err?.kind === API_ERROR_KINDS.AMBIGUOUS_TRANSPORT) {
        const recovered = await discoverSourceProofRuns(sourceType, sourceId, resolvedScope);
        if (recovered) return recovered;
      }
      throw err;
    } finally {
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
    }
  }, [discoverSourceProofRuns, rememberRun]);

  const getSourceState = useCallback((sourceType, sourceId, scope = null) => {
    const sourceKey = buildSourceKey(sourceType, sourceId, scope);
    const run = selectSourceProofRun(proofRuns, sourceKey);
    if (run) {
      return {
        ...run,
        candidateCount: run.candidateCount ?? null,
        statusLabel: proofRunStatusLabel(run),
      };
    }
    if (activeChecks[sourceKey]) {
      return activeChecks[sourceKey];
    }
    if (queuedChecks[sourceKey]) {
      return {
        status: 'queued',
        candidateCount: null,
      };
    }
    return null;
  }, [activeChecks, proofRuns, queuedChecks]);

  const isSourceBusy = useCallback((sourceType, sourceId) => {
    const state = getSourceState(sourceType, sourceId);
    return isProofRunBusy(state);
  }, [getSourceState]);

  const controlProofRun = useCallback(async (action, runOrId, generation = null) => {
    const run = typeof runOrId === 'string' ? proofRuns[runOrId] : runOrId;
    const proofRunId = typeof runOrId === 'string' ? runOrId : runOrId?.proof_run_id;
    const expectedGeneration = generation ?? run?.lifecycle_generation;
    if (!proofRunId || !expectedGeneration) {
      throw new Error('Proof run identity or lifecycle generation is missing. Refresh and try again.');
    }
    try {
      if (action !== 'stop') throw new Error(`Unsupported proof run action: ${action}`);
      const next = await autonomousAPI.stopProofRun(proofRunId, expectedGeneration);
      rememberRun(next);
      return next;
    } catch (error) {
      if (error?.kind === API_ERROR_KINDS.CONFLICT || error?.status === 409) {
        try {
          const authoritative = await autonomousAPI.getProofRun(proofRunId);
          rememberRun(authoritative);
        } catch {
          await discoverProofRuns();
        }
      }
      throw error;
    }
  }, [discoverProofRuns, proofRuns, rememberRun]);

  const currentProofRuntimeConfig = buildCurrentProofRuntimeConfig();
  const hasCurrentProofRuntimeConfig = isProofRuntimeConfigComplete(currentProofRuntimeConfig);
  const currentLocalRuntimeReady = hasCurrentProofRuntimeConfig && !getLeanRuntimeUnavailableMessage(proofStatus);

  const canQueueManualProofCheck = useCallback((sourceType, sourceId) => Boolean(
    proofStatus?.lean4_enabled &&
    (proofStatus?.manual_check_ready || canUseLocalProofRuntimeConfig(proofStatus, sourceType, sourceId))
  ), [proofStatus]);

  const getManualCheckReason = useCallback((sourceType, sourceId) => {
    if (!proofStatus) {
      return 'Loading proof runtime status...';
    }
    if (!proofStatus.lean4_enabled) {
      return 'Lean 4 proof checks are disabled.';
    }
    if (!proofStatus.manual_check_ready) {
      const localRuntimeConfig = hasProofRuntimeConfigForSource(sourceType, sourceId);
      const runtimeMessage = getLeanRuntimeUnavailableMessage(proofStatus);
      if (localRuntimeConfig && runtimeMessage) {
        return runtimeMessage;
      }
      if (!localRuntimeConfig) {
        return proofStatus.manual_check_message || 'Manual proof checks are not ready yet.';
      }
    }
    return '';
  }, [proofStatus]);

  const manualCheckReason = useMemo(() => {
    if (!proofStatus) {
      return 'Loading proof runtime status...';
    }
    if (!proofStatus.lean4_enabled) {
      return 'Lean 4 proof checks are disabled.';
    }
    if (!proofStatus.manual_check_ready) {
      const runtimeMessage = getLeanRuntimeUnavailableMessage(proofStatus);
      if (hasCurrentProofRuntimeConfig && runtimeMessage) {
        return runtimeMessage;
      }
      if (!hasCurrentProofRuntimeConfig) {
        return proofStatus.manual_check_message || 'Manual proof checks are not ready yet.';
      }
    }
    return '';
  }, [proofStatus, hasCurrentProofRuntimeConfig]);

  return {
    proofStatus,
    runtimeError,
    runDiscoveryError,
    proofRuns: Object.values(proofRuns),
    refreshProofStatus,
    discoverProofRuns,
    discoverSourceProofRuns,
    queueManualProofCheck,
    controlProofRun,
    stopProofRun: (run, generation) => controlProofRun('stop', run, generation),
    getSourceState,
    isSourceBusy,
    canQueueManualProofCheck,
    getManualCheckReason,
    manualCheckEnabled: Boolean(
      proofStatus?.lean4_enabled &&
      (proofStatus?.manual_check_ready || currentLocalRuntimeReady)
    ),
    manualCheckReason,
  };
}
