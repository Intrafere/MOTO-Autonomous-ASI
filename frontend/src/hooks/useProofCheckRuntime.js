import { useCallback, useEffect, useMemo, useState } from 'react';
import { autonomousAPI } from '../services/api';
import { websocket } from '../services/websocket';
import {
  getStoredAutonomousSettings,
  settingsToAutonomousConfig,
} from '../utils/autonomousProfiles';
import { isCloudAccessProvider } from '../utils/oauthProviders';

const DEVELOPER_MODE_STORAGE_KEY = 'developerModeSettingsEnabled';
export const MANUAL_AGGREGATOR_PROOF_SOURCE_ID = 'manual_aggregator';
export const MANUAL_COMPILER_CURRENT_PROOF_SOURCE_ID = 'manual_compiler_current';
const PROOF_STATUS_STARTUP_POLL_MS = 30000;

function isDeveloperModeEnabled() {
  return localStorage.getItem(DEVELOPER_MODE_STORAGE_KEY) === 'true';
}

function buildSourceKey(sourceType, sourceId) {
  return `${sourceType}:${sourceId}`;
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

export function buildCurrentProofRuntimeConfig() {
  try {
    const config = settingsToAutonomousConfig(getStoredAutonomousSettings());
    const firstSubmitter = roleFromSubmitterConfig(config.submitter_configs?.[0]);
    return {
      brainstorm: firstSubmitter,
      paper: roleFromAutonomousConfig(config, 'high_context', firstSubmitter.model_id),
      validator: roleFromAutonomousConfig(config, 'validator'),
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
  return {
    brainstorm: firstSubmitter,
    paper: firstSubmitter,
    validator,
  };
}

export function buildManualCompilerProofRuntimeConfig() {
  const settings = readStoredJson('compiler_settings') || {};
  const highContext = roleFromCompilerSettings(settings, 'highContext');
  const validator = roleFromCompilerSettings(settings, 'validator');
  return {
    brainstorm: highContext,
    paper: highContext,
    validator,
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
    config?.validator?.max_output_tokens
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
    !version ||
    version.includes('not found') ||
    version.includes('no such file') ||
    version.includes('not recognized')
  );
  if (versionUnavailable) {
    return 'Lean 4 executable is not available.';
  }
  if (!proofStatus.workspace_ready) {
    return 'Lean 4 is still starting up.';
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
  }, [refreshProofStatus]);

  useEffect(() => {
    const handleRefresh = () => {
      refreshProofStatus();
    };

    window.addEventListener('focus', handleRefresh);
    document.addEventListener('visibilitychange', handleRefresh);

    return () => {
      window.removeEventListener('focus', handleRefresh);
      document.removeEventListener('visibilitychange', handleRefresh);
    };
  }, [refreshProofStatus]);

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
    const unsubscribeStarted = websocket.on('proof_check_started', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: prev[sourceKey]?.candidateCount ?? null,
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
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
      setActiveChecks((prev) => ({
        ...prev,
        [sourceKey]: {
          status: 'running',
          candidateCount: data.count ?? null,
        },
      }));
    });

    const unsubscribeComplete = websocket.on('proof_check_complete', (data) => {
      const sourceKey = buildSourceKey(data.source_type, data.source_id);
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
      unsubscribeComplete();
    };
  }, [refreshProofStatus]);

  const queueManualProofCheck = useCallback(async ({ sourceType, sourceId }) => {
    const sourceKey = buildSourceKey(sourceType, sourceId);
    setQueuedChecks((prev) => ({
      ...prev,
      [sourceKey]: true,
    }));

    try {
      const proofRuntimeConfig = buildProofRuntimeConfigForSource(sourceType, sourceId);
      return await autonomousAPI.runProofCheck({
        sourceType,
        sourceId,
        proofRuntimeConfig: isProofRuntimeConfigComplete(proofRuntimeConfig) ? proofRuntimeConfig : null,
      });
    } catch (err) {
      setQueuedChecks((prev) => {
        if (!prev[sourceKey]) {
          return prev;
        }
        const next = { ...prev };
        delete next[sourceKey];
        return next;
      });
      throw err;
    }
  }, []);

  const getSourceState = useCallback((sourceType, sourceId) => {
    const sourceKey = buildSourceKey(sourceType, sourceId);
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
  }, [activeChecks, queuedChecks]);

  const isSourceBusy = useCallback((sourceType, sourceId) => {
    return Boolean(getSourceState(sourceType, sourceId));
  }, [getSourceState]);

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
    refreshProofStatus,
    queueManualProofCheck,
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
