import { useCallback, useEffect, useMemo, useState } from 'react';
import { autonomousAPI } from '../services/api';
import { websocket } from '../services/websocket';
import {
  getStoredAutonomousSettings,
  settingsToAutonomousConfig,
} from '../utils/autonomousProfiles';

const DEVELOPER_MODE_STORAGE_KEY = 'developerModeSettingsEnabled';

function isDeveloperModeEnabled() {
  return localStorage.getItem(DEVELOPER_MODE_STORAGE_KEY) === 'true';
}

function buildSourceKey(sourceType, sourceId) {
  return `${sourceType}:${sourceId}`;
}

function normalizeProvider(provider) {
  if (provider === 'openrouter' || provider === 'openai_codex_oauth') {
    return provider;
  }
  return 'lm_studio';
}

function toPositiveInteger(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : null;
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
      const proofRuntimeConfig = buildCurrentProofRuntimeConfig();
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

  const manualCheckReason = useMemo(() => {
    if (!proofStatus) {
      return 'Loading proof runtime status...';
    }
    if (!proofStatus.lean4_enabled) {
      return 'Lean 4 proof checks are disabled.';
    }
    if (!proofStatus.manual_check_ready && !hasCurrentProofRuntimeConfig) {
      return proofStatus.manual_check_message || 'Manual proof checks are not ready yet.';
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
    manualCheckEnabled: Boolean(
      proofStatus?.lean4_enabled &&
      (proofStatus?.manual_check_ready || hasCurrentProofRuntimeConfig)
    ),
    manualCheckReason,
  };
}
