import React, { useEffect, useState } from 'react';
import { api, cloudAccessAPI, openRouterAPI } from '../../services/api';
import {
  computeCodexAutoSettings,
  computeCloudAccessAutoSettings,
  computeOpenRouterAutoSettings,
  computeSakanaFuguAutoSettings,
  computeXAIGrokAutoSettings,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  findOpenRouterModel,
  formatOpenRouterProviderLabel,
  getOpenRouterProviderTitle,
  getProviderNames,
  getReasoningSupportInfo,
  hasEndpointMetadata,
  normalizeOpenRouterReasoningEffort,
  OPENROUTER_REASONING_EFFORT_OPTIONS,
  SAKANA_FUGU_REASONING_EFFORT_OPTIONS,
} from '../../utils/openRouterSelection';
import { refreshCredentialProviderState } from '../../utils/credentialProviderRefresh';
import {
  chooseCloudAccessProvider,
  getConfiguredCloudAccessProviders,
  isCloudAccessProvider,
  cloudAccessProviderLabel,
  SAKANA_FUGU_PROVIDER,
  XAI_GROK_PROVIDER,
} from '../../utils/oauthProviders';
import {
  LEANOJ_PROFILES_STORAGE_KEY,
  LEANOJ_RECOMMENDED_PROFILES,
  applyLeanOJProfileSelection,
  persistLeanOJSettings,
} from '../../utils/leanojProfiles';
import HelpTooltip from '../HelpTooltip';
import HighlightedModelsSidebar from '../HighlightedModelsSidebar';
import OpenRouterFreeModelsControl from '../OpenRouterFreeModelsControl';
import RawSettingsEditor from '../RawSettingsEditor';
import '../settings-common.css';

const RAW_VIEW_EXIT_WARNING = 'Switching back to the GUI view will restore your last GUI settings/profile and discard raw-only changes. Continue?';
const formatRawSettings = (value) => JSON.stringify(value, null, 2);
const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';

const ROLE_EDITOR_GROUPS = [
  { key: 'validator', title: 'Validator', roleKeys: ['topic_validator', 'brainstorm_validator'] },
  { key: 'assistant', title: 'Assistant', roleKeys: ['assistant'] },
  { key: 'final_solver', title: 'Final Proof Solver', roleKeys: ['final_solver'] },
];

const toRoleConfig = (config = {}) => {
  const { submitterId, ...roleConfig } = config;
  return roleConfig;
};

function ModelSelector({
  config,
  onChange,
  lmStudioModels,
  openRouterModels,
  openAICodexModels,
  xaiGrokModels,
  sakanaFuguModels,
  modelProviders,
  hasOpenRouterKey,
  hasOpenAICodexLogin,
  hasXAIGrokLogin,
  hasSakanaFuguKey,
  isRunning,
  lmStudioEnabled,
}) {
  const provider = lmStudioEnabled ? (config.provider || 'lm_studio') : 'openrouter';
  const oauthStatusByProvider = {
    openai_codex_oauth: { configured: hasOpenAICodexLogin },
    [XAI_GROK_PROVIDER]: { configured: hasXAIGrokLogin },
    [SAKANA_FUGU_PROVIDER]: { configured: hasSakanaFuguKey },
  };
  const configuredOAuthProviders = getConfiguredCloudAccessProviders(oauthStatusByProvider);
  const models = provider === 'openrouter'
    ? openRouterModels
    : (isCloudAccessProvider(provider)
      ? (provider === XAI_GROK_PROVIDER
        ? xaiGrokModels
        : (provider === SAKANA_FUGU_PROVIDER ? sakanaFuguModels : openAICodexModels))
      : lmStudioModels);
  const providers = provider === 'openrouter' && config.modelId
    ? getProviderNames(modelProviders[config.modelId])
    : [];
  const reasoningInfo = provider === 'openrouter'
    ? getReasoningSupportInfo(modelProviders[config.modelId], config.openrouterProvider || null)
    : { hasEndpointMetadata: false, supportsReasoning: false };

  return (
    <>
      <div className="settings-row">
        <label>Provider</label>
        {lmStudioEnabled ? (
          <div className="provider-toggle-group">
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'lm_studio' ? ' active-lm' : ''}`}
              disabled={isRunning}
              onClick={() => onChange({ ...config, provider: 'lm_studio', openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT })}
            >
              LM Studio
            </button>
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'openrouter' ? ' active-or-orange' : ''}`}
              disabled={isRunning || !hasOpenRouterKey}
              onClick={() => onChange({ ...config, provider: 'openrouter', openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT })}
              title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
            >
              OpenRouter
            </button>
            <button
              type="button"
              className={`provider-toggle-btn${isCloudAccessProvider(provider) ? ' active-or-orange' : ''}`}
              disabled={isRunning || configuredOAuthProviders.length === 0}
              onClick={() => onChange({
                ...config,
                provider: chooseCloudAccessProvider(oauthStatusByProvider, provider),
                modelId: '',
                openrouterProvider: null,
                openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
              })}
              title={configuredOAuthProviders.length === 0 ? 'Set up a cloud provider login or API key first' : 'Use a configured cloud provider'}
            >
              Cloud
            </button>
            {isCloudAccessProvider(provider) && configuredOAuthProviders.length > 1 && (
              <select
                value={provider}
                disabled={isRunning}
                onChange={(event) => onChange({
                  ...config,
                  provider: event.target.value,
                  modelId: '',
                  openrouterProvider: null,
                  openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
                })}
                title="Select cloud provider"
                className="input-dark"
                style={{ width: 'auto', minWidth: '150px' }}
              >
                {configuredOAuthProviders.map((oauthProvider) => (
                  <option key={oauthProvider.id} value={oauthProvider.id}>
                    {oauthProvider.label}
                  </option>
                ))}
              </select>
            )}
          </div>
        ) : (
          <small className="settings-hint">OpenRouter is required in this deployment.</small>
        )}
      </div>

      <div className="settings-row">
        <label>Model</label>
        <select
          value={config.modelId || ''}
          disabled={isRunning}
          onChange={(event) => onChange({ ...config, provider, modelId: event.target.value, openrouterProvider: null, openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT })}
        >
          <option value="">Select model...</option>
          {models.map((model) => {
            const isFree = provider === 'openrouter' && model.pricing?.prompt === '0' && model.pricing?.completion === '0';
            const contextInfo = model.context_length ? ` (${Math.round(model.context_length / 1000)}K)` : '';
            return (
              <option key={model.id} value={model.id}>
                {(model.name || model.id)}{contextInfo}{isFree ? ' [FREE]' : ''}
              </option>
            );
          })}
        </select>
      </div>

      {provider === 'openrouter' && config.modelId && (
        <div className="settings-row">
          <label>Host Provider</label>
          <select
            className="openrouter-host-provider-select"
            value={config.openrouterProvider || ''}
            disabled={isRunning}
            title={getOpenRouterProviderTitle(config.openrouterProvider)}
            onChange={(event) => onChange({ ...config, provider, openrouterProvider: event.target.value || null })}
          >
            <option value="">Auto</option>
            {providers.map((providerName) => (
              <option key={providerName} value={providerName} title={getOpenRouterProviderTitle(providerName)}>
                {formatOpenRouterProviderLabel(providerName)}
              </option>
            ))}
          </select>
        </div>
      )}

      {provider === 'openrouter' && config.modelId && (
        <div className="settings-row">
          <label>Reasoning Effort</label>
          <select
            value={normalizeOpenRouterReasoningEffort(config.openrouterReasoningEffort)}
            disabled={isRunning}
            onChange={(event) => onChange({ ...config, provider, openrouterReasoningEffort: event.target.value })}
          >
            {OPENROUTER_REASONING_EFFORT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <small className="settings-hint">
            {reasoningInfo.hasEndpointMetadata && !reasoningInfo.supportsReasoning
              ? 'This selected host does not advertise reasoning support; OpenRouter may ignore the setting.'
              : 'Auto sends OpenRouter max reasoning effort by default.'}
          </small>
        </div>
      )}

      {provider === SAKANA_FUGU_PROVIDER && config.modelId && (
        <div className="settings-row">
          <label>Reasoning Effort</label>
          <select
            value={normalizeOpenRouterReasoningEffort(config.openrouterReasoningEffort)}
            disabled={isRunning}
            onChange={(event) => onChange({ ...config, provider, openrouterReasoningEffort: event.target.value })}
          >
            {SAKANA_FUGU_REASONING_EFFORT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <small className="settings-hint">
            Sakana Fugu supports high and xhigh reasoning effort only; auto maps to xhigh.
          </small>
        </div>
      )}

      {provider !== 'lm_studio' && lmStudioEnabled && (
        <div className="settings-row">
          <label>LM Studio Fallback</label>
          <select
            value={config.lmStudioFallbackId || ''}
            disabled={isRunning}
            onChange={(event) => onChange({ ...config, provider, lmStudioFallbackId: event.target.value || null })}
          >
            <option value="">No fallback</option>
            {lmStudioModels.map((model) => (
              <option key={model.id} value={model.id}>{model.id}</option>
            ))}
          </select>
        </div>
      )}
    </>
  );
}
function RoleEditor(props) {
  const { title, config, onChange, isRunning, developerModeEnabled = false, disabled = false } = props;
  const controlsDisabled = isRunning || disabled;
  const updateNumber = (key, value, fallback) => {
    const parsed = parseInt(value, 10);
    onChange({ ...config, [key]: Number.isFinite(parsed) && parsed > 0 ? parsed : fallback });
  };

  return (
    <div
      className={`submitter-config-section${config.provider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
      aria-disabled={disabled}
      style={disabled ? { opacity: 0.55, pointerEvents: 'none' } : undefined}
    >
      <h4>{title}</h4>
      {title === 'Assistant' && (
        <p className="settings-info">
          Runs in parallel during topic, brainstorm, path, master-proof edit, and final proof work to retrieve up to 7 relevant verified proof-memory supports from Session History Memory and SyntheticLib4 when enabled. Validators never receive Assistant context.
        </p>
      )}
      {disabled && (
        <p className="settings-hint">
          Assistant requires Session History Memory. Enable it from Connectivity to edit or run this role.
        </p>
      )}
      <ModelSelector {...props} isRunning={controlsDisabled} />
      <div className="settings-row">
        <label>Context Window</label>
        <input
          type="number"
          min={4096}
          step={1024}
          value={config.contextWindow ?? DEFAULT_CONTEXT_WINDOW}
          disabled={controlsDisabled}
          onChange={(event) => updateNumber('contextWindow', event.target.value, '')}
        />
      </div>
      <div className="settings-row">
        <label>Max Output Tokens</label>
        <input
          type="number"
          min={1000}
          step={1000}
          value={config.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS}
          disabled={controlsDisabled}
          onChange={(event) => updateNumber('maxOutputTokens', event.target.value, '')}
        />
      </div>
      {developerModeEnabled && (
        <div className="settings-row settings-row--inline-checkbox">
          <label className="settings-checkbox-label settings-checkbox-label--supercharge">
            <input
              type="checkbox"
              checked={Boolean(config.superchargeEnabled)}
              disabled={controlsDisabled}
              onChange={(event) => onChange({ ...config, superchargeEnabled: event.target.checked })}
            />
            <HelpTooltip
              label="Learn about Supercharge"
              buttonContent="Supercharge"
              buttonClassName="help-tooltip-btn--text"
              popupClassName="help-tooltip-popup--fixed"
              useFixedPosition
            >
              {SUPERCHARGE_TOOLTIP}
            </HelpTooltip>
          </label>
        </div>
      )}
    </div>
  );
}
export default function LeanOJSettings({
  settings,
  onSettingsChange,
  capabilities,
  connectivityStatus,
  credentialStatusRefreshToken = 0,
  isRunning,
  developerModeEnabled = false,
}) {
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [openAICodexModels, setOpenAICodexModels] = useState([]);
  const [xaiGrokModels, setXaiGrokModels] = useState([]);
  const [sakanaFuguModels, setSakanaFuguModels] = useState([]);
  const [modelProviders, setModelProviders] = useState(settings.modelProviders || {});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [hasOpenAICodexLogin, setHasOpenAICodexLogin] = useState(false);
  const [hasXAIGrokLogin, setHasXAIGrokLogin] = useState(false);
  const [hasSakanaFuguKey, setHasSakanaFuguKey] = useState(false);
  const [openAICodexModelError, setOpenAICodexModelError] = useState('');
  const [xaiGrokModelError, setXaiGrokModelError] = useState('');
  const [sakanaFuguModelError, setSakanaFuguModelError] = useState('');
  const [userProfiles, setUserProfiles] = useState({});
  const [selectedProfile, setSelectedProfile] = useState(settings.selectedProfile || '');
  const [profileApplyError, setProfileApplyError] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [advancedSettingsExpanded, setAdvancedSettingsExpanded] = useState(false);
  const [editRawSettings, setEditRawSettings] = useState(false);
  const [rawSettingsText, setRawSettingsText] = useState('');
  const [rawSettingsMessage, setRawSettingsMessage] = useState('');
  const [guiSettingsBeforeRaw, setGuiSettingsBeforeRaw] = useState(null);
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const genericMode = Boolean(capabilities?.genericMode);
  const openAICodexOauthAvailable = !genericMode && capabilities?.openAICodexOauthAvailable !== false;
  const xaiGrokOauthAvailable = !genericMode && capabilities?.xaiGrokOauthAvailable !== false;
  const sakanaFuguAvailable = !genericMode && capabilities?.sakanaFuguAvailable !== false;
  const assistantMemoryEnabled = connectivityStatus?.skills?.agent_conversation_memory?.enabled === true;

  useEffect(() => {
    if (!developerModeEnabled && editRawSettings) {
      setEditRawSettings(false);
      setRawSettingsMessage('');
    }
  }, [developerModeEnabled, editRawSettings]);

  useEffect(() => {
    const load = async () => {
      try {
        const status = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(Boolean(status.has_key));
        if (status.has_key) {
          const openRouter = await openRouterAPI.getModels(null, settings.freeOnly);
          setOpenRouterModels(openRouter.models || []);
        }
      } catch (error) {
        console.error('Failed to load OpenRouter state for Proof Solver:', error);
      }

      if (openAICodexOauthAvailable) {
        try {
          const codexStatus = await cloudAccessAPI.getOpenAICodexStatus();
          const configured = Boolean(codexStatus.status?.configured);
          setHasOpenAICodexLogin(configured);
          if (configured) {
            const codexModels = await cloudAccessAPI.getOpenAICodexModels();
            const models = codexModels.models || [];
            setOpenAICodexModels(models);
            setHasOpenAICodexLogin(true);
            setOpenAICodexModelError(models.length > 0
              ? ''
              : 'OpenAI Codex OAuth is connected, but no Codex models were returned. Reconnect OAuth or check account access.'
            );
          } else {
            setOpenAICodexModelError('');
          }
        } catch (error) {
          console.error('Failed to load OpenAI Codex state for Proof Solver:', error);
          setOpenAICodexModels([]);
          setOpenAICodexModelError(`OpenAI Codex OAuth models could not be loaded: ${error.message || 'unknown error'}.`);
        }
      } else {
        setHasOpenAICodexLogin(false);
        setOpenAICodexModels([]);
        setOpenAICodexModelError('');
      }

      if (xaiGrokOauthAvailable) {
        try {
          const xaiStatus = await cloudAccessAPI.getXAIGrokStatus();
          const configured = Boolean(xaiStatus.status?.configured);
          setHasXAIGrokLogin(configured);
          if (configured) {
            const xaiModels = await cloudAccessAPI.getXAIGrokModels();
            const models = xaiModels.models || [];
            setXaiGrokModels(models);
            setHasXAIGrokLogin(true);
            setXaiGrokModelError(models.length > 0
              ? ''
              : 'xAI Grok OAuth is connected, but no Grok models were returned. Reconnect OAuth or check account access.'
            );
          } else {
            setXaiGrokModelError('');
          }
        } catch (error) {
          console.error('Failed to load xAI Grok state for Proof Solver:', error);
          setXaiGrokModels([]);
          setXaiGrokModelError(`xAI Grok OAuth models could not be loaded: ${error.message || 'unknown error'}.`);
        }
      } else {
        setHasXAIGrokLogin(false);
        setXaiGrokModels([]);
        setXaiGrokModelError('');
      }

      if (sakanaFuguAvailable) {
        try {
          const sakanaStatus = await cloudAccessAPI.getSakanaFuguStatus();
          const configured = Boolean(sakanaStatus.status?.configured);
          setHasSakanaFuguKey(configured);
          if (configured) {
            const sakanaModels = await cloudAccessAPI.getSakanaFuguModels();
            const models = sakanaModels.models || [];
            setSakanaFuguModels(models);
            setHasSakanaFuguKey(true);
            setSakanaFuguModelError(models.length > 0
              ? ''
              : 'Sakana Fugu API key is saved, but no Fugu models were returned. Check your Sakana subscription access.'
            );
          } else {
            setSakanaFuguModelError('');
          }
        } catch (error) {
          console.error('Failed to load Sakana Fugu state for Proof Solver:', error);
          setSakanaFuguModels([]);
          setSakanaFuguModelError(`Sakana Fugu models could not be loaded: ${error.message || 'unknown error'}.`);
        }
      } else {
        setHasSakanaFuguKey(false);
        setSakanaFuguModels([]);
        setSakanaFuguModelError('');
      }

      if (lmStudioEnabled) {
        try {
          const models = await api.getModels();
          setLmStudioModels(models.models || models || []);
        } catch (error) {
          console.error('Failed to load LM Studio models for Proof Solver:', error);
        }
      }

      try {
        setUserProfiles(JSON.parse(localStorage.getItem(LEANOJ_PROFILES_STORAGE_KEY) || '{}'));
      } catch {
        setUserProfiles({});
      }
    };
    load();
  }, [lmStudioEnabled, settings.freeOnly, openAICodexOauthAvailable, xaiGrokOauthAvailable, sakanaFuguAvailable]);

  useEffect(() => {
    if (credentialStatusRefreshToken === 0) {
      return;
    }

    let isCurrent = true;
    refreshCredentialProviderState({
      freeOnly: settings.freeOnly,
      openAICodexOauthAvailable,
      xaiGrokOauthAvailable,
      sakanaFuguAvailable,
      setHasOpenRouterKey,
      setOpenRouterModels,
      setHasOpenAICodexLogin,
      setOpenAICodexModels,
      setOpenAICodexModelError,
      setHasXAIGrokLogin,
      setXaiGrokModels,
      setXaiGrokModelError,
      setHasSakanaFuguKey,
      setSakanaFuguModels,
      setSakanaFuguModelError,
      shouldApply: () => isCurrent,
      logContext: 'Proof Solver settings',
    });
    return () => {
      isCurrent = false;
    };
  }, [credentialStatusRefreshToken]);

  useEffect(() => {
    setSelectedProfile(settings.selectedProfile || '');
  }, [settings.selectedProfile]);

  const persistSettings = (nextSettings, { markCustom = false } = {}) => {
    const finalSettings = markCustom
      ? { ...nextSettings, selectedProfile: '' }
      : nextSettings;
    const next = persistLeanOJSettings(finalSettings);
    if (markCustom && profileApplyError) {
      setProfileApplyError('');
    }
    if (markCustom && selectedProfile) {
      setSelectedProfile('');
    } else {
      setSelectedProfile(next.selectedProfile || '');
    }
    onSettingsChange(next);
    return next;
  };

  const updateSettings = (patch) => persistSettings({
    ...settings,
    ...patch,
    modelProviders,
  }, { markCustom: true });

  const shouldAutoFillRole = (previousConfig = {}, config = {}) => (
    (config.provider === 'openrouter' || isCloudAccessProvider(config.provider)) && config.modelId && (
      previousConfig.provider !== config.provider ||
      previousConfig.modelId !== config.modelId ||
      previousConfig.openrouterProvider !== config.openrouterProvider
    )
  );

  const updateRoles = (roleKeys, config) => {
    const shouldAutoFill = roleKeys.some((roleKey) => shouldAutoFillRole(settings.roles[roleKey], config));
    const roles = { ...settings.roles };
    roleKeys.forEach((roleKey) => {
      roles[roleKey] = config;
    });
    const nextSettings = updateSettings({
      roles,
    });
    if (shouldAutoFill) {
      maybeApplyAutoSettingsToRoles(roleKeys, config, nextSettings);
    }
  };

  const updateSubmitter = (index, config) => {
    const previousConfig = settings.submitterConfigs[index] || {};
    const shouldAutoFill = (config.provider === 'openrouter' || isCloudAccessProvider(config.provider)) && config.modelId && (
      previousConfig.provider !== config.provider ||
      previousConfig.modelId !== config.modelId ||
      previousConfig.openrouterProvider !== config.openrouterProvider
    );
    const submitterConfigs = [...settings.submitterConfigs];
    submitterConfigs[index] = { ...config, submitterId: index + 1 };
    const patch = { submitterConfigs };
    if (index === 0) {
      patch.roles = {
        ...settings.roles,
        topic_generator: toRoleConfig(submitterConfigs[index]),
      };
    }
    const nextSettings = updateSettings(patch);
    if (shouldAutoFill) {
      maybeApplyAutoSettingsToSubmitter(index, submitterConfigs[index], nextSettings);
    }
  };

  const fetchProvidersForModel = async (modelId, baseSettings = settings) => {
    if (!modelId) return null;
    if (hasEndpointMetadata(modelProviders[modelId])) {
      return modelProviders[modelId];
    }
    try {
      const result = await openRouterAPI.getProviders(modelId);
      const nextProviders = {
        ...modelProviders,
        [modelId]: {
          providers: result.providers || [],
          endpoints: result.endpoints || [],
        },
      };
      setModelProviders(nextProviders);
      persistSettings({ ...baseSettings, modelProviders: nextProviders });
      return nextProviders[modelId];
    } catch (error) {
      console.error('Failed to fetch Proof Solver provider list:', error);
      return null;
    }
  };

  const getAutoSettingsForModel = async (modelId, selectedProvider = null, baseSettings = settings) => {
    const model = findOpenRouterModel(openRouterModels, modelId);
    if (!model) {
      console.debug('[ProofSolverAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }
    const providerData = await fetchProvidersForModel(modelId, baseSettings);
    const autoSettings = computeOpenRouterAutoSettings(model, providerData, selectedProvider);
    if (autoSettings?.warnings?.length) {
      console.warn('[ProofSolverAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const getCodexAutoSettingsForModel = (modelId) => {
    const model = openAICodexModels.find((item) => item.id === modelId);
    if (!model) {
      console.debug('[ProofSolverCodexAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }
    const autoSettings = computeCodexAutoSettings(model);
    if (autoSettings.warnings.length > 0) {
      console.warn('[ProofSolverCodexAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const getOAuthModels = (provider) => {
    if (provider === 'openai_codex_oauth') return openAICodexModels;
    if (provider === XAI_GROK_PROVIDER) return xaiGrokModels;
    if (provider === SAKANA_FUGU_PROVIDER) return sakanaFuguModels;
    return [];
  };

  const getCloudAccessAutoSettingsForModel = (provider, modelId) => {
    if (provider === 'openai_codex_oauth') {
      return getCodexAutoSettingsForModel(modelId);
    }
    const model = getOAuthModels(provider).find((item) => item.id === modelId);
    if (!model) {
      console.debug('[ProofSolverOAuthAutoFill] model not in loaded list, skipping auto-fill', { provider, modelId });
      return null;
    }
    const autoSettings = provider === XAI_GROK_PROVIDER
      ? computeXAIGrokAutoSettings(model)
      : (provider === SAKANA_FUGU_PROVIDER
        ? computeSakanaFuguAutoSettings(model)
        : computeCloudAccessAutoSettings(model, cloudAccessProviderLabel(provider)));
    if (autoSettings.warnings.length > 0) {
      console.warn('[ProofSolverOAuthAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const applyAutoSettingsToConfig = async (config, baseSettings = settings) => {
    if (!(config.provider === 'openrouter' || isCloudAccessProvider(config.provider)) || !config.modelId) return;
    const auto = config.provider === 'openrouter'
      ? await getAutoSettingsForModel(config.modelId, config.openrouterProvider || null, baseSettings)
      : getCloudAccessAutoSettingsForModel(config.provider, config.modelId);
    if (!auto) return null;
    return {
      ...config,
      ...(auto.contextWindowKnown ? { contextWindow: auto.contextWindow } : {}),
      ...(auto.outputCapKnown ? { maxOutputTokens: auto.maxOutputTokens } : {}),
    };
  };

  const maybeApplyAutoSettingsToRoles = async (roleKeys, config, baseSettings) => {
    const nextConfig = await applyAutoSettingsToConfig(config, baseSettings);
    if (!nextConfig) return;
    const roles = { ...baseSettings.roles };
    roleKeys.forEach((roleKey) => {
      roles[roleKey] = nextConfig;
    });
    persistSettings({
      ...baseSettings,
      roles,
    });
  };

  const maybeApplyAutoSettingsToSubmitter = async (index, config, baseSettings) => {
    const nextConfig = await applyAutoSettingsToConfig(config, baseSettings);
    if (!nextConfig) return;
    const submitterConfigs = [...baseSettings.submitterConfigs];
    submitterConfigs[index] = { ...nextConfig, submitterId: index + 1 };
    const nextSettings = {
      ...baseSettings,
      submitterConfigs,
    };
    if (index === 0) {
      nextSettings.roles = {
        ...baseSettings.roles,
        topic_generator: toRoleConfig(submitterConfigs[index]),
      };
    }
    persistSettings(nextSettings);
  };

  const handleProfileSelect = async (profileKey) => {
    if (!profileKey) {
      setSelectedProfile('');
      setProfileApplyError('');
      persistSettings({ ...settings, selectedProfile: '' });
      return;
    }
    try {
      setProfileApplyError('');
      const { settings: nextSettings } = await applyLeanOJProfileSelection(profileKey, userProfiles);
      setSelectedProfile(nextSettings.selectedProfile || profileKey);
      onSettingsChange(nextSettings);
      setModelProviders(nextSettings.modelProviders || {});
    } catch (error) {
      console.error(error.message || 'Failed to apply Proof Solver profile:', error);
      setProfileApplyError(error.message || 'Failed to apply Proof Solver profile.');
    }
  };

  const saveCurrentAsProfile = () => {
    if (!newProfileName.trim()) {
      alert('Please enter a profile name');
      return;
    }
    const key = `user_${Date.now()}`;
    const nextProfiles = {
      ...userProfiles,
      [key]: {
        name: newProfileName.trim(),
        numSubmitters: settings.submitterConfigs.length,
        submitters: settings.submitterConfigs.map((config) => ({
          modelId: config.modelId,
          provider: config.provider,
          openrouterProvider: config.openrouterProvider,
          openrouterReasoningEffort: config.openrouterReasoningEffort,
          lmStudioFallbackId: config.lmStudioFallbackId,
          contextWindow: config.contextWindow,
          maxOutputTokens: config.maxOutputTokens,
          superchargeEnabled: Boolean(config.superchargeEnabled),
        })),
        roles: settings.roles,
        maxInitialBrainstormAccepts: settings.maxInitialBrainstormAccepts,
        maxRecursiveBrainstormAccepts: settings.maxRecursiveBrainstormAccepts,
        finalAttemptsPerCycle: settings.finalAttemptsPerCycle,
      },
    };
    localStorage.setItem(LEANOJ_PROFILES_STORAGE_KEY, JSON.stringify(nextProfiles));
    setUserProfiles(nextProfiles);
    const nextSettings = persistLeanOJSettings({ ...settings, selectedProfile: key });
    setSelectedProfile(key);
    setProfileApplyError('');
    onSettingsChange(nextSettings);
    setShowSaveDialog(false);
    setNewProfileName('');
  };

  const deleteProfile = (profileKey) => {
    if (!profileKey.startsWith('user_')) {
      alert('Cannot delete recommended profiles');
      return;
    }
    const profileToDelete = userProfiles[profileKey];
    if (!profileToDelete) {
      console.error(`Proof Solver profile ${profileKey} not found`);
      return;
    }
    if (!confirm(`Delete profile "${profileToDelete.name}"?`)) {
      return;
    }
    const nextProfiles = { ...userProfiles };
    delete nextProfiles[profileKey];
    localStorage.setItem(LEANOJ_PROFILES_STORAGE_KEY, JSON.stringify(nextProfiles));
    setUserProfiles(nextProfiles);
    if (selectedProfile === profileKey) {
      setSelectedProfile('');
      setProfileApplyError('');
      onSettingsChange(persistLeanOJSettings({ ...settings, selectedProfile: '' }));
    }
  };

  const setSubmitterCount = (count) => {
    const nextCount = Math.max(1, Math.min(10, Number(count) || 1));
    const submitterConfigs = [...settings.submitterConfigs];
    while (submitterConfigs.length < nextCount) {
      submitterConfigs.push({
        ...(submitterConfigs[submitterConfigs.length - 1] || settings.roles.final_solver),
        submitterId: submitterConfigs.length + 1,
      });
    }
    updateSettings({
      numSubmitters: nextCount,
      submitterConfigs: submitterConfigs.slice(0, nextCount).map((config, index) => ({ ...config, submitterId: index + 1 })),
    });
  };

  const toggleAdvancedSettings = () => {
    setAdvancedSettingsExpanded((expanded) => !expanded);
  };

  const refreshLmStudioModels = async () => {
    if (!lmStudioEnabled) return;
    try {
      const models = await api.getModels();
      setLmStudioModels(models.models || models || []);
    } catch (error) {
      console.error('Failed to refresh LM Studio models for Proof Solver:', error);
    }
  };

  const refreshOpenRouterModels = async (freeFilter = settings.freeOnly) => {
    if (!hasOpenRouterKey) return;
    try {
      const openRouter = await openRouterAPI.getModels(null, freeFilter);
      setOpenRouterModels(openRouter.models || []);
    } catch (error) {
      console.error('Failed to refresh OpenRouter models for Proof Solver:', error);
    }
  };

  const getLeanOJRawSettings = () => ({
    ...settings,
    modelProviders,
  });

  const applyLeanOJRawSettings = (rawSettings, { updateRawText = true } = {}) => {
    const nextSettings = persistLeanOJSettings({
      ...settings,
      ...rawSettings,
    });
    setModelProviders(nextSettings.modelProviders || {});
    setSelectedProfile(nextSettings.selectedProfile || '');
    setProfileApplyError('');
    onSettingsChange(nextSettings);
    if (updateRawText) {
      setRawSettingsText(formatRawSettings(nextSettings));
    }
  };

  const handleRawEditToggle = (checked) => {
    if (checked) {
      const currentSettings = getLeanOJRawSettings();
      setGuiSettingsBeforeRaw(currentSettings);
      setRawSettingsText(formatRawSettings(currentSettings));
      setRawSettingsMessage('');
      setEditRawSettings(true);
      return;
    }

    if (!confirm(RAW_VIEW_EXIT_WARNING)) {
      return;
    }

    if (guiSettingsBeforeRaw) {
      applyLeanOJRawSettings(guiSettingsBeforeRaw, { updateRawText: false });
    }
    setRawSettingsMessage('');
    setEditRawSettings(false);
  };

  const saveRawSettings = () => {
    try {
      const parsed = JSON.parse(rawSettingsText);
      applyLeanOJRawSettings(parsed);
      setRawSettingsMessage('Saved raw settings.');
    } catch (error) {
      setRawSettingsMessage(`Invalid JSON: ${error.message}`);
    }
  };

  const handleAdvancedSettingsKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleAdvancedSettings();
    }
  };

  return (
    <div className="settings-with-model-sidebar">
      <HighlightedModelsSidebar />
      <div className="settings-with-model-sidebar__main">
        <div className="settings-panel">
          <h2>Proof Solver Model Selection & Settings</h2>
          <p className="settings-hint">These profiles are dedicated to Proof Solver runs and do not change Autonomous, Aggregator, or Compiler profiles.</p>

      <div className="settings-section">
        <h3>Proof Solver Profiles</h3>
        <div className="settings-row">
          <label>Apply Profile</label>
          <select value={selectedProfile} disabled={isRunning} onChange={(event) => handleProfileSelect(event.target.value)}>
            <option value="">-- Custom Settings --</option>
            <optgroup label="Recommended Profiles">
              {Object.keys(LEANOJ_RECOMMENDED_PROFILES).map((key) => (
                <option key={key} value={key}>{LEANOJ_RECOMMENDED_PROFILES[key].name}</option>
              ))}
            </optgroup>
            {Object.keys(userProfiles).length > 0 && (
              <optgroup label="My Profiles">
                {Object.keys(userProfiles)
                  .sort((a, b) => userProfiles[a].name.localeCompare(userProfiles[b].name))
                  .map((key) => (
                    <option key={key} value={key}>{userProfiles[key].name}</option>
                  ))}
              </optgroup>
            )}
          </select>
          {profileApplyError && (
            <div className="settings-error" role="alert">
              {profileApplyError}
            </div>
          )}
        </div>
        <div className="settings-row">
          <button type="button" className="secondary" disabled={isRunning} onClick={() => setShowSaveDialog(true)}>
            Save as Profile
          </button>
          {selectedProfile && selectedProfile.startsWith('user_') && (
            <button
              type="button"
              className="secondary"
              disabled={isRunning}
              onClick={() => deleteProfile(selectedProfile)}
              style={{ marginLeft: '0.5rem', backgroundColor: '#e74c3c' }}
            >
              Delete Profile
            </button>
          )}
        </div>
      </div>

      {showSaveDialog && (
        <div className="inline-modal-overlay">
          <div className="inline-modal-content">
            <h3 style={{ marginTop: 0 }}>Save Proof Solver Profile</h3>
            <p className="label--muted">
              Enter a name for this profile. Current Proof Solver model settings will be saved.
            </p>
            <input
              type="text"
              value={newProfileName}
              onChange={(event) => setNewProfileName(event.target.value)}
              placeholder="Profile name..."
              className="input-dark"
              onKeyPress={(event) => {
                if (event.key === 'Enter') {
                  saveCurrentAsProfile();
                }
              }}
              autoFocus
            />
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
              <button
                className="secondary"
                onClick={() => {
                  setShowSaveDialog(false);
                  setNewProfileName('');
                }}
              >
                Cancel
              </button>
              <button className="btn-success-sm" onClick={saveCurrentAsProfile}>
                Save Profile
              </button>
            </div>
          </div>
        </div>
      )}

      {openAICodexModelError && (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {openAICodexModelError}
        </div>
      )}
      {xaiGrokModelError && (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {xaiGrokModelError}
        </div>
      )}

      <div className="model-refresh-controls">
        {lmStudioEnabled && (
          <button type="button" className="secondary" disabled={isRunning} onClick={refreshLmStudioModels}>
            Refresh LM Studio Models
          </button>
        )}
        {hasOpenRouterKey && (
          <>
            <button type="button" className="secondary" disabled={isRunning} onClick={() => refreshOpenRouterModels(settings.freeOnly)}>
              Refresh OpenRouter Models
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => window.open('https://openrouter.ai/models', '_blank', 'noopener,noreferrer')}
              title="Browse all available OpenRouter models"
            >
              🔗 OpenRouter Model List
            </button>
            <OpenRouterFreeModelsControl
              checked={settings.freeOnly}
              disabled={isRunning}
              onChange={(freeOnly) => updateSettings({ freeOnly })}
            />
          </>
        )}
        {developerModeEnabled && (
          <>
          {hasOpenRouterKey && <span className="model-refresh-controls__divider" aria-hidden="true" />}
          <label className="settings-checkbox-label model-refresh-controls__toggle" style={{ cursor: isRunning ? 'not-allowed' : 'pointer' }}>
            <input
              type="checkbox"
              checked={editRawSettings}
              onChange={(event) => handleRawEditToggle(event.target.checked)}
              disabled={isRunning}
            />
            Edit Raw
          </label>
          </>
        )}
      </div>

      {editRawSettings ? (
        <RawSettingsEditor
          value={rawSettingsText}
          onChange={setRawSettingsText}
          onSave={saveRawSettings}
          message={rawSettingsMessage}
          disabled={isRunning}
        />
      ) : (
        <>
      <div className="settings-section">
        <h3>Brainstorm Submitters</h3>
        <div className="settings-row">
          <label>Number of Submitters</label>
          <input type="number" min={1} max={10} disabled={isRunning} value={settings.submitterConfigs.length} onChange={(event) => setSubmitterCount(event.target.value)} />
        </div>
        {settings.submitterConfigs.map((submitter, index) => (
          <RoleEditor
            key={submitter.submitterId || index}
            title={index === 0 ? 'Brainstorm Submitter 1 + Topic Generator' : `Brainstorm Submitter ${index + 1}`}
            config={submitter}
            onChange={(next) => updateSubmitter(index, next)}
            lmStudioModels={lmStudioModels}
            openRouterModels={openRouterModels}
            openAICodexModels={openAICodexModels}
            xaiGrokModels={xaiGrokModels}
            sakanaFuguModels={sakanaFuguModels}
            modelProviders={modelProviders}
            hasOpenRouterKey={hasOpenRouterKey}
            hasOpenAICodexLogin={hasOpenAICodexLogin}
            hasXAIGrokLogin={hasXAIGrokLogin}
            hasSakanaFuguKey={hasSakanaFuguKey}
            isRunning={isRunning}
            lmStudioEnabled={lmStudioEnabled}
            developerModeEnabled={developerModeEnabled}
          />
        ))}
      </div>

      <div className="settings-section">
        <h3>Proof Solver Roles</h3>
        {ROLE_EDITOR_GROUPS.map((group) => (
          <div key={group.key}>
            <RoleEditor
              title={group.title}
              config={settings.roles[group.roleKeys[0]]}
              onChange={(next) => updateRoles(group.roleKeys, next)}
              lmStudioModels={lmStudioModels}
              openRouterModels={openRouterModels}
              openAICodexModels={openAICodexModels}
              xaiGrokModels={xaiGrokModels}
              sakanaFuguModels={sakanaFuguModels}
              modelProviders={modelProviders}
              hasOpenRouterKey={hasOpenRouterKey}
              hasOpenAICodexLogin={hasOpenAICodexLogin}
              hasXAIGrokLogin={hasXAIGrokLogin}
              hasSakanaFuguKey={hasSakanaFuguKey}
              isRunning={isRunning}
              lmStudioEnabled={lmStudioEnabled}
              developerModeEnabled={developerModeEnabled}
              disabled={group.key === 'assistant' && !assistantMemoryEnabled}
            />
          </div>
        ))}
      </div>

      <div className="settings-section">
        <div
          className="collapsible-trigger settings-trigger--multiline"
          onClick={toggleAdvancedSettings}
          onKeyDown={handleAdvancedSettingsKeyDown}
          role="button"
          tabIndex={0}
          aria-expanded={advancedSettingsExpanded}
          aria-controls="leanoj-advanced-settings-panel"
          style={{ marginBottom: advancedSettingsExpanded ? '1rem' : 0 }}
        >
          <div className="settings-heading-stack">
            <h3 className="settings-trigger-title">Advanced Settings</h3>
            <p className="settings-subsection-description">
              Tune Proof Solver run limits and proof-search loop budgets.
            </p>
          </div>
          <span className={`collapse-chevron${advancedSettingsExpanded ? ' collapse-chevron--open' : ''}`}>▼</span>
        </div>

        {advancedSettingsExpanded && (
          <div className="collapsible-body settings-advanced-content" id="leanoj-advanced-settings-panel">
            <div className="settings-subsection">
              <div className="settings-subsection-header">
                <h3 className="settings-subsection-title">Run Limits</h3>
                <p className="settings-subsection-description">
                  Control how many brainstorm ideas and final proof attempts Proof Solver will run. Inline brainstorm proof candidates use a fixed 5-attempt Lean repair gate.
                </p>
              </div>
              <div className="settings-row">
                <label>Initial Brainstorm Accepts</label>
                <input type="number" min={1} max={200} disabled={isRunning} value={settings.maxInitialBrainstormAccepts} onChange={(event) => updateSettings({ maxInitialBrainstormAccepts: Number(event.target.value) || 30 })} />
              </div>
              <div className="settings-row">
                <label>Recursive Brainstorm Accepts</label>
                <input type="number" min={1} max={100} disabled={isRunning} value={settings.maxRecursiveBrainstormAccepts} onChange={(event) => updateSettings({ maxRecursiveBrainstormAccepts: Number(event.target.value) || 10 })} />
              </div>
              <div className="settings-row">
                <label>Final Attempts Per Cycle</label>
                <input type="number" min={30} max={200} disabled={isRunning} value={settings.finalAttemptsPerCycle} onChange={(event) => updateSettings({ finalAttemptsPerCycle: Math.max(30, Number(event.target.value) || 30) })} />
              </div>
            </div>
          </div>
        )}
      </div>
        </>
      )}
        </div>
      </div>
    </div>
  );
}
