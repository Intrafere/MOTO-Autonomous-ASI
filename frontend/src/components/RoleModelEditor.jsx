import React, { useId } from 'react';
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  formatOpenRouterProviderLabel,
  getOpenRouterProviderTitle,
  getProviderNames,
  getReasoningSupportInfo,
  normalizeOpenRouterReasoningEffort,
  OPENROUTER_REASONING_EFFORT_OPTIONS,
} from '../utils/openRouterSelection';
import {
  chooseCloudAccessProvider,
  isCloudAccessProvider,
} from '../utils/oauthProviders';
import CloudReasoningControl from './CloudReasoningControl';
import HelpTooltip from './HelpTooltip';

const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';

// Inert view used when a caller has no cloud provider state (same shape as useCloudAccessProviders()).
const NO_CLOUD_PROVIDERS = {
  statusByProvider: {},
  configuredProviders: [],
  modelsFor: () => [],
};

// Provider / model / host / reasoning / LM Studio fallback rows for one role.
export function RoleModelSelector({
  config = {},
  onChange,
  lmStudioModels = [],
  openRouterModels = [],
  cloudProviders = NO_CLOUD_PROVIDERS,
  modelProviders = {},
  hasOpenRouterKey,
  isRunning,
  lmStudioEnabled,
}) {
  const provider = lmStudioEnabled ? (config.provider || 'lm_studio') : 'openrouter';
  const oauthStatusByProvider = cloudProviders.statusByProvider;
  const configuredOAuthProviders = cloudProviders.configuredProviders;
  const models = provider === 'openrouter'
    ? openRouterModels
    : (isCloudAccessProvider(provider)
      ? cloudProviders.modelsFor(provider)
      : lmStudioModels);
  const providers = provider === 'openrouter' && config.modelId
    ? getProviderNames(modelProviders[config.modelId])
    : [];
  const reasoningInfo = provider === 'openrouter'
    ? getReasoningSupportInfo(modelProviders[config.modelId], config.openrouterProvider || null)
    : { hasEndpointMetadata: false, supportsReasoning: false };
  const fieldId = useId();

  return (
    <>
      <div className="settings-row">
        <label id={`${fieldId}-provider`}>Provider</label>
        {lmStudioEnabled ? (
          <div className="provider-toggle-group" role="group" aria-labelledby={`${fieldId}-provider`}>
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'lm_studio' ? ' active-lm' : ''}`}
              disabled={isRunning}
              onClick={() => {
                if (provider !== 'lm_studio') {
                  onChange({
                    ...config,
                    provider: 'lm_studio',
                    modelId: '',
                    openrouterProvider: null,
                    openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
                  });
                }
              }}
            >
              LM Studio
            </button>
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'openrouter' ? ' active-or-orange' : ''}`}
              disabled={isRunning || !hasOpenRouterKey}
              onClick={() => {
                if (provider !== 'openrouter') {
                  onChange({
                    ...config,
                    provider: 'openrouter',
                    modelId: '',
                    openrouterProvider: null,
                    openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
                  });
                }
              }}
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
                className="input-dark provider-toggle-select"
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
        <label htmlFor={`${fieldId}-model`}>Model</label>
        <select
          id={`${fieldId}-model`}
          value={config.modelId || ''}
          disabled={isRunning}
          onChange={(event) => onChange({ ...config, provider, modelId: event.target.value, openrouterProvider: null, openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT })}
        >
          <option value="">Select model...</option>
          {config.modelId && !models.some((model) => model.id === config.modelId) && (
            <option value={config.modelId}>{config.modelId} (configured)</option>
          )}
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
          <label htmlFor={`${fieldId}-host`}>Host Provider</label>
          <select
            id={`${fieldId}-host`}
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
          <label htmlFor={`${fieldId}-reasoning`}>Reasoning Effort</label>
          <select
            id={`${fieldId}-reasoning`}
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

      {isCloudAccessProvider(provider) && (
        <CloudReasoningControl
          provider={provider}
          modelId={config.modelId}
          models={models}
          value={config.openrouterReasoningEffort}
          disabled={isRunning}
          onChange={(effort) => onChange({ ...config, provider, openrouterReasoningEffort: effort })}
        />
      )}

      {provider !== 'lm_studio' && lmStudioEnabled && (
        <div className="settings-row">
          <label htmlFor={`${fieldId}-fallback`}>LM Studio Fallback</label>
          <select
            id={`${fieldId}-fallback`}
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

// One role card: title, optional caller-provided notes (children), model selection,
// Context Window, Max Output Tokens, and the developer-only Supercharge checkbox.
export function RoleModelEditor({
  title,
  children,
  config = {},
  onChange,
  lmStudioModels,
  openRouterModels,
  cloudProviders,
  modelProviders,
  hasOpenRouterKey,
  lmStudioEnabled,
  isRunning,
  developerModeEnabled = false,
  disabled = false,
  showSupercharge = true,
}) {
  const controlsDisabled = isRunning || disabled;
  const fieldId = useId();
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
      {children}
      <RoleModelSelector
        config={config}
        onChange={onChange}
        lmStudioModels={lmStudioModels}
        openRouterModels={openRouterModels}
        cloudProviders={cloudProviders}
        modelProviders={modelProviders}
        hasOpenRouterKey={hasOpenRouterKey}
        isRunning={controlsDisabled}
        lmStudioEnabled={lmStudioEnabled}
      />
      <div className="settings-row">
        <label htmlFor={`${fieldId}-context`}>Context Window</label>
        <input
          id={`${fieldId}-context`}
          type="number"
          min={4096}
          step={1024}
          value={config.contextWindow ?? DEFAULT_CONTEXT_WINDOW}
          disabled={controlsDisabled}
          onChange={(event) => updateNumber('contextWindow', event.target.value, '')}
        />
      </div>
      <div className="settings-row">
        <label htmlFor={`${fieldId}-max-output`}>Max Output Tokens</label>
        <input
          id={`${fieldId}-max-output`}
          type="number"
          min={1000}
          step={1000}
          value={config.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS}
          disabled={controlsDisabled}
          onChange={(event) => updateNumber('maxOutputTokens', event.target.value, '')}
        />
      </div>
      {showSupercharge && developerModeEnabled && (
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

export default RoleModelEditor;
