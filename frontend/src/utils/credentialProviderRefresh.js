import { cloudAccessAPI, openRouterAPI } from '../services/api';

const noop = () => {};
const alwaysCurrent = () => true;

function applyIfCurrent(shouldApply, setter, ...args) {
  if (shouldApply()) {
    setter(...args);
  }
}

export async function refreshCredentialProviderState({
  freeOnly = false,
  openAICodexOauthAvailable = false,
  xaiGrokOauthAvailable = false,
  sakanaFuguAvailable = false,
  setHasOpenRouterKey = noop,
  setOpenRouterModels = noop,
  setLoadingOpenRouter = noop,
  setHasOpenAICodexLogin = noop,
  setOpenAICodexModels = noop,
  setOpenAICodexModelError = noop,
  setHasXAIGrokLogin = noop,
  setXaiGrokModels = noop,
  setXaiGrokModelError = noop,
  setHasSakanaFuguKey = noop,
  setSakanaFuguModels = noop,
  setSakanaFuguModelError = noop,
  shouldApply = alwaysCurrent,
  logContext = 'settings',
} = {}) {
  try {
    const status = await openRouterAPI.getApiKeyStatus();
    if (!shouldApply()) return;
    const configured = Boolean(status.has_key);
    applyIfCurrent(shouldApply, setHasOpenRouterKey, configured);
    if (configured) {
      applyIfCurrent(shouldApply, setLoadingOpenRouter, true);
      try {
        const result = await openRouterAPI.getModels(null, freeOnly);
        applyIfCurrent(shouldApply, setOpenRouterModels, result.models || []);
      } catch (error) {
        if (!shouldApply()) return;
        console.error(`Failed to refresh OpenRouter models for ${logContext}:`, error);
      } finally {
        applyIfCurrent(shouldApply, setLoadingOpenRouter, false);
      }
    } else {
      applyIfCurrent(shouldApply, setOpenRouterModels, []);
    }
  } catch (error) {
    if (!shouldApply()) return;
    console.error(`Failed to refresh OpenRouter state for ${logContext}:`, error);
    applyIfCurrent(shouldApply, setHasOpenRouterKey, false);
    applyIfCurrent(shouldApply, setOpenRouterModels, []);
    applyIfCurrent(shouldApply, setLoadingOpenRouter, false);
  }

  if (openAICodexOauthAvailable) {
    try {
      const status = await cloudAccessAPI.getOpenAICodexStatus();
      if (!shouldApply()) return;
      const configured = Boolean(status.status?.configured);
      applyIfCurrent(shouldApply, setHasOpenAICodexLogin, configured);
      if (configured) {
        const result = await cloudAccessAPI.getOpenAICodexModels();
        if (!shouldApply()) return;
        const models = result.models || [];
        applyIfCurrent(shouldApply, setOpenAICodexModels, models);
        applyIfCurrent(shouldApply, setOpenAICodexModelError, models.length > 0
          ? ''
          : 'OpenAI Codex OAuth is connected, but no Codex models were returned. Reconnect OAuth or check account access.'
        );
      } else {
        applyIfCurrent(shouldApply, setOpenAICodexModels, []);
        applyIfCurrent(shouldApply, setOpenAICodexModelError, '');
      }
    } catch (error) {
      if (!shouldApply()) return;
      console.error(`Failed to refresh OpenAI Codex state for ${logContext}:`, error);
      applyIfCurrent(shouldApply, setOpenAICodexModels, []);
      applyIfCurrent(
        shouldApply,
        setOpenAICodexModelError,
        `OpenAI Codex OAuth models could not be loaded: ${error.message || 'unknown error'}.`
      );
    }
  } else {
    applyIfCurrent(shouldApply, setHasOpenAICodexLogin, false);
    applyIfCurrent(shouldApply, setOpenAICodexModels, []);
    applyIfCurrent(shouldApply, setOpenAICodexModelError, '');
  }

  if (xaiGrokOauthAvailable) {
    try {
      const status = await cloudAccessAPI.getXAIGrokStatus();
      if (!shouldApply()) return;
      const configured = Boolean(status.status?.configured);
      applyIfCurrent(shouldApply, setHasXAIGrokLogin, configured);
      if (configured) {
        const result = await cloudAccessAPI.getXAIGrokModels();
        if (!shouldApply()) return;
        const models = result.models || [];
        applyIfCurrent(shouldApply, setXaiGrokModels, models);
        applyIfCurrent(shouldApply, setXaiGrokModelError, models.length > 0
          ? ''
          : 'xAI Grok OAuth is connected, but no Grok models were returned. Reconnect OAuth or check account access.'
        );
      } else {
        applyIfCurrent(shouldApply, setXaiGrokModels, []);
        applyIfCurrent(shouldApply, setXaiGrokModelError, '');
      }
    } catch (error) {
      if (!shouldApply()) return;
      console.error(`Failed to refresh xAI Grok state for ${logContext}:`, error);
      applyIfCurrent(shouldApply, setXaiGrokModels, []);
      applyIfCurrent(
        shouldApply,
        setXaiGrokModelError,
        `xAI Grok OAuth models could not be loaded: ${error.message || 'unknown error'}.`
      );
    }
  } else {
    applyIfCurrent(shouldApply, setHasXAIGrokLogin, false);
    applyIfCurrent(shouldApply, setXaiGrokModels, []);
    applyIfCurrent(shouldApply, setXaiGrokModelError, '');
  }

  if (sakanaFuguAvailable) {
    try {
      const status = await cloudAccessAPI.getSakanaFuguStatus();
      if (!shouldApply()) return;
      const configured = Boolean(status.status?.configured);
      applyIfCurrent(shouldApply, setHasSakanaFuguKey, configured);
      if (configured) {
        const result = await cloudAccessAPI.getSakanaFuguModels();
        if (!shouldApply()) return;
        const models = result.models || [];
        applyIfCurrent(shouldApply, setSakanaFuguModels, models);
        applyIfCurrent(shouldApply, setSakanaFuguModelError, models.length > 0
          ? ''
          : 'Sakana Fugu API key is saved, but no Fugu models were returned. Check your Sakana subscription access.'
        );
      } else {
        applyIfCurrent(shouldApply, setSakanaFuguModels, []);
        applyIfCurrent(shouldApply, setSakanaFuguModelError, '');
      }
    } catch (error) {
      if (!shouldApply()) return;
      console.error(`Failed to refresh Sakana Fugu state for ${logContext}:`, error);
      applyIfCurrent(shouldApply, setSakanaFuguModels, []);
      applyIfCurrent(
        shouldApply,
        setSakanaFuguModelError,
        `Sakana Fugu models could not be loaded: ${error.message || 'unknown error'}.`
      );
    }
  } else {
    applyIfCurrent(shouldApply, setHasSakanaFuguKey, false);
    applyIfCurrent(shouldApply, setSakanaFuguModels, []);
    applyIfCurrent(shouldApply, setSakanaFuguModelError, '');
  }
}
