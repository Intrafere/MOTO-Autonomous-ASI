import React, { useState, useEffect } from 'react';
import { cloudAccessAPI, openRouterAPI } from '../services/api';
import {
  CLOUD_ACCESS_PROVIDERS,
  OPENAI_CODEX_PROVIDER,
  SAKANA_FUGU_PROVIDER,
  XAI_GROK_PROVIDER,
  chooseDefaultCloudAccessProvider,
  getConfiguredCloudAccessProviders,
} from '../utils/oauthProviders';
import './settings-common.css';

/**
 * Modal for configuring cloud provider access.
 * 
 * Shows when:
 * 1. User clicks the OpenRouter & Cloud Subscriptions header row
 * 2. User clicks "Use OpenRouter" on any role but no API key is configured
 * 3. LM Studio is unavailable and user needs cloud access as primary provider
 */
export default function OpenRouterApiKeyModal({
  isOpen,
  onClose,
  onKeySet,
  onKeyCleared,
  onCloudAccessChanged,
  reason = 'setup',
  capabilities,
}) {
  const [apiKey, setApiKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState('');
  const [hasStoredKey, setHasStoredKey] = useState(false);
  const [codexStatus, setCodexStatus] = useState({ configured: false });
  const [codexLoading, setCodexLoading] = useState(false);
  const [codexState, setCodexState] = useState('');
  const [codexRedirectUri, setCodexRedirectUri] = useState('');
  const [codexCallbackInput, setCodexCallbackInput] = useState('');
  const [codexMessage, setCodexMessage] = useState('');
  const [codexLoginBaselineConfigured, setCodexLoginBaselineConfigured] = useState(false);
  const [codexLoginBaselineUpdatedAt, setCodexLoginBaselineUpdatedAt] = useState(0);
  const [codexModelsStatus, setCodexModelsStatus] = useState({
    checking: false,
    count: null,
    error: '',
  });
  const codexModelCheckRequestRef = React.useRef(0);
  const [xaiStatus, setXaiStatus] = useState({ configured: false });
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiState, setXaiState] = useState('');
  const [xaiRedirectUri, setXaiRedirectUri] = useState('');
  const [xaiCallbackInput, setXaiCallbackInput] = useState('');
  const [xaiMessage, setXaiMessage] = useState('');
  const [xaiLoginBaselineConfigured, setXaiLoginBaselineConfigured] = useState(false);
  const [xaiLoginBaselineUpdatedAt, setXaiLoginBaselineUpdatedAt] = useState(0);
  const [xaiModelsStatus, setXaiModelsStatus] = useState({
    checking: false,
    count: null,
    error: '',
  });
  const xaiModelCheckRequestRef = React.useRef(0);
  const [sakanaApiKey, setSakanaApiKey] = useState('');
  const [sakanaStatus, setSakanaStatus] = useState({ configured: false });
  const [sakanaLoading, setSakanaLoading] = useState(false);
  const [sakanaMessage, setSakanaMessage] = useState('');
  const [sakanaModelsStatus, setSakanaModelsStatus] = useState({
    checking: false,
    count: null,
    error: '',
  });
  const sakanaModelCheckRequestRef = React.useRef(0);
  const [selectedOAuthProvider, setSelectedOAuthProvider] = useState(OPENAI_CODEX_PROVIDER);
  const [oauthProviderTouched, setOauthProviderTouched] = useState(false);
  const genericMode = Boolean(capabilities?.genericMode);
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const oauthStatusByProvider = {
    [OPENAI_CODEX_PROVIDER]: codexStatus,
    [XAI_GROK_PROVIDER]: xaiStatus,
    [SAKANA_FUGU_PROVIDER]: sakanaStatus,
  };
  const configuredOAuthProviders = getConfiguredCloudAccessProviders(oauthStatusByProvider);
  const codexOAuthSuccess = Boolean(
    !genericMode
    && codexStatus?.configured
    && !codexModelsStatus.checking
    && codexModelsStatus.count > 0
    && !codexModelsStatus.error
  );
  const xaiOAuthSuccess = Boolean(
    !genericMode
    && xaiStatus?.configured
    && !xaiModelsStatus.checking
    && xaiModelsStatus.count > 0
    && !xaiModelsStatus.error
  );
  const sakanaSuccess = Boolean(
    !genericMode
    && sakanaStatus?.configured
    && !sakanaModelsStatus.checking
    && sakanaModelsStatus.count > 0
    && !sakanaModelsStatus.error
  );
  const oauthSuccessBannerStyle = {
    marginBottom: '1rem',
    border: '2px solid #39ff14',
    background: 'linear-gradient(135deg, rgba(24, 204, 23, 0.38), rgba(57, 255, 20, 0.18))',
    boxShadow: '0 0 22px rgba(57, 255, 20, 0.45)',
    color: '#eaffea',
    fontWeight: 800,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  };
  const getOAuthMessageBannerClass = (message) => {
    if (!message) return 'test-result-banner';
    if (message.includes('needs attention')) return 'test-result-banner test-result-banner--error';
    if (message.includes('saved and model list loaded')) return 'test-result-banner test-result-banner--success';
    return 'test-result-banner';
  };

  useEffect(() => {
    if (!isOpen || oauthProviderTouched) return;
    const nextProvider = chooseDefaultCloudAccessProvider(oauthStatusByProvider);
    if (nextProvider !== selectedOAuthProvider) {
      setSelectedOAuthProvider(nextProvider);
    }
  }, [
    isOpen,
    codexStatus?.configured,
    xaiStatus?.configured,
    sakanaStatus?.configured,
    oauthProviderTouched,
    selectedOAuthProvider,
  ]);

  const verifyCodexModels = async () => {
    if (genericMode) {
      codexModelCheckRequestRef.current += 1;
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      return false;
    }

    const requestId = codexModelCheckRequestRef.current + 1;
    codexModelCheckRequestRef.current = requestId;
    setCodexModelsStatus({ checking: true, count: null, error: '' });
    try {
      const result = await cloudAccessAPI.getOpenAICodexModels();
      if (codexModelCheckRequestRef.current !== requestId) {
        return null;
      }
      const models = Array.isArray(result.models) ? result.models : [];
      if (models.length === 0) {
        setCodexModelsStatus({
          checking: false,
          count: 0,
          error: 'Codex login is saved, but no Codex models were returned. Reconnect OAuth or check that this ChatGPT account has Codex model access.',
        });
        return false;
      }
      setCodexModelsStatus({ checking: false, count: models.length, error: '' });
      return true;
    } catch (err) {
      if (codexModelCheckRequestRef.current !== requestId) {
        return null;
      }
      setCodexModelsStatus({
        checking: false,
        count: null,
        error: `Codex login is saved, but models could not be loaded: ${err.message || 'unknown error'}. Reconnect OAuth or check this account's Codex access.`,
      });
      return false;
    }
  };

  const verifyXAIGrokModels = async () => {
    if (genericMode) {
      xaiModelCheckRequestRef.current += 1;
      setXaiModelsStatus({ checking: false, count: null, error: '' });
      return false;
    }

    const requestId = xaiModelCheckRequestRef.current + 1;
    xaiModelCheckRequestRef.current = requestId;
    setXaiModelsStatus({ checking: true, count: null, error: '' });
    try {
      const result = await cloudAccessAPI.getXAIGrokModels();
      if (xaiModelCheckRequestRef.current !== requestId) {
        return null;
      }
      const models = Array.isArray(result.models) ? result.models : [];
      if (models.length === 0) {
        setXaiModelsStatus({
          checking: false,
          count: 0,
          error: 'xAI Grok login is saved, but no Grok models were returned. Reconnect OAuth or check this account\'s SuperGrok/X Premium access.',
        });
        return false;
      }
      setXaiModelsStatus({ checking: false, count: models.length, error: '' });
      return true;
    } catch (err) {
      if (xaiModelCheckRequestRef.current !== requestId) {
        return null;
      }
      setXaiModelsStatus({
        checking: false,
        count: null,
        error: `xAI Grok login is saved, but models could not be loaded: ${err.message || 'unknown error'}. Reconnect OAuth or check this account's SuperGrok/X Premium access.`,
      });
      return false;
    }
  };

  const verifySakanaFuguModels = async () => {
    if (genericMode) {
      sakanaModelCheckRequestRef.current += 1;
      setSakanaModelsStatus({ checking: false, count: null, error: '' });
      return false;
    }

    const requestId = sakanaModelCheckRequestRef.current + 1;
    sakanaModelCheckRequestRef.current = requestId;
    setSakanaModelsStatus({ checking: true, count: null, error: '' });
    try {
      const result = await cloudAccessAPI.getSakanaFuguModels();
      if (sakanaModelCheckRequestRef.current !== requestId) {
        return null;
      }
      const models = Array.isArray(result.models) ? result.models : [];
      if (models.length === 0) {
        setSakanaModelsStatus({
          checking: false,
          count: 0,
          error: 'Sakana Fugu API key is saved, but no Fugu models were returned. Check your Sakana subscription access.',
        });
        return false;
      }
      setSakanaModelsStatus({ checking: false, count: models.length, error: '' });
      return true;
    } catch (err) {
      if (sakanaModelCheckRequestRef.current !== requestId) {
        return null;
      }
      setSakanaModelsStatus({
        checking: false,
        count: null,
        error: `Sakana Fugu API key is saved, but models could not be loaded: ${err.message || 'unknown error'}. Check the key and subscription access.`,
      });
      return false;
    }
  };

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setApiKey('');
      setTestResult(null);
      setError('');
      setCodexMessage('');
      setCodexLoginBaselineConfigured(false);
      setCodexLoginBaselineUpdatedAt(0);
      codexModelCheckRequestRef.current += 1;
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      setXaiMessage('');
      setXaiLoginBaselineConfigured(false);
      setXaiLoginBaselineUpdatedAt(0);
      xaiModelCheckRequestRef.current += 1;
      setXaiModelsStatus({ checking: false, count: null, error: '' });
      setSakanaApiKey('');
      setSakanaMessage('');
      sakanaModelCheckRequestRef.current += 1;
      setSakanaModelsStatus({ checking: false, count: null, error: '' });
      setOauthProviderTouched(false);
      setSelectedOAuthProvider(chooseDefaultCloudAccessProvider(oauthStatusByProvider));
      let isCancelled = false;

      const loadKeyStatus = async () => {
        try {
          const status = await openRouterAPI.getApiKeyStatus();
          if (!isCancelled) {
            setHasStoredKey(Boolean(status.has_key));
          }
        } catch {
          if (!isCancelled) {
            setHasStoredKey(false);
          }
        }
      };
      const loadCloudStatus = async () => {
        try {
          const status = await cloudAccessAPI.getOpenAICodexStatus();
          if (!isCancelled) {
            const nextStatus = status.status || { configured: false };
            setCodexStatus(nextStatus);
            if (nextStatus.configured) {
              verifyCodexModels();
            }
          }
        } catch {
          if (!isCancelled) {
            setCodexStatus({ configured: false });
            setCodexModelsStatus({ checking: false, count: null, error: '' });
          }
        }
        try {
          const status = await cloudAccessAPI.getXAIGrokStatus();
          if (!isCancelled) {
            const nextStatus = status.status || { configured: false };
            setXaiStatus(nextStatus);
            if (nextStatus.configured) {
              verifyXAIGrokModels();
            }
          }
        } catch {
          if (!isCancelled) {
            setXaiStatus({ configured: false });
            setXaiModelsStatus({ checking: false, count: null, error: '' });
          }
        }
      };
      const loadSakanaStatus = async () => {
        try {
          const status = await cloudAccessAPI.getSakanaFuguStatus();
          if (!isCancelled) {
            const nextStatus = status.status || { configured: false };
            setSakanaStatus(nextStatus);
            if (nextStatus.configured) {
              verifySakanaFuguModels();
            }
          }
        } catch {
          if (!isCancelled) {
            setSakanaStatus({ configured: false });
            setSakanaModelsStatus({ checking: false, count: null, error: '' });
          }
        }
      };

      loadKeyStatus();
      loadCloudStatus();
      loadSakanaStatus();

      return () => {
        isCancelled = true;
        codexModelCheckRequestRef.current += 1;
        xaiModelCheckRequestRef.current += 1;
        sakanaModelCheckRequestRef.current += 1;
      };
    }
    codexModelCheckRequestRef.current += 1;
    xaiModelCheckRequestRef.current += 1;
    sakanaModelCheckRequestRef.current += 1;
    setHasStoredKey(false);
    return undefined;
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !codexState) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const status = await cloudAccessAPI.getOpenAICodexStatus();
        const nextStatus = status.status || { configured: false };
        setCodexStatus(nextStatus);
        const statusUpdatedAt = Number(nextStatus.updated_at || 0);
        const loginCompleted = nextStatus.configured
          && (
            !codexLoginBaselineConfigured
            || (statusUpdatedAt > 0 && statusUpdatedAt > codexLoginBaselineUpdatedAt)
          );
        if (loginCompleted) {
          const modelsReady = await verifyCodexModels();
          if (modelsReady === null) {
            return;
          }
          setCodexState('');
          setCodexCallbackInput('');
          setCodexMessage(modelsReady
            ? 'OpenAI Codex login saved and model list loaded.'
            : 'OpenAI Codex login saved, but model loading needs attention.'
          );
          if (onCloudAccessChanged) {
            onCloudAccessChanged(true, 'openai_codex_oauth', { modelsReady });
          }
        }
      } catch {
        // Keep waiting; manual paste remains available.
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [isOpen, codexState, codexLoginBaselineConfigured, codexLoginBaselineUpdatedAt, onCloudAccessChanged]);

  useEffect(() => {
    if (!isOpen || !xaiState) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const status = await cloudAccessAPI.getXAIGrokStatus();
        const nextStatus = status.status || { configured: false };
        setXaiStatus(nextStatus);
        const statusUpdatedAt = Number(nextStatus.updated_at || 0);
        const loginCompleted = nextStatus.configured
          && (
            !xaiLoginBaselineConfigured
            || (statusUpdatedAt > 0 && statusUpdatedAt > xaiLoginBaselineUpdatedAt)
          );
        if (loginCompleted) {
          const modelsReady = await verifyXAIGrokModels();
          if (modelsReady === null) {
            return;
          }
          setXaiState('');
          setXaiCallbackInput('');
          setXaiMessage(modelsReady
            ? 'xAI Grok login saved and model list loaded.'
            : 'xAI Grok login saved, but model loading needs attention.'
          );
          if (onCloudAccessChanged) {
            onCloudAccessChanged(true, 'xai_grok_oauth', { modelsReady });
          }
        }
      } catch {
        // Keep waiting; manual paste remains available.
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [isOpen, xaiState, xaiLoginBaselineConfigured, xaiLoginBaselineUpdatedAt, onCloudAccessChanged]);

  const handleTestConnection = async () => {
    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    setTesting(true);
    setError('');
    setTestResult(null);

    try {
      const result = await openRouterAPI.testConnection(apiKey.trim());
      setTestResult(result);
      if (!result.connected) {
        setError(result.message || 'Connection failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to test connection');
      setTestResult({ connected: false });
    } finally {
      setTesting(false);
    }
  };

  const handleSaveKey = async () => {
    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    setSaving(true);
    setError('');

    try {
      // Save to backend
      await openRouterAPI.setApiKey(apiKey.trim());
      setHasStoredKey(true);

      // Notify parent
      if (onKeySet) {
        await onKeySet(apiKey.trim());
      }

      onClose();
    } catch (err) {
      setError(err.message || 'Failed to save API key');
    } finally {
      setSaving(false);
    }
  };

  const handleClearKey = async () => {
    try {
      await openRouterAPI.clearApiKey();
      setApiKey('');
      setTestResult(null);
      setError('');
      setHasStoredKey(false);
      if (onKeyCleared) {
        await onKeyCleared();
      }
    } catch (err) {
      setError(err.message || 'Failed to clear API key');
    }
  };

  const handleStartCodexLogin = async () => {
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    setCodexLoginBaselineConfigured(Boolean(codexStatus?.configured));
    setCodexLoginBaselineUpdatedAt(Number(codexStatus?.updated_at || 0));
    try {
      const result = await cloudAccessAPI.startOpenAICodexLogin();
      setCodexState(result.state || '');
      setCodexRedirectUri(result.redirect_uri || '');
      if (result.authorization_url) {
        window.open(result.authorization_url, '_blank', 'noopener,noreferrer');
      }
      setCodexMessage(result.callback_available
        ? 'OpenAI login opened. MOTO will capture the callback automatically; paste the callback URL or code below if the browser cannot return to MOTO.'
        : 'OpenAI login opened. The local callback port is unavailable, so paste the full callback URL or authorization code below after sign-in.'
      );
    } catch (err) {
      setError(err.message || 'Failed to start OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  const handleCompleteCodexLogin = async () => {
    if (!codexCallbackInput.trim()) {
      setError('Paste the OpenAI callback URL or authorization code first');
      return;
    }
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    try {
      const isUrl = /^https?:\/\//i.test(codexCallbackInput.trim());
      const result = await cloudAccessAPI.exchangeOpenAICodexCode({
        code: isUrl ? '' : codexCallbackInput.trim(),
        redirectUrl: isUrl ? codexCallbackInput.trim() : '',
        state: codexState,
        redirectUri: codexRedirectUri || null,
      });
      setCodexStatus(result.status || { configured: true });
      const modelsReady = await verifyCodexModels();
      if (modelsReady === null) {
        return;
      }
      setCodexCallbackInput('');
      setCodexState('');
      setCodexMessage(modelsReady
        ? 'OpenAI Codex login saved and model list loaded.'
        : 'OpenAI Codex login saved, but model loading needs attention.'
      );
      if (onCloudAccessChanged) {
        onCloudAccessChanged(true, 'openai_codex_oauth', { modelsReady });
      }
    } catch (err) {
      setError(err.message || 'Failed to complete OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  const handleClearCodexLogin = async () => {
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    codexModelCheckRequestRef.current += 1;
    try {
      await cloudAccessAPI.clearOpenAICodexLogin();
      setCodexStatus({ configured: false });
      setCodexCallbackInput('');
      setCodexState('');
      setCodexLoginBaselineConfigured(false);
      setCodexLoginBaselineUpdatedAt(0);
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      setCodexMessage('OpenAI Codex login cleared.');
      if (onCloudAccessChanged) {
        onCloudAccessChanged(false, 'openai_codex_oauth');
      }
    } catch (err) {
      setError(err.message || 'Failed to clear OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  const handleStartXaiLogin = async () => {
    setXaiLoading(true);
    setXaiMessage('');
    setError('');
    setXaiLoginBaselineConfigured(Boolean(xaiStatus?.configured));
    setXaiLoginBaselineUpdatedAt(Number(xaiStatus?.updated_at || 0));
    try {
      const result = await cloudAccessAPI.startXAIGrokLogin();
      setXaiState(result.state || '');
      setXaiRedirectUri(result.redirect_uri || '');
      if (result.authorization_url) {
        window.open(result.authorization_url, '_blank', 'noopener,noreferrer');
      }
      setXaiMessage(result.callback_available
        ? 'xAI Grok login opened. MOTO will capture the callback automatically; paste the callback URL or code below if the browser cannot return to MOTO.'
        : 'xAI Grok login opened. The local callback port is unavailable, so paste the full callback URL or authorization code below after sign-in.'
      );
    } catch (err) {
      setError(err.message || 'Failed to start xAI Grok login');
    } finally {
      setXaiLoading(false);
    }
  };

  const handleCompleteXaiLogin = async () => {
    if (!xaiCallbackInput.trim()) {
      setError('Paste the xAI Grok callback URL or authorization code first');
      return;
    }
    setXaiLoading(true);
    setXaiMessage('');
    setError('');
    try {
      const isUrl = /^https?:\/\//i.test(xaiCallbackInput.trim()) || xaiCallbackInput.trim().startsWith('?');
      const result = await cloudAccessAPI.exchangeXAIGrokCode({
        code: isUrl ? '' : xaiCallbackInput.trim(),
        redirectUrl: isUrl ? xaiCallbackInput.trim() : '',
        state: xaiState,
        redirectUri: xaiRedirectUri || null,
      });
      setXaiStatus(result.status || { configured: true });
      const modelsReady = await verifyXAIGrokModels();
      if (modelsReady === null) {
        return;
      }
      setXaiCallbackInput('');
      setXaiState('');
      setXaiMessage(modelsReady
        ? 'xAI Grok login saved and model list loaded.'
        : 'xAI Grok login saved, but model loading needs attention.'
      );
      if (onCloudAccessChanged) {
        onCloudAccessChanged(true, 'xai_grok_oauth', { modelsReady });
      }
    } catch (err) {
      setError(err.message || 'Failed to complete xAI Grok login');
    } finally {
      setXaiLoading(false);
    }
  };

  const handleClearXaiLogin = async () => {
    setXaiLoading(true);
    setXaiMessage('');
    setError('');
    xaiModelCheckRequestRef.current += 1;
    try {
      await cloudAccessAPI.clearXAIGrokLogin();
      setXaiStatus({ configured: false });
      setXaiCallbackInput('');
      setXaiState('');
      setXaiLoginBaselineConfigured(false);
      setXaiLoginBaselineUpdatedAt(0);
      setXaiModelsStatus({ checking: false, count: null, error: '' });
      setXaiMessage('xAI Grok login cleared.');
      if (onCloudAccessChanged) {
        onCloudAccessChanged(false, 'xai_grok_oauth');
      }
    } catch (err) {
      setError(err.message || 'Failed to clear xAI Grok login');
    } finally {
      setXaiLoading(false);
    }
  };

  const handleSaveSakanaKey = async () => {
    if (!sakanaApiKey.trim()) {
      setError('Please enter a Sakana Fugu API key');
      return;
    }
    setSakanaLoading(true);
    setSakanaMessage('');
    setError('');
    sakanaModelCheckRequestRef.current += 1;
    try {
      const result = await cloudAccessAPI.setSakanaFuguApiKey(sakanaApiKey.trim());
      setSakanaStatus(result.status || { configured: true });
      const models = Array.isArray(result.models) ? result.models : [];
      if (models.length > 0) {
        setSakanaModelsStatus({ checking: false, count: models.length, error: '' });
        setSakanaMessage('Sakana Fugu API key saved and model list loaded.');
        setSakanaApiKey('');
        if (onCloudAccessChanged) {
          onCloudAccessChanged(true, SAKANA_FUGU_PROVIDER, { modelsReady: true });
        }
      } else {
        const modelsReady = await verifySakanaFuguModels();
        setSakanaMessage(modelsReady
          ? 'Sakana Fugu API key saved and model list loaded.'
          : 'Sakana Fugu API key saved, but model loading needs attention.'
        );
        if (onCloudAccessChanged) {
          onCloudAccessChanged(true, SAKANA_FUGU_PROVIDER, { modelsReady: Boolean(modelsReady) });
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to save Sakana Fugu API key');
    } finally {
      setSakanaLoading(false);
    }
  };

  const handleClearSakanaKey = async () => {
    setSakanaLoading(true);
    setSakanaMessage('');
    setError('');
    sakanaModelCheckRequestRef.current += 1;
    try {
      await cloudAccessAPI.clearSakanaFuguApiKey();
      setSakanaApiKey('');
      setSakanaStatus({ configured: false });
      setSakanaModelsStatus({ checking: false, count: null, error: '' });
      setSakanaMessage('Sakana Fugu API key cleared.');
      if (onCloudAccessChanged) {
        onCloudAccessChanged(false, SAKANA_FUGU_PROVIDER);
      }
    } catch (err) {
      setError(err.message || 'Failed to clear Sakana Fugu API key');
    } finally {
      setSakanaLoading(false);
    }
  };

  if (!isOpen) return null;

  const reasonMessages = {
    setup: 'Configure cloud model access for MOTO roles.',
    startup_setup: 'Save cloud access credentials to unlock cloud models. MOTO will apply the recommended default profile immediately, and you can switch profiles later in Settings.',
    startup_codex_oauth: 'Cloud providers are supplementary model providers. Configure OpenRouter or LM Studio first so RAG embeddings are available.',
    startup_oauth: 'Cloud providers are supplementary model providers. Configure OpenRouter or LM Studio first so RAG embeddings are available.',
    lm_studio_unavailable: lmStudioEnabled
      ? 'LM Studio is not available. Configure cloud access to continue.'
      : 'This deployment disables LM Studio. Configure cloud access to continue.',
    no_key: 'An OpenRouter API key is required to use OpenRouter models.',
  };
  const storedKeyCopy = genericMode
    ? 'An OpenRouter API key is already loaded in this running backend instance. Enter a new key below to replace it for this session.'
    : 'An OpenRouter API key is already stored securely on the backend for this machine. Enter a new key below to replace it.';
  const keyStorageFooter = genericMode
    ? 'This API key is held in backend memory for the active hosted/runtime instance and sent to the backend for OpenRouter API calls. API Boost can reuse this key automatically, or you can override it inside the boost modal.'
    : 'This API key is stored securely through the backend keyring integration and sent to the backend for OpenRouter API calls. API Boost can reuse this key automatically, or you can override it inside the boost modal.';

  return (
    <div 
      className="inline-modal-overlay"
      style={{
        zIndex: 10000,
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div 
        className="inline-modal-content"
        style={{
          width: '640px',
          maxWidth: '90vw',
          backgroundColor: '#1a1a2e',
          borderRadius: '12px',
        }}
      >
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>
            OpenRouter & Cloud Subscriptions
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0.25rem',
            }}
          >
            ×
          </button>
        </div>

        <p style={{ color: '#aaa', marginBottom: '1.5rem', fontSize: '0.95rem' }}>
          {reasonMessages[reason] || reasonMessages.setup}
        </p>

        <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
          <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>OpenRouter API Key</h3>
          <p className="settings-hint">
            Use OpenRouter models across MOTO roles. This is the existing cloud-provider path and can still be reused by API Boost.
          </p>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-or-v1-..."
              className="input-dark"
              style={{
                fontSize: '0.95rem',
              }}
            />
            <small className="hint-text hint-text--dim">
              Get your API key at{' '}
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#18cc17' }}
              >
                openrouter.ai/keys
              </a>
            </small>
          </div>

          {testResult && testResult.connected && (
            <div className="test-result-banner test-result-banner--success" style={{
              marginBottom: '1rem',
            }}>
              Connection successful! {testResult.model_count} models available.
            </div>
          )}

          {hasStoredKey && !apiKey.trim() && (
            <div className="test-result-banner test-result-banner--success" style={{
              marginBottom: '1rem',
            }}>
              {storedKeyCopy}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
            <button
              onClick={handleTestConnection}
              disabled={testing || !apiKey.trim()}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#333',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                cursor: testing || !apiKey.trim() ? 'not-allowed' : 'pointer',
                opacity: testing || !apiKey.trim() ? 0.6 : 1,
                fontSize: '0.95rem',
              }}
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </button>

            <button
              onClick={handleSaveKey}
              disabled={saving || !apiKey.trim()}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#18cc17',
                border: 'none',
                borderRadius: '6px',
                color: '#fff',
                cursor: saving || !apiKey.trim() ? 'not-allowed' : 'pointer',
                opacity: saving || !apiKey.trim() ? 0.6 : 1,
                fontSize: '0.95rem',
                fontWeight: '500',
              }}
            >
              {saving ? 'Saving...' : 'Save API Key'}
            </button>
          </div>

          {(apiKey || hasStoredKey) && (
            <button
              onClick={handleClearKey}
              className="btn-ghost"
              style={{
                width: '100%',
                marginTop: '1rem',
                fontSize: '0.85rem',
              }}
            >
              Clear Stored API Key
            </button>
          )}
        </div>

        <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
          <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Cloud Provider Access</h3>
          <p className="settings-hint">
            Sign in with a subscription-backed OAuth provider or save a direct subscription API key for chat/model roles. These providers are supplementary: RAG embeddings still require OpenRouter, LM Studio, or hosted FastEmbed.
          </p>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              Provider
            </label>
            <select
              value={selectedOAuthProvider}
              onChange={(event) => {
                setOauthProviderTouched(true);
                setSelectedOAuthProvider(event.target.value);
              }}
              className="input-dark"
              disabled={genericMode}
            >
              {CLOUD_ACCESS_PROVIDERS.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.label}{oauthStatusByProvider[provider.id]?.configured ? ' (configured)' : ''}
                </option>
              ))}
            </select>
            {configuredOAuthProviders.length === 1 && (
              <small className="hint-text hint-text--dim">
                Auto-selected {configuredOAuthProviders[0].label} because it is the only configured cloud provider.
              </small>
            )}
          </div>

          {selectedOAuthProvider === OPENAI_CODEX_PROVIDER && (
            <>
          {genericMode ? (
            <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
              OpenAI Codex login is desktop-only until hosted callback/proxy support is designed.
            </div>
          ) : codexStatus?.configured ? (
            <div
              className={`test-result-banner ${
                codexModelsStatus.error
                  ? 'test-result-banner--error'
                  : (codexOAuthSuccess ? 'test-result-banner--success' : '')
              }`}
              style={{ marginBottom: '1rem' }}
            >
              {codexModelsStatus.error
                ? `OpenAI Codex OAuth token saved${codexStatus.email ? ` for ${codexStatus.email}` : ''}, but model access is not verified.`
                : `OpenAI Codex login configured${codexStatus.email ? ` for ${codexStatus.email}` : ''}.`}
            </div>
          ) : (
            <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
              OpenAI Codex login is not configured.
            </div>
          )}

          {codexModelsStatus.checking && (
            <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
              Checking Codex model list...
            </div>
          )}

          {!codexModelsStatus.checking && codexModelsStatus.count > 0 && (
            <div className="test-result-banner test-result-banner--success" style={oauthSuccessBannerStyle}>
              SUCCESS! OpenAI Codex OAuth connected. {codexModelsStatus.count} model{codexModelsStatus.count === 1 ? '' : 's'} available.
            </div>
          )}

          {!codexModelsStatus.checking && codexModelsStatus.error && (
            <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
              {codexModelsStatus.error}
            </div>
          )}

          {codexMessage && (
            <div
              className={getOAuthMessageBannerClass(codexMessage)}
              style={{ marginBottom: '1rem' }}
            >
              {codexMessage}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
            <button
              onClick={handleStartCodexLogin}
              disabled={codexLoading || genericMode}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#333',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                cursor: codexLoading || genericMode ? 'not-allowed' : 'pointer',
                opacity: codexLoading || genericMode ? 0.6 : 1,
                fontSize: '0.95rem',
              }}
            >
              {codexLoading ? 'Working...' : 'Start OpenAI Login'}
            </button>
            {codexStatus?.configured && (
              <button
                onClick={handleClearCodexLogin}
                disabled={codexLoading}
                className="btn-ghost"
                style={{ flex: 1 }}
              >
                Clear Codex Login
              </button>
            )}
          </div>

          {codexState && (
            <div style={{ marginTop: '1rem' }}>
              <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                Callback URL or Authorization Code
              </label>
              <input
                type="password"
                value={codexCallbackInput}
                onChange={(e) => setCodexCallbackInput(e.target.value)}
                placeholder="Paste callback URL or code from OpenAI login"
                className="input-dark"
                style={{ fontSize: '0.95rem' }}
              />
              <button
                onClick={handleCompleteCodexLogin}
                disabled={codexLoading || !codexCallbackInput.trim()}
                style={{
                  width: '100%',
                  marginTop: '0.75rem',
                  padding: '0.75rem 1rem',
                  backgroundColor: '#18cc17',
                  border: 'none',
                  borderRadius: '6px',
                  color: '#fff',
                  cursor: codexLoading || !codexCallbackInput.trim() ? 'not-allowed' : 'pointer',
                  opacity: codexLoading || !codexCallbackInput.trim() ? 0.6 : 1,
                  fontSize: '0.95rem',
                  fontWeight: '500',
                }}
              >
                Complete OpenAI Login
              </button>
            </div>
          )}
            </>
          )}

          {selectedOAuthProvider === XAI_GROK_PROVIDER && (
            <>
              <h4 style={{ marginTop: '0.25rem', color: '#ff9f4a', fontSize: '0.95rem' }}>
                xAI Grok Login (SuperGrok / X Premium)
              </h4>
              <p className="settings-hint">
                Sign in with xAI Grok OAuth for subscription-backed Grok models. xAI Console API keys are separate and may use API billing/credits instead of your subscription.
              </p>
              {genericMode ? (
                <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
                  xAI Grok login is desktop-only until hosted callback/proxy support is designed.
                </div>
              ) : xaiStatus?.configured ? (
                <div
                  className={`test-result-banner ${
                    xaiModelsStatus.error
                      ? 'test-result-banner--error'
                      : (xaiOAuthSuccess ? 'test-result-banner--success' : '')
                  }`}
                  style={{ marginBottom: '1rem' }}
                >
                  {xaiModelsStatus.error
                    ? `xAI Grok OAuth token saved${xaiStatus.email ? ` for ${xaiStatus.email}` : ''}, but model access is not verified.`
                    : `xAI Grok login configured${xaiStatus.email ? ` for ${xaiStatus.email}` : ''}.`}
                </div>
              ) : (
                <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
                  xAI Grok login is not configured.
                </div>
              )}

              {xaiModelsStatus.checking && (
                <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
                  Checking Grok model list...
                </div>
              )}

              {!xaiModelsStatus.checking && xaiModelsStatus.count > 0 && (
                <div className="test-result-banner test-result-banner--success" style={oauthSuccessBannerStyle}>
                  SUCCESS! xAI Grok OAuth connected. {xaiModelsStatus.count} model{xaiModelsStatus.count === 1 ? '' : 's'} available.
                </div>
              )}

              {!xaiModelsStatus.checking && xaiModelsStatus.error && (
                <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
                  {xaiModelsStatus.error}
                </div>
              )}

              {xaiMessage && (
                <div
                  className={getOAuthMessageBannerClass(xaiMessage)}
                  style={{ marginBottom: '1rem' }}
                >
                  {xaiMessage}
                </div>
              )}

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                <button
                  onClick={handleStartXaiLogin}
                  disabled={xaiLoading || genericMode}
                  style={{
                    flex: 1,
                    padding: '0.75rem 1rem',
                    backgroundColor: '#333',
                    border: '1px solid #444',
                    borderRadius: '6px',
                    color: '#fff',
                    cursor: xaiLoading || genericMode ? 'not-allowed' : 'pointer',
                    opacity: xaiLoading || genericMode ? 0.6 : 1,
                    fontSize: '0.95rem',
                  }}
                >
                  {xaiLoading ? 'Working...' : 'Start xAI Grok Login'}
                </button>
                {xaiStatus?.configured && (
                  <button
                    onClick={handleClearXaiLogin}
                    disabled={xaiLoading}
                    className="btn-ghost"
                    style={{ flex: 1 }}
                  >
                    Clear Grok Login
                  </button>
                )}
              </div>

              {xaiState && (
                <div style={{ marginTop: '1rem' }}>
                  <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                    Callback URL or Authorization Code
                  </label>
                  <input
                    type="password"
                    value={xaiCallbackInput}
                    onChange={(e) => setXaiCallbackInput(e.target.value)}
                    placeholder="Paste callback URL or code from xAI login"
                    className="input-dark"
                    style={{ fontSize: '0.95rem' }}
                  />
                  <button
                    onClick={handleCompleteXaiLogin}
                    disabled={xaiLoading || !xaiCallbackInput.trim()}
                    style={{
                      width: '100%',
                      marginTop: '0.75rem',
                      padding: '0.75rem 1rem',
                      backgroundColor: '#18cc17',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#fff',
                      cursor: xaiLoading || !xaiCallbackInput.trim() ? 'not-allowed' : 'pointer',
                      opacity: xaiLoading || !xaiCallbackInput.trim() ? 0.6 : 1,
                      fontSize: '0.95rem',
                      fontWeight: '500',
                    }}
                  >
                    Complete xAI Grok Login
                  </button>
                </div>
              )}
            </>
          )}

          {selectedOAuthProvider === SAKANA_FUGU_PROVIDER && (
            <>
              <h4 style={{ marginTop: '0.25rem', color: '#ff9f4a', fontSize: '0.95rem' }}>
                Sakana Fugu API
              </h4>
              <p className="settings-hint">
                Save your Sakana Fugu subscription API key to use Fugu or Fugu Ultra directly as a MOTO role provider.
              </p>
              {genericMode ? (
                <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
                  Sakana Fugu direct API access is desktop-only in this build. Hosted mode should use OpenRouter.
                </div>
              ) : sakanaStatus?.configured ? (
                <div
                  className={`test-result-banner ${
                    sakanaModelsStatus.error
                      ? 'test-result-banner--error'
                      : (sakanaSuccess ? 'test-result-banner--success' : '')
                  }`}
                  style={{ marginBottom: '1rem' }}
                >
                  {sakanaModelsStatus.error
                    ? 'Sakana Fugu API key is saved, but model access is not verified.'
                    : 'Sakana Fugu API key is configured.'}
                </div>
              ) : (
                <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
                  Sakana Fugu API key is not configured.
                </div>
              )}

              {sakanaModelsStatus.checking && (
                <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
                  Checking Sakana Fugu model list...
                </div>
              )}

              {!sakanaModelsStatus.checking && sakanaModelsStatus.count > 0 && (
                <div className="test-result-banner test-result-banner--success" style={oauthSuccessBannerStyle}>
                  SUCCESS! Sakana Fugu connected. {sakanaModelsStatus.count} model{sakanaModelsStatus.count === 1 ? '' : 's'} available.
                </div>
              )}

              {!sakanaModelsStatus.checking && sakanaModelsStatus.error && (
                <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
                  {sakanaModelsStatus.error}
                </div>
              )}

              {sakanaMessage && (
                <div
                  className={getOAuthMessageBannerClass(sakanaMessage)}
                  style={{ marginBottom: '1rem' }}
                >
                  {sakanaMessage}
                </div>
              )}

              <div style={{ marginTop: '1rem' }}>
                <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                  Sakana Fugu API Key
                </label>
                <input
                  type="password"
                  value={sakanaApiKey}
                  onChange={(e) => setSakanaApiKey(e.target.value)}
                  placeholder="Paste Sakana Fugu API key"
                  className="input-dark"
                  disabled={genericMode || sakanaLoading}
                  style={{ fontSize: '0.95rem' }}
                />
                <small className="hint-text hint-text--dim">
                  Get a key from{' '}
                  <a
                    href="https://console.sakana.ai"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#18cc17' }}
                  >
                    console.sakana.ai
                  </a>
                  . MOTO stores it in the backend keyring.
                </small>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                <button
                  onClick={handleSaveSakanaKey}
                  disabled={sakanaLoading || genericMode || !sakanaApiKey.trim()}
                  style={{
                    flex: 1,
                    padding: '0.75rem 1rem',
                    backgroundColor: '#18cc17',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#fff',
                    cursor: sakanaLoading || genericMode || !sakanaApiKey.trim() ? 'not-allowed' : 'pointer',
                    opacity: sakanaLoading || genericMode || !sakanaApiKey.trim() ? 0.6 : 1,
                    fontSize: '0.95rem',
                    fontWeight: '500',
                  }}
                >
                  {sakanaLoading ? 'Working...' : 'Save Sakana Key'}
                </button>
                {sakanaStatus?.configured && (
                  <button
                    onClick={handleClearSakanaKey}
                    disabled={sakanaLoading}
                    className="btn-ghost"
                    style={{ flex: 1 }}
                  >
                    Clear Sakana Key
                  </button>
                )}
              </div>
            </>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="test-result-banner test-result-banner--error" style={{
            marginBottom: '1rem',
          }}>
            {error}
          </div>
        )}

        {/* Info Note */}
        <p style={{ 
          color: '#666', 
          fontSize: '0.8rem', 
          marginTop: '1.5rem',
          padding: '0.75rem',
          backgroundColor: '#0d0d1a',
          borderRadius: '6px',
        }}>
          {keyStorageFooter}
        </p>
      </div>
    </div>
  );
}

