import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { api, openRouterAPI } from '../../services/api';
import useCloudAccessProviders from '../../hooks/useCloudAccessProviders';
import { refreshOpenRouterCredentialState } from '../../utils/credentialProviderRefresh';
import { getStoredAutonomousSettings } from '../../utils/autonomousProfiles';
import { isCloudAccessProvider } from '../../utils/oauthProviders';
import {
  computeCloudProviderAutoSettings,
  computeOpenRouterAutoSettings,
  findOpenRouterModel,
  hasEndpointMetadata,
  normalizeProviderData,
} from '../../utils/openRouterSelection';
import CloudProviderModelErrors from '../CloudProviderModelErrors';
import { RoleModelEditor } from '../RoleModelEditor';
import {
  TRANSLATOR_SUBMITTER_TITLE,
  TRANSLATOR_VALIDATOR_TITLE,
  applyCopiedRoles,
  describeTranslationError,
  draftFromSettings,
  draftToSettingsRoles,
  isDraftDirty,
  rolesFromAutonomousSettings,
  savedRolesSignature,
  validateTranslatorDraft,
} from './translatorRoleConfig';
import TranslationHelp from './TranslationHelp';

const LOG_CONTEXT = 'Auto Top-p Translate settings';

export const TRANSLATOR_MODELS_EXPLANATION = 'The translator works through each document in order, one logical part at a '
  + 'time, and can correct earlier parts. Each part keeps improving until the validator has accepted three attempts, '
  + 'or the translator confirms twice in a row that an accepted attempt is already perfect.';

export const TRANSLATOR_ROLES_HELP = `The ${TRANSLATOR_SUBMITTER_TITLE} writes each part of the translation and can `
  + `go back to fix earlier parts. The ${TRANSLATOR_VALIDATOR_TITLE} checks every attempt against the original and `
  + 'sends specific feedback when it rejects one. These roles are set only here: they never use Boost or '
  + 'Supercharge, and their model calls are left out of workflow token totals and paper model credits.';

// The draft holds one role per slot.
const ROLE_SLOTS = [
  { slot: 'translator', title: TRANSLATOR_SUBMITTER_TITLE },
  { slot: 'validator', title: TRANSLATOR_VALIDATOR_TITLE },
];

function sameModelRoute(left = {}, right = {}) {
  return left.provider === right.provider
    && left.modelId === right.modelId
    && (left.openrouterProvider || null) === (right.openrouterProvider || null);
}

function shouldAutoFill(previous = {}, next = {}) {
  return (next.provider === 'openrouter' || isCloudAccessProvider(next.provider))
    && Boolean(next.modelId)
    && !sameModelRoute(previous, next);
}

// Translator models tab: a local draft of the translator submitter and validator
// that is saved explicitly. Model lists load like the other settings panels.
export default function TranslatorModelsPanel({
  savedSettings,
  capabilities,
  credentialStatusRefreshToken = 0,
  developerModeEnabled = false,
  onSaveRoles,
}) {
  const idPrefix = useId();
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const modeOptions = useMemo(() => ({ lmStudioEnabled }), [lmStudioEnabled]);
  const rolesSignature = savedRolesSignature(savedSettings);
  const baseline = useMemo(
    () => draftFromSettings(savedSettings, modeOptions),
    [rolesSignature, modeOptions], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const [draft, setDraft] = useState(baseline);
  const [errors, setErrors] = useState([]);
  const [notice, setNotice] = useState('');
  const [saving, setSaving] = useState(false);
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [modelProviders, setModelProviders] = useState({});
  // Slots whose context/max output auto-fill is still loading. Save waits for them, so a
  // saved role never pairs a new model with the previous model's limits.
  const [autoFillPending, setAutoFillPending] = useState({});
  const mountedRef = useRef(false);
  const draftRef = useRef(draft);
  const openRouterModelsRef = useRef(openRouterModels);
  const modelProvidersRef = useRef(modelProviders);
  const autoFillTokensRef = useRef({});
  draftRef.current = draft;
  openRouterModelsRef.current = openRouterModels;

  const cloudProviders = useCloudAccessProviders({
    capabilities,
    refreshToken: credentialStatusRefreshToken,
    logContext: LOG_CONTEXT,
  });
  const cloudProvidersRef = useRef(cloudProviders);
  cloudProvidersRef.current = cloudProviders;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // A new saved role set (first load or a successful save) becomes the draft, and
  // auto-fills for the replaced draft are cancelled. The role cards are locked while a
  // save is in flight and Save waits for pending auto-fills, so no edit is lost to this reset.
  useEffect(() => {
    setDraft(baseline);
    setErrors([]);
    ROLE_SLOTS.forEach(({ slot }) => {
      autoFillTokensRef.current[slot] = (autoFillTokensRef.current[slot] || 0) + 1;
    });
    setAutoFillPending({});
  }, [baseline]);

  useEffect(() => {
    let isCurrent = true;
    const loadOpenRouter = async () => {
      try {
        const status = await openRouterAPI.getApiKeyStatus();
        if (!isCurrent) return;
        const hasKey = Boolean(status?.has_key);
        setHasOpenRouterKey(hasKey);
        if (hasKey) {
          const result = await openRouterAPI.getModels(null, false);
          if (isCurrent) {
            setOpenRouterModels(result?.models || []);
          }
        }
      } catch (error) {
        console.error(`Failed to load OpenRouter state for ${LOG_CONTEXT}:`, error);
      }
    };
    const loadLmStudio = async () => {
      if (!lmStudioEnabled) return;
      try {
        const result = await api.getModels();
        if (isCurrent) {
          setLmStudioModels(Array.isArray(result?.models) ? result.models : (Array.isArray(result) ? result : []));
        }
      } catch (error) {
        console.error(`Failed to load LM Studio models for ${LOG_CONTEXT}:`, error);
      }
    };
    loadOpenRouter();
    loadLmStudio();
    return () => {
      isCurrent = false;
    };
  }, [lmStudioEnabled]);

  useEffect(() => {
    if (credentialStatusRefreshToken <= 0) {
      return undefined;
    }
    let isCurrent = true;
    refreshOpenRouterCredentialState({
      freeOnly: false,
      setHasOpenRouterKey,
      setOpenRouterModels,
      shouldApply: () => isCurrent,
      logContext: LOG_CONTEXT,
    });
    return () => {
      isCurrent = false;
    };
  }, [credentialStatusRefreshToken]);

  // Endpoint metadata is kept in memory only, and reused only when it carries endpoints.
  const fetchProvidersForModel = async (modelId) => {
    const cached = modelProvidersRef.current[modelId];
    if (hasEndpointMetadata(cached)) {
      return cached;
    }
    try {
      const result = await openRouterAPI.getProviders(modelId);
      const data = normalizeProviderData(result);
      if (hasEndpointMetadata(result) && mountedRef.current) {
        modelProvidersRef.current = { ...modelProvidersRef.current, [modelId]: data };
        setModelProviders(modelProvidersRef.current);
      }
      return data;
    } catch (error) {
      console.error(`Failed to fetch OpenRouter hosts for ${LOG_CONTEXT}:`, error);
      return null;
    }
  };

  const computeAutoSettings = async (config) => {
    if (config.provider === 'openrouter') {
      const model = findOpenRouterModel(openRouterModelsRef.current, config.modelId);
      if (!model) return null;
      const providerData = await fetchProvidersForModel(config.modelId);
      return computeOpenRouterAutoSettings(model, providerData, config.openrouterProvider || null);
    }
    if (isCloudAccessProvider(config.provider)) {
      const model = cloudProvidersRef.current.modelsFor(config.provider).find((item) => item.id === config.modelId);
      return model ? computeCloudProviderAutoSettings(config.provider, model) : null;
    }
    return null;
  };

  const markAutoFillPending = (slot, pending) => {
    setAutoFillPending((current) => (
      Boolean(current[slot]) === pending ? current : { ...current, [slot]: pending }
    ));
  };

  // A newer token makes an auto-fill still loading for the slot ignore its result.
  const cancelAutoFill = (slot) => {
    autoFillTokensRef.current[slot] = (autoFillTokensRef.current[slot] || 0) + 1;
    markAutoFillPending(slot, false);
  };

  // Fill context/max output from known metadata only; a result whose role has since
  // changed model, provider, or host is ignored.
  const applyAutoFill = async (slot, config) => {
    const token = (autoFillTokensRef.current[slot] || 0) + 1;
    autoFillTokensRef.current[slot] = token;
    markAutoFillPending(slot, true);
    let auto = null;
    try {
      auto = await computeAutoSettings(config);
    } catch (error) {
      console.error(`Failed to compute model limits for ${LOG_CONTEXT}:`, error);
    }
    if (!mountedRef.current || autoFillTokensRef.current[slot] !== token) {
      return;
    }
    markAutoFillPending(slot, false);
    if (!auto || (!auto.contextWindowKnown && !auto.outputCapKnown)) {
      return;
    }
    setDraft((current) => {
      const role = current[slot];
      if (!role || !sameModelRoute(role, config)) {
        return current;
      }
      return {
        ...current,
        [slot]: {
          ...role,
          ...(auto.contextWindowKnown ? { contextWindow: auto.contextWindow } : {}),
          ...(auto.outputCapKnown ? { maxOutputTokens: auto.maxOutputTokens } : {}),
        },
      };
    });
  };

  const clearFeedback = () => {
    setErrors([]);
    setNotice('');
  };

  const updateRole = (slot, nextRole) => {
    const previous = draftRef.current[slot];
    setDraft((current) => ({ ...current, [slot]: nextRole }));
    clearFeedback();
    if (shouldAutoFill(previous, nextRole)) {
      applyAutoFill(slot, nextRole);
    } else if (!sameModelRoute(previous, nextRole)) {
      cancelAutoFill(slot);
    }
  };

  const copyFromAutonomous = () => {
    const copied = rolesFromAutonomousSettings(getStoredAutonomousSettings(), modeOptions);
    setErrors([]);
    if (!copied) {
      setNotice('No Autonomous Research model settings were found to copy.');
      return;
    }
    setDraft((current) => applyCopiedRoles(current, copied));
    ROLE_SLOTS.filter(({ slot }) => copied[slot]).forEach(({ slot }) => cancelAutoFill(slot));
    const parts = [];
    if (copied.translator) {
      parts.push(`Submitter 1 to the ${TRANSLATOR_SUBMITTER_TITLE}`);
    }
    if (copied.validator) {
      parts.push(`the Validator to the ${TRANSLATOR_VALIDATOR_TITLE}`);
    }
    setNotice(`Copied ${parts.join(' and ')} from Autonomous Research settings. Review, then save.`);
  };

  const dirty = isDraftDirty(draft, baseline, modeOptions);
  const loadingLimits = ROLE_SLOTS.some(({ slot }) => autoFillPending[slot]);

  const discard = () => {
    setDraft(baseline);
    clearFeedback();
    ROLE_SLOTS.forEach(({ slot }) => cancelAutoFill(slot));
  };

  const save = async () => {
    const validationErrors = validateTranslatorDraft(draft, modeOptions);
    setNotice('');
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      return;
    }
    setErrors([]);
    setSaving(true);
    try {
      await onSaveRoles(draftToSettingsRoles(draft, modeOptions));
      if (mountedRef.current) {
        setNotice('Translator models saved.');
      }
    } catch (error) {
      if (mountedRef.current) {
        setErrors([describeTranslationError(error, 'Translator models could not be saved.')]);
      }
    } finally {
      if (mountedRef.current) {
        setSaving(false);
      }
    }
  };

  const editorProps = {
    lmStudioModels,
    openRouterModels,
    cloudProviders,
    modelProviders,
    hasOpenRouterKey,
    lmStudioEnabled,
    isRunning: saving,
    developerModeEnabled,
    showSupercharge: false,
  };

  const headingId = `${idPrefix}-heading`;
  const rolesHelpId = `${idPrefix}-roles-help`;

  return (
    <div className="auto-translate-models">
      <section
        className="auto-translate-section auto-translate-models__head"
        aria-labelledby={headingId}
        aria-describedby={rolesHelpId}
      >
        <div className="auto-translate-section__head">
          <h3 id={headingId} className="auto-translate-section__title">Translator roles</h3>
          <TranslationHelp id={rolesHelpId} label="Learn about the translator roles">
            {TRANSLATOR_ROLES_HELP}
          </TranslationHelp>
          <button
            type="button"
            className="secondary auto-translate-models__copy"
            onClick={copyFromAutonomous}
            disabled={saving}
          >
            Copy from Autonomous Research settings
          </button>
        </div>
        <p className="auto-translate-hint auto-translate-models__intro">{TRANSLATOR_MODELS_EXPLANATION}</p>
        {!lmStudioEnabled && !hasOpenRouterKey && (
          <p className="auto-translate-hint">
            This deployment uses OpenRouter translator models. Add an OpenRouter API key to choose models.
          </p>
        )}
        <CloudProviderModelErrors modelErrors={cloudProviders.modelErrors} />
      </section>

      <div className="auto-translate-models__roles">
        {ROLE_SLOTS.map(({ slot, title }) => (
          <RoleModelEditor
            key={slot}
            title={title}
            config={draft[slot]}
            onChange={(next) => updateRole(slot, next)}
            {...editorProps}
          />
        ))}
      </div>

      {errors.length > 0 && (
        <div className="test-result-banner test-result-banner--error auto-translate-errors" role="alert">
          {errors.length === 1 ? (
            <p>{errors[0]}</p>
          ) : (
            <ul>
              {errors.map((message) => <li key={message}>{message}</li>)}
            </ul>
          )}
        </div>
      )}
      {notice && <p className="auto-translate-hint auto-translate-models__notice" role="status">{notice}</p>}

      <div className="auto-translate-models__actions">
        <button type="button" className="btn-success-sm" onClick={save} disabled={!dirty || saving || loadingLimits}>
          {saving ? 'Saving translator models…' : 'Save translator models'}
        </button>
        <button type="button" className="secondary" onClick={discard} disabled={!dirty || saving}>
          Discard changes
        </button>
        {dirty && !saving && (
          <span className="auto-translate-hint">
            {loadingLimits
              ? 'Loading the context window and max output for the selected model…'
              : 'You have unsaved model changes.'}
          </span>
        )}
      </div>
    </div>
  );
}
