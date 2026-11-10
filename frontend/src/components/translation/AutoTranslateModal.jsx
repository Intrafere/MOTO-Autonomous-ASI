import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { translationAPI } from '../../services/api';
import useModalDialogFocus from '../../hooks/useModalDialogFocus';
import { AUTO_TRANSLATE_NAME, ORIGINAL_LANGUAGE, useAutoTranslate } from '../../hooks/useAutoTranslate';
import TranslationHelp from './TranslationHelp';
import TranslationLanguagePicker from './TranslationLanguagePicker';
import TranslatorModelsPanel from './TranslatorModelsPanel';
import TranslationProgressPanel from './TranslationProgressPanel';
import TranslationModesDropdown, {
  DEFAULT_TRANSLATION_MODES,
  hasDeveloperOnlyTranslationModes,
  translationModesPayload,
} from './TranslationModesDropdown';
import {
  describeTranslationError,
  hasSavedTranslatorModels,
  sanitizeLanguagePayload,
  sanitizeRolePayload,
  savedRolesSignature,
} from './translatorRoleConfig';
import '../settings-common.css';
import './AutoTranslateModal.css';

export const AUTO_TRANSLATE_TABS = [
  { id: 'languages', label: 'Languages' },
  { id: 'models', label: 'Translator models' },
  { id: 'progress', label: 'Progress' },
];

const EMPTY_SETTINGS = Object.freeze({
  enabled: false,
  paused: false,
  batch_translation: false,
  languages: [],
  translator: null,
  validator: null,
  // null while the backend has sent no modes (it predates the checklist or has not restarted).
  modes: null,
});

const MODES_UNAVAILABLE_TEXT = 'Restart MOTO to choose translated modes.';

export const BATCH_TRANSLATION_HELP = 'Translate several documents at once when your connection allows it: '
  + 'OpenRouter and cloud providers, or LM Studio with several loaded copies of the model. More documents finish '
  + 'sooner, but it uses more model capacity and can slow down research and the rest of MOTO.';

export const TRANSLATION_MODES_HELP = 'Choose which modes have their current run translated automatically. New '
  + 'papers, proofs, final answers, completed brainstorms, and results from each checked mode\'s current run are '
  + 'translated as they appear. Older runs of a checked mode translate when you open an item in your language, or '
  + 'all at once with Translate all older items. Items from unchecked modes translate only when you open them. '
  + 'Unchecking a mode keeps the translations already finished.';

export const DEFAULT_LANGUAGE_HELP = "The language items open in. Each item's Language menu can switch just "
  + 'that item, without changing any other item or this default. A language an item has not finished yet is greyed '
  + 'out in its menu; choosing it shows the original until it is ready and translates that item first.';

export const TRANSLATE_ALL_HELP = 'Translates everything else you can still open — papers, final answers, and '
  + 'proofs from older runs in your libraries and history, plus pruned papers — for the modes checked in the list. '
  + 'It runs in the background at a lower priority than current-run translations and anything you are viewing. When '
  + 'your translator connection can run more than one job at once, it runs alongside them; otherwise it waits until '
  + 'they finish. Content you cleared is never included. It stops by itself when everything is translated, and '
  + 'Cancel stops it early; finished translations are kept.';

// The settings PUT body. enabled/paused are sent as last known (the service keeps its
// own values); translate_all belongs to control actions and is never sent; LeanOJ is
// a translated mode only in developer mode. Modes are sent only once the backend has
// sent them, because a backend that never did does not know the field.
export function buildTranslationSettingsPayload(settings, developerModeEnabled) {
  const payload = {
    enabled: Boolean(settings.enabled),
    paused: Boolean(settings.paused),
    batch_translation: Boolean(settings.batch_translation),
    languages: (settings.languages || []).map(sanitizeLanguagePayload),
    translator: sanitizeRolePayload(settings.translator),
    validator: sanitizeRolePayload(settings.validator),
  };
  if (Array.isArray(settings.modes)) {
    payload.modes = translationModesPayload(settings.modes, developerModeEnabled);
  }
  return payload;
}

// Every settings PUT, from the window and from the developer-mode cleanup below, and the
// window's re-read of settings another save changed, runs one at a time, so each builds
// on the settings the previous one left.
let settingsSaveChain = Promise.resolve();

function queueSettingsSave(save) {
  const run = settingsSaveChain.then(save);
  settingsSaveChain = run.catch(() => {});
  return run;
}

// Saved modes compared regardless of order, or null when the backend sent none.
function modesKey(modes) {
  return Array.isArray(modes) ? [...modes].sort().join(',') : null;
}

// LeanOJ is translated only in developer mode. Whenever developer mode is off while the
// saved modes still include it (checked in another browser, or an earlier cleanup
// failed), the settings are saved without it so its hidden content stops translating.
// The saved modes come with each status, so a failed cleanup runs again with the next
// one; a status without modes comes from a backend that has none to remove. A status or
// developer-mode change that arrives while a cleanup runs gets its own check afterwards.
export function TranslationDeveloperModesGate({ developerModeEnabled = false }) {
  const { status, refreshStatus } = useAutoTranslate();
  const developerModeRef = useRef(developerModeEnabled);
  developerModeRef.current = developerModeEnabled;
  const runningRef = useRef(false);
  const recheckRef = useRef(false);
  const [recheckToken, setRecheckToken] = useState(0);
  const needsCleanup = !developerModeEnabled && hasDeveloperOnlyTranslationModes(status?.modes);

  useEffect(() => {
    if (!needsCleanup) {
      return;
    }
    if (runningRef.current) {
      recheckRef.current = true;
      return;
    }
    runningRef.current = true;
    queueSettingsSave(async () => {
      const settings = (await translationAPI.getSettings())?.settings;
      if (developerModeRef.current || !hasDeveloperOnlyTranslationModes(settings?.modes)) {
        return false;
      }
      await translationAPI.saveSettings(buildTranslationSettingsPayload(settings, false));
      return true;
    })
      .then((saved) => {
        if (saved) {
          refreshStatus?.();
        }
      })
      .catch((error) => {
        console.warn('Auto Top-p Translate modes could not be updated:', error?.message || error);
      })
      .finally(() => {
        runningRef.current = false;
        if (recheckRef.current) {
          recheckRef.current = false;
          setRecheckToken((token) => token + 1);
        }
      });
  }, [needsCleanup, recheckToken, refreshStatus, status]);

  return null;
}

function wholeCount(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

// The status line shown while Translate all is on: the backend's message, which knows
// what it waits for, or a line built from its state when a backend sends none.
export function translateAllStatusText(translateAll, { paused = false } = {}) {
  const message = typeof translateAll?.message === 'string' ? translateAll.message.trim() : '';
  if (message) {
    return message;
  }
  if (paused) {
    return 'Paused.';
  }
  if (translateAll?.state === 'translating') {
    return `Translating older items: ${wholeCount(translateAll.complete_documents).toLocaleString()} of `
      + `${wholeCount(translateAll.documents).toLocaleString()} done.`;
  }
  if (translateAll?.state === 'waiting') {
    return 'Waiting to translate older items.';
  }
  return 'Finding older items…';
}

// How many documents translate at once, shown while Batch translation is on.
export function batchCapacityText(maxParallelJobs) {
  const count = Number(maxParallelJobs);
  if (!Number.isFinite(count) || count < 1) {
    return '';
  }
  return count === 1
    ? 'Your current models translate one document at a time.'
    : `Up to ${Math.floor(count)} documents translate at once.`;
}

const TOP_P_EXPLORATION_URL = 'https://intrafere.com/structured-brainstorming-validated-feedback/';

// The "How it works" disclosure under the intro.
const HOW_IT_WORKS = [
  {
    title: 'Translator',
    text: 'A translator model works through each document one part at a time and can go back to correct earlier parts.',
  },
  {
    title: 'Validator',
    text: 'A separate validator model checks every attempt and sends specific feedback when it rejects one.',
  },
  {
    title: 'Translator notes',
    text: 'The translator may add an optional note explaining its choices or disputing feedback, so a translator '
      + 'that knows the language better can make its case.',
  },
  {
    title: 'Left as written',
    text: 'Math, code, and Lean proof code stay untranslated, and the original is always one click away in each '
      + 'viewer\'s Language menu.',
  },
];

const STATE_FALLBACK_MESSAGES = {
  disabled: `${AUTO_TRANSLATE_NAME} is off.`,
  not_configured: 'Choose languages and translator models to start translating.',
  idle: `${AUTO_TRANSLATE_NAME} is ready.`,
  translating: 'Translating…',
  paused: `${AUTO_TRANSLATE_NAME} is paused.`,
  waiting_for_credits: 'Waiting for provider credits before translating again.',
  repair_required: `${AUTO_TRANSLATE_NAME} needs its translator settings repaired.`,
};

export function statusBannerVariant(state) {
  switch (state) {
    case 'idle':
    case 'translating':
      return 'success';
    case 'repair_required':
      return 'error';
    case 'paused':
    case 'waiting_for_credits':
    case 'not_configured':
      return 'warning';
    default:
      return 'neutral';
  }
}

function repairRouteText(repair) {
  const route = repair?.route_identity;
  if (typeof route === 'string') {
    return route.trim();
  }
  if (route && typeof route === 'object' && typeof route.route_identity === 'string') {
    return route.route_identity.trim();
  }
  return '';
}

// A setting saved as soon as it changes (Batch translation, the translated modes). The
// control shows the latest requested value until that save settles; a failed save shows
// the saved value again.
function useImmediateSetting({ field, saveSettings, mountedRef, failureMessage }) {
  const [pending, setPending] = useState(null);
  const [error, setError] = useState('');
  const sequenceRef = useRef(0);

  const change = async (value) => {
    sequenceRef.current += 1;
    const sequence = sequenceRef.current;
    setPending(value);
    setError('');
    try {
      await saveSettings({ [field]: value });
    } catch (saveError) {
      if (mountedRef.current) {
        setError(describeTranslationError(saveError, failureMessage));
      }
    } finally {
      if (mountedRef.current && sequence === sequenceRef.current) {
        setPending(null);
      }
    }
  };

  return { pending, error, change };
}

// "Translate all older items" with its inline confirm, or, while it is on, its status
// line and Cancel. One button element swaps its action so keyboard focus stays on it.
function TranslateAllControl({
  translateAll,
  paused,
  blocker,
  unavailable,
  busy,
  onStart,
  onCancel,
}) {
  const helpId = useId();
  const blockerId = useId();
  const [confirming, setConfirming] = useState(false);
  const buttonRef = useRef(null);
  const notNowRef = useRef(null);
  const confirmWasOpenRef = useRef(false);
  const requested = Boolean(translateAll?.requested);
  const showConfirm = confirming && !requested && !blocker && !unavailable;

  // Keep keyboard focus on the control that replaced the one just pressed.
  useEffect(() => {
    if (showConfirm) {
      notNowRef.current?.focus();
    } else if (confirmWasOpenRef.current) {
      setConfirming(false);
      buttonRef.current?.focus();
    }
    confirmWasOpenRef.current = showConfirm;
  }, [showConfirm]);

  const confirm = async () => {
    await onStart();
    setConfirming(false);
  };

  return (
    <>
      <span className="auto-translate-translate-all">
        {requested && (
          <span className="auto-translate-hint auto-translate-translate-all__status" role="status">
            {translateAllStatusText(translateAll, { paused })}
          </span>
        )}
        <button
          ref={buttonRef}
          type="button"
          className="secondary"
          aria-describedby={!requested && blocker ? `${blockerId} ${helpId}` : helpId}
          disabled={!requested && (Boolean(blocker) || unavailable)}
          onClick={() => (requested ? onCancel() : setConfirming(true))}
        >
          {requested ? 'Cancel Translate all' : 'Translate all older items'}
        </button>
        <TranslationHelp id={helpId} label="Learn about Translate all">{TRANSLATE_ALL_HELP}</TranslationHelp>
        {!requested && blocker && (
          <span id={blockerId} className="auto-translate-hint auto-translate-translate-all__blocker">{blocker}</span>
        )}
      </span>
      {showConfirm && (
        <div className="auto-translate-confirm auto-translate-translate-all__confirm" role="group" aria-label="Confirm Translate all">
          <p>
            Translate every older item still shown in your libraries for the checked modes? This can take a long
            time and uses translator model calls.
          </p>
          <div className="auto-translate-confirm__actions">
            <button type="button" className="btn-success-sm" onClick={confirm} disabled={busy}>
              Translate all
            </button>
            <button ref={notNowRef} type="button" className="secondary" onClick={() => setConfirming(false)}>
              Not now
            </button>
          </div>
        </div>
      )}
    </>
  );
}

function tabId(id) {
  return `auto-translate-tab-${id}`;
}

function panelId(id) {
  return `auto-translate-panel-${id}`;
}

function AutoTranslateDialog({
  onClose,
  capabilities,
  credentialStatusRefreshToken = 0,
  developerModeEnabled = false,
  onStatusChanged,
}) {
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const tabRefs = useRef([]);
  const mountedRef = useRef(false);
  const settingsGenerationRef = useRef(0);
  const controlBusyRef = useRef(false);
  const mutationsInFlightRef = useRef(0);
  const latestGenerationRef = useRef(null);
  const {
    status: contextStatus,
    recentActivity,
    refreshStatus,
    languages: contextLanguages,
    displayLanguage,
    setDisplayLanguage,
  } = useAutoTranslate();
  const [statusOverride, setStatusOverride] = useState(null);
  const [activeTab, setActiveTab] = useState(AUTO_TRANSLATE_TABS[0].id);
  const [visitedTabs, setVisitedTabs] = useState(() => new Set([AUTO_TRANSLATE_TABS[0].id]));
  const [savedSettings, setSavedSettings] = useState(null);
  const [settingsError, setSettingsError] = useState('');
  const [controlBusy, setControlBusy] = useState(false);
  const [controlError, setControlError] = useState('');
  const batchHelpId = useId();
  const modesHelpId = useId();
  const defaultLanguageId = useId();
  const defaultLanguageHelpId = useId();
  const developerModeRef = useRef(developerModeEnabled);
  developerModeRef.current = developerModeEnabled;

  useModalDialogFocus({ dialogRef, initialFocusRef: closeRef, onEscape: onClose });

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Requests can overlap (a language save sent while Enable is in flight) and their
  // replies can arrive in any order. The service generation grows with every settings
  // or control change, so while requests overlap an older status never replaces a
  // newer one. Outside an overlap every status applies (a restarted backend restarts
  // its generation). Returns whether a status with this generation may be shown.
  const acceptStatusGeneration = useCallback((generation) => {
    if (!Number.isFinite(generation)) {
      return true;
    }
    if (latestGenerationRef.current !== null && generation < latestGenerationRef.current) {
      return false;
    }
    if (mutationsInFlightRef.current > 0) {
      latestGenerationRef.current = generation;
    }
    return true;
  }, []);

  // A fresher provider status replaces the one returned by this dialog's last request.
  useEffect(() => {
    if (acceptStatusGeneration(contextStatus?.generation)) {
      setStatusOverride(null);
    }
  }, [acceptStatusGeneration, contextStatus]);

  const status = statusOverride || contextStatus;
  const statusRef = useRef(status);
  statusRef.current = status;
  const savedSettingsRef = useRef(savedSettings);
  savedSettingsRef.current = savedSettings;

  const notifyStatusChanged = useCallback(() => {
    refreshStatus?.();
    onStatusChanged?.();
  }, [onStatusChanged, refreshStatus]);

  // Show the status a request returned unless a newer one was already shown.
  const applyReturnedStatus = useCallback((nextStatus) => {
    if (!mountedRef.current || !nextStatus || typeof nextStatus !== 'object') {
      return;
    }
    if (acceptStatusGeneration(nextStatus.generation)) {
      statusRef.current = nextStatus;
      setStatusOverride(nextStatus);
    }
  }, [acceptStatusGeneration]);

  const trackMutation = useCallback(async (request) => {
    mutationsInFlightRef.current += 1;
    try {
      return await request();
    } finally {
      mutationsInFlightRef.current -= 1;
      if (mutationsInFlightRef.current === 0) {
        latestGenerationRef.current = null;
      }
    }
  }, []);

  const loadSettings = useCallback(async () => {
    settingsGenerationRef.current += 1;
    const generation = settingsGenerationRef.current;
    setSettingsError('');
    try {
      const response = await translationAPI.getSettings();
      if (!mountedRef.current || generation !== settingsGenerationRef.current) {
        return;
      }
      const nextSettings = { ...EMPTY_SETTINGS, ...(response?.settings || {}) };
      savedSettingsRef.current = nextSettings;
      setSavedSettings(nextSettings);
    } catch (error) {
      if (!mountedRef.current || generation !== settingsGenerationRef.current) {
        return;
      }
      setSettingsError(describeTranslationError(error, `${AUTO_TRANSLATE_NAME} settings could not be loaded.`));
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  // The developer-mode cleanup and other browser tabs save the translated modes too.
  // When the service's modes change to ones this window's saved settings lack, those
  // are read again in the save line, so the list shows them and later saves build on them.
  const statusModesKey = modesKey(status?.modes);
  const statusModesKeyRef = useRef(statusModesKey);
  useEffect(() => {
    const previousKey = statusModesKeyRef.current;
    statusModesKeyRef.current = statusModesKey;
    if (
      statusModesKey === null
      || statusModesKey === previousKey
      || modesKey(savedSettingsRef.current?.modes) === statusModesKey
    ) {
      return;
    }
    queueSettingsSave(loadSettings);
  }, [loadSettings, statusModesKey]);

  // PUT the saved settings with one part replaced. Language, role, Batch translation,
  // and mode edits each keep the other saved parts. Saves run one at a time (with the
  // developer-mode cleanup) and each builds on the previous save's result. enabled/paused are part of the settings
  // model and sent as last known, but the service keeps its own values (Enable,
  // Pause, and Resume go through control()), so nothing here depends on them.
  const saveSettings = useCallback((patch) => {
    const save = async () => {
      const base = savedSettingsRef.current || EMPTY_SETTINGS;
      const currentStatus = statusRef.current;
      const payload = buildTranslationSettingsPayload({
        ...base,
        ...patch,
        enabled: typeof currentStatus?.enabled === 'boolean' ? currentStatus.enabled : Boolean(base.enabled),
        paused: typeof currentStatus?.paused === 'boolean' ? currentStatus.paused : Boolean(base.paused),
      }, developerModeRef.current);
      settingsGenerationRef.current += 1;
      const response = await trackMutation(async () => {
        const reply = await translationAPI.saveSettings(payload);
        if (mountedRef.current && reply?.settings) {
          const nextSettings = { ...EMPTY_SETTINGS, ...reply.settings };
          savedSettingsRef.current = nextSettings;
          setSavedSettings(nextSettings);
        }
        applyReturnedStatus(reply?.status);
        return reply;
      });
      notifyStatusChanged();
      return response;
    };
    return queueSettingsSave(save);
  }, [applyReturnedStatus, notifyStatusChanged, trackMutation]);

  const saveLanguages = useCallback((languages) => saveSettings({ languages }), [saveSettings]);
  const saveRoles = useCallback((roles) => saveSettings(roles), [saveSettings]);

  const batch = useImmediateSetting({
    field: 'batch_translation',
    saveSettings,
    mountedRef,
    failureMessage: 'Batch translation could not be saved.',
  });
  const modes = useImmediateSetting({
    field: 'modes',
    saveSettings,
    mountedRef,
    failureMessage: 'The translated modes could not be saved.',
  });

  const runControl = async (action) => {
    if (controlBusyRef.current) {
      return;
    }
    controlBusyRef.current = true;
    setControlBusy(true);
    setControlError('');
    try {
      await trackMutation(async () => {
        const response = await translationAPI.control(action);
        applyReturnedStatus(response?.status);
      });
    } catch (error) {
      if (mountedRef.current) {
        setControlError(describeTranslationError(error, `${AUTO_TRANSLATE_NAME} could not be updated.`));
      }
    } finally {
      controlBusyRef.current = false;
      if (mountedRef.current) {
        setControlBusy(false);
      }
    }
    notifyStatusChanged();
  };

  const clearCache = async () => {
    const response = await trackMutation(async () => {
      const reply = await translationAPI.clearCache();
      applyReturnedStatus(reply?.status);
      return reply;
    });
    notifyStatusChanged();
    return response;
  };

  const selectTab = (id) => {
    setActiveTab(id);
    setVisitedTabs((previous) => (previous.has(id) ? previous : new Set([...previous, id])));
  };

  const handleTabKeyDown = (event, index) => {
    let nextIndex = null;
    if (event.key === 'ArrowRight') {
      nextIndex = (index + 1) % AUTO_TRANSLATE_TABS.length;
    } else if (event.key === 'ArrowLeft') {
      nextIndex = (index - 1 + AUTO_TRANSLATE_TABS.length) % AUTO_TRANSLATE_TABS.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = AUTO_TRANSLATE_TABS.length - 1;
    }
    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    selectTab(AUTO_TRANSLATE_TABS[nextIndex].id);
    tabRefs.current[nextIndex]?.focus();
  };

  const enabled = Boolean(status?.enabled);
  const paused = Boolean(status?.paused);
  const state = status?.state || '';
  const isRepair = state === 'repair_required';
  const repair = status?.repair || null;
  const bannerMessage = status
    ? (isRepair ? (repair?.message || status.message) : status.message) || STATE_FALLBACK_MESSAGES[state] || ''
    : `Checking ${AUTO_TRANSLATE_NAME} status…`;
  const repairGuidance = isRepair ? repair?.guidance || '' : '';
  const repairRoute = isRepair ? repairRouteText(repair) : '';
  const bannerVariant = statusBannerVariant(state);
  const modelsSaved = hasSavedTranslatorModels(savedSettings);
  const rolesSignature = savedRolesSignature(savedSettings);
  const batchChecked = batch.pending ?? Boolean(savedSettings?.batch_translation);
  const batchCapacity = batchChecked && batch.pending === null && status?.batch_translation === true
    ? batchCapacityText(status?.max_parallel_jobs)
    : '';
  const modesUnavailable = Boolean(savedSettings) && !Array.isArray(savedSettings.modes);
  const checkedModes = modes.pending ?? (savedSettings ? savedSettings.modes : DEFAULT_TRANSLATION_MODES);
  let translateAllBlocker = '';
  if (status && !enabled) {
    translateAllBlocker = 'Turn on translation first.';
  } else if (savedSettings && (!(savedSettings.languages || []).length || !modelsSaved)) {
    translateAllBlocker = 'Choose languages and translator models first.';
  } else if (Array.isArray(checkedModes) && translationModesPayload(checkedModes, developerModeEnabled).length === 0) {
    translateAllBlocker = 'Check at least one mode in Current runs first.';
  }

  const renderSettingsPanel = (render) => {
    if (savedSettings) {
      return render();
    }
    if (settingsError) {
      return (
        <div className="auto-translate-inline-error">
          <p className="settings-error" role="alert">{settingsError}</p>
          <button type="button" className="secondary" onClick={() => loadSettings()}>Try again</button>
        </div>
      );
    }
    return <p className="auto-translate-hint">Loading translation settings…</p>;
  };

  return (
    <div
      className="inline-modal-overlay auto-translate-overlay"
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        className="inline-modal-content inline-modal-content--fluid auto-translate-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auto-translate-title"
        aria-describedby="auto-translate-summary"
      >
        <header className="auto-translate-header">
          <span className="auto-translate-eyebrow">Skills and enhancements</span>
          <h2 id="auto-translate-title">{AUTO_TRANSLATE_NAME}</h2>
          <button
            ref={closeRef}
            type="button"
            className="modal-close-btn auto-translate-close"
            aria-label={`Close ${AUTO_TRANSLATE_NAME}`}
            onClick={onClose}
          >
            ×
          </button>
        </header>

        <div className="auto-translate-intro">
          <p id="auto-translate-summary" className="auto-translate-intro__lead">
            {AUTO_TRANSLATE_NAME} translates the papers, proofs, final answers, and brainstorms from your current
            runs into the languages you choose, in the background while you work.
          </p>
          <p className="auto-translate-intro__topp">
            It uses{' '}
            <a href={TOP_P_EXPLORATION_URL} target="_blank" rel="noopener noreferrer">Top-p Exploration</a>, the
            structured brainstorming with validated feedback that makes MOTO so smart and lets it run
            autonomously for hours.
          </p>
          <details className="auto-translate-howto">
            <summary>How it works</summary>
            <ul className="auto-translate-howto__list">
              {HOW_IT_WORKS.map((item) => (
                <li key={item.title} className="auto-translate-howto__item">
                  <span className="auto-translate-howto__title">{item.title}</span>
                  <p>{item.text}</p>
                </li>
              ))}
            </ul>
          </details>
        </div>

        <section className="auto-translate-service" aria-label={`${AUTO_TRANSLATE_NAME} controls`}>
          <div className="auto-translate-service__main">
            <label className="settings-checkbox-label auto-translate-enable">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(event) => runControl(event.target.checked ? 'enable' : 'disable')}
              />
              Enable {AUTO_TRANSLATE_NAME}
            </label>
            <div className="auto-translate-service__actions">
              {controlBusy && <span className="auto-translate-hint" role="status">Updating…</span>}
              {enabled && (
                <button type="button" className="secondary" onClick={() => runControl(paused ? 'resume' : 'pause')}>
                  {paused ? 'Resume' : 'Pause'}
                </button>
              )}
            </div>
          </div>
          <div className={`test-result-banner auto-translate-banner auto-translate-banner--${bannerVariant}${bannerVariant === 'success' || bannerVariant === 'error' ? ` test-result-banner--${bannerVariant}` : ''}`}>
            <p className="auto-translate-banner__message" role="status">{bannerMessage}</p>
            {repairGuidance && <p className="auto-translate-banner__detail">{repairGuidance}</p>}
            {repairRoute && <p className="auto-translate-banner__detail auto-translate-banner__route">{repairRoute}</p>}
            {isRepair && (
              <button type="button" className="secondary auto-translate-banner__retry" onClick={() => runControl('retry')}>
                Retry
              </button>
            )}
          </div>
          {controlError && <p className="settings-error" role="alert">{controlError}</p>}

          <div className="auto-translate-options">
            <div className="auto-translate-option">
              <label className="auto-translate-option__label" htmlFor={defaultLanguageId}>Default language</label>
              <div className="auto-translate-option__controls">
                <span className="auto-translate-default-language">
                  <select
                    id={defaultLanguageId}
                    className="auto-translate-default-language__select"
                    value={(contextLanguages || []).some((language) => language.key === displayLanguage)
                      ? displayLanguage
                      : ORIGINAL_LANGUAGE}
                    aria-describedby={defaultLanguageHelpId}
                    onChange={(event) => setDisplayLanguage(event.target.value)}
                  >
                    <option value={ORIGINAL_LANGUAGE}>Original</option>
                    {(contextLanguages || []).map((language) => (
                      <option key={language.key} value={language.key}>{language.name || language.key}</option>
                    ))}
                  </select>
                  <TranslationHelp id={defaultLanguageHelpId} label="Learn about the default language">
                    {DEFAULT_LANGUAGE_HELP}
                  </TranslationHelp>
                </span>
              </div>
            </div>
            <div className="auto-translate-option">
              <span className="auto-translate-option__label">Automatically</span>
              <div className="auto-translate-option__controls">
                <TranslationModesDropdown
                  modes={checkedModes}
                  developerModeEnabled={developerModeEnabled}
                  disabled={!savedSettings}
                  unavailableReason={modesUnavailable ? MODES_UNAVAILABLE_TEXT : ''}
                  describedBy={modesHelpId}
                  onChange={modes.change}
                  help={(
                    <TranslationHelp id={modesHelpId} label="Learn about which modes are translated">
                      {TRANSLATION_MODES_HELP}
                    </TranslationHelp>
                  )}
                />
                {modes.error && <p className="settings-error" role="alert">{modes.error}</p>}
              </div>
            </div>
            <div className="auto-translate-option">
              <span className="auto-translate-option__label">Older items</span>
              <div className="auto-translate-option__controls">
                <TranslateAllControl
                  translateAll={status?.translate_all}
                  paused={enabled && paused}
                  blocker={translateAllBlocker}
                  unavailable={!status || !savedSettings}
                  busy={controlBusy}
                  onStart={() => runControl('translate_all')}
                  onCancel={() => runControl('cancel_translate_all')}
                />
              </div>
            </div>
            <div className="auto-translate-option">
              <span className="auto-translate-option__label">Speed</span>
              <div className="auto-translate-option__controls">
                <span className="auto-translate-batch">
                  <label className="settings-checkbox-label auto-translate-batch__label">
                    <input
                      type="checkbox"
                      checked={batchChecked}
                      disabled={!savedSettings}
                      aria-describedby={batchHelpId}
                      onChange={(event) => batch.change(event.target.checked)}
                    />
                    Batch translation
                  </label>
                  <TranslationHelp id={batchHelpId} label="Learn about Batch translation">
                    {BATCH_TRANSLATION_HELP}
                  </TranslationHelp>
                </span>
                {batchCapacity && (
                  <span className="auto-translate-hint auto-translate-batch__capacity">{batchCapacity}</span>
                )}
                {batch.error && <p className="settings-error" role="alert">{batch.error}</p>}
              </div>
            </div>
          </div>
        </section>

        <div className="auto-translate-tabs" role="tablist" aria-label={`${AUTO_TRANSLATE_NAME} sections`}>
          {AUTO_TRANSLATE_TABS.map((tab, index) => {
            const selected = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                ref={(node) => { tabRefs.current[index] = node; }}
                type="button"
                role="tab"
                id={tabId(tab.id)}
                className={`auto-translate-tab${selected ? ' auto-translate-tab--active' : ''}`}
                aria-selected={selected}
                aria-controls={panelId(tab.id)}
                tabIndex={selected ? 0 : -1}
                onClick={() => selectTab(tab.id)}
                onKeyDown={(event) => handleTabKeyDown(event, index)}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {AUTO_TRANSLATE_TABS.map((tab) => (
          <div
            key={tab.id}
            id={panelId(tab.id)}
            role="tabpanel"
            aria-labelledby={tabId(tab.id)}
            className="auto-translate-tabpanel"
            hidden={tab.id !== activeTab}
            tabIndex={tab.id === activeTab ? 0 : -1}
          >
            {visitedTabs.has(tab.id) && tab.id === 'languages' && renderSettingsPanel(() => (
              <TranslationLanguagePicker
                active={activeTab === 'languages'}
                selectedLanguages={savedSettings.languages || []}
                onSaveLanguages={saveLanguages}
                modelsSaved={modelsSaved}
                rolesSignature={rolesSignature}
              />
            ))}
            {visitedTabs.has(tab.id) && tab.id === 'models' && renderSettingsPanel(() => (
              <TranslatorModelsPanel
                savedSettings={savedSettings}
                capabilities={capabilities}
                credentialStatusRefreshToken={credentialStatusRefreshToken}
                developerModeEnabled={developerModeEnabled}
                onSaveRoles={saveRoles}
              />
            ))}
            {visitedTabs.has(tab.id) && tab.id === 'progress' && (
              <TranslationProgressPanel
                status={status}
                recentActivity={recentActivity}
                onClearCache={clearCache}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Auto Top-p Translate settings dialog (languages, translator models, progress).
// Its state lives only while open, so each opening starts from saved settings.
export default function AutoTranslateModal({ isOpen, ...props }) {
  if (!isOpen) {
    return null;
  }
  return <AutoTranslateDialog {...props} />;
}
