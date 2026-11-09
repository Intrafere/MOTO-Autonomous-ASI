import React, { useEffect, useId, useRef, useState } from 'react';
import { ORIGINAL_LANGUAGE, useAutoTranslate } from '../../hooks/useAutoTranslate';
import { translationServiceWait } from './TranslationPendingNotice';
import { formatPercent } from './translationPercent';
import './TranslationControls.css';

// queuePosition 0 = translating now, n = n-th in line, null = no position yet.
function pendingLabel(overall) {
  const percent = overall.total > 0 ? ` ${formatPercent(overall.translated, overall.total)}` : '';
  if (overall.queuePosition === 0) {
    return `Translating${percent}`;
  }
  if (Number.isInteger(overall.queuePosition) && overall.queuePosition > 0) {
    return `Queued #${overall.queuePosition}`;
  }
  return `Pending${percent}`;
}

function statusChip(translation, state) {
  if (!translation?.active) {
    return null;
  }
  const overall = translation.overall || {};
  switch (overall.status) {
    case 'complete':
      return { modifier: 'complete', label: 'Translated' };
    case 'pending': {
      const wait = translationServiceWait(state);
      return wait
        ? { modifier: wait.chipModifier, label: wait.chip, role: 'status' }
        : { modifier: 'pending', label: pendingLabel(overall), role: 'status' };
    }
    case 'loading':
      return { modifier: 'loading', label: 'Checking…' };
    case 'error':
      return { modifier: 'error', label: 'Translation unavailable' };
    case 'unavailable':
      return { modifier: 'unavailable', label: 'Not selected' };
    default:
      return null;
  }
}

// Viewers often sit inside clickable cards (an expanded paper collapses on click), so
// using the switch must never reach ancestor handlers. Escape and Tab still bubble so
// dialogs keep closing and trapping focus.
function keepEventInSwitch(event) {
  event.stopPropagation();
}

function keepKeyInSwitch(event) {
  if (event.key !== 'Escape' && event.key !== 'Tab') {
    event.stopPropagation();
  }
}

// What the request button asks for: the chosen language, or every selected language while
// the original is shown.
function requestTarget(translation) {
  return translation?.active ? (translation.languageName || translation.language || 'this language') : '';
}

// "Request translation" for the viewer's documents. A request is saved, so they keep
// translating after the viewer closes; it hides once the backend reports them requested
// or covered by a current run.
function RequestTranslationButton({ translation }) {
  const requestKey = translation?.active ? (translation.language || '') : ORIGINAL_LANGUAGE;
  const [state, setState] = useState({ key: requestKey, status: 'idle' });
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const status = state.key === requestKey ? state.status : 'idle';
  if (status === 'sent') {
    return (
      <span className="translation-language-switch__request-note" role="status">
        Requested
      </span>
    );
  }
  if (typeof translation?.request !== 'function' || !translation?.overall?.requestable) {
    return null;
  }
  const target = requestTarget(translation);
  const description = target
    ? `Translate into ${target} in the background, even after you leave this page`
    : 'Translate into all your selected languages in the background';
  const send = async () => {
    setState({ key: requestKey, status: 'sending' });
    try {
      await translation.request();
      if (mountedRef.current) {
        setState({ key: requestKey, status: 'sent' });
      }
    } catch (error) {
      console.warn('Translation request failed:', error?.message || error);
      if (mountedRef.current) {
        setState({ key: requestKey, status: 'failed' });
      }
    }
  };
  return (
    <>
      <button
        type="button"
        className="translation-language-switch__request"
        onClick={send}
        disabled={status === 'sending'}
        aria-label={`Request translation. ${description}.`}
        title={description}
      >
        {status === 'sending' ? 'Requesting…' : 'Request translation'}
      </button>
      {status === 'failed' && (
        <span className="translation-language-switch__request-note translation-language-switch__request-note--error" role="alert">
          Request failed. Try again.
        </span>
      )}
    </>
  );
}

// Option text: unfinished languages say how far they are, so the greyed-out choice explains itself.
function optionLabel(option) {
  if (option.ready) {
    return option.name;
  }
  if (option.status === 'disabled') {
    return `${option.name} (translation off)`;
  }
  if (option.status === 'pending' && option.total > 0 && option.translated > 0) {
    return `${option.name} (${formatPercent(option.translated, option.total)})`;
  }
  return `${option.name} (not translated yet)`;
}

// Compact "Language" select for one viewer's documents, with a button to request the
// translation. It is always shown inside the app: every selected language is listed, and a
// language not yet finished for these documents is greyed out. Choosing it still shows the
// original until it is ready, and translates these documents first. The choice applies to
// this viewer's documents only; the default comes from the Auto Top-p Translate window.
export default function TranslationLanguageSwitch({ translation = null, className = '' }) {
  const { available, status, enabled } = useAutoTranslate();
  const selectId = useId();

  if (!available || !translation) {
    return null;
  }

  const options = Array.isArray(translation.languageOptions) ? translation.languageOptions : [];
  const selected = translation.selectedLanguage || ORIGINAL_LANGUAGE;
  const chip = statusChip(translation, status?.state);
  let hint = '';
  if (!enabled) {
    hint = 'Translation is off. Turn it on in the Foreign Language Translation w/ Top-p window.';
  } else if (!options.length) {
    hint = 'Choose languages in the Foreign Language Translation w/ Top-p window.';
  }

  return (
    <div
      className={`translation-language-switch${className ? ` ${className}` : ''}`}
      onClick={keepEventInSwitch}
      onMouseDown={keepEventInSwitch}
      onMouseUp={keepEventInSwitch}
      onPointerDown={keepEventInSwitch}
      onPointerUp={keepEventInSwitch}
      onKeyDown={keepKeyInSwitch}
    >
      <label className="translation-language-switch__label" htmlFor={selectId}>
        Language
      </label>
      <select
        id={selectId}
        className="translation-language-switch__select"
        value={options.some((option) => option.key === selected) ? selected : ORIGINAL_LANGUAGE}
        onChange={(event) => translation.setLanguage?.(event.target.value)}
        title={hint || undefined}
      >
        <option value={ORIGINAL_LANGUAGE}>Original</option>
        {options.map((option) => (
          <option
            key={option.key}
            value={option.key}
            className={option.ready ? undefined : 'translation-language-switch__option--unready'}
            data-ready={option.ready ? 'true' : 'false'}
          >
            {optionLabel(option)}
          </option>
        ))}
      </select>
      {chip && (
        <span
          className={`translation-status-chip translation-status-chip--${chip.modifier}`}
          role={chip.role}
        >
          {chip.label}
        </span>
      )}
      <RequestTranslationButton translation={translation} />
    </div>
  );
}
