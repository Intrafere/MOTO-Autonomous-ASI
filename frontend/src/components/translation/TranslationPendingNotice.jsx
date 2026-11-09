import React from 'react';
import { AUTO_TRANSLATE_NAME, useAutoTranslate } from '../../hooks/useAutoTranslate';
import { TRANSLATION_MODES } from './TranslationModesDropdown';
import { formatPercent } from './translationPercent';
import './TranslationControls.css';

const WAITING_STATUSES = new Set(['pending', 'loading']);

// Service states in which nothing is translated until the user or the provider acts, so a
// place in line would mislead: the notice and the chip say what translation waits for.
const SERVICE_WAITS = {
  repair_required: {
    variant: 'repair',
    chip: 'Needs repair',
    chipModifier: 'error',
    text: () => `${AUTO_TRANSLATE_NAME} is paused until its translator settings are repaired. `
      + `Open ${AUTO_TRANSLATE_NAME} from the Skills And Enhancements panel.`,
  },
  paused: {
    variant: 'paused',
    chip: 'Paused',
    chipModifier: 'waiting',
    text: () => `${AUTO_TRANSLATE_NAME} is paused.`,
  },
  not_configured: {
    variant: 'paused',
    chip: 'Needs translator models',
    chipModifier: 'waiting',
    text: (subject) => `${subject} is waiting for the translator models to be set up in ${AUTO_TRANSLATE_NAME}. `
      + `Open ${AUTO_TRANSLATE_NAME} from the Skills And Enhancements panel.`,
  },
  waiting_for_credits: {
    variant: 'paused',
    chip: 'Waiting for credits',
    chipModifier: 'waiting',
    text: (subject) => `${subject} is waiting for provider credits. Showing the original until it is ready.`,
  },
};

const TRANSLATE_ALL_WINDOW = `Translate all older items in the ${AUTO_TRANSLATE_NAME} window (Skills And Enhancements)`;

// What a waiting translation waits for when the service itself is not translating, or null.
export function translationServiceWait(state) {
  return SERVICE_WAITS[state] || null;
}

function modeNames(values) {
  return TRANSLATION_MODES.filter((mode) => values.includes(mode.value)).map((mode) => mode.label).join(' and ');
}

// Where a library viewer's whole library translates at once. viewerModes lists the modes
// the library holds items of, and Translate all skips the items of unchecked modes, so
// the hint names every viewer mode still to check. A status without saved modes (older
// backends) gets the plain hint.
export function translateAllHintText(viewerModes, savedModes) {
  if (!Array.isArray(viewerModes) || viewerModes.length === 0) {
    return '';
  }
  const unchecked = Array.isArray(savedModes) ? viewerModes.filter((mode) => !savedModes.includes(mode)) : [];
  if (unchecked.length === 0) {
    return `To translate your whole library in the background, use ${TRANSLATE_ALL_WINDOW}.`;
  }
  if (unchecked.length === viewerModes.length) {
    return `To translate your whole library in the background, check ${modeNames(unchecked)} in Current runs, `
      + `then use ${TRANSLATE_ALL_WINDOW}.`;
  }
  return `To translate your library in the background, use ${TRANSLATE_ALL_WINDOW}. It skips `
    + `${modeNames(unchecked)} items until you check ${modeNames(unchecked)} in Current runs.`;
}

// translateAllHint: the text a library viewer adds (translateAllHintText) while its item
// waits in line.
export function translationPendingMessage(translation, state, { translateAllHint = '' } = {}) {
  if (!translation?.active || !WAITING_STATUSES.has(translation.overall?.status)) {
    return null;
  }
  const language = translation.languageName || translation.language || '';
  const subject = language ? `${language} translation` : 'Translation';
  const wait = translationServiceWait(state);
  if (wait) {
    return { variant: wait.variant, text: wait.text(subject) };
  }
  const {
    status,
    translated = 0,
    total = 0,
    queuePosition = null,
  } = translation.overall;
  if (status === 'loading' && !total) {
    return {
      variant: 'pending',
      text: `Checking for ${language ? `a ${language}` : 'a'} translation. Showing the original until it is ready.`,
    };
  }
  const done = formatPercent(translated, total);
  if (status === 'pending' && queuePosition === 0) {
    return {
      variant: 'pending',
      text: `${language || 'Translation'}: translating now — ${done} done`,
    };
  }
  const hint = translateAllHint ? ` ${translateAllHint}` : '';
  if (status === 'pending' && Number.isInteger(queuePosition) && queuePosition > 0) {
    return {
      variant: 'pending',
      text: `${subject} queued — #${queuePosition} in line. Showing the original until it is ready.${hint}`,
    };
  }
  return {
    variant: 'pending',
    text: `${subject} pending — ${done} translated. Showing the original until it is ready.${hint}`,
  };
}

// Explains why a viewer still shows the original while its translation is being made.
// Library viewers pass translateAllModes, the modes their items belong to, to point to
// Translate all while it is off.
export default function TranslationPendingNotice({ translation = null, className = '', translateAllModes = null }) {
  const { status, translateAll } = useAutoTranslate();
  const message = translationPendingMessage(translation, status?.state, {
    translateAllHint: translateAll?.requested ? '' : translateAllHintText(translateAllModes, status?.modes),
  });
  if (!message) {
    return null;
  }
  return (
    <p
      className={`translation-pending-notice translation-pending-notice--${message.variant}${className ? ` ${className}` : ''}`}
      role="status"
    >
      {message.text}
    </p>
  );
}
