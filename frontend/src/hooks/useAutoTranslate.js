import { createContext, useContext } from 'react';

export const ORIGINAL_LANGUAGE = 'original';
// The name users see; "Auto Top-p Translate" stays the internal name.
export const AUTO_TRANSLATE_NAME = 'Foreign Language Translation w/ Top-p';
export const DISPLAY_LANGUAGE_STORAGE_KEY = 'auto_translate_display_language';

const noop = () => {};
const noopAsync = async () => {};
const noopSubscribe = () => noop;

// Value seen by any consumer rendered outside AutoTranslateProvider: translation is
// off, nothing is requested, and every viewer keeps showing original content.
export const INERT_AUTO_TRANSLATE_CONTEXT = Object.freeze({
  available: false,
  status: null,
  enabled: false,
  languages: Object.freeze([]),
  // The default reading language, set in the Auto Top-p Translate window.
  displayLanguage: ORIGINAL_LANGUAGE,
  setDisplayLanguage: noop,
  // Per-document choices from each viewer's language switch: { [docKey]: languageKey }.
  documentLanguages: Object.freeze({}),
  setDocumentLanguages: noop,
  refreshStatus: noopAsync,
  subscribeProgress: noopSubscribe,
  recentActivity: Object.freeze([]),
  translateAll: null,
});

export const AutoTranslateContext = createContext(INERT_AUTO_TRANSLATE_CONTEXT);

export function useAutoTranslate() {
  return useContext(AutoTranslateContext);
}

export default useAutoTranslate;
